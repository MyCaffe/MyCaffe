﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Threading;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;

/// <summary>
/// The MyCaffe.db.image namespace contains all classes used to create the MyCaffeImageDatabase in-memory database.
/// </summary>
namespace MyCaffe.db.image
{
    /// <summary>
    /// [V2 Image Database]
    /// The MyCaffeImageDatabase2 provides an enhanced in-memory image database used for quick image retrieval.
    /// </summary>
    /// <remarks>
    /// The MyCaffeImageDatbase2 manages a set of data sets, where each data sets comprise a pair of data sources: one source 
    /// for training and another source for testing.  Each data source contains a list of images and a list of label sets
    /// that point back into the list of images.  This organization allows for quick image selection by image or by label
    /// set and then by image from within the label set.
    /// </remarks>
    public partial class MyCaffeImageDatabase2 : Component, IXImageDatabase2
    {
        CryptoRandom m_random = null;
        DatasetFactory m_factory;
        string m_strID = "";
        int m_nStrIDHashCode = 0;
        EventWaitHandle m_evtInitializing = null;
        EventWaitHandle m_evtInitialized = null;
        EventWaitHandle m_evtAbortInitialization = null;
        bool m_bEnabled = false;
        static object m_syncObject = new object();
        static Dictionary<int, DatasetExCollection2> m_colDatasets = new Dictionary<int, DatasetExCollection2>();
        static Dictionary<int, LabelMappingCollection> m_rgLabelMappings = new Dictionary<int, LabelMappingCollection>();
        Dictionary<int, SimpleDatum> m_rgMeanCache = new Dictionary<int, SimpleDatum>();
        double m_dfSuperBoostProbability = 0;
        DB_ITEM_SELECTION_METHOD m_imageSelectionMethod = DB_ITEM_SELECTION_METHOD.RANDOM;
        DB_LABEL_SELECTION_METHOD m_labelSelectionMethod = DB_LABEL_SELECTION_METHOD.RANDOM;
        Log m_log;
        DB_LOAD_METHOD m_loadMethod = DB_LOAD_METHOD.LOAD_ON_DEMAND;
        int m_nLoadLimit = 0;
        bool m_bSkipMeanCheck = false;
        int m_nPadW = 0;
        int m_nPadH = 0;
        Guid m_userGuid;
        DateTime? m_dtMinDate = null;
        DateTime? m_dtMaxDate = null;


        /// <summary>
        /// The OnCalculateImageMean event fires each time the MyCaffeImageDatabase wants to access the image mean for a data set.
        /// </summary>
        public event EventHandler<CalculateImageMeanArgs> OnCalculateImageMean;

        /// <summary>
        /// The MyCaffeImageDatabase2 constructor.
        /// </summary>
        /// <param name="log">The Log for output.</param>
        /// <param name="strId">Specifies an identifier for this in memory database instance (default = "default").</param>
        /// <param name="nSeed">Optionally, specifies a seed for the random number generator (default = null).</param>
        public MyCaffeImageDatabase2(Log log = null, string strId = "default", int nSeed = 0)
        {
            m_factory = new DatasetFactory();
            m_userGuid = Guid.NewGuid();
            m_log = log;
            InitializeComponent();
            init(strId, nSeed);

            if (log != null)
                log.WriteLine("INFO: Using MyCaffe Image Database VERSION 2.");
        }

        /// <summary>
        /// The MyCaffeImageDatabase constructor.
        /// </summary>
        /// <param name="container">Specifies a container.</param>
        public MyCaffeImageDatabase2(IContainer container)
        {
            container.Add(this);

            InitializeComponent();
            init();
        }

        /// <summary>
        /// Returns the version of the MyCaffe Image Database being used.
        /// </summary>
        /// <returns>Returns the version.</returns>
        public DB_VERSION GetVersion()
        {
            return DB_VERSION.IMG_V2;
        }

        #region Initialization and Cleanup

        /// <summary>
        /// Set the database connection to use.
        /// </summary>
        /// <param name="ci">Specifies the dataase connection information to use.</param>
        public void SetConnection(ConnectInfo ci)
        {
            MyCaffe.db.image.EntitiesConnection.GlobalDatabaseConnectInfo = ci;
        }

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="strDs">Specifies the data set to load.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>Upon loading the dataset a handle to the default QueryState is returned (which is ordered by Index), or 0 on cancel.</returns>
        public long InitializeWithDsName(SettingsCaffe s, string strDs, string strEvtCancel = null)
        {
            return InitializeWithDs(s, new DatasetDescriptor(strDs), strEvtCancel);
        }

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="ds">Specifies the data set to load.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>Upon loading the dataset a handle to the default QueryState is returned (which is ordered by Index), or 0 on cancel.</returns>
        public long InitializeWithDs(SettingsCaffe s, DatasetDescriptor ds, string strEvtCancel = null)
        {
            string strDsName = ds.Name;

            if (String.IsNullOrEmpty(strDsName))
            {
                strDsName = m_factory.FindDatasetNameFromSourceName(ds.TrainingSourceName, ds.TestingSourceName);
                if (strDsName == null)
                    throw new Exception("Could not find the dataset! You must specify either the Datast name or at least the training or testing source name!");
            }

            int nDsId = m_factory.GetDatasetID(strDsName);

            return InitializeWithDsId(s, nDsId, strEvtCancel);
        }

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="nDataSetID">Specifies the database ID of the data set to load.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <param name="nPadW">Specifies the padding to add to each image width (default = 0).</param>
        /// <param name="nPadH">Specifies the padding to add to each image height (default = 0).</param>
        /// <returns>Upon loading the dataset a handle to the default QueryState is returned (which is ordered by Index), or 0 on cancel.</returns>
        public long InitializeWithDsId(SettingsCaffe s, int nDataSetID, string strEvtCancel = null, int nPadW = 0, int nPadH = 0)
        {
            Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD> selMethod = GetSelectionMethod(s);

            m_nPadW = nPadW;
            m_nPadH = nPadH;
            m_labelSelectionMethod = selMethod.Item1;
            m_imageSelectionMethod = selMethod.Item2;
            m_dfSuperBoostProbability = s.SuperBoostProbability;
            m_loadMethod = s.DbLoadMethod;
            m_nLoadLimit = s.DbLoadLimit;
            m_bSkipMeanCheck = s.SkipMeanCheck;
            m_dtMinDate = s.DbLoadMinDate;
            m_dtMaxDate = s.DbLoadMaxDate;

            if (m_loadMethod == DB_LOAD_METHOD.LOAD_EXTERNAL)
                m_loadMethod = DB_LOAD_METHOD.LOAD_ON_DEMAND_BACKGROUND;

            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtInitialized, m_evtInitializing, m_evtAbortInitialization }, 0);

            lock (m_syncObject)
            {
                if (nWait != WaitHandle.WaitTimeout)
                {
                    if (nWait == 0)     // already initialized.
                    {
                        if (m_colDatasets.ContainsKey(m_nStrIDHashCode))
                        {
                            DatasetExCollection2 col = m_colDatasets[m_nStrIDHashCode];

                            if (m_log != null)
                                m_log.WriteLine("The MyCaffe Image Database is already initialized.");

                            DatasetEx2 ds = col.FindDataset(nDataSetID);
                            if (ds != null)
                            {
                                ds.AddUser(m_userGuid);
                                return ds.DefaultQueryState;
                            }
                        }
                    }
                    else
                    {
                        return 0;
                    }
                }

                try
                {
                    m_evtInitializing.Set();

                    m_factory.SetLoadingParameters(s.ItemDbLoadDataCriteria, s.ItemDbLoadDebugData, s.DbLoadMinDate, s.DbLoadMaxDate);

                    DatasetDescriptor ds = m_factory.LoadDataset(nDataSetID);
                    if (ds == null)
                        throw new Exception("Could not find dataset with ID = " + nDataSetID.ToString());

                    List<WaitHandle> rgAbort = new List<WaitHandle>() { m_evtAbortInitialization };
                    CancelEvent evtCancel;

                    if (strEvtCancel != null)
                    {
                        evtCancel = new CancelEvent(strEvtCancel);
                        rgAbort.AddRange(evtCancel.Handles);
                    }

                    DatasetExCollection2 col = null;

                    if (m_colDatasets.ContainsKey(m_nStrIDHashCode))
                        col = m_colDatasets[m_nStrIDHashCode];
                    else
                        col = new DatasetExCollection2();

                    if (m_evtAbortInitialization.WaitOne(0))
                    {
                        col.Dispose();
                        return 0;
                    }

                    DatasetEx2 ds0 = new DatasetEx2(m_userGuid, m_factory, m_random);

                    if (OnCalculateImageMean != null)
                        ds0.OnCalculateImageMean += OnCalculateImageMean;

                    if (m_log != null)
                        m_log.WriteLine("Loading dataset '" + ds.Name + "'...", true);

                    long lQueryHandle = ds0.Initialize(ds, rgAbort.ToArray(), nPadW, nPadH, m_log, m_loadMethod, m_bSkipMeanCheck, m_nLoadLimit, s.DbAutoRefreshScheduledUpdateInMs, s.DbAutoRefreshScheduledReplacementPercent, s.VeriyDatasetOnLoad);
                    if (lQueryHandle == 0)
                    {
                        col.Dispose();
                        return 0;
                    }

                    if (m_log != null)
                        m_log.WriteLine("Dataset '" + ds.Name + "' loaded.", true);

                    col.Add(ds0);

                    if (!m_colDatasets.ContainsKey(m_nStrIDHashCode))
                        m_colDatasets.Add(m_nStrIDHashCode, col);

                    UseTrainingImagesForTesting = s.UseTrainingSourceForTesting;

                    return lQueryHandle;
                }
                finally
                {
                    m_evtInitialized.Set();
                    m_evtInitializing.Reset();
                }
            }
        }

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="strDs">Specifies the data set to load.</param>
        /// <param name="strEvtCancel">Optionally, specifies the name of the CancelEvent used to cancel load operations (default = null).</param>
        /// <param name="prop">Optionally, specifies the properties for the initialization (default = null).</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        public bool InitializeWithDsName1(SettingsCaffe s, string strDs, string strEvtCancel = null, PropertySet prop = null)
        {
            long lHandle = InitializeWithDsName(s, strDs, strEvtCancel);
            if (lHandle == 0)
                return false;

            return true;
        }

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="ds">Specifies the data set to load.</param>
        /// <param name="strEvtCancel">Optionally, specifies the name of the CancelEvent used to cancel load operations (default = null).</param>
        /// <param name="prop">Optionally, specifies the properties for the initialization (default = null).</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        public bool InitializeWithDs1(SettingsCaffe s, DatasetDescriptor ds, string strEvtCancel = null, PropertySet prop = null)
        {
            long lHandle = InitializeWithDs(s, ds, strEvtCancel);
            if (lHandle == 0)
                return false;

            return true;
        }

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="nDataSetID">Specifies the database ID of the data set to load.</param>
        /// <param name="strEvtCancel">Optionally, specifies the name of the CancelEvent used to cancel load operations (default = null).</param>
        /// <param name="nPadW">Optionally, specifies the padding to add to each image width (default = 0).</param>
        /// <param name="nPadH">Optionally, specifies the padding to add to each image height (default = 0).</param>
        /// <param name="prop">Optionally, specifies the properties for the initialization (default = null).</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        public bool InitializeWithDsId1(SettingsCaffe s, int nDataSetID, string strEvtCancel = null, int nPadW = 0, int nPadH = 0, PropertySet prop = null)
        {
            long lHandle = InitializeWithDsId(s, nDataSetID, strEvtCancel, nPadW, nPadH);
            if (lHandle == 0)
                return false;

            return true;
        }

        private void init(string strId = "", int nSeed = 0)
        {
            int nProcessID = Process.GetCurrentProcess().Id;

            m_random = new CryptoRandom(CryptoRandom.METHOD.DEFAULT, nSeed);
            m_strID = strId;
            m_nStrIDHashCode = strId.GetHashCode();
            m_evtInitializing = new EventWaitHandle(false, EventResetMode.ManualReset, "__CAFFE_IMAGEDATABASE__INITIALIZING__" + nProcessID.ToString());
            m_evtInitialized = new EventWaitHandle(false, EventResetMode.ManualReset, "__CAFFE_IMAGEDATABASE__INITIALIZED__" + nProcessID.ToString());
            m_evtAbortInitialization = new EventWaitHandle(false, EventResetMode.ManualReset, "__CAFFE_IMAGEDATABASE__ABORT_INITIALIZE__" + nProcessID.ToString());
        }

        /// <summary>
        /// Releases the image database, and if this is the last instance using the in-memory database, frees all memory used.
        /// </summary>
        /// <param name="nDsId">Optionally, specifies the dataset previously loaded.</param>
        /// <param name="bForce">Optionally, force the cleanup even if other users are using the database.</param>
        public void CleanUp(int nDsId = 0, bool bForce = false)
        {
            lock (m_syncObject)
            {
                if (m_evtInitializing != null && m_evtInitializing.WaitOne(0))
                {
                    m_evtAbortInitialization.Set();
                    return;
                }

                List<int> rgRemove = new List<int>();

                foreach (KeyValuePair<int, DatasetExCollection2> col in m_colDatasets)
                {
                    DatasetExCollection2 colDs = col.Value;

                    if (colDs.RemoveUser(m_userGuid) || bForce)
                    {
                        rgRemove.Add(col.Key);
                        colDs.Dispose();
                    }
                }

                foreach (int nKey in rgRemove)
                {
                    m_colDatasets.Remove(nKey);
                }

                if (m_colDatasets.Count == 0)
                {
                    if (m_evtInitialized.WaitOne(0))
                        m_evtInitialized.Reset();
                }
            }
        }

        private void dispose()
        {
            CleanUp(0, false);

            if (m_evtInitialized != null)
            {
                m_evtInitialized.Dispose();
                m_evtInitialized = null;
            }

            if (m_evtInitializing != null)
            {
                m_evtInitializing.Dispose();
                m_evtInitializing = null;
            }

            if (m_evtAbortInitialization != null)
            {
                m_evtAbortInitialization.Dispose();
                m_evtAbortInitialization = null;
            }

            if (m_random != null)
            {
                m_random.Dispose();
                m_random = null;
            }
        }

        /// <summary>
        /// Wait for the dataset loading to complete.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <param name="bTraining">Specifies to wait for the training data source to load.</param>
        /// <param name="bTesting">Specifies to wait for the testing data source to load.</param>
        /// <param name="nWait">Specifies the amount of time to wait in ms. (default = int.MaxValue).</param>
        /// <returns>If the data source(s) complete loading, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool WaitForDatasetToLoad(int nDsId, bool bTraining, bool bTesting, int nWait = int.MaxValue)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].WaitForDatasetToLoad(nDsId, bTraining, bTesting, nWait);
        }

        /// <summary>
        /// Wait for the dataset loading to complete.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="bTraining">Specifies to wait for the training data source to load.</param>
        /// <param name="bTesting">Specifies to wait for the testing data source to load.</param>
        /// <param name="nWait">Specifies the amount of time to wait in ms. (default = int.MaxValue).</param>
        /// <returns>If the data source(s) complete loading, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool WaitForDatasetToLoad(string strDs, bool bTraining, bool bTesting, int nWait = int.MaxValue)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].WaitForDatasetToLoad(strDs, bTraining, bTesting, nWait);
        }

        /// <summary>
        /// Reload the indexing for a data set.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <returns>If the data source(s) have their indexing reloaded, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool ReloadIndexing(int nDsId)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            m_colDatasets[m_nStrIDHashCode].ReloadIndexing(nDsId);
            return true;
        }

        /// <summary>
        /// Reload the indexing for a data set.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <returns>If the data source(s) have their indexing reloaded, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool ReloadIndexing(string strDs)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            m_colDatasets[m_nStrIDHashCode].ReloadIndexing(strDs);
            return true;
        }

        /// <summary>
        /// Create the database used by the MyCaffeImageDatabase.
        /// </summary>
        /// <param name="ci">Specifies the connection information for the database (recommended value: db = 'DNN', server = '.')</param>
        /// <param name="strPath">Specifies the file path where the database is to be created.</param>
        public static void CreateDatabase(ConnectInfo ci, string strPath)
        {
            if (ci == null)
                ci = EntitiesConnection.GlobalDatabaseConnectInfo;

            DatabaseManagement dbMgr = new DatabaseManagement(ci, strPath);
            Exception excpt = dbMgr.CreateDatabase();

            if (excpt != null)
                throw excpt;
        }

        #endregion // Initialization and Cleanup

        #region Refreshing

        /// <summary>
        /// Wait for the dataset refreshing to complete.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <param name="bTraining">Specifies to wait for the training data source to refresh.</param>
        /// <param name="bTesting">Specifies to wait for the testing data source to refresh.</param>
        /// <param name="nWait">Specifies the amount of time to wait in ms. (default = int.MaxValue).</param>
        /// <returns>If the data source(s) complete refreshing, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool WaitForDatasetToRefresh(int nDsId, bool bTraining, bool bTesting, int nWait = int.MaxValue)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].WaitForDatasetToRefresh(nDsId, bTraining, bTesting, nWait);
        }

        /// <summary>
        /// Wait for the dataset refreshing to complete.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="bTraining">Specifies to wait for the training data source to refresh.</param>
        /// <param name="bTesting">Specifies to wait for the testing data source to refresh.</param>
        /// <param name="nWait">Specifies the amount of time to wait in ms. (default = int.MaxValue).</param>
        /// <returns>If the data source(s) complete refreshing, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool WaitForDatasetToRefresh(string strDs, bool bTraining, bool bTesting, int nWait = int.MaxValue)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].WaitForDatasetToRefresh(strDs, bTraining, bTesting, nWait);
        }

        /// <summary>
        /// Returns true if the refresh operation running.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <param name="bTraining">Specifies to check the training data source for refresh.</param>
        /// <param name="bTesting">Specifies to check the testing data source for refresh.</param>
        /// <returns>If the refresh is running, true is returned, otherwise false.</returns>
        public bool IsRefreshRunning(int nDsId, bool bTraining, bool bTesting)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].IsRefreshRunning(nDsId, bTraining, bTesting);
        }

        /// <summary>
        /// Returns true if the refresh operation running.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="bTraining">Specifies to check the training data source for refresh.</param>
        /// <param name="bTesting">Specifies to check the testing data source for refresh.</param>
        /// <returns>If the refresh is running, true is returned, otherwise false.</returns>
        public bool IsRefreshRunning(string strDs, bool bTraining, bool bTesting)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].IsRefreshRunning(strDs, bTraining, bTesting);
        }

        /// <summary>
        /// Start a refresh on the dataset by replacing a specified percentage of the images with images from the physical database.
        /// </summary>
        /// <remarks>
        /// Note, this method is only valid when initialized with LoadLimit > 0.
        /// </remarks>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="bTraining">Specifies the training data source to refresh.</param>
        /// <param name="bTesting">Specifies the testing data source to refresh.</param>
        /// <param name="dfReplacementPct">Optionally, specifies the replacement percentage to use (default = 0.25 for 25%).</param>
        /// <returns>On succes, true is returned, otherwise false is returned.</returns>
        public bool StartRefresh(string strDs, bool bTraining, bool bTesting, double dfReplacementPct)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].StartRefresh(strDs, bTraining, bTesting, dfReplacementPct);
        }

        /// <summary>
        /// Stop a refresh operation running on the dataset.
        /// </summary>
        /// <remarks>
        /// Note, this method is only valid when initialized with LoadLimit > 0.
        /// </remarks>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="bTraining">Specifies the training data source to strop refreshing.</param>
        /// <param name="bTesting">Specifies the testing data source to stop refreshing.</param>
        /// <returns>On succes, true is returned, otherwise false is returned.</returns>
        public bool StopRefresh(string strDs, bool bTraining, bool bTesting)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].StopRefresh(strDs, bTraining, bTesting);
        }

        /// <summary>
        /// Start a refresh on the dataset by replacing a specified percentage of the images with images from the physical database.
        /// </summary>
        /// <remarks>
        /// Note, this method is only valid when initialized with LoadLimit > 0.
        /// </remarks>
        /// <param name="nDsID">Specifies the dataset ID.</param>
        /// <param name="bTraining">Specifies the training data source to refresh.</param>
        /// <param name="bTesting">Specifies the testing data source to refresh.</param>
        /// <param name="dfReplacementPct">Optionally, specifies the replacement percentage to use (default = 0.25 for 25%).</param>
        /// <returns>On succes, true is returned, otherwise false is returned.</returns>
        public bool StartRefresh(int nDsID, bool bTraining, bool bTesting, double dfReplacementPct)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].StartRefresh(nDsID, bTraining, bTesting, dfReplacementPct);
        }

        /// <summary>
        /// Stop a refresh operation running on the dataset.
        /// </summary>
        /// <remarks>
        /// Note, this method is only valid when initialized with LoadLimit > 0.
        /// </remarks>
        /// <param name="nDsID">Specifies the dataset ID.</param>
        /// <param name="bTraining">Specifies the training data source to strop refreshing.</param>
        /// <param name="bTesting">Specifies the testing data source to stop refreshing.</param>
        /// <returns>On succes, true is returned, otherwise false is returned.</returns>
        public bool StopRefresh(int nDsID, bool bTraining, bool bTesting)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].StopRefresh(nDsID, bTraining, bTesting);
        }

        /// <summary>
        /// Start the automatic refresh cycle to occur on specified period increments.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name for which the automatic refresh cycle is to run.</param>
        /// <param name="bTraining">Specifies the training data source to start refreshing.</param>
        /// <param name="bTesting">Specifies the testing data source to start refreshing.</param>
        /// <param name="nPeriodInMs">Specifies the period in milliseconds over which the auto refresh cycle is to run.</param>
        /// <param name="dfReplacementPct">Specifies the percentage of replacement to use on each cycle.</param>
        /// <returns>If successfully started, true is returned, otherwise false.</returns>
        public bool StartAutomaticRefreshSchedule(string strDs, bool bTraining, bool bTesting, int nPeriodInMs, double dfReplacementPct)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            if (nPeriodInMs <= 0 || dfReplacementPct <= 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].StartAutomaticRefreshSchedule(strDs, bTraining, bTesting, nPeriodInMs, dfReplacementPct);
        }

        /// <summary>
        /// Stop the automatic refresh schedule running on a dataset.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name for which the automatic refresh cycle is to run.</param>
        /// <param name="bTraining">Specifies the training data source to stop refreshing.</param>
        /// <param name="bTesting">Specifies the testing data source to stop refreshing.</param>
        /// <returns>If successfully stopped, true is returned, otherwise false.</returns>
        public bool StopAutomaticRefreshSchedule(string strDs, bool bTraining, bool bTesting)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].StopAutomaticRefreshSchedule(strDs, bTraining, bTesting);
        }

        /// <summary>
        /// Returns whether or not a scheduled refresh is running and if so at what period and replacement percent.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name for which the automatic refresh cycle is to run.</param>
        /// <param name="nPeriodInMs">Returns the period in milliseconds over which the auto refresh cycle is run.</param>
        /// <param name="dfReplacementPct">Returns the percentage of replacement to use on each cycle.</param>
        /// <param name="nTrainingRefreshCount">Returns the training refrsh count.</param>
        /// <param name="nTestingRefreshCount">Returns the testing refresh count.</param>
        /// <returns>If the refresh schedule is running, true is returned, otherwise false.</returns>
        public bool GetScheduledAutoRefreshInformation(string strDs, out int nPeriodInMs, out double dfReplacementPct, out int nTrainingRefreshCount, out int nTestingRefreshCount)
        {
            nPeriodInMs = 0;
            dfReplacementPct = 0;
            nTrainingRefreshCount = 0;
            nTestingRefreshCount = 0;

            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].GetScheduledAutoRefreshInformation(strDs, out nPeriodInMs, out dfReplacementPct, out nTrainingRefreshCount, out nTestingRefreshCount);
        }

        /// <summary>
        /// Start the automatic refresh cycle to occur on specified period increments.
        /// </summary>
        /// <param name="nDsID">Specifies the dataset ID for which the automatic refresh cycle is to run.</param>
        /// <param name="bTraining">Specifies the training data source to start refreshing.</param>
        /// <param name="bTesting">Specifies the testing data source to start refreshing.</param>
        /// <param name="nPeriodInMs">Specifies the period in milliseconds over which the auto refresh cycle is to run.</param>
        /// <param name="dfReplacementPct">Specifies the percentage of replacement to use on each cycle.</param>
        /// <returns>If successfully started, true is returned, otherwise false.</returns>
        public bool StartAutomaticRefreshSchedule(int nDsID, bool bTraining, bool bTesting, int nPeriodInMs, double dfReplacementPct)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            if (nPeriodInMs <= 0 || dfReplacementPct <= 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].StartAutomaticRefreshSchedule(nDsID, bTraining, bTesting, nPeriodInMs, dfReplacementPct);
        }

        /// <summary>
        /// Stop the automatic refresh schedule running on a dataset.
        /// </summary>
        /// <param name="nDsID">Specifies the dataset ID for which the automatic refresh cycle is to run.</param>
        /// <param name="bTraining">Specifies the training data source to stop refreshing.</param>
        /// <param name="bTesting">Specifies the testing data source to stop refreshing.</param>
        /// <returns>If successfully stopped, true is returned, otherwise false.</returns>
        public bool StopAutomaticRefreshSchedule(int nDsID, bool bTraining, bool bTesting)
        {
            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].StopAutomaticRefreshSchedule(nDsID, bTraining, bTesting);
        }

        /// <summary>
        /// Returns whether or not a scheduled refresh is running and if so at what period and replacement percent.
        /// </summary>
        /// <param name="nDsID">Specifies the dataset name for which the automatic refresh cycle is to run.</param>
        /// <param name="nPeriodInMs">Returns the period in milliseconds over which the auto refresh cycle is run.</param>
        /// <param name="dfReplacementPct">Returns the percentage of replacement to use on each cycle.</param>
        /// <param name="nTrainingRefreshCount">Returns the training refrsh count.</param>
        /// <param name="nTestingRefreshCount">Returns the testing refresh count.</param>
        /// <returns>If the refresh schedule is running, true is returned, otherwise false.</returns>
        public bool GetScheduledAutoRefreshInformation(int nDsID, out int nPeriodInMs, out double dfReplacementPct, out int nTrainingRefreshCount, out int nTestingRefreshCount)
        {
            nPeriodInMs = 0;
            dfReplacementPct = 0;
            nTrainingRefreshCount = 0;
            nTestingRefreshCount = 0;

            int nWait1 = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait1 == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].GetScheduledAutoRefreshInformation(nDsID, out nPeriodInMs, out dfReplacementPct, out nTrainingRefreshCount, out nTestingRefreshCount);
        }

        #endregion

        #region Query States

        /// <summary>
        /// Create a query state for a data set, optionally using a specific sorting method.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the data set.</param>
        /// <param name="bUseUniqueLabelIndexes">Optionally, specifies to use unique label indexes which is slightly slower, but ensures each label is hit per epoch equally (default = true).</param>
        /// <param name="bUseUniqueImageIndexes">Optionally, specifies to use unique image indexes which is slightly slower, but ensures each image is hit per epoch (default = true).</param>
        /// <param name="sort">Specifies the sorting method.</param>
        /// <returns>The new query state handle is returned.</returns>
        public long CreateQueryState(int nDsId, bool bUseUniqueLabelIndexes = true, bool bUseUniqueImageIndexes = true, IMGDB_SORT sort = IMGDB_SORT.NONE)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait == 0)
                return 0;

            return m_colDatasets[m_nStrIDHashCode].CreateQueryState(nDsId, bUseUniqueLabelIndexes, bUseUniqueImageIndexes, sort);
        }

        /// <summary>
        /// Create a query state for a data set, optionally using a specific sorting method.
        /// </summary>
        /// <param name="strDs">Specifies the name of the data set.</param>
        /// <param name="bUseUniqueLabelIndexes">Optionally, specifies to use unique label indexes which is slightly slower, but ensures each label is hit per epoch equally (default = true).</param>
        /// <param name="bUseUniqueImageIndexes">Optionally, specifies to use unique image indexes which is slightly slower, but ensures each image is hit per epoch (default = true).</param>
        /// <param name="sort">Specifies the sorting method.</param>
        /// <returns>The new query state handle is returned.</returns>
        public long CreateQueryState(string strDs, bool bUseUniqueLabelIndexes = true, bool bUseUniqueImageIndexes = true, IMGDB_SORT sort = IMGDB_SORT.NONE)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait == 0)
                return 0;

            return m_colDatasets[m_nStrIDHashCode].CreateQueryState(strDs, bUseUniqueLabelIndexes, bUseUniqueImageIndexes, sort);
        }

        /// <summary>
        /// Set the default query state to the query state specified for the dataset specified.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <param name="lQueryState">Specifies the query state to set.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        public bool SetDefaultQueryState(int nDsId, long lQueryState)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].SetDefaultQueryState(nDsId, lQueryState);
        }

        /// <summary>
        /// Set the default query state to the query state specified for the dataset specified.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <param name="lQueryState">Specifies the query state to set.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> on failure.</returns>
        public bool SetDefaultQueryState(string strDs, long lQueryState)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].SetDefaultQueryState(strDs, lQueryState);
        }

        /// <summary>
        /// Frees a query state from a given dataset.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset on which to free the query state.</param>
        /// <param name="lHandle">Specifies the handle to the query state to free.</param>
        /// <returns>If found and freed, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool FreeQueryState(int nDsId, long lHandle)
        {
            return m_colDatasets[m_nStrIDHashCode].FreeQueryState(nDsId, lHandle);
        }

        /// <summary>
        /// Frees a query state from a given dataset.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name on which to free the query state.</param>
        /// <param name="lHandle">Specifies the handle to the query state to free.</param>
        /// <returns>If found and freed, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool FreeQueryState(string strDs, long lHandle)
        {
            return m_colDatasets[m_nStrIDHashCode].FreeQueryState(strDs, lHandle);
        }

        /// <summary>
        /// Returns the percentage of boosted images queried as text.
        /// </summary>
        /// <param name="lQueryState">Specifies the handle to the query state.</param>
        /// <param name="strSource">Specifies the source to query.</param>
        /// <returns>The query boost percentage hit is returned as text.</returns>
        public string GetBoostQueryHitPercentsAsTextFromSourceName(long lQueryState, string strSource)
        {           
            return m_colDatasets[m_nStrIDHashCode].FindQueryState(lQueryState, strSource).GetQueryBoostHitPercentsAsText();
        }

        /// <summary>
        /// Returns a string with the query hit percent for each label (e.g. the percentage that each label has been queried).
        /// </summary>
        /// <param name="lQueryState">Specifies the handle to the query state.</param>
        /// <param name="strSource">Specifies the data source who's hit percentages are to be retrieved.</param>
        /// <returns>A string representing the query hit percentages is returned.</returns>
        public string GetLabelQueryHitPercentsAsTextFromSourceName(long lQueryState, string strSource)
        {
            return m_colDatasets[m_nStrIDHashCode].FindQueryState(lQueryState, strSource).GetQueryLabelHitPercentsAsText();
        }

        /// <summary>
        /// Returns a string with the query epoch counts for each label (e.g. the number of times all images with the label have been queried).
        /// </summary>
        /// <param name="lQueryState">Specifies the handle to the query state.</param>
        /// <param name="strSource">Specifies the data source who's query epochs are to be retrieved.</param>
        /// <returns>A string representing the query epoch counts is returned.</returns>
        public string GetLabelQueryEpocsAsTextFromSourceName(long lQueryState, string strSource)
        {
            return m_colDatasets[m_nStrIDHashCode].FindQueryState(lQueryState, strSource).GetQueryLabelEpochsAsText();
        }

        /// <summary>
        /// Returns a string with the query hit percent for each boost (e.g. the percentage that each boost value has been queried).
        /// </summary>
        /// <param name="strSource">Specifies the data source who's hit percentages are to be retrieved.</param>
        /// <returns>A string representing the query hit percentages is returned.</returns>
        public string GetBoostQueryHitPercentsAsTextFromSourceName(string strSource)
        {
            return GetBoostQueryHitPercentsAsTextFromSourceName(0, strSource);
        }

        /// <summary>
        /// Returns a string with the query hit percent for each label (e.g. the percentage that each label has been queried).
        /// </summary>
        /// <param name="strSource">Specifies the data source who's hit percentages are to be retrieved.</param>
        /// <returns>A string representing the query hit percentages is returned.</returns>
        public string GetLabelQueryHitPercentsAsTextFromSourceName(string strSource)
        {
            return GetLabelQueryHitPercentsAsTextFromSourceName(0, strSource);
        }

        /// <summary>
        /// Returns a string with the query epoch counts for each label (e.g. the number of times all images with the label have been queried).
        /// </summary>
        /// <param name="strSource">Specifies the data source who's query epochs are to be retrieved.</param>
        /// <returns>A string representing the query epoch counts is returned.</returns>
        public string GetLabelQueryEpocsAsTextFromSourceName(string strSource)
        {
            return GetLabelQueryEpocsAsTextFromSourceName(0, strSource);
        }

        #endregion // Query States

        #region Properties

        /// <summary>
        /// Get/set the output log.
        /// </summary>
        public Log OutputLog
        {
            get { return m_log; }
            set { m_log = value; }
        }

        /// <summary>
        /// Returns whether or not the image data criteria is loaded with each image.
        /// </summary>
        public bool GetLoadItemDataCriteria()
        {
            return m_factory.LoadDataCriteria;
        }

        /// <summary>
        /// Returns whether or not the image debug data is loaded with each image.
        /// </summary>
        public bool GetLoadItemDebugData()
        {
            return m_factory.LoadDebugData;
        }

        /// <summary>
        /// Returns the number of images in a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The number of images is returned.</returns>
        public int GetImageCount(int nSrcId)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });

            if (nWait == 0)
                return 0;

            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).GetTotalCount();
        }

        /// <summary>
        /// Returns the number of images in a given data source, optionally only counting the boosted images.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>The number of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public int GetItemCount(int nSrcId, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            return GetImageCount(0, nSrcId, strFilterVal, nBoostVal, bBoostValIsExact);
        }

        /// <summary>
        /// Returns the number of images in a given data source, optionally only counting the boosted images.
        /// </summary>
        /// <param name="lQueryState">Specifies a handle to the query state to use.</param>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>The number of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public int GetImageCount(long lQueryState, int nSrcId, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait == 0)
                return 0;

            QueryState qstate = m_colDatasets[m_nStrIDHashCode].FindQueryState(lQueryState, nSrcId);
            ImageSet2 imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId);

            return imgSet.GetCount(qstate, strFilterVal, nBoostVal, bBoostValIsExact);
        }

        /// <summary>
        /// Returns the label/image selection methods based on the SettingsCaffe settings.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <returns>The label/image selection method is returned.</returns>
        public static Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD> GetSelectionMethod(SettingsCaffe s)
        {
            DB_ITEM_SELECTION_METHOD imageSelectionMethod = DB_ITEM_SELECTION_METHOD.NONE;
            DB_LABEL_SELECTION_METHOD labelSelectionMethod = DB_LABEL_SELECTION_METHOD.NONE;

            if (s.EnableRandomInputSelection)
                imageSelectionMethod |= DB_ITEM_SELECTION_METHOD.RANDOM;

            if (s.SuperBoostProbability > 0)
                imageSelectionMethod |= DB_ITEM_SELECTION_METHOD.BOOST;

            if (s.EnableLabelBalancing)
            {
                labelSelectionMethod |= DB_LABEL_SELECTION_METHOD.RANDOM;

                if (s.EnableLabelBoosting)
                    labelSelectionMethod |= DB_LABEL_SELECTION_METHOD.BOOST;
            }
            else
            {
                if (s.EnablePairInputSelection)
                    imageSelectionMethod |= DB_ITEM_SELECTION_METHOD.PAIR;
            }

            return new Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD>(labelSelectionMethod, imageSelectionMethod);
        }

        /// <summary>
        /// Returns the label/image selection methods based on the ProjectEx settings.
        /// </summary>
        /// <param name="p">Specifies the project.</param>
        /// <returns>The label/image selection method is returned.</returns>
        public static Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD> GetSelectionMethod(ProjectEx p)
        {
            DB_ITEM_SELECTION_METHOD imageSelectionMethod = DB_ITEM_SELECTION_METHOD.NONE;
            DB_LABEL_SELECTION_METHOD labelSelectionMethod = DB_LABEL_SELECTION_METHOD.NONE;

            if (p.EnableRandomSelection)
                imageSelectionMethod |= DB_ITEM_SELECTION_METHOD.RANDOM;

            if (p.EnableLabelBalancing)
            {
                labelSelectionMethod |= DB_LABEL_SELECTION_METHOD.RANDOM;

                if (p.EnableLabelBoosting)
                    labelSelectionMethod |= DB_LABEL_SELECTION_METHOD.BOOST;
            }
            else
            {
                if (p.EnablePairSelection)
                    imageSelectionMethod |= DB_ITEM_SELECTION_METHOD.PAIR;
            }

            return new Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD>(labelSelectionMethod, imageSelectionMethod);
        }

        /// <summary>
        /// Returns the label and image selection method used.
        /// </summary>
        /// <returns>A KeyValue containing the Label and Image selection method.</returns>
        public Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD> GetSelectionMethod()
        {
            return new Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD>(m_labelSelectionMethod, m_imageSelectionMethod);
        }

        /// <summary>
        /// Sets the label and image selection methods.
        /// </summary>
        /// <param name="lbl">Specifies the label selection method or <i>null</i> to ignore.</param>
        /// <param name="img">Specifies the image selection method or <i>null</i> to ignore.</param>
        public void SetSelectionMethod(DB_LABEL_SELECTION_METHOD? lbl, DB_ITEM_SELECTION_METHOD? img)
        {
            if (lbl.HasValue)
                m_labelSelectionMethod = lbl.Value;

            if (img.HasValue)
                m_imageSelectionMethod = img.Value;
        }

        /// <summary>
        /// Get/set whether or not to use training images for the test set (default = false).
        /// </summary>
        public bool UseTrainingImagesForTesting
        {
            get { return m_colDatasets[m_nStrIDHashCode].UseTrainingSourcesForTesting; }
            set { m_colDatasets[m_nStrIDHashCode].EnableUsingTrainingSourcesForTesting(value); }
        }

        /// <summary>
        /// Get/set the super-boost probability which increases/decreases the probability of selecting a boosted image (default = 0).
        /// </summary>
        public double SuperBoostProbability
        {
            get { return m_dfSuperBoostProbability; }
            set { m_dfSuperBoostProbability = value; }
        }

        /// <summary>
        /// Returns whether or not to select ONLY from boosted images.
        /// </summary>
        public bool SelectFromBoostOnly
        {
            get
            {
                if (m_dfSuperBoostProbability > 0)
                {
                    double dfRandom = m_random.NextDouble(0, 1);

                    if (dfRandom <= m_dfSuperBoostProbability)
                        return true;
                }

                return false;
            }
        }

        /// <summary>
        /// Returns whether or not the image database is enabled.
        /// </summary>
        public bool IsEnabled
        {
            get { return m_bEnabled; }
        }

        /// <summary>
        /// Sets whether or not the image database is enabled.
        /// </summary>
        /// <param name="bEnable"></param>
        public void Enable(bool bEnable)
        {
            m_bEnabled = bEnable;
        }

        /// <summary>
        /// Returns whether or not the image database is initialized.
        /// </summary>
        public bool IsInitialized
        {
            get { return m_evtInitialized.WaitOne(0); }
        }

        #endregion // Properties

        #region Image Acquisition

        /// <summary>
        /// Returns the array of images in the image set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="nStartIdx">Specifies a starting index from which the query is to start within the set of images.</param>
        /// <param name="nQueryCount">Optionally, specifies a number of images to retrieve within the set (default = int.MaxValue).</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <param name="bAttemptDirectLoad">Optionaly, specifies to directly load all images not already loaded.</param>
        /// <returns>The list of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public List<SimpleDatum> GetItemsFromIndex(int nSrcId, int nStartIdx, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false, bool bAttemptDirectLoad = false)
        {
            return GetImagesFromIndex(0, nSrcId, nStartIdx, nQueryCount, strFilterVal, nBoostVal, bBoostValIsExact, bAttemptDirectLoad);
        }

        /// <summary>
        /// Returns the array of images in the image set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="dtStart">Specifies a starting time from which the query is to start within the set of images.</param>
        /// <param name="nQueryCount">Optionally, specifies a number of images to retrieve within the set (default = int.MaxValue).</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>The list of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public List<SimpleDatum> GetItemsFromTime(int nSrcId, DateTime dtStart, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            return GetImagesFromTime(0, nSrcId, dtStart, nQueryCount, strFilterVal, nBoostVal, bBoostValIsExact);
        }

        /// <summary>
        /// Returns the array of images in the image set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="lQueryState">Specifies a handle to the query state to use.</param>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="nStartIdx">Specifies a starting index from which the query is to start within the set of images.</param>
        /// <param name="nQueryCount">Optionally, specifies a number of images to retrieve within the set (default = int.MaxValue).</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <param name="bAttemptDirectLoad">Optionaly, specifies to directly load all images not already loaded.</param>
        /// <returns>The list of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public List<SimpleDatum> GetImagesFromIndex(long lQueryState, int nSrcId, int nStartIdx, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false, bool bAttemptDirectLoad = false)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait == 0)
                return null;

            QueryState qstate = m_colDatasets[m_nStrIDHashCode].FindQueryState(lQueryState, nSrcId);
            ImageSet2 imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId);

            return imgSet.GetImages(qstate, nStartIdx, nQueryCount, strFilterVal, nBoostVal, bBoostValIsExact, bAttemptDirectLoad);
        }

        /// <summary>
        /// Returns the array of images in the image set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="lQueryState">Specifies a handle to the query state to use.</param>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="dtStart">Specifies a starting time from which the query is to start within the set of images.</param>
        /// <param name="nQueryCount">Optionally, specifies a number of images to retrieve within the set (default = int.MaxValue).</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value.</param>
        /// <returns>The list of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public List<SimpleDatum> GetImagesFromTime(long lQueryState, int nSrcId, DateTime dtStart, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait == 0)
                return null;

            QueryState qstate = m_colDatasets[m_nStrIDHashCode].FindQueryState(lQueryState, nSrcId);
            ImageSet2 imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId);

            return imgSet.GetImages(qstate, dtStart, nQueryCount, strFilterVal, nBoostVal, bBoostValIsExact);
        }

        /// <summary>
        /// Returns the array of images in the image set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="rgIdx">Specifies an array of indexes to query.</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="bBoostValIsExact">Optionally, specifies whether or the boost value (if specified) is to be used literally (exact = true), or as a minimum boost value - currently, not used in Version 2.</param>
        /// <returns>The list of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public List<SimpleDatum> GetItems(int nSrcId, int[] rgIdx, string strFilterVal, int? nBoostVal, bool bBoostValIsExact = false)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait == 0)
                return null;

            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).GetImages(nBoostVal.HasValue, strFilterVal, nBoostVal, rgIdx);
        }

        /// <summary>
        /// Query an image in a given data source.
        /// </summary>
        /// <param name="lQueryState">Specifies a handle to the query state to use.</param>
        /// <param name="nSrcId">Specifies the databse ID of the data source.</param>
        /// <param name="nIdx">Specifies the image index to query.  Note, the index is only used in non-random image queries.</param>
        /// <param name="labelSelectionOverride">Optionally, specifies the label selection method override.  The default = null, which directs the method to use the label selection method specified during Initialization.</param>
        /// <param name="imageSelectionOverride">Optionally, specifies the image selection method override.  The default = null, which directs the method to use the image selection method specified during Initialization.</param>
        /// <param name="nLabel">Optionally, specifies a label set to use for the image selection.  When specified only images of this label are returned using the image selection method.</param>
        /// <param name="bLoadDataCriteria">Specifies to load the data criteria data (default = false).</param>
        /// <param name="bLoadDebugData">Specifies to load the debug data (default = false).</param>
        /// <param name="bThrowExceptions">Optionally, specifies to throw exceptions on error (default = true).</param>
        /// <returns>The image SimpleDatum is returned.</returns>
        public SimpleDatum QueryImage(long lQueryState, int nSrcId, int nIdx, DB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, DB_ITEM_SELECTION_METHOD? imageSelectionOverride = null, int? nLabel = null, bool bLoadDataCriteria = false, bool bLoadDebugData = false, bool bThrowExceptions = true)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait == 0)
                return null;

            DB_LABEL_SELECTION_METHOD labelSelectionMethod = m_labelSelectionMethod;
            DB_ITEM_SELECTION_METHOD imageSelectionMethod = m_imageSelectionMethod;

            if (labelSelectionOverride.HasValue)
                labelSelectionMethod = labelSelectionOverride.Value;

            if (imageSelectionOverride.HasValue)
                imageSelectionMethod = imageSelectionOverride.Value;

            if (SelectFromBoostOnly)
                imageSelectionMethod |= DB_ITEM_SELECTION_METHOD.BOOST;

            QueryState qstate = m_colDatasets[m_nStrIDHashCode].FindQueryState(lQueryState, nSrcId);
            ImageSet2 imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId);

            return imgSet.GetImage(qstate, labelSelectionMethod, imageSelectionMethod, m_log, nLabel, nIdx, bLoadDataCriteria, bLoadDataCriteria, bThrowExceptions);
        }

        /// <summary>
        /// Query an image in a given data source.
        /// </summary>
        /// <param name="lQueryState">Specifies a handle to the query state to use.</param>
        /// <param name="nSrcId">Specifies the databse ID of the data source.</param>
        /// <param name="dt">Specifies the image time to query.</param>
        /// <param name="labelSelectionOverride">Optionally, specifies the label selection method override.  The default = null, which directs the method to use the label selection method specified during Initialization.</param>
        /// <param name="imageSelectionOverride">Optionally, specifies the image selection method override.  The default = null, which directs the method to use the image selection method specified during Initialization.</param>
        /// <param name="nLabel">Optionally, specifies a label set to use for the image selection.  When specified only images of this label are returned using the image selection method.</param>
        /// <param name="bLoadDataCriteria">Specifies to load the data criteria data (default = false).</param>
        /// <param name="bLoadDebugData">Specifies to load the debug data (default = false).</param>
        /// <param name="bThrowExceptions">Optionally, specifies to throw exceptions on error (default = true).</param>
        /// <returns>The image SimpleDatum is returned.</returns>
        public SimpleDatum QueryImage(long lQueryState, int nSrcId, DateTime dt, DB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, DB_ITEM_SELECTION_METHOD? imageSelectionOverride = null, int? nLabel = null, bool bLoadDataCriteria = false, bool bLoadDebugData = false, bool bThrowExceptions = true)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait == 0)
                return null;

            DB_LABEL_SELECTION_METHOD labelSelectionMethod = m_labelSelectionMethod;
            DB_ITEM_SELECTION_METHOD imageSelectionMethod = m_imageSelectionMethod;

            if (labelSelectionOverride.HasValue)
                labelSelectionMethod = labelSelectionOverride.Value;

            if (imageSelectionOverride.HasValue)
                imageSelectionMethod = imageSelectionOverride.Value;

            if (SelectFromBoostOnly)
                imageSelectionMethod |= DB_ITEM_SELECTION_METHOD.BOOST;

            QueryState qstate = m_colDatasets[m_nStrIDHashCode].FindQueryState(lQueryState, nSrcId);
            ImageSet2 imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId);

            return imgSet.GetImage(qstate, labelSelectionMethod, imageSelectionMethod, m_log, dt, nLabel, bLoadDataCriteria, bLoadDataCriteria, bThrowExceptions);
        }

        /// <summary>
        /// Query an image in a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the databse ID of the data source.</param>
        /// <param name="nIdx">Specifies the image index to query.  Note, the index is only used in non-random image queries.</param>
        /// <param name="labelSelectionOverride">Optionally, specifies the label selection method override.  The default = null, which directs the method to use the label selection method specified during Initialization.</param>
        /// <param name="imageSelectionOverride">Optionally, specifies the image selection method override.  The default = null, which directs the method to use the image selection method specified during Initialization.</param>
        /// <param name="nLabel">Optionally, specifies a label set to use for the image selection.  When specified only images of this label are returned using the image selection method.</param>
        /// <param name="bLoadDataCriteria">Specifies to load the data criteria data (default = false).</param>
        /// <param name="bLoadDebugData">Specifies to load the debug data (default = false).</param>
        /// <param name="bThrowExceptions">Optionally, specifies to throw exceptions on error (default = true).</param>
        /// <returns>The image SimpleDatum is returned.</returns>
        public SimpleDatum QueryItem(int nSrcId, int nIdx, DB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, DB_ITEM_SELECTION_METHOD? imageSelectionOverride = null, int? nLabel = null, bool bLoadDataCriteria = false, bool bLoadDebugData = false, bool bThrowExceptions = true)
        {
            return QueryImage(0, nSrcId, nIdx, labelSelectionOverride, imageSelectionOverride, nLabel, bLoadDataCriteria, bLoadDebugData, bThrowExceptions);
        }

        /// <summary>
        /// Query an image in a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the databse ID of the data source.</param>
        /// <param name="dt">Specifies the image time to query.</param>
        /// <param name="labelSelectionOverride">Optionally, specifies the label selection method override.  The default = null, which directs the method to use the label selection method specified during Initialization.</param>
        /// <param name="imageSelectionOverride">Optionally, specifies the image selection method override.  The default = null, which directs the method to use the image selection method specified during Initialization.</param>
        /// <param name="nLabel">Optionally, specifies a label set to use for the image selection.  When specified only images of this label are returned using the image selection method.</param>
        /// <param name="bLoadDataCriteria">Specifies to load the data criteria data (default = false).</param>
        /// <param name="bLoadDebugData">Specifies to load the debug data (default = false).</param>
        /// <param name="bThrowExceptions">Optionally, specifies to throw exceptions on error (default = true).</param>
        /// <returns>The image SimpleDatum is returned.</returns>
        public SimpleDatum QueryItem(int nSrcId, DateTime dt, DB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, DB_ITEM_SELECTION_METHOD? imageSelectionOverride = null, int? nLabel = null, bool bLoadDataCriteria = false, bool bLoadDebugData = false, bool bThrowExceptions = true)
        {
            return QueryImage(0, nSrcId, dt, labelSelectionOverride, imageSelectionOverride, nLabel, bLoadDataCriteria, bLoadDebugData, bThrowExceptions);
        }

        /// <summary>
        /// Returns the image with a given Raw Image ID.
        /// </summary>
        /// <param name="nImageID">Specifies the Raw Image ID.</param>
        /// <param name="rgSrcId">Specifies a set of source ID's to query from.</param>
        /// <returns>If found, the SimpleDatum of the Raw Image is returned, otherwise, <i>null</i> is returned.</returns>
        public SimpleDatum GetItem(int nImageID, params int[] rgSrcId)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });

            if (nWait == 0)
                return null;

            foreach (int nSrcId in rgSrcId)
            {
                ImageSet2 imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId);
                SimpleDatum sd = imgSet.GetImage(nImageID);

                if (sd != null)
                    return sd;
            }

            return null;
        }

        /// <summary>
        /// Searches fro the image index of an image within a data source matching a DateTime/description pattern.
        /// </summary>
        /// <remarks>
        /// Optionally, images may have a time-stamp and/or description associated with each image.  In such cases
        /// searching by the time-stamp + description can be useful in some instances.
        /// </remarks>
        /// <param name="nSrcId">Specifies the data source ID of the data source to be searched.</param>
        /// <param name="dt">Specifies the time-stamp to search for.</param>
        /// <param name="strDescription">Specifies the description to search for.</param>
        /// <returns>If found the zero-based index of the image is returned, otherwise -1 is returned.</returns>
        public int FindItemIndex(int nSrcId, DateTime dt, string strDescription)
        {
            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).FindImageIndex(dt, strDescription);
        }

        #endregion // Image Acquisition

        #region Image Mean

        /// <summary>
        /// Returns the image mean for a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="rgParams">Optionally, specifies image mean parameters to query (default = none)</param>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        public SimpleDatum GetItemMean(int nSrcId, params string[] rgParams)
        {
            if (m_evtAbortInitialization.WaitOne(0))
                return null;

            if (!m_evtInitialized.WaitOne(0))
            {
                if (m_rgMeanCache.Keys.Contains(nSrcId))
                    return m_rgMeanCache[nSrcId];
            }

            SimpleDatum sd = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).GetImageMean(null, null, true, rgParams);

            if (!m_rgMeanCache.ContainsKey(nSrcId))
                m_rgMeanCache.Add(nSrcId, sd);
            else
                m_rgMeanCache[nSrcId] = sd;

            return sd;
        }

        /// <summary>
        /// Sets the image mean for a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="d">Specifies a SimpleDatum containing the image mean.</param>
        public void SetImageMean(int nSrcId, SimpleDatum d)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });

            if (nWait == 0)
                return;

            m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).SetImageMean(d);
        }

        /// <summary>
        /// Returns the image mean for the Training data source of a given data set.
        /// </summary>
        /// <param name="nDatasetId">Specifies the data set to use.</param>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        public SimpleDatum QueryItemMeanFromDataset(int nDatasetId)
        {
            DatasetEx2 ds = m_colDatasets[m_nStrIDHashCode].FindDataset(nDatasetId);
            if (ds == null)
                return null;

            return QueryItemMean(ds.Descriptor.TrainingSource.ID);
        }

        /// <summary>
        /// Queries the image mean for a data source from the database on disk.
        /// </summary>
        /// <remarks>
        /// If the image mean does not exist in the database, one is created, saved
        /// and then returned.
        /// </remarks>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        public SimpleDatum QueryItemMeanFromDb(int nSrcId)
        {
            SimpleDatum sd = QueryItemMean(nSrcId);

            if (sd != null)
                return sd;

            sd = GetItemMean(nSrcId);
            SaveImageMean(nSrcId, sd, false);

            return sd;
        }

        /// <summary>
        /// Saves the image mean to a data source on the database on disk.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="d">Specifies a SimpleDatum containing the image mean.</param>
        /// <param name="bUpdate">Specifies whether or not to update the mean image.</param>
        public void SaveImageMean(int nSrcId, SimpleDatum d, bool bUpdate)
        {
            if (m_colDatasets.ContainsKey(m_nStrIDHashCode))
                m_colDatasets[m_nStrIDHashCode].SaveImageMean(nSrcId, d, bUpdate);
        }

        /// <summary>
        /// Query the image mean for a data source and mask out (set to 0) all of the image except for the last columns.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns></returns>
        public SimpleDatum QueryItemMean(int nSrcId)
        {
            if (!m_colDatasets.ContainsKey(m_nStrIDHashCode))
                return null;

            return m_colDatasets[m_nStrIDHashCode].QueryImageMean(nSrcId);
        }

        /// <summary>
        /// Returns whether or not the image mean exists in the disk-based database for a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>Returns <i>true</i> if the image mean exists, <i>false</i> otherwise.</returns>
        public bool DoesImageMeanExists(int nSrcId)
        {
            using (DNNEntities entities = EntitiesConnection.CreateEntities())
            {
                List<Source> rgSrc = entities.Sources.Where(p => p.ID == nSrcId).ToList();
                if (rgSrc.Count == 0)
                    return false;

                IQueryable<RawImageMean> iQuery = entities.RawImageMeans.Where(p => p.SourceID == nSrcId);
                if (iQuery != null)
                {
                    List<RawImageMean> rgMean = iQuery.ToList();
                    if (rgMean.Count > 0)
                        return true;
                }

                return false;
            }
        }

        #endregion // Image Mean

        #region Datasets

        /// <summary>
        /// Load another 'secondary' dataset.
        /// </summary>
        /// <remarks>
        /// The primary dataset should be loaded using one of the 'Initialize' methods.  This method is provided to allow for loading
        /// multiple datasets.
        /// </remarks>
        /// <param name="strDs">Specifies the name of the data set.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>When loaded, the handle to the default query state is returned, otherwise 0 is returned.</returns>
        public long LoadDatasetByName(string strDs, string strEvtCancel = null)
        {
            int nDsId = m_factory.GetDatasetID(strDs);
            return LoadDatasetByID(nDsId, strEvtCancel);
        }

        /// <summary>
        /// Load another 'secondary' dataset.
        /// </summary>
        /// <remarks>
        /// The primary dataset should be loaded using one of the 'Initialize' methods.  This method is provided to allow for loading
        /// multiple datasets.
        /// </remarks>
        /// <param name="nDsId">Specifies the ID of the data set.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>When loaded, the handle to the default query state is returned, otherwise 0 is returned.</returns>
        public long LoadDatasetByID(int nDsId, string strEvtCancel = null)
        {
            if (!m_evtInitialized.WaitOne(0))
                throw new Exception("The image database must be initialized first before a secondary dataset can be loaded.");

            if (m_evtInitializing.WaitOne(0))
                throw new Exception("The image database is in the process of being initialized.");

            DatasetEx2 ds = m_colDatasets[m_nStrIDHashCode].FindDataset(nDsId);
            if (ds != null)
                return 0;

            DatasetDescriptor desc = m_factory.LoadDataset(nDsId);
            if (desc == null)
                throw new Exception("Could not find dataset with ID = " + nDsId.ToString());

            if (!m_colDatasets.ContainsKey(m_nStrIDHashCode))
                throw new Exception("The image database was not initialized properly.");

            DatasetExCollection2 col = m_colDatasets[m_nStrIDHashCode];
            DatasetEx2 ds0 = new DatasetEx2(m_userGuid, m_factory, m_random);

            if (OnCalculateImageMean != null)
                ds0.OnCalculateImageMean += OnCalculateImageMean;

            if (m_log != null)
                m_log.WriteLine("Loading dataset '" + desc.Name + "'...");

            CancelEvent evtCancel = new CancelEvent(strEvtCancel);
            List<WaitHandle> rgAbort = new List<WaitHandle>(evtCancel.Handles);

            long lQueryState = ds0.Initialize(desc, rgAbort.ToArray(), 0, 0, m_log, m_loadMethod, false, m_nLoadLimit);
            if (lQueryState == 0)
            {
                col.Dispose();
                return 0;
            }

            if (m_log != null)
                m_log.WriteLine("Dataset '" + desc.Name + "' loaded.");

            col.Add(ds0);

            return lQueryState;
        }

        /// <summary>
        /// Load another, 'secondary' dataset.
        /// </summary>
        /// <remarks>
        /// The primary dataset should be loaded using one of the 'Initialize' methods.  This method is provided to allow for loading
        /// multiple datasets.
        /// </remarks>
        /// <param name="nDsId">Specifies the ID of the data set.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>When loaded, the handle to the default query state is returned, otherwise 0 is returned.</returns>
        public bool LoadDatasetByID1(int nDsId, string strEvtCancel = null)
        {
            long lHandle = LoadDatasetByID(nDsId, strEvtCancel);
            if (lHandle == 0)
                return false;

            return true;
        }

        /// <summary>
        /// Load another, 'secondary' dataset.
        /// </summary>
        /// <remarks>
        /// The primary dataset should be loaded using one of the 'Initialize' methods.  This method is provided to allow for loading
        /// multiple datasets.
        /// </remarks>
        /// <param name="strDs">Specifies the name of the data set.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>When loaded, the handle to the default query state is returned, otherwise 0 is returned.</returns>
        public bool LoadDatasetByName1(string strDs, string strEvtCancel = null)
        {
            long lHandle = LoadDatasetByName(strDs, strEvtCancel);
            if (lHandle == 0)
                return false;

            return true;
        }

        /// <summary>
        /// Reload a data set.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the data set.</param>
        /// <returns>If the data set is found, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool ReloadDataset(int nDsId)
        {
            DatasetEx2 ds = m_colDatasets[m_nStrIDHashCode].FindDataset(nDsId);
            if (ds != null)
            {
                ds.ResetLabels();
                return true;
            }

            return false;
        }

        /// <summary>
        /// The UnloadDataset method removes the dataset specified from memory.
        /// </summary>
        /// <param name="strDataset">Specifies the dataset to remove.</param>
        /// <returns>If found and removed, this function returns <i>true</i>, otherwise <i>false</i> is returned.</returns>
        public bool UnloadDatasetByName(string strDataset)
        {
            bool bRemoved = false;

            lock (m_syncObject)
            {
                if (m_colDatasets.ContainsKey(m_nStrIDHashCode))
                {
                    DatasetExCollection2 col = m_colDatasets[m_nStrIDHashCode];
                    DatasetEx2 ds = col.FindDataset(strDataset);
                    if (ds != null)
                    {
                        if (m_log != null)
                            m_log.WriteLine("Unloading dataset '" + ds.DatasetName + "'.");

                        ds.Unload(false);
                        GC.Collect();
                        bRemoved = true;
                    }
                }
            }

            return bRemoved;
        }

        /// <summary>
        /// The UnloadDataset method removes the dataset specified from memory.
        /// </summary>
        /// <param name="nDataSetID">Specifies the dataset ID to remove.</param>
        /// <remarks>Specifiying a dataset ID of -1 directs the UnloadDatasetById to unload ALL datasets loaded.</remarks>
        /// <returns>If found and removed, this function returns <i>true</i>, otherwise <i>false</i> is returned.</returns>
        public bool UnloadDatasetById(int nDataSetID)
        {
            bool bRemoved = false;

            lock (m_syncObject)
            {
                if (m_colDatasets.ContainsKey(m_nStrIDHashCode))
                {
                    DatasetExCollection2 col = m_colDatasets[m_nStrIDHashCode];

                    foreach (DatasetEx2 ds in col)
                    {
                        if (ds != null)
                        {
                            if (ds.DatasetID == nDataSetID || nDataSetID == -1)
                            {
                                if (m_log != null)
                                    m_log.WriteLine("Unloading dataset '" + ds.DatasetName + "'.");

                                ds.Unload(false);
                                GC.Collect();
                                bRemoved = true;
                            }
                        }
                    }
                }
            }

            return bRemoved;
        }

        /// <summary>
        /// Returns the percentage that a dataset is loaded into memory.
        /// </summary>
        /// <param name="strDataset">Specifies the name of the dataset.</param>
        /// <param name="dfTraining">Specifies the percent of training images that are loaded.</param>
        /// <param name="dfTesting">Specifies the percent of testing images that are loaded.</param>
        /// <returns>The current image load percent for the dataset is returned..</returns>
        public double GetDatasetLoadedPercentByName(string strDataset, out double dfTraining, out double dfTesting)
        {
            dfTraining = 0;
            dfTesting = 0;

            if (!m_colDatasets.ContainsKey(m_nStrIDHashCode))
                return 0;

            DatasetExCollection2 col = m_colDatasets[m_nStrIDHashCode];
            DatasetEx2 ds = col.FindDataset(strDataset);

            if (ds == null)
                return 0;

            return ds.GetPercentageLoaded(out dfTraining, out dfTesting);
        }

        /// <summary>
        /// Returns the percentage that a dataset is loaded into memory.
        /// </summary>
        /// <param name="nDatasetID">Specifies the ID of the dataset.</param>
        /// <param name="dfTraining">Specifies the percent of training images that are loaded.</param>
        /// <param name="dfTesting">Specifies the percent of testing images that are loaded.</param>
        /// <returns>The current image load percent for the dataset is returned..</returns>
        public double GetDatasetLoadedPercentById(int nDatasetID, out double dfTraining, out double dfTesting)
        {
            dfTraining = 0;
            dfTesting = 0;

            if (!m_colDatasets.ContainsKey(m_nStrIDHashCode))
                return 0;

            DatasetExCollection2 col = m_colDatasets[m_nStrIDHashCode];
            DatasetEx2 ds = col.FindDataset(nDatasetID);

            if (ds == null)
                return 0;

            return ds.GetPercentageLoaded(out dfTraining, out dfTesting);
        }

        /// <summary>
        /// Returns the DatasetDescriptor for a given data set ID.
        /// </summary>
        /// <param name="nDsId">Specifies the data set ID.</param>
        /// <returns>The dataset Descriptor is returned.</returns>
        public DatasetDescriptor GetDatasetById(int nDsId)
        {
            DatasetEx2 ds = m_colDatasets[m_nStrIDHashCode].FindDataset(nDsId);
            return ds.Descriptor;
        }

        /// <summary>
        /// Returns the DatasetDescriptor for a given data set name.
        /// </summary>
        /// <param name="strDs">Specifies the data set name.</param>
        /// <returns>The dataset Descriptor is returned.</returns>
        public DatasetDescriptor GetDatasetByName(string strDs)
        {
            DatasetEx2 ds = m_colDatasets[m_nStrIDHashCode].FindDataset(strDs);
            return ds.Descriptor;
        }

        /// <summary>
        /// Returns a data set ID given its name.
        /// </summary>
        /// <param name="strDs">Specifies the data set name.</param>
        /// <returns>The data set ID is returned.</returns>
        public int GetDatasetID(string strDs)
        {
            DatasetDescriptor ds = GetDatasetByName(strDs);
            if (ds == null)
                return 0;

            return ds.ID;
        }

        /// <summary>
        /// Returns a data set name given its ID.
        /// </summary>
        /// <param name="nDsId">Specifies the data set ID.</param>
        /// <returns>The data set name is returned.</returns>
        public string GetDatasetName(int nDsId)
        {
            DatasetDescriptor ds = GetDatasetById(nDsId);
            if (ds == null)
                return null;

            return ds.Name;
        }

        /// <summary>
        /// Reloads the images of a data source.
        /// </summary>
        /// <param name="nSrcID">Specifies the ID of the data source.</param>
        /// <returns>If the data source is found, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool ReloadImageSet(int nSrcID)
        {
            ImageSet2 imgset = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcID);
            if (imgset != null)
            {
                imgset.ResetLabels();
                return true;
            }

            return false;
        }

        #endregion // Datasets

        #region Sources

        /// <summary>
        /// Returns the SourceDescriptor for a given data source ID.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The SourceDescriptor is returned.</returns>
        public SourceDescriptor GetSourceById(int nSrcId)
        {
            ImageSet2 imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId);
            if (imgSet != null)
                return imgSet.Source;

            return null;
        }

        /// <summary>
        /// Returns the SourceDescriptor for a given data source name.
        /// </summary>
        /// <param name="strSrc">Specifies the data source name.</param>
        /// <returns>The SourceDescriptor is returned.</returns>
        public SourceDescriptor GetSourceByName(string strSrc)
        {
            ImageSet2 imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(strSrc);
            if (imgSet != null)
                return imgSet.Source;

            return null;
        }

        /// <summary>
        /// Returns a data source ID given its name.
        /// </summary>
        /// <param name="strSrc">Specifies the data source name.</param>
        /// <returns>The data source ID is returned.</returns>
        public int GetSourceID(string strSrc)
        {
            SourceDescriptor desc = GetSourceByName(strSrc);
            if (desc == null)
                return 0;

            return desc.ID;
        }

        /// <summary>
        /// Returns a data source name given its ID.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The data source name is returned.</returns>
        public string GetSourceName(int nSrcId)
        {
            SourceDescriptor desc = GetSourceById(nSrcId);
            if (desc == null)
                return null;

            return desc.Name;
        }

        #endregion // Sources

        #region Labels

        /// <summary>
        /// Returns a list of LabelDescriptor%s associated with the labels within a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The list of LabelDescriptor%s is returned.</returns>
        public List<LabelDescriptor> GetLabels(int nSrcId)
        {
            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).GetLabels();
        }

        /// <summary>
        /// Returns the text name of a given label within a data source. 
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <returns>The laben name is returned as a string.</returns>
        public string GetLabelName(int nSrcId, int nLabel)
        {
            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).GetLabelName(nLabel);
        }

        /// <summary>
        /// Sets the label mapping to the database for a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="map">Specifies the label mapping to set.</param>
        public void SetLabelMapping(int nSrcId, LabelMapping map)
        {
            m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).SetLabelMapping(map);
        }

        /// <summary>
        /// Updates the label mapping in the database for a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="nNewLabel">Specifies a new label.</param>
        /// <param name="rgOriginalLabels">Specifies the original lables that are mapped to the new label.</param>
        public void UpdateLabelMapping(int nSrcId, int nNewLabel, List<int> rgOriginalLabels)
        {
            m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).UpdateLabelMapping(nNewLabel, rgOriginalLabels);
        }

        /// <summary>
        /// Resets all labels within a data source, used by a project, to their original labels.
        /// </summary>
        /// <param name="nProjectId">Specifies the ID of the project.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        public void ResetLabels(int nProjectId, int nSrcId)
        {
            m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).ResetLabels(nProjectId);
        }

        /// <summary>
        /// Updates the number of images of each label within a data source.
        /// </summary>
        /// <param name="nProjectID">Specifies a project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        public void UpdateLabelCounts(int nProjectID, int nSrcId)
        {
            m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).UpdateLabelCounts(nProjectID);
        }

        /// <summary>
        /// Returns a label lookup of counts for a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>A dictionary containing label,count pairs is returned.</returns>
        public Dictionary<int, int> LoadLabelCounts(int nSrcId)
        {
            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).LoadLabelCounts();
        }

        /// <summary>
        /// Returns a string with all label counts for a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>A string containing all label counts is returned.</returns>
        public string GetLabelCountsAsTextFromSourceId(int nSrcId)
        {
            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).GetLabelCountsAsText();
        }

        /// <summary>
        /// Returns a string with all label counts for a data source.
        /// </summary>
        /// <param name="strSource">Specifies the name of the data source.</param>
        /// <returns>A string containing all label counts is returned.</returns>
        public string GetLabelCountsAsTextFromSourceName(string strSource)
        {
            return m_colDatasets[m_nStrIDHashCode].FindImageset(strSource).GetLabelCountsAsText();
        }

        #endregion // Labels

        #region Boosts

        /// <summary>
        /// Reset all in-memory image boosts.
        /// </summary>
        /// <remarks>
        /// This does not impact the boost setting within the physical database.
        /// </remarks>
        /// <param name="nSrcId">Specifies the source ID of the data set to reset.</param>
        public void ResetAllBoosts(int nSrcId)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });
            if (nWait == 0)
                return;

            DatasetEx2 ds = m_colDatasets[m_nStrIDHashCode].FindDatasetFromSource(nSrcId);
            ds.ResetAllBoosts();
        }

        #endregion // Boosts

        #region Results

        /// <summary>
        /// Query all results for a given data source.
        /// </summary>
        /// <param name="strSource">Specifies the data source who's results are to be returned.</param>
        /// <param name="bRequireExtraData">specifies whether or not the Extra 'target' data is required or not.</param>
        /// <param name="nMax">Optionally, specifies the maximum number of items to load.</param>
        /// <returns>Each result is returned in a SimpleResult object.</returns>
        public List<SimpleResult> GetAllResults(string strSource, bool bRequireExtraData, int nMax = -1)
        {
            return m_colDatasets[m_nStrIDHashCode].FindImageset(strSource).GetAllResults(bRequireExtraData, nMax);
        }

        #endregion // Results
    }
}
