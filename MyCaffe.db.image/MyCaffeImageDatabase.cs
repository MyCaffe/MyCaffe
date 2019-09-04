using System;
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
    /// The MyCaffeImageDatabase provides an enhanced in-memory image database used for quick image retrieval.
    /// </summary>
    /// <remarks>
    /// The MyCaffeImageDatbase manages a set of data sets, where each data sets comprise a pair of data sources: one source 
    /// for training and another source for testing.  Each data source contains a list of images and a list of label sets
    /// that point back into the list of images.  This organization allows for quick image selection by image or by label
    /// set and then by image from within the label set.
    /// </remarks>
    public partial class MyCaffeImageDatabase : Component, IXImageDatabase
    {
        DatasetFactory m_factory;
        string m_strID = "";
        int m_nStrIDHashCode = 0;
        int m_nMaskOutAllButLastColumns = 0;
        EventWaitHandle m_evtInitializing = null;
        EventWaitHandle m_evtInitialized = null;
        EventWaitHandle m_evtAbortInitialization = null;
        bool m_bEnabled = false;
        static object m_syncObject = new object();
        static Dictionary<int, DatasetExCollection> m_colDatasets = new Dictionary<int, DatasetExCollection>();
        static Dictionary<int, LabelMappingCollection> m_rgLabelMappings = new Dictionary<int, LabelMappingCollection>();
        Dictionary<int, SimpleDatum> m_rgMeanCache = new Dictionary<int, SimpleDatum>();
        double m_dfSuperBoostProbability = 0;
        CryptoRandom m_random = new CryptoRandom();
        IMGDB_IMAGE_SELECTION_METHOD m_imageSelectionMethod = IMGDB_IMAGE_SELECTION_METHOD.RANDOM;
        IMGDB_LABEL_SELECTION_METHOD m_labelSelectionMethod = IMGDB_LABEL_SELECTION_METHOD.RANDOM;
        Log m_log;
        IMAGEDB_LOAD_METHOD m_loadMethod = IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND;
        int m_nLoadLimit = 0;
        int m_nPadW = 0;
        int m_nPadH = 0;
        Guid m_userGuid;

        /// <summary>
        /// The OnCalculateImageMean event fires each time the MyCaffeImageDatabase wants to access the image mean for a data set.
        /// </summary>
        public event EventHandler<CalculateImageMeanArgs> OnCalculateImageMean;

        /// <summary>
        /// The MyCaffeImageDatabase constructor.
        /// </summary>
        /// <param name="log">The Log for output.</param>
        /// <param name="strId">Specifies an identifier for this in memory database instance (default = "default").</param>
        public MyCaffeImageDatabase(Log log = null, string strId = "default")
        {
            m_factory = new DatasetFactory();
            m_userGuid = Guid.NewGuid();
            m_log = log;
            InitializeComponent();
            init(strId);
        }

        /// <summary>
        /// The MyCaffeImageDatabase constructor.
        /// </summary>
        /// <param name="container">Specifies a container.</param>
        public MyCaffeImageDatabase(IContainer container)
        {
            container.Add(this);

            InitializeComponent();
            init();
        }

        private void init(string strId = "")
        {
            int nProcessID = Process.GetCurrentProcess().Id;

            m_strID = strId;
            m_nStrIDHashCode = strId.GetHashCode();
            m_evtInitializing = new EventWaitHandle(false, EventResetMode.ManualReset, "__CAFFE_IMAGEDATABASE__INITIALIZING__" + nProcessID.ToString());
            m_evtInitialized = new EventWaitHandle(false, EventResetMode.ManualReset, "__CAFFE_IMAGEDATABASE__INITIALIZED__" + nProcessID.ToString());
            m_evtAbortInitialization = new EventWaitHandle(false, EventResetMode.ManualReset, "__CAFFE_IMAGEDATABASE__ABORT_INITIALIZE__" + nProcessID.ToString());
        }

        private void dispose()
        {
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
        /// Set the database instance to use.
        /// </summary>
        /// <param name="strInstance">Specifies the instance name to use in '.\\name' format.</param>
        public void SetInstance(string strInstance)
        {
            MyCaffe.db.image.EntitiesConnection.GlobalDatabaseServerName = strInstance;
        }

        /// <summary>
        /// Returns the label/image selection methods based on the SettingsCaffe settings.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <returns>The label/image selection method is returned.</returns>
        public static Tuple<IMGDB_LABEL_SELECTION_METHOD, IMGDB_IMAGE_SELECTION_METHOD> GetSelectionMethod(SettingsCaffe s)
        {
            IMGDB_IMAGE_SELECTION_METHOD imageSelectionMethod = IMGDB_IMAGE_SELECTION_METHOD.NONE;
            IMGDB_LABEL_SELECTION_METHOD labelSelectionMethod = IMGDB_LABEL_SELECTION_METHOD.NONE;

            if (s.EnableRandomInputSelection)
                imageSelectionMethod |= IMGDB_IMAGE_SELECTION_METHOD.RANDOM;

            if (s.SuperBoostProbability > 0)
                imageSelectionMethod |= IMGDB_IMAGE_SELECTION_METHOD.BOOST;

            if (s.EnableLabelBalancing)
            {
                labelSelectionMethod |= IMGDB_LABEL_SELECTION_METHOD.RANDOM;

                if (s.EnableLabelBoosting)
                    labelSelectionMethod |= IMGDB_LABEL_SELECTION_METHOD.BOOST;
            }
            else
            {
                if (s.EnablePairInputSelection)
                    imageSelectionMethod |= IMGDB_IMAGE_SELECTION_METHOD.PAIR;
            }

            return new Tuple<IMGDB_LABEL_SELECTION_METHOD, IMGDB_IMAGE_SELECTION_METHOD>(labelSelectionMethod, imageSelectionMethod);
        }

        /// <summary>
        /// Returns the label/image selection methods based on the ProjectEx settings.
        /// </summary>
        /// <param name="p">Specifies the project.</param>
        /// <returns>The label/image selection method is returned.</returns>
        public static Tuple<IMGDB_LABEL_SELECTION_METHOD, IMGDB_IMAGE_SELECTION_METHOD> GetSelectionMethod(ProjectEx p)
        {
            IMGDB_IMAGE_SELECTION_METHOD imageSelectionMethod = IMGDB_IMAGE_SELECTION_METHOD.NONE;
            IMGDB_LABEL_SELECTION_METHOD labelSelectionMethod = IMGDB_LABEL_SELECTION_METHOD.NONE;

            if (p.EnableRandomSelection)
                imageSelectionMethod |= IMGDB_IMAGE_SELECTION_METHOD.RANDOM;

            if (p.EnableLabelBalancing)
            {
                labelSelectionMethod |= IMGDB_LABEL_SELECTION_METHOD.RANDOM;

                if (p.EnableLabelBoosting)
                    labelSelectionMethod |= IMGDB_LABEL_SELECTION_METHOD.BOOST;
            }
            else
            {
                if (p.EnablePairSelection)
                    imageSelectionMethod |= IMGDB_IMAGE_SELECTION_METHOD.PAIR;
            }

            return new Tuple<IMGDB_LABEL_SELECTION_METHOD, IMGDB_IMAGE_SELECTION_METHOD>(labelSelectionMethod, imageSelectionMethod);
        }


        /// <summary>
        /// Returns the label and image selection method used.
        /// </summary>
        /// <returns>A KeyValue containing the Label and Image selection method.</returns>
        public Tuple<IMGDB_LABEL_SELECTION_METHOD, IMGDB_IMAGE_SELECTION_METHOD> GetSelectionMethod()
        {
            return new Tuple<IMGDB_LABEL_SELECTION_METHOD, IMGDB_IMAGE_SELECTION_METHOD>(m_labelSelectionMethod, m_imageSelectionMethod);
        }

        /// <summary>
        /// Sets the label and image selection methods.
        /// </summary>
        /// <param name="lbl">Specifies the label selection method or <i>null</i> to ignore.</param>
        /// <param name="img">Specifies the image selection method or <i>null</i> to ignore.</param>
        public void SetSelectionMethod(IMGDB_LABEL_SELECTION_METHOD? lbl, IMGDB_IMAGE_SELECTION_METHOD? img)
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

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="strDs">Specifies the data set to load.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        public bool InitializeWithDsName(SettingsCaffe s, string strDs, string strEvtCancel = null)
        {
            return InitializeWithDs(s, new DatasetDescriptor(strDs), strEvtCancel);
        }

        /// <summary>
        /// Initializes the image database.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <param name="ds">Specifies the data set to load.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        public bool InitializeWithDs(SettingsCaffe s, DatasetDescriptor ds, string strEvtCancel = null)
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
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        public bool InitializeWithDsId(SettingsCaffe s, int nDataSetID, string strEvtCancel = null, int nPadW = 0, int nPadH = 0)
        {
            Tuple<IMGDB_LABEL_SELECTION_METHOD, IMGDB_IMAGE_SELECTION_METHOD> selMethod = GetSelectionMethod(s);

            m_nPadW = nPadW;
            m_nPadH = nPadH;
            m_nMaskOutAllButLastColumns = s.MaskAllButLastColumns;
            m_labelSelectionMethod = selMethod.Item1;
            m_imageSelectionMethod = selMethod.Item2;
            m_dfSuperBoostProbability = s.SuperBoostProbability;
            m_loadMethod = s.ImageDbLoadMethod;
            m_nLoadLimit = s.ImageDbLoadLimit;

            if (m_loadMethod == IMAGEDB_LOAD_METHOD.LOAD_EXTERNAL)
                m_loadMethod = IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND;

            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtInitialized, m_evtInitializing, m_evtAbortInitialization }, 0);

            lock (m_syncObject)
            {
                if (nWait != WaitHandle.WaitTimeout)
                {
                    if (nWait == 0)     // already initialized.
                    {
                        if (m_colDatasets.ContainsKey(m_nStrIDHashCode))
                        {
                            DatasetExCollection col = m_colDatasets[m_nStrIDHashCode];

                            if (m_log != null)
                                m_log.WriteLine("The MyCaffe Image Database is already initialized.");

                            DatasetEx ds = col.FindDataset(nDataSetID);
                            if (ds != null)
                            {
                                ds.AddUser(m_userGuid);
                                return true;
                            }
                        }
                    }
                    else
                    {
                        return false;
                    }
                }

                try
                {
                    m_evtInitializing.Set();

                    m_factory.SetLoadingParameters(s.ImageDbLoadDataCriteria, s.ImageDbLoadDebugData);

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

                    DatasetExCollection col = null;

                    if (m_colDatasets.ContainsKey(m_nStrIDHashCode))
                        col = m_colDatasets[m_nStrIDHashCode];
                    else
                        col = new DatasetExCollection();

                    if (m_evtAbortInitialization.WaitOne(0))
                    {
                        col.Dispose();
                        return false;
                    }

                    DatasetEx ds0 = new DatasetEx(m_userGuid, m_factory);

                    if (OnCalculateImageMean != null)
                        ds0.OnCalculateImageMean += OnCalculateImageMean;

                    if (m_log != null)
                        m_log.WriteLine("Loading dataset '" + ds.Name + "'...");

                    if (!ds0.Initialize(ds, rgAbort.ToArray(), nPadW, nPadH, m_log, m_loadMethod, m_nLoadLimit))
                    {
                        col.Dispose();
                        return false;
                    }

                    if (m_log != null)
                        m_log.WriteLine("Dataset '" + ds.Name + "' loaded.");

                    col.Add(ds0);

                    if (!m_colDatasets.ContainsKey(m_nStrIDHashCode))
                        m_colDatasets.Add(m_nStrIDHashCode, col);

                    UseTrainingImagesForTesting = s.UseTrainingSourceForTesting;

                    return true;
                }
                finally
                {
                    m_evtInitialized.Set();
                    m_evtInitializing.Reset();
                }
            }
        }

        /// <summary>
        /// Releases the image database, and if this is the last instance using the in-memory database, frees all memory used.
        /// </summary>
        /// <param name="nDsId">Optionally, specifies the dataset previously loaded.</param>
        public void CleanUp(int nDsId = 0)
        {
            lock (m_syncObject)
            {
                if (m_evtInitializing.WaitOne(0))
                {
                    m_evtAbortInitialization.Set();
                    return;
                }

                List<int> rgRemove = new List<int>();

                foreach (KeyValuePair<int, DatasetExCollection> col in m_colDatasets)
                {
                    DatasetExCollection colDs = col.Value;

                    if (colDs.RemoveUser(m_userGuid))
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

        /// <summary>
        /// When using a <i>Load Limit</i> that is greater than 0, this function loads the next set of images.
        /// </summary>
        /// <param name="strEvtCancel">Specifies the name of the Cancel Event to abort loading the images.</param>
        /// <returns>Returns <i>true</i> on success, <i>false</i> otherwise.</returns>
        public bool LoadNextSet(string strEvtCancel)
        {
            if (m_nLoadLimit == 0)
                return false;

            lock (m_syncObject)
            {
                if (!m_evtInitialized.WaitOne(0))
                    return false;

                List<WaitHandle> rgAbort = new List<WaitHandle>() { m_evtAbortInitialization };
                CancelEvent evtCancel;

                if (strEvtCancel != null)
                {
                    evtCancel = new CancelEvent(strEvtCancel);
                    rgAbort.AddRange(evtCancel.Handles);
                }

                if (!m_colDatasets.ContainsKey(m_nStrIDHashCode))
                    return false;

                DatasetExCollection col = m_colDatasets[m_nStrIDHashCode];
                col.Reset();

                foreach (DatasetEx ds in col)
                {
                    if (m_evtAbortInitialization.WaitOne(0))
                        return false;

                    if (m_log != null)
                        m_log.WriteLine("Loading next limited image set for dataset '" + ds.DatasetName + "'...");

                    if (!ds.Initialize(null, rgAbort.ToArray(), m_nPadW, m_nPadH, m_log, m_loadMethod, m_nLoadLimit))
                        return false;

                    if (m_log != null)
                        m_log.WriteLine("Dataset '" + ds.DatasetName + "' re-loaded.");
                }

                return true;
            }
        }

        /// <summary>
        /// Updates the label boosts for the images based on the label boosts set for the given project.
        /// </summary>
        /// <param name="nProjectID">Specifies the project ID in the database.</param>
        /// <param name="nSrcID">Specifies the data source ID.</param>
        public void UpdateLabelBoosts(int nProjectID, int nSrcID)
        {
            m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcID).UpdateLabelBoosts(nProjectID);
        }

        /// <summary>
        /// Returns the number of images in a given data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The number of images is returned.</returns>
        public int ImageCount(int nSrcId)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });

            if (nWait == 0)
                return 0;

            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).Count;
        }

        /// <summary>
        /// Returns the number of images in a given data source, optionally only counting the boosted images.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="bSuperBoostOnly">Specifies whether or not to only count the super boosted images.</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <returns>The number of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public int ImageCount(int nSrcId, bool bSuperBoostOnly, string strFilterVal = null, int? nBoostVal = null)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });

            if (nWait == 0)
                return 0;

            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).GetCount(bSuperBoostOnly, strFilterVal, nBoostVal);
        }

        /// <summary>
        /// Returns the array of images in the image set, possibly filtered with the filtering parameters.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <param name="bSuperboostOnly">Specifies whether or not to return images with super-boost.</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nBoostVal">Optionally, specifies the boost value that the boost must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <param name="nStartIdx">Optionally, specifies a starting index from which the query is to start within the set of images (default = 0).</param>
        /// <param name="nQueryCount">Optionally, specifies a number of images to retrieve within the set (default = int.MaxValue).</param>
        /// <returns>The list of images is returned.</returns>
        /// <remarks>When using the 'nBoostValue' negative values are used to test the exact match of the boost value with the absolute value of the 'nBoostValue', ande
        /// positive values are used to test for boost values that are greater than or equal to the 'nBoostValue'.</remarks>
        public List<SimpleDatum> GetImages(int nSrcId, bool bSuperboostOnly, string strFilterVal = null, int? nBoostVal = null, int nStartIdx = 0, int nQueryCount = int.MaxValue)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });

            if (nWait == 0)
                return null;

            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).GetImages(bSuperboostOnly, strFilterVal, nBoostVal, nStartIdx, nQueryCount);
        }

        /// <summary>
        /// Get a set of images, listed in chronological order starting at the next date greater than or equal to 'dt'.
        /// </summary>
        /// <param name="nSrcId">Specifies the databse ID of the data source.</param>
        /// <param name="dt">Specifies the start date of the images sought.</param>
        /// <param name="nImageCount">Specifies the number of images to retrieve.</param>
        /// <param name="strFilterVal">Optionally, specifies the filter value that the description must match (default = <i>null</i>, which ignores this parameter).</param>
        /// <returns>The list of SimpleDatum is returned.</returns>
        /// <remarks> IMPORTANT: You must call Sort(ByDesc|ByDate) before using this function to ensure all loaded images are ordered by their descriptions then by their time.</remarks>
        public List<SimpleDatum> GetImagesByDate(int nSrcId, DateTime dt, int nImageCount, string strFilterVal = null)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });

            if (nWait == 0)
                return null;

            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).GetImages(dt, nImageCount, strFilterVal);
        }

        /// <summary>
        /// Sort the internal images.
        /// </summary>
        /// <param name="nSrcId">Specifies the database ID of the data source.</param>
        /// <param name="method">Specifies the sorting method.</param>
        /// <returns>If the sorting is successful, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Sort(int nSrcId, IMGDB_SORT method)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });

            if (nWait == 0)
                return false;

            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).Sort(method);
        }

        /// <summary>
        /// Create a dynamic dataset organized by time from a pre-existing dataset.
        /// </summary>
        /// <param name="nDsId">Specifies the database ID of the dataset to copy.</param>
        /// <returns>The dataset ID of the newly created dataset is returned.</returns>
        public int CreateDatasetOranizedByTime(int nDsId)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });

            if (nWait == 0)
                return 0;

            // If we have already created a re-organized dataset,
            //  return its DatasetID.
            foreach (DatasetEx ds1 in m_colDatasets[m_nStrIDHashCode])
            {
                if (ds1.OriginalDatasetID == nDsId)
                    return ds1.DatasetID;
            }

            DatasetEx ds = m_colDatasets[m_nStrIDHashCode].FindDataset(nDsId);
            DatasetEx dsNew = ds.Clone(true);

            m_colDatasets[m_nStrIDHashCode].Add(dsNew);

            return dsNew.Descriptor.ID;
        }

        /// <summary>
        /// Delete a dataset created with CreateDatasetOrganizedByTime.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID of the created dataset.</param>
        /// <returns>If successful, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool DeleteCreatedDataset(int nDsId)
        {
            if (nDsId >= 0)
                throw new Exception("The dataset specified is not a dynamic dataset.");

            DatasetEx ds = m_colDatasets[m_nStrIDHashCode].FindDataset(nDsId);
            if (ds == null)
                return false;

            return m_colDatasets[m_nStrIDHashCode].RemoveDataset(ds);
        }

        /// <summary>
        /// Delete all datasets created with CreateDatasetOrganizedByTime
        /// </summary>
        public void DeleteAllCreatedDatasets()
        {
            m_colDatasets[m_nStrIDHashCode].RemoveCreatedDatasets();
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
        /// <returns>The image SimpleDatum is returned.</returns>
        public SimpleDatum QueryImage(int nSrcId, int nIdx, IMGDB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, IMGDB_IMAGE_SELECTION_METHOD? imageSelectionOverride = null, int? nLabel = null, bool bLoadDataCriteria = false, bool bLoadDebugData = false)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });

            if (nWait == 0)
                return null;

            IMGDB_LABEL_SELECTION_METHOD labelSelectionMethod = m_labelSelectionMethod;
            IMGDB_IMAGE_SELECTION_METHOD imageSelectionMethod = m_imageSelectionMethod;

            if (labelSelectionOverride.HasValue)
                labelSelectionMethod = labelSelectionOverride.Value;

            if (imageSelectionOverride.HasValue)
                imageSelectionMethod = imageSelectionOverride.Value;
            else if (SelectFromBoostOnly)
                imageSelectionMethod |= IMGDB_IMAGE_SELECTION_METHOD.BOOST;

            SimpleDatum sd = null;
            ImageSet imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId);
            LabelSet lblSet = null;

            if (nLabel.HasValue)
            {
                lblSet = imgSet.GetLabelSet(nLabel.Value);
                if (lblSet.IsLoaded)
                    sd = lblSet.GetImage(0, imageSelectionMethod);
            }
           
            if (m_loadMethod == IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND && (imageSelectionMethod & IMGDB_IMAGE_SELECTION_METHOD.PAIR) == IMGDB_IMAGE_SELECTION_METHOD.PAIR)
                throw new Exception("PAIR selection is not supported whith the LOAD_ON_DEMAND loading method.");

            if (sd == null)
            {
                sd = imgSet.GetImage(nIdx, labelSelectionMethod, imageSelectionMethod, m_log, bLoadDataCriteria, bLoadDebugData);
                if (sd == null)
                {
                    Exception err = new Exception("Could not acquire an image - re-index the dataset.");
                    m_log.WriteError(err);
                    throw err;
                }

                if (nLabel.HasValue)
                {
                    while (sd.Label != nLabel.Value)
                    {
                        sd = imgSet.GetImage(nIdx, labelSelectionMethod, IMGDB_IMAGE_SELECTION_METHOD.RANDOM, m_log, bLoadDataCriteria, bLoadDebugData);
                    }

                    lblSet.Add(sd);
                }
            }

            sd.MaskOutAllButLastColumns(m_nMaskOutAllButLastColumns, 0);
            return sd;
        }

        /// <summary>
        /// Returns the image with a given Raw Image ID.
        /// </summary>
        /// <param name="nImageID">Specifies the Raw Image ID.</param>
        /// <param name="rgSrcId">Specifies a set of source ID's to query from.</param>
        /// <returns>If found, the SimpleDatum of the Raw Image is returned, otherwise, <i>null</i> is returned.</returns>
        public SimpleDatum GetImage(int nImageID, params int[] rgSrcId)
        {
            int nWait = WaitHandle.WaitAny(new WaitHandle[] { m_evtAbortInitialization, m_evtInitialized });

            if (nWait == 0)
                return null;

            foreach (int nSrcId in rgSrcId)
            {
                ImageSet imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId);
                SimpleDatum sd = imgSet.GetImage(nImageID);

                if (sd != null)
                    return sd;
            }

            return null;
        }

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

            ImageSet imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId);
            imgSet.ResetAllBoosts();
        }

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
        /// Returns the DatasetDescriptor for a given data set ID.
        /// </summary>
        /// <param name="nDsId">Specifies the data set ID.</param>
        /// <returns>The dataset Descriptor is returned.</returns>
        public DatasetDescriptor GetDatasetById(int nDsId)
        {
            DatasetEx ds = m_colDatasets[m_nStrIDHashCode].FindDataset(nDsId);
            return ds.Descriptor;
        }

        /// <summary>
        /// Returns the DatasetDescriptor for a given data set name.
        /// </summary>
        /// <param name="strDs">Specifies the data set name.</param>
        /// <returns>The dataset Descriptor is returned.</returns>
        public DatasetDescriptor GetDatasetByName(string strDs)
        {
            DatasetEx ds = m_colDatasets[m_nStrIDHashCode].FindDataset(strDs);
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
        /// Returns the SourceDescriptor for a given data source ID.
        /// </summary>
        /// <param name="nSrcId">Specifies the data source ID.</param>
        /// <returns>The SourceDescriptor is returned.</returns>
        public SourceDescriptor GetSourceById(int nSrcId)
        {
            ImageSet imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId);
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
            ImageSet imgSet = m_colDatasets[m_nStrIDHashCode].FindImageset(strSrc);
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
        public int FindImageIndex(int nSrcId, DateTime dt, string strDescription)
        {
            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).FindImageIndex(dt, strDescription);
        }

        /// <summary>
        /// Returns the image mean for a data source.
        /// </summary>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The image mean is returned as a SimpleDatum.</returns>
        public SimpleDatum GetImageMean(int nSrcId)
        {
            if (m_evtAbortInitialization.WaitOne(0))
                return null;

            if (!m_evtInitialized.WaitOne(0))
            {
                if (m_rgMeanCache.Keys.Contains(nSrcId))
                    return m_rgMeanCache[nSrcId];
            }

            SimpleDatum sd = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).GetImageMean(null, null);
            sd.MaskOutAllButLastColumns(m_nMaskOutAllButLastColumns, 0);

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
        public SimpleDatum QueryImageMeanFromDataset(int nDatasetId)
        {
            DatasetEx ds = m_colDatasets[m_nStrIDHashCode].FindDataset(nDatasetId);
            if (ds == null)
                return null;

            return QueryImageMean(ds.Descriptor.TrainingSource.ID);
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
        public SimpleDatum QueryImageMeanFromDb(int nSrcId)
        {
            SimpleDatum sd = QueryImageMean(nSrcId);

            if (sd != null)
                return sd;

            sd = GetImageMean(nSrcId);
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
        public SimpleDatum QueryImageMean(int nSrcId)
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
        /// Delete all label boosts for a given data source associated with a given project.
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        public void DeleteLabelBoosts(int nProjectId, int nSrcId)
        {
            m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).DeleteLabelBoosts(nProjectId);
        }

        /// <summary>
        /// Add a label boost for a data source associated with a given project.
        /// </summary>
        /// <param name="nProjectID">Specifies the project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="dfBoost">Specifies the new boost for the label.</param>
        public void AddLabelBoost(int nProjectID, int nSrcId, int nLabel, double dfBoost)
        {
            m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).AddLabelBoost(nProjectID, nLabel, dfBoost);
        }

        /// <summary>
        /// Returns the label boosts as a text string for all boosted labels within a data source associated with a given project. 
        /// </summary>
        /// <param name="nProjectId">Specifies the project ID.</param>
        /// <param name="nSrcId">Specifies the ID of the data source.</param>
        /// <returns>The label boosts are returned as a text string.</returns>
        public string GetLabelBoostsAsTextFromProject(int nProjectId, int nSrcId)
        {
            return m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcId).GetLabelBoostsAsText(nProjectId);
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


        /// <summary>
        /// Load another 'secondary' dataset.
        /// </summary>
        /// <remarks>
        /// The primary dataset should be loaded using one of the 'Initialize' methods.  This method is provided to allow for loading
        /// multiple datasets.
        /// </remarks>
        /// <param name="strDs">Specifies the name of the data set.</param>
        /// <param name="strEvtCancel">Specifies the name of the CancelEvent used to cancel load operations.</param>
        /// <returns>When the dataset is loaded <i>true</i> is returned, otherwise if the dataset is already loaded <i>false</i> is returned.</returns>
        public bool LoadDatasetByName(string strDs, string strEvtCancel = null)
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
        /// <returns>When the dataset is loaded <i>true</i> is returned, otherwise if the dataset is already loaded <i>false</i> is returned.</returns>
        public bool LoadDatasetByID(int nDsId, string strEvtCancel = null)
        {
            if (!m_evtInitialized.WaitOne(0))
                throw new Exception("The image database must be initialized first before a secondary dataset can be loaded.");

            if (m_evtInitializing.WaitOne(0))
                throw new Exception("The image database is in the process of being initialized.");

            DatasetEx ds = m_colDatasets[m_nStrIDHashCode].FindDataset(nDsId);
            if (ds != null)
                return false;

            DatasetDescriptor desc = m_factory.LoadDataset(nDsId);
            if (desc == null)
                throw new Exception("Could not find dataset with ID = " + nDsId.ToString());

            if (!m_colDatasets.ContainsKey(m_nStrIDHashCode))
                throw new Exception("The image database was not initialized properly.");

            DatasetExCollection col = m_colDatasets[m_nStrIDHashCode];
            DatasetEx ds0 = new DatasetEx(m_userGuid, m_factory);

            if (OnCalculateImageMean != null)
                ds0.OnCalculateImageMean += OnCalculateImageMean;

            if (m_log != null)
                m_log.WriteLine("Loading dataset '" + desc.Name + "'...");

            CancelEvent evtCancel = new CancelEvent(strEvtCancel);
            List<WaitHandle> rgAbort = new List<WaitHandle>(evtCancel.Handles);

            if (!ds0.Initialize(desc, rgAbort.ToArray(), 0, 0, m_log, m_loadMethod, m_nLoadLimit))
            {
                col.Dispose();
                return false;
            }

            if (m_log != null)
                m_log.WriteLine("Dataset '" + desc.Name + "' loaded.");

            col.Add(ds0);

            return true;
        }

        /// <summary>
        /// Reload a data set.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the data set.</param>
        /// <returns>If the data set is found, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool ReloadDataset(int nDsId)
        {
            DatasetEx ds = m_colDatasets[m_nStrIDHashCode].FindDataset(nDsId);
            if (ds != null)
            {
                ds.ReloadLabelSets();
                return true;
            }

            return false;
        }

        /// <summary>
        /// Reloads the images of a data source.
        /// </summary>
        /// <param name="nSrcID">Specifies the ID of the data source.</param>
        /// <returns>If the data source is found, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool ReloadImageSet(int nSrcID)
        {
            ImageSet imgset = m_colDatasets[m_nStrIDHashCode].FindImageset(nSrcID);
            if (imgset != null)
            {
                imgset.ReloadLabelSets();
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
                    DatasetExCollection col = m_colDatasets[m_nStrIDHashCode];
                    DatasetEx ds = col.FindDataset(strDataset);
                    if (ds != null)
                    {
                        if (m_log != null)
                            m_log.WriteLine("Unloading dataset '" + ds.DatasetName + "'.");

                        ds.Unload();
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
                    DatasetExCollection col = m_colDatasets[m_nStrIDHashCode];

                    foreach (DatasetEx ds in col)
                    {
                        if (ds != null)
                        {
                            if (ds.DatasetID == nDataSetID || nDataSetID == -1)
                            {
                                if (m_log != null)
                                    m_log.WriteLine("Unloading dataset '" + ds.DatasetName + "'.");

                                ds.Unload();
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

            DatasetExCollection col = m_colDatasets[m_nStrIDHashCode];
            DatasetEx ds = col.FindDataset(strDataset);

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

            DatasetExCollection col = m_colDatasets[m_nStrIDHashCode];
            DatasetEx ds = col.FindDataset(nDatasetID);

            if (ds == null)
                return 0;

            return ds.GetPercentageLoaded(out dfTraining, out dfTesting);
        }

        /// <summary>
        /// Create the database used by the CaffeImageDatabase.
        /// </summary>
        /// <param name="strName">Specifies the name of the database (recommended value = "DNN").</param>
        /// <param name="strPath">Specifies the file path where the database is to be created.</param>
        /// <param name="strInstance">Optionally, specifies the SQL Instance.  By default this is <i>null</i>, which sets the instance to the default global instance.</param>
        public static void CreateDatabase(string strName, string strPath, string strInstance = null)
        {
            if (strInstance == null)
                strInstance = EntitiesConnection.GlobalDatabaseServerName;

            DatabaseManagement dbMgr = new DatabaseManagement(strName, strPath, strInstance);
            Exception excpt = dbMgr.CreateDatabase();

            if (excpt != null)
                throw excpt;
        }
    }
}
