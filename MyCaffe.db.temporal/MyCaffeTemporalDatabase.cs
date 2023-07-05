using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.db.image;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

/// <summary>
/// The MyCaffe.db.temporal namespace contains all classes used to create the MyCaffeTemporalDatabase in-memory database.
/// </summary>
namespace MyCaffe.db.temporal
{
    /// <summary>
    /// [Temporal Database]
    /// The MyCaffeTemporalDatabase provides an enhanced in-memory temporal database used for quick temporal data retrieval.
    /// </summary>
    /// <remarks>
    /// The MyCaffeTemporalDatbase manages a set of data sets, where each data sets comprise a pair of data sources: one source 
    /// for training and another source for testing.  Each data source contains a set of temporal data streams.  
    /// This organization allows for quick temporal data selection and is used with temporal models such as the Temporal Fusion Transformer (TFT).
    /// </remarks>
    public partial class MyCaffeTemporalDatabase : Component, IXTemporalDatabaseBase
    {
        SettingsCaffe m_settings;
        PropertySet m_prop;
        CryptoRandom m_random = null;
        EventWaitHandle m_evtInitializing = null;
        EventWaitHandle m_evtInitialized = null;
        EventWaitHandle m_evtAbortInitialization = null;
        bool m_bEnabled = false;
        static object m_syncObject = new object();
        DB_ITEM_SELECTION_METHOD m_valueSelectionMethod = DB_ITEM_SELECTION_METHOD.RANDOM;
        DB_LABEL_SELECTION_METHOD m_itemSelectionMethod = DB_LABEL_SELECTION_METHOD.RANDOM;
        Log m_log;
        DB_LOAD_METHOD m_loadMethod = DB_LOAD_METHOD.LOAD_ON_DEMAND;
        int m_nLoadLimit = 0;
        Guid m_userGuid;
        DatasetCollection m_rgDataSets = new DatasetCollection();
        Dictionary<int, TemporalSet> m_rgTemporalSets = new Dictionary<int, TemporalSet>();
        
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="log">The Log for output.</param>
        /// <param name="prop">Specifies the initialization properties.</param>
        public MyCaffeTemporalDatabase(Log log, PropertySet prop)
        {
            m_log = log;
            m_prop = prop;
            InitializeComponent();
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="container">Specifies the container for the component.</param>
        public MyCaffeTemporalDatabase(IContainer container)
        {
            container.Add(this);

            InitializeComponent();
        }

        /// <summary>
        /// Cleanup all resources used by the dataset specified (or all when no dataset is specified).
        /// </summary>
        /// <param name="nDsId">Specifies the dataset to cleanup, or 0 for all datasets.</param>
        /// <param name="bForce">Not used.</param>
        public void CleanUp(int nDsId = 0, bool bForce = false)
        {
            m_rgTemporalSets.Clear();
            m_rgDataSets.CleanUp(nDsId);
        }

        /// <summary>
        /// Create the database tables used by the MyCaffeTemporalDatabase.
        /// </summary>
        /// <param name="ci">Specifies the connection information for the database (recommended value: db = 'DNN', server = '.')</param>
        /// <param name="bTemporalOnly">Specifies to only update the temporal tables.</param>
        public static void CreateDatabaseTables(ConnectInfo ci, bool bTemporalOnly)
        {
            if (ci == null)
                ci = EntitiesConnection.GlobalDatabaseConnectInfo;

            DatabaseManagement dbMgr = new DatabaseManagement(ci);
            if (bTemporalOnly)
            {
                dbMgr.CreateTemporalTables();
            }
            else
            {
                Exception excpt = dbMgr.CreateDatabase();

                if (excpt != null)
                    throw excpt;
            }
        }


        /// <summary>
        /// Reset the database indexes.
        /// </summary>
        /// <remarks>
        /// This method is only used when using sequential selection.
        /// </remarks>
        public void Reset()
        {
            foreach (KeyValuePair<int, TemporalSet> kv in m_rgTemporalSets)
            {
                kv.Value.Reset();
            }
        }

        /// <summary>
        /// Returns the total number of blocks in the database where one block is a set of (historical and future steps).
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <param name="phase">Specifies the phase who's data size is to be returned.</param>
        /// <param name="nHistoricalSteps">Specifies the number of historical steps.</param>
        /// <param name="nFutureSteps">Specifies the number of future steps.</param>
        /// <returns>The total number of blocks is returned.</returns>
        public int GetTotalSize(int nDsId, Phase phase, int nHistoricalSteps, int nFutureSteps)
        {
            DataSet ds = m_rgDataSets.Find(nDsId);
            if (ds == null)
                return 0;

            SourceDescriptor sd = (phase == Phase.TRAIN) ? ds.Dataset.TrainingSource : ds.Dataset.TestingSource;

            if (sd == null)
                return 0;

            if (sd.TemporalDescriptor == null)
                return 0;

            int nStepsPerBlock = nHistoricalSteps + nFutureSteps;
            int nTotalSteps = sd.TemporalDescriptor.ValueStreamDescriptors.Max(p => p.Steps);
            if (nTotalSteps < nStepsPerBlock)
                m_log.FAIL("Stream: " + sd.Name + " - The number of historical and future steps must be less than the number of steps in the temporal data stream.  The steps per block = " + nStepsPerBlock.ToString() + " and the total steps = " + nTotalSteps.ToString() + " items.");

            int nRowBlocks = nTotalSteps - nStepsPerBlock;
            int nTotal = sd.TemporalDescriptor.ValueItemDescriptors.Count * nRowBlocks;

            return nTotal;
        }

        /// <summary>
        /// Return the dataset descriptor of the dataset with the specified ID.
        /// </summary>
        /// <param name="nDsId">Specifies the ID of the dataset to find.</param>
        /// <returns>The DatasetDescriptor is returned when found, or null.</returns>
        public DatasetDescriptor GetDatasetById(int nDsId)
        {
            DataSet ds = m_rgDataSets.Find(nDsId);
            if (ds == null)
                return null;

            return ds.Dataset;
        }

        /// <summary>
        /// Return the dataset descriptor of the dataset with the specified dataset name.
        /// </summary>
        /// <param name="strDs">Specifies the name of the dataset to find.</param>
        /// <returns>The DatasetDescriptor is returned when found, or null.</returns>
        public DatasetDescriptor GetDatasetByName(string strDs)
        {
            DataSet ds = m_rgDataSets.Find(strDs);
            if (ds == null)
            {
                DatabaseLoader dbLoader = new DatabaseLoader();
                return dbLoader.LoadDatasetFromDb(strDs);
            }

            return ds.Dataset;
        }

        /// <summary>
        /// Return the dataset ID of the dataset with the specified dataset name.
        /// </summary>
        /// <param name="strDs">Specifies the dataset name.</param>
        /// <returns>The dataset ID is returned when found, or 0.</returns>
        public int GetDatasetID(string strDs)
        {
            return m_rgDataSets.GetDatasetID(strDs);
        }

        /// <summary>
        /// Return the load percentage for dataset ID.
        /// </summary>
        /// <param name="nDatasetID">Specifies the dataset ID.</param>
        /// <param name="dfTraining">Specifies the training percent loaded.</param>
        /// <param name="dfTesting">Specifies the testing percent loaded.</param>
        /// <returns>Returns the total load percent.</returns>
        public double GetDatasetLoadedPercentById(int nDatasetID, out double dfTraining, out double dfTesting)
        {
            DataSet ds = m_rgDataSets.Find(nDatasetID);
            if (ds == null)
            {
                dfTraining = 0;
                dfTesting = 0;
                return 0;
            }

            return ds.GetLoadPercent(out dfTraining, out dfTesting);
        }

        /// <summary>
        /// Return the load percentage for a dataset.
        /// </summary>
        /// <param name="strDataset">Specifies the dataset name.</param>
        /// <param name="dfTraining">Specifies the training percent loaded.</param>
        /// <param name="dfTesting">Specifies the testing percent loaded.</param>
        /// <returns>Returns the total load percent.</returns>
        public double GetDatasetLoadedPercentByName(string strDataset, out double dfTraining, out double dfTesting)
        {
            return GetDatasetLoadedPercentById(GetDatasetID(strDataset), out dfTraining, out dfTesting);
        }

        /// <summary>
        /// Returns the dataset name given its ID.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <returns>The dataset name is returned, or null if not found.</returns>
        public string GetDatasetName(int nDsId)
        {
            DataSet ds = m_rgDataSets.Find(nDsId);
            if (ds == null)
                return null;

            return ds.Dataset.Name;
        }

        /// <summary>
        /// Returns the item/value selection methods based on the SettingsCaffe settings.
        /// </summary>
        /// <param name="s">Specifies the caffe settings.</param>
        /// <returns>The item/value selection method is returned.</returns>
        public static Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD> GetSelectionMethod(SettingsCaffe s)
        {
            DB_ITEM_SELECTION_METHOD itemSelectionMethod = DB_ITEM_SELECTION_METHOD.NONE;
            DB_LABEL_SELECTION_METHOD valueSelectionMethod = DB_LABEL_SELECTION_METHOD.NONE;

            if (s.EnableRandomInputSelection)
                itemSelectionMethod |= DB_ITEM_SELECTION_METHOD.RANDOM;

            if (s.EnableLabelBalancing)
                valueSelectionMethod |= DB_LABEL_SELECTION_METHOD.RANDOM;

            return new Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD>(valueSelectionMethod, itemSelectionMethod);
        }

        /// <summary>
        /// Returns the item/value selection methods based on the ProjectEx settings.
        /// </summary>
        /// <param name="p">Specifies the project.</param>
        /// <returns>The item/value selection method is returned.</returns>
        public static Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD> GetSelectionMethod(ProjectEx p)
        {
            DB_ITEM_SELECTION_METHOD valueSelectionMethod = DB_ITEM_SELECTION_METHOD.NONE;
            DB_LABEL_SELECTION_METHOD itemSelectionMethod = DB_LABEL_SELECTION_METHOD.NONE;

            if (p.EnableRandomSelection)
                valueSelectionMethod |= DB_ITEM_SELECTION_METHOD.RANDOM;

            if (p.EnableLabelBalancing)
                itemSelectionMethod |= DB_LABEL_SELECTION_METHOD.RANDOM;

            return new Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD>(itemSelectionMethod, valueSelectionMethod);
        }

        /// <summary>
        /// Returns the item and value selection method used.
        /// </summary>
        /// <returns>A KeyValue containing the Item and Value selection method.</returns>
        public Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD> GetSelectionMethod()
        {
            return new Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD>(m_itemSelectionMethod, m_valueSelectionMethod);
        }

        /// <summary>
        /// Get the source given its ID.
        /// </summary>
        /// <param name="nSrcId">Specifies the Source ID.</param>
        /// <returns>Returns the source descriptor or null if not loaded.</returns>
        public SourceDescriptor GetSourceById(int nSrcId)
        {
            return m_rgDataSets.FindSourceByID(nSrcId);
        }

        /// <summary>
        /// Get the source given its name.
        /// </summary>
        /// <param name="strSrc">Specifies the Source Name.</param>
        /// <returns>Returns the source descriptor or null if not loaded.</returns>
        public SourceDescriptor GetSourceByName(string strSrc)
        {
            return m_rgDataSets.FindSourceByName(strSrc);
        }

        /// <summary>
        /// Get the source ID given its name.
        /// </summary>
        /// <param name="strSrc">Specifies the Source Name.</param>
        /// <returns>Returns the source ID or 0 if not loaded.</returns>
        public int GetSourceID(string strSrc)
        {
            SourceDescriptor sd = GetSourceByName(strSrc);
            if (sd == null)
                return 0;

            return sd.ID;
        }

        /// <summary>
        /// Get the source name given its ID.
        /// </summary>
        /// <param name="nSrcId">Specifies the Source Id.</param>
        /// <returns>Returns the source name or null if not loaded.</returns>
        public string GetSourceName(int nSrcId)
        {
            SourceDescriptor sd = GetSourceById(nSrcId);
            if (sd == null)
                return null;

            return sd.Name;
        }

        /// <summary>
        /// Reload the dataset with the specified dataset ID.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <returns>If reloaded successfully, returns true.</returns>
        public bool ReloadDataset(int nDsId) 
        {
            UnloadDatasetById(nDsId);
            return InitializeWithDsId1(m_settings, nDsId);
        }

        /// <summary>
        /// Unload the dataset specified by the dataset ID.
        /// </summary>
        /// <param name="nDsId">Specifies the dataset ID.</param>
        /// <returns>Returns true if unloaded successfully.</returns>
        public bool UnloadDatasetById(int nDsId)
        {
            DataSet ds = m_rgDataSets.Find(nDsId);
            if (ds != null)
            {
                m_rgDataSets.CleanUp(nDsId);
                return true;
            }

            return false;
        }

        /// <summary>
        /// Unload the dataset specified by the dataset name.
        /// </summary>
        /// <param name="strDataset">Specifies the dataset name.</param>
        /// <returns>Returns true if unloaded successfully.</returns>
        public bool UnloadDatasetByName(string strDataset) 
        {
            return UnloadDatasetById(GetDatasetID(strDataset));
        }

        /// <summary>
        /// Returns the version of the database (e.g., TEMPORAL).
        /// </summary>
        /// <returns>The database version is returned.</returns>
        public DB_VERSION GetVersion()
        {
            return DB_VERSION.TEMPORAL;
        }

        /// <summary>
        /// Set the initialization properties to use when initializing a dataset.
        /// </summary>
        /// <param name="prop">Specifies the initialization properties (see remarks).</param>
        /// <remarks>
        /// The initialization properties for the dataset must include the following properties:
        ///   NormalizeData (bool) - whether or not to use the normalized data or raw data.
        ///   HistoricalSteps (int) - value > 0 specifiying the number of historical steps to use.
        ///   FutureSteps (int) - value > 0 specifiying the number of future steps to use.
        ///   Chunks (int) - Optional, value > 0 specifying the number of chunks to use (default = 1024).
        /// </remarks>
        public void SetInitializationProperties(PropertySet prop)
        {
            m_prop = prop;
        }

        /// <summary>
        /// Initialize the database with the specified dataset ID.
        /// </summary>
        /// <param name="s">Specifies the initial settings that specify the load method and load limit.</param>
        /// <param name="nDataSetID">Specifies the dataset ID to load.</param>
        /// <param name="strEvtCancel">Specifies the name of a global cancel event.</param>
        /// <param name="nPadW">Not Used</param>
        /// <param name="nPadH">Not Used</param>
        /// <returns>If the datset is loaded, true is returned, otherwise false.</returns>
        /// <remarks>
        /// You must call SetInitializationProperties before calling the InitializeWithDsId method.
        /// </remarks>
        /// <exception cref="Exception">An exception is thrown on error, e.g, when missing initialization properties.</exception>
        public bool InitializeWithDsId1(SettingsCaffe s, int nDataSetID, string strEvtCancel = null, int nPadW = 0, int nPadH = 0)
        {
            m_settings = s;
            m_evtAbortInitialization = new AutoResetEvent(false);

            DataSet ds = m_rgDataSets.Find(nDataSetID);
            if (ds == null)
            {
                DatabaseLoader dsLoader = new DatabaseLoader();
                DatasetDescriptor dsd = dsLoader.LoadDatasetFromDb(nDataSetID);
                ds = m_rgDataSets.Add(dsd, m_log);
            }

            if (m_prop == null)
                throw new Exception("You must first call SetInitializationProperties with the properties to use for initialization.");

            bool bNormalizeData = m_prop.GetPropertyAsBool("NormalizedData", true);
            int nHistSteps = m_prop.GetPropertyAsInt("HistoricalSteps", 0);
            int nFutureSteps = m_prop.GetPropertyAsInt("FutureSteps", 0);
            int nChunks = m_prop.GetPropertyAsInt("Chunks", 1024);

            if (nHistSteps == 0)
                throw new Exception("The historical steps are missing from the properties, please add the 'HistoricalSteps' property with a value > 0.");
            if (nFutureSteps == 0)
                throw new Exception("The future steps are missing from the properties, please add the 'FutureSteps' property with a value > 0.");

            if (nHistSteps < 0 || nFutureSteps < 0)
                throw new Exception("The historical and future steps must be > 0.");

            return ds.Load(s.DbLoadMethod, s.DbLoadLimit, bNormalizeData, nHistSteps, nFutureSteps, nChunks, m_evtAbortInitialization);
        }

        /// <summary>
        /// Initialize the database with the specified dataset descriptor.
        /// </summary>
        /// <param name="s">Specifies the initial settings that specify the load method and load limit.</param>
        /// <param name="ds1">Specifies the dataset descriptor to load.</param>
        /// <param name="strEvtCancel">Specifies the name of a global cancel event.</param>
        /// <returns>If the datset is loaded, true is returned, otherwise false.</returns>
        /// <remarks>
        /// You must call SetInitializationProperties before calling the InitializeWithDsId method.
        /// </remarks>
        /// <exception cref="Exception">An exception is thrown on error, e.g, when missing initialization properties.</exception>
        public bool InitializeWithDs1(SettingsCaffe s, DatasetDescriptor ds1, string strEvtCancel = null)
        {
            return InitializeWithDsId1(s, ds1.ID, strEvtCancel);
        }

        /// <summary>
        /// Initialize the database with the specified dataset name.
        /// </summary>
        /// <param name="s">Specifies the initial settings that specify the load method and load limit.</param>
        /// <param name="strDs">Specifies the name of the dataset to load.</param>
        /// <param name="strEvtCancel">Specifies the name of a global cancel event.</param>
        /// <returns>If the datset is loaded, true is returned, otherwise false.</returns>
        /// <remarks>
        /// You must call SetInitializationProperties before calling the InitializeWithDsId method.
        /// </remarks>
        /// <exception cref="Exception">An exception is thrown on error, e.g, when missing initialization properties.</exception>
        public bool InitializeWithDsName1(SettingsCaffe s, string strDs, string strEvtCancel = null)
        {
            int nDsID = m_rgDataSets.GetDatasetID(strDs);
            if (nDsID == 0)
            {
                DatabaseTemporal db = new DatabaseTemporal();
                nDsID = db.GetDatasetID(strDs);
                if (nDsID == 0)
                    return false;
            }

            return InitializeWithDsId1(s, nDsID, strEvtCancel);
        }

        /// <summary>
        /// Load the dataset specified by the dataset ID.
        /// </summary>
        /// <param name="nDsId">Specifies the datset ID of the dataset to load.</param>
        /// <param name="strEvtCancel">Optionally, specifies the name of the global cancel event.</param>
        /// <returns>When loaded, true is returned, otherwise false.</returns>
        public bool LoadDatasetByID1(int nDsId, string strEvtCancel = null)
        {
            SettingsCaffe s = new SettingsCaffe();
            s.DbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;
            s.DbLoadLimit = 0;
            return InitializeWithDsId1(s, nDsId, strEvtCancel);
        }

        /// <summary>
        /// Load the dataset specified by the dataset name.
        /// </summary>
        /// <param name="strDs">Specifies the name of the dataset to load.</param>
        /// <param name="strEvtCancel">Optionally, specifies the name of the global cancel event.</param>
        /// <returns>When loaded, true is returned, otherwise false.</returns>
        public bool LoadDatasetByName1(string strDs, string strEvtCancel = null)
        {
            SettingsCaffe s = new SettingsCaffe();
            s.DbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;
            s.DbLoadLimit = 0;
            return InitializeWithDsName1(s, strDs, strEvtCancel);
        }

        private TemporalSet getTemporalSet(int nSrcId)
        {
            TemporalSet ts = null;

            if (m_rgTemporalSets.ContainsKey(nSrcId))
            {
                ts = m_rgTemporalSets[nSrcId];
            }
            else
            {
                ts = m_rgDataSets.FindTemporalSetBySourceID(nSrcId);
                m_rgTemporalSets.Add(nSrcId, ts);
            }

            return ts;
        }

        /// <summary>
        /// Returns a block of static, observed and known data from the database where one block is a set of (historical and future steps).
        /// </summary>
        /// <param name="nQueryIdx">Specifies the index of the query within a batch.</param>
        /// <param name="nSrcId">Specifies the source ID of the data source.</param>
        /// <param name="nItemIdx">Returns the item index used.</param>
        /// <param name="nIdx">Returns the index used with in the item.</param>
        /// <param name="itemSelectionOverride">Optionally, specifies the item selection method used to select the item (e.g., customer, station, stock symbol)</param>
        /// <param name="valueSelectionOverride">Optionally, specifies the value selection method used to select the index within the temporal data of the selected item.</param>
        /// <param name="bEnableDebug">Optionally, specifies to enable debug output (default = false).</param>
        /// <param name="strDebugPath">Optionally, specifies the debug path where debug images are placed when 'EnableDebug' = true.</param>
        /// <returns>A tuple containing the static, observed and known data is returned.</returns>
        public SimpleDatum[] QueryTemporalItem(int nQueryIdx, int nSrcId, out int nItemIdx, out int nIdx, DB_LABEL_SELECTION_METHOD? itemSelectionOverride = null, DB_ITEM_SELECTION_METHOD? valueSelectionOverride = null, bool bEnableDebug = false, string strDebugPath = null)
        {
            TemporalSet ts = getTemporalSet(nSrcId);
            DB_LABEL_SELECTION_METHOD itemSelection = (itemSelectionOverride.HasValue) ? itemSelectionOverride.Value : m_itemSelectionMethod;
            DB_ITEM_SELECTION_METHOD valueSelection = (valueSelectionOverride.HasValue) ? valueSelectionOverride.Value : m_valueSelectionMethod;

            return ts.GetData(nQueryIdx, out nItemIdx, out nIdx, itemSelection, valueSelection, 1, bEnableDebug, strDebugPath);
        }


        /// <summary>
        /// Set the database connection to use.
        /// </summary>
        /// <param name="ci">Specifies the dataase connection information to use.</param>
        public void SetConnection(ConnectInfo ci)
        {
            MyCaffe.db.image.EntitiesConnection.GlobalDatabaseConnectInfo = ci;
        }

        /// <summary>
        /// Sets the label and image selection methods.
        /// </summary>
        /// <param name="item">Specifies the item selection method or <i>null</i> to ignore.</param>
        /// <param name="value">Specifies the value selection method or <i>null</i> to ignore.</param>
        public void SetSelectionMethod(DB_LABEL_SELECTION_METHOD? item, DB_ITEM_SELECTION_METHOD? value)
        {
            if (item.HasValue)
                m_itemSelectionMethod = item.Value;
            if (value.HasValue)
                m_valueSelectionMethod = value.Value;
        }

        //---------------------------------------------------------------------
        //  Not Implemented
        //---------------------------------------------------------------------
        #region Not Implemented
        public int FindItemIndex(int nSrcId, DateTime dt, string strDescription) /**@private */
        {
            throw new NotImplementedException();
        }

        public SimpleDatum GetItem(int nItemID, params int[] rgSrcId) /**@private */
        {
            throw new NotImplementedException();
        }

        public int GetItemCount(int nSrcId, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false) /**@private */
        {
            throw new NotImplementedException();
        }

        public SimpleDatum GetItemMean(int nSrcId) /**@private */
        {
            throw new NotImplementedException();
        }

        public List<SimpleDatum> GetItems(int nSrcId, int[] rgIdx, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false) /**@private */
        {
            throw new NotImplementedException();
        }

        public List<SimpleDatum> GetItemsFromIndex(int nSrcId, int nStartIdx, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false, bool bAttemptDirectLoad = false) /**@private */
        {
            throw new NotImplementedException();
        }

        public List<SimpleDatum> GetItemsFromTime(int nSrcId, DateTime dtStart, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false) /**@private */
        {
            throw new NotImplementedException();
        }

        public bool GetLoadItemDataCriteria() /**@private */
        {
            throw new NotImplementedException();
        }

        public bool GetLoadItemDebugData() /**@private */
        {
            throw new NotImplementedException();
        }

        public SimpleDatum QueryItem(int nSrcId, int nIdx, DB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, DB_ITEM_SELECTION_METHOD? imageSelectionOverride = null, int? nLabel = null, bool bLoadDataCriteria = false, bool bLoadDebugData = false) /**@private */
        {
            throw new NotImplementedException();
        }

        public SimpleDatum QueryItemMean(int nSrcId) /**@private */
        {
            throw new NotImplementedException();
        }

        public SimpleDatum QueryItemMeanFromDataset(int nDatasetId) /**@private */
        {
            throw new NotImplementedException();
        }

        public SimpleDatum QueryItemMeanFromDb(int nSrcId) /**@private */
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
