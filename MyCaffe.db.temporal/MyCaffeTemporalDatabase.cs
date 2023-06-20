using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
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
        CryptoRandom m_random = null;
        EventWaitHandle m_evtInitializing = null;
        EventWaitHandle m_evtInitialized = null;
        EventWaitHandle m_evtAbortInitialization = null;
        bool m_bEnabled = false;
        static object m_syncObject = new object();
        DB_ITEM_SELECTION_METHOD m_itemSelectionMethod = DB_ITEM_SELECTION_METHOD.RANDOM;
        DB_LABEL_SELECTION_METHOD m_labelSelectionMethod = DB_LABEL_SELECTION_METHOD.RANDOM;
        Log m_log;
        DB_LOAD_METHOD m_loadMethod = DB_LOAD_METHOD.LOAD_ON_DEMAND;
        int m_nLoadLimit = 0;
        Guid m_userGuid;
        DatasetCollection m_rgDataSets = new DatasetCollection();

        /// <summary>
        /// The constructor.
        /// </summary>
        public MyCaffeTemporalDatabase()
        {
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
            m_rgDataSets.CleanUp(nDsId);
        }

        public int FindItemIndex(int nSrcId, DateTime dt, string strDescription)
        {
            throw new NotImplementedException();
        }

        public DatasetDescriptor GetDatasetById(int nDsId)
        {
            throw new NotImplementedException();
        }

        public DatasetDescriptor GetDatasetByName(string strDs)
        {
            throw new NotImplementedException();
        }

        public int GetDatasetID(string strDs)
        {
            throw new NotImplementedException();
        }

        public double GetDatasetLoadedPercentById(int nDatasetID, out double dfTraining, out double dfTesting)
        {
            throw new NotImplementedException();
        }

        public double GetDatasetLoadedPercentByName(string strDataset, out double dfTraining, out double dfTesting)
        {
            throw new NotImplementedException();
        }

        public string GetDatasetName(int nDsId)
        {
            throw new NotImplementedException();
        }

        public SimpleDatum GetItem(int nItemID, params int[] rgSrcId)
        {
            throw new NotImplementedException();
        }

        public int GetItemCount(int nSrcId, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            throw new NotImplementedException();
        }

        public SimpleDatum GetItemMean(int nSrcId)
        {
            throw new NotImplementedException();
        }

        public List<SimpleDatum> GetItems(int nSrcId, int[] rgIdx, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            throw new NotImplementedException();
        }

        public List<SimpleDatum> GetItemsFromIndex(int nSrcId, int nStartIdx, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false, bool bAttemptDirectLoad = false)
        {
            throw new NotImplementedException();
        }

        public List<SimpleDatum> GetItemsFromTime(int nSrcId, DateTime dtStart, int nQueryCount = int.MaxValue, string strFilterVal = null, int? nBoostVal = null, bool bBoostValIsExact = false)
        {
            throw new NotImplementedException();
        }

        public bool GetLoadItemDataCriteria()
        {
            throw new NotImplementedException();
        }

        public bool GetLoadItemDebugData()
        {
            throw new NotImplementedException();
        }

        public Tuple<DB_LABEL_SELECTION_METHOD, DB_ITEM_SELECTION_METHOD> GetSelectionMethod()
        {
            throw new NotImplementedException();
        }

        public SourceDescriptor GetSourceById(int nSrcId)
        {
            throw new NotImplementedException();
        }

        public SourceDescriptor GetSourceByName(string strSrc)
        {
            throw new NotImplementedException();
        }

        public int GetSourceID(string strSrc)
        {
            throw new NotImplementedException();
        }

        public string GetSourceName(int nSrcId)
        {
            throw new NotImplementedException();
        }

        public DB_VERSION GetVersion()
        {
            throw new NotImplementedException();
        }

        public bool InitializeWithDs1(SettingsCaffe s, DatasetDescriptor ds, string strEvtCancel = null)
        {
            throw new NotImplementedException();
        }

        public bool InitializeWithDsId1(SettingsCaffe s, int nDataSetID, string strEvtCancel = null, int nPadW = 0, int nPadH = 0)
        {
            throw new NotImplementedException();
        }

        public bool InitializeWithDsName1(SettingsCaffe s, string strDs, string strEvtCancel = null)
        {
            throw new NotImplementedException();
        }

        public bool LoadDatasetByID1(int nDsId, string strEvtCancel = null)
        {
            throw new NotImplementedException();
        }

        public bool LoadDatasetByName1(string strDs, string strEvtCancel = null)
        {
            throw new NotImplementedException();
        }

        public SimpleDatum QueryItem(int nSrcId, int nIdx, DB_LABEL_SELECTION_METHOD? labelSelectionOverride = null, DB_ITEM_SELECTION_METHOD? imageSelectionOverride = null, int? nLabel = null, bool bLoadDataCriteria = false, bool bLoadDebugData = false)
        {
            throw new NotImplementedException();
        }

        public SimpleDatum QueryItemMean(int nSrcId)
        {
            throw new NotImplementedException();
        }

        public SimpleDatum QueryItemMeanFromDataset(int nDatasetId)
        {
            throw new NotImplementedException();
        }

        public SimpleDatum QueryItemMeanFromDb(int nSrcId)
        {
            throw new NotImplementedException();
        }

        public bool ReloadDataset(int nDsId)
        {
            throw new NotImplementedException();
        }

        public void SetConnection(ConnectInfo ci)
        {
            throw new NotImplementedException();
        }

        public void SetSelectionMethod(DB_LABEL_SELECTION_METHOD? lbl, DB_ITEM_SELECTION_METHOD? img)
        {
            throw new NotImplementedException();
        }

        public bool UnloadDatasetById(int nDatasetID)
        {
            throw new NotImplementedException();
        }

        public bool UnloadDatasetByName(string strDataset)
        {
            throw new NotImplementedException();
        }
    }
}
