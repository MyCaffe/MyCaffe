using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.db.image;
using SimpleGraphing;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.db.temporal
{
    /// <summary>
    /// The TemporalSet manages a set of temporal data for a given data source.
    /// </summary>
    [Serializable]
    public class TemporalSet : IDisposable
    {
        Exception m_loadException = null;
        int m_nHistoricSteps;
        int m_nFutureSteps;
        int m_nTotalSteps;
        DateTime m_dtStart;
        DateTime m_dtEnd;
        ManualResetEvent m_evtCancel = new ManualResetEvent(false);
        AutoResetEvent m_evtDone = new AutoResetEvent(false);
        List<ItemSet> m_rgItems = new List<ItemSet>();
        int m_nLoadLimit = 0;
        DB_LOAD_METHOD m_loadMethod;
        DatabaseTemporal m_db;
        CryptoRandom m_random = null;
        SourceDescriptor m_src;
        Log m_log;
        Thread m_loadThread = null;
        int m_nChunks = 1024;
        int m_nItemIdx = 0;
        bool m_bNormalizeData = false;
        double m_dfLoadPct = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="log">Specifies the log for status output.</param>
        /// <param name="db">Specifies the database connection.</param>
        /// <param name="src">Specifies the data source.</param>
        /// <param name="loadMethod">Specifies the data load method.</param>
        /// <param name="nLoadLimit">Specifies the data load limit.</param>
        /// <param name="random">Specifies the random number object.</param>
        /// <param name="nHistoricSteps">Specifies the historical steps in a step block.</param>
        /// <param name="nFutureSteps">Specifies the future steps in a step block.</param>
        /// <param name="nChunks">Specifies the number of step items to load on each cycle.</param>
        public TemporalSet(Log log, DatabaseTemporal db, SourceDescriptor src, DB_LOAD_METHOD loadMethod, int nLoadLimit, CryptoRandom random, int nHistoricSteps, int nFutureSteps, int nChunks)
        {
            m_log = log;
            m_random = random;
            m_db = db;
            m_src = src;
            m_loadMethod = loadMethod;
            m_nLoadLimit = nLoadLimit;
            m_nHistoricSteps = nHistoricSteps;
            m_nFutureSteps = nFutureSteps;
            m_nTotalSteps = m_nHistoricSteps + m_nFutureSteps;
            m_nChunks = nChunks;
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            m_evtCancel.Set();
        }

        /// <summary>
        /// Release all resources used by the temporal set and shut down all internal threads.
        /// </summary>
        public void CleanUp()
        {
            if (m_loadThread != null)
            {
                m_evtCancel.Set();
                m_loadThread.Join();
                m_loadThread = null;
            }

            foreach (ItemSet item in m_rgItems)
            {
                item.CleanUp();
            }

            m_rgItems.Clear();
        }

        /// <summary>
        /// Reset all indexes to their starting locations.
        /// </summary>
        public void Reset()
        {
            m_nItemIdx = 0;

            foreach (ItemSet item in m_rgItems)
            {
                item.Reset();
            }
        }

        /// <summary>
        /// Load the data based on the data load method.
        /// </summary>
        /// <param name="bNormalizedData">Specifies to load the normalized data.</param>
        /// <param name="evtCancel">Specifies the auto reset event used to cancel waiting for the data to load.</param>
        /// <returns>True is returned after starting the load thread, otherwise false is returned if it is already started.</returns>
        public bool Initialize(bool bNormalizedData, EventWaitHandle evtCancel)
        {
            m_bNormalizeData = bNormalizedData;

            if (m_loadThread == null)
            {
                m_loadThread = new Thread(new ThreadStart(loadThread));
                m_loadThread.Start();
            }

            if (m_loadMethod == DB_LOAD_METHOD.LOAD_ALL)
                return WaitForLoadingToComplete(evtCancel);

            return true;
        }

        private void loadThread()
        {
            DatabaseLoader loader = new DatabaseLoader();
            List<ValueItem> rgItems = m_db.GetAllValueItems(m_src.ID);

            try
            {
                m_dfLoadPct = 0;

                TemporalDescriptor td = loader.LoadTemporalFromDb(m_src.ID);
                OrderedValueStreamDescriptorSet rgStrm = td.OrderedValueStreamDescriptors;
                
                m_dtStart = td.StartDate;
                m_dtEnd = td.EndDate;

                if (m_dtEnd <= m_dtStart)
                    throw new Exception("The end date must be greater than the start date.");

                // Load the data.
                DateTime dt = m_dtStart;
                bool bEOD = false;

                Stopwatch sw = new Stopwatch();
                sw.Start();
                m_dfLoadPct = 0;

                while (!m_evtCancel.WaitOne(0))
                {
                    bool bEOD1;

                    // Load one chunk for each item.
                    for (int i=0; i<rgItems.Count; i++)
                    {
                        ItemSet item = new ItemSet(m_random, m_db, rgItems[i], rgStrm);
                        DateTime dtEnd = item.Load(dt, out bEOD1);
                        m_rgItems.Add(item);

                        m_dfLoadPct = (double)i / rgItems.Count;                       
                        if (bEOD1)
                            bEOD = true;

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            sw.Restart();
                            m_log.Progress = m_dfLoadPct;
                            m_log.WriteLine("Loading '" + m_src.Name + "' data for item " + item.Item.Name + " (" + m_dfLoadPct.ToString("P") + ")...", true);
                        }
                    }

                    if (bEOD)
                        break;
                }
            }
            catch (Exception excpt)
            {
                m_loadException = excpt;
                return;
            }
            finally
            {
                m_evtDone.Set();
            }
        }

        /// <summary>
        /// Wait for the image set to complete loading.
        /// </summary>
        /// <param name="evtCancel">Specifies the cancel event to abort waiting.</param>
        /// <param name="nWaitMs">Specifies the maximum number of ms to wait (default = int.MaxValue).</param>
        /// <returns>If the load has completed <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool WaitForLoadingToComplete(EventWaitHandle evtCancel, int nWaitMs = int.MaxValue)
        {
            WaitHandle[] rgWait = new WaitHandle[] { m_evtCancel, evtCancel, m_evtDone };
            int nWait = WaitHandle.WaitAny(rgWait, nWaitMs);

            if (nWait <= 1)
                return false;

            return true;
        }

        /// <summary>
        /// Returns the percentage of the load completed.
        /// </summary>
        public double LoadPercent
        {
            get { return m_dfLoadPct; }
        }

        /// <summary>
        /// Get the image load method used on initialization.
        /// </summary>
        public DB_LOAD_METHOD LoadMethod
        {
            get { return m_loadMethod; }
        }

        /// <summary>
        /// Get a data set consisting of the static, historical, and future data for a selected item where the static data is not bound by time, 
        /// the historical data set contains nHistSteps * nStreamCount items, and the future data set contains (nHistSteps + nFutSteps) * nStreamCount items.
        /// </summary>
        /// <param name="nQueryIdx">Specifies the index location of the query within a batch.</param>
        /// <param name="itemSelectionMethod">Specifies the item index selection method.</param>
        /// <param name="valueSelectionMethod">Specifies the value starting point selection method.</param>
        /// <param name="nValueStepOffset">Optionally, specifies the value step offset from the previous query (default = 1, this parameter only applies when using non random selection).</param>
        /// <param name="bEnableDebug">Optionally, specifies to enable debug output (default = false).</param>
        /// <param name="strDebugPath">Optionally, specifies the debug path where debug images are placed when 'EnableDebug' = true.</param>
        /// <returns>An array of SimpleDatum is returned where: [0] = static num, [1] = static cat, [2] = historical num, [3] = historical cat, [4] = future num, [5] = future cat, [6] = target, and [7] = target history
        /// for a given item at the temporal selection point.</returns>
        /// <remarks>Note, the ordering for historical value streams is: observed, then known.  Future value streams only contiain known value streams.  If a dataset does not have one of the data types noted above, null
        /// is returned in the array slot (for example, if the dataset does not produce static numeric values, the array slot is set to [0] = null.</remarks>
        public SimpleDatum[] GetData(int nQueryIdx, DB_LABEL_SELECTION_METHOD itemSelectionMethod, DB_ITEM_SELECTION_METHOD valueSelectionMethod, int nValueStepOffset = 1, bool bEnableDebug = false, string strDebugPath = null)
        {
            if (itemSelectionMethod == DB_LABEL_SELECTION_METHOD.RANDOM)
            {
                m_nItemIdx = m_random.Next(m_rgItems.Count);
            }
            else
            {
                if (m_nItemIdx >= m_rgItems.Count)
                    m_nItemIdx = 0;
            }

            SimpleDatum[] data = m_rgItems[m_nItemIdx].GetData(nQueryIdx, valueSelectionMethod, m_nHistoricSteps, m_nFutureSteps, nValueStepOffset, bEnableDebug, strDebugPath);

            int nRetryCount = 0;
            while (data == null && nRetryCount < 5)
            {
                if (itemSelectionMethod == DB_LABEL_SELECTION_METHOD.RANDOM)
                {
                    m_nItemIdx = m_random.Next(m_rgItems.Count);
                }
                else
                {
                    m_nItemIdx++;

                    if (m_nItemIdx >= m_rgItems.Count)
                        m_nItemIdx = 0;
                }

                data = m_rgItems[m_nItemIdx].GetData(nQueryIdx, valueSelectionMethod, m_nHistoricSteps, m_nFutureSteps, nValueStepOffset, bEnableDebug, strDebugPath);
                nRetryCount++;
            }

            return data;
        }
    }
}
