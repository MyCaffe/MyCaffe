using Microsoft.VisualBasic.Devices;
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
using static MyCaffe.basecode.descriptors.ValueStreamDescriptor;

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
        DateTime? m_dtStart;
        DateTime? m_dtEnd;
        ManualResetEvent m_evtCancel = new ManualResetEvent(false);
        AutoResetEvent m_evtDone = new AutoResetEvent(false);
        List<ItemSet> m_rgItems = new List<ItemSet>();
        int m_nLoadLimit = 0;
        double m_dfReplacementPct;
        int m_nRefreshUpdateMs;
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
        object m_objSync = new object();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="log">Specifies the log for status output.</param>
        /// <param name="db">Specifies the database connection.</param>
        /// <param name="src">Specifies the data source.</param>
        /// <param name="loadMethod">Specifies the data load method.</param>
        /// <param name="nLoadLimit">Specifies the data load limit.</param>
        /// <param name="dfReplacementPct">Specifies the percent of replacement on a load limit event.</param>
        /// <param name="nRefreshUpdateMs">Specifies the refresh update perod in milliseconds.</param>
        /// <param name="random">Specifies the random number object.</param>
        /// <param name="nHistoricSteps">Specifies the historical steps in a step block.</param>
        /// <param name="nFutureSteps">Specifies the future steps in a step block.</param>
        /// <param name="nChunks">Specifies the number of step items to load on each cycle.</param>
        public TemporalSet(Log log, DatabaseTemporal db, SourceDescriptor src, DB_LOAD_METHOD loadMethod, int nLoadLimit, double dfReplacementPct, int nRefreshUpdateMs, CryptoRandom random, int nHistoricSteps, int nFutureSteps, int nChunks)
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
            m_dfReplacementPct = dfReplacementPct;
            m_nRefreshUpdateMs = nRefreshUpdateMs;

            if (m_nLoadLimit < 0)
                m_nLoadLimit = 0;

            if (m_nLoadLimit > 0 && m_nLoadLimit < 1000)
                m_nLoadLimit = 1000;

            if (m_dfReplacementPct > 0.9)
                m_dfReplacementPct = 0.9;

            if (m_nRefreshUpdateMs < 250)
                m_nRefreshUpdateMs = 250;
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
                m_loadThread = null;
            }

            m_rgItems.Clear();
        }

        /// <summary>
        /// Add a direct item set to the temporal set.
        /// </summary>
        /// <param name="random">Specifies the random generator.</param>
        /// <param name="item">Specifies the value item description.</param>
        /// <param name="rgStrm">Specifies the ordered stream descriptors for the item.</param>
        /// <param name="src">Optionally, specifies a source descriptor to update with temporal information.</param>
        /// <returns>The index of the item is returned.</returns>
        public int AddDirectItemSet(CryptoRandom random, ValueItem item, OrderedValueStreamDescriptorSet rgStrm, SourceDescriptor src = null)
        {
            ItemSet itemSet = new ItemSet(random, m_db, item, rgStrm);

            if (src != null)
            {
                if (src.TemporalDescriptor == null)
                    src.TemporalDescriptor = new TemporalDescriptor();

                bool bFound = false;
                foreach (ValueItemDescriptor vid in src.TemporalDescriptor.ValueItemDescriptors)
                {
                    if (vid.ID == item.ID)
                    { 
                        bFound = true; 
                        break; 
                    }
                }

                if (!bFound)
                {
                    ValueItemDescriptor vid1 = new ValueItemDescriptor(item.ID, item.Name, item.StartTime, item.EndTime, item.Steps);
                    src.TemporalDescriptor.ValueItemDescriptors.Add(vid1);
                }

                foreach (ValueStreamDescriptor vsd1 in rgStrm.Descriptors)
                {
                    bFound = false;
                    foreach (ValueStreamDescriptor vsd in src.TemporalDescriptor.ValueStreamDescriptors)
                    {
                        if (vsd.ID == vsd1.ID)
                        {
                            bFound = true;
                            break;
                        }
                    }

                    if (!bFound)
                        src.TemporalDescriptor.ValueStreamDescriptors.Add(vsd1);
                }
            }

            m_rgItems.Add(itemSet);
            return m_rgItems.Count - 1;
        }

        /// <summary>
        /// Add a set of values directly to the temporal set item values.
        /// </summary>
        /// <param name="nItemId">Specifies the ID of the item values to add data to.</param>
        /// <param name="plots">Specifies the data to add.</param>
        /// <param name="nStartIdx">Specifies the start index.</param>
        /// <param name="nEndIdx">Specifies the end index.</param>
        /// <param name="nValIdx">Specifies the value index into the Y_values to add (default = -1, to add all Y_values).</param>
        /// <returns>The new start/end date and count are returned.</returns>
        /// <exception cref="IndexOutOfRangeException">An exception is thrown if the item index is out of range.</exception>
        public Tuple<DateTime, DateTime, int> AddDirectValues(int nItemId, PlotCollection plots, int nStartIdx, int nEndIdx, int nValIdx = -1)
        {
            if (nItemId < 0 || nItemId >= m_rgItems.Count)
                throw new IndexOutOfRangeException("The item index '" + nItemId.ToString() + "' is out of range.");

            return m_rgItems[nItemId].AddDirectValues(plots, nStartIdx, nEndIdx, nValIdx);
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

        private void removeItem(List<ItemSet> rgItems)
        {
            lock (m_objSync)
            {
                if (m_rgItems == null)
                    return;

                if (m_rgItems.Count == 0)
                    return;

                ItemSet item = rgItems[0];
                rgItems.RemoveAt(0);
                item.CleanUp();
            }
        }

        private void loadThread()
        {
            DatabaseLoader loader = new DatabaseLoader();
            List<ValueItem> rgItems = m_db.GetAllValueItems(m_src.ID);
            double dfMemMin = 1000.0;

            try
            {
                m_dfLoadPct = 0;

                TemporalDescriptor td = loader.LoadTemporalFromDb(m_src.ID);
                OrderedValueStreamDescriptorSet rgStrm = td.OrderedValueStreamDescriptors;
                
                m_dtStart = td.StartDate;
                m_dtEnd = td.EndDate;

                if (m_dtEnd.HasValue && m_dtStart.HasValue && m_dtEnd <= m_dtStart)
                    throw new Exception("The end date must be greater than the start date.");

                // Load the data.
                DateTime? dt = m_dtStart;
                bool bEOD = false;

                Stopwatch sw = new Stopwatch();
                sw.Start();
                m_dfLoadPct = 0;
                int nIdx = 0;

                ComputerInfo info = new ComputerInfo();
                Stopwatch swReplacement = new Stopwatch();

                swReplacement.Start();

                while (!m_evtCancel.WaitOne(0))
                {
                    bool bEOD1;

                    // Load one chunk for each item.
                    while (m_rgItems.Count < rgItems.Count && (m_nLoadLimit <= 0 || m_rgItems.Count < m_nLoadLimit))
                    {
                        ItemSet item = new ItemSet(m_random, m_db, rgItems[nIdx], rgStrm);
                        item.Load(out bEOD1);

                        lock (m_objSync)
                        {
                            m_rgItems.Add(item);
                        }

                        m_dfLoadPct = (double)m_rgItems.Count / rgItems.Count;                       
                        if (bEOD1)
                            bEOD = true;

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            sw.Restart();
                            m_log.Progress = m_dfLoadPct;
                            m_log.WriteLine("Loading '" + m_src.Name + "' data for item " + item.Item.Name + " (" + m_dfLoadPct.ToString("P") + ")...", true);
                        }

                        if (m_evtCancel.WaitOne(0))
                            break;

                        nIdx++;

                        if (nIdx >= rgItems.Count)
                            nIdx = 0;

                        if (nIdx % m_nChunks == 0)
                            break;
                    }

                    if (m_nLoadLimit > 0)
                    {
                        int nLoadLimit = m_nLoadLimit;
                        
                        if ((int)swReplacement.Elapsed.TotalMilliseconds > m_nRefreshUpdateMs)
                            nLoadLimit = (int)(m_nLoadLimit * (1 - m_dfReplacementPct));

                        if (nLoadLimit > 1000)
                        {
                            while (m_rgItems.Count > nLoadLimit)
                            {
                                removeItem(m_rgItems);
                            }

                            GC.Collect(2, GCCollectionMode.Forced);
                        }
                    }

                    ulong nMem = info.AvailablePhysicalMemory;
                    double dfMb = (double)nMem / (1024.0 * 1024.0);
                    int nRetryCount = 0;

                    while (dfMb < dfMemMin && nRetryCount < 10 && m_rgItems.Count > 1000)
                    {
                        m_log.WriteLine("Waiting for memory to free up...", true);

                        for (int i=0; i<1000 && m_rgItems.Count > 1000; i++)
                        {
                            removeItem(m_rgItems);
                        }

                        GC.Collect(2, GCCollectionMode.Forced);
                        nMem = info.AvailablePhysicalMemory;
                        dfMb = (double)nMem / (1024.0 * 1024.0);

                        nRetryCount++;
                        Thread.Sleep(250);
                    }

                    if (dfMb < dfMemMin)
                        return;

                    if (m_rgItems.Count == rgItems.Count && m_nLoadLimit <= 0)
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
        /// <param name="nItemIdx">Specifies the item index override when not null, returns the item index used.</param>
        /// <param name="nValueIdx">Specifies the value index override when not null, returns the index used with in the item.</param>
        /// <param name="itemSelectionMethod">Specifies the item index selection method.</param>
        /// <param name="valueSelectionMethod">Specifies the value starting point selection method.</param>
        /// <param name="nValueStepOffset">Optionally, specifies the value step offset from the previous query (default = 1, this parameter only applies when using non random selection).</param>
        /// <param name="bOutputTime">Optionally, output the time data.</param>
        /// <param name="bOutputMask">Optionally, output the mask data.</param>
        /// <param name="bEnableDebug">Optionally, specifies to enable debug output (default = false).</param>
        /// <param name="strDebugPath">Optionally, specifies the debug path where debug images are placed when 'EnableDebug' = true.</param>
        /// <returns>An collection of SimpleTemporalDatums is returned where: [0] = static num, [1] = static cat, [2] = historical num, [3] = historical cat, [4] = future num, [5] = future cat, [6] = target, and [7] = target history
        /// for a given item at the temporal selection point.</returns>
        /// <remarks>Note, the ordering for historical value streams is: observed, then known.  Future value streams only contiain known value streams.  If a dataset does not have one of the data types noted above, null
        /// is returned in the array slot (for example, if the dataset does not produce static numeric values, the array slot is set to [0] = null.</remarks>
        public SimpleTemporalDatumCollection GetData(int nQueryIdx, ref int? nItemIdx, ref int? nValueIdx, DB_LABEL_SELECTION_METHOD itemSelectionMethod, DB_ITEM_SELECTION_METHOD valueSelectionMethod, int nValueStepOffset = 1, bool bOutputTime = false, bool bOutputMask = false, bool bEnableDebug = false, string strDebugPath = null)
        {
            SimpleTemporalDatumCollection data = null;

            lock (m_objSync)
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

                if (nItemIdx.HasValue)
                    m_nItemIdx = nItemIdx.Value;
                nItemIdx = m_nItemIdx;

                if (m_nItemIdx >= m_rgItems.Count)
                    return null;

                data = m_rgItems[m_nItemIdx].GetData(nQueryIdx, ref nValueIdx, valueSelectionMethod, m_nHistoricSteps, m_nFutureSteps, nValueStepOffset, bOutputTime, bOutputMask, bEnableDebug, strDebugPath);

                int nRetryCount = 0;
                while (data == null && nRetryCount < 40)
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

                    nItemIdx = m_nItemIdx;
                    data = m_rgItems[m_nItemIdx].GetData(nQueryIdx, ref nValueIdx, valueSelectionMethod, m_nHistoricSteps, m_nFutureSteps, nValueStepOffset, bOutputTime, bOutputMask, bEnableDebug, strDebugPath);
                    nRetryCount++;
                }
            }

            return data;
        }

        /// <summary>
        /// Return the total number of queries available in the temporal set.
        /// </summary>
        /// <returns>The number of queries available is returned.</returns>
        public int GetCount()
        {
            lock (m_objSync)
            {
                return m_rgItems.Sum(p => p.GetCount(m_nHistoricSteps + m_nFutureSteps));
            }
        }

        /// <summary>
        /// Return the total number of queries available in the temporal set.
        /// </summary>
        public int Count
        {
            get 
            {
                lock (m_objSync)
                {
                    return m_rgItems.Count;
                }
            }
        }
    }
}
