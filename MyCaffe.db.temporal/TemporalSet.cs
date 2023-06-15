using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.db.image;
using SimpleGraphing;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.db.temporal
{
    /// <summary>
    /// The TemporalSet manages a set of temporal data for a given data source.
    /// </summary>
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
        Dictionary<ValueItem, List<ValueStream>> m_rgSchema = new Dictionary<ValueItem, List<ValueStream>>();
        List<ValueItem> m_rgKeys = new List<ValueItem>();
        int m_nLoadLimit = 0;
        DB_LOAD_METHOD m_loadMethod;
        DatabaseTemporal m_db;
        CryptoRandom m_random = null;
        SourceDescriptor m_src;
        PlotCollectionSet m_itemSet;
        Log m_log;
        Thread m_loadThread = null;
        int m_nChunks = 1024;
        int m_nItemIdx = 0;
        int m_nValueIdx = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="log">Specifies the log for status output.</param>
        /// <param name="db">Specifies the database connection.</param>
        /// <param name="src">Specifies the data source.</param>
        /// <param name="loadMethod">Specifies the data load method.</param>
        /// <param name="nLoadLimit">Specifies the data load limit.</param>
        /// <param name="random">Specifies the random number object.</param>
        /// <param name="dtStart">Specifies the start date.</param>
        /// <param name="dtEnd">Specifies the end date.</param>
        /// <param name="nHistoricSteps">Specifies the historical steps in a step block.</param>
        /// <param name="nFutureSteps">Specifies the future steps in a step block.</param>
        /// <param name="nChunks">Specifies the number of step items to load on each cycle.</param>
        public TemporalSet(Log log, DatabaseTemporal db, SourceDescriptor src, DB_LOAD_METHOD loadMethod, int nLoadLimit, CryptoRandom random, DateTime dtStart, DateTime dtEnd, int nHistoricSteps, int nFutureSteps, int nChunks)
        {
            m_log = log;
            m_random = random;
            m_db = db;
            m_src = src;
            m_loadMethod = loadMethod;
            m_nLoadLimit = nLoadLimit;
            m_dtStart = dtStart;
            m_dtEnd = dtEnd;
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
        }

        /// <summary>
        /// Load the data based on the data load method.
        /// </summary>
        /// <returns>True is returned after starting the load thread, otherwise false is returned if it is already started.</returns>
        public bool Initialize()
        {
            if (m_loadThread == null)
            {
                m_loadThread = new Thread(new ThreadStart(loadThread));
                m_loadThread.Start();
            }

            if (m_loadMethod == DB_LOAD_METHOD.LOAD_ALL)
                return WaitForLoadingToComplete();

            return true;
        }

        private void loadThread()
        {
            List<ValueItem> rgItems = m_db.GetAllValueItems(m_src.ID);
            DateTime dtMinStart = DateTime.MaxValue;
            DateTime dtMaxEnd = DateTime.MinValue;
            int nSecondsPerStep = 0;
            int nTotalChunks = 1;
            int nItemCount = 0;

            try
            {
                m_itemSet = new PlotCollectionSet();

                // Load the data schema.
                foreach (ValueItem item in rgItems)
                {
                    List<ValueStream> rgStreams = m_db.GetAllValueStreams(item.ID);

                    DateTime dtStart = rgStreams.Min(p => p.StartTime.Value);
                    DateTime dtEnd = rgStreams.Max(p => p.EndTime.Value);

                    dtMinStart = (dtStart < dtMinStart) ? dtStart : dtMinStart;
                    dtMaxEnd = (dtEnd > dtMaxEnd) ? dtEnd : dtMaxEnd;

                    int nMinSecPerStep = rgStreams.Min(p => p.SecondsPerStep.Value);
                    int nMaxSecPerStep = rgStreams.Max(p => p.SecondsPerStep.Value);
                    int nMaxItems = rgStreams.Max(p => p.ItemCount.Value);

                    if (nMinSecPerStep != nMaxSecPerStep)
                        throw new Exception("All streams must have the same number of seconds per step.");

                    if (nSecondsPerStep == 0)
                        nSecondsPerStep = nMinSecPerStep;
                    else if (nSecondsPerStep != nMinSecPerStep)
                        throw new Exception("All streams must have the same number of seconds per step.");

                    if (nItemCount == 0)
                        nItemCount = nMaxItems;
                    else if (nItemCount != nMaxItems)
                        throw new Exception("All streams must have the same number of items.");

                    m_rgSchema.Add(item, rgStreams);
                }

                m_rgKeys = m_rgSchema.Keys.ToList();

                if (nItemCount < m_nTotalSteps)
                    throw new Exception("The number of items in the data source is less than the number of steps requested.");

                if (m_dtEnd <= m_dtStart)
                    throw new Exception("The end date must be greater than the start date.");

                if (m_dtEnd < dtMinStart)
                    throw new Exception("The end date must be greater than the minimum start date of the data source.");

                if (m_dtStart > dtMaxEnd)
                    throw new Exception("The start date must be less than the maximum end date of the data source.");

                nTotalChunks = (int)Math.Floor((double)nItemCount/(double)m_nTotalSteps);
                if (m_nLoadLimit > 0 && m_nLoadLimit < nTotalChunks)
                    nTotalChunks = m_nLoadLimit;

                int nChunks = Math.Min(nTotalChunks, m_nChunks);

                // Load the data.
                DateTime dtStart1 = (dtMinStart > m_dtStart) ? dtMinStart : m_dtStart;
                DateTime dtEnd1 = dtStart1;
                DateTime dt = dtStart1;
                bool bEOD = false;
                int nStepsToLoad = m_nTotalSteps * nChunks;
                int nLoadedChunks = 0;

                while (!m_evtCancel.WaitOne(0))
                {
                    PlotCollectionSet set = new PlotCollectionSet();

                    // Load one chunk for each item.
                    foreach (KeyValuePair<ValueItem, List<ValueStream>> kv in m_rgSchema)
                    {
                        ValueItem item = kv.Key;

                        bool bEOD1 = false;
                        PlotCollection plots = m_db.GetRawValues(m_src.ID, item.ID, dt, nStepsToLoad, false, out dtEnd1, out bEOD1);
                        set.Add(plots);

                        if (bEOD1)
                            bEOD = true;
                    }

                    m_itemSet.Add(set);

                    if (bEOD)
                        dt = dtStart1;
                    else
                        dt = dtEnd1;

                    nLoadedChunks += nChunks;

                    if (m_nLoadLimit > 0)
                    {
                        while (m_itemSet[0].Count > m_nLoadLimit)
                        {
                            m_itemSet.RemoveAt(0);
                        }
                    }
                    else
                    {
                        if (nLoadedChunks >= nTotalChunks)
                            break;
                    }
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
        /// <param name="nWaitMs">Specifies the maximum number of ms to wait (default = int.MaxValue).</param>
        /// <returns>If the load has completed <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool WaitForLoadingToComplete(int nWaitMs = int.MaxValue)
        {
            WaitHandle[] rgWait = new WaitHandle[] { m_evtCancel, m_evtDone };
            int nWait = WaitHandle.WaitAny(rgWait, nWaitMs);
            m_loadThread = null;

            if (nWait == 0)
                return false;

            return true;
        }

        /// <summary>
        /// Get the image load method used on initialization.
        /// </summary>
        public DB_LOAD_METHOD LoadMethod
        {
            get { return m_loadMethod; }
        }

        /// <summary>
        /// Get a single temporal data set for a selected item where the temporal data set contains (nHistSteps + nFutSteps) * nStreamCount items.
        /// </summary>
        /// <param name="itemSelectionMethod">Specifies the item index selection method.</param>
        /// <param name="valueSelectionMethod">Specifies the value starting point selection method.</param>
        /// <param name="nValueStepOffset">Optionally, specifies the value step offset from the previous query (this parameter only applies when using non random selection).</param>
        /// <returns>A SimpleDatum is returned containing (nHistSteps + nFutSteps) for the all streams for a given item.</returns>
        public SimpleDatum GetTemporalData(DB_LABEL_SELECTION_METHOD itemSelectionMethod, DB_ITEM_SELECTION_METHOD valueSelectionMethod, int nValueStepOffset)
        {
            if (itemSelectionMethod == DB_LABEL_SELECTION_METHOD.RANDOM)
                m_nItemIdx = m_random.Next(m_rgSchema.Count);
            else
            {
                if (m_nItemIdx >= m_rgSchema.Count)
                    m_nItemIdx = 0;
            }

            ValueItem item = m_rgKeys[m_nItemIdx];
            int nValueCount = m_rgSchema[item][0].ItemCount.GetValueOrDefault(0);
            int nStreamCount = m_rgSchema[item].Count;

            if (valueSelectionMethod == DB_ITEM_SELECTION_METHOD.RANDOM)
            {
                m_nValueIdx = m_random.Next(nValueCount - m_nTotalSteps);
            }
            else if (m_nValueIdx + m_nTotalSteps >= nValueCount)
            {
                m_nValueIdx = 0;
                m_nItemIdx++;
            }

            int nC = 1;
            int nW = nStreamCount; // streams
            int nH = m_nTotalSteps;   // value streams

            float[] rgf = new float[nC * nH * nW];

            for (int i = 0; i < nH; i++)
            {
                for (int j = 0; j < nW; j++)
                {
                    int nSrcIdx = m_nValueIdx + i;
                    int nDstIdx = (i * nW) + j;
                    rgf[nDstIdx] = m_itemSet[m_nItemIdx][nSrcIdx].Y_values[j];
                }
            }

            SimpleDatum sd = new SimpleDatum(nC, nW, nH, rgf, 0, rgf.Length);
            sd.Index = m_nHistoricSteps;
            sd.TimeStamp = (DateTime)m_itemSet[m_nItemIdx][m_nValueIdx + m_nHistoricSteps].Tag;

            m_nValueIdx += nValueStepOffset;

            return sd;
        }
    }
}
