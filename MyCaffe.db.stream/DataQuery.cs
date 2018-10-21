using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    /// <summary>
    /// The DataQuery manages a custom query interface and queues data from the custom query via an internal query thread.
    /// </summary>
    public class DataQuery : IDisposable
    {
        CancelEvent m_evtCancel = new CancelEvent();
        ManualResetEvent m_evtQueryEnabled = new ManualResetEvent(false);
        ManualResetEvent m_evtPaused = new ManualResetEvent(false);
        bool m_bQueryEnd = false;
        object m_objSync = new object();
        Queue<double[]> m_rgDataQueue = new Queue<double[]>();
        Task m_queryTask;
        IXCustomQuery m_iquery;
        DateTime m_dtStart;
        DateTime m_dt;
        TimeSpan m_tsInc;
        int m_nMaxCount = 0;
        int m_nSegmentSize = 1;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="iquery">Specifies the custom query managed.</param>
        /// <param name="dtStart">Specifies the start date for queries.</param>
        /// <param name="tsInc">Specifies the time increment between data items within a query.</param>
        /// <param name="nSegmentSize">Specifies the number of items to collect on each query.</param>
        /// <param name="nMaxCount">Specifies the maximum number of items to store in memory.</param>
        public DataQuery(IXCustomQuery iquery, DateTime dtStart, TimeSpan tsInc, int nSegmentSize, int nMaxCount)
        {
            m_nSegmentSize = nSegmentSize;
            m_nMaxCount = nMaxCount;
            m_dtStart = dtStart;
            m_dt = dtStart;
            m_tsInc = tsInc;
            m_iquery = iquery;
            m_queryTask = Task.Factory.StartNew(new Action(queryThread));
        }

        /// <summary>
        /// Release all resources used and shutdown.
        /// </summary>
        public void Dispose()
        {
            Shutdown();
        }

        /// <summary>
        /// Stop the internal query thread.
        /// </summary>
        public void Shutdown()
        {
            m_evtCancel.Set();
        }

        /// <summary>
        /// Returns the number of fields (including the sync field) that this query manages.
        /// </summary>
        public int FieldCount
        {
            get { return m_iquery.FieldCount; }
        }

        /// <summary>
        /// Enable/disable the internal query thread.
        /// </summary>
        public bool EnableQueueThread
        {
            get
            {
                if (m_evtQueryEnabled.WaitOne(0))
                    return true;
                else
                    return false;
            }
            set
            {
                if (value)
                    m_evtQueryEnabled.Set();
                else
                    m_evtQueryEnabled.Reset();
            }
        }

        /// <summary>
        /// Returns the number of items in the data queue.
        /// </summary>
        public int Count
        {
            get { return m_rgDataQueue.Count; }
        }

        /// <summary>
        /// Returns <i>true</i> when data is ready, <i>false</i> otherwise.
        /// </summary>
        /// <param name="nCount">Specifies the number of items in the data queue required to consider the data 'ready'.</param>
        /// <returns></returns>
        public bool DataReady(int nCount)
        {
            if (m_rgDataQueue.Count < nCount)
                return false;

            return true;
        }

        /// <summary>
        /// Returns <i>true</i> when there is no more data to query.
        /// </summary>
        /// <returns>Returns <i>true</i> when there is no more data to query.</returns>
        public bool DataDone()
        {
            return m_bQueryEnd;
        }

        /// <summary>
        /// Returns data at an index within the queue without removing it, or <i>null</i> if no data exists at the index.
        /// </summary>
        /// <param name="nIdx">Specifies the index to check.</param>
        /// <returns>The data at the index is returned, or <i>null</i> if not data exists at that index.</returns>
        public double[] PeekDataAt(int nIdx)
        {
            if (nIdx >= m_rgDataQueue.Count)
                return null;

            return m_rgDataQueue.ElementAt(nIdx);
        }

        /// <summary>
        /// Returns data at an index and field within the queue without removing it.
        /// </summary>
        /// <param name="nIdx">Specifies the index to check.</param>
        /// <param name="nFieldIdx">Specifies the field to check.</param>
        /// <returns>The data at the index and field is returned.</returns>
        public double PeekDataAt(int nIdx, int nFieldIdx)
        {
            double[] rg = m_rgDataQueue.ElementAt(nIdx);
            return rg[nFieldIdx];
        }

        /// <summary>
        /// Returns the next data and removes it from the queue.
        /// </summary>
        /// <returns>The next data is returned.  When no data exists, <i>null</i> is returned.</returns>
        public double[] GetNextData()
        {
            lock (m_objSync)
            {
                if (m_rgDataQueue.Count == 0)
                    return null;

                return m_rgDataQueue.Dequeue();
            }
        }

        /// <summary>
        /// Reset the data query to and offset from the start date.
        /// </summary>
        /// <param name="nStartOffset">Specifies the offset to use.</param>
        public void Reset(int nStartOffset)
        {
            m_evtQueryEnabled.Reset();
            m_evtPaused.WaitOne();

            m_iquery.Reset();
            m_dt = m_dtStart;

            if (nStartOffset != 0)
                m_dt += TimeSpan.FromMilliseconds(nStartOffset * m_tsInc.TotalMilliseconds);

            m_rgDataQueue.Clear();
            m_evtQueryEnabled.Set();
        }

        /// <summary>
        /// The query thread is where all data is collected from the underlying custom query managed.
        /// </summary>
        private void queryThread()
        {
            try
            {
                int nWait = 0;

                m_iquery.Open();

                while (!m_evtCancel.WaitOne(nWait))
                {
                    if (!m_evtQueryEnabled.WaitOne(0))
                    {
                        m_evtPaused.Set();
                        nWait = 250;
                        continue;
                    }
                    
                    m_evtPaused.Reset();

                    if (m_rgDataQueue.Count >= m_nMaxCount)
                    {
                        nWait = 10;
                        continue;
                    }

                    double[] rgData = m_iquery.QueryByTime(m_dt, m_tsInc, m_nSegmentSize);
                    if (rgData == null)
                    {
                        m_bQueryEnd = true;
                        nWait = 10;
                        continue;
                    }

                    nWait = 0;
                    m_bQueryEnd = false;

                    int nItemCount = rgData.Length / m_nSegmentSize;
                    int nSrcIdx = 0;

                    lock (m_objSync)
                    {
                        for (int i = 0; i < m_nSegmentSize; i++)
                        {
                            double[] rgItem = new double[nItemCount];
                            Array.Copy(rgData, nSrcIdx, rgItem, 0, nItemCount);
                            nSrcIdx += nItemCount;

                            m_rgDataQueue.Enqueue(rgItem);
                        }
                    }

                    m_dt += TimeSpan.FromMilliseconds(m_nSegmentSize * m_tsInc.TotalMilliseconds);
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                m_iquery.Close();
            }
        }
    }
}
