using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
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

        public void Dispose()
        {
            Shutdown();
        }

        public void Shutdown()
        {
            m_evtCancel.Set();
        }

        public int FieldCount
        {
            get { return m_iquery.FieldCount; }
        }

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

        public int Count
        {
            get { return m_rgDataQueue.Count; }
        }

        public bool DataReady(int nCount)
        {
            if (m_rgDataQueue.Count < nCount)
                return false;

            return true;
        }

        public bool DataDone()
        {
            return m_bQueryEnd;
        }

        public double[] PeekDataAt(int nIdx)
        {
            if (nIdx >= m_rgDataQueue.Count)
                return null;

            return m_rgDataQueue.ElementAt(nIdx);
        }

        public double PeekDataAt(int nIdx, int nFieldIdx)
        {
            double[] rg = m_rgDataQueue.ElementAt(nIdx);
            return rg[nFieldIdx];
        }

        public double[] GetNextData()
        {
            lock (m_objSync)
            {
                if (m_rgDataQueue.Count == 0)
                    return null;

                return m_rgDataQueue.Dequeue();
            }
        }

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
