using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    public class DataItemCollection
    {
        object m_objSync = new object();
        List<DataItem> m_rgItems = new List<DataItem>();
        ManualResetEvent m_evtDataReady = new ManualResetEvent(false);
        ManualResetEvent m_evtAtCount = new ManualResetEvent(false);
        ManualResetEvent m_evtQueryEnd = new ManualResetEvent(false);
        CancelEvent m_evtCancel = new CancelEvent();
        int m_nAtCount = 0;

        public DataItemCollection(int nAtCount)
        {
            m_nAtCount = nAtCount;
        }

        public int Count
        {
            get { return m_rgItems.Count; }
        }

        public CancelEvent Cancel
        {
            get { return m_evtCancel; }
        }

        public ManualResetEvent QueryEnd
        {
            get { return m_evtQueryEnd; }
        }

        public bool WaitForCount(int nWait)
        {          
            List<WaitHandle> rgWait = new List<WaitHandle>();
            rgWait.AddRange(m_evtCancel.Handles);
            rgWait.Add(m_evtAtCount);
            rgWait.Add(m_evtQueryEnd);
            WaitHandle[] rgWait1 = rgWait.ToArray();

            int nWaitItem = WaitHandle.WaitAny(rgWait1, 10);
            if (nWaitItem == rgWait.Count - 2)
                return true;

            Stopwatch sw = new Stopwatch();
            sw.Start();

            while (nWaitItem >= m_evtCancel.Handles.Length)
            {
                if (nWaitItem == rgWait.Count - 2)
                { 
                    return true;
                }
                else if (nWaitItem == rgWait.Count - 1)
                {
                    lock (m_objSync)
                    {
                        if (m_rgItems.Count == 0)
                            return false;
                    }
                }

                if (sw.Elapsed.TotalMilliseconds > nWait)
                    return false;

                nWaitItem = WaitHandle.WaitAny(rgWait1, 10);
            }

            return false;
        }

        public void Add(DataItem di)
        {
            lock (m_objSync)
            {
                m_rgItems.Add(di);
                m_evtDataReady.Set();

                if (m_rgItems.Count >= m_nAtCount)
                    m_evtAtCount.Set();
            }
        }

        public DataItem GetData(int nWait)
        {
            if (!m_evtDataReady.WaitOne(nWait))
                return null;

            lock (m_objSync)
            {
                DataItem di = m_rgItems[0];
                m_rgItems.RemoveAt(0);

                if (m_rgItems.Count == 0)
                    m_evtDataReady.Reset();

                if (m_rgItems.Count < m_nAtCount)
                    m_evtAtCount.Reset();

                return di;
            }
        }

        public bool WaitData(int nWait)
        {
            return m_evtDataReady.WaitOne(nWait);
        }

        public void Clear()
        {
            lock (m_objSync)
            {
                m_rgItems.Clear();
                m_evtQueryEnd.Reset();
                m_evtDataReady.Reset();
                m_evtAtCount.Reset();
            }
        }
    }
}
