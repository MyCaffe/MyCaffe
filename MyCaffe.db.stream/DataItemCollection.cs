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
    /// <summary>
    /// The DataItemCollection contains the collection of synchronized data items collected from all custom queries.
    /// </summary>
    public class DataItemCollection
    {
        object m_objSync = new object();
        List<DataItem> m_rgItems = new List<DataItem>();
        ManualResetEvent m_evtDataReady = new ManualResetEvent(false);
        ManualResetEvent m_evtAtCount = new ManualResetEvent(false);
        ManualResetEvent m_evtQueryEnd = new ManualResetEvent(false);
        CancelEvent m_evtCancel = new CancelEvent();
        int m_nAtCount = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nAtCount">Specifies the number of items that trigger the 'AtCount' event.</param>
        public DataItemCollection(int nAtCount)
        {
            m_nAtCount = nAtCount;
        }

        /// <summary>
        /// Returns the number of items in the queue.
        /// </summary>
        public int Count
        {
            get { return m_rgItems.Count; }
        }

        /// <summary>
        /// Cancels the internal WaitForCount.
        /// </summary>
        public CancelEvent Cancel
        {
            get { return m_evtCancel; }
        }

        /// <summary>
        /// The QueryEnd is set when the data reaches the data end.
        /// </summary>
        public ManualResetEvent QueryEnd
        {
            get { return m_evtQueryEnd; }
        }

        /// <summary>
        /// The WaitForCount function waits for the data queue to either fill to a given number of items (e.g. the 'at count'), or 
        /// if no items remain in the queue and the query end has been reached, or the cancel event has been set.
        /// </summary>
        /// <param name="nWait">Specifies the maximum amount of time to wait.</param>
        /// <returns><i>true</i> is returned if there is an 'at count' amount of data in the queue, otherwise <i>false</i> is returned.</returns>
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

        /// <summary>
        /// Add a new data item to the queue.
        /// </summary>
        /// <param name="di">Specifies the synchronized data item.</param>
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

        /// <summary>
        /// Returns the next data item from the back of the queue.
        /// </summary>
        /// <param name="nWait">Specifies the amount of time to wait for the data.</param>
        /// <returns>The synchronized data item is returned.</returns>
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

        /// <summary>
        /// The WaitData function waits a given amount of time for data to be ready.
        /// </summary>
        /// <param name="nWait">Specifies the amount of time to wait.</param>
        /// <returns>When data is ready, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool WaitData(int nWait)
        {
            return m_evtDataReady.WaitOne(nWait);
        }

        /// <summary>
        /// The Clear method removes all data from the data queue.
        /// </summary>
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
