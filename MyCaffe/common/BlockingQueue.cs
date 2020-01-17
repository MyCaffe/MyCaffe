using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using MyCaffe.basecode;

namespace MyCaffe.common
{
    /// <summary>
    /// The BlockingQueue is used for synchronized Queue operations.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class BlockingQueue<T> : IDisposable 
    {
        CancelEvent m_evtCancel;
        ManualResetEvent m_evtReady = new ManualResetEvent(false);
        ManualResetEvent m_evtAbort = new ManualResetEvent(false);
        object m_syncObj = new object();
        List<T> m_rgData = new List<T>();
        int m_nDataCount = 0;

        /// <summary>
        /// The BlockingQueue constructor.
        /// </summary>
        /// <param name="evtCancel">Specifies a CancelEvent used to terminate wait sates within the Queue.</param>
        public BlockingQueue(CancelEvent evtCancel)
        {
            m_evtCancel = evtCancel;
        }

        /// <summary>
        /// Reset the abort event.
        /// </summary>
        public void Reset()
        {
            m_evtAbort.Reset();
        }

        /// <summary>
        /// Cancel the blocking queue operations.
        /// </summary>
        public void Abort()
        {
            m_evtAbort.Set();
        }

        /// <summary>
        /// Return the number of items in the queue.
        /// </summary>
        public int Count
        {
            get { return m_nDataCount; }
        }

        /// <summary>
        /// Returns the item at a given index within the queue.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <param name="t">Specifies the data value at the index.</param>
        /// <returns>If the CancelEvent is null or is set, <i>false</i> is returned, otherwise if the data is successfully retrieved <i>true</i> is returned.</returns>
        public bool GetAt(int nIdx, ref T t)
        {
            while (Count <= nIdx)
            {
                if (m_evtCancel == null)
                    return false;

                WaitHandle[] rghCancel = m_evtCancel.Handles;

                if (m_evtReady == null || m_evtAbort == null)
                    return false;

                ManualResetEvent evtReady = m_evtReady;
                ManualResetEvent evtAbort = m_evtAbort;

                List<WaitHandle> rgWait = new List<WaitHandle>();
                rgWait.AddRange(rghCancel);
                rgWait.Add(evtAbort);
                rgWait.Add(evtReady);

                int nWait = WaitHandle.WaitAny(rgWait.ToArray());
                if (nWait < rgWait.Count - 1)
                {
                    evtAbort.Reset();
                    return false;
                }

                evtReady.Reset();
            }

            lock (m_syncObj)
            {
                t = m_rgData[nIdx];
                return true;
            }
        }

        /// <summary>
        /// Remove all items from the queue.
        /// </summary>
        public void Clear()
        {
            lock (m_syncObj)
            {
                m_rgData.Clear();
                m_nDataCount = 0;             
            }
        }

        /// <summary>
        /// Add an item to the back of the queue.
        /// </summary>
        /// <param name="t">Specifies the item to add.</param>
        public void Push(T t)
        {
            lock (m_syncObj)
            {
                if (m_evtReady != null)
                {
                    m_rgData.Add(t);
                    m_nDataCount = m_rgData.Count;
                    m_evtReady.Set();
                }
            }
        }

        /// <summary>
        /// Remove an item from the front of the queue.
        /// </summary>
        /// <remarks>
        /// This function will wait until either data is added to the queue or the CancelEvent or AbortEvent are set.
        /// </remarks>
        /// <param name="t">Specifies the item removed.</param>
        /// <returns>If a CancelEvent occurs, <i>false</i> is returned, otherwise if the data is successfully removed from the queue <i>true</i> is returned.</returns>
        public bool Pop(ref T t)
        {
            while (Count == 0)
            {
                WaitHandle[] rghCancel = m_evtCancel.Handles;

                if (m_evtCancel == null || m_evtAbort == null)
                    return false;

                ManualResetEvent evtAbort = m_evtAbort;
                ManualResetEvent evtReady = m_evtReady;

                List<WaitHandle> rgWait = new List<WaitHandle>();
                rgWait.AddRange(rghCancel);
                rgWait.Add(evtAbort);
                rgWait.Add(evtReady);

                int nWait = WaitHandle.WaitAny(rgWait.ToArray());
                if (nWait < rgWait.Count - 1)
                    return false;

                evtReady.Reset();
            }

            lock (m_syncObj)
            {
                t = m_rgData[0];
                m_rgData.RemoveAt(0);
                m_nDataCount = m_rgData.Count;
                return true;
            }
        }

        /// <summary>
        /// Retrieve an item from the front of the queue, but do not remove it.
        /// </summary>
        /// <remarks>
        /// This function will wait until either data is added to the queue or the CancelEvent or AbortEvent are set.
        /// </remarks>
        /// <param name="t">Specifies the item removed.</param>
        /// <returns>If a CancelEvent occurs, <i>false</i> is returned, otherwise if the data is successfully removed from the queue <i>true</i> is returned.</returns>
        public bool Peek(ref T t)
        {
            while (Count == 0)
            {
                if (m_evtReady == null)
                    return false;

                if (m_evtCancel.WaitOne(0))
                    return false;

                Thread.Sleep(0);
            }

            lock (m_syncObj)
            {
                t = m_rgData[0];
                return true;
            }
        }

        /// <summary>
        /// Release all resources used by the queue.
        /// </summary>
        /// <param name="bDisposing">Set to <i>true</i> when called from Dispose().</param>
        protected virtual void Dispose(bool bDisposing)
        {
            if (m_evtAbort != null)
            {
                m_evtAbort.Set();
                Thread.Sleep(50);
                m_evtAbort.Dispose();
                m_evtAbort = null;
            }

            if (m_evtReady != null)
            {
                m_evtReady.Dispose();
                m_evtReady = null;
            }

            Clear();
        }

        /// <summary>
        /// Release all resources used by the queue.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }
    }
}
