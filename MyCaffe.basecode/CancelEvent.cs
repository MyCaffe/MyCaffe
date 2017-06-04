using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The CancelEvent provides an extension to the manual cancel event that allows for overriding the
    /// manual cancel event.
    /// </summary>
    /// <remarks>
    /// The CancelEvent is used by the CaffeControl to cancel training and testing operations.
    /// </remarks>
    public class CancelEvent : IDisposable
    {
        bool m_bOwnOriginal = false;
        ManualResetEvent m_evtOriginalCancel = null;
        ManualResetEvent m_evtCancel = null;

        /// <summary>
        /// The CancelEvent constructor.
        /// </summary>
        public CancelEvent()
        {
            m_bOwnOriginal = true;
            m_evtOriginalCancel = new ManualResetEvent(false);
            m_evtCancel = m_evtOriginalCancel;
        }

        /// <summary>
        /// The CancelEvent constructor.
        /// </summary>
        /// <param name="evtCancel">Specifies an external manual reset event to wrap.</param>
        public CancelEvent(ManualResetEvent evtCancel)
        {
            m_evtOriginalCancel = evtCancel;
            m_evtCancel = evtCancel;
        }

        /// <summary>
        /// Sets the actual cancel event used to an external manual reset event.
        /// </summary>
        /// <param name="evtCancel">Specifies an external manual reset event to wrap.</param>
        public void SetCancelOverride(ManualResetEvent evtCancel)
        {
            if (evtCancel == null)
                evtCancel = m_evtOriginalCancel;

            m_evtCancel = evtCancel;
        }

        /// <summary>
        /// Sets the event to the signaled state.
        /// </summary>
        public void Set()
        {
            m_evtCancel.Set();
        }

        /// <summary>
        /// Resets the event clearing any signaled state.
        /// </summary>
        public void Reset()
        {
            m_evtCancel.Reset();
        }

        /// <summary>
        /// Waits for the signal state to occur.
        /// </summary>
        /// <param name="nMs">Specifies the number of milliseconds to wait.</param>
        /// <returns>If the CancelEvent is in the signal state, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool WaitOne(int nMs = int.MaxValue)
        {
            return m_evtCancel.WaitOne(nMs);
        }

        /// <summary>
        /// Returns the internal wait handle of the CancelEvent.
        /// </summary>
        public WaitHandle Handle
        {
            get { return m_evtCancel; }
        }

        #region IDisposable Support

        /// <summary>
        /// Releases all resources used by the CancelEvent.
        /// </summary>
        /// <param name="disposing">Specifies whether or not this was called from Dispose().</param>
        protected virtual void Dispose(bool disposing)
        {
            if (m_bOwnOriginal && m_evtOriginalCancel != null)
            {
                m_evtOriginalCancel.Dispose();
                m_evtOriginalCancel = null;
            }
        }

        /// <summary>
        /// Releases all resources used by the CancelEvent.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }
        #endregion
    }
}
