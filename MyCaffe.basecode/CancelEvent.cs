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
        bool m_bOwnOriginal = true;
        WaitHandle m_hOriginalCancel = null;
        List<Tuple<WaitHandle, bool, string>> m_rgCancel = new List<Tuple<WaitHandle, bool, string>>();
        string m_strName = null;

        /// <summary>
        /// The CancelEvent constructor.
        /// </summary>
        public CancelEvent()
        {
            m_hOriginalCancel = new EventWaitHandle(false, EventResetMode.ManualReset, m_strName);
        }


        /// <summary>
        /// The CancelEvent constructor that accepts a global name.
        /// </summary>
        /// <param name="strGlobalName">Specifies the global name.</param>
        public CancelEvent(string strGlobalName)
        {
            m_strName = strGlobalName;
            m_hOriginalCancel = EventWaitHandle.OpenExisting(strGlobalName, System.Security.AccessControl.EventWaitHandleRights.Synchronize | System.Security.AccessControl.EventWaitHandleRights.Modify);
        }

        /// <summary>
        /// Create a new Cancel Event and add another to this ones overrides.
        /// </summary>
        /// <param name="evtCancel">Specifies the Cancel Event to add to the overrides.</param>
        public CancelEvent(CancelEvent evtCancel)
        {
            m_hOriginalCancel = new EventWaitHandle(false, EventResetMode.ManualReset, m_strName);
            m_rgCancel.Add(new Tuple<WaitHandle, bool, string>(evtCancel.m_hOriginalCancel, false, evtCancel.Name));
        }

        /// <summary>
        /// Add a new cancel override.
        /// </summary>
        /// <param name="strName">Specifies the name of the cancel event to add.</param>
        public void AddCancelOverride(string strName)
        {
            EventWaitHandle evtWait = EventWaitHandle.OpenExisting(strName, System.Security.AccessControl.EventWaitHandleRights.Synchronize | System.Security.AccessControl.EventWaitHandleRights.Modify);
            m_rgCancel.Add(new Tuple<WaitHandle, bool, string>(evtWait, true, strName));
        }

        /// <summary>
        /// Add a new cancel override.
        /// </summary>
        /// <param name="evtCancel">Specifies the cancel override to add.</param>
        public void AddCancelOverride(CancelEvent evtCancel)
        {
            m_rgCancel.Add(new Tuple<WaitHandle, bool, string>(evtCancel.m_hOriginalCancel, false, evtCancel.Name));
        }

        /// <summary>
        /// Remove a new cancel override.
        /// </summary>
        /// <param name="strName">Specifies the name of the cancel event to remove.</param>
        /// <returns>If removed, <i>true</i> is returned.</returns>
        public bool RemoveCancelOverride(string strName)
        {
            int nIdx = -1;

            for (int i = 0; i < m_rgCancel.Count; i++)
            {
                if (m_rgCancel[i].Item3 == strName)
                {
                    nIdx = i;
                    break;
                }
            }

            if (nIdx >= 0)
            {
                if (m_rgCancel[nIdx].Item2)
                    m_rgCancel[nIdx].Item1.Dispose();

                m_rgCancel.RemoveAt(nIdx);
                return true;
            }

            return false;
        }

        /// <summary>
        /// Remove a new cancel override.
        /// </summary>
        /// <param name="evtCancel">Specifies the cancel override to remove.</param>
        /// <returns>If removed, <i>true</i> is returned.</returns>
        public bool RemoveCancelOverride(CancelEvent evtCancel)
        {
            int nIdx = -1;

            for (int i = 0; i < m_rgCancel.Count; i++)
            {
                if (m_rgCancel[i].Item1 == evtCancel.m_hOriginalCancel)
                {
                    nIdx = i;
                    break;
                }
            }

            if (nIdx >= 0)
            {
                if (m_rgCancel[nIdx].Item2)
                    m_rgCancel[nIdx].Item1.Dispose();

                m_rgCancel.RemoveAt(nIdx);
                return true;
            }

            return false;
        }

        /// <summary>
        /// Return the name of the cancel event.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Sets the event to the signaled state.
        /// </summary>
        public void Set()
        {
            if (m_hOriginalCancel is EventWaitHandle)
                ((EventWaitHandle)m_hOriginalCancel).Set();
        }

        /// <summary>
        /// Resets the event clearing any signaled state.
        /// </summary>
        public void Reset()
        {
            if (m_hOriginalCancel is EventWaitHandle)
                ((EventWaitHandle)m_hOriginalCancel).Reset();
        }

        /// <summary>
        /// Waits for the signal state to occur.
        /// </summary>
        /// <param name="nMs">Specifies the number of milliseconds to wait.</param>
        /// <returns>If the CancelEvent is in the signal state, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool WaitOne(int nMs = int.MaxValue)
        {
            if (WaitHandle.WaitAny(Handles, nMs) == WaitHandle.WaitTimeout)
                return false;

            return true;
        }

        /// <summary>
        /// Returns the internal wait handle of the CancelEvent.
        /// </summary>
        public WaitHandle[] Handles
        {
            get
            {
                List<WaitHandle> rgHandles = new List<WaitHandle>() { m_hOriginalCancel };

                foreach (Tuple<WaitHandle, bool, string> item in m_rgCancel)
                {
                    rgHandles.Add(item.Item1);
                }

                return rgHandles.ToArray();
            }
        }

        #region IDisposable Support

        /// <summary>
        /// Releases all resources used by the CancelEvent.
        /// </summary>
        /// <param name="disposing">Specifies whether or not this was called from Dispose().</param>
        protected virtual void Dispose(bool disposing)
        {
            if (m_bOwnOriginal && m_hOriginalCancel != null)
            {
                m_hOriginalCancel.Dispose();
                m_hOriginalCancel = null;
            }

            foreach (Tuple<WaitHandle, bool, string> item in m_rgCancel)
            {
                if (item.Item2)
                    item.Item1.Dispose();
            }

            m_rgCancel.Clear();
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
