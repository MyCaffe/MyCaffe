using MyCaffe.basecode;
using MyCaffe.gym;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The ObservationReadyCollection manages a set of gym observations.
    /// </summary>
    public class ObservationReadyCollection
    {
        Dictionary<int, ObservationReady> m_rgItems = new Dictionary<int, ObservationReady>();
        CancelEvent m_evtCancel = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        public ObservationReadyCollection()
        {
        }

        /// <summary>
        /// The cancel event used to abort wating for a new observation.
        /// </summary>
        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
            set { m_evtCancel = value; }
        }

        /// <summary>
        /// Clear the set of observation items.
        /// </summary>
        public void Clear()
        {
            m_rgItems.Clear();
        }

        /// <summary>
        /// Add a new observation to the colleciton with its associated index.
        /// </summary>
        /// <remarks>
        /// Indexes are used to indicate observations from different gym instances.
        /// </remarks>
        /// <param name="nIdx">Specifies the gym instance index.</param>
        /// <param name="obs">Specifies the observation.</param>
        public void Add(int nIdx, Observation obs)
        {
            if (!m_rgItems.ContainsKey(nIdx))
                m_rgItems.Add(nIdx, new ObservationReady(obs));
            else
                m_rgItems[nIdx].Observation = obs;
        }

        /// <summary>
        /// Retrive the observation (when recievied) for a given gym instance index.
        /// </summary>
        /// <param name="nIdx">Specifies the gym instance index.</param>
        /// <param name="nMaxWait">Specifies the maximum number of milliseconds to wait for an observation.</param>
        /// <returns>If the wait expires, <i>null</i> is returned, otherwise the observation is returned.</returns>
        public Observation GetObservation(int nIdx, int nMaxWait)
        {
            int nWait = 0;

            while (!m_rgItems.ContainsKey(nIdx))
            {
                Thread.Sleep(100);
                nWait += 100;

                if (nWait >= nMaxWait)
                    return null;

                if (m_evtCancel != null && m_evtCancel.WaitOne(0))
                    return null;
            }

            if (!m_rgItems[nIdx].IsReady(nMaxWait))
                return null;

            return m_rgItems[nIdx].Observation;
        }
    }
}
