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
    /// The ObservationReady contains a single observation and notifies an event when it is ready (e.g. has been recieved).
    /// </summary>
    public class ObservationReady
    {
        AutoResetEvent m_evtReady = new AutoResetEvent(false);
        Observation m_observation = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="obs">Specifies the observation received.</param>
        public ObservationReady(Observation obs = null)
        {
            m_observation = obs;

            if (obs != null)
                m_evtReady.Set();
        }

        /// <summary>
        /// Returns <i>true</i> if an observation is recieved within the given wait time.
        /// </summary>
        /// <param name="nWait">Specifies the maximim amount of time to wait (in milliseconds) for an observation.</param>
        /// <returns>After receiving an observation, <i>true</i> is returned, otherwise on timeout <i>false</i> is returned.</returns>
        public bool IsReady(int nWait)
        {
            return m_evtReady.WaitOne(nWait);
        }

        /// <summary>
        /// Get/set the observation.
        /// </summary>
        public Observation Observation
        {
            set
            {
                m_observation = value;
                m_evtReady.Set();
            }

            get
            {
                Observation obs = m_observation;
                m_observation = null;
                return obs;
            }
        }
    }
}
