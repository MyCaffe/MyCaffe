using MyCaffe.basecode;
using MyCaffe.common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The Observation class manages a single observation made up of the data (e.g. image), results from running the network on the data (e.g. the actions),
    /// the state (defined by the actual trainer overriding the MyCaffeCustomTrainer), and the previous action taken (if any) to reach the observation data.
    /// </summary>
    public class Observation
    {
        SimpleDatum m_sdObservation;
        ResultCollection m_Actions1;
        StateBase m_state;
        int m_nAction;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="sd">Specifies the data of the observation based on the previous action.</param>
        /// <param name="res">Specifies the new set of actions based on running the network on this observation's data and state.</param>
        /// <param name="state">Specifies the state.</param>
        /// <param name="nAction">Specifies the previous action taken (if any).</param>
        public Observation(SimpleDatum sd, ResultCollection res, StateBase state, int nAction)
        {
            m_sdObservation = sd;
            m_Actions1 = res;
            m_state = state;
            m_nAction = nAction;
        }

        /// <summary>
        /// Returns the observation data in the form of a SimpleDatum.
        /// </summary>
        public SimpleDatum Data
        {
            get { return m_sdObservation; }
        }

        /// <summary>
        /// Returns the actions output by running the network on the data and state.
        /// </summary>
        public ResultCollection Actions
        {
            get { return m_Actions1; }
        }

        /// <summary>
        /// Returns the current state data.
        /// </summary>
        public StateBase CurrentState
        {
            get { return m_state; }
        }

        /// <summary>
        /// Returns a string representation of the observation.
        /// </summary>
        /// <returns>The string representation of the observation is returned.</returns>
        public override string ToString()
        {
            return m_nAction.ToString();
        }
    }
}
