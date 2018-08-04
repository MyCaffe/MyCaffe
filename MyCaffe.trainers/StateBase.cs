using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The StateBase is the base class for the state of each observation - this is defined by actual trainer that overrides the MyCaffeCustomTrainer.
    /// </summary>
    public class StateBase
    {
        bool m_bValid = true;
        double m_dfReward = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        public StateBase()
        {
        }

        /// <summary>
        /// Get/set whether or not the state is valid.
        /// </summary>
        public bool IsValid
        {
            get { return m_bValid; }
            set { m_bValid = value; }
        }

        /// <summary>
        /// Get/set the reward of the state.
        /// </summary>
        public double Reward
        {
            get { return m_dfReward; }
            set { m_dfReward = value; }
        }
    }
}
