using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The memory item is a single item stored within the Memory making up an experience.
    /// </summary>
    public class MemoryItem
    {
        StateBase m_state;
        int m_nAction;
        double m_dfReward = 0;
        double m_dfTarget = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="s">Specifies the state information.</param>
        /// <param name="nAction">Specifies the action taken.</param>
        /// <param name="dfReward">Specifies the reward for taking the action.</param>
        public MemoryItem(StateBase s, int nAction, double dfReward)
        {
            m_state = s;
            m_nAction = nAction;
            m_dfReward = dfReward;
        }

        /// <summary>
        /// Get the state information.
        /// </summary>
        public StateBase State
        {
            get { return m_state; }
        }

        /// <summary>
        /// Get the action taken.
        /// </summary>
        public int Action
        {
            get { return m_nAction; }
        }

        /// <summary>
        /// Get the reward for taking the action.
        /// </summary>
        public double Reward
        {
            get { return m_dfReward; }
        }

        /// <summary>
        /// Get/set the target reward.
        /// </summary>
        public double Target
        {
            get { return m_dfTarget; }
            set { m_dfTarget = value; }
        }
    }
}
