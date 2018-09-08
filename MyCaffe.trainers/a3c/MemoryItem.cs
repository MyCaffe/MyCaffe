using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers.a3c
{
    /// <summary>
    /// The memory item is a single item stored within the Memory making up an experience.
    /// </summary>
    public class MemoryItem /** @private */
    {
        StateBase m_state0;
        StateBase m_state1;
        int m_nAction;
        double m_dfReward = 0;
        int m_nFrameIdx;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="s0">Specifies the state information.</param>
        /// <param name="nAction">Specifies the action taken.</param>
        /// <param name="dfReward">Specifies the reward for taking the action.</param>
        /// <param name="s1">Specifies the new state information.</param>
        public MemoryItem(StateBase s0, int nAction, double dfReward, StateBase s1, int nFrameIdx)
        {
            m_state0 = s0;
            m_state1 = s1;
            m_nAction = nAction;
            m_dfReward = dfReward;
            m_nFrameIdx = nFrameIdx;
        }

        /// <summary>
        /// Return the frame index (episode) under which this memory item belongs.
        /// </summary>
        public int FrameIndex
        {
            get { return m_nFrameIdx; }
        }

        /// <summary>
        /// Get the state information.
        /// </summary>
        public StateBase State0
        {
            get { return m_state0; }
        }

        /// <summary>
        /// Get the new state information.
        /// </summary>
        public StateBase State1
        {
            get { return m_state1; }
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
        /// Return a string representation of the MemoryItem.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "Action=" + m_nAction.ToString() + " Reward=" + m_dfReward.ToString() + " Mask = " + ((m_state1 == null) ? "0" : "1");
        }
    }
}
