using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Drawing;
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
        bool m_bDone = false;
        bool m_bValid = true;
        double m_dfReward = 0;
        int m_nActionCount = 0;
        SimpleDatum m_data = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        public StateBase(int nActionCount)
        {
            m_nActionCount = nActionCount;
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

        /// <summary>
        /// Get/set whether the state is done or not.
        /// </summary>
        public bool Done
        {
            get { return m_bDone; }
            set { m_bDone = value; }
        }

        /// <summary>
        /// Returns the number of actions.
        /// </summary>
        public int ActionCount
        {
            get { return m_nActionCount; }           
        }

        /// <summary>
        /// Returns other data associated with the state.
        /// </summary>
        public SimpleDatum Data
        {
            get { return m_data; }
            set { m_data = value; }
        }
    }
}
