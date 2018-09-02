using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{
    /// <summary>
    /// Specifies the memory making up an experience.
    /// </summary>
    class Memory<T> : GenericList<MemoryItem>
    {
        T m_tOne = Utility.ConvertVal<T>(1.0);
        T m_tZero = Utility.ConvertVal<T>(0.0);
        double m_dfRewardScale = 1.0;

        public enum STATE_TYPE
        {
            OLD,
            NEW
        }

        public enum VALUE_TYPE
        {
            VALUE,
            ONEHOT
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        public Memory(double dfRewardScale)
        {
            m_dfRewardScale = dfRewardScale;
        }


        /// <summary>
        /// Clone the memory into a new memory.
        /// </summary>
        /// <param name="bClear">If <i>true</i>, clear the original memory.</param>
        /// <returns>The cloned memory is returned.</returns>
        public Memory<T> Clone(bool bClear)
        {
            Memory<T> col = new Memory<T>(m_dfRewardScale);

            col.m_rgItems.AddRange(m_rgItems);

            if (bClear)
                m_rgItems.Clear();

            return col;
        }

        /// <summary>
        /// Get all states as a list of Datums.
        /// </summary>
        /// <param name="type">Specifies the information type to retrieve.</param>
        /// <returns>The list of datums is returned.</returns>
        public List<Datum> GetStates(STATE_TYPE type)
        {
            List<Datum> rg = new List<Datum>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                if (type == STATE_TYPE.OLD)
                    rg.Add(new Datum(m_rgItems[i].State0.Data));
                else
                    rg.Add(m_rgItems[i].State1 == null ? null : new Datum(m_rgItems[i].State1.Data));
            }

            return rg;
        }

        /// <summary>
        /// Get all actions in memory as an array of values or one hot vectors.
        /// </summary>
        /// <param name="type">Specifies the type value or one-hot-vector.</param>
        /// <returns>The array of action values or one hot vectors is returned.</returns>
        public T[] GetActions(VALUE_TYPE type)
        {
            List<T> rg = new List<T>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                int nAction = m_rgItems[i].Action;

                if (type == VALUE_TYPE.ONEHOT)
                {
                    float[] rgOneHot = new float[m_rgItems[i].State0.ActionCount];
                    rgOneHot[nAction] = 1;

                    rg.AddRange(Utility.ConvertVec<T>(rgOneHot));
                }
                else
                {
                    rg.Add(Utility.ConvertVal<T>((double)nAction));
                }
            }

            return rg.ToArray();
        }

        /// <summary>
        /// Returns the total rewards for all items in the memory.
        /// </summary>
        public double TotalReward
        {
            get { return m_rgItems.Sum(p => p.Reward) * m_dfRewardScale; }
        }

        /// <summary>
        /// Get all rewards in memory as an array of base types.
        /// </summary>
        /// <returns>The array of rewards is returned.</returns>
        public T[] GetRewards()
        {
            List<T> rg = new List<T>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                double dfVal = m_rgItems[i].Reward * m_dfRewardScale;
                rg.Add(Utility.ConvertVal<T>(dfVal));
            }

            return rg.ToArray();
        }

        /// <summary>
        /// Get all state masks (0 for done, 1 for not done) in memory as an array of base types.
        /// </summary>
        /// <returns>The array of state masks is returned.</returns>
        public T[] GetStateMask()
        {
            List<T> rg = new List<T>();

            for (int i = 0; i < m_rgItems.Count; i++)
            {
                rg.Add(m_rgItems[i].State1 == null ? m_tZero : m_tOne);
            }

            return rg.ToArray();
        }
    }
}
