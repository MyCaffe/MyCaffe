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
        /// <summary>
        /// The constructor.
        /// </summary>
        public Memory()
        {
        }

        /// <summary>
        /// Get all targets in memory as an array of the base type.
        /// </summary>
        /// <returns>The array of target values is returned.</returns>
        public T[] GetTargets(int nMax)
        {
            List<T> rg = new List<T>();

            for (int i = 0; i < m_rgItems.Count && i < nMax; i++)
            {
                T fVal = Utility.ConvertVal<T>(m_rgItems[i].Target);
                rg.Add(fVal);
            }

            return rg.ToArray();
        }

        /// <summary>
        /// Get all actions in memory as an array of the base type.
        /// </summary>
        /// <returns>The array of action values is returned.</returns>
        public T[] GetActions(int nMax)
        {
            List<T> rg = new List<T>();

            for (int i = 0; i < m_rgItems.Count && i < nMax; i++)
            {
                T fVal = Utility.ConvertVal<T>(m_rgItems[i].Action);
                rg.Add(fVal);
            }

            return rg.ToArray();
        }

        /// <summary>
        /// Return the actions as a vector of one-hot-vectors each of size nActionCount.
        /// </summary>
        /// <param name="nMax">Specifies the maximum number of items to return.</param>
        /// <param name="nActionCount">Specifies the action count of each one-hot-vector of the batch of one-hot-vectors.</param>
        /// <returns>The set of one-hot-vectors is returned as an array.</returns>
        public T[] GetActionOneHot(int nMax, int nActionCount)
        {
            T tOne = (T)Convert.ChangeType(1.0, typeof(T));
            T tZero = (T)Convert.ChangeType(0.0, typeof(T));
            List<T> rg = new List<T>();

            for (int i = 0; i < m_rgItems.Count && i < nMax; i++)
            {
                int nAction = m_rgItems[i].Action;

                for (int j = 0; j < nActionCount; j++)
                {
                    rg.Add((j == nAction) ? tOne : tZero);
                }
            }

            return rg.ToArray();
        }
    }
}
