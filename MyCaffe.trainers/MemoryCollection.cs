using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The MemoryCollection contains a collection of Memory objects.
    /// </summary>
    /// <typeparam name="T">Specifies the base type used, either <i>float</i> or <i>double</i> - this should be the same base type used with MyCaffe.</typeparam>
    class MemoryCollection<T>
    {
        GenericList<Memory<T>> m_colMemory = new GenericList<Memory<T>>();
        int m_nMin = 0;
        int m_nMax = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nMin">Specifies the minimum number of Memory objects that must exists, afterwhich new Memories are only added if their total return is greater than the minimum total return in the set.</param>
        /// <param name="nMax">Specifies the maximum number of Memory objects, afterwhich the Memory with the minimum total return is removed from the set.</param>
        public MemoryCollection(int nMin, int nMax)
        {
            m_nMin = nMin;
            m_nMax = nMax;
        }

        /// <summary>
        /// Returns the Memory object with the minimum total reward.
        /// </summary>
        public Memory<T> MinimumTotalReward
        {
            get
            {
                Memory<T> mem = null;
                double dfTotalReward = double.MaxValue;

                for (int i = 0; i < m_colMemory.Count; i++)
                {
                    double dfReward = m_colMemory[i].TotalReward;

                    if (dfTotalReward > dfReward)
                    {
                        dfTotalReward = dfReward;
                        mem = m_colMemory[i];
                    }
                }

                return mem;
            }
        }

        /// <summary>
        /// Adds a new Memory to the collection base on the nMin and nMax settings passed to the constructor.
        /// </summary>
        /// <param name="mem">Specifies the new Memory item.</param>
        public void Add(Memory<T> mem)
        {
            Memory<T> memMin = null;

            if (m_colMemory.Count > m_nMin)
            {
                memMin = MinimumTotalReward;
                if (mem.TotalReward < memMin.TotalReward)
                    return;
            }
                
            m_colMemory.Add(mem);

            if (m_colMemory.Count > m_nMax)
                m_colMemory.Remove(memMin);
        }

        /// <summary>
        /// Returns a Memory, randomly selected from the collection.
        /// </summary>
        /// <param name="random">Specifies the random number generator.</param>
        /// <returns>The Memory object is returned.</returns>
        public Memory<T> GetMemory(Random random)
        {
            int nIdx = random.Next(m_colMemory.Count);
            return m_colMemory[nIdx];
        }
    }
}
