using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;

namespace MyCaffe.trainers.common
{
    /// <summary>
    /// The RandomMemoryCollection is used to randomly sample the collection of items.
    /// </summary>
    public class RandomMemoryCollection : IMemoryCollection
    {
        MemoryCollection m_mem;
        double[] m_rgWts = null;
        int[] m_rgIdx = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nMax">Specifies the maximum number of items.</param>
        public RandomMemoryCollection(int nMax)
        {
            m_mem = new MemoryCollection(nMax);
        }

        /// <summary>
        /// Complete any final processing.
        /// </summary>
        public void CleanUp()
        {
        }

        /// <summary>
        /// Returns the number of items in the collection.
        /// </summary>
        public int Count
        {
            get { return m_mem.Count; }
        }

        /// <summary>
        /// Add a new item to the collection.
        /// </summary>
        /// <param name="m">Specifies the item to add.</param>
        public void Add(MemoryItem m)
        {
            m_mem.Add(m);
        }

        /// <summary>
        /// Return a batch of items.
        /// </summary>
        /// <param name="random">Specifies the random number generator.</param>
        /// <param name="nCount">Specifies the number of items to sample.</param>
        /// <param name="dfBeta">Not used.</param>
        /// <returns>The random array of items is returned.</returns>
        public MemoryCollection GetSamples(CryptoRandom random, int nCount, double dfBeta)
        {
            if (m_rgWts == null || m_rgWts.Length != nCount)
            {
                m_rgWts = new double[nCount];
                for (int i = 0; i < m_rgWts.Length; i++)
                {
                    m_rgWts[i] = 1.0;
                }
            }

            if (m_rgIdx == null || m_rgIdx.Length != nCount)
            {
                m_rgIdx = new int[nCount];
            }

            MemoryCollection mem = m_mem.GetRandomSamples(random, nCount);

            mem.Indexes = m_rgIdx;
            mem.Priorities = m_rgWts;

            return mem;
        }

        /// <summary>
        /// Update - does nothing.
        /// </summary>
        /// <param name="rgSamples">Specifies the list of samples.</param>
        public void Update(MemoryCollection rgSamples)
        {
        }
    }
}
