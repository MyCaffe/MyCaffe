using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers.common
{
    /// <summary>
    /// The PrioritizedMemoryCollection provides a sampling based on prioritizations.
    /// </summary>
    public class PrioritizedMemoryCollection : IMemoryCollection 
    {
        float m_fAlpha;
        double m_fMaxPriority = 1.0f;
        int m_nItCapacity = 1;
        MemoryCollection m_mem;
        SumSegmentTree m_ItSum;
        MinSegmentTree m_ItMin;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nMax">Specifies the maximum number of items in the collection.</param>
        /// <param name="fAlpha">Specifies how much prioritization is used (0 = no prioritization, 1 = full prioritization).</param>
        public PrioritizedMemoryCollection(int nMax, float fAlpha)
        {
            m_mem = new MemoryCollection(nMax);
            m_fAlpha = fAlpha;

            while (m_nItCapacity < nMax)
            {
                m_nItCapacity *= 2;
            }

            m_ItSum = new SumSegmentTree(m_nItCapacity);
            m_ItMin = new MinSegmentTree(m_nItCapacity);
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
            int nIdx = m_mem.NextIndex;
            m_mem.Add(m);

            int nVal = (int)Math.Pow(m_fMaxPriority, m_fAlpha);
            m_ItSum[nIdx] = nVal;
            m_ItMin[nIdx] = nVal;
        }

        private int[] getSamplesProportional(CryptoRandom random, int nCount)
        {
            int[] rgIdx = new int[nCount];

            for (int i = 0; i < nCount; i++)
            {
                double dfRand = random.NextDouble();
                double dfSum1 = m_ItSum.sum(0, m_mem.Count - 1);
                double dfMass = dfRand * dfSum1;
                int nIdx = m_ItSum.find_prefixsum_idx((float)dfMass);
                rgIdx[i] = nIdx;
            }

            return rgIdx;
        }

        /// <summary>
        /// Return a batch of items.
        /// </summary>
        /// <param name="random">Specifies the random number generator.</param>
        /// <param name="nCount">Specifies the number of items to sample.</param>
        /// <param name="dfBeta">Specifies the degree to use importance weights (0 = no corrections, 1 = full corrections).</param>
        /// <returns>The prioritized array of items is returned along with the weights and indexes.</returns>
        public MemoryCollection GetSamples(CryptoRandom random, int nCount, double dfBeta)
        {
            int[] rgIdx = getSamplesProportional(random, nCount);
            double[] rgfWeights = new double[nCount];
            double fSum = m_ItSum.sum();
            double fMin = m_ItMin.min();
            double fPMin = fMin / fSum;
            double fMaxWeight = (float)Math.Pow(fPMin * m_mem.Count, -dfBeta);
            MemoryCollection mem = new MemoryCollection(nCount);

            for (int i = 0; i < rgIdx.Length; i++)
            {
                int nIdx = rgIdx[i];
                double fItSum = m_ItSum[nIdx];
                double fPSample = fItSum / fSum;
                double fWeight = Math.Pow(fPSample * m_mem.Count, -dfBeta);
                rgfWeights[i] = fWeight / fMaxWeight;

                mem.Add(m_mem[nIdx]);
            }

            mem.Indexes = rgIdx;
            mem.Priorities = rgfWeights;

            return mem;
        }

        /// <summary>
        /// Update the priorities of sampled transitions.
        /// </summary>
        /// <remarks>
        /// Sets priority of transitions at index rgIdx[i] in buffer to priorities[i].
        /// </remarks>
        /// <param name="rgSamples">Specifies the list of samples with updated priorities.</param>
        public void Update(MemoryCollection rgSamples)
        {
            int[] rgIdx = rgSamples.Indexes;
            double[] rgfPriorities = rgSamples.Priorities;

            if (rgIdx.Length != rgfPriorities.Length)
                throw new Exception("The index and priority arrays must have the same length.");

            for (int i = 0; i < rgIdx.Length; i++)
            {
                int nIdx = rgIdx[i];
                double fPriority = rgfPriorities[i];

                if (fPriority <= 0)
                    throw new Exception("The priority at index '" + i.ToString() + "' is zero!");

                if (nIdx < 0 || nIdx >= m_mem.Count)
                    throw new Exception("The index at index '" + i.ToString() + "' is out of range!");

                double fNewPriority = Math.Pow(fPriority, m_fAlpha);
                m_ItSum[nIdx] = fNewPriority;
                m_ItMin[nIdx] = fNewPriority;
                m_fMaxPriority = Math.Max(m_fMaxPriority, fPriority);
            }
        }
    } 
}
