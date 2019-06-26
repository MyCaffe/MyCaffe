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
    public class PrioritizedMemoryCollection : MemoryCollection
    {
        float m_fAlpha;
        float m_fMaxPriority = 1.0f;
        int m_nItCapacity = 1;
        SumSegmentTree m_ItSum;
        MinSegmentTree m_ItMin;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nMax">Specifies the maximum number of items in the collection.</param>
        /// <param name="fAlpha">Specifies how much prioritization is used (0 = no prioritization, 1 = full prioritization).</param>
        public PrioritizedMemoryCollection(int nMax, float fAlpha)
            : base(nMax)
        {
            m_fAlpha = fAlpha;

            while (m_nItCapacity < nMax)
            {
                m_nItCapacity *= 2;
            }

            m_ItSum = new SumSegmentTree(m_nItCapacity);
            m_ItMin = new MinSegmentTree(m_nItCapacity);
        }

        /// <summary>
        /// Add a new item to the collection.
        /// </summary>
        /// <param name="m"></param>
        public override void Add(MemoryItem m)
        {
            int nIdx = m_nNextIdx;
            base.Add(m);

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
                double dfSum1 = m_ItSum.sum(0, Count - 1);
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
        public Tuple<MemoryCollection, int[], float[]> GetSamples(CryptoRandom random, int nCount, double dfBeta)
        {
            int[] rgIdx = getSamplesProportional(random, nCount);
            float[] rgfWeights = new float[nCount];
            float fSum = m_ItSum.sum();
            float fMin = m_ItMin.min();
            float fPMin = fMin / fSum;
            float fMaxWeight = (float)Math.Pow(fPMin * Count, -dfBeta);
            MemoryCollection col = new MemoryCollection(nCount);

            for (int i = 0; i < rgIdx.Length; i++)
            {
                int nIdx = rgIdx[i];
                float fItSum = m_ItSum[nIdx];
                float fPSample = fItSum / fSum;
                float fWeight = (float)Math.Pow(fPSample * Count, -dfBeta);
                rgfWeights[i] = fWeight / fMaxWeight;

                col.Add(m_rgItems[nIdx]);
            }

            return new Tuple<MemoryCollection, int[], float[]>(col, rgIdx, rgfWeights);
        }

        /// <summary>
        /// Update the priorities of sampled transitions.
        /// </summary>
        /// <remarks>
        /// Sets priority of transitions at index rgIdx[i] in buffer to priorities[i].
        /// </remarks>
        /// <param name="rgIdx">Specifies the list of indexed sampled transitions.</param>
        /// <param name="rgfPriorities">Specifies the list of updated priorities corresponding to transitions at the sampled indexes donated by variable 'rgIdx'.</param>
        public void UpdatePriorities(int[] rgIdx, float[] rgfPriorities)
        {
            if (rgIdx.Length != rgfPriorities.Length)
                throw new Exception("The index and priority arrays must have the same length.");

            for (int i = 0; i < rgIdx.Length; i++)
            {
                int nIdx = rgIdx[i];
                float fPriority = rgfPriorities[i];

                if (fPriority <= 0)
                    throw new Exception("The priority at index '" + i.ToString() + "' is zero!");

                if (nIdx < 0 || nIdx >= m_rgItems.Length)
                    throw new Exception("The index at index '" + i.ToString() + "' is out of range!");

                float fNewPriority = (float)Math.Pow(fPriority, m_fAlpha);
                m_ItSum[nIdx] = fNewPriority;
                m_ItMin[nIdx] = fNewPriority;
                m_fMaxPriority = Math.Max(m_fMaxPriority, fPriority);
            }
        }
    } 
}
