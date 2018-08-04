using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MyCaffe.basecode.SimpleDatum;

namespace MyCaffe.trainers
{
    /// <summary>
    /// The Batch class manages a batch of observation collections, where each obervation collection constitutes a sequence of actions making up an experience.
    /// </summary>
    public class Batch
    {
        List<ObservationCollection> m_rgBatch = new List<ObservationCollection>();
        int m_nObsColIdx = 0;
        int m_nObsIdx = 0;
        int m_nMax;
        Random m_random = new Random();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nMax">Maximum number of items allowed in the batch (older items are removed).</param>
        public Batch(int nMax = int.MaxValue)
        {
            m_nMax = nMax;
        }

        /// <summary>
        /// Retrieves the training data which is made up of all Datum's from each item within each experience.  The input
        /// data is placed in the regular Datum location and the set of actions taken are placed in a multi-valued label
        /// which is stored in each Datum's DataCriteria byte array.
        /// </summary>
        /// <returns>A list of the training data is returned.</returns>
        public List<Datum> GetTrainingData()
        {
            List<Datum> rgDatum = new List<Datum>();

            Reset();

            Observation obs = GetNextObservation();

            while (obs != null)
            {
                SimpleDatum sd = new SimpleDatum(obs.Data);
                DATA_FORMAT fmt;

                List<float> rgActions = new List<float>();
                List<KeyValuePair<int, double>> rgKv = obs.Actions.ResultsOriginal.OrderBy(p => p.Key).ToList();

                foreach (KeyValuePair<int, double> kv in rgKv)
                {
                    rgActions.Add((float)kv.Value);
                }

                sd.DataCriteria = BinaryData.Pack(rgActions, out fmt);
                sd.DataCriteriaFormat = fmt;

                rgDatum.Add(new Datum(sd));

                obs = GetNextObservation();
            }

            return rgDatum;
        }

        /// <summary>
        /// Add a new collection of observations making up an experience.
        /// </summary>
        /// <param name="col">Specifies the collection of observations.</param>
        public void Add(ObservationCollection col)
        {
            if (col.TotalValidCount > 0)
                m_rgBatch.Add(col);
        }

        /// <summary>
        /// Add another batch to this one.
        /// </summary>
        /// <param name="batch">Specifies the batch to add.</param>
        public void Add(Batch batch)
        {
            if (m_rgBatch.Count > 0 && batch.Count + m_rgBatch.Count > m_nMax)
            {
                m_rgBatch = m_rgBatch.OrderByDescending(p => p.TotalReward).ToList();

                // Remove the top 25% from the first half at random
                int nFirstQuarter = (int)((m_rgBatch.Count - batch.Count) * 0.25);
                int nCount = (int)Math.Ceiling(batch.Count * 0.5);

                for (int i = 0; i < nCount; i++)
                {
                    int nIdx = m_random.Next(nFirstQuarter);
                    m_rgBatch.RemoveAt(nIdx);
                }

                // Remove the bottom worst until we reach our count.
                while (m_rgBatch.Count > 0 && batch.Count + m_rgBatch.Count > m_nMax)
                {
                    m_rgBatch.RemoveAt(m_rgBatch.Count - 1);
                }
            }

            m_rgBatch.AddRange(batch.m_rgBatch);
        }

        /// <summary>
        /// Returns the number of items within the batch.
        /// </summary>
        public int Count
        {
            get { return m_rgBatch.Count; }
        }

        /// <summary>
        /// Orders the batch by best reward to worst, and removes items only keeping the top percernt of items.
        /// </summary>
        /// <param name="dfTopPercent">Specifies the top percent to keep.  For a value of 0.33 will keep the top 33% of the items with the best result.</param>
        /// <param name="nBatchSize">Specifies the size of the batch.</param>
        public void TrimToBest(double dfTopPercent, int nBatchSize)
        {
            m_rgBatch = m_rgBatch.OrderByDescending(p => p.TotalReward).ToList();
            int nTopCount = (int)(m_rgBatch.Count * dfTopPercent);
            int nSetSize = nBatchSize / m_rgBatch[0].Count;
            int nTopCountAligned = (int)(Math.Ceiling((double)nTopCount / (double)nSetSize)) * nSetSize;

            for (int i=m_rgBatch.Count-1; i>=nTopCountAligned; i--)
            {
                m_rgBatch.RemoveAt(i);
            }
        }

        /// <summary>
        /// Retrieves the total obervation count across all batches.
        /// </summary>
        /// <returns></returns>
        public int GetObservationCount()
        {
            int nCount = 0;

            foreach (ObservationCollection col in m_rgBatch)
            {
                nCount += col.Count;
            }

            return nCount;
        }

        /// <summary>
        /// Resets the iterators to the first item within the first batch.
        /// </summary>
        public void Reset()
        {
            m_nObsColIdx = 0;
            m_nObsIdx = 0;
        }

        /// <summary>
        /// Gets the next observation within the batch.
        /// </summary>
        /// <returns>The next observation is returned.</returns>
        public Observation GetNextObservation()
        {
            if (m_nObsColIdx == m_rgBatch.Count)
                return null;

            Observation obs = m_rgBatch[m_nObsColIdx][m_nObsIdx];

            m_nObsIdx++;

            if (m_nObsIdx == m_rgBatch[m_nObsColIdx].Count)
            {
                m_nObsIdx = 0;
                m_nObsColIdx++;
            }

            return obs;
        }
    }
}
