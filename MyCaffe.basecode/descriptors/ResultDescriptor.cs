using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The ResultDescriptor class describes the results of a run.
    /// </summary>
    [Serializable]
    public class ResultDescriptor : BaseDescriptor
    {
        int m_nIdx;
        int m_nLabel;
        int m_nSourceID;
        DateTime m_dt;
        List<KeyValuePair<int, double>> m_rgResults = new List<KeyValuePair<int, double>>();

        /// <summary>
        /// The ResultDescriptor constructor.
        /// </summary>
        /// <param name="nID">Specifies the database ID of the item.</param>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="strOwner">Specifies the identifier of the item's owner.</param>
        /// <param name="nIdx">Specifies the index of the results.</param>
        /// <param name="nLabel">Specifies the expected label of the result.</param>
        /// <param name="nResultCount">Specifies the number of items (classes) participating in the results.</param>
        /// <param name="rgResults">Specifies the raw result data that is converted into the full list of (int nLabel, double dfResult) pairs from the run.</param>
        /// <param name="nSrcId">Specifiesthe data source ID.</param>
        /// <param name="dt">Specifies the Timestamp of the result.</param>
        public ResultDescriptor(int nID, string strName, string strOwner, int nIdx, int nLabel, int nResultCount, byte[] rgResults, int nSrcId, DateTime dt)
            : base(nID, strName, strOwner)
        {
            m_nIdx = nIdx;
            m_nLabel = nLabel;
            m_nSourceID = nSrcId;
            m_dt = dt;
            m_rgResults = createResults(nResultCount, rgResults);
        }

        /// <summary>
        /// Returns the index of the results.
        /// </summary>
        public int Index
        {
            get { return m_nIdx; }
        }

        /// <summary>
        /// Returns the expected label of the result.
        /// </summary>
        public int Label
        {
            get { return m_nLabel; }
        }

        /// <summary>
        /// Returns the number of items (classes) participating in the results.
        /// </summary>
        public int ResultCount
        {
            get { return m_rgResults.Count; }
        }

        /// <summary>
        /// Returns the raw result data that is converted into the full list of (int nLabel, double dfResult) pairs from the run.
        /// </summary>
        public List<KeyValuePair<int, double>> Results
        {
            get { return m_rgResults; }
        }

        /// <summary>
        /// Returns the data source ID.
        /// </summary>
        public int SourceID
        {
            get { return m_nSourceID; }
        }

        /// <summary>
        /// Returns the time-stamp of the result.
        /// </summary>
        public DateTime TimeStamp
        {
            get { return m_dt; }
        }

        /// <summary>
        /// The CreateResults function converts the list of (int nLabel, double dfResult) pairs into a array of <i>bytes</i>.
        /// </summary>
        /// <param name="rgResults">Specifies the list of (int nLabel, double dfResult) result pairs.</param>
        /// <param name="bInvert">Specifies whether or not to invert the value by subtracting it from the maximum value within the result pairs.</param>
        /// <returns>A <i>byte</i> array containing the result data is returned.</returns>
        public static byte[] CreateResults(List<KeyValuePair<int, double>> rgResults, bool bInvert)
        {
            List<byte> rgData = new List<byte>();
            double dfMax = double.MinValue;
            double dfMin = double.MaxValue;

            if (bInvert)
            {
                foreach (KeyValuePair<int, double> kv in rgResults)
                {
                    if (dfMax < kv.Value)
                        dfMax = kv.Value;

                    if (dfMin > kv.Value)
                        dfMin = kv.Value;
                }
            }

            foreach (KeyValuePair<int, double> kv in rgResults)
            {
                rgData.AddRange(BitConverter.GetBytes(kv.Key));
                double dfValue = kv.Value;

                if (bInvert)
                    dfValue = dfMax - dfValue;

                rgData.AddRange(BitConverter.GetBytes(dfValue));
            }

            return rgData.ToArray();
        }

        private List<KeyValuePair<int, double>> createResults(int nCount, byte[] rgData)
        {
            List<KeyValuePair<int, double>> rgResults = new List<KeyValuePair<int, double>>();
            int nIdx = 0;

            for (int i = 0; i < nCount; i++)
            {
                int nVal = BitConverter.ToInt32(rgData, nIdx);
                nIdx += sizeof(int);
                double dfVal = BitConverter.ToDouble(rgData, nIdx);
                nIdx += sizeof(double);

                rgResults.Add(new KeyValuePair<int, double>(nVal, dfVal));
            }

            return rgResults;
        }

        /// <summary>
        /// Creates the string representation of the descriptor.
        /// </summary>
        /// <returns>The string representation of the descriptor is returned.</returns>
        public override string ToString()
        {
            return m_dt.ToString() + " ~ " + m_nLabel.ToString();
        }
    }
}
