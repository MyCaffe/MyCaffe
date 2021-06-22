using System;
using System.Collections.Generic;
using System.IO;
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
        /// <param name="rgResults">Specifies the list of (int nLabel, double dfResult) result data.</param>
        /// <param name="bInvert">Specifies whether or not to invert the value by subtracting it from the maximum value within the result pairs.</param>
        /// <returns>A <i>byte</i> array containing the result data is returned.</returns>
        public static byte[] CreateResults(List<Result> rgResults, bool bInvert)
        {
            List<byte> rgData = new List<byte>();
            double dfMax = double.MinValue;
            double dfMin = double.MaxValue;

            if (bInvert)
            {
                foreach (Result kv in rgResults)
                {
                    if (dfMax < kv.Score)
                        dfMax = kv.Score;

                    if (dfMin > kv.Score)
                        dfMin = kv.Score;
                }
            }

            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                bw.Write(rgResults.Count);

                foreach (Result kv in rgResults)
                {
                    bw.Write(kv.Label);

                    double dfValue = kv.Score;

                    if (bInvert)
                        dfValue = dfMax - dfValue;

                    bw.Write(dfValue);
                }

                ms.Flush();
                return ms.ToArray();
            }
        }

        /// <summary>
        /// Extract the results from the binary data.
        /// </summary>
        /// <param name="rgData">Specifies the binary data holding the results.</param>
        /// <returns>The list of results is returned.</returns>
        public static List<Result> GetResults(byte[] rgData)
        {
            List<Result> rgRes = new List<Result>();

            using (MemoryStream ms = new MemoryStream(rgData))
            using (BinaryReader br = new BinaryReader(ms))
            {
                return GetResults(br);
            }
        }

        /// <summary>
        /// Extract the results from the binary data.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <returns>The list of results is returned.</returns>
        public static List<Result> GetResults(BinaryReader br)
        {
            List<Result> rgRes = new List<Result>();

            int nCount = br.ReadInt32();

            for (int i = 0; i < nCount; i++)
            {
                int nLabel = br.ReadInt32();
                double dfVal = br.ReadDouble();

                Result r = new Result(nLabel, dfVal);
                rgRes.Add(r);
            }

            return rgRes;
        }

        /// <summary>
        /// The CreateResults function converts the batch of lists of (int nLabel, double dfResult) pairs into a array of <i>bytes</i>.
        /// </summary>
        /// <param name="rgrgResults">Specifies the batch of lists of (int nLabel, double dfResult) result data.</param>
        /// <returns>A <i>byte</i> array containing the result data is returned.</returns>
        public static byte[] CreateResults(List<Tuple<SimpleDatum, List<Result>>> rgrgResults)
        {
            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                bw.Write(rgrgResults.Count);

                for (int i = 0; i < rgrgResults.Count; i++)
                {
                    bw.Write(rgrgResults[i].Item1.ImageID);
                    bw.Write(rgrgResults[i].Item1.Index);
                    bw.Write(rgrgResults[i].Item1.TimeStamp.ToFileTimeUtc());
                    bw.Write(CreateResults(rgrgResults[i].Item2, false));
                }

                ms.Flush();
                return ms.ToArray();
            }
        }

        /// <summary>
        /// Extracts the raw image result batch from the result binary data.
        /// </summary>
        /// <param name="nBatchCount">Specifies the number of results in the batch.</param>
        /// <param name="rgData">Specifies the binary batch data.</param>
        /// <returns>An array of tuples containing SimpleDatum/Result pairs is returned.</returns>
        public static List<Tuple<SimpleDatum, List<Result>>> GetResults(int nBatchCount, byte[] rgData)
        {
            if (nBatchCount <= 0)
                throw new Exception("The batch count must be >= 1!");

            List<Tuple<SimpleDatum, List<Result>>> rgRes1 = new List<Tuple<SimpleDatum, List<Result>>>();

            using (MemoryStream ms = new MemoryStream(rgData))
            using (BinaryReader br = new BinaryReader(ms))
            {
                int nCount = br.ReadInt32();
                if (nCount != nBatchCount)
                    throw new Exception("The batch count does not match the expected count of " + nCount.ToString());

                for (int i = 0; i < nCount; i++)
                {
                    int nImageID = br.ReadInt32();
                    int nIdx = br.ReadInt32();
                    long lTime = br.ReadInt64();
                    DateTime dt = DateTime.FromFileTimeUtc(lTime);
                    List<Result> rgRes = GetResults(br);

                    SimpleDatum sd = new SimpleDatum();
                    sd.SetImageID(nImageID);
                    sd.Index = nIdx;
                    sd.TimeStamp = dt;

                    rgRes1.Add(new Tuple<SimpleDatum, List<Result>>(sd, rgRes));
                }
            }

            return rgRes1;
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
