using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The SimpleResult class holds the result data stored in the RawImageResults table.
    /// </summary>
    [Serializable]
    public class SimpleResult
    {
        int m_nSourceID;
        int m_nIdx;
        DateTime m_dt;
        int m_nBatchCount;
        int m_nResultCount;
        float[] m_rgResult;
        int[] m_rgTarget;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nSrcID">Specifies the source ID of the data source associated with the results.</param>
        /// <param name="nIdx">Specifies the image index corresponding to the result.</param>
        /// <param name="dt">Specifies the time-stamp of the result.</param>
        /// <param name="nBatchCount">Specifies the number of result data sets.</param>
        /// <param name="nResultCount">Specifies the number of results in the packed rgResult data.</param>
        /// <param name="rgResult">Specifies the results.</param>
        /// <param name="rgTarget">Specifies the target.</param>
        public SimpleResult(int nSrcID, int nIdx, DateTime dt, int nBatchCount, int nResultCount, float[] rgResult, int[] rgTarget)
        {
            m_nSourceID = nSrcID;
            m_nIdx = nIdx;
            m_dt = dt;
            m_nBatchCount = nBatchCount;
            m_nResultCount = nResultCount;
            m_rgResult = rgResult;
            m_rgTarget = rgTarget;
        }

        /// <summary>
        /// Returns the source ID of the data source associated with the result.
        /// </summary>
        public int SourceID
        {
            get { return m_nSourceID; }
        }

        /// <summary>
        /// Returns the image index associated with the result.
        /// </summary>
        public int Index
        {
            get { return m_nIdx; }
        }

        /// <summary>
        /// Returns the time-stamp of the result.
        /// </summary>
        public DateTime TimeStamp
        {
            get { return m_dt; }
        }

        /// <summary>
        /// Returns the number of results in the result data sets.
        /// </summary>
        public int BatchCount
        {
            get { return m_nBatchCount; }
        }

        /// <summary>
        /// Returns the number of results in the result array.
        /// </summary>
        public int ResultCount
        {
            get { return m_nResultCount; }
        }

        /// <summary>
        /// Returns the results.
        /// </summary>
        public float[] Result
        {
            get { return m_rgResult; }
        }

        /// <summary>
        /// Returns the Target.
        /// </summary>
        public int[] Target
        {
            get { return m_rgTarget; }
        }
    }
}
