using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    /// <summary>
    /// The DataItem manages one synchronized data item where the first element is the sync field.
    /// </summary>
    public class DataItem
    {
        double[] m_rgdfData;
        int m_nFilled;
        int m_nFull;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nFieldCount">Specifies the total number of fields to be collected.</param>
        public DataItem(int nFieldCount)
        {
            m_rgdfData = new double[nFieldCount];
            m_nFilled = 0;
            m_nFull = ((int)Math.Pow(2, nFieldCount)) - 1;
        }

        /// <summary>
        /// Adds a new set of raw data to the synchronized data.
        /// </summary>
        /// <param name="nFieldIdx">Specifies the field index where the data is to be added.</param>
        /// <param name="nItemIdx">Specifies the item index of the data.</param>
        /// <param name="rg">Specifies the raw data.</param>
        /// <param name="nFieldCount">Specifies the local number of fields contained in the 'rg' parameter.</param>
        /// <returns>The next field index is returned.</returns>
        public int Add(int nFieldIdx, int nItemIdx, double[] rg, int nFieldCount)
        {
            int nStart = (nFieldIdx == 0) ? 0 : 1;

            for (int j = nStart; j < nFieldCount; j++)
            {
                int nIdx = (nItemIdx * nFieldCount) + j;
                Add(nFieldIdx, rg[nIdx]);
                nFieldIdx++;
            }

            return nFieldIdx;
        }

        /// <summary>
        /// Add a new data item at a specified field index.
        /// </summary>
        /// <param name="nFieldIdx">Specifies the field index.</param>
        /// <param name="df">Specifies the raw data.</param>
        /// <returns>When the data item fields are full, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Add(int nFieldIdx, double df)
        {
            m_rgdfData[nFieldIdx] = df;
            m_nFilled |= (0x0001 << nFieldIdx);

            if (m_nFilled == m_nFull)
                return true;

            return false;
        }

        /// <summary>
        /// Returns the synchronized data fields.
        /// </summary>
        /// <returns></returns>
        public double[] GetData()
        {
            return m_rgdfData;
        }

        /// <summary>
        /// Clears the data fields and the filled status.
        /// </summary>
        public void Reset()
        {
            Array.Clear(m_rgdfData, 0, m_rgdfData.Length);
            m_nFilled = 0;
        }
    }
}
