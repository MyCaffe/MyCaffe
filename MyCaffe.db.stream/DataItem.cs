using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    public class DataItem
    {
        double[] m_rgdfData;
        int m_nFilled;
        int m_nFull;

        public DataItem(int nFieldCount)
        {
            m_rgdfData = new double[nFieldCount];
            m_nFilled = 0;
            m_nFull = ((int)Math.Pow(2, nFieldCount)) - 1;
        }

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

        public bool Add(int nIdx, double df)
        {
            m_rgdfData[nIdx] = df;
            m_nFilled |= (0x0001 << nIdx);

            if (m_nFilled == m_nFull)
                return true;

            return false;
        }

        public double[] GetData()
        {
            return m_rgdfData;
        }

        public void Reset()
        {
            Array.Clear(m_rgdfData, 0, m_rgdfData.Length);
            m_nFilled = 0;
        }
    }
}
