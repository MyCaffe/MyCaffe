using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.app
{
    public class ProgressInfo
    {
        int m_nIdx;
        int m_nTotal;
        string m_strMsg;
        Exception m_err;
        bool? m_bAlive = null;

        public ProgressInfo(int nIdx, int nTotal, string str, Exception err = null, bool? bAlive = null)
        {
            m_nIdx = nIdx;
            m_nTotal = nTotal;
            m_strMsg = str;
            m_err = err;
            m_bAlive = bAlive;
        }

        public double Percentage
        {
            get { return (m_nTotal == 0) ? 0 : (double)m_nIdx / (double)m_nTotal; }
        }

        public string Message
        {
            get { return m_strMsg; }
        }

        public Exception Error
        {
            get { return m_err; }
        }

        public bool? Alive
        {
            get { return m_bAlive; }
            set { m_bAlive = value; }
        }
    }
}
