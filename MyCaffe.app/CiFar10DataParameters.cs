using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.app
{
    public class CiFar10DataParameters
    {
        string m_strDataBatchFile1;
        string m_strDataBatchFile2;
        string m_strDataBatchFile3;
        string m_strDataBatchFile4;
        string m_strDataBatchFile5;
        string m_strTestBatchFile;

        public CiFar10DataParameters(string strDataBatchFile1, string strDataBatchFile2, string strDataBatchFile3, string strDataBatchFile4, string strDataBatchFile5, string strTestBatchFile)
        {
            m_strDataBatchFile1 = strDataBatchFile1;
            m_strDataBatchFile2 = strDataBatchFile2;
            m_strDataBatchFile3 = strDataBatchFile3;
            m_strDataBatchFile4 = strDataBatchFile4;
            m_strDataBatchFile5 = strDataBatchFile5;
            m_strTestBatchFile = strTestBatchFile;
        }

        public string DataBatchFile1
        {
            get { return m_strDataBatchFile1; }
        }

        public string DataBatchFile2
        {
            get { return m_strDataBatchFile2; }
        }

        public string DataBatchFile3
        {
            get { return m_strDataBatchFile3; }
        }

        public string DataBatchFile4
        {
            get { return m_strDataBatchFile4; }
        }

        public string DataBatchFile5
        {
            get { return m_strDataBatchFile5; }
        }

        public string TestBatchFile
        {
            get { return m_strTestBatchFile; }
        }
    }
}
