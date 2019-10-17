using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.app
{
    public class VOCDataParameters
    {
        string m_strDataFile1;
        string m_strDataFile2;
        string m_strDataFile3;
        bool m_bExtractFiles = true;

        public VOCDataParameters(string strDataBatchFile1, string strDataBatchFile2, string strDataBatchFile3, bool bExtractFiles)
        {
            m_strDataFile1 = strDataBatchFile1;
            m_strDataFile2 = strDataBatchFile2;
            m_strDataFile3 = strDataBatchFile3;
            m_bExtractFiles = bExtractFiles;
        }

        public string DataBatchFile1
        {
            get { return m_strDataFile1; }
        }

        public string DataBatchFile2
        {
            get { return m_strDataFile2; }
        }

        public string DataBatchFile3
        {
            get { return m_strDataFile3; }
        }

        public bool ExtractFiles
        {
            get { return m_bExtractFiles; }
        }
    }
}
