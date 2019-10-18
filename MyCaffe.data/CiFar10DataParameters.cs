using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.data
{
    /// <summary>
    /// Contains the dataset parameters used to create the CIFAR-10 dataset.
    /// </summary>
    public class CiFar10DataParameters
    {
        string m_strDataBatchFile1;
        string m_strDataBatchFile2;
        string m_strDataBatchFile3;
        string m_strDataBatchFile4;
        string m_strDataBatchFile5;
        string m_strTestBatchFile;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strDataBatchFile1">Specifies the first training dataset file 'data_batch_1.bin'.</param>
        /// <param name="strDataBatchFile2">Specifies the second training dataset file 'data_batch_2.bin'.</param>
        /// <param name="strDataBatchFile3">Specifies the third training dataset file 'data_batch_3.bin'.</param>
        /// <param name="strDataBatchFile4">Specifies the fourth training dataset file 'data_batch_4.bin'.</param>
        /// <param name="strDataBatchFile5">Specifies the fifth training dataset file 'data_batch_5.bin'</param>
        /// <param name="strTestBatchFile">Specifies the testing dataset file 'test_batch.bin'.</param>
        public CiFar10DataParameters(string strDataBatchFile1, string strDataBatchFile2, string strDataBatchFile3, string strDataBatchFile4, string strDataBatchFile5, string strTestBatchFile)
        {
            m_strDataBatchFile1 = strDataBatchFile1;
            m_strDataBatchFile2 = strDataBatchFile2;
            m_strDataBatchFile3 = strDataBatchFile3;
            m_strDataBatchFile4 = strDataBatchFile4;
            m_strDataBatchFile5 = strDataBatchFile5;
            m_strTestBatchFile = strTestBatchFile;
        }

        /// <summary>
        /// Specifies the first training dataset file 'data_batch_1.bin'
        /// </summary>
        public string DataBatchFile1
        {
            get { return m_strDataBatchFile1; }
        }

        /// <summary>
        /// Specifies the second training dataset file 'data_batch_2.bin'
        /// </summary>
        public string DataBatchFile2
        {
            get { return m_strDataBatchFile2; }
        }

        /// <summary>
        /// Specifies the third training dataset file 'data_batch_3.bin'
        /// </summary>
        public string DataBatchFile3
        {
            get { return m_strDataBatchFile3; }
        }

        /// <summary>
        /// Specifies the fourth training dataset file 'data_batch_4.bin'
        /// </summary>
        public string DataBatchFile4
        {
            get { return m_strDataBatchFile4; }
        }

        /// <summary>
        /// Specifies the fifth training dataset file 'data_batch_5.bin'
        /// </summary>
        public string DataBatchFile5
        {
            get { return m_strDataBatchFile5; }
        }

        /// <summary>
        /// Specifies the testing dataset file 'test_batch.bin'
        /// </summary>
        public string TestBatchFile
        {
            get { return m_strTestBatchFile; }
        }
    }
}
