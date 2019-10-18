using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.data
{
    /// <summary>
    /// Contains the dataset parameters used to create the VOC0712 dataset.
    /// </summary>
    public class VOCDataParameters
    {
        string m_strDataFileTrain2007;
        string m_strDataFileTrain2012;
        string m_strDataFileTest2007;
        bool m_bExtractFiles = true;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strDataBatchFileTrain2007">Specifies the training file 'VOCtrain_06-Nov-2007.tar'.</param>
        /// <param name="strDataBatchFileTrain2012">Specifies the training file 'VOCtrain_11-May-2012.tar'.</param>
        /// <param name="strDataBatchFileTest2007">"Specifies the testing file 'VOCtest_06-Nov-2007.tar'.</param>
        /// <param name="bExtractFiles">Specifies whether or not to extract the tar files (when false, the files must have already been extracted at least once).</param>
        public VOCDataParameters(string strDataBatchFileTrain2007, string strDataBatchFileTrain2012, string strDataBatchFileTest2007, bool bExtractFiles)
        {
            m_strDataFileTrain2007 = strDataBatchFileTrain2007;
            m_strDataFileTrain2012 = strDataBatchFileTrain2012;
            m_strDataFileTest2007 = strDataBatchFileTest2007;
            m_bExtractFiles = bExtractFiles;
        }

        /// <summary>
        /// Returns the training file 'VOCtrain_06-Nov-2007.tar'.
        /// </summary>
        public string DataBatchFileTrain2007
        {
            get { return m_strDataFileTrain2007; }
        }

        /// <summary>
        /// Specifies the training file 'OCtrain_11-May-2012.tar'.
        /// </summary>
        public string DataBatchFileTrain2012
        {
            get { return m_strDataFileTrain2012; }
        }

        /// <summary>
        /// Returns the testing file 'VOCtest_06-Nov-2007.tar'.
        /// </summary>
        public string DataBatchFileTest2007
        {
            get { return m_strDataFileTest2007; }
        }

        /// <summary>
        /// Returns whether or not to extract the tar files.
        /// </summary>
        public bool ExtractFiles
        {
            get { return m_bExtractFiles; }
        }
    }
}
