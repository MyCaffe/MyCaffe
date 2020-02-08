using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.data
{
    /// <summary>
    /// Contains the dataset parameters used to create the MNIST dataset.
    /// </summary>
    public class MnistDataParameters
    {
        string m_strTrainImagesFile;
        string m_strTrainLabelsFile;
        string m_strTestImagesFile;
        string m_strTestLabelsFile;
        bool m_bExportToFileOnly;
        string m_strExportPath;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strTrainImages">Specifies the training image file 'train-images-idx3-ubyte.gz'.</param>
        /// <param name="strTrainLabels">Specifies the training label file 'train-labels-idx1.ubyte.gz'.</param>
        /// <param name="strTestImages">Specifies the testing image file 't10k-images-idx3-ubyte.gz'.</param>
        /// <param name="strTestLabels">Specifies the testing label file 't10k-labels-idx1-ubyte.gz'.</param>
        /// <param name="bExportToFileOnly">Specifies to extract and export the *.gz files to all images files contained within each *.gz.</param>
        /// <param name="strExportPath">Specifies the path where the images are to be exported.</param>
        public MnistDataParameters(string strTrainImages, string strTrainLabels, string strTestImages, string strTestLabels, bool bExportToFileOnly, string strExportPath)
        {
            m_strTrainImagesFile = strTrainImages;
            m_strTrainLabelsFile = strTrainLabels;
            m_strTestImagesFile = strTestImages;
            m_strTestLabelsFile = strTestLabels;
            m_bExportToFileOnly = bExportToFileOnly;
            m_strExportPath = strExportPath;
        }

        /// <summary>
        /// Specifies the training image file 'train-images-idx3-ubyte.gz'.
        /// </summary>
        public string TrainImagesFile
        {
            get { return m_strTrainImagesFile; }
        }

        /// <summary>
        /// Specifies the training label file 'train-labels-idx1.ubyte.gz'.
        /// </summary>
        public string TrainLabelsFile
        {
            get { return m_strTrainLabelsFile; }
        }

        /// <summary>
        /// Specifies the testing image file 't10k-images-idx3-ubyte.gz'.
        /// </summary>
        public string TestImagesFile
        {
            get { return m_strTestImagesFile; }
        }

        /// <summary>
        /// Specifies the testing label file 't10k-labels-idx1-ubyte.gz'.
        /// </summary>
        public string TestLabelsFile
        {
            get { return m_strTestLabelsFile; }
        }

        /// <summary>
        /// Specifies where to export the files when 'ExportToFile' = <i>true</i>.
        /// </summary>
        public string ExportPath
        {
            get { return m_strExportPath; }
        }

        /// <summary>
        /// Specifies to export the images to a folder.
        /// </summary>
        public bool ExportToFile
        {
            get { return m_bExportToFileOnly; }
        }
    }
}
