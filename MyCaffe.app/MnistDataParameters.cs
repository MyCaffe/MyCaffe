using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.app
{
    public class MnistDataParameters
    {
        string m_strTrainImagesFile;
        string m_strTrainLabelsFile;
        string m_strTestImagesFile;
        string m_strTestLabelsFile;

        public MnistDataParameters(string strTrainImages, string strTrainLabels, string strTestImages, string strTestLabels)
        {
            m_strTrainImagesFile = strTrainImages;
            m_strTrainLabelsFile = strTrainLabels;
            m_strTestImagesFile = strTestImages;
            m_strTestLabelsFile = strTestLabels;
        }

        public string TrainImagesFile
        {
            get { return m_strTrainImagesFile; }
        }

        public string TrainLabelsFile
        {
            get { return m_strTrainLabelsFile; }
        }

        public string TestImagesFile
        {
            get { return m_strTestImagesFile; }
        }

        public string TestLabelsFile
        {
            get { return m_strTestLabelsFile; }
        }
    }
}
