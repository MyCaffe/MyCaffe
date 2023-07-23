using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The MyCaffeModelData object contains the model descriptor, model weights and optionally the image mean.
    /// </summary>
    public class MyCaffeModelData
    {
        string m_strModelDescription;
        string m_strSolverDescription = null;
        byte[] m_rgWeights;
        byte[] m_rgImageMean;
        string m_strLastModelFileName = "";
        string m_strLastSolverFileName = "";
        string m_strLastImageMeanFileName = "";
        string m_strLastWeightsFileName = "";
        string m_strOriginalDownloadFile = null;
        List<int> m_rgInputShape = new List<int>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strModelDesc">Specifies the model descriptor.</param>
        /// <param name="rgWeights">Specifies the model weights.</param>
        /// <param name="rgImageMean">Optionally, specifies the image mean (default = null).</param>
        /// <param name="strSolverDescription">Optionally, specifies the solver description.</param>
        /// <param name="rgInputShape">Specifies the input data shape.</param>
        public MyCaffeModelData(string strModelDesc, byte[] rgWeights, byte[] rgImageMean = null, string strSolverDescription = null, List<int> rgInputShape = null)
        {
            m_strModelDescription = strModelDesc;
            m_rgWeights = rgWeights;
            m_rgImageMean = rgImageMean;
            m_strSolverDescription = strSolverDescription;

            if (rgInputShape != null)
                m_rgInputShape = rgInputShape;
        }

        /// <summary>
        /// Get/set the data input shape.
        /// </summary>
        public List<int> InputShape
        {
            get { return m_rgInputShape; }
            set { m_rgInputShape = value; }
        }

        /// <summary>
        /// Get/set the model descriptor.
        /// </summary>
        public string ModelDescription
        {
            get { return m_strModelDescription; }
            set { m_strModelDescription = value; }
        }

        /// <summary>
        /// Get/set the solver description.
        /// </summary>
        public string SolverDescription
        {
            get { return m_strSolverDescription; }
            set { m_strSolverDescription = value; }
        }

        /// <summary>
        /// Returns the model weights.
        /// </summary>
        public byte[] Weights
        {
            get { return m_rgWeights; }
        }

        /// <summary>
        /// Returns the image mean if one was specified, or null.
        /// </summary>
        public byte[] ImageMean
        {
            get { return m_rgImageMean; }
        }

        /// <summary>
        /// Specifies the original download file, if any.
        /// </summary>
        public string OriginalDownloadFile
        {
            get { return m_strOriginalDownloadFile; }
            set { m_strOriginalDownloadFile = value; }
        }

        /// <summary>
        /// Save the model data to the specified folder under the specified name.
        /// </summary>
        /// <param name="strFolder">Specifies the folder where the data is to be saved.</param>
        /// <param name="strName">Specifies the base name of the files.</param>
        public void Save(string strFolder, string strName)
        {
            string strModel = strFolder.TrimEnd('\\') + "\\" + strName + "_model_desc.prototxt";

            if (File.Exists(strModel))
                File.Delete(strModel);

            m_strLastModelFileName = strModel;
            using (StreamWriter sr = new StreamWriter(strModel))
            {
                sr.WriteLine(m_strModelDescription);
            }

            if (!string.IsNullOrEmpty(m_strSolverDescription))
            {
                string strSolver = strFolder.TrimEnd('\\') + "\\" + strName + "_solver_desc.prototxt";

                if (File.Exists(strSolver))
                    File.Delete(strSolver);

                m_strLastSolverFileName = strSolver;
                using (StreamWriter sr = new StreamWriter(strSolver))
                {
                    sr.WriteLine(strSolver);
                }
            }

            if (m_rgWeights != null)
            {
                string strWts = strFolder.TrimEnd('\\') + "\\" + strName + "_weights.mycaffemodel";

                if (File.Exists(strWts))
                    File.Delete(strWts);

                m_strLastWeightsFileName = strWts;
                using (FileStream fs = new FileStream(strWts, FileMode.CreateNew, FileAccess.Write))
                using (BinaryWriter bw = new BinaryWriter(fs))
                {
                    bw.Write(m_rgWeights);
                }
            }
            else
            {
                m_strLastWeightsFileName = "";
            }

            if (m_rgImageMean != null)
            {
                string strWts = strFolder.TrimEnd('\\') + "\\" + strName + "_image_mean.bin";

                if (File.Exists(strWts))
                    File.Delete(strWts);

                m_strLastImageMeanFileName = strWts;
                using (FileStream fs = new FileStream(strWts, FileMode.CreateNew, FileAccess.Write))
                using (BinaryWriter bw = new BinaryWriter(fs))
                {
                    bw.Write(m_rgImageMean);
                }
            }
            else
            {
                m_strLastImageMeanFileName = "";
            }
        }

        /// <summary>
        /// Returns the file name of the last saved model description file.
        /// </summary>
        public string LastSavedModeDescriptionFileName
        {
            get { return m_strLastModelFileName; }
        }

        /// <summary>
        /// Returns the file name of the last saved solver description file.
        /// </summary>
        public string LastSaveSolverDescriptionFileName
        {
            get { return m_strLastSolverFileName; }
        }

        /// <summary>
        /// Returns the file name of the last saved weights file.
        /// </summary>
        public string LastSavedWeightsFileName
        {
            get { return m_strLastWeightsFileName; }
        }

        /// <summary>
        /// Returns the file name of the last saved image mean file.
        /// </summary>
        public string LastSavedImageMeanFileName
        {
            get { return m_strLastImageMeanFileName; }
        }
    }
}
