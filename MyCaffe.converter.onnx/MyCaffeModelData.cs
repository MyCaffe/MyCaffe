using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.converter.onnx
{
    /// <summary>
    /// The MyCaffeModelData object contains the model descriptor, model weights and optionally the image mean.
    /// </summary>
    public class MyCaffeModelData
    {
        string m_strModelDescription;
        byte[] m_rgWeights;
        byte[] m_rgImageMean;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strModelDesc">Specifies the model descriptor.</param>
        /// <param name="rgWeights">Specifies the model weights.</param>
        /// <param name="rgImageMean">Optionally, specifies the image mean (default = null).</param>
        public MyCaffeModelData(string strModelDesc, byte[] rgWeights, byte[] rgImageMean = null)
        {
            m_strModelDescription = strModelDesc;
            m_rgWeights = rgWeights;
            m_rgImageMean = rgImageMean;
        }

        /// <summary>
        /// Returns the model descriptor.
        /// </summary>
        public string ModelDescription
        {
            get { return m_strModelDescription; }
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
    }
}
