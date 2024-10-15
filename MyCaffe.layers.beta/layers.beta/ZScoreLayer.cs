using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The ZScoreLayer normalizes the inputs using the Z-Score method, by subtracting the mean and dividing by the standard deviation.
    /// </summary>
    /// <remarks>
    /// When using the ZScoreLayer, the layer must be configured with the following parameters:
    ///  - source: Specifies the data source to use for the mean and standard deviation values.
    ///  - mean_param: Specifies the name of the RawImage parameter that has the mean value.
    ///  - stdev_param: Specifies the name of the RawImage parameter that has the stdev value.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ZScoreLayer<T> : NeuronLayer<T>
    {
        float m_fMean;
        float m_fStdev;

        /// <summary>
        /// The MishLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Mish with parameter Mish_param
        /// </param>
        public ZScoreLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXDatabaseBase db)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.Z_SCORE;

            SourceDescriptor src = db.GetSourceByName(p.z_score_param.source);
            if (src == null)
                throw new Exception("Could not find the data source '" + p.z_score_param.source + "!");

            SimpleDatum sd = db.GetItemMean(src.ID, p.z_score_param.mean_param, p.z_score_param.stdev_param);
            if (sd == null)
                throw new Exception("Could not find the item mean for the data source '" + p.z_score_param.source + "!  Make sure the item mean is created for this dataset.");

            float? fVal = sd.GetParameter(p.z_score_param.mean_param);
            if (!fVal.HasValue)
                throw new Exception("Layer: '" + layer_param.name + "' - Could not find the mean parameter '" + p.z_score_param.mean_param + "'!  The image mean has the following parameters: " + sd.GetParameterNames() + " Make sure the item mean is created for this dataset.");

            m_fMean = fVal.Value;

            fVal = sd.GetParameter(p.z_score_param.stdev_param);
            if (!fVal.HasValue)
                throw new Exception("Layer: '" + layer_param.name + "' - Could not find the stdev parameter '" + p.z_score_param.stdev_param + "'!  The image mean has the following parameters: " + sd.GetParameterNames() + " Make sure the item stdev is created for this dataset.");

            m_fStdev = fVal.Value;
        }

        /// <summary>
        /// Z-score normalize the input value.
        /// </summary>
        /// <param name="fVal">Specifies the un-normalized value.</param>
        /// <returns>The normalized value is returned.</returns>
        public float Normalize(float fVal)
        {
            return (fVal - m_fMean) / m_fStdev;
        }

        /// <summary>
        /// UnNormalize a normalized input value.
        /// </summary>
        /// <param name="fVal">Specifies the normalized value.</param>
        /// <returns>The un-normalized value is returned.</returns>
        public float UnNormalize(float fVal)
        {
            return (fVal * m_fStdev) + m_fMean;
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs @f$ 
        ///         y = (x - \mu) / \sigma
        ///     @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (colTop[0] != colBottom[0])
                colTop[0].CopyFrom(colBottom[0], false, true);

            if (m_param.z_score_param.enabled)
            {
                m_cuda.add_scalar(colTop[0].count(), -m_fMean, colTop[0].mutable_gpu_data);
                m_cuda.mul_scalar(colTop[0].count(), 1.0f / m_fStdev, colTop[0].mutable_gpu_data);
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the Mish value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$; Backward fills their diff with 
        ///     gradients @f$
        ///     \frac{\partial E}{\partial x}
        ///         = \frac{\partial E}{\partial y} \cdot \frac{1}{m\_fStdev}
        ///     @$f
        ///     since y = (x - m_fMean) / m_fStdev.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (colTop[0] != colBottom[0])
                colBottom[0].CopyFrom(colTop[0], true, true);

            if (m_param.z_score_param.enabled)
            {
                m_cuda.mul_scalar(colBottom[0].count(), m_fStdev, colBottom[0].mutable_gpu_diff);
            }
        }
    }
}
