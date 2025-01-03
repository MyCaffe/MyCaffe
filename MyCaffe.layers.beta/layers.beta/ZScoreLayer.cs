﻿using System;
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
    public class ZScoreLayer<T> : NeuronLayer<T>, IXNormalize<T>
    {
        SCORE_AS_LABEL_NORMALIZATION m_method = SCORE_AS_LABEL_NORMALIZATION.Z_SCORE;
        float m_fMeanPos;
        float m_fStdevPos;
        float m_fMeanNeg;
        float m_fStdevNeg;

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
            m_method = p.z_score_param.score_method;

            SourceDescriptor src = db.GetSourceByName(p.z_score_param.source);
            if (src == null)
                throw new Exception("Could not find the data source '" + p.z_score_param.source + "!");

            SimpleDatum sd = db.GetItemMean(src.ID, p.z_score_param.mean_pos_param, p.z_score_param.stdev_pos_param, p.z_score_param.mean_neg_param, p.z_score_param.stdev_neg_param);
            if (sd == null)
                throw new Exception("Could not find the item mean for the data source '" + p.z_score_param.source + "!  Make sure the item mean is created for this dataset.");

            float? fVal = sd.GetParameter(p.z_score_param.mean_pos_param);
            if (!fVal.HasValue)
                throw new Exception("Layer: '" + layer_param.name + "' - Could not find the positive mean parameter '" + p.z_score_param.mean_pos_param + "'!  The image mean has the following parameters: " + sd.GetParameterNames() + " Make sure the item mean is created for this dataset.");

            m_fMeanPos = fVal.Value;

            fVal = sd.GetParameter(p.z_score_param.stdev_pos_param);
            if (!fVal.HasValue)
                throw new Exception("Layer: '" + layer_param.name + "' - Could not find the positive stdev parameter '" + p.z_score_param.stdev_pos_param + "'!  The image mean has the following parameters: " + sd.GetParameterNames() + " Make sure the item stdev is created for this dataset.");

            m_fStdevPos = fVal.Value;

            fVal = sd.GetParameter(p.z_score_param.mean_neg_param);
            if (!fVal.HasValue)
                throw new Exception("Layer: '" + layer_param.name + "' - Could not find the negative mean parameter '" + p.z_score_param.mean_neg_param + "'!  The image mean has the following parameters: " + sd.GetParameterNames() + " Make sure the item mean is created for this dataset.");

            m_fMeanNeg = fVal.Value;

            fVal = sd.GetParameter(p.z_score_param.stdev_neg_param);
            if (!fVal.HasValue)
                throw new Exception("Layer: '" + layer_param.name + "' - Could not find the negative stdev parameter '" + p.z_score_param.stdev_neg_param + "'!  The image mean has the following parameters: " + sd.GetParameterNames() + " Make sure the item stdev is created for this dataset.");

            m_fStdevNeg = fVal.Value;
        }

        /// <summary>
        /// Normalize a bucket collection configuration string.
        /// </summary>
        /// <param name="strConfig">Specifies the configuration string.</param>
        /// <returns>The list of normalized tuples is returned.</returns>
        public List<Tuple<double, double>> Normalize(string strConfig)
        {
            List<Tuple<double, double>> rgVal = BucketCollection.Parse(strConfig);
            List<Tuple<double, double>> rgNorm = new List<Tuple<double, double>>();

            foreach (Tuple<double, double> tuple in rgVal)
            {
                double fMin = tuple.Item1;
                double fMax = tuple.Item2;

                double fMinNorm = (fMin == 0) ? 0 : Normalize((float)fMin);
                double fMaxNorm = (fMax == 0) ? 0 : Normalize((float)fMax);

                rgNorm.Add(new Tuple<double, double>(fMinNorm, fMaxNorm));
            }

            return rgNorm;
        }

        /// <summary>
        /// Normalize the un-normalized value.
        /// </summary>
        /// <param name="val">Specifies the un-normalized value.</param>
        /// <returns>The normalized value is returned.</returns>
        public T Normalize(T val)
        {
            float fVal = Utility.ConvertValF<T>(val);
            float fNormVal = Normalize(fVal);
            return Utility.ConvertVal<T>(fNormVal);
        }

        /// <summary>
        /// Unnormalize the normalized value.
        /// </summary>
        /// <param name="val">Specifies normalized value.</param>
        /// <returns>The un-normalized value is returned.</returns>
        public T Unnormalize(T val)
        {
            float fVal = Utility.ConvertValF<T>(val);
            float fUnnormVal = UnNormalize(fVal);
            return Utility.ConvertVal<T>(fUnnormVal);
        }

        /// <summary>
        /// Z-score normalize the input value.
        /// </summary>
        /// <param name="fVal">Specifies the un-normalized value.</param>
        /// <returns>The normalized value is returned.</returns>
        public float Normalize(float fVal)
        {
            float fMean = m_fMeanPos;
            float fStdev = m_fStdevPos;

            if (m_method == SCORE_AS_LABEL_NORMALIZATION.Z_SCORE_POSNEG)
            {
                if (fVal < 0)
                {
                    fMean = m_fMeanNeg;
                    fStdev = m_fStdevNeg;
                }
            }

            return (fVal - fMean) / fStdev;
        }

        /// <summary>
        /// UnNormalize a normalized input value.
        /// </summary>
        /// <param name="fVal">Specifies the normalized value.</param>
        /// <returns>The un-normalized value is returned.</returns>
        public float UnNormalize(float fVal)
        {
            float fMean = m_fMeanPos;
            float fStdev = m_fStdevPos;

            if (m_method == SCORE_AS_LABEL_NORMALIZATION.Z_SCORE_POSNEG)
            {
                if (fVal < 0)
                {
                    fMean = m_fMeanNeg;
                    fStdev = m_fStdevNeg;
                }
            }

            return (fVal * fStdev) + fMean;
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
                m_cuda.z_score(colTop[0].count(), colTop[0].gpu_data, m_fMeanPos, m_fStdevPos, m_fMeanNeg, m_fStdevNeg, colTop[0].mutable_gpu_data, DIR.FWD, m_method);
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
                colBottom[0].CopyFrom(colTop[0], true, false);

            if (m_param.z_score_param.enabled)
                m_cuda.z_score(colBottom[0].count(), colBottom[0].mutable_gpu_diff, m_fMeanPos, m_fStdevPos, m_fMeanNeg, m_fStdevNeg, colBottom[0].gpu_diff, DIR.BWD, m_method);
        }
    }
}
