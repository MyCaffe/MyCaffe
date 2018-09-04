using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The LossLayer provides an interface for Layer's that take two blobs as input -- usually
    /// (1) predictions and (2) ground-truth labels -- and output a 
    /// singleton blob representing the loss.
    /// This layer is initialized with the MyCaffe.param.LossParameter.
    /// </summary>
    /// <remarks>
    /// LossLayers are typically only capable of backpropagating to their first input
    /// -- the predictions.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public abstract class LossLayer<T> : Layer<T>
    {
        /// <summary>
        /// Specifies the minimum threshold for loss values.
        /// </summary>
        public const double kLOG_THRESHOLD = 1e-20;
        /// <summary>
        /// Set to <i>true</i> when labels are to be ignored.
        /// </summary>
        protected bool m_bIgnoreLabels = false;
        /// <summary>
        /// Specifies the normalization mode used to normalize the loss.
        /// </summary>
        protected LossParameter.NormalizationMode m_normalization = LossParameter.NormalizationMode.NONE;
        /// <summary>
        /// Specifies the outer num, such as the batch count (e.g. count(0, axis)).  Each derivative class must set this value appropriately.
        /// </summary>
        protected int m_nOuterNum = 0;
        /// <summary>
        /// Specifies the inner num, such as the channel + height + width (e.g. count(axis + 1)).  Each derivative class must set this value appropriately.
        /// </summary>
        protected int m_nInnerNum = 0;

        /// <summary>
        /// The LossLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type LOSS with parameter loss_param,
        /// with options:
        ///     - ignore_label (\b optional, default null). When specified, instances with the label are ignored.
        ///     
        ///     - normalization. The normalization mode to use: FULL, VALID, or BATCH_SIZE. 
        /// </param>
        public LossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.LOSS;
        }

        /// <summary>
        /// Returns the normalizer used to normalize the loss.
        /// </summary>
        /// <param name="normalization_mode">Specifies the normalization mode to use.</param>
        /// <param name="nValidCount">Specifies the number of valid.</param>
        /// <returns>The normalization value is returned.</returns>
        protected virtual double get_normalizer(LossParameter.NormalizationMode normalization_mode, int nValidCount)
        {
            double dfNormalizer = 0.0;

            switch (normalization_mode)
            {
                case LossParameter.NormalizationMode.FULL:
                    m_log.CHECK_GT(m_nInnerNum, 0, "The inner number must be set.");
                    m_log.CHECK_GT(m_nOuterNum, 0, "The outer number must be set.");
                    dfNormalizer = m_nOuterNum * m_nInnerNum;
                    break;

                case LossParameter.NormalizationMode.VALID:
                    if (nValidCount == -1)
                    {
                        m_log.CHECK_GT(m_nInnerNum, 0, "The inner number must be set.");
                        m_log.CHECK_GT(m_nOuterNum, 0, "The outer number must be set.");
                        dfNormalizer = m_nOuterNum * m_nInnerNum;
                    }
                    else
                        dfNormalizer = nValidCount;
                    break;

                case LossParameter.NormalizationMode.BATCH_SIZE:
                    m_log.CHECK_GT(m_nOuterNum, 0, "The outer number must be set.");
                    dfNormalizer = m_nOuterNum;
                    break;

                case LossParameter.NormalizationMode.NONE:
                    dfNormalizer = 1.0;
                    break;

                default:
                    m_log.FAIL("Unknown normalization mode " + normalization_mode.ToString());
                    break;
            }

            // Some users will have no labels for some examples in order to 'turn off' a 
            // particular loss in a multi-taks setup.  The max prevents Nans in that case.
            return Math.Max(dfNormalizer, 1.0);
        }

        /// <summary>
        /// Returns the exact number of required bottom (intput) Blobs: prediction, label
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: loss
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// For convenience and backwards compatibility, insturct the Net to 
        /// automatically allocate a single top Blob for LossLayers, into which
        /// they output their singleton loss, (even if the user didn't specify
        /// one in the prototxt, etc.).
        /// </summary>
        public override bool AutoTopBlobs
        {
            get { return true; }
        }

        /// <summary>
        /// We usually cannot backpropagate to the labels; ignore force_backward 
        /// for these inputs.
        /// </summary>
        /// <param name="nBottomIdx"></param>
        /// <returns></returns>
        public override bool AllowForceBackward(int nBottomIdx)
        {
            if (nBottomIdx != 1)
                return true;

            return false;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // LossLayers have non-zero (1) loss by default.
            if (m_param.loss_weight.Count == 0)
                m_param.loss_weight.Add(1.0);

            m_log.CHECK(!m_param.loss_param.normalize, "normalize is drepreciated, use 'normalization'.");
            if (!m_param.loss_param.normalization.HasValue)
                m_normalization = (m_param.loss_param.normalize) ? LossParameter.NormalizationMode.VALID : LossParameter.NormalizationMode.BATCH_SIZE;
            else
                m_normalization = m_param.loss_param.normalization.Value;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(colBottom[0].shape(0), colBottom[1].shape(0), "The data and label should have the same first dimension.  Data has shape '" + colBottom[0].shape_string + "' and label has shape '" + colBottom[1].shape_string + "'.");
            List<int> rgLossShape = new List<int>(); // Loss layers output a scalar, 0 axes.
            colTop[0].Reshape(rgLossShape);
            colTop[0].type = Blob<T>.BLOB_TYPE.LOSS;
        }
    }
}
