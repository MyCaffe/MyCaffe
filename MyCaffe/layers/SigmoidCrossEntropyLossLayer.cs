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
    /// The SigmoidCrossEntropyLayer computes the corss-entropy (logisitc) loss and is
    /// often used for predicting targets interpreted as probabilities.
    /// </summary>
    /// <remarks>
    /// Computation: @f$
    ///     E = \frac{-1}{n} \sum\limits_{n=1}^N \left[
    ///             p_n \log \hat{p}_n +
    ///             (1 - p_n) \log(1 - \hat{p}_n)
    ///         \right]
    ///     @f$
    /// <br/>
    /// This layer is implemented rather than separate 
    /// SigmoidLayer + CrossEntropyLayer as its gradient
    /// computation is more numerically stable.
    /// <br/>
    /// At test time, this layer can be replaced simply by a SigmoidLayer.
    /// 
    /// @see [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/abs/1612.02295) by Weiyang Liu, Yandong Wen, Zhiding Yu, and Meng Yang, 2016.
    /// @see [Information Dropout: Learning Optimal Representations Through Noisy Computation](https://arxiv.org/abs/1611.01353) by Alessandro Achille and Stefano Soatto, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SigmoidCrossEntropyLossLayer<T> : LossLayer<T>
    {
        SigmoidLayer<T> m_sigmoidLayer;
        Blob<T> m_blobSigmoidOutput;
        BlobCollection<T> m_colSigmoidBottomVec = new BlobCollection<T>();
        BlobCollection<T> m_colSigmoidTopVec = new BlobCollection<T>();

        // The label indicating that an instance should be ignored.
        int? m_nIgnoreLabel = null;
        // How to normalize the loss.
        LossParameter.NormalizationMode m_normalization;
        double m_dfNormalizer = 0;
        int m_nOuterNum;
        int m_nInnerNum;


        /// <summary>
        /// The SigmoidCrossEntropyLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type SIGMOIDCROSSENTROPY_LOSS.
        /// </param>
        public SigmoidCrossEntropyLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SIGMOIDCROSSENTROPY_LOSS;
            m_blobSigmoidOutput = new Blob<T>(cuda, log);
            m_sigmoidLayer = new SigmoidLayer<T>(cuda, log, p);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobSigmoidOutput.Dispose();
            m_sigmoidLayer.Dispose();
            base.dispose();
        }

        /// <summary>
        /// Read the normalization mode parameter and compute the normalizer based
        /// on the Blob size.  If normalizeation_mode is VALID, the count of valid
        /// outputs will be read from valid_count, unless it is -1 in which case
        /// all outputs are assumed to be valid.
        /// </summary>
        /// <param name="normalization_mode">Specifies the normalization mode.</param>
        /// <param name="valid_count">Specifies the valid count.</param>
        /// <returns>The normalizer value is returned.</returns>
        protected virtual double get_normalizer(LossParameter.NormalizationMode normalization_mode, int valid_count)
        {
            double dfNormalizer = 0;

            switch (normalization_mode)
            {
                case LossParameter.NormalizationMode.FULL:
                    dfNormalizer = m_nOuterNum * m_nInnerNum;
                    break;

                case LossParameter.NormalizationMode.VALID:
                    if (valid_count == -1)
                        dfNormalizer = m_nOuterNum * m_nInnerNum;
                    else
                        dfNormalizer = valid_count;
                    break;

                case LossParameter.NormalizationMode.BATCH_SIZE:
                    dfNormalizer = m_nOuterNum;
                    break;

                case LossParameter.NormalizationMode.NONE:
                    dfNormalizer = 1;
                    break;

                default:
                    m_log.FAIL("Unknown normalization mode: " + normalization_mode.ToString());
                    break;
            }

            // Some users will have no labels for some examples in order to 'turn off' a
            // particular loss in a multi-task setup.  The max prevents NaNs in that case.
            return Math.Max(1.0, dfNormalizer);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            m_colSigmoidBottomVec = new BlobCollection<T>();
            m_colSigmoidBottomVec.Add(colBottom[0]);
            m_colSigmoidTopVec = new BlobCollection<T>();
            m_colSigmoidTopVec.Add(m_blobSigmoidOutput);
            m_sigmoidLayer.Setup(m_colSigmoidBottomVec, m_colSigmoidTopVec);

            m_nIgnoreLabel = m_param.loss_param.ignore_label;

            if (m_param.loss_param.normalization != LossParameter.NormalizationMode.NONE)
                m_normalization = m_param.loss_param.normalization;
            else
                m_normalization = (m_param.loss_param.normalize) ? LossParameter.NormalizationMode.VALID : LossParameter.NormalizationMode.BATCH_SIZE;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);
            m_nOuterNum = colBottom[0].shape(0); // batch size
            m_nInnerNum = colBottom[0].count(1); // instance size: |output| == |target|
            m_log.CHECK_EQ(colBottom[0].count(), colBottom[1].count(), "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.");
            m_sigmoidLayer.Reshape(m_colSigmoidBottomVec, m_colSigmoidTopVec);
        }

        /// <summary>
        /// Forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the scores @f$ x \in [-\infty, +\infty] @f$,
        ///     which this layer maps to probability predictions @f$
        ///     \hat{p}_n = \sigma(x_n) \in [0,1]
        ///     @f$
        ///     using the sigmoid function @f$ \sigma(.) @f$ (see SigmoidLayer).
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the targets @f$ y \in [0,1] @f$.
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed cross-entropy loss: @f$
        ///       E = \frac{-1}{n} \sum\limits_{n=1}^N \left[
        ///               p_n \log \hat{p}_n + (1 - p_n) \log(1 - \hat{p}_n)
        ///           \right]
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // The forward pass computes the sigmoid outputs.
            m_colSigmoidBottomVec[0] = colBottom[0];
            m_sigmoidLayer.Forward(m_colSigmoidBottomVec, m_colSigmoidTopVec);

            // Compute the loss (negative log likelihood)
            int nCount = colBottom[0].count();

            // Stable version of loss computation for input data.
            long hInputData = colBottom[0].gpu_data;
            long hTarget = colBottom[1].gpu_data;

            // Since this memory is not used for anything, we use it here to avoid having
            // to allocate the GPU memory to accumulate intermediate results.
            long hLossData = colBottom[0].mutable_gpu_diff;
            long hCountData = colBottom[1].mutable_gpu_diff;

            m_cuda.sigmoid_cross_entropy_fwd(nCount, hInputData, hTarget, hLossData, m_nIgnoreLabel.HasValue, m_nIgnoreLabel.GetValueOrDefault(-1), hCountData);

            double dfValidCount = nCount;
            // Only launch another CUDA kernel if we actually need the valid count.
            if (m_normalization == LossParameter.NormalizationMode.VALID && m_nIgnoreLabel.HasValue)
                dfValidCount = m_cuda.asum_double(nCount, hCountData);

            double dfLoss = m_cuda.asum_double(nCount, hLossData);
            m_dfNormalizer = get_normalizer(m_normalization, (int)dfValidCount);

            colTop[0].SetData(dfLoss / m_dfNormalizer, 0);

            // Clear scratch memory to prevent interfering with the backward pass (see #6202)
            colBottom[0].SetDiff(0);
            colBottom[1].SetDiff(0);
        }

        /// <summary>
        /// Computes the sigmoid cross-entropy loss error gradient w.r.t. the 
        /// predictions.
        /// </summary>
        /// <remarks>
        /// Gradients cannot be computed with respect to the target inputs (bottom[1]),
        /// so this method ignores bottom[1] and requires propagate_down[1] == false, 
        /// crashing otherwise.
        /// </remarks>
        /// <param name="colTop">top output blob (length 1), providing the error gradient with
        /// respect to the otuputs
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     This blob's diff will simply contain the loss_weight * @f$ \lambda @f$,
        ///     as @f$ \lambda @f$ is the coefficient of this layer's output
        ///     @f$ \ell_i @f$ in the overall Net loss @f$
        ///       E = \lambda_i \ell_i + \mbox{other loss terms} @f$; hence @f$
        ///       \frac{\partial E}{\partial \ell_i} = \lambda_i.
        ///       @f$
        ///       (*Assuming that this top blob is not used as a bottom (input) by any
        ///       other layer of the Net.)</param>
        /// <param name="rgbPropagateDown">see Layer::Backward.  propagate_down[1] must be false
        /// as gradient computation with respect to the targets is not implemented.
        /// </param>
        /// <param name="colBottom">input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$; Backward computes diff @f$
        ///       \frac{\partial E}{\partial x} = 
        ///         \frac{1}{n} \sum\limits_{n=1}^N (\hat{p}_n - p_n)
        ///     @f$
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels -- ignored as we can't compute their error gradients.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[1])
                m_log.FAIL(m_type.ToString() + " Layer cannot backpropagate to label inputs.");

            if (rgbPropagateDown[0])
            {
                // First, compute the diff.
                int nCount = colBottom[0].count();
                long hSigmoidOutputData = m_blobSigmoidOutput.gpu_data;
                long hTarget = colBottom[1].gpu_data;
                long hBottomDiff = colBottom[0].mutable_gpu_diff;

                m_cuda.copy(nCount, hSigmoidOutputData, hBottomDiff);
                m_cuda.axpy(nCount, convert(-1.0), hTarget, hBottomDiff);

                // Zero out gradient for ignored targets
                if (m_nIgnoreLabel.HasValue)
                    m_cuda.sigmoid_cross_entropy_ignore(nCount, m_nIgnoreLabel.Value, hTarget, hBottomDiff);

                // Scale down gradient
                double dfLossWeight = convertD(colTop[0].GetDiff(0)) / m_dfNormalizer;
                m_cuda.scal(nCount, dfLossWeight, hBottomDiff);
            }
        }
    }
}
