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
    /// The SoftmaxCrossEntropyLayer computes the cross-entropy (logisitic) loss and is
    /// often used for predicting targets interpreted as probabilities in reinforcement learning.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SoftmaxCrossEntropyLossLayer<T> : LossLayer<T>
    {
        SoftmaxLayer<T> m_softmaxLayer;
        Blob<T> m_blobSoftmaxOutput;
        Blob<T> m_blobLoss;
        BlobCollection<T> m_colSoftmaxBottomVec = new BlobCollection<T>();
        BlobCollection<T> m_colSoftmaxTopVec = new BlobCollection<T>();

        // How to normalize the loss.
        double m_dfNormalizer = 0;


        /// <summary>
        /// The SoftmaxCrossEntropyLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type SOFTMAXCROSSENTROPY_LOSS.
        /// </param>
        public SoftmaxCrossEntropyLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SOFTMAXCROSSENTROPY_LOSS;
            m_blobSoftmaxOutput = new Blob<T>(cuda, log);
            m_blobSoftmaxOutput.Name = m_param.name + " softmax out";
            m_blobLoss = new Blob<T>(cuda, log);
            m_blobLoss.Name = m_param.name + " loss";

            LayerParameter param_softmax = p.Clone(false);
            param_softmax.loss_weight.Clear();
            m_softmaxLayer = new SoftmaxLayer<T>(cuda, log, param_softmax);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobSoftmaxOutput.Dispose();
            m_softmaxLayer.Dispose();
            m_blobLoss.Dispose();
            base.dispose();
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs as variable.
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: loss.
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of required top (output) Blobs: loss, loss values
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            m_colSoftmaxBottomVec = new BlobCollection<T>();
            m_colSoftmaxBottomVec.Add(colBottom[0]);
            m_colSoftmaxTopVec = new BlobCollection<T>();
            m_colSoftmaxTopVec.Add(m_blobSoftmaxOutput);
            m_softmaxLayer.Setup(m_colSoftmaxBottomVec, m_colSoftmaxTopVec);
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
            m_log.CHECK_EQ(colBottom[0].count(), colBottom[1].count(), "SOFTMAX_CROSS_ENTROPY_LOSS layer inputs must have the same count.");
            m_softmaxLayer.Reshape(m_colSoftmaxBottomVec, m_colSoftmaxTopVec);
            m_blobLoss.ReshapeLike(colBottom[0]);
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
        ///     using the softmax function @f$ \sigma(.) @f$ (see SoftmaxLayer).
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
            // The forward pass computes the softmax outputs.
            m_colSoftmaxBottomVec[0] = colBottom[0];
            m_softmaxLayer.Forward(m_colSoftmaxBottomVec, m_colSoftmaxTopVec);

            // Compute the loss (negative log likelihood)
            int nCount = colBottom[0].count();

            // Stable version of loss computation for input data.
            long hInputData = colBottom[0].gpu_data;
            long hTarget = colBottom[1].gpu_data;

            // Since this memory is not used for anything, we use it here to avoid having
            // to allocate the GPU memory to accumulate intermediate results.
            long hLossData = colBottom[0].mutable_gpu_diff;
            long hCountData = colBottom[1].mutable_gpu_diff;

            m_cuda.cross_entropy_fwd(nCount, hInputData, hTarget, hLossData, false, -1, hCountData);

            double dfValidCount = nCount;
            double dfLoss = m_cuda.asum_double(nCount, hLossData);
            m_dfNormalizer = get_normalizer(m_normalization, (int)dfValidCount);

            colTop[0].SetData(dfLoss / m_dfNormalizer, 0);

            // Return the losses in colTop[1] if it exists.
            if (colTop.Count == 2)
            {
                m_cuda.copy(nCount, hLossData, m_blobLoss.mutable_gpu_data);
                colTop[1].ShareData(m_blobLoss);
            }

            // Clear scratch memory to prevent interfering with the backward pass (see #6202)
            colBottom[0].SetDiff(0);
            colBottom[1].SetDiff(0);
        }

        /// <summary>
        /// Computes the softmax cross-entropy loss error gradient w.r.t. the 
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
                long hSoftmaxOutputData = m_blobSoftmaxOutput.gpu_data;
                long hTarget = colBottom[1].gpu_data;
                long hBottomDiff = colBottom[0].mutable_gpu_diff;

                m_cuda.copy(nCount, hSoftmaxOutputData, hBottomDiff);
                m_cuda.axpy(nCount, convert(-1.0), hTarget, hBottomDiff);

                // Scale down gradient
                double dfLossWeight = convertD(colTop[0].GetDiff(0)) / m_dfNormalizer;
                m_cuda.scal(nCount, dfLossWeight, hBottomDiff);
            }
        }
    }
}
