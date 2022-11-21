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
    /// The SoftmaxCrossEntropy2Layer computes the cross-entropy (logisitic) loss and is
    /// often used for predicting targets interpreted as probabilities.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SoftmaxCrossEntropy2LossLayer<T> : LossLayer<T>
    {
        Layer<T> m_softmaxLayer;
        Layer<T> m_logLayer;
        Blob<T> m_blobProb;
        Blob<T> m_blobLogProb;
        Blob<T> m_blobLoss;
        BlobCollection<T> m_colSoftmaxBottomVec = new BlobCollection<T>();
        BlobCollection<T> m_colSoftmaxTopVec = new BlobCollection<T>();
        BlobCollection<T> m_colLogBottomVec = new BlobCollection<T>();
        BlobCollection<T> m_colLogTopVec = new BlobCollection<T>();

        // How to normalize the loss.
        int? m_nIgnoreLabel = null;
        double m_dfNormalizer = 0;
        int m_nSoftmaxAxis = 1;

        /// <summary>
        /// The SoftmaxCrossEntropyLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type SOFTMAXCROSSENTROPY2_LOSS.
        /// </param>
        public SoftmaxCrossEntropy2LossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SOFTMAXCROSSENTROPY2_LOSS;
            m_blobProb = new Blob<T>(cuda, log);
            m_blobProb.Name = m_param.name + " prob";
            m_blobLogProb = new Blob<T>(cuda, log);
            m_blobLogProb.Name = m_param.name + " logprob";
            m_blobLoss = new Blob<T>(cuda, log);
            m_blobLoss.Name = m_param.name + " loss";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobProb.Dispose();

            if (m_softmaxLayer != null)
                m_softmaxLayer.Dispose();
            
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();
                col.Add(m_blobProb);
                return col;
            }
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

            m_nIgnoreLabel = m_param.loss_param.ignore_label;

            LayerParameter param_softmax = m_param.Clone(false);
            param_softmax.SetType(LayerParameter.LayerType.SOFTMAX);
            param_softmax.softmax_param = m_param.softmax_param.Clone() as SoftmaxParameter;
            param_softmax.loss_weight.Clear();

            m_softmaxLayer = Layer<T>.Create(m_cuda, m_log, param_softmax, null);
            m_colSoftmaxBottomVec = new BlobCollection<T>() { colBottom[0] };
            m_colSoftmaxTopVec = new BlobCollection<T>() { m_blobProb };

            m_softmaxLayer.Setup(m_colSoftmaxBottomVec, m_colSoftmaxTopVec);

            LayerParameter param_log = new LayerParameter(LayerParameter.LayerType.LOG);

            m_logLayer = Layer<T>.Create(m_cuda, m_log, param_log, null);
            m_colLogBottomVec = new BlobCollection<T>() { m_blobProb };
            m_colLogTopVec = new BlobCollection<T>() { m_blobLogProb };

            m_logLayer.Setup(m_colLogBottomVec, m_colLogTopVec);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_blobLoss.ReshapeLike(colBottom[0]);            

            m_softmaxLayer.Reshape(m_colSoftmaxBottomVec, m_colSoftmaxTopVec);
            m_logLayer.Reshape(m_colLogBottomVec, m_colLogTopVec);
            
            m_nSoftmaxAxis = colBottom[0].CanonicalAxisIndex(m_param.softmax_param.axis);
            m_nOuterNum = colBottom[0].count(0, m_nSoftmaxAxis);
            m_nInnerNum = colBottom[0].count(m_nSoftmaxAxis + 1);

            if (!m_bIgnoreLabels)
            {
                m_log.CHECK_EQ(colBottom[0].count(0, m_nSoftmaxAxis), colBottom[1].count(0, m_nSoftmaxAxis), "Number of labels must match number of predictions; e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), label count (number of labels) must be N*H*W, with integer values in {0, 1, ..., C-1}.");

                if (colTop.Count >= 2)
                {
                    // softmax output
                    colTop[1].ReshapeLike(colBottom[0]);
                }
            }
        }

        /// <summary>
        /// The forward computation for softmax cross entropy loss.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$, a blob with values in
        ///     @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
        ///     the K = CHW classes.  This layer maps these scores to a
        ///     probability distribution over classes using the softmax function @f$
        ///     \hat{p}_{nk} = \exp(x_{nk}) /
        ///     \left[\sum_{k'} \exp(x_{nk'})\right] @f$ (see SoftmaxLayer).
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels l, an integer valued blob with values @f$ l_n \in [0, 1, 2, ..., K-1] @f$
        ///     indicating the correct class label among the K classes.</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///     the computed cross_entropy classification loss: @f$ E = 
        ///     \frac{-1}{N} \sum\limits_{n=1}^N \log(\hat{p}_{n,l_n})
        ///     @f$ for softmax output class probabilities @f$ \hat{p} @f$.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // The forward pass computes the sotmax outputs (which are probabilities).
            m_softmaxLayer.Forward(m_colSoftmaxBottomVec, m_colSoftmaxTopVec);

            // Run the log on the Probabilities to get LogSoftmax
            m_logLayer.Forward(m_colLogBottomVec, m_colLogTopVec);

            // Use the softmax output for input data.
            long hProbData = m_blobLogProb.gpu_data;
            long hTarget = colBottom[1].gpu_data;
            int nInputCount = m_blobProb.count();
            int nDim = m_blobProb.shape()[m_nSoftmaxAxis];
            int nCount = m_nOuterNum * m_nInnerNum;

            m_blobLoss.SetDiff(0.0);
            long hLossData = m_blobLoss.mutable_gpu_data;
            long hLossDiff = m_blobLoss.mutable_gpu_diff;

            // Since this memory is not used for anything, we use it here to avoid having
            // to allocate the GPU memory to accumulate intermediate results.
            colBottom[1].SetDiff(0);
            long hCountData = colBottom[1].mutable_gpu_diff;

            // Run the NLL Loss portion to get the loss.
            m_cuda.softmax_cross_entropy_fwd(colBottom[0].count(), hProbData, hTarget, hLossDiff, hLossData, m_nOuterNum, nDim, m_nInnerNum, hCountData, m_nIgnoreLabel.GetValueOrDefault(-1));
            double dfLoss = m_cuda.asum_double(colBottom[0].count(), hLossData);

            double dfValidCount = nCount;
            // Only launch another CUDA kernel if we actually need the valid count.
            if (m_normalization == LossParameter.NormalizationMode.VALID && m_nIgnoreLabel.HasValue)
                dfValidCount = m_cuda.asum_double(nCount, hCountData);

            m_dfNormalizer = get_normalizer(m_normalization, (int)dfValidCount);
            double dfFinalLoss = dfLoss / m_dfNormalizer;

            colTop[0].SetData(dfFinalLoss, 0);

            // Return the losses in colTop[1] if it exists.
            if (colTop.Count == 2)
                colTop[1].CopyFrom(m_blobLoss);

            // Clear scratch memory to prevent interfering with the backward pass (see #6202)
            colBottom[1].SetDiff(0);
        }

        /// <summary>
        /// Computes the softmax cross entropy loss error gradient w.r.t the predictions.
        /// </summary>
        /// <remarks>
        /// Gradients cannot be computed with respect to the label inputs (bottom[1]),
        /// so this method ignores bottom[1] and requires !propagate_down[1], crashing
        /// if propagate_down[1] == true.
        /// </remarks>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///   -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///      This blob's diff will simply contain the loss_weight * @f$ \lambda @f$ as
        ///      @f$ \lambda @f$ is the coefficient of this layer's output
        ///      @f$ \ell_i @f$ in the overall Net loss.
        ///      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}; hence
        ///        \frac{partial E}{\partial \ell_i} = \lambda_i
        ///      @f$
        ///        (*Assuming that this top blob is not used as a bottom (input) by any
        ///        other layer of the Net.)
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.  propagate_down[1] must be false as
        /// we can't compute gradients with respect to the labels.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$; backward computes diff @f$
        ///       \frac{\partial E}{\partial x}
        ///     @f$
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels -- ignored as we can't compute their error gradients.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            // Calculate the NLL Loss Gradient
            float fGrad = convertF(colTop[0].GetDiff(0));
            fGrad = -1.0f * fGrad / (float)m_dfNormalizer;
            
            m_blobLoss.scale_diff(fGrad);

            // Calculate the log gradient.
            m_blobLogProb.CopyFrom(m_blobLoss, true);
            m_logLayer.Backward(m_colLogTopVec, rgbPropagateDown, m_colLogBottomVec);

            // Calculate the Softmax gradient.
            m_softmaxLayer.Backward(m_colSoftmaxTopVec, rgbPropagateDown, m_colSoftmaxBottomVec);
        }
    }
}
