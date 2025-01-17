﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// Computes the multinomial logistic loss for a one-of-many
    /// classification task, passing real-valued predictions through a
    /// softmax to get a probability distribution over classes.
    /// </summary>
    /// <remarks>
    /// This layer should be preferred over separate
    /// SofmaxLayer + MultinomialLogisticLossLayer
    /// as its gradient computation is more numerically stable.
    /// At test time, this layer can be replaced simply by a SofmaxLayer.
    /// 
    /// @see [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/abs/1612.02295) by Weiyang Liu, Yandong Wen, Zhiding Yu and Meng Yang, 2016. 
    /// @see [Generalisation and Sharing in Triplet Convnets for Sketch based Visual Search](https://arxiv.org/abs/1611.05301v1) by Tu Bui, Leonardo Ribeiro, Moacir Ponti, and John Collomosse, 2016.
    /// </remarks>
    /// <typeparam name="T"></typeparam>
    public class SoftmaxLossLayer<T> : LossLayer<T>
    {
        Layer<T> m_softmaxLayer;
        Blob<T> m_blobProb;
        BlobCollection<T> m_colSoftmaxBottom;
        BlobCollection<T> m_colSoftmaxTop;
        int? m_nIgnoreLabel = null;
        int m_nSoftmaxAxis;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides LossParameter loss_param, with options:
        ///  - ignore_label (optional)
        ///    Specify a label value that whould be ignored when computing the loss.
        ///  - normalize (optional, default true)
        ///    If true, the loss is normalized by the number of (nonignored) labels
        ///    present; otherwise the loss is imply summed over spatial locations.
        /// </param>
        public SoftmaxLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SOFTMAXWITH_LOSS;
            m_blobProb = new Blob<T>(cuda, log);
            m_blobProb.Name = m_param.name + " prob";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobProb.Dispose();

            if (m_softmaxLayer != null)
                m_softmaxLayer.Dispose();
            
            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobProb);
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
        /// Returns the maximum number of required top (output) Blobs: loss, labels
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

            LayerParameter param_softmax = m_param.Clone(false);
            param_softmax.SetType(LayerParameter.LayerType.SOFTMAX);
            param_softmax.softmax_param = m_param.softmax_param.Clone() as SoftmaxParameter;
            param_softmax.loss_weight.Clear();

            m_softmaxLayer = new SoftmaxLayer<T>(m_cuda, m_log, param_softmax);
            m_colSoftmaxBottom = new BlobCollection<T>();
            m_colSoftmaxTop = new BlobCollection<T>();

            m_colSoftmaxBottom.Add(colBottom[0]);
            m_colSoftmaxTop.Add(m_blobProb);
            m_softmaxLayer.Setup(m_colSoftmaxBottom, m_colSoftmaxTop);

            m_nIgnoreLabel = m_param.loss_param.ignore_label;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_softmaxLayer.Reshape(m_colSoftmaxBottom, m_colSoftmaxTop);
            m_nSoftmaxAxis = colBottom[0].CanonicalAxisIndex(m_param.softmax_param.axis);
            m_nOuterNum = colBottom[0].count(0, m_nSoftmaxAxis);
            m_nInnerNum = colBottom[0].count(m_nSoftmaxAxis + 1);

            if (!m_bIgnoreLabels)
            {
                m_log.CHECK_EQ(m_nOuterNum * m_nInnerNum, colBottom[1].count(), "Number of labels must match number of predictions; e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), label count (number of labels) must be N*H*W, with integer values in {0, 1, ..., C-1}.");

                if (colTop.Count >= 2)
                {
                    // softmax output
                    colTop[1].ReshapeLike(colBottom[0]);
                }
            }
        }

        /// <summary>
        /// The forward computation.
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
            m_softmaxLayer.Forward(m_colSoftmaxBottom, m_colSoftmaxTop);

            long hProbData = m_blobProb.gpu_data;
            long hLabel = colBottom[1].gpu_data;
            int nDim = m_blobProb.count() / m_nOuterNum;
            int nCount = m_nOuterNum * m_nInnerNum;

            // Since this memory is not used for anything, we use it here to avoid having
            // to allocate new GPU memory to accumulate intermediate results.
            long hLossData = colBottom[0].mutable_gpu_diff;

            // Similarly, this memory is never used elsewhere, and thus we can use it
            // to avoid having to allocate additional GPU memory.
            long hCounts = m_blobProb.mutable_gpu_diff;

            m_cuda.softmaxloss_fwd(nCount, hProbData, hLabel, hLossData, m_nOuterNum, nDim, m_nInnerNum, hCounts, m_nIgnoreLabel);
            T fLoss = m_cuda.asum(nCount, hLossData);
            double dfValidCount = -1;

            // Only launch another cuda kernel if we actually need the count of valid
            // outputs.
            if (m_normalization == LossParameter.NormalizationMode.VALID && m_nIgnoreLabel.HasValue)
                dfValidCount = convertD(m_cuda.asum(nCount, hCounts));

            double dfLoss = convertD(fLoss);
            double dfNormalizer = get_normalizer(m_normalization, (int)dfValidCount);
            double dfFinalLoss = dfLoss / dfNormalizer;

            colTop[0].SetData(dfFinalLoss, 0);

            if (colTop.Count == 2)
                colTop[1].ShareData(m_blobProb);

            // Clear scratch memory to prevent with interfering with backward pass (see #602)
            colBottom[0].SetDiff(0);
        }

        /// <summary>
        /// Computes the softmax loss error gradient w.r.t the predictions.
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

            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            long hProbData = m_blobProb.gpu_data;
            long hTopData = colTop[0].gpu_data;

            m_cuda.copy(m_blobProb.count(), hProbData, hBottomDiff);

            long hLabel = colBottom[1].gpu_data;
            int nDim = m_blobProb.count() / m_nOuterNum;
            int nCount = m_nOuterNum * m_nInnerNum;

            // Since this memory is not used for anything else,
            // we use to avoid allocating new GPU memory.
            long hCounts = m_blobProb.mutable_gpu_diff;

            m_cuda.softmaxloss_bwd(nCount, hTopData, hLabel, hBottomDiff, m_nOuterNum, nDim, m_nInnerNum, hCounts, m_nIgnoreLabel);

            double dfValidCount = -1;

            // Only launch another cuda kernel if we acutally need the count of valid
            // outputs.
            if (m_normalization == LossParameter.NormalizationMode.VALID && m_nIgnoreLabel.HasValue)
                dfValidCount = convertD(m_cuda.asum(nCount, hCounts));

            double dfTopDiff = convertD(colTop[0].GetDiff(0));
            double dfNormalizer = get_normalizer(m_normalization, (int)dfValidCount);
            double dfLossWeight = dfTopDiff / dfNormalizer;

            m_cuda.scal(m_blobProb.count(), convert(dfLossWeight), hBottomDiff);
        }
    }
}
