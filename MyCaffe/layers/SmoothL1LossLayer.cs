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
    /// Fast R-CNN
    /// Copyright (c) Microsoft
    /// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
    /// Originally written in C by Ross Girshick
    /// Modified by Wei Liu in C
    /// Rewritten in C# by SignalPop LLC
    /// 
    /// Computes the SmoothL1 loss as introduced in: 
    /// Fast R-CNN, Ross Girschick, ICCV 2015
    /// </summary>
    /// <remarks>
    /// @see [Fast R-CNN](https://arxiv.org/abs/1504.08083) by Ross Girshick, 2015
    /// @see [GitHub: rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn), by Ross Girschick, 2015
    /// @see [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/abs/1612.02295) by Weiyang Liu, Yandong Wen, Zhiding Yu and Meng Yang, 2016. 
    /// @see [Generalisation and Sharing in Triplet Convnets for Sketch based Visual Search](https://arxiv.org/abs/1611.05301v1) by Tu Bui, Leonardo Ribeiro, Moacir Ponti, and John Collomosse, 2016.
    /// </remarks>
    /// <typeparam name="T"></typeparam>
    public class SmoothL1LossLayer<T> : LossLayer<T>
    {
        Blob<T> m_blobDiff;
        Blob<T> m_blobErrors;
        bool m_bHasWeights;

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
        public SmoothL1LossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SMOOTHL1_LOSS;

            m_blobDiff = new Blob<T>(cuda, log, false);
            m_blobDiff.Name = "diff";
            m_blobErrors = new Blob<T>(cuda, log, false);
            m_blobErrors.Name = "errors";

            m_bHasWeights = false;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_blobDiff != null)
            {
                m_blobDiff.Dispose();
                m_blobDiff = null;
            }

            if (m_blobErrors != null)
            {
                m_blobErrors.Dispose();
                m_blobErrors = null;
            }

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();
                col.Add(m_blobDiff);
                col.Add(m_blobErrors);
                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (output) Blobs as variable.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: loss, labels
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the maximum number of required top (output) Blobs: loss, labels, weights
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 3; }
        }

        /// <summary>
        /// Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
        /// to both inputs -- override to return true and always allow force_backward.
        /// </summary>
        /// <param name="nBottomIdx"></param>
        /// <returns></returns>
        public override bool AllowForceBackward(int nBottomIdx)
        {
            return true;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            if (colBottom.Count == 3)
                m_bHasWeights = true;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_log.CHECK_EQ(colBottom[0].channels, colBottom[1].channels, "The bottom(0) and bottom(1) must have the same channels.");
            m_log.CHECK_EQ(colBottom[0].height, colBottom[1].height, "The bottom(0) and bottom(1) must have the same height.");
            m_log.CHECK_EQ(colBottom[0].width, colBottom[1].width, "The bottom(0) and bottom(1) must have the same width.");

            if (m_bHasWeights)
            {
                m_log.CHECK_EQ(colBottom[0].channels, colBottom[2].channels, "The bottom(0) and bottom(2) must have the same channels.");
                m_log.CHECK_EQ(colBottom[0].height, colBottom[2].height, "The bottom(0) and bottom(2) must have the same height.");
                m_log.CHECK_EQ(colBottom[0].width, colBottom[2].width, "The bottom(0) and bottom(2) must have the same width.");
            }

            m_blobDiff.ReshapeLike(colBottom[0]);
            m_blobErrors.ReshapeLike(colBottom[0]);
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$, a blob with values in
        ///     @f$ [-\infty, +\infty] @f$ indicating the predicted score for eachy of
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
            int nCount = colBottom[0].count();

            // d := b0 - b1
            m_cuda.sub(nCount, colBottom[0].gpu_data, colBottom[1].gpu_data, m_blobDiff.mutable_gpu_data);

            // d := w * (b0 - b1)
            if (m_bHasWeights)
                m_cuda.mul(nCount, colBottom[2].gpu_data, m_blobDiff.gpu_data, m_blobDiff.mutable_gpu_data);

            m_cuda.smoothl1_fwd(nCount, m_blobDiff.gpu_data, m_blobErrors.mutable_gpu_data);

            double dfLoss = Utility.ConvertVal<T>(m_blobErrors.asum_data());
            colTop[0].SetData(dfLoss / colBottom[0].num, 0);
        }

        /// <summary>
        /// Computes the smooth L1 loss error gradient w.r.t the predictions.
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
            int nCount = m_blobDiff.count();

            m_cuda.smoothl1_bwd(nCount, m_blobDiff.gpu_data, m_blobDiff.mutable_gpu_data);

            for (int i = 0; i < 2; i++)
            {
                if (rgbPropagateDown[i])
                {
                    double dfSign = (i == 0) ? 1 : -1;
                    double dfAlpha = Utility.ConvertVal<T>(colTop[0].GetDiff(0));

                    dfAlpha = dfSign * dfAlpha / colBottom[i].num;
                    m_cuda.axpby(colBottom[i].count(), dfAlpha, m_blobDiff.gpu_data, 0.0, colBottom[i].mutable_gpu_diff);
                }
            }
        }
    }
}
