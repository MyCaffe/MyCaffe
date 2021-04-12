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
    /// Computes the mean absolute error as a loss which computes:
    /// @f$ L(y, y\hat) = \frac{1}{N} \sum_{i=0}^{N} |y - \hat{y}{i}| @f$ where @f$ \hat{y} @f$ is the predicted value.
    /// </summary>
    /// <remarks>
    /// Used with regression models, such as those used with time-series prediction.
    /// @see [Mean Absolute Error](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-absolute-error) by Peltarion.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.</typeparam>
    public class MAELossLayer<T> : LossLayer<T>
    {
        int m_nAxis = 1;

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
        public MAELossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.MAE_LOSS;
            m_nAxis = p.mae_loss_param.axis;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();
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
        /// Returns the maximum number of required top (output) Blobs: loss.
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$, a blob with values in
        ///     @f$ [-\infty, +\infty] @f$ indicating the predicted values.
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$, a blob with values in
        ///     @f$ [-\infty, +\infty] @f$ indicating the target values.
        /// <param name="colTop">top output blob vector (length 1)
        ///     the computed mean absolute error loss: 
        ///     @f$ E = \frac{1}{N} \sum_{i=0}^{N} |y - \hat{y}{i}| @f$ where @f$ \hat{y} @f$ is the predicted value.
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hPredicted = colBottom[0].gpu_data;
            long hTarget = colBottom[1].gpu_data;
            int nCount = colBottom[0].count();

            m_log.CHECK_EQ(nCount, colBottom[1].count(), "The bottom(0) predicted and bottom(1) target must have the same shapes!");

            m_cuda.sub(nCount, hTarget, hPredicted, colBottom[0].mutable_gpu_diff);
            m_cuda.abs(nCount, colBottom[0].gpu_diff, colBottom[0].mutable_gpu_diff);
            double dfLoss = m_cuda.asum_double(nCount, colBottom[0].gpu_diff);

            dfLoss /= colBottom[0].shape(m_nAxis);

            colTop[0].SetData(dfLoss, 0);

            // Clear scratch memory to prevent with interfering with backward pass (see #602)
            colBottom[0].SetDiff(0);
        }

        /// <summary>
        /// Computes the softmax loss error gradient w.r.t the predictions.
        /// </summary>
        /// <remarks>
        /// The gradient is set to:
        ///     +1 when predicted greater than target,
        ///     -1 when predicted less than target,
        ///      0 when predicted equal to target.
        /// if propagate_down[1] == true.
        /// 
        /// @see [Mean Absolute Error (MAE) derivative](https://stats.stackexchange.com/questions/312737/mean-absolute-error-mae-derivative)
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
        ///     the predictions @f$ \hat{x} @f$; backward computes diff @f$
        ///       \frac{\partial E}{\partial x}
        ///     @f$
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the targets @f$ x @f$.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            long hPredicted = colBottom[0].gpu_data;
            long hTarget = colBottom[1].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int nCount = colBottom[0].count();

            m_cuda.mae_loss_bwd(nCount, hPredicted, hTarget, hBottomDiff);

            double dfTopDiff = convertD(colTop[0].GetDiff(0));
            double dfNormalizer = colBottom[0].shape(m_nAxis);
            double dfLossWeight = dfTopDiff / dfNormalizer;

            m_cuda.scal(nCount, dfLossWeight, hBottomDiff);
        }
    }
}
