using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.beta;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The MeanErrorLossLayer computes losses based on various different Mean Error methods as shown below.  This layer is used
    /// to solve regression problems such as time-series predictions.
    /// 
    /// Mean Squared Error (MSE)
    /// @f$ L(y, y\hat) = \frac{1}{N} \sum_{i=0}^{N} (y - \hat{y}{i})^2 @f$ where @f$ \hat{y} @f$ is the predicted value.
    /// 
    /// Mean Absolute Error (MAE)
    /// @f$ L(y, y\hat) = \frac{1}{N} \sum_{i=0}^{N} |y - \hat{y}{i}| @f$ where @f$ \hat{y} @f$ is the predicted value.
    /// </summary>
    /// <remarks>
    /// @see [Methods for forecasts of continuous variables](https://www.cawcr.gov.au/projects/verification/#Methods_for_foreasts_of_continuous_variables) by WCRP, 2017.
    /// @see [MAD vs RMSE vs MAE vs MSLE vs R^2: When to use which?](https://datascience.stackexchange.com/questions/42760/mad-vs-rmse-vs-mae-vs-msle-vs-r%C2%B2-when-to-use-which), StackExchange, 2018.
    /// @see [Mean Absolute Error](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-absolute-error) by Peltarion.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.</typeparam>
    public class MeanErrorLossLayer<T> : LossLayer<T>
    {
        int m_nAxis = 1;
        MEAN_ERROR m_meanType = MEAN_ERROR.MAE;
        Blob<T> m_blobWork;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides LossParameter loss_param, with options:
        ///  - axis (optional)
        ///    Specify a label value that whould be ignored when computing the loss.
        ///  - normalize (optional, default true)
        ///    If true, the loss is normalized by the number of (nonignored) labels
        ///    present; otherwise the loss is imply summed over spatial locations.
        ///  - axis (optional, default 1)
        ///    Specify the axis of the probability.
        ///  - mean_type (optional, default ME.MAE)
        ///    Specify the type of mean error to use.
        /// </param>
        public MeanErrorLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.MEAN_ERROR_LOSS;
            m_nAxis = p.mean_error_loss_param.axis;
            m_meanType = p.mean_error_loss_param.mean_error_type;

            m_blobWork = new Blob<T>(cuda, log);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_blobWork != null)
            {
                m_blobWork.Dispose();
                m_blobWork = null;
            }

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

            m_blobWork.ReshapeLike(colBottom[0]);

            if (m_nOuterNum == 0)
                m_nOuterNum = (int)colBottom[0].count(0, m_nAxis);

            if (m_nInnerNum == 0)
                m_nInnerNum = colBottom[0].count(m_nAxis);
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
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///     MSE - the computed mean squared error loss: 
        ///     @f$ E = \frac{1}{N} \sum_{i=0}^{N} (y - \hat{y}{i})^2 @f$ where @f$ \hat{y} @f$ is the predicted value.
        ///     
        ///     MAE - the computed mean absolute error loss: 
        ///     @f$ E = \frac{1}{N} \sum_{i=0}^{N} |y - \hat{y}{i}| @f$ where @f$ \hat{y} @f$ is the predicted value.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hPredicted = colBottom[0].gpu_data;
            long hTarget = colBottom[1].gpu_data;
            int nCount = colBottom[0].count();

            double dfLoss = 0;

            m_log.CHECK_EQ(nCount, colBottom[1].count(), "The bottom(0) predicted and bottom(1) target must have the same shapes!");

            switch (m_meanType)
            {
                case MEAN_ERROR.MSE:
                    m_cuda.sub(nCount, hTarget, hPredicted, colBottom[0].mutable_gpu_diff);
                    m_cuda.powx(nCount, colBottom[0].gpu_diff, 2.0, colBottom[0].mutable_gpu_diff);
                    dfLoss = m_cuda.asum_double(nCount, colBottom[0].gpu_diff);
                    break;

                case MEAN_ERROR.MAE:
                    m_cuda.sub(nCount, hTarget, hPredicted, colBottom[0].mutable_gpu_diff);
                    m_cuda.abs(nCount, colBottom[0].gpu_diff, colBottom[0].mutable_gpu_diff);
                    dfLoss = m_cuda.asum_double(nCount, colBottom[0].gpu_diff);
                    break;
            }

            double dfNormalizer = get_normalizer(m_normalization, -1);
            colTop[0].SetData(dfLoss / dfNormalizer, 0);

            // Clear scratch memory to prevent with interfering with backward pass (see #602)
            colBottom[0].SetDiff(0);
        }

        /// <summary>
        /// Computes the softmax loss error gradient w.r.t the predictions.
        /// </summary>
        /// <remarks>
        /// MSE Gradient
        /// @see [Wolframe Alpha: derivative of (t - p)^2 = d/dp((t - p)^2) = -2 (t - p)](https://www.wolframalpha.com/input?i=derivative+of+%28t+-+p%29%5E2)
        /// 
        /// MAE Gradient
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

            m_cuda.mean_error_loss_bwd(nCount, hPredicted, hTarget, hBottomDiff, m_meanType);

            double dfTopDiff = convertD(colTop[0].GetDiff(0));
            double dfNormalizer = get_normalizer(m_normalization, -1);
            double dfLossWeight = dfTopDiff / dfNormalizer;

            m_cuda.scal(nCount, dfLossWeight, hBottomDiff);

            if (colBottom.Count > 1)
            {
                long hBottomDiff2 = colBottom[1].mutable_gpu_diff;
                m_cuda.scale(nCount, -1, hBottomDiff, hBottomDiff2);
            }
        }
    }
}
