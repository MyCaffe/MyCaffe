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
    /// @see [The Proximal Map of the Weighted Mean Absolute Error](https://arxiv.org/abs/2209.13545) by Lukas Baumgärtner, Roland Herzog, Stephan Schmidt, Manuel Weiß, 2022, arXiv 2209.13545
    /// @see [Mean Absolute Error In Machine Learning: What You Need To Know](https://arize.com/blog-course/mean-absolute-error-in-machine-learning-what-you-need-to-know/) by David Burch, 2023, ariz.com
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.</typeparam>
    public class MeanErrorLossLayer<T> : LossLayer<T>
    {
        RollingBucketCollection m_buckets = null;
        RollingBucketCollection m_bucketsCorrect = null;
        bool m_bLossWeightsReady = false;
        int m_nCurrentIteration = 0;
        int m_nAxis = 1;
        MEAN_ERROR m_meanType = MEAN_ERROR.MAE;
        Blob<T> m_blobWork;
        Blob<T> m_blobWeights;
        float m_fMin = float.MaxValue;
        float m_fMax = -float.MaxValue;
        float[] m_rgWeights = null;
        float m_fAlpha;

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
            m_blobWork.Name = m_param.name + " workspace";
            m_blobWeights = new Blob<T>(cuda, log);
            m_blobWeights.Name = m_param.name + " weights";
            m_fAlpha = p.mean_error_loss_param.weight_frequency_error_alpha;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobWork);
            dispose(ref m_blobWeights);

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
            m_blobWeights.ReshapeLike(colBottom[0]);

            if (m_nOuterNum == 0)
                m_nOuterNum = (int)colBottom[0].count(0, m_nAxis);

            if (m_nInnerNum == 0)
                m_nInnerNum = colBottom[0].count(m_nAxis);
        }

        private void ComputeWeights(float[] rgPred, float[] rgTarget, int nCount)
        {
            // Create bucket collection for target values
            if (m_buckets == null)
                return;

            if (m_rgWeights == null)
                m_rgWeights = new float[nCount];

            // Add values to buckets
            for (int i = 0; i < nCount; i++)
            {
                float fTarget = rgTarget[i];
                float fPred = rgPred[i];
                List<float> rgReturns = null;

                int nIdxTgt = m_buckets.Add(fTarget, false, rgReturns);
                int nIdxPred = m_bucketsCorrect.Add(fPred, true);

                if (nIdxTgt == nIdxPred)
                    m_bucketsCorrect.Add(fPred, false, rgReturns);
            }

            // Get bucket with max count for weight normalization
            Bucket bMax = m_buckets.Current.GetBucketWithMaxCount();

            // Compute inverse frequency weights
            for (int i = 0; i < nCount; i++)
            {
                // Get range-based weight
                int nBucketIdx = m_buckets.Add(rgTarget[i], true);
                Bucket bFreq = m_buckets.Current[nBucketIdx];
                Bucket bCor = m_bucketsCorrect.Current[nBucketIdx];

                // Calculate frequency weight with alpha
                float fFrequencyWeight = (1.0f + m_fAlpha) * ((float)bMax.Count / bFreq.Count);

                // Calculate error weight with beta
                float fErrorWeight = (1.0f - m_fAlpha) * (1.0f - ((float)bCor.Count / bFreq.Count));

                m_rgWeights[i] = fFrequencyWeight * fErrorWeight;


                // Constrain weights to the max.
                if (m_rgWeights[i] > m_param.mean_error_loss_param.max_weight)
                    m_rgWeights[i] = m_param.mean_error_loss_param.max_weight;
            }

            // Copy weights to device
            m_blobWeights.mutable_cpu_data = convert(m_rgWeights);
            m_bLossWeightsReady = true;
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$, a blob with values in
        ///     @f$ [-\infty, +\infty] @f$ indicating the predicted values.
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the targets @f$ y @f$, a blob with values in
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

            m_nCurrentIteration++;

            double dfLoss = 0;

            m_log.CHECK_EQ(nCount, colBottom[1].count(), "The bottom(0) predicted and bottom(1) target must have the same shapes!");

            // When using weighted loss, run warmup iterations to collect target values for bucketing
            if (m_param.mean_error_loss_param.enable_weighted_loss &&
                m_nCurrentIteration >= m_param.mean_error_loss_param.weight_warmup_iterations)
            {
                if (m_buckets == null)
                    m_buckets = new RollingBucketCollection(m_fMin, m_fMax, m_param.mean_error_loss_param.weight_bucket_count, m_param.mean_error_loss_param.weight_max_history);

                if (m_bucketsCorrect == null)
                    m_bucketsCorrect = new RollingBucketCollection(m_fMin, m_fMax, m_param.mean_error_loss_param.weight_bucket_count, m_param.mean_error_loss_param.weight_max_history);

                ComputeWeights(convertF(colBottom[0].update_cpu_data()), convertF(colBottom[1].update_cpu_data()), nCount);
            }
            else
            {
                Tuple<double, double, double, double> minmax = colBottom[1].minmax_data(m_blobWork);
                m_fMin = Math.Min(m_fMin, (float)minmax.Item1);
                m_fMax = Math.Max(m_fMax, (float)minmax.Item2);
            }

            switch (m_meanType)
            {
                case MEAN_ERROR.MSE:
                    m_cuda.sub(nCount, hTarget, hPredicted, colBottom[0].mutable_gpu_diff);
                    m_cuda.powx(nCount, colBottom[0].gpu_diff, 2.0, colBottom[0].mutable_gpu_diff);

                    if (m_param.mean_error_loss_param.penalize_values_below_threshold.HasValue)
                    {
                        float fPenalty = m_param.mean_error_loss_param.penalize_values_below_threshold.Value;
                        float fLambda = m_param.mean_error_loss_param.below_penalty_portion_lambda;
                        m_cuda.copy(nCount, colBottom[0].gpu_diff, m_blobWork.mutable_gpu_data);
                        m_cuda.threshold(nCount, m_blobWork.gpu_data, fPenalty, SIDE.BELOW, m_blobWork.mutable_gpu_data);
                        m_cuda.powx(nCount, m_blobWork.gpu_data, 2.0, m_blobWork.mutable_gpu_data);
                        m_cuda.axpy(nCount, fLambda, m_blobWork.gpu_data, colBottom[0].mutable_gpu_diff);
                    }

                    if (m_param.mean_error_loss_param.penalize_values_above_threshold.HasValue)
                    {
                        float fPenalty = m_param.mean_error_loss_param.penalize_values_above_threshold.Value;
                        float fLambda = m_param.mean_error_loss_param.above_penalty_portion_lambda;
                        m_cuda.copy(nCount, colBottom[0].gpu_diff, m_blobWork.mutable_gpu_data);
                        m_cuda.threshold(nCount, m_blobWork.gpu_data, fPenalty, SIDE.ABOVE, m_blobWork.mutable_gpu_data);
                        m_cuda.powx(nCount, m_blobWork.gpu_data, 2.0, m_blobWork.mutable_gpu_data);
                        m_cuda.axpy(nCount, fLambda, m_blobWork.gpu_data, colBottom[0].mutable_gpu_diff);
                    }

                    dfLoss = m_cuda.asum_double(nCount, colBottom[0].gpu_diff);
                    break;

                case MEAN_ERROR.MAE:
                    m_cuda.sub(nCount, hTarget, hPredicted, colBottom[0].mutable_gpu_diff);
                    m_cuda.abs(nCount, colBottom[0].gpu_diff, colBottom[0].mutable_gpu_diff);

                    if (m_bLossWeightsReady)
                        m_cuda.mul(nCount, colBottom[0].gpu_diff, m_blobWeights.gpu_data, colBottom[0].mutable_gpu_diff);

                    if (m_param.mean_error_loss_param.penalize_values_below_threshold.HasValue)
                    {
                        float fPenalty = m_param.mean_error_loss_param.penalize_values_below_threshold.Value;
                        float fLambda = m_param.mean_error_loss_param.below_penalty_portion_lambda;
                        m_cuda.copy(nCount, colBottom[0].gpu_diff, m_blobWork.mutable_gpu_data);
                        m_cuda.threshold(nCount, m_blobWork.gpu_data, fPenalty, SIDE.BELOW, m_blobWork.mutable_gpu_data);
                        m_cuda.powx(nCount, m_blobWork.gpu_data, 2.0, m_blobWork.mutable_gpu_data);
                        m_cuda.axpy(nCount, fLambda, m_blobWork.gpu_data, colBottom[0].mutable_gpu_diff);
                    }

                    if (m_param.mean_error_loss_param.penalize_values_above_threshold.HasValue)
                    {
                        float fPenalty = m_param.mean_error_loss_param.penalize_values_above_threshold.Value;
                        float fLambda = m_param.mean_error_loss_param.above_penalty_portion_lambda;
                        m_cuda.copy(nCount, colBottom[0].gpu_diff, m_blobWork.mutable_gpu_data);
                        m_cuda.threshold(nCount, m_blobWork.gpu_data, fPenalty, SIDE.ABOVE, m_blobWork.mutable_gpu_data);
                        m_cuda.powx(nCount, m_blobWork.gpu_data, 2.0, m_blobWork.mutable_gpu_data);
                        m_cuda.axpy(nCount, fLambda, m_blobWork.gpu_data, colBottom[0].mutable_gpu_diff);
                    }

                    dfLoss = m_cuda.asum_double(nCount, colBottom[0].gpu_diff);
                    break;
            }

            double dfNormalizer = get_normalizer(m_normalization, -1);
            double dfLossFinal;

            if ((m_param.mean_error_loss_param.enable_weighted_loss && m_bLossWeightsReady))
            {
                double dfWeightSum = m_cuda.asum_double(nCount, m_blobWeights.gpu_data);
                dfLossFinal = dfLoss / dfWeightSum;
            }
            else
            {
                dfLossFinal = dfLoss / dfNormalizer;
            }

            colTop[0].SetData(dfLossFinal, 0);

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

            // Apply weights if weighted MAE is enabled and warmup is complete
            if (m_bLossWeightsReady)
                m_cuda.mul(nCount, hBottomDiff, m_blobWeights.gpu_data, hBottomDiff);

            double dfTopDiff = convertD(colTop[0].GetDiff(0));
            double dfNormalizer = get_normalizer(m_normalization, -1);
            double dfLossWeight = dfTopDiff / dfNormalizer;

            if (dfLossWeight != 1.0)
                m_cuda.scal(nCount, dfLossWeight, hBottomDiff);

            // Apply below-threshold penalty
            if (m_param.mean_error_loss_param.penalize_values_below_threshold.HasValue)
            {
                float fPenalty = m_param.mean_error_loss_param.penalize_values_below_threshold.Value;
                float fLambda = m_param.mean_error_loss_param.below_penalty_portion_lambda;

                // Adjust gradients for below-threshold values
                m_cuda.copy(nCount, hBottomDiff, m_blobWork.mutable_gpu_data);
                m_cuda.threshold(nCount, hBottomDiff, fPenalty, SIDE.BELOW, m_blobWork.mutable_gpu_data);
                m_cuda.axpy(nCount, fLambda, m_blobWork.gpu_data, hBottomDiff);  // Apply gradient penalty
            }

            // Apply above-threshold penalty
            if (m_param.mean_error_loss_param.penalize_values_above_threshold.HasValue)
            {
                float fPenalty = m_param.mean_error_loss_param.penalize_values_above_threshold.Value;
                float fLambda = m_param.mean_error_loss_param.above_penalty_portion_lambda;

                // Adjust gradients for above-threshold values
                m_cuda.copy(nCount, hBottomDiff, m_blobWork.mutable_gpu_data);
                m_cuda.threshold(nCount, hBottomDiff, fPenalty, SIDE.ABOVE, m_blobWork.mutable_gpu_data);
                m_cuda.axpy(nCount, fLambda, m_blobWork.gpu_data, hBottomDiff);  // Apply gradient penalty
            }

            if (colBottom.Count > 1 && rgbPropagateDown[1])
                m_cuda.scale(nCount, -1, hBottomDiff, colBottom[1].mutable_gpu_diff);
        }
    }
}
