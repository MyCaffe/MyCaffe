using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The QuantileLossLayer computes the quantile loss
    /// @f$
    ///     E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
    ///         \right| \right|_2^2 
    /// @f$ for real-valued regression tasks. 
    /// </summary>
    /// <remarks>
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch/loss.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/loss.py) by Playtika Research, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class QuantileLossLayer<T> : LossLayer<T>
    {
        List<int> m_rgShape = new List<int>(4);
        int m_nCount;
        int m_nChannels;
        Blob<T> m_blobTargetsFull;
        Blob<T> m_blobErrors;
        Blob<T> m_blobQuantile1;
        Blob<T> m_blobQuantile2;
        Blob<T> m_blobDesiredQuantiles;
        Blob<T> m_blobLoss;
        Blob<T> m_blobLossSum;
        Blob<T> m_blobLossSumMean;
        Blob<T> m_blobWork;

        /// <summary>
        /// The QuantileLossLayer constructor
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type QUANTILE_LOSS.</param>
        public QuantileLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.QUANTILE_LOSS;

            m_blobErrors = new Blob<T>(cuda, log);
            m_blobErrors.Name = m_param.name + ".diff";
            m_blobTargetsFull = new Blob<T>(cuda, log);
            m_blobTargetsFull.Name = m_param.name + ".trgtfull";
            m_blobQuantile1 = new Blob<T>(cuda, log);
            m_blobQuantile1.Name = m_param.name + ".qtl1";
            m_blobQuantile2 = new Blob<T>(cuda, log);
            m_blobQuantile2.Name = m_param.name + ".qtl2";
            m_blobDesiredQuantiles = new Blob<T>(cuda, log);
            m_blobDesiredQuantiles.Name = m_param.name + ".desqtl";
            m_blobLoss = new Blob<T>(cuda, log);
            m_blobLoss.Name = m_param.name + ".loss";
            m_blobLossSum = new Blob<T>(cuda, log);
            m_blobLossSum.Name = m_param.name + ".losssum";
            m_blobLossSumMean = new Blob<T>(cuda, log);
            m_blobLossSumMean.Name = m_param.name + ".losssum.mean";
            m_blobWork = new Blob<T>(m_cuda, m_log);
            m_blobWork.Name = m_param.name + ".work";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobErrors);
            dispose(ref m_blobQuantile1);
            dispose(ref m_blobQuantile2);
            dispose(ref m_blobTargetsFull);
            dispose(ref m_blobDesiredQuantiles);
            dispose(ref m_blobLoss);
            dispose(ref m_blobLossSum);
            dispose(ref m_blobLossSumMean);
            dispose(ref m_blobWork);

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
        /// Returns the maximum number of required top (output) Blobs: loss, q_risk
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Unlike most loss layers, in the QuantileLossLayer we can backpropagate
        /// to both inputs -- override to return true and always allow force_backward.
        /// </summary>
        /// <param name="nBottomIdx">Specifies the index of the bottom element.</param>
        /// <returns>Returns <i>true</i>.</returns>
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

            List<int> rgShape = new List<int>(1);
            rgShape.Add(m_param.quantile_loss_param.desired_quantiles.Count);
            m_blobDesiredQuantiles.Reshape(rgShape);

            float[] rgDeqQtl1 = new float[m_param.quantile_loss_param.desired_quantiles.Count];
            float[] rgDeqQtl2 = new float[m_param.quantile_loss_param.desired_quantiles.Count];

            for (int i = 0; i < rgDeqQtl1.Length; i++)
            {
                rgDeqQtl1[i] = m_param.quantile_loss_param.desired_quantiles[i];
                rgDeqQtl2[i] = rgDeqQtl1[i] - 1;
            }

            m_blobDesiredQuantiles.mutable_cpu_data = convert(rgDeqQtl1);
            m_blobDesiredQuantiles.mutable_cpu_diff = convert(rgDeqQtl2);

            
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            int nAxes = colBottom[0].num_axes;
            m_nCount = colBottom[0].count();
            m_nOuterNum = colBottom[0].num;
            m_nChannels = (nAxes == 2) ? 1 : colBottom[0].channels;
            m_nInnerNum = (nAxes == 2) ? colBottom[0].channels : colBottom[0].count(2);

            m_log.CHECK_EQ(colBottom[0].num, colBottom[1].num, "Input and target must have same 'num' size.");
            m_log.CHECK_EQ(colBottom[0].channels, colBottom[1].channels, "Input and target must have same 'channel' size.");
            m_log.CHECK_EQ(colBottom[0].height, colBottom[1].height * m_param.quantile_loss_param.desired_quantiles.Count, "Input must have 'desired_quantile.Count' * target 'height' size.");

            m_blobErrors.ReshapeLike(colBottom[0]);
            m_blobTargetsFull.ReshapeLike(colBottom[0]);
            m_blobQuantile1.ReshapeLike(colBottom[0]);
            m_blobQuantile2.ReshapeLike(colBottom[0]);
            m_blobLoss.ReshapeLike(colBottom[0]);
            m_blobWork.ReshapeLike(colBottom[0]);

            m_rgShape.Clear();
            m_rgShape.Add(m_nOuterNum);
            m_rgShape.Add(m_nChannels);
            m_blobLossSum.Reshape(m_rgShape);

            m_rgShape.Clear();
            m_rgShape.Add(m_nOuterNum);
            m_blobLossSumMean.Reshape(m_rgShape);

            m_rgShape.Clear();
            m_rgShape.Add(1);
            colTop[0].Reshape(m_rgShape);

            if (colTop.Count > 1)
            {
                m_rgShape[0] = m_nChannels;
                colTop[1].Reshape(m_rgShape);
            }
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 2)
        ///  -# @f$ (N \times C \times QuantileCount \times 1) @f$
        ///     the quantile predictions @f$ \hat{y} \in [-\infty, +\infty] @f$
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the targets @f$ y \in [-infty, +\infty] @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed quantile loss.
        ///  -# @f$ (C \times 1 \times 1 \times 1) @f$
        ///     the quantile risk for each future item.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Fill the targets accross all output quantiles.
            m_cuda.channel_fillfrom(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, colBottom[1].gpu_data, m_blobTargetsFull.mutable_gpu_data, DIR.FWD);

            // Compute the actual error between the observed target and each predicted quantile
            m_cuda.sub(m_nCount, m_blobTargetsFull.gpu_data, colBottom[0].gpu_data, m_blobErrors.mutable_gpu_data);

            // Compute the loss separately for each sample, time-step, quantile
            m_cuda.channel_copyall(m_nCount, m_nOuterNum * m_nChannels, 1, m_nInnerNum, m_blobDesiredQuantiles.gpu_diff, m_blobWork.mutable_gpu_data);
            m_cuda.mul(m_nCount, m_blobWork.gpu_data, m_blobErrors.gpu_data, m_blobQuantile1.mutable_gpu_data);

            m_cuda.channel_copyall(m_nCount, m_nOuterNum * m_nChannels, 1, m_nInnerNum, m_blobDesiredQuantiles.gpu_data, m_blobWork.mutable_gpu_data);
            m_cuda.mul(m_nCount, m_blobWork.gpu_data, m_blobErrors.gpu_data, m_blobQuantile2.mutable_gpu_data);

            m_cuda.max(m_nCount, m_blobQuantile2.gpu_data, m_blobQuantile1.gpu_data, m_blobLoss.mutable_gpu_data);

            // Sum losses over the quantiles
            m_cuda.channel_sum(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_blobLoss.gpu_data, m_blobLossSum.mutable_gpu_data, false);

            // Mean of Sum losses over time
            m_cuda.channel_mean(m_blobLossSum.count(), m_nOuterNum, 1, m_nChannels, m_blobLossSum.gpu_data, m_blobLossSumMean.mutable_gpu_data);

            // Average across time and observations
            double dfQLoss = m_blobLossSumMean.mean();
            colTop[0].SetData(dfQLoss, 0);

            // Calculate the q-risk for each quantile
            if (colTop.Count > 1)
            {
                double dfTargetSum = convertD(colBottom[1].asum_data());
                m_cuda.channel_sum(m_blobLossSum.count(), 1, m_nOuterNum, m_nChannels, m_blobLossSum.gpu_data, colTop[1].mutable_gpu_data, true);

                colTop[1].scale_data(2.0 / dfTargetSum);
            }

            callLossEvent(m_blobLossSumMean);
        }

        /// <summary>
        /// Computes the QuantileLoss error gradient w.r.t. the inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     This blob's diff will simply contain the loss_weight * @f$ \lambda @f$,
        ///     as @f$ \lambda @f$ is the coefficient of this layer's output
        ///     @f$ \ell_i @f$ in the overall Net loss.
        ///     @f$ E = \lambda_i \ell_i + \mbox{other loss terms} @f$; hence
        ///     @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
        ///     (*Assuming that this top Blob is not used by a bottom (input) by any
        ///     other layer of the Net.)</param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">inpub Blob vector (length 2)
        ///  -# @f$ (N \times C \times QuantileCount \times 1) @f$
        ///     the quantile predictions @f$ \hat{y} \in [-\infty, +\infty] @f$; 
        ///     Backward fills their diff with gradients.
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the targets @f$ y \in [-infty, +\infty] @f$; 
        ///     ignored as we can't compute their error gradients.
        ///  </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            // Sum and average over quantiles, time and observations
            double dfGrad = convertD(colTop[0].GetDiff(0));
            m_blobLoss.SetDiff(dfGrad / (m_nOuterNum * m_nChannels));

            // Compute the grad separately for each sample, time-step, quantile
            m_cuda.max_bwd(m_nCount, m_blobQuantile2.gpu_data, m_blobQuantile1.gpu_data, m_blobLoss.gpu_diff, m_blobQuantile2.mutable_gpu_diff, m_blobQuantile1.mutable_gpu_diff);

            m_cuda.channel_copyall(m_nCount, m_nOuterNum * m_nChannels, 1, m_nInnerNum, m_blobDesiredQuantiles.gpu_data, m_blobWork.mutable_gpu_data);
            m_cuda.mul(m_nCount, m_blobWork.gpu_data, m_blobQuantile2.gpu_diff, m_blobQuantile2.mutable_gpu_diff);

            m_cuda.channel_copyall(m_nCount, m_nOuterNum * m_nChannels, 1, m_nInnerNum, m_blobDesiredQuantiles.gpu_diff, m_blobWork.mutable_gpu_data);
            m_cuda.mul(m_nCount, m_blobWork.gpu_data, m_blobQuantile1.gpu_diff, m_blobQuantile1.mutable_gpu_diff);

            m_cuda.add(m_nCount, m_blobQuantile1.gpu_diff, m_blobQuantile2.gpu_diff, m_blobErrors.mutable_gpu_diff);

            // Compute the actual grad between the observed target and each predicted quantile
            m_cuda.scale(m_nCount, -1.0, m_blobErrors.gpu_diff, colBottom[0].mutable_gpu_diff);
        }
    }
}
