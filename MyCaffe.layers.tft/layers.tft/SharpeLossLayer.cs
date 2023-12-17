using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using static System.Net.Mime.MediaTypeNames;

namespace MyCaffe.layers
{
    /// <summary>
    /// The SharpeLossLayer computes the Sharpe loss
    /// @f$
    ///     \begin{aligned}
    ///         captured\_returns = weights* y\_true
    ///         mean\_returns = \text{tf.reduce\_mean}(captured\_returns) 
    ///         E = -\frac{mean\_returns}{\sqrt{\text{ tf.reduce\_mean} (\text{ tf.square} (captured\_returns)) - \text{ tf.square} (mean\_returns) +1e-9} * \sqrt{ 252.0} }
    ///     \end{aligned}
    /// @f$. 
    /// </summary>
    /// <remarks>
    /// @see [Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture](https://arxiv.org/abs/2112.08534) by Kieran Wood, Sven Giegerich, Stephen Roberts, and Stefan Zohren, 2022, arXiv:2112.08534
    /// @see [Github - kieranjwood/trading-momentum-transformer](https://github.com/kieranjwood/trading-momentum-transformerh) by Kieran Wood, 2022.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SharpeLossLayer<T> : LossLayer<T>
    {
        List<int> m_rgShape = new List<int>(4);
        Blob<T> m_blobCapturedReturns;
        Blob<T> m_blobCapturedReturnsSq;
        Blob<T> m_blobMeanCapturedReturns;
        Blob<T> m_blobMeanCapturedReturnsSq;
        Blob<T> m_blobSquareMeanCapturedReturns;
        Blob<T> m_blobStdev;
        Blob<T> m_blobLoss;

        /// <summary>
        /// The SharpeLossLayer constructor
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type QUANTILE_LOSS.</param>
        public SharpeLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SHARPE_LOSS;

            m_blobCapturedReturns = new Blob<T>(cuda, log);
            m_blobCapturedReturns.Name = m_param.name + ".captured_returns";
            m_blobCapturedReturnsSq = new Blob<T>(cuda, log);
            m_blobCapturedReturnsSq.Name = m_param.name + ".captured_returns_sq";
            m_blobMeanCapturedReturns = new Blob<T>(cuda, log);
            m_blobMeanCapturedReturns.Name = m_param.name + ".mean.captured_returns";
            m_blobMeanCapturedReturnsSq = new Blob<T>(cuda, log);
            m_blobMeanCapturedReturnsSq.Name = m_param.name + ".mean.captured_returns_sq";
            m_blobSquareMeanCapturedReturns = new Blob<T>(cuda, log);
            m_blobSquareMeanCapturedReturns.Name = m_param.name + ".square.mean.captured_returns";
            m_blobStdev = new Blob<T>(cuda, log);
            m_blobStdev.Name = m_param.name + ".stdev";
            m_blobLoss = new Blob<T>(cuda, log);
            m_blobLoss.Name = m_param.name + ".loss";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobCapturedReturns);
            dispose(ref m_blobCapturedReturnsSq);
            dispose(ref m_blobMeanCapturedReturns);
            dispose(ref m_blobMeanCapturedReturnsSq);
            dispose(ref m_blobSquareMeanCapturedReturns);
            dispose(ref m_blobStdev);
            dispose(ref m_blobLoss);

            base.dispose();
        }

        /// <summary>
        /// Unlike most loss layers, in the SharpeLossLayer we can backpropagate
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

            m_log.CHECK_EQ(colBottom.Count, 2, "There should be two inputs: position predictions and target returns.");
            m_log.CHECK_EQ(colBottom[0].count(), colBottom[1].count(), "The inputs must have the same count.");
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_nOuterNum = colBottom[0].num;
            m_nInnerNum = colBottom[0].count(1);

            m_blobCapturedReturns.ReshapeLike(colBottom[0]);
            m_blobCapturedReturnsSq.ReshapeLike(colBottom[0]);

            m_blobMeanCapturedReturns.Reshape(m_nOuterNum, 1, 1, 1);
            m_blobMeanCapturedReturnsSq.Reshape(m_nOuterNum, 1, 1, 1);
            m_blobSquareMeanCapturedReturns.Reshape(m_nOuterNum, 1, 1, 1);
            m_blobStdev.Reshape(m_nOuterNum, 1, 1, 1);
            m_blobLoss.Reshape(m_nOuterNum, 1, 1, 1);

            m_rgShape.Clear();
            m_rgShape.Add(1);
            colTop[0].Reshape(m_rgShape);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 2)
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the position predictions @f$ \hat{y} \in [-\infty, +\infty] @f$
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the target returns @f$ y \in [-infty, +\infty] @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed sharpe loss.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // captured_returns = weights * y_true
            m_cuda.mul(m_blobCapturedReturns.count(), colBottom[0].gpu_data, colBottom[1].gpu_data, m_blobCapturedReturns.mutable_gpu_data);
            // captured_returns_sq = captured_returns^2
            m_cuda.powx(m_blobCapturedReturns.count(), m_blobCapturedReturns.gpu_data, 2.0, m_blobCapturedReturnsSq.mutable_gpu_data);

            // mean_returns = tf.reduce_mean(captured_returns)
            m_cuda.channel_mean(m_blobCapturedReturns.count(), m_nOuterNum, 1, m_nInnerNum, m_blobCapturedReturns.gpu_data, m_blobMeanCapturedReturns.mutable_gpu_data);

            // mean_returns_sq = tf.reduce_mean(captured_returns_sq)
            m_cuda.channel_mean(m_blobCapturedReturnsSq.count(), m_nOuterNum, 1, m_nInnerNum, m_blobCapturedReturnsSq.gpu_data, m_blobMeanCapturedReturnsSq.mutable_gpu_data);

            m_blobMeanCapturedReturns.Reshape(m_nOuterNum, 1, 1, 1);
            m_blobMeanCapturedReturnsSq.Reshape(m_nOuterNum, 1, 1, 1);

            // mean_returns_sq = mean_returns^2
            m_cuda.powx(m_blobMeanCapturedReturns.count(), m_blobMeanCapturedReturns.gpu_data, 2.0, m_blobSquareMeanCapturedReturns.mutable_gpu_data);
            // subtract mean_returns_sq from mean_returns_sq
            m_cuda.sub(m_blobMeanCapturedReturnsSq.count(), m_blobMeanCapturedReturnsSq.gpu_data, m_blobSquareMeanCapturedReturns.gpu_data, m_blobStdev.mutable_gpu_data);
            // Add 1e-9 to avoid division by zero.
            m_cuda.add_scalar(m_blobStdev.count(), convert(1e-9), m_blobStdev.mutable_gpu_data);

            // take the square root of the difference.
            m_cuda.powx(m_blobStdev.count(), m_blobStdev.gpu_data, 0.5, m_blobStdev.mutable_gpu_data);

            // divide mean_returns by the difference.
            m_cuda.div(m_blobStdev.count(), m_blobMeanCapturedReturns.gpu_data, m_blobStdev.gpu_data, m_blobLoss.mutable_gpu_data);
            // multiply by 252.0 to annualize.
            m_cuda.scal(m_blobLoss.count(), convert(Math.Sqrt(252.0)), m_blobLoss.mutable_gpu_data);

            // average the loss over the batches.
            double dfLoss = m_blobLoss.mean();
            colTop[0].SetData(-1.0 * dfLoss, 0);
        }

        /// <summary>
        /// Computes the SharpeLoss error gradient w.r.t. the inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     @f$ \frac{\partial E}{\partial E} = \frac{y_i}{\sigma^2} \times \sqrt{252} @f$
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">inpub Blob vector (length 1)</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            // sharpe = mean_returns / stdev * sqrt(252.0)
            m_cuda.div(m_blobStdev.count(), m_blobMeanCapturedReturns.gpu_data, m_blobStdev.gpu_data, m_blobMeanCapturedReturns.mutable_gpu_diff);
            m_cuda.scal(m_blobMeanCapturedReturns.count(), convert(Math.Sqrt(252.0)), m_blobMeanCapturedReturns.mutable_gpu_diff);
            // grad = y * sharpe / (stdev^2)
            m_cuda.powx(m_blobStdev.count(), m_blobStdev.gpu_data, 2.0, m_blobStdev.mutable_gpu_diff);
            m_cuda.div(m_blobStdev.count(), m_blobMeanCapturedReturns.mutable_gpu_diff, m_blobStdev.gpu_diff, m_blobMeanCapturedReturns.mutable_gpu_diff);
            // grad fill along all values of the channel.
            m_cuda.channel_fillfrom(colBottom[0].count(), m_nOuterNum, 1, m_nInnerNum, m_blobMeanCapturedReturns.gpu_diff, colBottom[0].mutable_gpu_diff, DIR.FWD);
            // grad = y * grad
            m_cuda.mul(colBottom[0].count(), colBottom[1].gpu_data, colBottom[0].gpu_diff, colBottom[0].mutable_gpu_diff);
        }
    }
}
