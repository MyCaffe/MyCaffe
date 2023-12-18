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
        Blob<T> m_blobMeanCapturedReturns;
        Blob<T> m_blobStdevCapturedReturns;
        Blob<T> m_blobStdevCapturedReturnsFull;
        Blob<T> m_blobMeanCapturedReturnsFull;
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
            m_blobMeanCapturedReturns = new Blob<T>(cuda, log);
            m_blobMeanCapturedReturns.Name = m_param.name + ".mean.captured_returns";
            m_blobStdevCapturedReturns = new Blob<T>(cuda, log);
            m_blobStdevCapturedReturns.Name = m_param.name + ".stdev";
            m_blobStdevCapturedReturnsFull = new Blob<T>(cuda, log);
            m_blobStdevCapturedReturnsFull.Name = m_param.name + ".stdev.full";
            m_blobMeanCapturedReturnsFull = new Blob<T>(cuda, log);
            m_blobMeanCapturedReturnsFull.Name = m_param.name + ".stdev.full";
            m_blobLoss = new Blob<T>(cuda, log);
            m_blobLoss.Name = m_param.name + ".loss";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobCapturedReturns);
            dispose(ref m_blobMeanCapturedReturns);
            dispose(ref m_blobStdevCapturedReturns);
            dispose(ref m_blobStdevCapturedReturnsFull);
            dispose(ref m_blobMeanCapturedReturnsFull);
            dispose(ref m_blobLoss);

            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobCapturedReturns);
            col.Add(m_blobMeanCapturedReturns);
            col.Add(m_blobStdevCapturedReturns);
            col.Add(m_blobStdevCapturedReturnsFull);
            col.Add(m_blobMeanCapturedReturnsFull);
            col.Add(m_blobLoss);
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
            m_blobStdevCapturedReturnsFull.ReshapeLike(colBottom[0]);
            m_blobMeanCapturedReturnsFull.ReshapeLike(colBottom[0]);

            m_blobMeanCapturedReturns.Reshape(m_nOuterNum, 1, 1, 1);
            m_blobStdevCapturedReturns.Reshape(m_nOuterNum, 1, 1, 1);
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
            // mean and stdev of captured_returns
            m_cuda.add_scalar(m_blobCapturedReturns.count(), 1e-9f, m_blobCapturedReturns.mutable_gpu_data);
            m_cuda.channel_stdev(m_blobCapturedReturns.count(), m_nOuterNum, 1, m_nInnerNum, m_blobCapturedReturns.gpu_data, m_blobStdevCapturedReturns.mutable_gpu_data, m_blobMeanCapturedReturns.gpu_data, 1e-9f, true);
            // mean / stdev
            m_cuda.div(m_blobLoss.count(), m_blobMeanCapturedReturns.gpu_data, m_blobStdevCapturedReturns.gpu_data, m_blobLoss.mutable_gpu_data);

            // average the loss over the batches.
            m_blobLoss.scale_data(-1.0);
            double dfLoss = m_blobLoss.mean();
            colTop[0].SetData(dfLoss, 0);
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
        /// <remarks>
        /// Calculate the gradients:
        ///     mean_grad  --> 1 / n broadcast acorss all elements
        ///     stdev_grad --> (2.0 / (n-1)) * grad_output * (x - mean) / (std * 2)
        ///     div_grad   --> ((mean_grad * stdev) - (stdev_grad * mean)) / (stdev * stdev)
        ///     final_grad = div_grad * -1.0
        ///     
        /// @see [Notes on PyTorch implementation of std_backward](https://github.com/vishwakftw/pytorch/blob/ede9bc97c3d734f3c80f4c0c08e1fe3dc2ab0250/tools/autograd/templates/Functions.cpp#L758-L770)
        /// </remarks>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            // calculate the mean gradient: (1/n)
            m_blobMeanCapturedReturns.SetDiff(1.0 / m_nInnerNum, 0);

            // calculate the stdev gradient: (2.0 / (n-1)) * grad_output * (x - mean) / (std * 2)
            //                                                             ^^^^^^^^^^
            m_cuda.channel_fillfrom(m_blobMeanCapturedReturnsFull.count(), m_nOuterNum, 1, m_nInnerNum, m_blobMeanCapturedReturns.gpu_data, m_blobMeanCapturedReturnsFull.mutable_gpu_data, DIR.FWD);
            m_cuda.sub(m_blobMeanCapturedReturnsFull.count(), m_blobCapturedReturns.gpu_data, m_blobMeanCapturedReturnsFull.gpu_data, m_blobMeanCapturedReturnsFull.mutable_gpu_diff);

            // calculate the stdev gradient: (2.0 / (n-1)) * grad_output * (x - mean) / (std * 2)
            //                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            double dfScale = (2.0 / (m_nInnerNum - 1)) * convertD(colTop[0].GetDiff(0));
            m_blobMeanCapturedReturnsFull.scale_diff(dfScale);

            // calculate the stdev gradient: (2.0 / (n-1)) * grad_output * (x - mean) / (std * 2)
            //                                                                        ^^^^^^^^^^^
            m_cuda.scale(m_blobStdevCapturedReturns.count(), 2.0, m_blobStdevCapturedReturns.gpu_data, m_blobStdevCapturedReturns.mutable_gpu_diff);
            m_cuda.channel_fillfrom(m_blobStdevCapturedReturnsFull.count(), m_nOuterNum, 1, m_nInnerNum, m_blobStdevCapturedReturns.gpu_data, m_blobStdevCapturedReturnsFull.mutable_gpu_data, DIR.FWD);
            m_cuda.channel_fillfrom(m_blobStdevCapturedReturnsFull.count(), m_nOuterNum, 1, m_nInnerNum, m_blobStdevCapturedReturns.gpu_diff, m_blobStdevCapturedReturnsFull.mutable_gpu_diff, DIR.FWD);

            m_cuda.div(m_blobMeanCapturedReturnsFull.count(), m_blobMeanCapturedReturnsFull.gpu_diff, m_blobStdevCapturedReturnsFull.gpu_diff, m_blobStdevCapturedReturnsFull.mutable_gpu_diff);

            // calculate the div gradient, Apply the quotient rule for mean/stdev
            // ((mean_grad * stdev) - (stdev_grad * mean)) / (stdev * stdev)
            m_cuda.channel_fillfrom(m_blobMeanCapturedReturnsFull.count(), m_nOuterNum, 1, m_nInnerNum, m_blobMeanCapturedReturns.gpu_diff, m_blobMeanCapturedReturnsFull.mutable_gpu_diff, DIR.FWD);
            m_cuda.mul(m_blobMeanCapturedReturnsFull.count(), m_blobMeanCapturedReturnsFull.gpu_diff, m_blobStdevCapturedReturnsFull.gpu_data, m_blobMeanCapturedReturnsFull.mutable_gpu_diff);
            m_cuda.mul(m_blobStdevCapturedReturnsFull.count(), m_blobMeanCapturedReturnsFull.gpu_data, m_blobStdevCapturedReturnsFull.gpu_diff, m_blobStdevCapturedReturnsFull.mutable_gpu_diff);
            m_cuda.sub(colBottom[0].count(), m_blobMeanCapturedReturnsFull.gpu_diff, m_blobStdevCapturedReturnsFull.gpu_diff, colBottom[0].mutable_gpu_diff);
            m_cuda.powx(m_blobStdevCapturedReturnsFull.count(), m_blobStdevCapturedReturnsFull.gpu_data, 2.0, m_blobStdevCapturedReturnsFull.mutable_gpu_data);
            m_cuda.div(colBottom[0].count(), colBottom[0].gpu_diff, m_blobStdevCapturedReturnsFull.gpu_data, colBottom[0].mutable_gpu_diff);

            m_cuda.scal(colBottom[0].count(), -1.0, colBottom[0].gpu_diff);
        }
    }
}
