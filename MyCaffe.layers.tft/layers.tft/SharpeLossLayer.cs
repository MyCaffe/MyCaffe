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
        Blob<T> m_blobCapturedReturns1;
        Blob<T> m_blobWeightReturns1;
        double m_dfMeanReturns;
        double m_dfMeanCapturedReturnsSqMinusMeanReturnsSq;
        double m_dfMeanCapturedReturnsSqMinusMeanReturnsSqSqrt;

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
            m_blobCapturedReturnsSq.Name = m_param.name + ".captured_returns.sq";
            m_blobMeanCapturedReturns = new Blob<T>(cuda, log);
            m_blobMeanCapturedReturns.Name = m_param.name + ".mean.captured_returns";
            m_blobCapturedReturns1 = new Blob<T>(cuda, log);
            m_blobCapturedReturns1.Name = m_param.name + ".stdev";
            m_blobWeightReturns1 = new Blob<T>(cuda, log);
            m_blobWeightReturns1.Name = m_param.name + ".stdev.full";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobCapturedReturns);
            dispose(ref m_blobCapturedReturnsSq);
            dispose(ref m_blobMeanCapturedReturns);
            dispose(ref m_blobCapturedReturns1);
            dispose(ref m_blobWeightReturns1);

            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobCapturedReturns);
            col.Add(m_blobCapturedReturnsSq);
            col.Add(m_blobMeanCapturedReturns);
            col.Add(m_blobCapturedReturns1);
            col.Add(m_blobWeightReturns1);
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
            m_blobWeightReturns1.ReshapeLike(colBottom[0]);
            m_blobCapturedReturns1.ReshapeLike(colBottom[0]);
            m_blobMeanCapturedReturns.ReshapeLike(colBottom[0]);

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
            // captured_returns_sq = captured_returns * captured_returns
            m_cuda.powx(m_blobCapturedReturns.count(), m_blobCapturedReturns.gpu_data, 2.0, m_blobCapturedReturnsSq.mutable_gpu_data);

            // mean returns = torch.mean(captured_returns)
            m_cuda.channel_mean(m_blobCapturedReturns.count(), 1, 1, m_blobCapturedReturns.count(), m_blobCapturedReturns.gpu_data, m_blobMeanCapturedReturns.mutable_gpu_data);
            double dfMeanReturns = convertD(m_blobMeanCapturedReturns.GetData(0));
            double dfMeanReturnsSq = dfMeanReturns * dfMeanReturns;

            // mean captured_returns_sq = torch.mean(captured_returns_sq)
            m_cuda.channel_mean(m_blobCapturedReturnsSq.count(), 1, 1, m_blobCapturedReturnsSq.count(), m_blobCapturedReturnsSq.gpu_data, m_blobMeanCapturedReturns.mutable_gpu_data);
            double dfMeanCapturedReturnsSq = convertD(m_blobMeanCapturedReturns.GetData(0));
            double dfMeanCapturedReturnsSqMinusMeanReturnsSq = dfMeanCapturedReturnsSq - dfMeanReturnsSq + 1e-9;
            double dfMeanCapturedReturnsSqMinusMeanReturnsSqSqrt = Math.Sqrt(dfMeanCapturedReturnsSqMinusMeanReturnsSq);

            // E = -mean_returns / Math.Sqrt(mean_captured_returns_sq - mean_returns_sq + 1e-9)
            double dfLoss = (dfMeanReturns / dfMeanCapturedReturnsSqMinusMeanReturnsSqSqrt);

            dfLoss *= Math.Sqrt(252);

            dfLoss *= -1;

            // Set the loss output
            colTop[0].SetData(dfLoss, 0);
            m_dfMeanReturns = dfMeanReturns;
            m_dfMeanCapturedReturnsSqMinusMeanReturnsSq = dfMeanCapturedReturnsSqMinusMeanReturnsSq;
            m_dfMeanCapturedReturnsSqMinusMeanReturnsSqSqrt = dfMeanCapturedReturnsSqMinusMeanReturnsSqSqrt;
        }

        /// <summary>
        /// Computes the SharpeLoss error gradient w.r.t. the inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$ 
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">inpub Blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$ predicted positions
        ///  -# @f$ (N \times C \times H \times W) @f$ y_true actual returns
        /// </param>
        /// <remarks>
        /// @see [Notes on PyTorch implementation of std_backward](https://github.com/vishwakftw/pytorch/blob/ede9bc97c3d734f3c80f4c0c08e1fe3dc2ab0250/tools/autograd/templates/Functions.cpp#L758-L770)
        /// </remarks>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;
            
            // Get the grad input
            double dfLossGrad = -1 * convertD(colTop[0].GetDiff(0)) * Math.Sqrt(252);

            double dfMeanCapturedReturnsSqMinusMeanReturnsSqSqrtGrad = -1 * m_dfMeanReturns  / Math.Pow(m_dfMeanCapturedReturnsSqMinusMeanReturnsSqSqrt, 2) * dfLossGrad;
            double dfMeanCapturedReturnsSqMinusMeanReturnsSqGrad = 0.5 * 1.0 / Math.Sqrt(m_dfMeanCapturedReturnsSqMinusMeanReturnsSq + 1e-9) * dfMeanCapturedReturnsSqMinusMeanReturnsSqSqrtGrad;
            double dfMeanCapturedReturnsSqGrad = dfMeanCapturedReturnsSqMinusMeanReturnsSqGrad;
            m_blobCapturedReturns.SetDiff(dfMeanCapturedReturnsSqGrad / m_blobCapturedReturns.count());
            double dfMeanReturnsSqGrad = -1 * dfMeanCapturedReturnsSqGrad;
            double dfMeanReturnsGrad = (2 * m_dfMeanReturns * dfMeanReturnsSqGrad) + (1 / m_dfMeanCapturedReturnsSqMinusMeanReturnsSqSqrt * dfLossGrad);

            // captured_returns_grad_1 = mean_returns_grad / captured_returns.numel()
            m_blobCapturedReturns1.SetData(dfMeanReturnsGrad / m_blobCapturedReturns1.count());
            // captured_returns_grad_2 =  2 * captured_returns * captured_returns_sq_grad
            m_cuda.mul(m_blobCapturedReturns.count(), m_blobCapturedReturns.gpu_data, m_blobCapturedReturns.gpu_diff, m_blobCapturedReturns1.mutable_gpu_diff);
            m_blobCapturedReturns1.scale_diff(2.0);
            // captured_returns_grad = captured_returns_grad_1 + captured_returns_grad_2
            //m_cuda.add(m_blobCapturedReturns.count(), m_blobCapturedReturns1.gpu_data, m_blobCapturedReturns1.gpu_diff, m_blobCapturedReturns.mutable_gpu_diff);

            // weight_returns_grad_1 = y_true * captured_returns_grad_1
            m_cuda.mul(colBottom[1].count(), colBottom[1].gpu_data, m_blobCapturedReturns1.gpu_data, m_blobWeightReturns1.mutable_gpu_data);
            // weight_returns_grad_2 = y_true * captured_returns_grad_2
            m_cuda.mul(colBottom[1].count(), colBottom[1].gpu_data, m_blobCapturedReturns1.gpu_diff, m_blobWeightReturns1.mutable_gpu_diff);
            // weight_returns_grad = weight_returns_grad_1 + weight_returns_grad_2
            m_cuda.add(colBottom[0].count(), m_blobWeightReturns1.gpu_data, m_blobWeightReturns1.gpu_diff, colBottom[0].mutable_gpu_diff);
        }
    }
}
