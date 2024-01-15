using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.tft
{
    /// <summary>
    /// The SharpeAccuracyLayer implements the Sharpe Accuracy Layer used in TFT models.
    /// </summary>
    /// <remarks>
    /// The accuracy returns the actual sharpe ratio value for the given prediction and target.
    /// 
    /// @see [Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture](https://arxiv.org/abs/2112.08534) by Kieran Wood, Sven Giegerich, Stephen Roberts, and Stefan Zohren, 2022, arXiv:2112.08534
    /// @see [Github - kieranjwood/trading-momentum-transformer](https://github.com/kieranjwood/trading-momentum-transformerh) by Kieran Wood, 2022.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SharpeAccuracyLayer<T> : Layer<T>
    {
        Blob<T> m_blobPredictedPos;
        Blob<T> m_blobCapturedReturns;
        Blob<T> m_blobMeanCapturedReturns;
        Blob<T> m_blobStdevCapturedReturns;
        Blob<T> m_blobCapturedReturnsSum;
        T m_tLarge;
        List<double> m_rgValues = new List<double>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public SharpeAccuracyLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SHARPE_ACCURACY;
            m_tLarge = (T)Convert.ChangeType(1e9, typeof(T));

            m_blobPredictedPos = new Blob<T>(cuda, log);
            m_blobPredictedPos.Name = m_param.name + ".predicted_pos";
            m_blobCapturedReturns = new Blob<T>(cuda, log);
            m_blobCapturedReturns.Name = m_param.name + ".captured_returns";
            m_blobMeanCapturedReturns = new Blob<T>(cuda, log);
            m_blobMeanCapturedReturns.Name = m_param.name + ".mean_captured_returns";
            m_blobStdevCapturedReturns = new Blob<T>(cuda, log);
            m_blobStdevCapturedReturns.Name = m_param.name + ".stdev_captured_returns";
            m_blobCapturedReturnsSum = new Blob<T>(cuda, log);
            m_blobCapturedReturnsSum.Name = m_param.name + ".mean_captured_returns.sum";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobPredictedPos);
            dispose(ref m_blobCapturedReturns);
            dispose(ref m_blobMeanCapturedReturns);
            dispose(ref m_blobStdevCapturedReturns);
            dispose(ref m_blobCapturedReturnsSum);

            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: x, target
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: one per accuracy range, each containing an accuracy % value for the accuracy range.
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nN = colBottom[0].num;
            int nC = colBottom[0].channels;
            m_log.CHECK_EQ(colBottom[0].height, 1, "Currently, the Sharpe Accuracy Layer only supports 1 position prediction.");
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_blobPredictedPos.ReshapeLike(colBottom[0]);
            m_blobCapturedReturns.ReshapeLike(colBottom[0]);

            if (m_param.sharpe_accuracy_param.accuracy_type == param.tft.SharpeAccuracyParameter.ACCURACY_TYPE.SHARPE)
            {
                m_blobMeanCapturedReturns.Reshape(colBottom[0].channels, 1, 1, 1);
                m_blobStdevCapturedReturns.Reshape(colBottom[0].channels, 1, 1, 1);
                m_blobCapturedReturnsSum.Reshape(colBottom[0].channels, 1, 1, 1);
            }

            colTop[0].Reshape(1, 1, 1, 1);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the numeric inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector)
        ///  -# @f$ (N \times C \times H \times W size) @f$
        ///     the computed outputs @f$ y @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            double dfVal = 0;
            int nN = colBottom[0].num;
            int nC = colBottom[0].channels;

            // captured_returns = weights * y_true
            m_cuda.mul(m_blobCapturedReturns.count(), colBottom[0].gpu_data, colBottom[1].gpu_data, m_blobCapturedReturns.mutable_gpu_data);
            // Add the returns of all assets in the batch.
            m_cuda.channel_sum(m_blobCapturedReturns.count(), 1, nN, nC, m_blobCapturedReturns.gpu_data, m_blobCapturedReturnsSum.mutable_gpu_data, true);

            if (m_param.sharpe_accuracy_param.accuracy_type == param.tft.SharpeAccuracyParameter.ACCURACY_TYPE.RETURNS)
            {
                double dfAnnualized = Math.Sqrt(252.0 / nC);
                double dfReturns = m_blobCapturedReturns.sum();
                dfReturns *= dfAnnualized;

                dfVal = dfReturns;
            }
            else
            {
                // mean and stdev of captured_returns
                m_cuda.channel_stdev(m_blobCapturedReturnsSum.count(), 1, 1, nC, m_blobCapturedReturnsSum.gpu_data, m_blobStdevCapturedReturns.gpu_data, m_blobMeanCapturedReturns.mutable_gpu_data, 1e-9f, true);

                double dfSum = m_blobCapturedReturnsSum.sum();
                double dfStdev = convertD(m_blobStdevCapturedReturns.GetData(0));
                double dfAnnualStdev = dfStdev * Math.Sqrt(252.0);

                double dfAnnualized = Math.Sqrt(252.0 / nC);
                double dfAnnualReturns = dfSum * dfAnnualized;

                double dfSharpe = dfAnnualReturns / dfAnnualStdev;
                dfSharpe *= dfAnnualized;

                dfVal = dfSharpe;
            }

            m_rgValues.Add(dfVal);
            while (m_rgValues.Count > m_param.sharpe_accuracy_param.averaging_periods)
            {
                m_rgValues.RemoveAt(0);
            }

            double dfAveVal = m_rgValues.Average();
            colTop[0].SetData(dfAveVal, 0);
        }

        /// @brief Not implemented -- SharpeAccuracyLayer cannot be used as a loss.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
                throw new NotImplementedException();
        }
    }
}
