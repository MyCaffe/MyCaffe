using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The AccuracyMapeLayer computes the regression accuracy using the MAPE formula:
    ///     @f$
    ///         MAPE accuracy = \frac{1}{N} \sum\limits_{n=1}^N \left| \frac{p_n - t_n}{t_n} \right| \times 100
    ///         SMAPE accuracy = \frac{1}{N} \sum\limits_{n=1}^N \left (|frac{p_n - t_n|}{(|p_n| + |t_n|)/2} \right) \times 100
    ///     @f$
    /// </summary>
    /// <remarks>
    /// @see [A better measure of relative prediction accuracy for model selection and model estimation](https://arxiv.org/abs/2105.05249) by Chris Tofallis, 2021.
    /// @see [Mean Absolute Percentage Error for regression models](https://arxiv.org/abs/2105.05249) by Arnaud De Myttenaere (Viadeo, SAMM), Boris Golden (Viadeo), Bénédicte Le Grand (CRI), and Fabrice Rossi (SAMM), 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class AccuracyMapeLayer<T> : Layer<T>
    {
        int m_nLabelAxis;
        int m_nOuterNum;
        int m_nInnerNum;
        Blob<T> m_blobWork = null;
        Blob<T> m_blobWork2 = null;
        AccuracyMapeParameter.MAPE_ALGORITHM m_alg = AccuracyMapeParameter.MAPE_ALGORITHM.MAPE;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        public AccuracyMapeLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.ACCURACY_MAPE;
            m_alg = p.accuracy_mape_param.algorithm;

            m_blobWork = new Blob<T>(cuda, log);
            m_blobWork.Name = layer_param.name + ".work";

            if (m_alg == AccuracyMapeParameter.MAPE_ALGORITHM.SMAPE)
            {
                m_blobWork2 = new Blob<T>(cuda, log);
                m_blobWork2.Name = layer_param.name + ".work2";
            }
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobWork);
            dispose(ref m_blobWork2);
            base.dispose();
        }

        /// <summary>
        /// Returns the number of bottom blobs used: predicted, label
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the minimum number of top blobs: accuracy
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of top blobs: accuracy, labels
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
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nLabelAxis = colBottom[0].CanonicalAxisIndex(m_param.accuracy_param.axis);
            m_nOuterNum = colBottom[0].count(0, m_nLabelAxis);
            m_nInnerNum = colBottom[0].count(m_nLabelAxis + 1);
            int nLabelDim = m_nOuterNum * m_nInnerNum;

            if (m_param.accuracy_param.axis != 0 || nLabelDim != 1)
                m_log.CHECK_EQ(m_nOuterNum * m_nInnerNum, colBottom[1].count(), "Number of labels must match number of predictions; e.g., if label axis = 1 and prediction shape is (N, C, H, W), label count (number of labels) must be N*H*W, with integer values in {0, 1, ..., C=1}.");

            m_blobWork.ReshapeLike(colBottom[0]);
            if (m_blobWork2 != null)
                m_blobWork2.ReshapeLike(colBottom[0]);

            List<int> rgTopShape = new List<int>(); // Accuracy is a scalar; 0 axes.
            colTop[0].Reshape(rgTopShape);
            colTop[0].blob_type = BLOB_TYPE.ACCURACY;
        }

        /// <summary>
        /// Forward compuation.
        /// </summary>
        /// <param name="colBottom">bottom input blob (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$, a blob with values in
        ///     @f$ [-\infty, +\infty] @f$ indicating the predicted score of each item.
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels l, an integer-valued blob with ground truth score values
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed accuracy: @f$
        ///         accuracy = \frac{1}{N} \sum\limits_{n=1}^N \left| \frac{p_n - t_n}{t_n} \right|
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            float fAccuracy = 0;
            float fAcc = 0;

            // Calculate Symetric Mean Absolute Percentage Error
            // - SMAPE adjusts for the size of actual values and can handle zero values better.
            if (m_alg == AccuracyMapeParameter.MAPE_ALGORITHM.SMAPE)
            {
                // N = Calculate |y(i) - yhat(i)|, where y(i) is the target and yhat(i) is the predicted.
                m_cuda.sub(colBottom[0].count(), colBottom[0].gpu_data, colBottom[1].gpu_data, m_blobWork.mutable_gpu_data);
                m_cuda.abs(m_blobWork.count(), m_blobWork.gpu_data, m_blobWork.mutable_gpu_data);
                // D = Calculate (|y(i)| + |yhat(i)|)/2
                m_cuda.abs(colBottom[0].count(), colBottom[0].gpu_data, m_blobWork2.mutable_gpu_data);
                m_cuda.abs(colBottom[1].count(), colBottom[1].gpu_data, m_blobWork2.mutable_gpu_diff);
                m_cuda.add(m_blobWork2.count(), m_blobWork2.gpu_data, m_blobWork2.gpu_diff, m_blobWork2.mutable_gpu_data);
                m_cuda.mul_scalar(m_blobWork2.count(), 0.5, m_blobWork2.mutable_gpu_data);
                // Calculate N/D
                m_cuda.div(m_blobWork.count(), m_blobWork.gpu_data, m_blobWork2.gpu_data, m_blobWork.mutable_gpu_data);
                m_cuda.denan(m_blobWork.count(), m_blobWork.mutable_gpu_data, 0);
                // Sum all values
                m_blobWork.SetDiff(0);
                m_cuda.channel_sum(m_blobWork.count(), 1, 1, m_blobWork.count(), m_blobWork.gpu_data, m_blobWork.mutable_gpu_diff);
                fAcc = convertF(m_blobWork.GetDiff(0));
            }
            else
            {
                // N = Calculate |y(i) - yhat(i)|, where y(i) is the target and yhat(i) is the predicted.
                m_cuda.sub(colBottom[0].count(), colBottom[0].gpu_data, colBottom[1].gpu_data, m_blobWork.mutable_gpu_data);
                // Divide by y(i)
                m_cuda.div(colBottom[1].count(), m_blobWork.gpu_data, colBottom[1].gpu_data, m_blobWork.mutable_gpu_data);
                // Denan, setting all nan's to 0.
                m_cuda.denan(m_blobWork.count(), m_blobWork.mutable_gpu_data, 0);
                // Sum absolute value of all values for each batch.
                fAcc = m_cuda.asum_float(m_blobWork.count(), m_blobWork.gpu_data);
            }

            fAccuracy = fAcc / m_blobWork.count();

            colTop[0].SetData(fAccuracy, 0);
        }

        /// @brief Not implemented -- AccuracyMapeLayer cannot be used as a loss.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
                throw new NotImplementedException();
        }
    }
}
