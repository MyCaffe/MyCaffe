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
    /// The AccuracyRegressionLayer computes the regression accuracy using the MAPE, SMAPE or Bucketing formula:
    ///     @f$
    ///         MAPE accuracy = \frac{1}{N} \sum\limits_{n=1}^N \left| \frac{p_n - t_n}{t_n} \right| \times 100
    ///         SMAPE accuracy = \frac{1}{N} \sum\limits_{n=1}^N \left (|frac{p_n - t_n|}{(|p_n| + |t_n|)/2} \right) \times 100
    ///     @f$
    /// </summary>
    /// <remarks>
    /// @see [A better measure of relative prediction accuracy for model selection and model estimation](https://arxiv.org/abs/2105.05249) by Chris Tofallis, 2021.
    /// @see [Mean Absolute Percentage Error for regression models](https://arxiv.org/abs/2105.05249) by Arnaud De Myttenaere (Viadeo, SAMM), Boris Golden (Viadeo), Bénédicte Le Grand (CRI), and Fabrice Rossi (SAMM), 2016.
    /// @see [MAPE vs sMAPE - When to choose what?](https://medium.com/illumination/mape-vs-smape-when-to-choose-what-be51a170df16) by Thiruthuvaraj Rajasekhar, Medium, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class AccuracyRegressionLayer<T> : Layer<T>
    {
        int m_nLabelAxis;
        int m_nOuterNum;
        int m_nInnerNum;
        Blob<T> m_blobWork = null;
        Blob<T> m_blobWork2 = null;
        AccuracyRegressionParameter.ALGORITHM m_alg = AccuracyRegressionParameter.ALGORITHM.MAPE;
        RollingBucketAccuracy m_rgBucketAccuracy = null;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        public AccuracyRegressionLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.ACCURACY_REGRESSION;
            m_alg = p.accuracy_regression_param.algorithm;

            if (m_alg == AccuracyRegressionParameter.ALGORITHM.MAPE ||
                m_alg == AccuracyRegressionParameter.ALGORITHM.SMAPE)
            {
                m_blobWork = new Blob<T>(cuda, log);
                m_blobWork.Name = layer_param.name + ".work";

                if (m_alg == AccuracyRegressionParameter.ALGORITHM.SMAPE)
                {
                    m_blobWork2 = new Blob<T>(cuda, log);
                    m_blobWork2.Name = layer_param.name + ".work2";
                }
            }
            else
            {
                m_rgBucketAccuracy = new RollingBucketAccuracy(p.accuracy_regression_param.bucket_min, p.accuracy_regression_param.bucket_max, p.accuracy_regression_param.bucket_count, 100, 200);
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

            if (m_blobWork != null)
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

            // Calculate bucketing method of accuracy where bucket counts are used to determine overall accuracy.
            if (m_alg == AccuracyRegressionParameter.ALGORITHM.BUCKETING)
            {
                float[] rgPredicted = convertF(colBottom[0].mutable_cpu_data);
                float[] rgTarget = convertF(colBottom[1].mutable_cpu_data);
                m_rgBucketAccuracy.Add(rgPredicted, rgTarget);
                fAccuracy = (float)m_rgBucketAccuracy.CalculateAccuracy();
            }
            // Calculate Symetric Mean Absolute Percentage Error
            // - SMAPE adjusts for the size of actual values and can handle zero values better.
            else if (m_alg == AccuracyRegressionParameter.ALGORITHM.SMAPE)
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
                m_cuda.channel_sumEx(m_blobWork.count(), 1, 1, m_blobWork.count(), m_blobWork.gpu_data, m_blobWork.mutable_gpu_diff, false, DIR.FWD);
                fAcc = convertF(m_blobWork.GetDiff(0));
            }
            // Calculate Mean Absolute Percentage Error
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

    class RollingBucketAccuracy
    {
        double m_dfMin;
        double m_dfMax;
        int m_nCount;
        int m_nMinIterations;
        int m_nMaxIterations;
        List<BucketAccuracy> m_rgItems = new List<BucketAccuracy>();

        public RollingBucketAccuracy(double dfMin, double dfMax, int nCount, int nMinIterations, int nMaxIterations)
        {
            m_dfMin = dfMin;
            m_dfMax = dfMax;
            m_nCount = nCount;
            m_nMinIterations = nMinIterations;
            m_nMaxIterations = nMaxIterations;

            m_rgItems.Add(new BucketAccuracy(dfMin, dfMax, nCount, nMinIterations, nMaxIterations));
        }

        public void Add(float[] rgPred, float[] rgTgt)
        {
            bool? bRes = m_rgItems[0].Add(rgPred, rgTgt);
            if (bRes.GetValueOrDefault(false))
                m_rgItems.RemoveAt(0);

            BucketAccuracy b = new BucketAccuracy(m_dfMin, m_dfMax, m_nCount, m_nMinIterations, m_nMaxIterations);
            m_rgItems.Add(b);

            foreach (BucketAccuracy b1 in m_rgItems)
            {
                b.Add(rgPred, rgTgt);
            }
        }

        public double CalculateAccuracy()
        {
            if (m_rgItems.Count == 0)
                return 0;

            return m_rgItems[0].CalculateAccuracy();
        }
    }

    class BucketAccuracy
    {
        BucketCollection m_colPredPos = null;
        BucketCollection m_colPredNeg = null;
        BucketCollection m_colTgtPos = null;
        BucketCollection m_colTgtNeg = null;
        int m_nMinIterations = 0;
        int m_nMaxIterations = 0;
        int m_nIterations = 0;

        public BucketAccuracy(double dfMin, double dfMax, int nCount, int nMinIterations, int nMaxIterations)
        {
            m_nMinIterations = nMinIterations;
            m_nMaxIterations = nMaxIterations;

            int nBucketCount = nCount;
            if (dfMin < 0)
            {
                nBucketCount /= 2;
                m_colPredNeg = new BucketCollection(dfMin, 0, nBucketCount);
                m_colTgtNeg = new BucketCollection(dfMin, 0, nBucketCount);
            }

            m_colPredPos = new BucketCollection(0, dfMax, nBucketCount);
            m_colTgtPos = new BucketCollection(0, dfMax, nBucketCount);
        }

        public bool? Add(float[] rgPred, float[] rgTgt)
        {
            for (int i = 0; i < rgPred.Length; i++)
            {
                if (rgPred[i] < 0 && m_colPredNeg != null)
                    m_colPredNeg.Add(rgPred[i]);
                else
                    m_colPredPos.Add(rgPred[i]);

                if (rgTgt[i] < 0 && m_colTgtNeg != null)
                    m_colTgtNeg.Add(rgTgt[i]);
                else
                    m_colTgtPos.Add(rgTgt[i]);
            }

            m_nIterations++;

            if (m_nIterations < m_nMinIterations)
                return null;

            if (m_nIterations > m_nMaxIterations)
                return true;

            return false;
        }

        public double CalculateAccuracy()
        {
            if (m_nIterations < m_nMinIterations)
                return 0;

            int nTotalError = 0;
            int nTotalGroundTruth = 0;

            if (m_colPredNeg != null)
            {
                for (int i = 0; i < m_colPredNeg.Count; i++)
                {
                    int nGroundTruthCount = m_colTgtNeg[i].Count;
                    int nPredictedCount = m_colPredNeg[i].Count;

                    nTotalError += (Math.Abs(nPredictedCount - nGroundTruthCount));
                    nTotalGroundTruth += nGroundTruthCount;
                }
            }

            for (int i = 0; i < m_colPredPos.Count; i++)
            {
                int nGroundTruthCount = m_colTgtPos[i].Count;
                int nPredictedCount = m_colPredPos[i].Count;

                nTotalError += (Math.Abs(nPredictedCount - nGroundTruthCount));
                nTotalGroundTruth += nGroundTruthCount;
            }

            double dfAccuracy = (1 - (nTotalError / (double)nTotalGroundTruth));
            return dfAccuracy;
        }
    }
}
