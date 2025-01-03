using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.common;
using MyCaffe.param;
using static MyCaffe.basecode.ConfusionMatrixStats;
using static MyCaffe.param.beta.DecodeParameter;

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
    public class AccuracyRegressionLayer<T> : Layer<T>, IXAccuracyTest
    {
        int m_nLabelAxis;
        int m_nOuterNum;
        int m_nInnerNum;
        Blob<T> m_blobWork = null;
        Blob<T> m_blobWork2 = null;
        AccuracyRegressionParameter.ALGORITHM m_alg = AccuracyRegressionParameter.ALGORITHM.MAPE;
        BucketAccuracy m_trainingAccuracy = null;
        BucketAccuracy m_testingAccuracy1 = null;
        BucketAccuracy m_testingAccuracy2 = null;
        BucketCollection m_colLabel = null;
        ZScoreLayer<T> m_zscore = null;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">Specifies the layer parameter.</param>
        /// <param name="db">Specifies the database.</param>
        public AccuracyRegressionLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXDatabaseBase db)
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
                if (p.accuracy_regression_param.bucket_count <= 1)
                    throw new Exception("The accuracy regression bucket count must be > 1.");

                if (p.z_score_param != null && p.z_score_param.enabled)
                {
                    LayerParameter pzs = new LayerParameter(LayerParameter.LayerType.Z_SCORE);
                    pzs.z_score_param = p.z_score_param;
                    m_zscore = Layer<T>.Create(cuda, log, pzs, null, db) as ZScoreLayer<T>;

                    SourceDescriptor src = db.GetSourceByName(p.z_score_param.source);
                    if (src != null)
                    {
                        int nPos = src.Name.IndexOf('.');
                        string strDs = src.Name.Substring(0, nPos);
                        DatasetDescriptor ds = db.GetDatasetByName(strDs);

                        if (ds != null && m_param.accuracy_regression_param.enable_override)
                        {
                            ParameterDescriptor p1 = ds.Parameters.Find("LabelBucketConfig");
                            if (p1 != null && !string.IsNullOrEmpty(p1.Value))
                                m_colLabel = new BucketCollection(p1.Value);
                        }
                    }

                    if (m_colLabel == null || m_colLabel.Count == 0)
                    {
                        if (m_param.accuracy_regression_param.enable_override)
                            m_log.WriteLine("WARNING: The source '" + p.z_score_param.source + "' was not found, the ZScoreLayer will not be used.");
                    }
                    else
                    {
                        // Normalize the label buckets.
                        foreach (Bucket b in m_colLabel)
                        {
                            float fMin = m_zscore.Normalize((float)b.Minimum);
                            if (b.Minimum == 0)
                                fMin = 0;

                            float fMax = m_zscore.Normalize((float)b.Maximum);
                            if (b.Maximum == 0)
                                fMax = 0;

                            b.Update(fMin, fMax);
                        }
                    }
                }

                if (m_colLabel != null)
                    m_log.WriteLine("INFO: Using Label Bucket Collection from 'LabelBucketConfig' dataset parameter instead of accuraccy regression parameters.");

                m_trainingAccuracy = new BucketAccuracy(m_colLabel, layer_param.accuracy_regression_param.bucket_min, layer_param.accuracy_regression_param.bucket_max, layer_param.accuracy_regression_param.bucket_count, 100, 1000, layer_param.accuracy_regression_param.bucket_ignore_min, layer_param.accuracy_regression_param.bucket_ignore_max);
                m_testingAccuracy1 = new BucketAccuracy(m_colLabel, p.accuracy_regression_param.bucket_min, p.accuracy_regression_param.bucket_max, p.accuracy_regression_param.bucket_count, 100, 1000, p.accuracy_regression_param.bucket_ignore_min, p.accuracy_regression_param.bucket_ignore_max);
                m_testingAccuracy2 = new BucketAccuracy(m_colLabel, p.accuracy_regression_param.bucket_min, p.accuracy_regression_param.bucket_max, p.accuracy_regression_param.bucket_count, 100, 1000, p.accuracy_regression_param.bucket_ignore_min, p.accuracy_regression_param.bucket_ignore_max);
            }
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobWork);
            dispose(ref m_blobWork2);

            if (m_zscore != null)
            {
                m_zscore.Dispose();
                m_zscore = null;
            }

            base.dispose();
        }

        /// <summary>
        /// Reset the testing run.
        /// </summary>
        /// <exception cref="Exception">Exceptions are thrown when not using the BUCKETING algorithm.</exception>
        public void ResetTesting()
        {
            if (layer_param.accuracy_regression_param.algorithm != AccuracyRegressionParameter.ALGORITHM.BUCKETING)
                throw new Exception("The 'ResetTesting' is only supported when using the 'BUCKETING' algorithm.");

            m_trainingAccuracy = new BucketAccuracy(m_colLabel, layer_param.accuracy_regression_param.bucket_min, layer_param.accuracy_regression_param.bucket_max, layer_param.accuracy_regression_param.bucket_count, 100, 1000, layer_param.accuracy_regression_param.bucket_ignore_min, layer_param.accuracy_regression_param.bucket_ignore_max);
            m_testingAccuracy1 = new BucketAccuracy(m_colLabel, layer_param.accuracy_regression_param.bucket_min, layer_param.accuracy_regression_param.bucket_max, layer_param.accuracy_regression_param.bucket_count, 100, 1000, layer_param.accuracy_regression_param.bucket_ignore_min, layer_param.accuracy_regression_param.bucket_ignore_max);
            m_testingAccuracy2 = null;
        }

        /// <summary>
        /// Add new values to the testing accuracy buckets.
        /// </summary>
        /// <param name="fPredicted">Specifies the predicted value.</param>
        /// <param name="fGroundTruth">Specifies the ground truth target value.</param>
        /// <param name="fGroundTruth2">Optional, secondary ground truth value.</param>
        /// <param name="bNormalize">Optionally, specifies to normalize the ground truth (default = true).</param>
        public void AddTesting(float fPredicted, float fGroundTruth, float? fGroundTruth2 = null, bool bNormalize = true)
        {
            if (m_testingAccuracy1 != null)
            {
                if (m_zscore != null && bNormalize)
                    fGroundTruth = m_zscore.Normalize(fGroundTruth);

                m_testingAccuracy1.Add(new float[] { fPredicted }, new float[] { fGroundTruth });

                if (fGroundTruth2.HasValue)
                {
                    if (m_zscore != null && bNormalize)
                        fGroundTruth2 = m_zscore.Normalize(fGroundTruth2.Value);

                    if (m_testingAccuracy2 == null)
                        m_testingAccuracy2 = new BucketAccuracy(m_colLabel, layer_param.accuracy_regression_param.bucket_min, layer_param.accuracy_regression_param.bucket_max, layer_param.accuracy_regression_param.bucket_count, 100, 1000, layer_param.accuracy_regression_param.bucket_ignore_min, layer_param.accuracy_regression_param.bucket_ignore_max);
                    m_testingAccuracy2.Add(new float[] { fPredicted }, new float[] { fGroundTruth2.Value });
                }
            }
        }

        /// <summary>
        /// Calculate the accuracy using the testing accuracy bucket collection.
        /// </summary>
        /// <param name="bGetDetails">Specifies to retrieve the details.</param>
        /// <param name="strDetails">Specifies details on the testing.</param>
        /// <returns>The accuracy value is returned.</returns>
        public double CalculateTestingAccuracy(bool bGetDetails, out string strDetails)
        {
            strDetails = "";

            if (m_testingAccuracy1 == null)
                return 0;

            double dfAccuracy = m_testingAccuracy1.CalculateAccuracy(bGetDetails, out strDetails);

            if (m_testingAccuracy2 != null)
            {
                string strDetails2;
                double dfAccuracy2 = m_testingAccuracy2.CalculateAccuracy(bGetDetails, out strDetails2);

                strDetails += Environment.NewLine;
                strDetails += "Secondary Ground Truth" + Environment.NewLine;
                strDetails += "Accuracy = " + dfAccuracy2.ToString("P3");
                strDetails += Environment.NewLine;
                strDetails += strDetails2;
            }

            return dfAccuracy;
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
                m_trainingAccuracy.Add(rgPredicted, rgTarget);
                fAccuracy = (float)m_trainingAccuracy.CalculateAccuracy();
            }
            else
            {
                // Calculate Symetric Mean Absolute Percentage Error
                // - SMAPE adjusts for the size of actual values and can handle zero values better.
                if (m_alg == AccuracyRegressionParameter.ALGORITHM.SMAPE)
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
            }

            colTop[0].SetData(fAccuracy, 0);
        }

        /// @brief Not implemented -- AccuracyMapeLayer cannot be used as a loss.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
                throw new NotImplementedException();
        }
    }

    /// <summary>
    /// The RollingBucketAccuracy provides rolling accuracy.
    /// </summary>
    public class BucketAccuracy
    {
        BucketCollection m_colCorrect;
        BucketCollection m_colTotal;
        List<Bucket> m_rgBucketTotal = new List<Bucket>();
        List<Bucket> m_rgBucketCorrect = new List<Bucket>();
        int m_nMax = 0;
        int m_nMin = 0;
        int m_nIteration = 0;
        double? m_dfMinIgnore = null;
        double? m_dfMaxIgnore = null;


        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="colOverride">Optional bucket collection override that when specified is used instead of the other fixed buckets.</param>
        /// <param name="dfMin">Specifies the minimum value.</param>
        /// <param name="dfMax">Specifies the maximum value.</param>
        /// <param name="nCount">Specfies the number of buckets.</param>
        /// <param name="nMinIterations">Specifies the minimum number of iterations.</param>
        /// <param name="nMaxIterations">Specifies the maximum number of iterations.</param>
        /// <param name="dfIgnoreMin">Specifies the minimum ignore value.</param>
        /// <param name="dfIgnoreMax">Specifies the maximum ignore value.</param>
        public BucketAccuracy(BucketCollection colOverride, double dfMin, double dfMax, int nCount, int nMinIterations, int nMaxIterations, double? dfIgnoreMin, double? dfIgnoreMax)
        {
            m_nMin = nMinIterations;
            m_nMax = nMaxIterations;
            m_dfMinIgnore = dfIgnoreMin;
            m_dfMaxIgnore = dfIgnoreMax;

            if (colOverride != null)
            {
                string strConfig = colOverride.ToConfigString();
                m_colCorrect = new BucketCollection(strConfig);
                m_colTotal = new BucketCollection(strConfig);
            }
            else
            {
                m_colCorrect = new BucketCollection(dfMin, dfMax, nCount);
                m_colTotal = new BucketCollection(dfMin, dfMax, nCount);
            }
        }

        /// <summary>
        /// Add a set of predictions and target values.
        /// </summary>
        /// <param name="rgPred">Specifies the predictions.</param>
        /// <param name="rgTgt">Specifies the targets.</param>
        public void Add(float[] rgPred, float[] rgTgt)
        {
            for (int i = 0; i < rgTgt.Length; i++)
            {
                int nIdxTgt = m_colTotal.Add(rgTgt[i]);
                int nIdxPred = m_colCorrect.Add(rgPred[i], true);

                if (nIdxTgt == nIdxPred)
                {
                    m_colCorrect.Add(rgPred[i]);
                    m_rgBucketCorrect.Add(m_colCorrect.GetBucketAt(nIdxPred));
                }
                else
                {
                    m_rgBucketCorrect.Add(null);
                }

                m_rgBucketTotal.Add(m_colTotal.GetBucketAt(nIdxTgt));

                if (m_rgBucketTotal.Count > m_nMax)
                {
                    Bucket b = m_rgBucketTotal[0];
                    m_rgBucketTotal.RemoveAt(0);

                    b.Count--;

                    b = m_rgBucketCorrect[0];
                    if (b != null)
                        b.Count--;
                    m_rgBucketCorrect.RemoveAt(0);
                }

                m_nIteration++;
            }
        }

        /// <summary>
        /// Calculates the accuracy.
        /// </summary>
        /// <returns>The accuracy value is returned.</returns>
        public double CalculateAccuracy()
        {
            if (m_nIteration < m_nMin)
                return 0;

            int nCorrect = 0;
            int nTotal = 0;

            for (int i = 0; i < m_rgBucketTotal.Count; i++)
            {
                nTotal += m_rgBucketTotal[i].Count;
                nCorrect += (m_rgBucketCorrect[i] == null) ? 0 : m_rgBucketCorrect[i].Count;
            }

            return (nTotal == 0) ? 0 : (double)nCorrect / nTotal;
        }

        /// <summary>
        /// Calculates the accuracy.
        /// </summary>
        /// <returns>The accuracy value is returned.</returns>
        public double CalculateAccuracy(bool bGetDetails, out string strDetails)
        {
            strDetails = "";

            if (m_nIteration < m_nMin)
                return 0;

            int nCorrect = 0;
            int nTotal = 0;

            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < m_rgBucketTotal.Count; i++)
            {
                sb.Append(m_rgBucketTotal[0].ToString());
                double dfAcc = (m_rgBucketCorrect[i].Count == 0) ? 0 : (double)m_rgBucketCorrect[i].Count / m_rgBucketTotal[i].Count;
                sb.Append(" = ");
                sb.Append(dfAcc.ToString("P"));
                sb.AppendLine();

                nTotal += m_rgBucketTotal[i].Count;
                nCorrect += m_rgBucketCorrect[i].Count;
            }

            double dfTotalAcc = (nTotal == 0) ? 0 : (double)nCorrect / nTotal;

            sb.Append("Total Accuracy = ");
            sb.AppendLine(dfTotalAcc.ToString("P"));
            strDetails = sb.ToString();

            return dfTotalAcc;
        }
    }
}
