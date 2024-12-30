using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.common;
using MyCaffe.param;
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
        RollingBucketAccuracy m_rgBucketAccuracy = null;
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

                m_rgBucketAccuracy = new RollingBucketAccuracy(m_colLabel, p.accuracy_regression_param.bucket_min, p.accuracy_regression_param.bucket_max, p.accuracy_regression_param.bucket_count, 100, 200, p.accuracy_regression_param.bucket_ignore_min, p.accuracy_regression_param.bucket_ignore_max);
                m_testingAccuracy1 = new BucketAccuracy(m_colLabel, p.accuracy_regression_param.bucket_min, p.accuracy_regression_param.bucket_max, p.accuracy_regression_param.bucket_count, p.accuracy_regression_param.bucket_ignore_min, p.accuracy_regression_param.bucket_ignore_max);
                m_testingAccuracy2 = new BucketAccuracy(m_colLabel, p.accuracy_regression_param.bucket_min, p.accuracy_regression_param.bucket_max, p.accuracy_regression_param.bucket_count, p.accuracy_regression_param.bucket_ignore_min, p.accuracy_regression_param.bucket_ignore_max);
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

            m_testingAccuracy1 = new BucketAccuracy(m_colLabel, layer_param.accuracy_regression_param.bucket_min, layer_param.accuracy_regression_param.bucket_max, layer_param.accuracy_regression_param.bucket_count, layer_param.accuracy_regression_param.bucket_ignore_min, layer_param.accuracy_regression_param.bucket_ignore_max);
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
                        m_testingAccuracy2 = new BucketAccuracy(m_colLabel, layer_param.accuracy_regression_param.bucket_min, layer_param.accuracy_regression_param.bucket_max, layer_param.accuracy_regression_param.bucket_count, layer_param.accuracy_regression_param.bucket_ignore_min, layer_param.accuracy_regression_param.bucket_ignore_max);
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
                m_rgBucketAccuracy.Add(rgPredicted, rgTarget);
                fAccuracy = (float)m_rgBucketAccuracy.CalculateAccuracy();
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
    public class RollingBucketAccuracy
    {
        double m_dfMin;
        double m_dfMax;
        int m_nCount;
        int m_nIteration = 0;
        int m_nMinIterations;
        int m_nMaxIterations;
        double? m_dfIgnoreMax;
        double? m_dfIgnoreMin;
        List<BucketAccuracy> m_rgItems = new List<BucketAccuracy>();
        BucketCollection m_colOverride = null;

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
        public RollingBucketAccuracy(BucketCollection colOverride, double dfMin, double dfMax, int nCount, int nMinIterations, int nMaxIterations, double? dfIgnoreMin, double? dfIgnoreMax)
        {
            m_colOverride = colOverride;
            m_dfMin = dfMin;
            m_dfMax = dfMax;
            m_nCount = nCount;
            m_nIteration = 0;
            m_nMinIterations = nMinIterations;
            m_nMaxIterations = nMaxIterations;
            m_dfIgnoreMax = dfIgnoreMax;
            m_dfIgnoreMin = dfIgnoreMin;

            m_rgItems.Add(new BucketAccuracy(colOverride, dfMin, dfMax, nCount, dfIgnoreMin, dfIgnoreMax));
        }

        /// <summary>
        /// Add a set of predictions and target values.
        /// </summary>
        /// <param name="rgPred">Specifies the predictions.</param>
        /// <param name="rgTgt">Specifies the targets.</param>
        public void Add(float[] rgPred, float[] rgTgt)
        {
            m_nIteration++;

            BucketAccuracy b = new BucketAccuracy(m_colOverride, m_dfMin, m_dfMax, m_nCount, m_dfIgnoreMax, m_dfIgnoreMin);
            m_rgItems.Add(b);

            foreach (BucketAccuracy b1 in m_rgItems)
            {
                b1.Add(rgPred, rgTgt);
            }

            if (m_nIteration >= m_nMaxIterations)
                m_rgItems.RemoveAt(0);
        }

        /// <summary>
        /// Calculates the accuracy.
        /// </summary>
        /// <returns>The accuracy value is returned.</returns>
        public double CalculateAccuracy()
        {
            if (m_rgItems.Count == 0 || m_nIteration < m_nMinIterations)
                return 0;

            string strTmp;
            return m_rgItems[0].CalculateAccuracy(false, out strTmp);
        }
    }

    /// <summary>
    /// The BucketAccuracy layer tracks the accuracy across both positive and negative bucket collections between the target and predicted values.
    /// </summary>
    public class BucketAccuracy
    {
        BucketCollection m_colPredPos = null;
        BucketCollection m_colPredNeg = null;
        BucketCollection m_colTgtPos = null;
        BucketCollection m_colTgtNeg = null;
        double? m_dfIgnoreMax = null;
        double? m_dfIgnoreMin = null;
        Dictionary<Bucket, int> m_rgBucketCorrectHits = new Dictionary<Bucket, int>();
        Dictionary<Bucket, Dictionary<int, int>> m_rgBucketIncorrectHits = new Dictionary<Bucket, Dictionary<int, int>>();
        BucketCollection m_colOverride = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="colOverride">Optional bucket collection override that when specified is used instead of the other fixed buckets.</param>
        /// <param name="dfMin">Specifies the minimum of all values.</param>
        /// <param name="dfMax">Specifies the maximum of all values.</param>
        /// <param name="nCount">Specifies the number of buckets.</param>
        /// <param name="dfIgnoreMin">Specifies to the minimum ignore range (default = -double.MaxValue).</param>
        /// <param name="dfIgnoreMax">Specifies to the maximum ignore range (default = double.MaxValue).</param>
        public BucketAccuracy(BucketCollection colOverride, double dfMin, double dfMax, int nCount, double? dfIgnoreMin, double? dfIgnoreMax)
        {
            m_colOverride = colOverride;
            m_dfIgnoreMax = dfIgnoreMax;
            m_dfIgnoreMin = dfIgnoreMin;

            if (m_colOverride != null)
            {
                string strCfg = m_colOverride.ToConfigString();
                string[] rgstr = strCfg.Split(';');
                List<string> rgstrNeg = new List<string>();
                List<string> rgstrPos = new List<string>();

                for (int i = 1; i < rgstr.Length; i++)
                {
                    if (string.IsNullOrEmpty(rgstr[i]))
                        continue;

                    string str = rgstr[i].Trim('[', ']');
                    string[] rgstr1 = str.Split(',');
                    double dfMin1 = double.Parse(rgstr1[0]);
                    double dfMax1 = double.Parse(rgstr1[1]);

                    if (dfMin1 < 0)
                        rgstrNeg.Add(rgstr[i]);
                    else
                        rgstrPos.Add(rgstr[i]);
                }

                string strNegConfig = "Count=" + rgstrNeg.Count.ToString() + ";";
                for (int i = 0; i < rgstrNeg.Count; i++)
                {
                    strNegConfig += rgstrNeg[i];
                    if (i < rgstrNeg.Count - 1)
                        strNegConfig += ";";
                }

                string strPosConfig = "Count=" + rgstrPos.Count.ToString() + ";";
                for (int i = 0; i < rgstrPos.Count; i++)
                {
                    strPosConfig += rgstrPos[i];
                    if (i < rgstrPos.Count - 1)
                        strPosConfig += ";";
                }

                m_colPredNeg = new BucketCollection(strNegConfig);
                m_colTgtNeg = new BucketCollection(strNegConfig);
                m_colPredPos = new BucketCollection(strPosConfig);
                m_colTgtPos = new BucketCollection(strPosConfig);
            }
            else
            {
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
        }

        /// <summary>
        /// Add an array of predicted and target values.
        /// </summary>
        /// <param name="rgPred">Specifies the predicted values.</param>
        /// <param name="rgTgt">Specifies the target values.</param>
        public int Add(float[] rgPred, float[] rgTgt)
        {
            int nPredIdxNeg = -1;
            int nPredIdxPos = -1;

            for (int i = 0; i < rgPred.Length; i++)
            {
                int nTgtIdxNeg = -1;
                int nTgtIdxPos = -1;
                nPredIdxNeg = -1;
                nPredIdxPos = -1;

                if (m_dfIgnoreMin.HasValue && m_dfIgnoreMax.HasValue)
                {
                    if (rgPred[i] > m_dfIgnoreMin.Value && rgPred[i] < m_dfIgnoreMax.Value)
                        continue;
                }

                if (rgTgt[i] < 0 && m_colTgtNeg != null)
                    nTgtIdxNeg = m_colTgtNeg.Add(rgTgt[i]);
                else
                    nTgtIdxPos = m_colTgtPos.Add(rgTgt[i]);

                if (rgPred[i] < 0 && m_colPredNeg != null)
                    nPredIdxNeg = m_colPredNeg.Add(rgPred[i]);
                else
                    nPredIdxPos = m_colPredPos.Add(rgPred[i]);

                if (nTgtIdxNeg >= 0)
                {
                    Bucket b = m_colTgtNeg[nTgtIdxNeg];

                    if (nTgtIdxNeg == nPredIdxNeg)
                    {
                        if (!m_rgBucketCorrectHits.ContainsKey(b))
                            m_rgBucketCorrectHits.Add(b, 0);
                        m_rgBucketCorrectHits[b]++;
                    }
                    else
                    {
                        if (!m_rgBucketIncorrectHits.ContainsKey(b))
                            m_rgBucketIncorrectHits.Add(b, new Dictionary<int, int>());

                        if (nPredIdxNeg >= 0)
                        {
                            if (!m_rgBucketIncorrectHits[b].ContainsKey(nPredIdxNeg))
                                m_rgBucketIncorrectHits[b].Add(nPredIdxNeg, 0);
                            m_rgBucketIncorrectHits[b][nPredIdxNeg]++;
                        }
                        else if (nPredIdxPos >= 0)
                        {
                            if (!m_rgBucketIncorrectHits[b].ContainsKey(nPredIdxPos))
                                m_rgBucketIncorrectHits[b].Add(nPredIdxPos, 0);
                            m_rgBucketIncorrectHits[b][nPredIdxPos]++;
                        }
                    }
                }
                else if (nTgtIdxPos >= 0)
                {
                    Bucket b = m_colTgtPos[nTgtIdxPos];
                    if (nTgtIdxPos == nPredIdxPos)
                    {
                        if (!m_rgBucketCorrectHits.ContainsKey(b))
                            m_rgBucketCorrectHits.Add(b, 0);
                        m_rgBucketCorrectHits[b]++;
                    }
                    else
                    {
                        if (!m_rgBucketIncorrectHits.ContainsKey(b))
                            m_rgBucketIncorrectHits.Add(b, new Dictionary<int, int>());

                        if (nPredIdxPos >= 0)
                        {
                            if (!m_rgBucketIncorrectHits[b].ContainsKey(nPredIdxPos))
                                m_rgBucketIncorrectHits[b].Add(nPredIdxPos, 0);
                            m_rgBucketIncorrectHits[b][nPredIdxPos]++;
                        }
                        else if (nPredIdxNeg >= 0)
                        {
                            if (!m_rgBucketIncorrectHits[b].ContainsKey(nPredIdxNeg))
                                m_rgBucketIncorrectHits[b].Add(nPredIdxNeg, 0);
                            m_rgBucketIncorrectHits[b][nPredIdxNeg]++;
                        }
                    }
                }
            }

            if (nPredIdxNeg >= 0)
                return -nPredIdxNeg;
            else if (nPredIdxPos >= 0)
                return nPredIdxPos;
            else
                return 0;
        }

        /// <summary>
        /// Calculates the overall accuracy.
        /// </summary>
        /// <param name="bGetDetails">Specifies to fill out the details string.</param>
        /// <param name="strDetails">Specifies the string to receive the details, when specified.</param>
        /// <returns>The accuracy is returned.</returns>
        public double CalculateAccuracy(bool bGetDetails, out string strDetails)
        {
            strDetails = (bGetDetails) ? "" : null;

            int nTotalCorrect = m_rgBucketCorrectHits.Sum(p => p.Value);
            int nTotalPredictions = m_rgBucketCorrectHits.Sum(p => p.Key.Count);

            if (bGetDetails)
            {
                List<KeyValuePair<Bucket, int>> rgBuckets = m_rgBucketCorrectHits.OrderBy(p => p.Key.MidPoint).ToList();
                foreach (KeyValuePair<Bucket, int> kv in rgBuckets)
                {
                    double dfPct = kv.Value / (double)kv.Key.Count;
                    strDetails += "Bucket: " + kv.Key.ToString() + " Accuracy: " + dfPct.ToString("P2") + Environment.NewLine;
                }
            }

            strDetails += CreateConfusionMatrix();

            // Calculate accuracy as a percentage
            double accuracy = nTotalPredictions > 0 ? (nTotalCorrect / (double)nTotalPredictions) : 0;
            return accuracy;
        }

        private int[,] createConfusionMatrix(List<string> rgstrTargetLabels, List<string> rgstrPredLabels, List<Bucket> rgTargets)
        {
            int[,] confusionMatrix = new int[rgstrTargetLabels.Count, rgstrPredLabels.Count];

            for (int i = 0; i < rgstrTargetLabels.Count; i++)
            {
                Bucket bTarget = rgTargets[i]; // Current target bucket

                for (int j = 0; j < rgstrPredLabels.Count; j++)
                {
                    // Assuming m_rgBucketCorrectHits stores hits per target, indexed by prediction label
                    if (m_rgBucketCorrectHits.ContainsKey(bTarget))
                    {
                        int nPredCount = m_rgBucketCorrectHits[bTarget]; // Fetch prediction count for this target-prediction pair
                        int nVal = 0;

                        if (i == j) // Diagonal: correct predictions
                        {
                            nVal = Math.Min(bTarget.Count, nPredCount);
                        }
                        else
                        {
                            if (m_rgBucketIncorrectHits.ContainsKey(bTarget))
                            {
                                if (j > i)
                                {
                                    if (m_rgBucketIncorrectHits[bTarget].ContainsKey(i))
                                        nVal = m_rgBucketIncorrectHits[bTarget][i];
                                }
                                else if (j < i)
                                {
                                    if (m_rgBucketIncorrectHits[bTarget].ContainsKey(j))
                                        nVal = m_rgBucketIncorrectHits[bTarget][j];
                                }
                            }
                        }

                        confusionMatrix[i, j] = nVal;
                    }
                }
            }

            return confusionMatrix;
        }

        private double[,] createPercentMatrix(List<string> rgstrTargetLabels, List<string> rgstrPredLabels, List<Bucket> rgTarget, int[,] confusionMatrix)
        {
            double[,] percentageMatrix = new double[rgstrTargetLabels.Count, rgstrPredLabels.Count];

            // Calculate percentages for the confusion matrix
            for (int i = 0; i < rgstrTargetLabels.Count; i++)
            {
                int total = rgTarget[i].Count;
                for (int j = 0; j < rgstrPredLabels.Count; j++)
                {
                    percentageMatrix[i, j] = total > 0 ? (double)confusionMatrix[i, j] / total * 100 : 0;
                }
            }

            return percentageMatrix;
        }

        /// <summary>
        /// Prints a matrix with labels and values, formatted as counts or percentages.
        /// </summary>
        private void printMatrix(StringBuilder sb, string[] labels, dynamic matrix, int maxLabelWidth, int maxCellWidth, bool isPercentage)
        {
            // Print the matrix title and determine if it's showing percentages
            sb.Append("Confusion Matrix");
            if (isPercentage)
                sb.Append(" (Percentages)");
            sb.AppendLine(":");

            // Print the header row with labels
            sb.Append(new string('_', maxLabelWidth + 1));  // Adjusted space to align with the data rows
            foreach (string label in labels)
            {
                sb.Append($"| {label.PadRight(maxCellWidth)} ");
            }
            sb.AppendLine();

            // Print each row of the matrix
            for (int i = 0; i < labels.Length; i++)
            {
                // Pad the row label to align with the header row
                sb.Append($"{labels[i].PadRight(maxLabelWidth)} ");

                // Print each cell in the row
                for (int j = 0; j < labels.Length; j++)
                {
                    // Determine formatting based on whether values are percentages
                    string value = isPercentage ? $"{matrix[i, j]:F2}%" : $"{matrix[i, j]}";
                    sb.Append($"| {value.PadLeft(maxCellWidth + 2)} ");
                }
                sb.AppendLine();
            }
        }


        /// <summary>
        /// Create a confusion matrix of the values.
        /// </summary>
        /// <returns>The confusion matrix is returned as a pretty-print string.</returns>
        public string CreateConfusionMatrix()
        {
            List<string> rgstrPredLabels = new List<string>();
            List<string> rgstrTargetLabels = new List<string>();
            List<Bucket> rgTargetLabels = new List<Bucket>();

            foreach (Bucket b in m_colTgtNeg)
            {
                rgstrTargetLabels.Add(b.Minimum.ToString("N2") + " - " + b.Maximum.ToString("N2"));
                rgTargetLabels.Add(b);
            }

            foreach (Bucket b in m_colPredNeg)
            {
                rgstrPredLabels.Add(b.Minimum.ToString("N2") + " - " + b.Maximum.ToString("N2"));
            }

            foreach (Bucket b in m_colTgtPos)
            {
                rgstrTargetLabels.Add(b.Minimum.ToString("N2") + " - " + b.Maximum.ToString("N2"));
                rgTargetLabels.Add(b);
            }

            foreach (Bucket b in m_colPredPos)
            {
                rgstrPredLabels.Add(b.Minimum.ToString("N2") + " - " + b.Maximum.ToString("N2"));
            }

            // Create the confusion matrix
            int[,] confusionMatrix = createConfusionMatrix(rgstrTargetLabels, rgstrPredLabels, rgTargetLabels);
            double[,] percentageMatrix = createPercentMatrix(rgstrTargetLabels, rgstrPredLabels, rgTargetLabels, confusionMatrix);
            StringBuilder sb = new StringBuilder();

            string[] labels = rgstrTargetLabels.ToArray();
            int maxLabelWidth = labels.Max(label => label.Length);
            int maxCellWidth = 10;

            printMatrix(sb, labels, confusionMatrix, maxLabelWidth, maxCellWidth, false);
            sb.AppendLine();
            printMatrix(sb, labels, percentageMatrix, maxLabelWidth, maxCellWidth, true);

            int nTotal = m_colTgtPos.Sum(p => p.Count) + m_colTgtNeg.Sum(p => p.Count);
            double dfGtPercentPos = (double)m_colTgtPos.Sum(p => p.Count) / nTotal;
            double dfGtPercentNeg = (double)m_colTgtNeg.Sum(p => p.Count) / nTotal;
            sb.AppendLine();
            sb.AppendLine("Ground Truth Sample Size = " + nTotal.ToString("N0"));
            sb.AppendLine("Ground Truth % Positive = " + dfGtPercentPos.ToString("P2"));
            sb.AppendLine("Ground Truth % Negative = " + dfGtPercentNeg.ToString("P2"));

            return sb.ToString();
        }
    }
}
