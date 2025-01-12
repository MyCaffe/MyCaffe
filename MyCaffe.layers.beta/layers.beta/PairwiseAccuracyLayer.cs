using System;
using System.Collections.Generic;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The PairwiseAccuracyLayer computes ranking accuracy metrics for the PairwiseLossLayer
    /// by measuring the percentage of correctly ordered pairs and weighted ranking correlation:
    /// @f$
    ///     accuracy = \frac{\sum_{i,j} w_{ij} * correct_{ij}}{\sum_{i,j} w_{ij}}
    /// @f$
    /// where correct_{ij} = 1 if sign(pred_i - pred_j) = sign(true_i - true_j), 0 otherwise
    /// and w_{ij} = |true_i - true_j| * (1 + |true_i - true_j|)
    /// </summary>
    /// <remarks>
    /// This layer works in conjunction with PairwiseLossLayer by:
    /// 1. Using the same weighting scheme for pair importance
    /// 2. Focusing on relative ordering accuracy rather than absolute values
    /// 3. Handling both long and short positions correctly
    /// </remarks>
    public class PairwiseAccuracyLayer<T> : Layer<T>
    {
        int m_nBatchSize;
        Blob<T> m_blobDiffTrue;
        Blob<T> m_blobDiffPred;
        Blob<T> m_blobValidPairs;
        Blob<T> m_blobCorrectPairs;
        Blob<T> m_blobWeights;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type PAIRWISE_ACCURACY.</param>
        public PairwiseAccuracyLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.PAIRWISE_ACCURACY;

            // Initialize internal blobs for computation
            m_blobDiffTrue = new Blob<T>(cuda, log);
            m_blobDiffTrue.Name = m_param.name + ".diff_true";
            m_blobDiffPred = new Blob<T>(cuda, log);
            m_blobDiffPred.Name = m_param.name + ".diff_pred";
            m_blobValidPairs = new Blob<T>(cuda, log);
            m_blobValidPairs.Name = m_param.name + ".valid_pairs";
            m_blobCorrectPairs = new Blob<T>(cuda, log);
            m_blobCorrectPairs.Name = m_param.name + ".correct_pairs";
            m_blobWeights = new Blob<T>(cuda, log);
            m_blobWeights.Name = m_param.name + ".weights";
        }

        /// <summary>
        /// Cleanup
        /// </summary>
        protected override void dispose()
        {
            dispose(ref m_blobDiffTrue);
            dispose(ref m_blobDiffPred);
            dispose(ref m_blobValidPairs);
            dispose(ref m_blobCorrectPairs);
            dispose(ref m_blobWeights);
            base.dispose();
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) blobs: predictions and targets
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) blobs: accuracy
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Collection of bottom (input) blobs.</param>
        /// <param name="colTop">Collection of top (output) blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(colBottom[0].count(), colBottom[1].count(), "The predictions and targets must have the same count.");
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Collection of bottom (input) blobs.</param>
        /// <param name="colTop">Collection of top (output) blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nBatchSize = colBottom[0].num;  // Get batch size from input blob

            // Reset internal blobs
            List<int> pairShape = new List<int> { m_nBatchSize, m_nBatchSize };
            m_blobDiffTrue.Reshape(pairShape);
            m_blobDiffPred.Reshape(pairShape);
            m_blobValidPairs.Reshape(pairShape);
            m_blobCorrectPairs.Reshape(pairShape);
            m_blobWeights.Reshape(pairShape);

            // Accuracy is a scalar
            List<int> topShape = new List<int> { 1 };
            colTop[0].Reshape(topShape);
            colTop[0].blob_type = BLOB_TYPE.ACCURACY;
        }

        /// <summary>
        /// Forward computation of pairwise ranking accuracy
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (length 2)
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the predictions @f$ \hat{y} @f$
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the targets @f$ y @f$
        /// </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed accuracy between 0 and 1.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            float[] rgfPred = convertF(colBottom[0].mutable_cpu_data);
            float[] rgfTrue = convertF(colBottom[1].mutable_cpu_data);
            float[] rgfDiffTrue = convertF(m_blobDiffTrue.mutable_cpu_data);
            float[] rgfDiffPred = convertF(m_blobDiffPred.mutable_cpu_data);
            float[] rgfValidPairs = convertF(m_blobValidPairs.mutable_cpu_data);
            float[] rgfCorrectPairs = convertF(m_blobCorrectPairs.mutable_cpu_data);
            float[] rgfWeights = convertF(m_blobWeights.mutable_cpu_data);

            float fTotalCorrect = 0.0f;
            float fTotalWeight = 0.0f;

            // Compute pairwise comparisons
            for (int i = 0; i < m_nBatchSize; i++)
            {
                float fTrueI = rgfTrue[i];
                float fPredI = rgfPred[i];

                for (int j = 0; j < m_nBatchSize; j++)
                {
                    if (i == j) continue;

                    int idx = i * m_nBatchSize + j;
                    float fTrueJ = rgfTrue[j];
                    float fPredJ = rgfPred[j];

                    float fTrueDiff = fTrueI - fTrueJ;
                    float fPredDiff = fPredI - fPredJ;

                    rgfDiffTrue[idx] = fTrueDiff;
                    rgfDiffPred[idx] = fPredDiff;

                    // Calculate importance weight based on return difference magnitude
                    float fReturnDiffAbs = Math.Abs(fTrueDiff);
                    float fWeight = fReturnDiffAbs * (1.0f + fReturnDiffAbs);  // Quadratic weighting
                    rgfWeights[idx] = fWeight;

                    bool bValidPair = fReturnDiffAbs > 1e-6;
                    rgfValidPairs[idx] = bValidPair ? 1.0f : 0.0f;

                    if (bValidPair)
                    {
                        // Check if prediction ordering matches true ordering
                        bool bCorrect = Math.Sign(fPredDiff) == Math.Sign(fTrueDiff);
                        rgfCorrectPairs[idx] = bCorrect ? 1.0f : 0.0f;

                        if (bCorrect)
                            fTotalCorrect += fWeight;

                        fTotalWeight += fWeight;
                    }
                    else
                    {
                        rgfCorrectPairs[idx] = 0.0f;
                    }
                }
            }

            // Update the internal blobs with computed values
            m_blobDiffTrue.mutable_cpu_data = convert(rgfDiffTrue);
            m_blobDiffPred.mutable_cpu_data = convert(rgfDiffPred);
            m_blobValidPairs.mutable_cpu_data = convert(rgfValidPairs);
            m_blobCorrectPairs.mutable_cpu_data = convert(rgfCorrectPairs);
            m_blobWeights.mutable_cpu_data = convert(rgfWeights);

            // Calculate weighted accuracy
            float fAccuracy = fTotalWeight > 0.0f ? fTotalCorrect / fTotalWeight : 0.0f;
            colTop[0].SetData(fAccuracy, 0);
        }

        /// <summary>
        /// The accuracy layer doesn't need backward computation
        /// </summary>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
                throw new NotImplementedException();
        }
    }
}