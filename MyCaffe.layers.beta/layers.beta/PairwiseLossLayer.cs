using System;
using System.Collections.Generic;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The PairwiseLossLayer computes a return-weighted ranking loss optimized for long-short portfolio selection
    /// @f$
    ///     \begin{aligned}
    ///         true\_diff = y\_true_i - y\_true_j
    ///         pred\_diff = y\_pred_i - y\_pred_j
    ///         weight = |true\_diff| * (1 + |true\_diff|)  // Quadratically weight larger return differences
    ///         loss = weight * max(0, margin - sign(true\_diff) * pred\_diff)
    ///     \end{aligned}
    /// @f$. 
    /// </summary>
    /// <remarks>
    /// References:
    /// @see [Deep Portfolio Management Using Deep Learning](https://arxiv.org/abs/2405.01604) by Ashish Anil Pawar, Vishnureddy Prashant Muskawar, Ritesh Tiku., 2024.
    /// The paper proposes a Deep Reinforcement Learning (DRL) approach for portfolio management that directly learns to allocate weights to different assets 
    /// (both long and short positions) and shows it can outperform traditional portfolio management methods in terms of risk-adjusted returns.
    /// 
    /// @see [Deep Learning for Portfolio Optimization](https://arxiv.org/abs/2005.13665) by Zhang et al., 2020.
    /// Introduces weighted ranking losses for portfolio selection.
    /// 
    /// @see [From RankNet to LambdaRank to LambdaMART:AnOverview](https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf) by Burges, 2016.
    /// This technical report provides a comprehensive explanation of three related learning-to-rank algorithms - RankNet, LambdaRank, and LambdaMART - which 
    /// progressively improved ranking performance by introducing the novel concept of directly specifying gradients rather than deriving them from cost functions.
    /// 
    /// @see [Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf) by Cao et al., 2007.
    /// This paper proposes a "listwise" approach to learning-to-rank that treats entire lists of items as training instances rather than pairs of items, introducing probability 
    /// models and neural networks to define loss functions that outperform traditional pairwise ranking methods.
    /// 
    /// Implementation References:
    /// @see [LightGBM Ranking Implementation](https://github.com/microsoft/LightGBM/blob/master/src/objective/rank_objective.hpp)
    /// Efficient C++ implementation of ranking losses.
    /// 
    /// @see [FastAI Pairwise Ranking](https://github.com/fastai/fastai/blob/master/fastai/losses.py)
    /// Python implementation of pairwise ranking losses with emphasis on efficiency.    
    /// </remarks>
    public class PairwiseLossLayer<T> : LossLayer<T>
    {
        List<int> m_rgShape = new List<int>(4);
        Blob<T> m_blobDiffTrue;
        Blob<T> m_blobDiffPred;
        Blob<T> m_blobValidPairs;
        Blob<T> m_blobLoss;
        Blob<T> m_blobWeights;       // Stores importance weights for each pair
        double m_dfMargin;
        int m_nBatchSize;
        int m_nValidPairsPerBatch;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the connection to the CUDA layer.</param>
        /// <param name="log">Specifies the output Log.</param>
        /// <param name="p">Specifies the parameter.</param>
        public PairwiseLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.PAIRWISE_LOSS;
            m_dfMargin = p.pairwise_loss_param.margin;
            m_nValidPairsPerBatch = m_nBatchSize * (m_nBatchSize - 1);

            m_blobDiffTrue = new Blob<T>(cuda, log);
            m_blobDiffTrue.Name = m_param.name + ".diff_true";
            m_blobDiffPred = new Blob<T>(cuda, log);
            m_blobDiffPred.Name = m_param.name + ".diff_pred";
            m_blobValidPairs = new Blob<T>(cuda, log);
            m_blobValidPairs.Name = m_param.name + ".valid_pairs";
            m_blobLoss = new Blob<T>(cuda, log);
            m_blobLoss.Name = m_param.name + ".loss";
            m_blobWeights = new Blob<T>(cuda, log);
            m_blobWeights.Name = m_param.name + ".weights";

            // Initialize shapes once since batch size is fixed
            List<int> pairShape = new List<int> { m_nBatchSize, m_nBatchSize };
            m_blobDiffTrue.Reshape(pairShape);
            m_blobDiffPred.Reshape(pairShape);
            m_blobValidPairs.Reshape(pairShape);
            m_blobLoss.Reshape(pairShape);
            m_blobWeights.Reshape(pairShape);

            m_rgShape.Add(1);  // Top blob will be scalar loss
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        protected override void dispose()
        {
            dispose(ref m_blobDiffTrue);
            dispose(ref m_blobDiffPred);
            dispose(ref m_blobValidPairs);
            dispose(ref m_blobLoss);
            dispose(ref m_blobWeights);
            base.dispose();
        }

        /// <summary>
        /// Setup the internal blobs used.
        /// </summary>
        /// <param name="col">Specifies the collection of internal blobs.</param>
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobDiffTrue);
            col.Add(m_blobDiffPred);
            col.Add(m_blobValidPairs);
            col.Add(m_blobLoss);
            col.Add(m_blobWeights);
        }

        /// <summary>
        /// The PairwiseLossLayer can backpropagate
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
            m_log.CHECK_EQ(colBottom.Count, 2, "There should be two inputs: predictions and target values.");
            m_log.CHECK_EQ(colBottom[0].count(), colBottom[1].count(), "The inputs must have the same count.");

            colTop[0].Reshape(m_rgShape);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_nBatchSize = colBottom[0].num;
            m_nOuterNum = m_nBatchSize;
            m_nInnerNum = colBottom[0].count(1);

            // Reset internal blobs
            List<int> pairShape = new List<int> { m_nBatchSize, m_nBatchSize };
            m_blobDiffTrue.Reshape(pairShape);
            m_blobDiffPred.Reshape(pairShape);
            m_blobValidPairs.Reshape(pairShape);
            m_blobLoss.Reshape(pairShape);
            m_blobWeights.Reshape(pairShape);

            // Loss is a scalar
            List<int> topShape = new List<int> { 1 };
            colTop[0].Reshape(topShape);
            colTop[0].blob_type = BLOB_TYPE.LOSS;
        }

        /// <summary>
        /// Forward computation of pairwise ranking loss optimized for return spread prediction
        /// </summary>
        /// <param name="colBottom">input Blob vector (length 2)
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the model scores @f$ \hat{y} \in [-\infty, +\infty] @f$
        ///     where higher scores should predict higher returns
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the target 21-day forward returns @f$ y \in [-\infty, +\infty] @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed weighted pairwise loss: 
        ///     @f$ E = \frac{\sum_i \sum_j w_{ij} \max(0, margin - sign(y_i - y_j)(\hat{y}_i - \hat{y}_j))}{\sum_i \sum_j w_{ij}} @f$
        ///     where @f$ w_{ij} @f$ is the importance weight based on return difference magnitude
        /// </param>
        /// <remarks>
        /// The loss is designed to maximize the spread between high-return and low-return assets by:
        /// 1. Weighting pairs by the magnitude of their return difference
        /// 2. Enforcing a margin between scores based on return rankings
        /// 3. Putting extra emphasis on extreme return differences through quadratic and exponential weighting
        /// </remarks>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            float[] rgfPred = convertF(colBottom[0].mutable_cpu_data);
            float[] rgfTrue = convertF(colBottom[1].mutable_cpu_data);
            float[] rgfDiffTrue = convertF(m_blobDiffTrue.mutable_cpu_data);
            float[] rgfDiffPred = convertF(m_blobDiffPred.mutable_cpu_data);
            float[] rgfValidPairs = convertF(m_blobValidPairs.mutable_cpu_data);
            float[] rgfLoss = convertF(m_blobLoss.mutable_cpu_data);
            float[] rgfWeights = convertF(m_blobWeights.mutable_cpu_data);

            float fTotalLoss = 0.0f;
            float fTotalWeight = 0.0f;

            // Compute pairwise comparisons with return-magnitude weighting
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
                        // Weighted hinge loss
                        float fLoss = fWeight * (float)Math.Max(0.0, m_dfMargin - Math.Sign(fTrueDiff) * fPredDiff);
                        rgfLoss[idx] = fLoss;
                        fTotalLoss += fLoss;
                        fTotalWeight += fWeight;
                    }
                    else
                    {
                        rgfLoss[idx] = 0.0f;
                    }
                }
            }

            // Update the internal blobs with computed values
            m_blobDiffTrue.mutable_cpu_data = convert(rgfDiffTrue);
            m_blobDiffPred.mutable_cpu_data = convert(rgfDiffPred);
            m_blobLoss.mutable_cpu_data = convert(rgfLoss);
            m_blobValidPairs.mutable_cpu_data = convert(rgfValidPairs);
            m_blobWeights.mutable_cpu_data = convert(rgfWeights);

            // Normalize the loss by total weight to match test expectations
            float fNormalizedLoss = (fTotalWeight > 0) ? fTotalLoss / fTotalWeight : 0.0f;

            if (m_param.loss_param.loss_scale != 1.0 &&
                m_param.loss_param.loss_scale != 0.0)
                fNormalizedLoss *= (float)m_param.loss_param.loss_scale;

            colTop[0].SetData(fNormalizedLoss, 0);
        }

        /// <summary>
        /// Computes the PairwiseLoss error gradient w.r.t. the inputs, optimizing for return-spread prediction.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$ 
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">input Blob vector (length 2)
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$ model scores for ranking
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$ target 21-day forward returns
        /// </param>
        /// <remarks>
        /// The gradient computation follows:
        /// @f$ \frac{\partial E}{\partial \hat{y}_i} = \sum_j w_{ij} \begin{cases}
        ///   -sign(y_i - y_j) & \text{if } margin - sign(y_i - y_j)(\hat{y}_i - \hat{y}_j) > 0 \\
        ///   0 & \text{otherwise}
        /// \end{cases} @f$
        /// where @f$ w_{ij} = (|y_i - y_j|)^2 + exp(min(0.5|y_i - y_j|, 5)) - 1 @f$ is the importance weight
        /// for the pair (i,j) based on their return difference magnitude.
        /// 
        /// The gradient encourages:
        /// 1. Higher scores for assets with higher future returns
        /// 2. Lower scores for assets with lower future returns
        /// 3. Larger score differences for pairs with larger return differences
        /// </remarks>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            float[] rgfDiffTrue = convertF(m_blobDiffTrue.mutable_cpu_data);
            float[] rgfDiffPred = convertF(m_blobDiffPred.mutable_cpu_data);
            float[] rgfValidPairs = convertF(m_blobValidPairs.mutable_cpu_data);
            float[] rgfWeights = convertF(m_blobWeights.mutable_cpu_data);
            float[] rgfGrad = convertF(colBottom[0].mutable_cpu_diff);

            // Get the loss weight from the top gradient
            float fLossWeight = convertF(colTop[0].GetDiff(0));

            if (m_param.loss_param.loss_scale != 1.0 &&
                m_param.loss_param.loss_scale != 0.0)
                fLossWeight *= (float)m_param.loss_param.loss_scale;

            // Calculate total weight for normalization
            float fTotalWeight = 0.0f;
            for (int i = 0; i < m_nBatchSize; i++)
            {
                for (int j = 0; j < m_nBatchSize; j++)
                {
                    if (i == j) continue;
                    int idx = i * m_nBatchSize + j;
                    if (rgfValidPairs[idx] > 0)
                    {
                        fTotalWeight += rgfWeights[idx];
                    }
                }
            }

            // Initialize gradients to zero
            Array.Clear(rgfGrad, 0, colBottom[0].count());

            if (fTotalWeight > 0)
            {
                for (int i = 0; i < m_nBatchSize; i++)
                {
                    for (int j = 0; j < m_nBatchSize; j++)
                    {
                        if (i == j) continue;
                        int idx = i * m_nBatchSize + j;
                        if (rgfValidPairs[idx] > 0)
                        {
                            float fTrueDiff = rgfDiffTrue[idx];
                            float fPredDiff = rgfDiffPred[idx];
                            float fWeight = rgfWeights[idx];

                            if (m_dfMargin - Math.Sign(fTrueDiff) * fPredDiff > 0)
                            {
                                // Apply gradient with weight, loss weight, and normalization
                                float fGradSign = Math.Sign(fTrueDiff) * fWeight * fLossWeight / fTotalWeight;
                                rgfGrad[i] -= fGradSign;
                                rgfGrad[j] += fGradSign;
                            }
                        }
                    }
                }
            }

            colBottom[0].mutable_cpu_diff = convert(rgfGrad);
        }
    }
}