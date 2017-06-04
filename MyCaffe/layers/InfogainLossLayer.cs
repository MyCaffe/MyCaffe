using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The InforgainLossLayer is a generalization of MultinomialLogisticLossLayer that takes an
    /// 'information gain' (infogain) matrix specifying the 'value of all label
    /// pairs.
    /// This layer is initialized with the MyCaffe.param.InfogainLossParameter.
    /// </summary>
    /// <remarks>
    /// Equivalent to the MultinomialLogisticLossLayer if the inforgain matrix is the
    /// identity.
    /// 
    /// @see [DeepGaze II: Reading fixations from deep features trained on object recognition](https://arxiv.org/abs/1610.01563) by Matthias Kümmerer, Thomas S. A. Wallis, and Matthias Bethge, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class InfogainLossLayer<T> : LossLayer<T>
    {
        // The internal SoftmaxLayer used to map predictions to a distribution.
        SoftmaxLayer<T> m_softmaxLayer;
        // Stores the output probability preductions from the SoftmaxLayer.
        Blob<T> m_blobProb = null;
        // Bottom vector holder used in the call to the underlying SoftmaxLayer::Forward
        BlobCollection<T> m_colSoftmaxBottomVec = new BlobCollection<T>();
        // Top vector holder used in the call to the underlying SoftmaxLayer::Forward
        BlobCollection<T> m_colSoftmaxTopVec = new BlobCollection<T>();
        // The infogain blob.
        Blob<T> m_blobInfoGain = null;
        // Cache holding the rows sums of H.
        Blob<T> m_blobSumRowsOfH = null;
        // The optional lable indicating that an instance should be ignored.
        int? m_nIgnoreLabel = null;
        // how to normalize the output loss.
        LossParameter.NormalizationMode m_normalization = LossParameter.NormalizationMode.NONE;

        int m_nInfogainAxis = 0;
        int m_nOuterNum = 0;
        int m_nInnerNum = 0;
        int m_nNumLabels = 0;

        /// <summary>
        /// The InfogainLossLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LossParameter loss_param, with options:
        ///  - ignore_label (optional)
        ///    Specify a label value that whould be ignored when computing the loss.
        ///    
        ///  - normalize (optional, default true)
        ///    If true, the loss is normalized by the number of (nonignored) labels
        ///    present; otherwise the loss is imply summed over spatial locations.
        /// </param>
        public InfogainLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.INFOGAIN_LOSS;

            m_blobProb = new Blob<T>(m_cuda, m_log);
            m_blobProb.Name = "softmax prob";
            m_blobInfoGain = new Blob<T>(m_cuda, m_log);
            m_blobInfoGain.Name = "infogain";
            m_blobSumRowsOfH = new Blob<T>(m_cuda, m_log);
            m_blobSumRowsOfH.Name = "sum rows of H";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            if (m_blobInfoGain != null)
            {
                m_blobInfoGain.Dispose();
                m_blobInfoGain = null;
            }

            if (m_blobProb != null)
            {
                m_blobProb.Dispose();
                m_blobProb = null;
            }

            if (m_blobSumRowsOfH != null)
            {
                m_blobSumRowsOfH.Dispose();
                m_blobSumRowsOfH = null;
            }
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = base.internal_blobs;
                col.Add(m_blobInfoGain);
                col.Add(m_blobProb);
                col.Add(m_blobSumRowsOfH);
                return col;
            }
        }

        /// <summary>
        /// InfogainLossLayer takes 2-3 bottom blobs; if there are 3 the third should
        /// be the infogain matrix. (Otherwise the infogain matrix is loaded from a
        /// file specified by the LayerParameter.)
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the minimum number of required bottom (intput) Blobs: pred, label
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the maximum number of required bottom (intput) Blobs: pred, label, matrix
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 3; }
        }

        /// <summary>
        /// InfogainLossLayer computes softmax probability internally.
        /// optional second 'top' outputs the sofmax probability.
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: infogain
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of requried top (output) Blobs: infogain, softmax
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Read the normalization mode parameter and compute the normalizer based on the blob size.
        /// If the <i>normalization_mode</i> is VALID, the count of valid outputs will be read from
        /// <i>valid_count</i>, unless it is -1 in which case all outputs are assumed to be valid.
        /// </summary>
        /// <param name="normalization_mode">Specifies the normalization mode to use.</param>
        /// <param name="nValidCount">Specifies the valid count.</param>
        /// <returns>The normalization value is returned.</returns>
        protected virtual double get_normalizer(LossParameter.NormalizationMode normalization_mode, int nValidCount)
        {
            double dfNormalizer = 0;

            switch (normalization_mode)
            {
                case LossParameter.NormalizationMode.FULL:
                    dfNormalizer = m_nOuterNum * m_nInnerNum;
                    break;

                case LossParameter.NormalizationMode.VALID:
                    dfNormalizer = (nValidCount == -1) ? (m_nOuterNum * m_nInnerNum) : nValidCount;
                    break;

                case LossParameter.NormalizationMode.BATCH_SIZE:
                    dfNormalizer = m_nOuterNum;
                    break;

                case LossParameter.NormalizationMode.NONE:
                    dfNormalizer = 1;
                    break;

                default:
                    m_log.FAIL("Unknown normalization mode: " + normalization_mode.ToString());
                    break;
            }

            // Some users will have no labels for some examples in order to 'turn off' a 
            // particular loss in a multi-task setup.  The max prevents NaNs in that case.
            return Math.Max(1.0, dfNormalizer);
        }

        /// <summary>
        /// Fill <i>m_blobSumRowsOfH</i> according to matrix H.
        /// </summary>
        /// <param name="blobH">Specifies the Sum Rows of H blob.</param>
        protected virtual void sum_rows_of_H(Blob<T> blobH)
        {
            m_log.CHECK_EQ(blobH.count(), m_nNumLabels * m_nNumLabels, "H must be " + m_nNumLabels.ToString() + " x " + m_nNumLabels.ToString());
            float[] rgInfogainMat = convertF(blobH.update_cpu_data());
            float[] rgSum = convertF(m_blobSumRowsOfH.mutable_cpu_data);

            for (int row = 0; row < m_nNumLabels; row++)
            {
                rgSum[row] = 0;

                for (int col = 0; col < m_nNumLabels; col++)
                {
                    rgSum[row] += rgInfogainMat[row * m_nNumLabels + col];
                }
            }

            m_blobSumRowsOfH.mutable_cpu_data = convert(rgSum);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            // Internal softmax layer.
            LayerParameter softmax_param = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            softmax_param.softmax_param.axis = m_param.infogain_loss_param.axis;
            softmax_param.loss_weight.Clear();
            softmax_param.loss_weight.Add(1);
            m_softmaxLayer = new SoftmaxLayer<T>(m_cuda, m_log, softmax_param);
            m_colSoftmaxBottomVec.Clear();
            m_colSoftmaxBottomVec.Add(colBottom[0]);
            m_colSoftmaxTopVec.Clear();
            m_colSoftmaxTopVec.Add(m_blobProb);
            m_softmaxLayer.Setup(m_colSoftmaxBottomVec, m_colSoftmaxTopVec);

            // ignore label.
            m_nIgnoreLabel = m_param.loss_param.ignore_label;

            // normalization
            m_log.CHECK(!m_param.loss_param.normalize, "normalize is drepreciated, use 'normalization'.");
            m_normalization = m_param.loss_param.normalization;

            // matrix H
            if (colBottom.Count < 3)
            {
                m_log.CHECK(m_param.infogain_loss_param.source != null, "Infogain matrix source must be specified.");
                PersistCaffe<T> persist = new PersistCaffe<T>(m_log, true);              
                BlobProto blobProto = persist.LoadBlobProto(m_param.infogain_loss_param.source, 1);
                m_blobInfoGain.FromProto(blobProto);
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_softmaxLayer.Reshape(m_colSoftmaxBottomVec, m_colSoftmaxTopVec);
            m_nInfogainAxis = colBottom[0].CanonicalAxisIndex(m_param.infogain_loss_param.axis);
            m_nOuterNum = colBottom[0].count(0, m_nInfogainAxis);
            m_nInnerNum = colBottom[0].count(m_nInfogainAxis + 1);

            m_log.CHECK_EQ(m_nOuterNum * m_nInnerNum, colBottom[1].count(), "Number of labels must match the number of predictions; e.g., if infogain_axis == 1 and predictions shape is (N, C, H, W), label count (number of labels) must be N*H*W, with integer values in {0, 1, ..., C-1}.");
            m_nNumLabels = colBottom[0].shape(m_nInfogainAxis);

            Blob<T> blobInfoGain = null;

            if (colBottom.Count < 3)
                blobInfoGain = m_blobInfoGain;
            else
                blobInfoGain = colBottom[2];

            m_log.CHECK_EQ(blobInfoGain.count(), m_nNumLabels * m_nNumLabels, "The infogain count must equal 'num_labels' * 'num_labels'.");
            m_blobSumRowsOfH.Reshape(new List<int>() { m_nNumLabels });
            if (colBottom.Count == 2)
            {
                // H is provided as a parameter and will not change.  Sum rows once.
                sum_rows_of_H(blobInfoGain);
            }
            if (colTop.Count >= 2)
            {
                // softmax output.
                colTop[1].ReshapeLike(colBottom[0]);
            }
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 2-3)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ \hat{p} @f$, a Blob with values in
        ///     [0,1] indicating the predicted probability of each of the
        ///     K = CHW classes.  Each prediction vector @f$ \hat{p}_n @f$
        ///     should sum to 1 as in a probability distribution:
        ///       @f$ \forall n \sum\limits_{k=1}^K \hat{p}_{nk} = 1 @f$
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels @f$ l @f$, an integer-valued Blob with values
        ///     @f$ l_n \in [0, 1, 2, ..., K-1] @f$
        ///     indicating the correct class label among the @f$ K @f$ classes
        ///  -# @f$ (1 \times 1 \times K \times K) @f$
        ///     (\b optional) the infogain matrix @f$ H @f$.  This must be provided as
        ///     the third bottom blob input if not provided as the inforgain_mat in the
        ///     InfogainLossParameter.  If @f$ H = I @f$, this layer is equivalent to the
        ///     MultinomialLogisticsLossLayer.
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed infogain multinomial logistic loss: @f$ E = 
        ///       \frac{-1}{N} \sum\limits_{n=1}^N H_{l_n} \log(\hat{p}_n) =
        ///       \frac{-1}{N} \sum\limits_{n=1}^N \sum\limits_{k=1}^{K} H_{l_n,k}
        ///       \log(\hat{p}_{n,k})
        ///     @f$
        ///     where @f$ H_{l_n} @f$ denotes row @f$ l_n @f$ of @f$ H @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // The forward pass computes the softmax prob values.
            m_softmaxLayer.Forward(m_colSoftmaxBottomVec, m_colSoftmaxTopVec);

            Blob<T> blobInfoGain = (colBottom.Count < 3) ? m_blobInfoGain : colBottom[2];
            int nCount = 0;
            int nLabel = -1;
            double dfLoss = 0;
            double dfProb = 0;
            double dfProbLog = 0;
            double dfVal;

            if (typeof(T) == typeof(double))
            {
                double[] rgProbData = (double[])Convert.ChangeType(m_blobProb.update_cpu_data(), typeof(double[]));
                double[] rgBottomLabel = (double[])Convert.ChangeType(colBottom[1].update_cpu_data(), typeof(double[]));
                double[] rgInfoGainMat = (double[])Convert.ChangeType(blobInfoGain.update_cpu_data(), typeof(double[]));

                for (int i = 0; i < m_nOuterNum; i++)
                {
                    for (int j = 0; j < m_nInnerNum; j++)
                    {
                        nLabel = (int)rgBottomLabel[i * m_nInnerNum + j];
                        if (m_nIgnoreLabel.HasValue && m_nIgnoreLabel.Value == nLabel)
                            continue;

                        m_log.CHECK_GE(nLabel, 0, "The label should be greater than or equal to 0.");
                        m_log.CHECK_LT(nLabel, m_nNumLabels, "The label should be less than the number of labels '" + m_nNumLabels.ToString() + "'");

                        for (int l = 0; l < m_nNumLabels; l++)
                        {
                            dfProb = Math.Max(rgProbData[i * m_nInnerNum * m_nNumLabels + l * m_nInnerNum + j], kLOG_THRESHOLD);
                            dfProbLog = Math.Log(dfProb);
                            dfVal = rgInfoGainMat[nLabel * m_nNumLabels + l] * dfProbLog;
                            dfLoss -= dfVal;
                        }

                        nCount++;
                    }
                }
            }
            else
            {
                float[] rgProbData = (float[])Convert.ChangeType(m_blobProb.update_cpu_data(), typeof(float[]));
                float[] rgBottomLabel = (float[])Convert.ChangeType(colBottom[1].update_cpu_data(), typeof(float[]));
                float[] rgInfoGainMat = (float[])Convert.ChangeType(blobInfoGain.update_cpu_data(), typeof(float[]));

                for (int i = 0; i < m_nOuterNum; i++)
                {
                    for (int j = 0; j < m_nInnerNum; j++)
                    {
                        nLabel = (int)rgBottomLabel[i * m_nInnerNum + j];
                        if (m_nIgnoreLabel.HasValue && m_nIgnoreLabel.Value == nLabel)
                            continue;

                        m_log.CHECK_GE(nLabel, 0, "The label should be greater than or equal to 0.");
                        m_log.CHECK_LT(nLabel, m_nNumLabels, "The label should be less than the number of labels '" + m_nNumLabels.ToString() + "'");

                        for (int l = 0; l < m_nNumLabels; l++)
                        {
                            dfProb = Math.Max(rgProbData[i * m_nInnerNum * m_nNumLabels + l * m_nInnerNum + j], kLOG_THRESHOLD);
                            dfProbLog = Math.Log(dfProb);
                            dfVal = rgInfoGainMat[nLabel * m_nNumLabels + l] * dfProbLog;
                            dfLoss -= dfVal;
                        }

                        nCount++;
                    }
                }
            }

            double dfNormalizer = get_normalizer(m_normalization, nCount);
            double dfNormalizedLoss = dfLoss / dfNormalizer;
            colTop[0].SetData(dfNormalizedLoss, 0);

            if (colTop.Count == 2)
                colTop[1].ShareData(m_blobProb);
        }

        /// <summary>
        /// Computes the infogain loss error gradient w.r.t the predictions.
        /// </summary>
        /// <remarks>
        /// Gradients cannot be computed with respect to the label inputs (bottom[1]),
        /// so this method ignores bottom[1] and requires !propagate_down[1], crashing
        /// if propagate_down[1] == true.  (The same applies to the infogain matrix, if
        /// provided as bottom[2] rather than in the layer_param.)
        /// </remarks>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///   -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///      This blob's diff will simply contain the loss_weight * @f$ \lambda @f$ as
        ///      @f$ \lambda @f$ is the coefficient of this layer's output
        ///      @f$ \ell_i @f$ in the overall Net loss.
        ///      @f$ E = \lambda_i \ell_i + \mbox{other loss terms} @f$; hence
        ///      @f$ \frac{partial E}{\partial \ell_i} = \lambda_i @f$
        ///        (*Assuming that this top blob is not used as a bottom (input) by any
        ///        other layer of the Net.)
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.  propagate_down[1] must be false as
        /// we can't compute gradients with respect to the labels (similarly for progagate_down[2] and
        /// the infogain matrix, if provided as bottom[2]).</param>
        /// <param name="colBottom">bottom input blob vector (length 2-3)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ \hat{p} @f$; backward computes diff
        ///       @f$ \frac{\partial E}{\partial \hat{p}} @f$
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels -- ignored as we can't compute their error gradients.
        ///  -# @f$ (1 \times 1 \times K \times K) @f$
        ///     (\b optional) the information gain matrix -- ignored as its error
        ///     gradient computation is not implemented.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[1])
                m_log.FAIL(type.ToString() + " Layer cannot backpropagate to label inputs.");

            if (rgbPropagateDown.Count > 2 && rgbPropagateDown[2])
                m_log.FAIL(type.ToString() + " Layer cannot backpropagate to infogain inputs.");

            if (rgbPropagateDown[0])
            {
                Blob<T> blobInfoGainMat = m_blobInfoGain;
                int nDim = colBottom[0].count() / m_nOuterNum;
                int nCount = 0;
                int nLabelValue = 0;

                if (colBottom.Count >= 3)
                {
                    blobInfoGainMat = colBottom[2];
                    // H is provided as a 'bottom' and might change so sum rows every time.
                    sum_rows_of_H(colBottom[2]);
                }

                if (typeof(T) == typeof(double))
                {
                    double[] rgProbData = (double[])Convert.ChangeType(m_blobProb.update_cpu_data(), typeof(double[]));
                    double[] rgBottomLabel = (double[])Convert.ChangeType(colBottom[1].update_cpu_data(), typeof(double[]));
                    double[] rgInfoGainMat = (double[])Convert.ChangeType(blobInfoGainMat.update_cpu_data(), typeof(double[]));
                    double[] rgSumsRowsH = (double[])Convert.ChangeType(m_blobSumRowsOfH.update_cpu_data(), typeof(double[]));
                    double[] rgBottomDiff = (double[])Convert.ChangeType(colBottom[0].mutable_cpu_diff, typeof(double[]));

                    for (int i=0; i<m_nOuterNum; i++)
                    {
                        for (int j=0; j<m_nInnerNum; j++)
                        {
                            nLabelValue = (int)rgBottomLabel[i * m_nInnerNum + j];

                            m_log.CHECK_GE(nLabelValue, 0, "The label should be greater than or equal to 0.");
                            m_log.CHECK_LT(nLabelValue, m_nNumLabels, "The label should be less than the number of labels '" + m_nNumLabels.ToString() + "'");

                            if (m_nIgnoreLabel.HasValue && m_nIgnoreLabel.Value == nLabelValue)
                            {
                                for (int l = 0; l < m_nNumLabels; l++)
                                {
                                    rgBottomDiff[i * nDim + l * m_nInnerNum + j] = 0;
                                }
                            }
                            else
                            {
                                for (int l = 0; l < m_nNumLabels; l++)
                                {
                                    int nIdx = i * nDim + l * m_nInnerNum + j;
                                    rgBottomDiff[nIdx] = rgProbData[nIdx] * rgSumsRowsH[nLabelValue] - rgInfoGainMat[nLabelValue * m_nNumLabels + l];
                                }

                                nCount++;
                            }
                        }
                    }

                    colBottom[0].mutable_cpu_diff = convert(rgBottomDiff);
                }
                else
                {
                    float[] rgProbData = (float[])Convert.ChangeType(m_blobProb.update_cpu_data(), typeof(float[]));
                    float[] rgBottomLabel = (float[])Convert.ChangeType(colBottom[1].update_cpu_data(), typeof(float[]));
                    float[] rgInfoGainMat = (float[])Convert.ChangeType(blobInfoGainMat.update_cpu_data(), typeof(float[]));
                    float[] rgSumsRowsH = (float[])Convert.ChangeType(m_blobSumRowsOfH.update_cpu_data(), typeof(float[]));
                    float[] rgBottomDiff = (float[])Convert.ChangeType(colBottom[0].mutable_cpu_diff, typeof(float[]));

                    for (int i = 0; i < m_nOuterNum; i++)
                    {
                        for (int j = 0; j < m_nInnerNum; j++)
                        {
                            nLabelValue = (int)rgBottomLabel[i * m_nInnerNum + j];

                            m_log.CHECK_GE(nLabelValue, 0, "The label should be greater than or equal to 0.");
                            m_log.CHECK_LT(nLabelValue, m_nNumLabels, "The label should be less than the number of labels '" + m_nNumLabels.ToString() + "'");

                            if (m_nIgnoreLabel.HasValue && m_nIgnoreLabel.Value == nLabelValue)
                            {
                                for (int l = 0; l < m_nNumLabels; l++)
                                {
                                    rgBottomDiff[i * nDim + l * m_nInnerNum + j] = 0;
                                }
                            }
                            else
                            {
                                for (int l = 0; l < m_nNumLabels; l++)
                                {
                                    int nIdx = i * nDim + l * m_nInnerNum + j;
                                    rgBottomDiff[nIdx] = rgProbData[nIdx] * rgSumsRowsH[nLabelValue] - rgInfoGainMat[nLabelValue * m_nNumLabels + l];
                                }

                                nCount++;
                            }
                        }
                    }

                    colBottom[0].mutable_cpu_diff = convert(rgBottomDiff);
                }

                // Scale the gradient
                double dfNormalizer = get_normalizer(m_normalization, nCount);
                double dfLossWeight = convertD(colTop[0].GetDiff(0)) / dfNormalizer;
                m_cuda.scal(colBottom[0].count(), dfLossWeight, colBottom[0].mutable_gpu_diff);
            }
        }
    }
}
