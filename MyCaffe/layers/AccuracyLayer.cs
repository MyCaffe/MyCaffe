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
    /// The AccuracyLayer computes the classification accuracy for a one-of-many
    /// classification task.
    /// This layer is initialized with the MyCaffe.param.AccuracyParameter.
    /// </summary>
    /// <remarks>
    /// @see [Convolutional Architecture Exploration for Action Recognition and Image Classification](https://arxiv.org/abs/1512.07502v1) by J. T. Turner, David Aha, Leslie Smith, and Kalyan Moy Gupta, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class AccuracyLayer<T> : Layer<T>
    {
        int m_nLabelAxis;
        int m_nOuterNum;
        int m_nInnerNum;
        int m_nTopK;
        int? m_nIgnoreLabel = null;
        Blob<T> m_blobNumsBuffer;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides AccuracyParameter accuracy_param,
        /// with AccuracyLayer options:
        ///  - top_k (optional, default 1)
        ///          Sets the maximumrank k at which prediction is considered
        ///          correct, For example, if k = 5, a prediction is counted
        ///          correct if the correct label is among the top 5 predicted labels.</param>
        public AccuracyLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.ACCURACY;
            m_blobNumsBuffer = new Blob<T>(cuda, log);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobNumsBuffer.Dispose();
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
            m_nTopK = (int)m_param.accuracy_param.top_k;
            m_nIgnoreLabel = m_param.accuracy_param.ignore_label;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_LE(m_nTopK, colBottom[0].count() / colBottom[1].count(), "top_k must be less than or equal to the number of classes.");

            m_nLabelAxis = colBottom[0].CanonicalAxisIndex(m_param.accuracy_param.axis);
            m_nOuterNum = colBottom[0].count(0, m_nLabelAxis);
            m_nInnerNum = colBottom[0].count(m_nLabelAxis + 1);

            m_log.CHECK_EQ(m_nOuterNum * m_nInnerNum, colBottom[1].count(), "Number of labels must match number of predictions; e.g., if label axis = 1 and prediction shape is (N, C, H, W), label count (number of labels) must be N*H*W, with integer values in {0, 1, ..., C=1}.");
            List<int> rgTopShape = new List<int>(); // Accuracy is a scalar; 0 axes.
            colTop[0].Reshape(rgTopShape);
            colTop[0].type = Blob<T>.BLOB_TYPE.ACCURACY;

            if (colTop.Count > 1)
            {
                // Per-class accuracy is a vector; 1 axes.
                List<int> rgTopShapePerClass = new List<int>() { colBottom[0].shape(m_nLabelAxis) };
                colTop[1].Reshape(rgTopShapePerClass);
                m_blobNumsBuffer.Reshape(rgTopShapePerClass);
            }
        }

        /// <summary>
        /// Forward compuation.
        /// </summary>
        /// <param name="colBottom">bottom input blob (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$, a blob with values in
        ///     @f$ [-\infty, +\infty] @f$ indicating the predicted score of each of
        ///     the @f$ K = CHW @f$ classes.  Each @f$ x_n @f$ is mapped to a predicted 
        ///     label @f$ \hat{l}_n @f$ given by its maximal index:
        ///     @f$ \hat{l}_n = \arg\max\limits_k x_{nk} @f$
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels l, an integer-valued blob with values
        ///     @f$ l_n \in [0, 1, 2, ..., K-1] @f$
        ///     indicating the correct class label among the @f$ K @f$ classes.</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed accuracy: @f$
        ///       \frac{1}{N} \sum\limits_{n=1}^N \delta\{ \hat{l}_n = l_n \}
        ///     @f$
        ///     where @f$ 
        ///       \delta\{\mathrm{condition}\} = \left\{
        ///         \begin{array}{lr}
        ///           1 & \mbox{if condition} \\
        ///           0 & \mbox{otherwise}
        ///         \end{array} \right.
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            double dfAccuracy = 0;
            double[] rgBottomData = convertD(colBottom[0].update_cpu_data());
            double[] rgBottomLabel = convertD(colBottom[1].update_cpu_data());
            int nDim = colBottom[0].count() / m_nOuterNum;
            int nNumLabels = colBottom[0].shape(m_nLabelAxis);
            List<double> rgMaxVal = Utility.Create<double>(m_nTopK + 1, 0.0);
            List<int> rgMaxIdx = Utility.Create<int>(m_nTopK + 1, 0);
            double[] rgNumsBuffer = null;
            double[] rgTopLabel = null;

            if (colTop.Count > 1)
            {
                m_blobNumsBuffer.SetData(0.0);
                colTop[1].SetData(0.0);
                rgTopLabel = convertD(colTop[1].mutable_cpu_data);
                rgNumsBuffer = convertD(m_blobNumsBuffer.update_cpu_data());
            }

            int nCount = 0;

            for (int i = 0; i < m_nOuterNum; i++)
            {
                for (int j = 0; j < m_nInnerNum; j++)
                {
                    int nLabelValue = (int)rgBottomLabel[i * m_nInnerNum + j];

                    if (m_nIgnoreLabel.HasValue && m_nIgnoreLabel.Value == nLabelValue)
                        continue;

                    if (colTop.Count > 1)
                        rgNumsBuffer[nLabelValue]++;

                    m_log.CHECK_GE(nLabelValue, 0, "The lable value must be >= 0.");
                    m_log.CHECK_LT(nLabelValue, nNumLabels, "The label value must be < " + nNumLabels.ToString() + ".  Make sure that the prototxt 'num_outputs' setting is > the highest label number.");

                    // Top-k accuracy
                    List<KeyValuePair<double, int>> rgBottomDataItems = new List<KeyValuePair<double, int>>();

                    for (int k = 0; k < nNumLabels; k++)
                    {
                        rgBottomDataItems.Add(new KeyValuePair<double, int>(rgBottomData[i * nDim + k * m_nInnerNum + j], k));
                    }

                    rgBottomDataItems.Sort(new Comparison<KeyValuePair<double, int>>(sortDataItems));

                    // Check if true label is in top_k predictions
                    for (int k = 0; k < m_nTopK; k++)
                    {
                        if (rgBottomDataItems[k].Value == nLabelValue)
                        {
                            dfAccuracy += 1.0;

                            if (colTop.Count > 1)
                                rgTopLabel[nLabelValue] += 1.0;

                            break;
                        }
                    }

                    nCount++;
                }
            }

            dfAccuracy /= (double)nCount;
            colTop[0].SetData(dfAccuracy, 0);
            colTop[0].Tag = m_param.accuracy_param.top_k;

            if (colTop.Count > 1)
            {
                for (int i = 0; i < colTop[1].count(); i++)
                {
                    double dfVal = 0.0;

                    if (rgNumsBuffer[i] != 0)
                        dfVal = rgTopLabel[i] / rgNumsBuffer[i];

                    rgTopLabel[i] = dfVal;
                }

                colTop[1].mutable_cpu_data = convert(rgTopLabel);
            }

            // Accuracy layer should not be used as a loss function.
        }

        private int sortDataItems(KeyValuePair<double, int> a, KeyValuePair<double, int> b)
        {
            if (a.Key < b.Key)
                return 1;

            if (a.Key > b.Key)
                return -1;

            return 0;
        }

        /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
                throw new NotImplementedException();
        }
    }
}
