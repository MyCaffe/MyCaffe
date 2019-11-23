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
    /// The AccuracyEncodingLayer computes the classification accuracy for an encoding used in a 
    /// classification task that uses a Siamese Network or similar type of net that creates an encoding 
    /// mapped to a label.
    /// This layer is initialized with the MyCaffe.param.AccuracyParameter.
    /// </summary>
    /// <remarks>
    /// @see [Convolutional Architecture Exploration for Action Recognition and Image Classification](https://arxiv.org/abs/1512.07502v1) by J. T. Turner, David Aha, Leslie Smith, and Kalyan Moy Gupta, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class AccuracyEncodingLayer<T> : Layer<T>
    {
        int m_nCentroidThreshold = 10;
        int m_nNum = 0;
        int m_nEncodingDim = 0;
        Blob<T> m_blobEncodings;
        Blob<T> m_blobData;
        Blob<T> m_blobDistSq; 
        Blob<T> m_blobSummerVec; 
        Dictionary<int, int> m_rgLabelCounts = new Dictionary<int, int>();

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides AccuracyParameter accuracy_param,
        /// with EncodingAccuracyLayer options:
        ///  - top_k (optional, default 1)
        ///          Sets the maximumrank k at which prediction is considered
        ///          correct, For example, if k = 5, a prediction is counted
        ///          correct if the correct label is among the top 5 predicted labels.</param>
        public AccuracyEncodingLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.ACCURACY_ENCODING;
            m_blobEncodings = new Blob<T>(cuda, log);
            m_blobEncodings.Name = m_param.name + " encodings";
            m_blobDistSq = new Blob<T>(cuda, log, false);
            m_blobDistSq.Name = m_param.name + " distsq";
            m_blobSummerVec = new Blob<T>(cuda, log, false);
            m_blobSummerVec.Name = m_param.name + " sum";
            m_blobData = new Blob<T>(cuda, log);
            m_blobData.Name = m_param.name + " data";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {

            if (m_blobEncodings != null)
            {
                m_blobEncodings.Dispose();
                m_blobEncodings = null;
            }

            if (m_blobDistSq != null)
            {
                m_blobDistSq.Dispose();
                m_blobDistSq = null;
            }

            if (m_blobSummerVec != null)
            {
                m_blobSummerVec.Dispose();
                m_blobSummerVec = null;
            }

            if (m_blobData != null)
            {
                m_blobData.Dispose();
                m_blobData = null;
            }

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();
                col.Add(m_blobEncodings);
                return col;
            }
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
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ((int)m_param.accuracy_param.top_k, 1, "Accuracy Encoding Layer only supports a topk = 1.");
            m_log.CHECK_EQ((int)m_param.accuracy_param.axis, 1, "Accuracy Encoding Layer expects axis to = 1.");

            if (m_param.accuracy_param.ignore_label.HasValue)
                m_log.WriteLine("WARNING: The Accuracy Encoding Layer does not use the 'ignore_label' parameter.");
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNum = colBottom[0].num;
            bool bFirstReshape = (nNum != m_nNum) ? true : false;
            m_nNum = nNum;
            m_nEncodingDim = colBottom[0].channels;

            m_log.CHECK_EQ(colBottom[1].num, m_nNum, "The number of labels does not match the number of items at bottom[0].");

            List<int> rgTopShape = new List<int>(); // Accuracy is a scalar; 0 axes.
            colTop[0].Reshape(rgTopShape);
            colTop[0].type = Blob<T>.BLOB_TYPE.ACCURACY;

            // vector of ones used to sum along channels.
            m_blobSummerVec.Reshape(colBottom[0].channels, 1, 1, 1);
            m_blobSummerVec.SetData(1.0);
        }

        /// <summary>
        /// Forward compuation.
        /// </summary>
        /// <param name="colBottom">bottom input blob (length 2)
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the encoding predictions @f$ x @f$, a blob with values in
        ///     @f$ [-\infty, +\infty] @f$ indicating the embedding of each of
        ///     the @f$ K @f$ classes.  Each embedding @f$ x @f$ is mapped to a predicted 
        ///     label.
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels l, an integer-valued blob with values
        ///     @f$ l_n \in [0, 1, 2, ..., K-1] @f$
        ///     indicating the correct class label among the @f$ K @f$ classes.
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed accuracy each calculated by finding the label with the minimum
        ///     distance to each encoding.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(colBottom[1].count() % 3, 0, "The bottom[1] count must be a factor of 3 for {sim, lbl1, lbl2}.");

            double dfAccuracy = 0;
            double[] rgBottomLabel = convertD(colBottom[1].update_cpu_data());
            int nMinCount = m_nCentroidThreshold;
            int nCorrectCount = 0;
            int nComparedCount = 0;
            List<int> rgLabels = new List<int>();
            int nIdx = 0;

            while (nIdx < rgBottomLabel.Length)
            {
                nIdx++;
                rgLabels.Add((int)rgBottomLabel[nIdx]);
                nIdx++;
                nIdx++;
            }

            int nMaxLabel = rgLabels.Max();
            if (nMaxLabel != m_rgLabelCounts.Count-1)
            {
                m_rgLabelCounts = new Dictionary<int, int>(nMaxLabel + 1);
                m_blobEncodings.Reshape(nMaxLabel + 1, m_nEncodingDim, 1, 1);
                m_blobData.Reshape(nMaxLabel + 1, m_nEncodingDim, 1, 1);
                m_blobDistSq.Reshape(nMaxLabel + 1, 1, 1, 1);
            }

            for (int i = 0; i < colBottom[0].num; i++)
            {
                int nLabel = rgLabels[i];

                if (!m_rgLabelCounts.ContainsKey(nLabel))
                {
                    m_rgLabelCounts.Add(nLabel, 1);
                    m_cuda.copy(m_nEncodingDim, colBottom[0].gpu_data, m_blobEncodings.mutable_gpu_data, i * m_nEncodingDim, nLabel * m_nEncodingDim);
                }
                else
                {
                    m_rgLabelCounts[nLabel]++;

                    double dfAlpha = (1.0 / (double)m_rgLabelCounts[nLabel]);
                    double dfBeta = ((double)(m_rgLabelCounts[nLabel] - 1) / m_rgLabelCounts[nLabel]);

                    // Add to centroids for each label.
                    m_cuda.add(m_nEncodingDim, colBottom[0].gpu_data, m_blobEncodings.gpu_data, m_blobEncodings.mutable_gpu_data, dfAlpha, dfBeta, i * m_nEncodingDim, nLabel * m_nEncodingDim, nLabel * m_nEncodingDim);
                }

                nMinCount = m_rgLabelCounts.Min(p => p.Value);
                if (nMinCount >= m_nCentroidThreshold)
                {
                    // Load data with the current data embedding across each label 'slot'.
                    for (int k = 0; k < m_rgLabelCounts.Count; k++)
                    {
                        m_cuda.copy(m_nEncodingDim, colBottom[0].gpu_data, m_blobData.mutable_gpu_data, i * m_nEncodingDim, k * m_nEncodingDim);
                    }

                    int nCount = m_blobData.count();

                    m_cuda.sub(nCount,
                               m_blobData.gpu_data,                // a
                               m_blobEncodings.gpu_data,           // b
                               m_blobEncodings.mutable_gpu_diff);  // a_i - b_i

                    m_cuda.powx(nCount,
                               m_blobEncodings.gpu_diff,           // a_i - b_i
                               2.0,
                               m_blobEncodings.mutable_gpu_diff);  // (a_i - b_i)^2

                    m_cuda.gemv(false,
                               m_blobData.num,                     // label count.
                               m_blobData.channels,                // encoding size.
                               1.0,
                               m_blobEncodings.gpu_diff,           // (a_i - b_i)^2
                               m_blobSummerVec.gpu_data,
                               0.0,
                               m_blobDistSq.mutable_gpu_data);  // \Sum (a_i - b_i)^2

                    // The label with the smallest distance is the detected label.
                    double[] rgLabelDist = convertD(m_blobDistSq.mutable_cpu_data);
                    int nDetectedLabel = -1;
                    double dfMin = double.MaxValue;

                    for (int l = 0; l < rgLabelDist.Length; l++)
                    {
                        if (rgLabelDist[l] < dfMin)
                        {
                            dfMin = rgLabelDist[l];
                            nDetectedLabel = l;
                        }
                    }

                    if (nDetectedLabel == nLabel)
                        nCorrectCount++;

                    nComparedCount++;
                }
            }

            dfAccuracy = (double)nCorrectCount / nComparedCount;

            colTop[0].SetData(dfAccuracy, 0);
            colTop[0].Tag = m_param.accuracy_param.top_k;
        }

        /// @brief Not implemented -- EncodingAccuracyLayer cannot be used as a loss.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
                throw new NotImplementedException();
        }
    }
}
