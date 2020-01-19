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
    /// The AccuracyEncodingLayer computes the classification accuracy for an encoding used in a 
    /// classification task that uses a Siamese Network or similar type of net that creates an encoding 
    /// mapped to a label.
    /// This layer is initialized with the MyCaffe.param.AccuracyParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class AccuracyEncodingLayer<T> : Layer<T>
    {
        int m_nCacheSize = 100;
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
                col.Add(m_blobDistSq);
                col.Add(m_blobSummerVec);
                col.Add(m_blobData);
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
        /// Returns the number of top blobs: accuracy
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

            m_nCacheSize = m_param.decode_param.cache_size;
            m_log.CHECK_GT(m_nCacheSize, 0, "The cache size must be > 0.");

            if (m_colBlobs.Count == 0)
            {
                Blob<T> blobCentroids = new Blob<T>(m_cuda, m_log, false);
                blobCentroids.Name = m_param.name + " centroids";
                blobCentroids.reshape_when_sharing = true;

                List<int> rgCentroidShape = new List<int>() { 0 }; // skip size check.
                if (!shareParameter(blobCentroids, rgCentroidShape))
                {
                    blobCentroids.Reshape(2, m_nEncodingDim, 1, 1); // set to at least two labels initially (may get expanded in forward).
                    blobCentroids.SetData(0);
                }

                m_colBlobs.Add(blobCentroids);
            }
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
            colTop[0].type = BLOB_TYPE.ACCURACY;

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
            m_log.CHECK_EQ(colBottom[1].count() % 2, 0, "The bottom[1] count must be a factor of 2 for {lbl1, lbl2}.");
            int nItemNum = colBottom[0].num;
            int nItemCount = nItemNum * m_param.decode_param.cache_size;
            double dfAlpha = 1.0 / (double)nItemCount;

            double dfAccuracy = 0;
            double[] rgBottomLabel = convertD(colBottom[1].update_cpu_data());
            int nCorrectCount = 0;
            int nComparedCount = 0;

            int nMaxLabel = rgBottomLabel.Max(p => (int)p);
            int nMaxKey = (m_rgLabelCounts.Count == 0) ? 0 : m_rgLabelCounts.Max(p => p.Key);
            if (nMaxLabel > nMaxKey)
            {
                int nNumLabels = nMaxLabel + 1;

                m_colBlobs[0].Reshape(nNumLabels, m_nEncodingDim, 1, 1);
                m_colBlobs[0].SetData(0);
                m_blobData.Reshape(nNumLabels, m_nEncodingDim, 1, 1);
                m_blobDistSq.Reshape(nNumLabels, 1, 1, 1);
                m_rgLabelCounts.Clear();
            }

            for (int i = 0; i < colBottom[0].num; i++)
            {
                int nLabel = (int)rgBottomLabel[i * 2]; // Only the first embedding and first label are used (second is ignored).
                int nLabelItemCount = 0;

                if (m_rgLabelCounts.ContainsKey(nLabel))
                    nLabelItemCount = m_rgLabelCounts[nLabel];

                // Create the centroid when counts fall between Centroid Start and Centroid End by
                // averaging all items within these counts together to create the centroid.
                if (nLabelItemCount == 0)
                {
                    // Add initial centroid portion for the label.
                    m_cuda.copy(m_nEncodingDim, colBottom[0].gpu_data, m_colBlobs[0].mutable_gpu_data, i * m_nEncodingDim, nLabel * m_nEncodingDim);
                    m_cuda.scale(m_nEncodingDim, convert(dfAlpha), m_colBlobs[0].gpu_data, m_colBlobs[0].mutable_gpu_data, nLabel * m_nEncodingDim, nLabel * m_nEncodingDim);
                }
                else if (nLabelItemCount < nItemCount)
                {
                    dfAlpha = 1.0 / (nLabelItemCount + 1);
                    // Add portion of current item to centroids for the label.
                    m_cuda.add(m_nEncodingDim, colBottom[0].gpu_data, m_colBlobs[0].gpu_data, m_colBlobs[0].mutable_gpu_data, dfAlpha, 1.0 - dfAlpha, i * m_nEncodingDim, nLabel * m_nEncodingDim, nLabel * m_nEncodingDim);
                }
                else
                {
                    // Add portion of current item to centroids for the label.
                    m_cuda.add(m_nEncodingDim, colBottom[0].gpu_data, m_colBlobs[0].gpu_data, m_colBlobs[0].mutable_gpu_data, dfAlpha, 1.0 - dfAlpha, i * m_nEncodingDim, nLabel * m_nEncodingDim, nLabel * m_nEncodingDim);
                }

                m_colBlobs[0].snapshot_requested = true;

                if (!m_rgLabelCounts.ContainsKey(nLabel))
                    m_rgLabelCounts.Add(nLabel, 1);
                else
                    m_rgLabelCounts[nLabel]++;

                // Load data with the current data embedding across each label 'slot' in blobData.
                int nCount = m_blobData.count();
                int nItems = m_blobData.num;
                m_cuda.fill(nItems, m_nEncodingDim, colBottom[0].gpu_data, i * m_nEncodingDim, nCount, m_blobData.mutable_gpu_data);

                m_cuda.sub(nCount,
                           m_blobData.gpu_data,              // a
                           m_colBlobs[0].gpu_data,           // b (centroid)
                           m_blobData.mutable_gpu_diff);     // a_i - b_i

                m_cuda.powx(nCount,
                           m_blobData.gpu_diff,              // a_i - b_i
                           2.0,
                           m_blobData.mutable_gpu_diff);     // (a_i - b_i)^2

                m_cuda.gemv(false,
                           m_blobData.num,                   // label count.
                           m_blobData.channels,              // encoding size.
                           1.0,
                           m_blobData.gpu_diff,              // (a_i - b_i)^2
                           m_blobSummerVec.gpu_data,
                           0.0,
                           m_blobDistSq.mutable_gpu_data);   // \Sum (a_i - b_i)^2

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

            dfAccuracy = (nComparedCount == 0) ? 0 : (double)nCorrectCount / nComparedCount;

            colTop[0].SetData(dfAccuracy, 0);
            colTop[0].Tag = m_param.accuracy_param.top_k;
        }

        /// @brief Not implemented -- EncodingAccuracyLayer cannot be used as a loss.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            // do nothing.
        }
    }
}
