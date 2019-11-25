using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The DecodeLayer decodes the label of a classification for an encoding produced by a Siamese Network or similar type of net that creates 
    /// an encoding mapped to a set of distances where the smallest distance indicates the label for which the encoding belongs.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DecodeLayer<T> : Layer<T>
    {
        int m_nCentroidThreshold = 20;
        double m_dfMinAlpha = 0.0001;
        int m_nNum = 0;
        int m_nEncodingDim = 0;
        Blob<T> m_blobData;
        Blob<T> m_blobDistSq; 
        Blob<T> m_blobSummerVec;
        T[] m_rgTopLabels = null;
        Dictionary<int, int> m_rgLabelCounts = new Dictionary<int, int>();

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides the generic parameter for the DecodeLayer.</param>
        public DecodeLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.DECODE;
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
                col.Add(m_blobDistSq);
                col.Add(m_blobSummerVec);
                col.Add(m_blobData);
                return col;
            }
        }

        /// <summary>
        /// Returns the minimum number of bottom blobs used: predicted (RUN phase)
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of bottom blobs used: predicted, label (TRAIN and TEST phase)
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the min number of top blobs: distances
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the min number of top blobs: distances, labels
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
            m_nEncodingDim = colBottom[0].channels;

            m_nCentroidThreshold = m_param.decode_param.centroid_threshold;
            m_log.CHECK_GE(m_nCentroidThreshold, 10, "The centroid threshold must be >= 10, and the recommended setting is 20.");
            m_dfMinAlpha = m_param.decode_param.min_alpha;
            m_log.CHECK_GE(m_dfMinAlpha, 0, "The minimum alpha must be >= 0.");

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

            if (colBottom.Count > 1)
                m_log.CHECK_EQ(colBottom[1].num, m_nNum, "The number of labels does not match the number of items at bottom[0].");

            // vector of ones used to sum along channels.
            m_blobSummerVec.Reshape(colBottom[0].channels, 1, 1, 1);
            m_blobSummerVec.SetData(1.0);

            if (colTop.Count > 1)
            {
                colTop[1].Reshape(colBottom[0].num, 1, 1, 1);
                int nCount = colTop[0].count();
                if (m_rgTopLabels == null || m_rgTopLabels.Length != nCount)
                    m_rgTopLabels = new T[nCount];
            }
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
            int nMinCount = m_nCentroidThreshold;
            double[] rgBottomLabel = null;

            if (m_param.phase == Phase.TRAIN)
            {
                m_log.CHECK_EQ(colBottom[1].count() % 2, 0, "The bottom[1] count must be a factor of 2 for {lbl1, lbl2}.");
                rgBottomLabel = convertD(colBottom[1].update_cpu_data());

                int nMaxLabel = rgBottomLabel.Max(p => (int)p);
                if (nMaxLabel != m_rgLabelCounts.Count - 1)
                {
                    int nNumLabels = nMaxLabel + 1;

                    m_colBlobs[0].Reshape(nNumLabels, m_nEncodingDim, 1, 1);
                    m_colBlobs[0].SetData(0);
                    m_blobData.Reshape(nNumLabels, m_nEncodingDim, 1, 1);
                    m_blobDistSq.Reshape(nNumLabels, 1, 1, 1);
                    m_rgLabelCounts.Clear();
                }
            }
            else
            {
                m_blobData.ReshapeLike(m_colBlobs[0]);
                m_blobDistSq.Reshape(m_colBlobs[0].num, 1, 1, 1);
            }

            colTop[0].Reshape(colBottom[0].num, m_colBlobs[0].num, 1, 1);

            for (int i = 0; i < colBottom[0].num; i++)
            {
                if (rgBottomLabel != null)
                {
                    int nLabel = (int)rgBottomLabel[i * 2]; // Only the first embedding and first label are used (second is ignored).

                    if (!m_rgLabelCounts.ContainsKey(nLabel))
                    {
                        m_rgLabelCounts.Add(nLabel, 1);
                        m_cuda.copy(m_nEncodingDim, colBottom[0].gpu_data, m_colBlobs[0].mutable_gpu_data, i * m_nEncodingDim, nLabel * m_nEncodingDim);
                    }
                    else
                    {
                        m_rgLabelCounts[nLabel]++;

                        double dfAlpha = (1.0 / (double)m_rgLabelCounts[nLabel]);
                        if (dfAlpha < m_dfMinAlpha)
                            dfAlpha = m_dfMinAlpha;

                        double dfBeta = 1.0 - dfAlpha;

                        // Add to centroids for each label.
                        m_cuda.add(m_nEncodingDim, colBottom[0].gpu_data, m_colBlobs[0].gpu_data, m_colBlobs[0].mutable_gpu_data, dfAlpha, dfBeta, i * m_nEncodingDim, nLabel * m_nEncodingDim, nLabel * m_nEncodingDim);
                    }
                }

                if (m_phase != Phase.TRAIN || (m_rgLabelCounts.Count > 0 && m_rgLabelCounts.Min(p => p.Value) >= m_nCentroidThreshold))
                {
                    int nLabelCount = m_colBlobs[0].num;
                    if (nLabelCount == 0)
                        break;

                    // Load data with the current data embedding across each label 'slot'.
                    for (int k = 0; k < nLabelCount; k++)
                    {
                        m_cuda.copy(m_nEncodingDim, colBottom[0].gpu_data, m_blobData.mutable_gpu_data, i * m_nEncodingDim, k * m_nEncodingDim);
                    }

                    int nCount = m_blobData.count();

                    m_cuda.sub(nCount,
                               m_blobData.gpu_data,              // a
                               m_colBlobs[0].gpu_data,           // b
                               m_blobData.mutable_gpu_diff);  // a_i - b_i

                    m_cuda.powx(nCount,
                               m_blobData.gpu_diff,           // a_i - b_i
                               2.0,
                               m_blobData.mutable_gpu_diff);  // (a_i - b_i)^2

                    m_cuda.gemv(false,
                               m_blobData.num,                   // label count.
                               m_blobData.channels,              // encoding size.
                               1.0,
                               m_blobData.gpu_diff,           // (a_i - b_i)^2
                               m_blobSummerVec.gpu_data,
                               0.0,
                               m_blobDistSq.mutable_gpu_data);   // \Sum (a_i - b_i)^2

                    // The distances are returned in top[0], where the smallest distance is the detected label.
                    m_cuda.copy(nLabelCount, m_blobDistSq.gpu_data, colTop[0].mutable_gpu_data, 0, i * nLabelCount);

                    // The label with the smallest distance is the detected label.
                    if (m_rgTopLabels != null)
                    {
                        double[] rgdfLabelDist = convertD(m_blobDistSq.mutable_cpu_data);
                        int nDetectedLabel = -1;
                        double dfMin = double.MaxValue;

                        for (int l = 0; l < rgdfLabelDist.Length; l++)
                        {
                            if (rgdfLabelDist[l] < dfMin)
                            {
                                dfMin = rgdfLabelDist[l];
                                nDetectedLabel = l;
                            }
                        }

                        m_rgTopLabels[i] = Utility.ConvertVal<T>((double)nDetectedLabel);
                    }
                }
            }

            if (colTop.Count > 1)
                colTop[1].mutable_cpu_data = m_rgTopLabels;
        }

        /// @brief Not implemented -- DecodeLayer cannot be used as a loss.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
                throw new NotImplementedException();
        }
    }
}
