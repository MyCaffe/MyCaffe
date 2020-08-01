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
    /// <remarks>
    /// Centroids:
    /// @see [A New Loss Function for CNN Classifier Based on Pre-defined Evenly-Distributed Class Centroids](https://arxiv.org/abs/1904.06008) by Qiuyu Zhu, Pengju Zhang, and Xin Ye, arXiv:1904.06008, 2019.
    /// 
    /// KNN:
    /// @see [Constellation Loss: Improving the efficiency of deep metric learning loss functions for optimal embedding](https://arxiv.org/abs/1905.10675) by Alfonso Medela and Artzai Picon, arXiv:1905.10675, 2019
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DecodeLayer<T> : Layer<T>
    {
        List<int> m_rgIgnoreLabels = new List<int>();
        int m_nCentroidOutputIteration = 300;
        int m_nCacheSize = 100;
        int m_nNum = 0;
        int m_nEncodingDim = 0;
        Blob<T> m_blobData;
        Blob<T> m_blobDistSq; 
        Blob<T> m_blobSummerVec;
        Blob<T> m_blobWork;
        int m_nLabelCount = 0;
        int m_nIteration = 0;
        long m_hMin = 0;
        long m_hMax = 0;
        double m_dfPreGenAlpha = 0;
        bool m_bInitializePreGenTargets = true;

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

            m_hMin = cuda.AllocHostBuffer(m_param.decode_param.k);
            m_hMax = cuda.AllocHostBuffer(m_param.decode_param.k);
            m_blobWork = new Blob<T>(cuda, log);
            m_blobWork.Name = "work";
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

            if (m_blobWork != null)
            {
                m_blobWork.Dispose();
                m_blobWork = null;
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
        /// Returns the min number of top blobs: distances, centroids
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
            m_rgIgnoreLabels = m_param.decode_param.ignore_labels;
            m_nEncodingDim = colBottom[0].count(1);

            if (m_param.decode_param.target != param.beta.DecodeParameter.TARGET.PREGEN)
            {
                if (m_param.decode_param.enable_centroid_update)
                {
                    m_nCentroidOutputIteration = m_param.decode_param.centroid_output_iteration;
                    if (m_nCentroidOutputIteration < 10)
                        m_log.WriteLine("WARNING: Centroid output iteration is set at " + m_nCentroidOutputIteration.ToString() + ", a value above 10 is recommended.");
                }
            }
            else
            {
                m_nCentroidOutputIteration = 0;
            }

            m_nCacheSize = m_param.decode_param.cache_size;
            m_log.CHECK_GT(m_nCacheSize, 0, "The cache size must be > 0.");

            m_dfPreGenAlpha = m_param.decode_param.pregen_alpha;

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

                Blob<T> blobStatus = new Blob<T>(m_cuda, m_log, false);
                blobStatus.Name = m_param.name + " status";
                blobStatus.reshape_when_sharing = true;

                List<int> rgStatusShape = new List<int>() { 0 }; // skip size check.
                if (!shareParameter(blobStatus, rgStatusShape))
                {
                    blobStatus.Reshape(1, 1, 1, 1); // This will be resized to the label count x 2
                    blobStatus.SetData(0);
                }

                m_colBlobs.Add(blobStatus);

                Blob<T> blobEncodingCounts = new Blob<T>(m_cuda, m_log, false);
                blobEncodingCounts.Name = m_param.name + " enc_counts";
                blobEncodingCounts.reshape_when_sharing = true;

                List<int> rgEncCountShape = new List<int>() { 0 }; // skip size check.
                if (!shareParameter(blobEncodingCounts, rgEncCountShape))
                {
                    blobEncodingCounts.Reshape(1, 1, 1, 1); // This will be resized to the label count.
                    blobEncodingCounts.SetData(0);
                }

                m_colBlobs.Add(blobEncodingCounts);

                if (m_param.decode_param.target == param.beta.DecodeParameter.TARGET.KNN)
                {
                    Blob<T> blobEncodings = new Blob<T>(m_cuda, m_log, false);
                    blobEncodings.Name = m_param.name + " encodings";
                    blobEncodings.reshape_when_sharing = true;

                    List<int> rgEncShape = new List<int>() { 0 }; // skip size check.
                    if (!shareParameter(blobEncodings, rgEncShape))
                    {
                        blobEncodings.Reshape(1, 1, m_nEncodingDim, 1); // This will be resized to the label count x nMaxItems x nEncDim.
                        blobEncodings.SetData(0);
                    }

                    m_colBlobs.Add(blobEncodings);
                }
            }

            m_nIteration = 0;
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

            m_log.CHECK_EQ(m_nEncodingDim, colBottom[0].count(1), "The encoding dim changed!");

            if (colBottom.Count > 1)
                m_log.CHECK_EQ(colBottom[1].num, m_nNum, "The number of labels does not match the number of items at bottom[0].");
        }


        /// <summary>
        /// Creates the pre-distanced pre-generated targets, only made public for testing.
        /// </summary>
        /// <param name="b">Specifies the blob to fill with pre-generated, pre-spaced targets.</param>
        /// <param name="dfMinDist">Specifies the minimum acceptable distance between all targets.</param>
        public void createPreGenTargets(Blob<T> b, double dfMinDist)
        {
            Random rand = new Random();
            bool bDone = false;
            int nNum = b.num;
            int nDim = b.count(1);
            float[] rgData = convertF(b.mutable_cpu_data);

            while (!bDone)
            {
                List<List<double>> rgDist = new List<List<double>>();

                double dfAbsMinDist = double.MaxValue;

                for (int i = 0; i < nNum; i++)
                {
                    rgDist.Add(new List<double>());

                    for (int j = 0; j < nNum; j++)
                    {
                        if (i != j)
                        {
                            double dfDiff = 0;
                            double dfDist = 0;

                            for (int k = 0; k < nDim; k++)
                            {
                                dfDiff = rgData[i * nDim + k] - rgData[j * nDim + k];
                                dfDist += (dfDiff * dfDiff);
                            }

                            rgDist[i].Add(dfDist);
                            dfAbsMinDist = Math.Min(dfAbsMinDist, dfDist);

                            if (dfDist < dfMinDist)
                            {
                                for (int k = 0; k < nDim; k++)
                                {
                                    rgData[j * nDim + k] += (float)(rand.NextDouble() * (dfMinDist / 4));
                                }
                            }
                        }
                    }
                }

                if (dfAbsMinDist > dfMinDist)
                    bDone = true;
            }

            b.mutable_cpu_data = convert(rgData);
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
        ///  -# @f$ (N \times L \times 1 \times 1) @f$
        ///     the computed distance of each item where the label with the smallest
        ///     distance represents the selected label.  The L dimension size equals
        ///     the number of labels in the data set (e.g. with MNIST L = 10).
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nItemNum = colBottom[0].num;
            int nItemCount = nItemNum * m_nCacheSize;
            int nLabelDim = 0;
            double dfAlpha = 1.0 / (double)nItemCount;
            double[] rgBottomLabel = null;

            if (m_param.phase == Phase.TRAIN)
            {
                nLabelDim = colBottom[1].count(1);
                m_log.CHECK(colBottom[1].count() % nLabelDim == 0, "The bottom[1] count must be a factor of 2 for {lbl1, lbl2}, or 3 for {anc, pos, neg}.");

                rgBottomLabel = convertD(colBottom[1].update_cpu_data());

                int nMaxLabel = rgBottomLabel.Max(p => (int)p);
                if (nMaxLabel > m_nLabelCount)
                {
                    int nNumLabels = nMaxLabel + 1;

                    if (m_param.decode_param.target == param.beta.DecodeParameter.TARGET.PREGEN)
                        nNumLabels = m_param.decode_param.pregen_label_count;

                    if (m_colBlobs[0].count() != nNumLabels * m_nEncodingDim)
                    {
                        m_colBlobs[0].Reshape(nNumLabels, m_nEncodingDim, 1, 1);
                        m_colBlobs[0].SetData(0);
                    }

                    if (m_colBlobs[1].count() != nNumLabels)
                    {
                        m_colBlobs[1].Reshape(nNumLabels, 1, 1, 1); // status
                        m_colBlobs[1].SetData(0);
                    }

                    if (m_colBlobs[2].count() != nNumLabels)
                    {
                        m_colBlobs[2].Reshape(nNumLabels, 1, 1, 1); // label counts
                        m_colBlobs[2].SetData(0);
                    }

                    if (m_param.decode_param.target == param.beta.DecodeParameter.TARGET.KNN)
                    {
                        if (m_colBlobs[3].count() != nNumLabels * nItemCount * m_nEncodingDim)
                        {
                            m_colBlobs[3].Reshape(nNumLabels, nItemCount, m_nEncodingDim, 1);
                            m_colBlobs[3].SetData(0);
                        }
                    }

                    m_nLabelCount = nNumLabels;
                }

                if (m_param.decode_param.target == param.beta.DecodeParameter.TARGET.PREGEN)
                {
                    if (m_bInitializePreGenTargets)
                    {
                        createPreGenTargets(m_colBlobs[0], m_dfPreGenAlpha * 2);
                        m_colBlobs[0].snapshot_requested = true;
                        m_bInitializePreGenTargets = false;
                    }
                }
            }

            int nActiveLabels = m_colBlobs[1].num - m_rgIgnoreLabels.Count;

            if (m_param.decode_param.target == param.beta.DecodeParameter.TARGET.KNN)
            {
                m_blobData.ReshapeLike(m_colBlobs[3]);
                m_blobSummerVec.Reshape(m_blobData.channels, 1, 1, 1);
                m_blobSummerVec.SetData(1.0);
                m_blobDistSq.ReshapeLike(m_blobSummerVec);
            }
            else
            {
                m_blobData.ReshapeLike(m_colBlobs[0]);
                m_blobSummerVec.Reshape(m_blobData.num, 1, 1, 1);
                m_blobSummerVec.SetData(1.0);
                m_blobDistSq.ReshapeLike(m_blobSummerVec);
            }

            if (nActiveLabels <= 0)
                nActiveLabels = m_colBlobs[0].num;

            colTop[0].Reshape(colBottom[0].num, m_colBlobs[0].num, 1, 1);

            for (int i = 0; i < colBottom[0].num; i++)
            {
                // When training, we calculate the targets during observations between nTargetStart and nTargetEnd.
                if (rgBottomLabel != null)
                {
                    // Pre-gen targets are ready to go.
                    if (m_param.decode_param.target == param.beta.DecodeParameter.TARGET.PREGEN)
                        m_colBlobs[1].SetData(1.0);

                    int nLabel = (int)rgBottomLabel[i * nLabelDim]; // Only the first embedding and first label are used (second is ignored).
                    int nReady = (int)convertD(m_colBlobs[1].GetData(nLabel));
                    int nLabelItemCount = (int)convertD(m_colBlobs[2].GetData(nLabel));

                    // Create the centroid when counts fall between Centroid Start and Centroid End by
                    // averaging all items within these counts together to create the centroid.
                    if (m_param.decode_param.target == param.beta.DecodeParameter.TARGET.CENTROID)
                    {
                        if (m_param.decode_param.enable_centroid_update)
                        {
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

                                if (nReady == 0 && !m_rgIgnoreLabels.Contains(nLabel))
                                    m_colBlobs[1].SetData(1.0, nLabel);
                            }
                        }
                        else
                        {
                            m_colBlobs[1].SetData(1.0);
                        }
                    }
                    // Save all items observed to the KNN cache.
                    else if (m_param.decode_param.target == param.beta.DecodeParameter.TARGET.KNN)
                    {
                        // Items added as a rolling list and are ordered by label, then by encoding as each encoding is received.  
                        int nSrcOff = i * m_nEncodingDim;
                        int nDstOff = (nLabel * nItemCount * m_nEncodingDim) + ((nLabelItemCount % nItemCount) * m_nEncodingDim);
                        m_cuda.copy(m_nEncodingDim, colBottom[0].gpu_data, m_colBlobs[3].mutable_gpu_data, nSrcOff, nDstOff);
                    }

                    m_colBlobs[2].SetData(nLabelItemCount + 1, nLabel);
                }

                // Request a snapshot when completed to make sure to save latest cache and centroids.
                int nCompletedTargets = (int)convertD(m_colBlobs[1].asum_data());
                if (nCompletedTargets == nActiveLabels)
                {
                    if (m_param.phase == Phase.TRAIN)
                        m_colBlobs[0].snapshot_requested = true;
                }

                m_log.CHECK_GE(m_blobData.num, m_colBlobs[0].num, "The data blob is not sized correctly!");

                // Load data with the current data embedding across each label 'slot' in blobData.
                int nCount = m_blobData.count();
                int nItems = m_blobData.num;

                if (m_param.decode_param.target == param.beta.DecodeParameter.TARGET.KNN)
                    nItems *= m_blobData.channels;

                m_cuda.fill(nItems, m_nEncodingDim, colBottom[0].gpu_data, i * m_nEncodingDim, nCount, m_blobData.mutable_gpu_data);

                // When using KNN, find the nearest neighbors from within the cached items.
                if (m_param.decode_param.target == param.beta.DecodeParameter.TARGET.KNN)
                {
                    if (nCompletedTargets == nActiveLabels)
                    {
                        m_blobDistSq.ReshapeLike(m_blobSummerVec);

                        m_cuda.sub(nCount,
                                   m_blobData.gpu_data,              // a
                                   m_colBlobs[3].gpu_data,           // b (saved encodings per label)
                                   m_blobData.mutable_gpu_diff);     // a_i - b_i

                        m_cuda.powx(nCount,
                                   m_blobData.gpu_diff,              // a_i - b_i
                                   2.0,
                                   m_blobData.mutable_gpu_diff);     // (a_i - b_i)^2

                        // Calculate distances of the label items.
                        int nDim = m_blobData.count(1);
                        float[] rgMinDist = new float[m_blobData.num];
                        for (int j = 0; j < m_blobData.num; j++)
                        {

                            m_cuda.gemv(false,
                                       m_blobData.channels,              // item count.
                                       m_blobData.height,                // encoding size.
                                       m_tOne,
                                       m_blobData.gpu_diff,              // (a_i - b_i)^2
                                       m_blobSummerVec.gpu_data,
                                       m_tZero,
                                       m_blobDistSq.mutable_gpu_data,    // \Sum (a_i - b_i)^2
                                       j * nDim,
                                       0,
                                       0);

                            m_cuda.minmax(m_blobDistSq.count(), 0, 0, 0, m_param.decode_param.k, m_hMin, m_hMax, true);
                            double[] rgMinD = m_cuda.GetHostMemoryDouble(m_hMin);
                            m_blobWork.Reshape((int)rgMinD[0], 1, 1, 1);
                            m_cuda.minmax(m_blobDistSq.count(), m_blobDistSq.gpu_data, m_blobWork.gpu_data, m_blobWork.gpu_diff, m_param.decode_param.k, m_hMin, m_hMax, true);

                            float[] rgMin = m_cuda.GetHostMemoryFloat(m_hMin);
                            List<float> rgMin1 = rgMin.Where(p => p < float.MaxValue).Take(m_param.decode_param.k).ToList();

                            rgMinDist[j] = rgMin1.Average();
                        }

                        m_blobDistSq.Reshape(rgMinDist.Length, 1, 1, 1);
                        m_blobDistSq.mutable_cpu_data = convert(rgMinDist);
                    }
                    else
                    {
                        m_blobDistSq.Reshape(m_blobData.num, 1, 1, 1);
                        m_blobDistSq.SetData(0);

                        if (i == 0 && m_param.phase != Phase.TRAIN)
                            m_log.WriteLine("WARNING: KNN cache still filling...");
                    }
                }
                // Otherwise, when using CENTROID or PREGEN, calculate the distance using the latest centroids.
                else
                {
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
                }

                // Set all ignore labels to the float maximum value.
                foreach (int nIgnoreLabel in m_rgIgnoreLabels)
                {
                    m_blobDistSq.SetData(float.MaxValue, nIgnoreLabel);
                }

                // The distances are returned in top[0], where the smallest distance is the detected label.
                m_cuda.copy(m_blobDistSq.num, m_blobDistSq.gpu_data, colTop[0].mutable_gpu_data, 0, i * m_blobDistSq.num);
            }

            // If we are to output the centroids, only do so when they are complete, otherwise output 0's.
            if (colTop.Count > 1)
            {
                colTop[1].ReshapeLike(m_colBlobs[0]);
                
                int nCompletedCentroids = (int)convertD(m_colBlobs[1].asum_data());
                if (m_nIteration >= m_nCentroidOutputIteration && nCompletedCentroids == nActiveLabels)
                {
                    m_cuda.copy(m_colBlobs[0].count(), m_colBlobs[0].gpu_data, colTop[1].mutable_gpu_data);
                }
                else
                {
                    if (m_phase != Phase.TRAIN)
                        m_log.WriteLine("WARNING: The centroids for the decode layer are not completed!  You must train the model first to calculate the centroids.");

                    colTop[1].SetData(0);
                }
            }

            m_nIteration++;
        }

        /// @brief Not implemented -- DecodeLayer cannot be used as a loss.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            // do nothing.
        }
    }
}
