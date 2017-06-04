using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.alpha
{
    /// <summary>
    /// <H3>PRE ALPHA</H3>
    /// 
    /// Knn Layer - this converts embeddings received into the nearest neighbor and outputs
    /// the inverse sum of distances between the input and all previously received inputs.
    /// 
    /// This layer is initialized with the MyCaffe.param.KnnParameter.
    /// </summary>
    /// <remarks>
    /// As an example, when using a 128 item embedding for a 10 class problem, the Knn layer
    /// takes each input and calculates the distance between the input and all other inputs
    /// collected for each class.  The resulting collection of distances are then summed for
    /// each class.  At this point the class with the lowest sum is the nearest neighbor.
    /// 
    /// However, in order to work with the Accuracy, SoftmaxLoss and Softmax layers, the
    /// summed values are normalized to the range between 0 and 1 and then inverted so that
    /// the maximum value is accociated with the nearest neighbor class.
    /// </remarks> 
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class KnnLayer<T> : Layer<T>
    {
        Blob<T> m_blobItem;
        Blob<T> m_blobCompare;
        BlobCollection<T> m_rgBatchData = new BlobCollection<T>();
        List<float[]> m_rgrgLabels = new List<float[]>();
        int m_nBatchSize;
        int m_nMaxBatches;
        int m_nCurrentBatchIdx = 0;
        int m_nNumOutput;
        int m_nVectorDim = 0;
        int m_nK = -1;
        bool m_bBufferFull = false;
        int m_nBatchDataCount = 0;
        int m_nIteration = 0;

        /// <summary>
        /// The KnnLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type KNN with parameter knn_param,
        /// with options:
        ///   - num_output (\b optional, default 10). The number of output items (e.g. classes).
        ///   
        ///   - k (\b optional, default 100). The number of nearest neighbors to compare (per class).
        ///   
        ///   - max_stored_batches (\b optional, default = 1000). The maximum number of batches to store before releasing batches.
        /// </param>
        public KnnLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_param = p;
            m_type = LayerParameter.LayerType.KNN;
            m_nNumOutput = p.knn_param.num_output;
            m_nK = p.knn_param.k;
            m_nMaxBatches = p.knn_param.max_stored_batches;
            m_blobItem = new common.Blob<T>(m_cuda, m_log);
            m_blobItem.Name = "item";
            m_blobCompare = new common.Blob<T>(m_cuda, m_log);
            m_blobCompare.Name = "compare";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_rgBatchData.Dispose();
            m_blobCompare.Dispose();
            m_blobItem.Dispose();
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobItem);
                col.Add(m_blobCompare);

                return col;
            }
        }

        /// <summary>
        /// Returns the minimum number of required bottom (intput) Blobs: data
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }   // data (embeddings) - only in RUN mode
        }

        /// <summary>
        /// Returns the maximum number of required bottom (intput) Blobs: data, label
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 2; }   // data (embeddings), label - only in TRAIN/TEST mode
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: knn
        /// </summary>
        public override int MinTopBlobs
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
            if (m_phase == Phase.TEST || m_phase == Phase.TRAIN)
                m_log.CHECK_EQ(2, colBottom.Count, "There should be two bottom items: data (embeddings) and labels.");

            m_nBatchSize = colBottom[0].shape(0);

            // Allocate the temp batch storage.
            for (int i = 0; i < m_nMaxBatches; i++)
            {
                Blob<T> data = new Blob<T>(m_cuda, m_log, false);
                data.ReshapeLike(colBottom[0]);
                m_rgBatchData.Add(data);
            }

            // Setup the weights (which stores the centroid embedding for each class)
            Blob<T> blobCentroid = new Blob<T>(m_cuda, m_log);
            List<int> rgShape = Utility.Clone<int>(colBottom[0].shape());
            rgShape[0] = m_nNumOutput;
            blobCentroid.Reshape(rgShape);
            m_colBlobs.Add(blobCentroid);

            for (int i = 0; i < colBottom.Count; i++)
            {
                m_param.propagate_down.Add(false);
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Reshape the temp batch storage.
            for (int i = 0; i < m_nMaxBatches; i++)
            {
                m_rgBatchData[i].ReshapeLike(colBottom[0]);
            }

            // Size the compare blob to one element int a single batch.
            List<int> rgShape = Utility.Clone<int>(colBottom[0].shape());
            rgShape[0] = 1;
            m_blobCompare.Reshape(rgShape);
            m_blobItem.Reshape(rgShape);
            m_nVectorDim = m_blobItem.count();

            // Setup the outputs where each item has 'num_output' elements, one per class.
            rgShape = new List<int>() { m_nBatchSize, m_nNumOutput, 1, 1 };
            colTop[0].Reshape(rgShape);

            // Reshape the weights (centroids)
            rgShape = Utility.Clone<int>(colBottom[0].shape());
            rgShape[0] = m_nNumOutput;
            m_colBlobs[0].Reshape(rgShape);

            m_nBatchDataCount = (m_bBufferFull) ? m_nMaxBatches : (m_nCurrentBatchIdx + 1);
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times K \times 1 \times 1) @f$ computed outputs, where @f$ K @f$ equals 
        ///     the <i>num_output</i> parameter. 
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // When training, move the centroids closer to their
            //  true locations, store in the weights.
            if (m_phase == Phase.TRAIN || m_param.phase == Phase.TRAIN) 
                forward_train(colBottom, colTop);

            // Find the closest centroid that matches the class.
            forward_test(colBottom, colTop);
        }

        /// <summary>
        /// Computes the forward calculation, run during the Phase.TEST cycle to find the 
        /// closest centroid stored in the internal blob cache.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times K \times 1 \times 1) @f$ computed outputs, where @f$ K @f$ equals 
        ///     the <i>num_output</i> parameter. 
        /// </param>
        protected void forward_test(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            List<float> rgFullSet = new List<float>();

            for (int i=0; i<m_nBatchSize; i++)
            {
                float[] rgDist = new float[m_nNumOutput];
                float fTotal = 0;

                for (int j = 0; j < m_nNumOutput; j++)
                {
                    m_cuda.sub(m_blobCompare.count(), colBottom[0].gpu_data, m_colBlobs[0].gpu_data, m_blobCompare.mutable_gpu_data, i * m_nVectorDim, j * m_nVectorDim);
                    rgDist[j] = m_cuda.dot_float(m_blobCompare.count(), m_blobCompare.gpu_data, m_blobCompare.gpu_data);
                    fTotal += rgDist[j];
                }

                // Normalize and invert so that the shortest distance has the largest value for that class
                // Softmax and Accuracy layers look for the max.
                for (int j=0; j<m_nNumOutput; j++)
                {
                    rgDist[j] = 1.0f - (rgDist[j] / fTotal);
                }

                rgFullSet.AddRange(rgDist);
            }

            colTop[0].mutable_cpu_data = convert(rgFullSet.ToArray());
        }

        /// <summary>
        /// Computes the forward calculation, run during the Phase.TRAIN cycle to store the batch
        /// in the internal cache and calculate the centroids using the nearest neighbors.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times K \times 1 \times 1) @f$ computed outputs, where @f$ K @f$ equals 
        ///     the <i>num_output</i> parameter. 
        /// </param>
        protected void forward_train(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(2, colBottom.Count, "When training, the bottom must have both 'data' and 'labels'.");

            if (m_nCurrentBatchIdx == m_nMaxBatches)
            {
                m_bBufferFull = true;
                m_nCurrentBatchIdx = 0;
            }

            // Copy the data into the batch storage.
            m_cuda.copy(colBottom[0].count(), colBottom[0].gpu_data, m_rgBatchData[m_nCurrentBatchIdx].mutable_gpu_data);

            // Copy the labels into the label storage on the host side.
            float[] rgLabels = convertF(colBottom[1].update_cpu_data());
            if (m_bBufferFull)
                m_rgrgLabels[m_nCurrentBatchIdx] = rgLabels;
            else
                m_rgrgLabels.Add(rgLabels);

            colTop[0].SetData(0.0);

            // Calculate all of the distances between each item within the current batch and
            // all other items.
            List<DistItem> rgDistances = new List<DistItem>();
            for (int j = 0; j < m_nBatchSize; j++)
            {
                m_cuda.copy(m_blobItem.count(), m_rgBatchData[m_nCurrentBatchIdx].gpu_data, m_blobItem.mutable_gpu_data, j * m_nVectorDim);
                int nLabel = (int)m_rgrgLabels[m_nCurrentBatchIdx][j];

                for (int l = 0; l < m_nBatchDataCount; l++)
                {
                    if (l != m_nCurrentBatchIdx)
                    {
                        for (int m = 0; m < m_nBatchSize; m++)
                        {
                            // Only add if we have not already made the distance comparison.
                            // For example, if a->b distance has been added, don't add b->a.
                            int nDuplicateCount = rgDistances.Where(p => p.CompareBatchIdx == m_nCurrentBatchIdx && p.CompareItemIdx == j && p.MainBatchIdx == l && p.MainItemIdx == m).Count();

                            if (nDuplicateCount == 0)
                            {
                                m_cuda.sub(m_blobCompare.count(), m_blobItem.gpu_data, m_rgBatchData[l].gpu_data, m_blobCompare.mutable_gpu_data, 0, m * m_nVectorDim);
                                int nCompareLabel = (int)m_rgrgLabels[l][m];
                                double dfDist = m_cuda.dot_double(m_blobCompare.count(), m_blobCompare.gpu_data, m_blobCompare.gpu_data);
                                rgDistances.Add(new DistItem(nLabel, nCompareLabel, m_nCurrentBatchIdx, j, l, m, dfDist));
                            }
                        }
                    }
                }
            }

            // Make sure to compare at least num_output+1 items to avoid ties.
            int nCompareCount = m_nK;
            if (nCompareCount < m_nNumOutput + 1)
                nCompareCount = m_nNumOutput + 1;

            if (rgDistances.Count >= nCompareCount)
            {
                // Sort all, with shortest first.
                rgDistances = rgDistances.OrderBy(p => p.Distance).ToList();

                // Load K items with the shortest distances into their respective classes.
                Dictionary<int, List<DistItem>> rgClassItems = new Dictionary<int, List<DistItem>>();
                for (int i = 0; i < nCompareCount; i++)
                {
                    DistItem item = rgDistances[i];
                    if (!rgClassItems.ContainsKey(item.MainItemLabel))
                        rgClassItems.Add(item.MainItemLabel, new List<DistItem>());

                    rgClassItems[item.MainItemLabel].Add(item);
                }

                // Go through the cache of batches and find the most 'centered' point for each class and set
                //  it as the centroid in the weights - the centered values are used to find the class
                //  when running in TEST or RUN mode.
                m_colBlobs[0].SetDiff(0);

                Dictionary<int, CentroidItem<T>> rgCentroids = new Dictionary<int, CentroidItem<T>>();
                for (int i = 0; i < m_nNumOutput; i++)
                {
                    rgCentroids.Add(i, new CentroidItem<T>(m_colBlobs[0], i, m_nVectorDim));
                }

                // Calculate the average data point for each class.
                foreach (KeyValuePair<int, List<DistItem>> kv in rgClassItems)
                {
                    int nLabel = kv.Key;
                    CentroidItem<T> centroid = rgCentroids[nLabel];
                    double dfScale = 1.0 / (double)kv.Value.Count;

                    for (int j = 0; j < kv.Value.Count; j++)
                    {
                        int nOffset = kv.Value[j].MainItemIdx * m_nVectorDim;
                        m_cuda.add(centroid.VectorDim, centroid.Data.gpu_diff, m_rgBatchData[m_nCurrentBatchIdx].gpu_data, centroid.Data.mutable_gpu_diff, 1.0, dfScale, centroid.Offset, nOffset, centroid.Offset);
                    }
                }

                // Scale the current centroid by half and add half of the new centroid
                // to 'move' toward the real centroid.
                double dfScale1 = (m_nIteration == 0) ? 1.0 : 0.5;
                m_cuda.add(m_colBlobs[0].count(), m_colBlobs[0].gpu_data, m_colBlobs[0].gpu_diff, m_colBlobs[0].mutable_gpu_data, dfScale1, dfScale1);
                m_nIteration++;
            }

            m_nCurrentBatchIdx++;
        }


        /// @brief Not implemented - the KNN Layer does not perform backward.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
                throw new NotImplementedException();
        }
    }

    class CentroidItem<T> /** @private */
    {
        Blob<T> m_blob;
        int m_nOffset;
        int m_nVectorDim;
        int m_nCount;

        public CentroidItem(Blob<T> blob, int nOffset, int nVectorDim)
        {
            m_blob = blob;
            m_nOffset = nOffset * nVectorDim;
            m_nVectorDim = nVectorDim;
            m_nCount = 0;
        }

        public Blob<T> Data
        {
            get { return m_blob; }
        }

        public int Offset
        {
            get { return m_nOffset; }
        }

        public int VectorDim
        {
            get { return m_nVectorDim; }
        }

        public int Count
        {
            get { return m_nCount; }
            set { m_nCount = value; }
        }
    }

    class DistItem /** @private */
    {
        int m_nMainItemLabel;
        int m_nMainBatchIdx;
        int m_nMainItemIdx;
        int m_nCompareItemLabel;
        int m_nCompareBatchCacheIdx;
        int m_nCompareBatchCacheItemIdx;
        double m_dfDistanceMainToCompare;

        public DistItem(float fMainItemLabel, float fCompareItemLabel, int nMainBatchIdx, int nMainItemIdx, int nCompareBatchIdx, int nCompareDistItemIdx, double dfDistance)
        {
            m_nMainItemLabel = (int)fMainItemLabel;
            m_nMainBatchIdx = nMainBatchIdx;
            m_nMainItemIdx = nMainItemIdx;
            m_nCompareItemLabel = (int)fCompareItemLabel;
            m_nCompareBatchCacheIdx = nCompareBatchIdx;
            m_nCompareBatchCacheItemIdx = nCompareDistItemIdx;
            m_dfDistanceMainToCompare = dfDistance;
        }

        public int MainItemLabel
        {
            get { return m_nMainItemLabel; }
        }

        public int MainBatchIdx
        {
            get { return m_nMainBatchIdx; }
        }

        public int MainItemIdx
        {
            get { return m_nMainItemIdx; }
        }

        public int CompareItemLabel
        {
            get { return m_nCompareItemLabel; }
        }

        public int CompareBatchIdx
        {
            get { return m_nCompareBatchCacheIdx; }
        }

        public int CompareItemIdx
        {
            get { return m_nCompareBatchCacheItemIdx; }
        }

        public double Distance
        {
            get { return m_dfDistanceMainToCompare; }
        }

        public override string ToString()
        {
            return "[" + m_nMainItemLabel.ToString() + "](" + m_nMainBatchIdx.ToString() + "," + m_nMainItemIdx.ToString() + ") -> [" + m_nCompareItemLabel.ToString() + "](" + m_nCompareBatchCacheIdx.ToString() + "," + m_nCompareBatchCacheItemIdx.ToString() + ") = " + m_dfDistanceMainToCompare.ToString();
        }
    }
}
