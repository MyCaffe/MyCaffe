using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// <H3>BETA</H3>
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
    /// 
    /// IMPORTANT: The KNN layer requires that both the training and testing phases use 
    /// the same batch sizes and requires both the 'data' and 'label' bottom items.
    /// </remarks> 
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class KnnLayer<T> : Layer<T>
    {
        Blob<T> m_blobCompare;
        int m_nBatchSize;
        int m_nMaxBatches;
        int m_nCurrentBatchIdx = 0;
        int m_nNumOutput;
        int m_nVectorDim = 0;
        int m_nK = -1;
        bool m_bBufferFull = false;
        int m_nBatchDataCount = 0;

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
            m_blobCompare = new common.Blob<T>(m_cuda, m_log, false);
            m_blobCompare.Name = "compare";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobCompare.Dispose();
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

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

            // Make sure to not have a diff so as to avoid
            // being normalized or regularized in the
            // ApplyUpdate.
            Blob<T> blobInfo = new Blob<T>(m_cuda, m_log, false);
            blobInfo.Name = m_param.name + " info";
            blobInfo.Reshape(1, 1, 1, 1);
            blobInfo.SetData(0, 0);
            m_colBlobs.Add(blobInfo);

            // Setup the weights where
            //  weight[0] stores the last 'max' batches and
            //  weight[1] stores the last 'max' labels
            for (int i = 0; i < m_nMaxBatches; i++)
            {
                Blob<T> blobData = new Blob<T>(m_cuda, m_log, false);
                blobData.Name = m_param.name + " data_" + i.ToString();
                Blob<T> blobLabel = new Blob<T>(m_cuda, m_log, false);
                blobLabel.Name = m_param.name + " label_" + i.ToString();
                m_colBlobs.Add(blobData);
                m_colBlobs.Add(blobLabel);
            }

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
                m_colBlobs[1 + (i * 2 + 0)].ReshapeLike(colBottom[0]);

                if (colBottom.Count > 1)
                    m_colBlobs[1 + (i * 2 + 1)].ReshapeLike(colBottom[1]);
            }

            // Size the compare blob to one element int a single batch.
            List<int> rgShape = Utility.Clone<int>(colBottom[0].shape());
            rgShape[0] = 1;
            m_blobCompare.Reshape(rgShape);
            m_nVectorDim = m_blobCompare.count();

            // Setup the outputs where each item has 'num_output' elements, one per class.
            rgShape = new List<int>() { m_nBatchSize, m_nNumOutput, 1, 1 };
            colTop[0].Reshape(rgShape);

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
            if (m_phase == Phase.TRAIN || m_param.phase == Phase.TRAIN)
            {
                // Save the last 'max' items.
                forward_save(colBottom, colTop);
                return;
            }

            colTop[0].SetData(0);

            // Find the label with the closest (smallest)
            // averaged distance.
            forward_test(colBottom, colTop);
        }

        /// <summary>
        /// Computes the forward calculation, run during the Phase.TEST cycle to find the 
        /// closest averaged distance stored in the inernal blobs.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times K \times 1 \times 1) @f$ computed outputs, where @f$ K @f$ equals 
        ///     the <i>num_output</i> parameter. 
        /// </param>
        protected void forward_test(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            float fDataCount = convertF(m_colBlobs[0].GetData(0));
            int nDataCount = (int)fDataCount;
           
            if (nDataCount == 0)
                return;

            m_log.CHECK_EQ(colBottom.Count, 2, "The KNN Layer is used for testing and expects both the 'data' and 'label' bottom items.");

            Dictionary<int, List<Tuple<int, int>>> rgData = new Dictionary<int, List<Tuple<int, int>>>();
            float[] rgFullSet = new float[m_nBatchSize * m_nNumOutput];

            // Organize all stored data items by label.
            for (int i = 0; i < nDataCount; i++)
            {
                Blob<T> blobLabel = m_colBlobs[1 + (i * 2 + 1)];
                float[] rgLabels1 = convertF(blobLabel.update_cpu_data());

                for (int j = 0; j < rgLabels1.Length; j++)
                {
                    int nLabel = (int)rgLabels1[j];

                    if (!rgData.ContainsKey(nLabel))
                        rgData.Add(nLabel, new List<Tuple<int, int>>());

                    rgData[nLabel].Add(new Tuple<int, int>(i, j));
                }
            }

            // Get the current set of labels.
            float[] rgLabels = convertF(colBottom[1].update_cpu_data());
            Stopwatch sw = new Stopwatch();

            sw.Start();

            // Calculate all distances between each item in the current bottom and those in the stored
            //  blobs 'weight' data (which are actually the last trained 'max' data items and labels).
            for (int i = 0; i < m_nBatchSize; i++)
            {
                int nLabel = (int)rgLabels[i];
                Dictionary<int, float> rgKDist = new Dictionary<int, float>();

                foreach (KeyValuePair<int, List<Tuple<int, int>>> kvItem in rgData.OrderBy(p => p.Key))
                {
                    List<float> rgDist = new List<float>();

                    foreach (Tuple<int, int> offset in kvItem.Value)
                    {
                        Blob<T> blobData = m_colBlobs[1 + (offset.Item1 * 2 + 0)];
                        m_cuda.sub(m_blobCompare.count(), colBottom[0].gpu_data, blobData.gpu_data, m_blobCompare.mutable_gpu_data, i * m_nVectorDim, offset.Item2 * m_nVectorDim);
                        float fDist1 = m_cuda.dot_float(m_blobCompare.count(), m_blobCompare.gpu_data, m_blobCompare.gpu_data);
                        float fDist = (float)Math.Sqrt(convertF(m_blobCompare.sumsq_data()));
                        rgDist.Add(fDist);
                    }

                    rgDist.Sort();
                    int k = (m_nK <= 0 || m_nK > rgDist.Count) ? rgDist.Count : m_nK;
                    float fTotal = 0;

                    for (int j = 0; j < k; j++)
                    {
                        fTotal += rgDist[j];
                    }

                    float fAveDist = fTotal / k;
                    rgKDist.Add(kvItem.Key, fAveDist);
                }

                List<KeyValuePair<int, float>> rgKDistSorted = rgKDist.OrderBy(p => p.Key).ToList();
                float fMax = rgKDistSorted.Max(p => p.Value);
                float fMin = rgKDistSorted.Min(p => p.Value);

                for (int j = 0; j < rgKDistSorted.Count; j++)
                {
                    float fVal = (rgKDistSorted[j].Value - fMin)/(fMax - fMin);
                    fVal = 1.0f - fVal; // invert so that max value is the shortest distance (softmax looks for max);
                    rgFullSet[i * m_nNumOutput + j] = fVal;
                }

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    double dfPct = (double)i / (double)m_nBatchSize;
                    m_log.WriteLine("KNN testing cycle at " + dfPct.ToString("P") + "...");
                    sw.Restart();
                }
            }

            colTop[0].mutable_cpu_data = convert(rgFullSet.ToArray());
        }


        /// <summary>
        /// Save the data in the batch storage.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times K \times 1 \times 1) @f$ computed outputs, where @f$ K @f$ equals 
        ///     the <i>num_output</i> parameter. 
        /// </param>
        protected bool forward_save(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(2, colBottom.Count, "When training, the bottom must have both 'data' and 'labels'.");

            if (m_nCurrentBatchIdx == m_nMaxBatches)
            {
                m_bBufferFull = true;
                m_nCurrentBatchIdx = 0;
            }

            // Copy the data into the batch storage.
            m_cuda.copy(colBottom[0].count(), colBottom[0].gpu_data, m_colBlobs[1 + (m_nCurrentBatchIdx * 2 + 0)].mutable_gpu_data);
            m_cuda.copy(colBottom[1].count(), colBottom[1].gpu_data, m_colBlobs[1 + (m_nCurrentBatchIdx * 2 + 1)].mutable_gpu_data);
            m_nCurrentBatchIdx++;

            double dfCount = (m_bBufferFull) ? m_nMaxBatches : m_nCurrentBatchIdx;
            m_colBlobs[0].SetData(dfCount, 0);

            return m_bBufferFull;
        }

        /// @brief Not implemented - the KNN Layer does not perform backward.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
                throw new NotImplementedException();
        }
    }
}
