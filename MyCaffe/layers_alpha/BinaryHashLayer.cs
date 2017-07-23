using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using System.Diagnostics;

namespace MyCaffe.layers.alpha
{
    /// <summary>
    /// <H3>PRE ALPHA</H3>
    /// 
    /// The BinaryHashLayer uses distance calculations (Hamming and Euclidean) to find similar items and increase categorization accuracy.
    /// </summary>
    /// <remarks>
    /// @see [Deep Learning of Binary Hash Codes for Fast Image Retrieval](http://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W03/html/Lin_Deep_Learning_of_2015_CVPR_paper.html) by Kevin Lin, Heui-Fang Yang, Jen-Hao Hsiao and Chu-Song Chen, 2015. 
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class BinaryHashLayer<T> : Layer<T>
    {
        Blob<T> m_blobDebug;
        Blob<T> m_blobWork;
        IndexItemCollection m_rgCache1CurrentIndex;
        IndexItemCollection m_rgCache2CurrentIndex;
        int m_nLayer2Dim = 0;
        int m_nLayer3Dim = 0;
        int m_nLabelCount = 0;
        int m_nIteration = 0;
        bool m_bIsFull = false;
        double m_dfBinaryThreshold1 = 0;
        double m_dfBinaryThreshold2 = 0;
        float[] m_rgLabels = null;

        /// <summary>
        /// The BinaryHashLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type GRN.
        /// </param>
        public BinaryHashLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {           
            m_type = LayerParameter.LayerType.BINARYHASH;

            m_blobDebug = new common.Blob<T>(cuda, log, false);
            m_blobDebug.Name = "debug";

            m_blobWork = new common.Blob<T>(cuda, log);
            m_blobWork.Name = "work";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
            m_blobDebug.Dispose();
            m_blobWork.Dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();
                return col;
            }
        }

        /// <summary>
        /// Returns the minimum number of required bottom (input) Blobs: input1, input2*, input3*
        /// </summary>
        /// <remarks>
        /// <i>input2</i> contains the outputs for cache #1 and <i>input3</i> contains the outputs for cache #2.
        /// <i>input1</i> is the layer just before the Softmax layer and is used to augments it output based
        /// on the search results, and for sizing of the caches for it contains the label count.
        /// </remarks>
        public override int MinBottomBlobs
        {
            get { return 3; }
        }

        /// <summary>
        /// Returns the maximum number of required bottom (input) Blobs: input1, input2, label (only during training)
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 4; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: output
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of required top (output) Blobs: output, debug (only when enabled)
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
            m_nLabelCount = colBottom[0].count() / colBottom[0].num;

            // The pools contains tuples, where 
            //  Item1 = label and 
            //  Item2 = index of the item associated with the label.
            //  Item3 = distance from input item and this item in the cache.
            m_log.CHECK_LE(m_param.binary_hash_param.pool_size, m_param.binary_hash_param.cache_depth * m_nLabelCount, "The pool depth is too big, it must be less than the cache_depth * the label count.");

            Blob<T> blobCache1 = new common.Blob<T>(m_cuda, m_log, false);
            blobCache1.Name = "cache 1";
            m_colBlobs.Add(blobCache1);

            Blob<T> blobCache2 = new common.Blob<T>(m_cuda, m_log, false);
            blobCache2.Name = "cache 2";
            m_colBlobs.Add(blobCache2);

            Blob<T> blobParam = new common.Blob<T>(m_cuda, m_log, false);
            blobParam.Name = "params";
            m_colBlobs.Add(blobParam);

            // The current indexes tell when to 'roll' back
            // to the start and maintain the rolling buffer
            // so as to only store the latest outputs.
            m_rgCache1CurrentIndex = new IndexItemCollection(m_nLabelCount);
            m_rgCache2CurrentIndex = new IndexItemCollection(m_nLabelCount);

            m_nIteration = 0;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nLabelCount = colBottom[0].count() / colBottom[0].num;
            m_log.CHECK_EQ(nLabelCount, m_nLabelCount, "The label counts are not the same!");

            m_nLayer2Dim = colBottom[1].count() / colBottom[1].num;
            m_nLayer3Dim = colBottom[2].count() / colBottom[2].num;

            m_blobWork.Reshape(new List<int>() { m_nLabelCount, m_param.binary_hash_param.cache_depth, Math.Max(m_nLayer2Dim, m_nLayer3Dim) });

            m_colBlobs[0].Reshape(new List<int>() { nLabelCount, m_param.binary_hash_param.cache_depth, m_nLayer2Dim });
            m_colBlobs[1].Reshape(new List<int>() { nLabelCount, m_param.binary_hash_param.cache_depth, m_nLayer3Dim });
            m_colBlobs[2].Reshape(new List<int>() { 2 });

            colTop[0].ReshapeLike(colBottom[0]);

            if (m_phase != Phase.TEST || m_param.binary_hash_param.enable_test)
            {
                Tuple<double, double, double, double> workSize1 = m_cuda.minmax(m_colBlobs[0].count(), 0, 0, 0);
                Tuple<double, double, double, double> workSize2 = m_cuda.minmax(m_colBlobs[1].count(), 0, 0, 0);

                int nMax = (int)Math.Max(workSize1.Item1, workSize2.Item1);
                if (m_blobWork.count() < nMax)
                    m_blobWork.Reshape((int)Math.Max(workSize1.Item1, workSize2.Item1), 1, 1, 1);
            }
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the outputs.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNum = colBottom[0].num;

            m_log.CHECK_EQ(nNum, colBottom[1].num, "The bottom[1] should have the same num as bottom[0]!");
            m_log.CHECK_EQ(nNum, colBottom[2].num, "The bottom[2] should have the same num as bottom[0]!");

            m_cuda.copy(colBottom[0].count(), colBottom[0].gpu_data, colTop[0].mutable_gpu_data);

            if (m_phase == Phase.TRAIN)
            {
                m_log.CHECK_EQ(colBottom.Count, 4, "When training, the bottom should have 4 items: input1, input2 (rough pass), input3 (fine pass), labels");
                m_log.CHECK_EQ(nNum, colBottom[3].num, "The bottom[3] should have the same num as bottom[0]!");

                if (m_param.binary_hash_param.iteration_enable == 0 || m_nIteration > m_param.binary_hash_param.iteration_enable)
                {
                    m_rgLabels = convertF(colBottom[3].update_cpu_data());

                    // Load the first cache with colBottom[1] inputs.
                    for (int i = 0; i < nNum; i++)
                    {
                        int nLabel = (int)m_rgLabels[i];
                        int nClassIdx = m_rgCache1CurrentIndex[nLabel].Index;
                        int nIdx = (nLabel * (m_nLayer2Dim * m_param.binary_hash_param.cache_depth)) + (nClassIdx * m_nLayer2Dim);

                        m_cuda.copy(m_nLayer2Dim, colBottom[1].gpu_data, m_colBlobs[0].mutable_gpu_data, i * m_nLayer2Dim, nIdx);

                        m_rgCache1CurrentIndex[nLabel].Increment(m_param.binary_hash_param.cache_depth);
                    }

                    if (m_rgCache1CurrentIndex.IsFull)
                        m_colBlobs[2].SetData(1, 0);

                    // Load the second cache with colBottom[2] inputs.
                    for (int i = 0; i < nNum; i++)
                    {
                        int nLabel = (int)m_rgLabels[i];
                        int nClassIdx = m_rgCache2CurrentIndex[nLabel].Index;
                        int nIdx = (nLabel * (m_nLayer3Dim * m_param.binary_hash_param.cache_depth)) + (nClassIdx * m_nLayer3Dim);

                        m_cuda.copy(m_nLayer3Dim, colBottom[2].gpu_data, m_colBlobs[1].mutable_gpu_data, i * m_nLayer3Dim, nIdx);

                        m_rgCache2CurrentIndex[nLabel].Increment(m_param.binary_hash_param.cache_depth);
                    }

                    if (m_rgCache2CurrentIndex.IsFull)
                        m_colBlobs[2].SetData(1, 1);
                }
            }
            else if (m_phase != Phase.TEST || m_param.binary_hash_param.enable_test)
            {
                run(nNum, colBottom, colTop);
            }
        }

        private List<PoolItemCollection> run(int nNum, BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            List<PoolItemCollection> rgrgPool = new List<PoolItemCollection>();

            // This happens on the test or run net so we need to see if the cache that is shared (or loaded) is ready to use.
            if (!m_bIsFull)
            {
                float[] rgIsFull = convertF(m_colBlobs[2].mutable_cpu_data);
                if (rgIsFull[0] == 1 && rgIsFull[1] == 1)
                {
                    m_bIsFull = true;
                    m_log.WriteLine("The Binary Hash Cache is ready to use.");
                }

                if (m_param.binary_hash_param.dist_calc_pass1 == BinaryHashParameter.DISTANCE_TYPE.HAMMING)
                {
                    Tuple<double, double, double, double> minmax = m_cuda.minmax(m_colBlobs[0].count(), m_colBlobs[0].gpu_data, m_blobWork.mutable_gpu_data, m_blobWork.mutable_gpu_diff);

                    double dfMin = minmax.Item1;
                    double dfMax = minmax.Item2;

                    m_dfBinaryThreshold1 = ((dfMax - dfMin) * m_param.binary_hash_param.binary_threshold);

                    if (dfMin > 0)
                        m_dfBinaryThreshold1 += dfMin;
                }

                if (m_param.binary_hash_param.dist_calc_pass2 == BinaryHashParameter.DISTANCE_TYPE.HAMMING)
                {
                    Tuple<double, double, double, double> minmax = m_cuda.minmax(m_colBlobs[1].count(), m_colBlobs[1].gpu_data, m_blobWork.mutable_gpu_data, m_blobWork.mutable_gpu_diff);

                    double dfMin = minmax.Item1;
                    double dfMax = minmax.Item2;

                    m_dfBinaryThreshold2 = ((dfMax - dfMin) * m_param.binary_hash_param.binary_threshold);

                    if (dfMin > 0)
                        m_dfBinaryThreshold2 += dfMin;
                }
            }

            if (!m_bIsFull)
                return null;

            float[] rgOutput = convertF(colTop[0].mutable_cpu_data);
            bool bUpdate = false;

            Stopwatch sw = new Stopwatch();
            

            // Find the distance between each input and each element of cache #1 
            // and then from cache #2.
            for (int i = 0; i < nNum; i++)
            {
                //-----------------------------------------
                //  Build the pool using the Rough pass
                //-----------------------------------------
                PoolItemCollection rgPool1 = new alpha.PoolItemCollection();

                int[,] rgOffsets1 = new int[m_nLabelCount * m_param.binary_hash_param.cache_depth, 2];
                for (int j = 0; j < m_nLabelCount; j++)
                {
                    for (int k = 0; k < m_param.binary_hash_param.cache_depth; k++)
                    {
                        int nIdx1 = j * m_param.binary_hash_param.cache_depth + k;
                        rgOffsets1[nIdx1, 0] = i * m_nLayer2Dim;
                        rgOffsets1[nIdx1, 1] = j * m_nLayer2Dim * m_param.binary_hash_param.cache_depth + k * m_nLayer2Dim;
                    }
                }

                sw.Restart();
                DistanceMethod distMethod1 = (m_param.binary_hash_param.dist_calc_pass1 == BinaryHashParameter.DISTANCE_TYPE.EUCLIDEAN) ? DistanceMethod.EUCLIDEAN : DistanceMethod.HAMMING;
                double[] rgDist1 = m_cuda.calculate_batch_distances(distMethod1, m_dfBinaryThreshold1, m_nLayer2Dim, colBottom[1].gpu_data, m_colBlobs[0].gpu_data, m_blobWork.mutable_gpu_data, rgOffsets1);
                sw.Stop();
                Trace.WriteLine("timing = " + sw.Elapsed.TotalMilliseconds.ToString());

                for (int j = 0; j < m_nLabelCount; j++)
                {
                    for (int k = 0; k < m_param.binary_hash_param.cache_depth; k++)
                    {
                        int nIdx1 = j * m_param.binary_hash_param.cache_depth + k;
                        rgPool1.Add(new PoolItem(j, k, rgDist1[nIdx1]));
                    }
                }

                rgPool1.Sort();


                //-----------------------------------------
                //  Fine tuned pass
                //
                //  Find the 'pool_size' number of 
                //  minimum distances.
                //-----------------------------------------
                PoolItemCollection rgPool2 = new PoolItemCollection();

                int[,] rgOffsets2 = new int[m_param.binary_hash_param.pool_size, 2];
                for (int k = 0; k < m_param.binary_hash_param.pool_size; k++)
                {
                    PoolItem poolItem = rgPool1[k];
                    rgOffsets2[k, 0] = i * m_nLayer3Dim;
                    rgOffsets2[k, 1] = poolItem.Label * m_nLayer3Dim * m_param.binary_hash_param.cache_depth + poolItem.IndexIntoClass * m_nLayer3Dim;
                }

                DistanceMethod distMethod2 = (m_param.binary_hash_param.dist_calc_pass2 == BinaryHashParameter.DISTANCE_TYPE.EUCLIDEAN) ? DistanceMethod.EUCLIDEAN : DistanceMethod.HAMMING;
                double[] rgDist2 = m_cuda.calculate_batch_distances(distMethod2, m_dfBinaryThreshold2, m_nLayer3Dim, colBottom[2].gpu_data, m_colBlobs[1].gpu_data, m_blobWork.mutable_gpu_data, rgOffsets2);

                for (int k = 0; k < m_param.binary_hash_param.pool_size; k++)
                {
                    PoolItem poolItem = rgPool1[k];
                    rgPool2.Add(new PoolItem(poolItem.Label, poolItem.IndexIntoClass, rgDist2[k]));
                }

                rgPool2.Sort();


                //-----------------------------------------
                //  Select the label from the 'top_k' 
                //  minimum items of the fine-tuned pass.
                //-----------------------------------------

                int nNewLabel = rgPool2.SelectNewLabel(m_param.binary_hash_param.selection_method, (int)m_param.binary_hash_param.top_k);
                int nIdx = i * m_nLabelCount;
                int nPredictionLabel = getLabel(rgOutput, nIdx, m_nLabelCount);

                // If the new label is different from the previously predicted label, replace it.
                if (nNewLabel != nPredictionLabel)
                {
                    setLabel(rgOutput, nIdx, m_nLabelCount, nNewLabel);
                    bUpdate = true;
                }

                rgrgPool.Add(rgPool2);
            }

            if (bUpdate)
                colTop[0].mutable_cpu_data = convert(rgOutput);

            return rgrgPool;
        }

        private int getLabel(float[] rg, int nIdx, int nCount)
        {
            int nIdxMax = -1;
            float fMax = -float.MaxValue;

            for (int i = 0; i < nCount; i++)
            {
                if (rg[nIdx + i] > fMax)
                {
                    nIdxMax = i;
                    fMax = rg[nIdx + i];
                }
            }

            return nIdxMax;
        }

        private void setLabel(float[] rg, int nIdx, int nCount, int nSetIdx)
        {
            for (int i = 0; i<nCount; i++)
            {
                if (i == nSetIdx)
                    rg[nIdx + i] = 1.0f;
                else
                    rg[nIdx + i] = 0.0f;
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
            {
                m_cuda.copy(colTop[0].count(), colTop[0].gpu_diff, colBottom[0].mutable_gpu_diff);

                if (m_param.binary_hash_param.enable_triplet_loss && (m_param.binary_hash_param.iteration_enable == 0 || m_nIteration > m_param.binary_hash_param.iteration_enable))
                {
                    int nNum = colBottom[1].num;
                    int nDim = colBottom[1].count() / colBottom[1].num;

                    List<PoolItemCollection> rgrgPool = run(nNum, colBottom, colTop);
                    if (rgrgPool == null)
                        return;

                    float[] rgDiff = convertF(colTop[0].mutable_cpu_diff);
                    bool bUpdate = false;

                    for (int i = 0; i < nNum; i++)
                    {
                        PoolItemCollection rgPool = rgrgPool[i];
                        PoolItem itemPos = null;
                        PoolItem itemNeg = null;
                        int nLabel = (int)m_rgLabels[i];

                        // Find values with smallest differences for negatives (argmin)
                        for (int j = 0; j < rgPool.Count; j++)
                        {
                            int nPooledLabel = rgPool[j].Label;
                            if (rgPool[j].IndexIntoClass != m_rgCache2CurrentIndex[nPooledLabel].LastIndex)
                            {
                                if (itemNeg == null && nPooledLabel != nLabel)
                                {
                                    itemNeg = rgPool[j];
                                    break;
                                }
                            }
                        }

                        // Find values with largest differences for positives (argmax)
                        for (int j = rgPool.Count-1; j >= 0; j--)
                        {
                            int nPooledLabel = rgPool[j].Label;
                            if (rgPool[j].IndexIntoClass != m_rgCache2CurrentIndex[nPooledLabel].LastIndex)
                            {
                                if (itemPos == null && nPooledLabel == nLabel)
                                {
                                    itemPos = rgPool[j];
                                    break;
                                }
                            }
                        }

                        if (itemNeg != null && itemPos != null)
                        {
                            // See page 3 of https://arxiv.org/pdf/1503.03832.pdf
                            int nAOff = i * nDim;
                            int nBOffPos = (itemPos.Label * (m_param.binary_hash_param.cache_depth * nDim)) + (itemPos.IndexIntoClass * nDim);
                            double dfPos = m_cuda.sumsqdiff(nDim, m_blobWork.mutable_gpu_data, colTop[0].gpu_data, m_colBlobs[1].gpu_data, nAOff, nBOffPos);
                            int nBOffNeg = (itemNeg.Label * (m_param.binary_hash_param.cache_depth * nDim)) + (itemNeg.IndexIntoClass * nDim);
                            double dfNeg = m_cuda.sumsqdiff(nDim, m_blobWork.mutable_gpu_data, colTop[0].gpu_data, m_colBlobs[1].gpu_data, nAOff, nBOffNeg);
                            double dfErr = (Math.Sqrt(dfPos) - Math.Sqrt(dfNeg)) + m_param.binary_hash_param.alpha;

                            if (dfErr > 0)
                            {
                                for (int k = 0; k < m_nLabelCount; k++)
                                {
                                    rgDiff[i * m_nLabelCount + k] += (float)(dfErr / nDim);
                                }

                                bUpdate = true;
                            }
                        }
                    }

                    if (bUpdate)
                        colBottom[0].mutable_cpu_diff = convert(rgDiff);
                }
            }
        }
    }

    class PoolItem /** @private */
    {
        int m_nLabel;
        int m_nClassIdx;
        double m_dfDistance;

        public PoolItem(int nLabel, int nClassIdx, double dfDist)
        {
            m_nLabel = nLabel;
            m_nClassIdx = nClassIdx;
            m_dfDistance = dfDist;
        }

        public int Label
        {
            get { return m_nLabel; }
        }

        public int IndexIntoClass
        {
            get { return m_nClassIdx; }
        }

        public double Distance
        {
            get { return m_dfDistance; }
            set { m_dfDistance = value; }
        }

        public override string ToString()
        {
            return m_nLabel.ToString() + ":" + m_nClassIdx.ToString() + " -> " + m_dfDistance.ToString();
        }
    }

    class PoolItemCollection : IEnumerable<PoolItem> /** @private */
    {
        List<PoolItem> m_rg;

        public PoolItemCollection()
        {
            m_rg = new List<PoolItem>();
        }

        public int Count
        {
            get { return m_rg.Count; }
        }

        public PoolItem this[int nIdx]
        {
            get { return m_rg[nIdx]; }
            set { m_rg[nIdx] = value; }
        }

        public void Add(PoolItem p)
        {
            m_rg.Add(p);
        }

        public void Clear()
        {
            m_rg.Clear();
        }

        public void ShrinkToSize(int nMax)
        {
            if (m_rg.Count <= nMax)
                return;

            List<PoolItem> rg = new List<PoolItem>();

            for (int i = nMax; i < m_rg.Count; i++)
            {
                rg.Add(m_rg[i]);
            }

            foreach (PoolItem p in rg)
            {
                m_rg.Remove(p);
            }
        }

        public void Sort()
        {
            m_rg.Sort(new Comparison<PoolItem>(sortAscending));
        }

        int sortAscending(PoolItem a, PoolItem b)
        {
            if (a.Distance < b.Distance)
                return -1;

            if (a.Distance > b.Distance)
                return 1;

            return 0;
        }

        public int SelectNewLabel(BinaryHashParameter.SELECTION_METHOD method, int nK)
        {
            if (method == BinaryHashParameter.SELECTION_METHOD.MINIMUM_DISTANCE)
                return m_rg[0].Label;

            // Select the item with the most votes.
            Dictionary<int, int> rgCounts = new Dictionary<int, int>();

            for (int i=0; i<m_rg.Count && i<nK; i++)
            {
                PoolItem p = m_rg[i];

                if (!rgCounts.ContainsKey(p.Label))
                    rgCounts.Add(p.Label, 0);

                rgCounts[p.Label]++;
            }

            List<KeyValuePair<int, int>> rg = rgCounts.ToList();

            if (rgCounts.Count == 1)
                return m_rg[0].Label;

            rg.Sort(new Comparison<KeyValuePair<int, int>>(sortItemsDescending));

            // If we get two items with the same vote,
            // default back to the minimum distance.
            if (rg[0].Value == rg[1].Value)
                return m_rg[0].Label;

            return rg[0].Key;
        }

        private int sortItemsDescending(KeyValuePair<int, int> a, KeyValuePair<int, int> b)
        {
            if (a.Value < b.Value)
                return 1;

            if (a.Value > b.Value)
                return -1;

            return 0;
        }

        public IEnumerator<PoolItem> GetEnumerator()
        {
            return m_rg.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rg.GetEnumerator();
        }
    }

    class IndexItem /** @private */
    {
        int m_nLastIdx = -1;
        int m_nIdx = 0;
        bool m_bFull = false;

        public IndexItem()
        {
        }

        public int Index
        {
            get { return m_nIdx; }
            set { m_nIdx = value; }
        }

        public int LastIndex
        {
            get { return m_nLastIdx; }
        }

        public bool Full
        {
            get { return m_bFull; }
            set { m_bFull = value; }
        }

        public void Increment(int nMax)
        {
            m_nLastIdx = m_nIdx;
            m_nIdx++;

            if (m_nIdx >= nMax)
            {
                m_bFull = true;
                m_nIdx = 0;
            }
        }

        public override string ToString()
        {
            return m_nIdx.ToString() + " -> " + m_bFull.ToString();
        }
    }

    class IndexItemCollection /** @private */
    {
        IndexItem[] m_rg;

        public IndexItemCollection(int nCount)
        {
            m_rg = new IndexItem[nCount];

            for (int i = 0; i < nCount; i++)
            {
                m_rg[i] = new IndexItem();
            }
        }

        public IndexItem this[int nIdx]
        {
            get { return m_rg[nIdx]; }
            set { m_rg[nIdx] = value; }
        }

        public int Count
        {
            get { return m_rg.Length; }
        }

        public bool IsFull
        {
            get
            {
                foreach (IndexItem item in m_rg)
                {
                    if (!item.Full)
                        return false;
                }

                return true;
            }
        }
    }
}
