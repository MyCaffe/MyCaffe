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
    /// DataSequence Layer - this caches inputs by label and then outputs data item tuplets that include
    /// an 'anchor', optionally a 'positive' match, and at least one 'negative' match.
    /// 
    /// This layer is initialized with the MyCaffe.param.DataSequenceParameter.
    /// </summary>
    /// <remarks>
    /// SiameseNet's and TripletLoss based nets use this layer to help organize the data inputs to the 
    /// parallel networks making up each network architecture.
    /// 
    /// The following settings should be used with each network architecture.
    /// 
    /// SiameseNet - k = 0, causing this layer to output (anchor, negative1)
    /// TripletLoss1 - k = 1, causing this layer to output (anchor, positive, negative1)
    /// TripletLoss5 - k = 5, causing this layer to output (anchor, positive, negative1, negative2, negative3, negative4, negative5)
    /// </remarks> 
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DataSequenceLayer<T> : Layer<T>
    {
        int m_nK;
        int m_nCacheSize;
        bool m_bOutputLabels = false;
        Blob<T> m_blobLabeledDataCache = null;
        int m_nLabelStart = 0;
        int m_nLabelCount = 0;
        bool m_bBalanceMatches = false;
        long m_hCacheCursors = 0;
        long m_hWorkDataHost = 0;

        /// <summary>
        /// The DataSequenceLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type DataSequence with parameter DataSequence_param,
        /// with options:
        ///   - cache_size (\b optional, default 256). The size of each labeled image cache.
        ///   
        ///   - k (\b optional, default 0). When 0, output is an anchor and one negative match, when > 0 output is an anchor, positive match + 'k' negative matches.
        /// </param>
        public DataSequenceLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_param = p;
            m_type = LayerParameter.LayerType.DATA_SEQUENCE;
            m_nK = m_param.data_sequence_param.k;
            m_nCacheSize = m_param.data_sequence_param.cache_size;
            m_bOutputLabels = m_param.data_sequence_param.output_labels;
            m_nLabelCount = m_param.data_sequence_param.label_count;
            m_nLabelStart = m_param.data_sequence_param.label_start;
            m_bBalanceMatches = m_param.data_sequence_param.balance_matches;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_blobLabeledDataCache != null)
            {
                m_blobLabeledDataCache.Dispose();
                m_blobLabeledDataCache = null;
            }

            if (m_hCacheCursors != 0)
            {
                m_cuda.FreeHostBuffer(m_hCacheCursors);
                m_hCacheCursors = 0;
                m_nLabelCount = 0;
            }

            if (m_hWorkDataHost != 0)
            {
                m_cuda.FreeHostBuffer(m_hWorkDataHost);
                m_hWorkDataHost = 0;
            }

            base.dispose();
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
        /// Returns the exact number of required bottom (intput) Blobs: data, label
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; } // data, label
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: anchor, positve (k > 0), negative (depending on k value), labels
        /// </summary>
        public override int MinTopBlobs
        {
            get { return m_nK + 2 + ((m_bOutputLabels) ? 1 : 0); } // anchor, positive (if k > 0), negative * (1 + m_nK)
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Disable back-propagation for all outputs of this layer.
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
            colTop[0].ReshapeLike(colBottom[0]);
            colTop[1].ReshapeLike(colBottom[0]);

            for (int k = 0; k < m_nK; k++)
            {
                colTop[2 + k].ReshapeLike(colBottom[0]);
            }

            if (m_bOutputLabels)
            {
                int nLabelDim = 2 + m_nK;
                colTop[2 + m_nK].Reshape(colBottom[0].num, nLabelDim, 1, 1);
            }
        }

        /// <summary>
        /// During the forward pass, each input data item is cached by label and then sequencing is performed on the 
        /// cached items to produce the desired data sequencing output.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times K \times 1 \times 1) @f$ computed outputs, where @f$ K @f$ equals 
        ///     the <i>num_output</i> parameter. 
        /// </param>
        /// <remarks>
        /// IMPORTANT: The data batch size must be sufficiently large enough to contain at least one instance of each label in the dataset.
        /// </remarks>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> data = colBottom[0];
            Blob<T> labels = colBottom[1];

            if (m_blobLabeledDataCache == null)
            {
                List<int> rgLabels = new List<int>();

                // Use dynamic label discovery - requires that all labels of the dataset are in the first batch.
                if (m_nLabelCount == 0)
                {
                    float[] rgfLabels = convertF(labels.update_cpu_data());

                    foreach (float fLabel in rgfLabels)
                    {
                        int nLabel = (int)fLabel;
                        if (!rgLabels.Contains(nLabel))
                            rgLabels.Add(nLabel);
                    }

                    rgLabels.Sort();

                    m_nLabelCount = rgLabels.Count;
                    m_nLabelStart = rgLabels.Min();
                }
                else
                {
                    for (int i = 0; i < m_nLabelCount; i++)
                    {
                        rgLabels.Add(m_nLabelStart + i);
                    }
                }

                int nNum = rgLabels.Count * m_nCacheSize;
                m_blobLabeledDataCache = new Blob<T>(m_cuda, m_log, nNum, colBottom[0].channels, colBottom[0].height, colBottom[0].width);
                m_blobLabeledDataCache.SetData(0);
                m_hCacheCursors = m_cuda.AllocHostBuffer(rgLabels.Count * 2);
                m_hWorkDataHost = m_cuda.AllocHostBuffer(labels.count());                
            }

            m_log.CHECK_EQ(data.num, labels.count(), "The label counts do not match the batch size!");

            m_cuda.copy_batch(data.count(), data.num, data.count(1), data.gpu_data, labels.gpu_data, m_blobLabeledDataCache.count(), m_blobLabeledDataCache.mutable_gpu_data, m_blobLabeledDataCache.mutable_gpu_diff, m_nLabelStart, m_nLabelCount, m_nCacheSize, m_hCacheCursors, m_hWorkDataHost);

            int nK = m_nK;
            List<long> rgTop = new List<long>();
            List<int> rgTopCount = new List<int>();

            for (int i = 0; i < colTop.Count; i++)
            {
                rgTop.Add(colTop[i].mutable_gpu_data);
                rgTopCount.Add(colTop[i].count());
            }

            m_cuda.copy_sequence(nK, data.num, data.count(1), data.gpu_data, labels.gpu_data, m_blobLabeledDataCache.count(), m_blobLabeledDataCache.gpu_data, m_nLabelStart, m_nLabelCount, m_nCacheSize, m_hCacheCursors, m_bOutputLabels, rgTop, rgTopCount, m_hWorkDataHost, m_bBalanceMatches);
        }

        /// @brief Not implemented - the DataSequence Layer does not perform backward.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
                throw new NotImplementedException();
        }
    }
}
