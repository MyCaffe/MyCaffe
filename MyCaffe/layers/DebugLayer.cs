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
    /// The DebugLayer merely stores, up to max_stored_batches, batches of input which
    /// are then optionally used by various debug visualizers.
    /// This layer is initialized with the MyCaffe.param.DebugParameter.
    /// </summary>
    /// <remarks>
    /// The data collected by the DebugLayer can later be used for debugging analysis.
    /// </remarks> 
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DebugLayer<T> : Layer<T>, IXDebugData<T>
    {
        BlobCollection<T> m_rgBatchData = new BlobCollection<T>();
        BlobCollection<T> m_rgBatchLabels = new BlobCollection<T>();
        int m_nCurrentBatchIdx = 0;
        int m_nLastBatchIdx = 0;
        int m_nMaxBatches;
        bool m_bBufferFull = false;

        /// <summary>
        /// The DebugLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides DebugParameter debug_param with options:
        /// - max_stored_batches. Specifies the number of batches that the DebugLayer should store.
        /// </param>
        public DebugLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.DEBUG;
            m_nMaxBatches = p.debug_param.max_stored_batches;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_rgBatchData.Dispose();
            m_rgBatchLabels.Dispose();
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
        /// Returns a collection of Blobs containing the data stored by the DebugLayer.
        /// </summary>
        public BlobCollection<T> data
        {
            get { return m_rgBatchData; }
        }

        /// <summary>
        /// Returns a collection of Blobs containing the labels stored by the DebugLayer.
        /// </summary>
        public BlobCollection<T> labels
        {
            get { return m_rgBatchLabels; }
        }

        /// <summary>
        /// Returns the name of the DebugLayer.
        /// </summary>
        public string name
        {
            get { return m_param.name; }
        }

        /// <summary>
        /// Returns the handle to the CudaDnn kernel where the debug GPU memory resides.
        /// </summary>
        public long kernel_handle
        {
            get { return m_cuda.KernelHandle; }
        }

        /// <summary>
        /// Returns the number of batches actually loaded into the DebugLayer.
        /// </summary>
        public int load_count
        {
            get { return (m_bBufferFull) ? m_rgBatchData.Count : m_nCurrentBatchIdx; }
        }

        /// <summary>
        /// Returns the minimum number of required bottom (input) Blobs: data, label
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 2; }   // data (embeddings), label
        }

        /// <summary>
        /// Returns the minimum number of top (output) Blobs: data (passthrough)
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }   // debug layer passes through the data, but not the label.
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_GE(colBottom.Count, 2, "There should be at least two bottom items: data (embeddings) and labels.");
            m_log.CHECK_EQ(colTop.Count, colBottom.Count - 1, "The top count should equal the bottom count - 1");

            // Allocate the temp batch storage.
            for (int i = 0; i < m_nMaxBatches; i++)
            {
                Blob<T> data = new Blob<T>(m_cuda, m_log);
                data.ReshapeLike(colBottom[0]);
                m_rgBatchData.Add(data);

                Blob<T> label = new common.Blob<T>(m_cuda, m_log, false);
                label.ReshapeLike(colBottom[1]);
                m_rgBatchLabels.Add(label);
            }
        }

        /// <summary>
        /// Reshape the top (output) to match the bottom (input), and reshape internal buffers.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Reshape the temp batch storage.
            for (int i = 0; i < m_nMaxBatches; i++)
            {
                m_rgBatchData[i].ReshapeLike(colBottom[0]);
                m_rgBatchLabels[i].ReshapeLike(colBottom[1]);
            }

            colTop[0].ReshapeLike(colBottom[0]);
        }

        /// <summary>
        /// Forward cache and pass through
        /// </summary>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_nCurrentBatchIdx == m_nMaxBatches)
            {
                m_nCurrentBatchIdx = 0;
                m_bBufferFull = true;
            }

            // Copy the data into the batch storage.
            m_cuda.copy(colBottom[0].count(), colBottom[0].gpu_data, m_rgBatchData[m_nCurrentBatchIdx].mutable_gpu_data);
            m_cuda.copy(colBottom[1].count(), colBottom[1].gpu_data, m_rgBatchLabels[m_nCurrentBatchIdx].mutable_gpu_data);

            m_nLastBatchIdx = m_nCurrentBatchIdx;
            m_nCurrentBatchIdx++;

            m_cuda.copy(colBottom[0].count(), colBottom[0].gpu_data, colTop[0].mutable_gpu_data);
        }

        /// <summary>
        /// Backward passthrough
        /// </summary>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        /// <param name="rgbPropagateDown">Specifies whether or not to propagate each blob back.</param>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            // Copy the diff into the batch storage.
            m_cuda.copy(colTop[0].count(), colTop[0].gpu_diff, m_rgBatchData[m_nLastBatchIdx].mutable_gpu_diff);

            m_cuda.copy(colTop[0].count(), colTop[0].gpu_diff, colBottom[0].mutable_gpu_diff);
        }
    }
}
