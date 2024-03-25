﻿using System;
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
        int m_nTotalItems = 1;
        int m_nLabelItems = 1;
        int m_nIgnoreItems = 0;
        List<int> m_rgLabelShape;
        List<int> m_rgDataShape;

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
        /// Returns the maximum number of bottom (input) Blobs: data, label, ignore
        /// </summary>
        public override int MaxBottomBlobs  
        {
            get { return 3; }
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
            m_log.CHECK_EQ(colTop.Count, 1, "The top count should equal 1");

            m_nLabelItems = colBottom[1].height;
            if (colBottom.Count == 3)
                m_nIgnoreItems = colBottom[2].height;

            m_nTotalItems = m_nLabelItems + m_nIgnoreItems;

            m_rgDataShape = Utility.Clone<int>(colBottom[0].shape());
            m_rgLabelShape = Utility.Clone<int>(colBottom[1].shape());

            // Allocate the temp batch storage.
            for (int i = 0; i < m_nMaxBatches; i++)
            {
                Blob<T> data = new Blob<T>(m_cuda, m_log);
                Blob<T> label = new common.Blob<T>(m_cuda, m_log, false);

                data.Reshape(m_rgDataShape);
                m_rgBatchData.Add(data);

                label.Reshape(m_rgLabelShape);
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
            if (!reshapeNeeded(colBottom, colTop))
                return;

            // Reshape the temp batch storage.
            for (int i = 0; i < m_nMaxBatches; i++)
            {
                m_rgBatchData[i].Reshape(m_rgDataShape);
                m_rgBatchLabels[i].Reshape(m_rgLabelShape);
            }

            colTop[0].ReshapeLike(colBottom[0]);
        }

        /// <summary>
        /// Forward cache and pass through
        /// </summary>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = m_nTotalItems - m_nIgnoreItems;

            for (int i = 0; i < nCount; i++)
            {
                if (m_param.debug_param.item_index < 0 || i == m_param.debug_param.item_index)
                {
                    if (m_nCurrentBatchIdx == m_nMaxBatches)
                    {
                        m_nCurrentBatchIdx = 0;
                        m_bBufferFull = true;
                    }

                    // Copy the data into the batch storage.
                    m_rgBatchData[m_nCurrentBatchIdx].CopyFrom(colBottom[0]);
                    m_rgBatchLabels[m_nCurrentBatchIdx].CopyFrom(colBottom[1]);

                    m_nLastBatchIdx = m_nCurrentBatchIdx;
                    m_nCurrentBatchIdx++;
                }
            }

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
            if (m_nTotalItems - m_nIgnoreItems == 1)
                m_cuda.copy(colTop[0].count(), colTop[0].gpu_diff, m_rgBatchData[m_nLastBatchIdx].mutable_gpu_diff);

            m_cuda.copy(colTop[0].count(), colTop[0].gpu_diff, colBottom[0].mutable_gpu_diff);
        }
    }
}
