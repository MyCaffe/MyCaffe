using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Diagnostics;
using MyCaffe.basecode;
using MyCaffe.imagedb;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.data;

/// <summary>
/// The MyCaffe.layers.alpha contains layers that are considered at a pre-alpha level and may experience a high degree of change.
/// This layer is initialized with the MyCaffe.param.BatchDataParameter.
/// </summary>
namespace MyCaffe.layers.alpha
{
    /// <summary>
    /// <H3>PRE ALPHA</H3>
    /// 
    /// The BatchDataLayer loads data into a batch and is used with custom learning.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class BatchDataLayer<T> : BaseDataLayer<T>
    {
        CancelEvent m_evtCancel;
        AutoResetEvent m_evtDataReady = new AutoResetEvent(false);
        List<BlockingQueue<Datum>> m_rgrgDataQueue = null;
        T[] m_rgInput = null;
        InternalThread<T> m_internalThread;
        int m_nCurrentBatchIdx = 0;
        int m_nNumBatches = 0;
        int m_nBatchSize = 0;
        int m_nCurrentIteration = 0;
        Batch<T> m_batch = null;
        TransferInput.fnSetInputData m_fnSetInput = null;


        /// <summary>
        /// The DataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter with the batchdata_param.</param>
        /// <param name="db">Specifies the external database to use.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel any pre-fetching operations.</param>
        /// <param name="fnSet">Specifies the delegate used to set the input data.</param>
        public BatchDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXImageDatabase db, CancelEvent evtCancel, TransferInput.fnSetInputData fnSet)
            : base(cuda, log, p, db)
        {
            log.CHECK(p.type == LayerParameter.LayerType.BATCHDATA, "The layer type should be BATCHDATA.");

            m_type = LayerParameter.LayerType.BATCHDATA;
            m_evtCancel = evtCancel;
            m_fnSetInput = fnSet;
            m_batch = new layers.Batch<T>(m_cuda, m_log);
            m_internalThread = new common.InternalThread<T>();
            m_internalThread.DoWork += m_internalThread_DoWork;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_internalThread.StopInternalThread();

            if (m_rgrgDataQueue != null)
            {
                foreach (BlockingQueue<Datum> q in m_rgrgDataQueue)
                {
                    q.Dispose();
                }

                m_rgrgDataQueue.Clear();
                m_rgrgDataQueue = null;
            }

            if (m_batch != null)
            {
                m_batch.Dispose();
                m_batch = null;
            }

            base.dispose();
        }

        private void m_internalThread_DoWork(object sender, ActionStateArgs<T> e)
        {
            while (!m_internalThread.CancellationPending)
            {
                if (m_evtDataReady.WaitOne(10))
                {
                    for (int i = 0; i < m_nNumBatches; i++)
                    {
                        for (int j = 0; j < m_nBatchSize; j++)
                        {
                            if (m_evtCancel.WaitOne(0))
                                return;

                            int nIdx = (i * m_nBatchSize) + j;
                            int nImgIdx = (int)Convert.ChangeType(m_rgInput[nIdx], typeof(int));

                            SimpleDatum d = m_imgdb.QueryImage(m_src.ID, nImgIdx, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                            m_rgrgDataQueue[i].Push(new Datum(d));
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Setup the DataLayer.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void DataLayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nCurrentBatchIdx = 0;
            m_nCurrentIteration = 0;
            m_nBatchSize = (int)m_param.batch_data_param.batch_size;
            m_nNumBatches = 0;

            if (m_rgrgDataQueue != null)
            {
                foreach (BlockingQueue<Datum> q in m_rgrgDataQueue)
                {
                    q.Dispose();
                }

                m_rgrgDataQueue = null;
            }

            m_rgInput = null;

            m_log.WriteLine("Initializing batch data layer...");
            m_transformer.InitRand();
            m_internalThread.StartInternalThread(m_cuda, m_log, m_cuda.GetDeviceID());
            m_log.WriteLine("Batch data layer initialized.");
            m_rgInput = null;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            if (colTop.Count > 0)
                colTop[0].Reshape(new List<int>() { m_nBatchSize, m_src.ImageChannels, m_src.ImageHeight, m_src.ImageWidth });
        }

        /// <summary>
        /// Returns the minimum number of required bottom (input) Blobs which is 0.
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 0; }
        }

        /// <summary>
        /// Returns the maximum number or required bottom (input) Blobs: batchIdx 
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: data
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of required top (output) Blobs: data, label
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Load a batch of data in the background (this is run on an internal thread within the BasePrefetchingDataLayer class).
        /// </summary>
        /// <param name="batch">Specifies the Batch of data to load.</param>
        protected bool load_batch(Batch<T> batch)
        {
            Datum d = null;

            // Reshape according to the first datum of each batch
            // on single input batches allows for inputs of varying
            // dimension.

            if (!m_rgrgDataQueue[m_nCurrentBatchIdx].GetAt(0, ref d))
                return false;

            // Use data transformer to infer the expected blob shape for datum.
            List<int> rgTopShape = m_transformer.InferBlobShape(d);


            // Reshape batch according to the batch size.
            rgTopShape[0] = m_nBatchSize;
            batch.Data.Reshape(rgTopShape);

            T[] rgTopLabel = null;

            if (m_bOutputLabels)
                rgTopLabel = batch.Label.mutable_cpu_data;

            List<T> rgData = new List<T>();

            for (int i = 0; i < m_nBatchSize; i++)
            {
                if (m_evtCancel.WaitOne(0))
                    return false;

                // Apply data transformations (mirrow, scaling, crop, etc)
                rgData.AddRange(m_transformer.Transform(d));

                // Copy label.
                if (m_bOutputLabels)
                    rgTopLabel[i] = (T)Convert.ChangeType(d.Label, typeof(T));

                if (i < m_nBatchSize - 1)
                {
                    if (!m_rgrgDataQueue[m_nCurrentBatchIdx].GetAt(i + 1, ref d))
                        return false;
                }
            }

            batch.Data.mutable_cpu_data = rgData.ToArray();
            m_transformer.SetRange(batch.Data);

            if (m_bOutputLabels)
                batch.Label.mutable_cpu_data = rgTopLabel;

            return true;
        }

        /// <summary>
        /// The forward override implements the functionality to load data and feed it into the 
        /// top (output) Blobs.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_bEnablePassthrough)
            {
                for (int i = 0; i < colBottom.Count; i++)
                {
                    colTop[i].CopyFrom(colBottom[i], false, false);
                }

                return;
            }

            m_log.CHECK_EQ(colBottom.Count, 1, "The bottom must contain one blob with at least one set of image indexes to load in the batch.");
            int nNumBatches = colBottom[0].shape(0);
            m_log.CHECK_GT(nNumBatches, 0, "The bottom[0].shape(0) must have at least one batch to load in the batch.");
            int nBatchSize = colBottom[0].shape(1);
            m_log.CHECK_GT(nBatchSize, 0, "The bottom[0].shape(1) must have at least one image index in the batch.");

            if (m_nNumBatches != 0)
                m_log.CHECK_EQ(nNumBatches, m_nNumBatches, "The current batch count does not equal the count used to load the data!");

            if (m_nBatchSize != 0)
                m_log.CHECK_EQ(nBatchSize, m_nBatchSize, "The current batch size does not equal the size used to load the data!");

            m_nNumBatches = nNumBatches;
            m_nBatchSize = nBatchSize;

            if (m_rgrgDataQueue == null)
            {
                m_rgrgDataQueue = new List<BlockingQueue<Datum>>();

                for (int i = 0; i < nNumBatches; i++)
                {
                    m_rgrgDataQueue.Add(new common.BlockingQueue<Datum>(m_evtCancel));
                }
            }

            if (m_rgInput == null)
            {
                m_rgInput = colBottom[0].update_cpu_data();
                m_evtDataReady.Set();
            }
            
            if (load_batch(m_batch))
            {
                // Reshape to loaded data.
                colTop[0].ReshapeLike(m_batch.Data);

                // Copy the data.
                m_cuda.copy(m_batch.Data.count(), m_batch.Data.gpu_data, colTop[0].mutable_gpu_data);

                if (m_bOutputLabels)
                {
                    // Reshape to loaded labels.
                    colTop[1].ReshapeLike(m_batch.Label);

                    // Copy the labels.
                    m_cuda.copy(m_batch.Label.count(), m_batch.Label.gpu_data, colTop[1].mutable_gpu_data);
                }

                if (m_fnSetInput != null)
                {
                    List<int> rgFwdInput = new List<int>();

                    for (int i = 0; i < m_nBatchSize; i++)
                    {
                        int nIdx = (m_nCurrentBatchIdx * m_nBatchSize) + i;
                        float fIdx = (float)Convert.ChangeType(m_rgInput[nIdx], typeof(float));
                        rgFwdInput.Add((int)fIdx);
                    }

                    m_fnSetInput(new BatchInput(m_nCurrentBatchIdx, rgFwdInput));
                }

                m_nCurrentBatchIdx++;

                if (m_nCurrentBatchIdx == m_rgrgDataQueue.Count)
                {
                    m_nCurrentBatchIdx = 0;
                    m_nCurrentIteration++;

                    if (m_nCurrentIteration == m_param.batch_data_param.iterations)
                    {
                        m_nCurrentIteration = 0;

                        if (m_param.batch_data_param.CompletedEvent != null)
                            m_param.batch_data_param.CompletedEvent.Set();
                    }
                }
            }
        }
    }
}
