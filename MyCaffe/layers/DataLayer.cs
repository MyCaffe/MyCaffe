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

namespace MyCaffe.layers
{
    /// <summary>
    /// The DataLayer loads data from the IXImageDatabase database.
    /// This layer is initialized with the MyCaffe.param.DataParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DataLayer<T> : BasePrefetchingDataLayer<T>
    {
        /// <summary>
        /// Specifies the database.
        /// </summary>
        protected DB m_db;
        /// <summary>
        /// Specifies the database used to traverse through the database.
        /// </summary>
        protected Cursor m_cursor;
        UInt64 m_nOffset;
        /// <summary>
        /// Specifies a first timer used to calcualte the batch time.
        /// </summary>
        protected Stopwatch m_swTimerBatch;
        /// <summary>
        /// Specfies a second timer used to calculate the transaction time.
        /// </summary>
        protected Stopwatch m_swTimerTransaction;
        /// <summary>
        /// Specifies the read time.
        /// </summary>
        protected double m_dfReadTime;
        /// <summary>
        /// Specifies the transaction time.
        /// </summary>
        protected double m_dfTransTime;
        private T[] m_rgTopData = null;

        private LabelCollection m_rgBatchLabels = null;

        /// <summary>
        /// This event fires (only when set) each time a batch is loaded form this dataset.
        /// </summary>
        public event EventHandler<LastBatchLoadedArgs> OnBatchLoad;

        /// <summary>
        /// The DataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter data_param</param>
        /// <param name="db">Specifies the external database to use.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel any pre-fetching operations.</param>
        public DataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXImageDatabase db, CancelEvent evtCancel)
            : base(cuda, log, p, db, evtCancel)
        {
            m_type = LayerParameter.LayerType.DATA;

            if (p.data_param.synchronize_target)
                m_rgBatchLabels = new LabelCollection();

            Tuple<IMGDB_LABEL_SELECTION_METHOD, IMGDB_IMAGE_SELECTION_METHOD> kvSel = db.GetSelectionMethod();
            IMGDB_IMAGE_SELECTION_METHOD imgSel = kvSel.Item2;

            if (m_param.data_param.enable_pair_selection.HasValue)
            {
                if (m_param.data_param.enable_pair_selection.Value)
                    imgSel |= IMGDB_IMAGE_SELECTION_METHOD.PAIR;
                else
                    imgSel &= (~IMGDB_IMAGE_SELECTION_METHOD.PAIR);
            }

            if (m_param.data_param.enable_random_selection.HasValue)
            {
                if (m_param.data_param.enable_random_selection.Value)
                    imgSel |= IMGDB_IMAGE_SELECTION_METHOD.RANDOM;
                else
                    imgSel &= (~IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
            }

            db.SetSelectionMethod(null, imgSel);

            m_db = new data.DB(db);
            m_db.Open(p.data_param.source);
            m_cursor = m_db.NewCursor();

            if (p.data_param.display_timing)
            {
                m_swTimerBatch = new Stopwatch();
                m_swTimerTransaction = new Stopwatch();
            }
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_rgBatchLabels != null)
            {
                m_rgBatchLabels.Dispose();
                m_rgBatchLabels = null;
            }

            base.dispose();
        }

        /// <summary>
        /// Specifies to delay the prefetch when using a synchronized Data Layer.
        /// </summary>
        protected override bool delayPrefetch
        {
            get
            {
                if (m_param.data_param.synchronize_with != null || m_param.data_param.synchronize_target)
                    return true;

                return false;
            }
        }

        /// <summary>
        /// The Connect method connects one Data Layer to another so that they can synchronize.
        /// </summary>
        /// <param name="src">Specifies the source Data Layer whos OnBatchLoad event fires and
        /// is handled by this Data Layer.
        /// </param>
        public void Connect(DataLayer<T> src)
        {
            src.OnBatchLoad += Src_OnBatchLoad;
            m_log.WriteLine("DataLayer '" + m_param.name + "' is now connected to DataLayer '" + src.m_param.name + "'.");

            statupPrefetch();
            src.statupPrefetch();
        }

        /// <summary>
        /// Disconnect any previously connected Data Layers.
        /// </summary>
        public void Disconnect()
        {
            m_rgBatchLabels.Cancel();
        }

        private void Src_OnBatchLoad(object sender, LastBatchLoadedArgs e)
        {
            int nWait = m_rgBatchLabels.WaitProcessing;
            if (nWait == 0)
                return;

            m_rgBatchLabels.Set(e.Labels);
        }

        /// <summary>
        /// Setup the DataLayer by starting up the pre-fetching.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void DataLayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nBatchSize = (int)m_param.data_param.batch_size;

            // Read a data point, and use it to initialize the top blob.
            Datum datum = m_cursor.GetValue();

            // Use data transformer to infer the expected blob shape from the datum.
            List<int> rgTopShape = m_transformer.InferBlobShape(datum);
           

            // Reshape colTop[0] and prefetch data according to the batch size.
            rgTopShape[0] = nBatchSize;

            colTop[0].Reshape(rgTopShape);

            for (int i = 0; i < m_rgPrefetch.Length; i++)
            {
                m_rgPrefetch[i].Data.Reshape(rgTopShape);
            }

            m_log.WriteLine("output data size: " + colTop[0].ToSizeString());

            // Label
            if (m_bOutputLabels)
            {
                List<int> rgLabelShape = new List<int>() { nBatchSize };

                // When using multi-labels, resize to batch x the number of multiple 
                // labels per image.
                if (m_param.data_param.load_multiple_labels)
                {
                    if (datum.DataCriteria == null || datum.DataCriteria.Length == 0)
                        m_log.FAIL("Could not find the multi-label data.  The data source '" + m_param.data_param.source + "' does not appear to have any Image Criteria data.");

                    // Get the number of items and the item size from the end of the data.
                    int nLen = BitConverter.ToInt32(datum.DataCriteria, datum.DataCriteria.Length - (sizeof(int) * 4));
                    int nItemSize = BitConverter.ToInt32(datum.DataCriteria, datum.DataCriteria.Length - (sizeof(int) * 3));

                    m_log.CHECK_EQ(nItemSize, 1, "Currently only byte sized labels are supported in multi-label scenarios.");
                    rgLabelShape.Add(nLen); 
                }

                colTop[1].Reshape(rgLabelShape);

                for (int i = 0; i < m_rgPrefetch.Length; i++)
                {
                    m_rgPrefetch[i].Label.Reshape(rgLabelShape);
                }
            }
        }

        /// <summary>
        /// No bottom blobs are used by this layer.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 0; }
        }

        /// <summary>
        /// Specifies the minimum number of required top (output) Blobs: data
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Specifies the maximum number of required top (output) Blobs: data, lable
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Retrieves the next item from the database and rolls the cursor over once the end 
        /// of the dataset is reached.
        /// </summary>
        protected void Next()
        {
            m_cursor.Next();

            if (!m_cursor.IsValid)
            {
                m_log.WriteLine("Restarting data prefetching from start.");
                m_cursor.SeekToFirst();
            }

            m_nOffset++;
        }

        /// <summary>
        /// Skip to the next value - used when training in a multi-GPU scenario.
        /// </summary>
        /// <returns></returns>
        protected bool Skip()
        {
            UInt64 nSize = (UInt64)m_param.solver_count;
            UInt64 nRank = (UInt64)m_param.solver_rank;
            // In test mode, only rank 0 runs, so avoid skipping.
            bool bKeep = (m_nOffset % nSize) == nRank || m_param.phase == Phase.TEST;

            return !bKeep;
        }

        /// <summary>
        /// Load a batch of data in the background (this is run on an internal thread within the BasePrefetchingDataLayer class).
        /// </summary>
        /// <param name="batch">Specifies the Batch of data to load.</param>
        protected override void load_batch(Batch<T> batch)
        {
            m_log.CHECK(batch.Data.count() > 0, "There is no space allocated for data!");
            int nBatchSize = (int)m_param.data_param.batch_size;

            T[] rgTopLabel = null;

            if (m_bOutputLabels)
                rgTopLabel = batch.Label.mutable_cpu_data;

            if (m_param.data_param.display_timing)
            {
                m_swTimerBatch.Restart();
                m_dfReadTime = 0;
                m_dfTransTime = 0;
            }

            Datum datum;
            int nDim = 0;
            List<int> rgLabels = new List<int>();
            List<int> rgTargetLabels = null;

            // If we are synced with another dataset, wait for it to load the initial data set.
            if (m_param.data_param.synchronize_target)
            {
                int nWait = m_rgBatchLabels.WaitReady;
                if (nWait == 0)
                    return;

                rgTargetLabels = m_rgBatchLabels.Get();
                m_log.CHECK_EQ(nBatchSize, m_rgBatchLabels.Count, "The batch label count (previously loaded by the primary dataset) does not match the batch size '" + m_param.data_param.batch_size.ToString() + "' of this layer!");
            }

            for (int i = 0; i < nBatchSize; i++)
            {
                if (m_param.data_param.display_timing)
                    m_swTimerTransaction.Restart();

                while (Skip())
                {
                    Next();
                }

                if (rgTargetLabels == null)
                    datum = m_cursor.GetValue();
                else
                    datum = m_cursor.GetValue(rgTargetLabels[i]);

                if (m_param.data_param.display_timing)
                {
                    m_dfReadTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;
                    m_swTimerTransaction.Restart();
                }

                if (i == 0)
                {
                    // Reshape according to the first datum of each batch
                    // on single input batches allows for inputs of varying dimension.
                    // Use data transformer to infer the expected blob shape for datum.
                    List<int> rgTopShape = m_transformer.InferBlobShape(datum);

                    // Reshape batch according to the batch size.
                    rgTopShape[0] = nBatchSize;
                    batch.Data.Reshape(rgTopShape);

                    nDim = 1;
                    for (int k = 1; k < rgTopShape.Count; k++)
                    {
                        nDim *= rgTopShape[k];
                    }

                    int nTopLen = nDim * nBatchSize;
                    if (m_rgTopData == null || m_rgTopData.Length != nTopLen)
                        m_rgTopData = new T[nTopLen];
                }

                // Apply data transformations (mirrow, scaling, crop, etc)
                T[] rgTrans = m_transformer.Transform(datum);
                Array.Copy(rgTrans, 0, m_rgTopData, nDim * i, nDim);

                // Copy label.
                if (m_bOutputLabels)
                {
                    if (m_param.data_param.load_multiple_labels)
                    {
                        if (datum.DataCriteria == null || datum.DataCriteria.Length == 0)
                            m_log.FAIL("Could not find the multi-label data.  The data source '" + m_param.data_param.source + "' does not appear to have any Image Criteria data.");

                        // Get the number of items and the item size from the end of the data.
                        int nLen = BitConverter.ToInt32(datum.DataCriteria, datum.DataCriteria.Length - (sizeof(int) * 4));
                        int nItemSize = BitConverter.ToInt32(datum.DataCriteria, datum.DataCriteria.Length - (sizeof(int) * 3));
                        int nDstIdx = i * nLen;

                        m_log.CHECK_EQ(nItemSize, 1, "Currently only byte sized labels are supported in multi-label scenarios.");
                        Array.Copy(datum.DataCriteria, 0, rgTopLabel, nDstIdx, nLen);
                    }
                    else
                    {
                        rgTopLabel[i] = (T)Convert.ChangeType(datum.Label, typeof(T));
                    }
                }

                if (m_param.data_param.display_timing)
                    m_dfTransTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;

                rgLabels.Add(datum.Label);

                Next();
            }

            batch.Data.SetCPUData(m_rgTopData);

            if (m_bOutputLabels)
                batch.Label.SetCPUData(rgTopLabel);

            if (m_param.data_param.display_timing)
            {
                m_swTimerBatch.Stop();
                m_swTimerTransaction.Stop();
                m_log.WriteLine("Prefetch batch: " + m_swTimerBatch.ElapsedMilliseconds.ToString() + " ms.", true);
                m_log.WriteLine("     Read time: " + m_dfReadTime.ToString() + " ms.", true);
                m_log.WriteLine("Transform time: " + m_dfTransTime.ToString() + " ms.", true);
            }

            if (m_param.data_param.synchronize_target)
                m_rgBatchLabels.Done();

            if (OnBatchLoad != null)
                OnBatchLoad(this, new LastBatchLoadedArgs(rgLabels));
        }
    }

    class LabelCollection : IDisposable /** @private */
    {
        ManualResetEvent m_evtReady = new ManualResetEvent(false);
        ManualResetEvent m_evtDone = new ManualResetEvent(true);
        AutoResetEvent m_evtCancel = new AutoResetEvent(false);
        List<int> m_rgLabels = new List<int>();
        object m_sync = new object();

        public LabelCollection()
        {
        }

        public void Dispose()
        {
            if (m_evtReady != null)
            {
                m_evtReady.Dispose();
                m_evtReady = null;
            }

            if (m_evtDone != null)
            {
                m_evtDone.Dispose();
                m_evtDone = null;
            }

            if (m_evtCancel != null)
            {
                m_evtCancel.Dispose();
                m_evtCancel = null;
            }
        }

        public void Cancel()
        {
            m_evtCancel.Set();
        }

        public int WaitReady
        {
            get
            {
                List<WaitHandle> rgWait = new List<WaitHandle>() { m_evtCancel, m_evtReady };
                return WaitHandle.WaitAny(rgWait.ToArray());
            }
        }

        public int WaitProcessing
        {
            get
            {
                List<WaitHandle> rgWait = new List<WaitHandle>() { m_evtCancel, m_evtDone };
                return WaitHandle.WaitAny(rgWait.ToArray());
            }
        }

        public void Done()
        {
            m_evtDone.Set();
        }

        public void Set(List<int> rg)
        {
            lock (m_sync)
            {
                m_rgLabels = new List<int>(rg);
                m_evtReady.Set();
            }
        }

        public int Count
        {
            get
            {
                lock (m_sync)
                {
                    return m_rgLabels.Count;
                }
            }
        }

        public List<int> Get()
        {
            lock (m_sync)
            {
                List<int> rg = new List<int>(m_rgLabels);
                m_evtReady.Reset();
                m_evtDone.Reset();
                return rg;
            }
        }
    }

    /// <summary>
    /// Specifies the arguments sent to the OnBatchLoad event used when synchronizing between Data Layers.
    /// </summary>
    public class LastBatchLoadedArgs : EventArgs
    {
        List<int> m_rgLabels;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="rgLabels">Specifies the labels loaded.</param>
        public LastBatchLoadedArgs(List<int> rgLabels)
        {
            m_rgLabels = new List<int>(rgLabels);
        }

        /// <summary>
        /// Returns the labels loaded.
        /// </summary>
        public List<int> Labels
        {
            get { return m_rgLabels; }
        }
    }
}
