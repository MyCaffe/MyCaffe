using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Remoting.Messaging;
using System.Security.Cryptography;
using System.Security.Policy;
using System.Text;
using System.Threading;
using System.Xml.Linq;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.common;
using MyCaffe.db.temporal;
using MyCaffe.param;
using MyCaffe.param.tft;
//using SimpleGraphing;

namespace MyCaffe.layers.tft
{
    /// <summary>
    /// The DataTemporalLayer implements the data layer used to load the temporal data into the model.
    /// </summary>
    /// <remarks>
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L405) by Playtika Research, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DataTemporalLayer<T> : Layer<T>
    {
        IXTemporalDatabaseBase m_db;
        List<int> m_rgShape = new List<int>(4);
        uint m_nBatchSize;
        uint m_nNumHistoricalSteps;
        uint m_nNumFutureSteps;
        RawData<T> m_data = null;
        CancelEvent m_evtCancel;
        int[,] m_rgIdx = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type DATA_TEMPORAL with parameter data_temporal_param</param>
        /// <param name="evtCancel">Specifies the cancel event used to cancel background data loading.</param>
        /// <param name="db">Specifies the in-memory database.</param>
        public DataTemporalLayer(CudaDnn<T> cuda, Log log, LayerParameter p, CancelEvent evtCancel, IXDatabaseBase db)
            : base(cuda, log, p)
        {
            m_evtCancel = evtCancel;
            m_type = LayerParameter.LayerType.DATA_TEMPORAL;
            m_db = db as IXTemporalDatabaseBase;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;
        }

        /// <summary>
        /// The data layer has no bottom blobs.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 0; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: static_numeric, static_categorical, hist_numeric, hist_categorical, future_numeric, future_categorical
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 6; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: static_numeric, static_categorical, hist_numeric, hist_categorical, future_numeric, future_categorical, [target_hist], [time], [mask]
        /// </summary>
        public override int MaxTopBlobs
        {
            get 
            {
                int nTops = 7;

                if (m_param.data_temporal_param.output_target_historical)
                    nTops++;

                if (m_param.data_temporal_param.output_time)
                    nTops++;

                if (m_param.data_temporal_param.output_mask)
                    nTops++;

                return nTops;
            }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nBatchSize = m_param.data_temporal_param.batch_size;
            m_nNumHistoricalSteps = m_param.data_temporal_param.num_historical_steps;
            m_nNumFutureSteps = m_param.data_temporal_param.num_future_steps;

            if (m_data == null)
            {
                if (m_param.data_temporal_param.source_type == DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE)
                    m_data = new RawFileData<T>(m_param.data_temporal_param.seed, m_param.data_temporal_param.output_target_historical, m_param.data_temporal_param.output_time, m_param.data_temporal_param.output_mask, m_param.data_temporal_param.max_load_percent, m_param.data_temporal_param.drip_refresh_rate_in_sec, m_param.data_temporal_param.chunk_count);
                else if (m_param.data_temporal_param.source_type == DataTemporalParameter.SOURCE_TYPE.SQL_DB)
                    m_data = new RawSqlData<T>(m_param.data_temporal_param.seed, m_param.data_temporal_param.output_target_historical, m_param.data_temporal_param.output_time, m_param.data_temporal_param.output_mask, m_db, m_log, m_param.data_temporal_param.max_load_percent, m_param.data_temporal_param.drip_refresh_rate_in_sec, m_param.data_temporal_param.chunk_count);
                else if (m_param.data_temporal_param.source_type == DataTemporalParameter.SOURCE_TYPE.DIRECT)
                    m_data = new RawDirectData<T>(m_param.data_temporal_param.seed, m_param.data_temporal_param.output_target_historical, m_param.data_temporal_param.output_time, m_param.data_temporal_param.output_mask, m_db, m_log);
                else
                    throw new Exception("Unknown source type: " + m_param.data_temporal_param.source_type.ToString());
            }

            Phase phase = m_phase;
            if (m_param.data_temporal_param.forced_phase.HasValue)
            {
                m_log.WriteLine("INFO: Using forced phase = " + m_param.data_temporal_param.forced_phase.Value.ToString() + ".");
                phase = m_param.data_temporal_param.forced_phase.Value;
            }

            if (!m_data.LoadData(phase, m_param.data_temporal_param.source, m_param.data_temporal_param.shuffle_data, (int)m_param.data_temporal_param.batch_size, (int)m_nNumHistoricalSteps, (int)m_nNumFutureSteps, m_log, m_evtCancel))
                throw new Exception("DataTemporalLayer - could not find the data for '" + m_param.data_temporal_param.source + "'. You may need to run the SignalPop AI Designer to create this " + m_param.data_temporal_param.source_type.ToString() + " dataset.");

            int nTotalSize = m_data.GetTotalSize();
            m_log.CHECK_GE(nTotalSize, m_nBatchSize, "There must be enough items for at least one batch - items found = " + nTotalSize.ToString() + ", batch size = " + m_nBatchSize.ToString());
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int[] rgShape;

            if ((rgShape = m_data.GetShape(DataNpy<T>.OUTPUT_TYPE.STATIC_NUMERIC)) != null)
                colTop[0].Reshape(rgShape);

            if ((rgShape = m_data.GetShape(DataNpy<T>.OUTPUT_TYPE.STATIC_CATEGORICAL)) != null)
                colTop[1].Reshape(rgShape);

            if ((rgShape = m_data.GetShape(DataNpy<T>.OUTPUT_TYPE.HISTORICAL_NUMERIC)) != null)
                colTop[2].Reshape(rgShape);

            if ((rgShape = m_data.GetShape(DataNpy<T>.OUTPUT_TYPE.HISTORICAL_CATEGORICAL)) != null)
                colTop[3].Reshape(rgShape);

            if ((rgShape = m_data.GetShape(DataNpy<T>.OUTPUT_TYPE.FUTURE_NUMERIC)) != null)
                colTop[4].Reshape(rgShape);

            if ((rgShape = m_data.GetShape(DataNpy<T>.OUTPUT_TYPE.FUTURE_CATEGORICAL)) != null)
                colTop[5].Reshape(rgShape);

            if (colTop.Count > 6)
            {
                if ((rgShape = m_data.GetShape(DataNpy<T>.OUTPUT_TYPE.TARGET)) != null)
                {
                    colTop[6].Reshape(rgShape);
                    colTop[6].type = BLOB_TYPE.TARGET;
                }

                int nIdx = 7;
                if (m_param.data_temporal_param.output_target_historical && colTop.Count > nIdx)
                {
                    rgShape[1] = (int)m_nNumHistoricalSteps;
                    colTop[nIdx].Reshape(rgShape);
                    colTop[nIdx].type = BLOB_TYPE.TARGET | BLOB_TYPE.DATA;
                    nIdx++;
                }

                if (m_param.data_temporal_param.output_time && colTop.Count > nIdx)
                {
                    rgShape[1] = (int)m_nNumHistoricalSteps;
                    colTop[nIdx].Reshape(rgShape);
                    colTop[nIdx].type = BLOB_TYPE.TIME | BLOB_TYPE.DATA;
                    nIdx++;
                }

                if (m_param.data_temporal_param.output_mask && colTop.Count > nIdx)
                {
                    rgShape[1] = (int)m_nNumHistoricalSteps;
                    colTop[nIdx].Reshape(rgShape);
                    colTop[nIdx].type = BLOB_TYPE.MASK | BLOB_TYPE.DATA;
                    nIdx++;
                }
            }
        }

        /// <summary>
        /// Connect the loss layer to the data layer so that we can rank the data values.
        /// </summary>
        /// <param name="layer">Specifies the loss layer to connect.</param>
        public override void ConnectLoss(LossLayer<T> layer)
        {
            layer.OnLoss += Layer_OnLoss;
        }

        private void Layer_OnLoss(object sender, LossArgs e)
        {
            if (m_rgIdx != null)
                m_data.Add(e, m_rgIdx);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector)
        ///  -# @f$ (N \times C \times H \times W size) @f$
        ///     the computed outputs @f$ y @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Phase phase = layer_param.data_temporal_param.forced_phase.GetValueOrDefault(m_phase);
            m_rgIdx = m_data.LoadBatch(phase, (int)m_nBatchSize, colTop, m_param.data_temporal_param.enable_debug_output, m_param.data_temporal_param.debug_output_path);

            if (m_param.data_temporal_param.enable_debug_output)
                m_log.WriteLine("WARNING: Debugging is enabled with path = " + m_param.data_temporal_param.debug_output_path + " and will slow down training!");
        }

        /// @brief Not implemented - data Layers do not perform backward.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }
    }

    /// <summary>
    /// The RawData class is the base class for all raw data types.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    abstract class RawData<T>
    {
        /// <summary>
        /// Specifies the base data object used to store data blocks loaded from disk or database.
        /// </summary>
        protected Data<T> m_data;
        /// <summary>
        /// Specifies the random number generator used to shuffle the data.
        /// </summary>
        protected Random m_random;
        /// <summary>
        /// Specifies the batch size.
        /// </summary>
        protected int m_nBatchSize;
        /// <summary>
        /// Specifies to output the target historical data.
        /// </summary>
        protected bool m_bOutputTargetHistorical;
        /// <summary>
        /// Specifies to output the time for each item in a separate blob.
        /// </summary>
        protected bool m_bOutputTime;
        /// <summary>
        /// Specifies to output the mask (all 1's) matching each item in a separate blob.
        /// </summary>
        protected bool m_bOutputMask;
        /// <summary>
        /// Specifies that the data is loaded and ready.
        /// </summary>
        protected ManualResetEvent m_evtReady = new ManualResetEvent(false);
        /// <summary>
        /// Specifies that the data is done loading.
        /// </summary>
        protected ManualResetEvent m_evtDone = new ManualResetEvent(false);

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nSeed">Specifies the random number generator seed.</param>
        /// <param name="bOutputTargetHistorical">Specifies to output the target historical data.</param>
        /// <param name="bOutputTime">Specifies to output the time matching each item in a separate blob.</param>
        /// <param name="bOutputMask">Specifies to output the mask (all 1's) matching each item in a separate blob.</param>
        public RawData(uint? nSeed, bool bOutputTargetHistorical, bool bOutputTime, bool bOutputMask)
        {
            m_bOutputTargetHistorical = bOutputTargetHistorical;
            m_bOutputTime = bOutputTime;
            m_bOutputMask = bOutputMask;

            if (nSeed.HasValue)
                m_random = new Random((int)nSeed.Value);
            else
                m_random = new Random();
        }

        /// <summary>
        /// Specifies the random number generator used.
        /// </summary>
        public Random Random
        {
            get { return m_random; }
        }

        /// <summary>
        /// Loads all data values for the phase specified.
        /// </summary>
        /// <param name="phase">Specifies the phase to load.</param>
        /// <param name="strPath">Specifies the base path for all data.</param>
        /// <param name="bShuffleData">Specifies to randomly select from the data.</param>
        /// <param name="nBatchSize">Specifies the batch size.</param>
        /// <param name="nHistoricalSteps">Specifies the number of historical steps.</param>
        /// <param name="nFutureSteps">Specifies the number of future steps.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        public virtual bool LoadData(Phase phase, string strPath, bool bShuffleData, int nBatchSize, int nHistoricalSteps, int nFutureSteps, Log log, CancelEvent evtCancel)
        {
            m_nBatchSize = nBatchSize;

            Thread threadLoad = new Thread(new ParameterizedThreadStart(loadDataFunction));
            DataLoadParameters p = getDataLoadParameters(phase, strPath, bShuffleData, nHistoricalSteps, nFutureSteps, log, evtCancel, m_evtReady, m_evtDone);
            threadLoad.Start(p);

            while (!m_evtReady.WaitOne(1000))
            {
                if (evtCancel.WaitOne(0))
                    return false;

                Thread.Sleep(50);
            }

            return true;
        }

        /// <summary>
        /// Get the data load parameters sent to the data load function thread.
        /// </summary>
        /// <param name="phase">Specifies the phase to load.</param>
        /// <param name="strPath">Specifies the base path for all data.</param>
        /// <param name="bShuffleData">Specifies to randomly select from the data.</param>
        /// <param name="nHistoricalSteps">Specifies the number of historical steps.</param>
        /// <param name="nFutureSteps">Specifies the number of future steps.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="evtReady">Specifies the event set when the data is ready.</param>
        /// <param name="evtDone">Specifies the event set when the data loading is done.</param>
        /// <returns>A new DataLoadParameters parameter is returned.</returns>
        protected virtual DataLoadParameters getDataLoadParameters(Phase phase, string strPath, bool bShuffleData, int nHistoricalSteps, int nFutureSteps, Log log, CancelEvent evtCancel, ManualResetEvent evtReady, ManualResetEvent evtDone)
        {
            return new DataLoadParameters(phase, strPath, nHistoricalSteps, nFutureSteps, 0, 0, 0, bShuffleData, log, evtCancel, evtReady, evtDone);
        }

        /// <summary>
        /// The virtual load data function override by the derived class to load the data in the background.
        /// </summary>
        /// <param name="obj">Specifies the user state.</param>
        protected virtual void loadDataFunction(object obj)
        {
        }

        /// <summary>
        /// Loads a batch of data items into the BlobCollection.
        /// </summary>
        /// <param name="nBatchSize">Specifies the batch size.</param>
        /// <param name="col">Specifies the blob collection to load the batch into.</param>
        /// <param name="phase">Specifies the phase.</param>
        /// <param name="bEnableDebug">Optionally, specifies to enable debug output (default = false).</param>
        /// <param name="strDebugPath">Optionally, specifies the debug path where debug images are placed when 'EnableDebug' = true.</param>
        /// <returns>An array of the selected item and indexes is returned.</returns>
        public virtual int[,] LoadBatch(Phase phase, int nBatchSize, BlobCollection<T> col, bool bEnableDebug = false, string strDebugPath = null)
        {
            return m_data.LoadBatch(nBatchSize, col, bEnableDebug, strDebugPath);
        }

        /// <summary>
        /// Adds the selected indexes along with the loss data (used by the BatchPerfSet to select worst cases).
        /// </summary>
        /// <param name="e">Specifies the loss args.</param>
        /// <param name="rgIdx">Specifies the selected item/value indexes for the batch.</param>
        public virtual void Add(LossArgs e, int[,] rgIdx)
        {
        }

        /// <summary>
        /// Returns the total size of the data.
        /// </summary>
        /// <returns>The total size is returned.</returns>
        public virtual int GetTotalSize()
        {
            return m_data.GetTotalSize();
        }

        /// <summary>
        /// Returns the shape of a given output type.
        /// </summary>
        /// <param name="ot">Specifies the output type.</param>
        /// <returns>The shape returned can be used to reshape the Blob used to store the data on the GPU.</returns>
        public virtual int[] GetShape(DataNpy<T>.OUTPUT_TYPE ot)
        {
            return m_data.GetShape(ot);
        }
    }

    /// <summary>
    /// The RawSqlData class loads data from a database.
    /// </summary>
    /// <typeparam name="T">Specifies the base type.</typeparam>
    class RawDirectData<T> : RawData<T>
    {
        /// <summary>
        /// Specifies the dataset descriptor for the data.
        /// </summary>
        protected DatasetDescriptor m_ds;
        /// <summary>
        /// Specifies the in-memory temporal database.
        /// </summary>
        protected IXTemporalDatabaseBase m_db;
        /// <summary>
        /// Specifies to shuffle the data when building batches.
        /// </summary>
        protected bool m_bShuffleData;
        /// <summary>
        /// Specifies the number of historical steps.
        /// </summary>
        protected int m_nHistoricalSteps;
        /// <summary>
        /// Specifies the number of future steps.
        /// </summary>
        protected int m_nFutureSteps;
        /// <summary>
        /// Specifies the output log.
        /// </summary>
        protected Log m_log;
        /// <summary>
        /// Specifies the phase.
        /// </summary>
        protected Phase m_phase = Phase.NONE;
        /// <summary>
        /// Specifies the static numeric data buffer.
        /// </summary>
        protected float[] m_rgStaticNum = null;
        /// <summary>
        /// Specifies the static categorical data buffer.
        /// </summary>
        protected float[] m_rgStaticCat = null;
        /// <summary>
        /// Specifies the historical numeric data buffer.
        /// </summary>
        protected float[] m_rgHistoricalNum = null;
        /// <summary>
        /// Specifies the historical categorical data buffer.
        /// </summary>
        protected float[] m_rgHistoricalCat = null;
        /// <summary>
        /// Specifies the future numeric data buffer.
        /// </summary>
        protected float[] m_rgFutureNum = null;
        /// <summary>
        /// Specifies the future categorical data buffer.
        /// </summary>
        protected float[] m_rgFutureCat = null;
        /// <summary>
        /// Specifies the target data buffer.
        /// </summary>
        protected float[] m_rgTarget = null;
        /// <summary>
        /// Specifies the historical target data buffer.
        /// </summary>
        protected float[] m_rgTargetHist = null;
        /// <summary>
        /// Specifies the time data buffer.
        /// </summary>
        protected float[] m_rgTime = null;
        /// <summary>
        /// Specifies the mask data buffer.
        /// </summary>
        protected float[] m_rgMask = null;
        /// <summary>
        /// Specifies the indexes used.
        /// </summary>
        protected int[,] m_rgIdx = null;
        /// <summary>
        /// Specifies the batch performance set used to select the worst cases.
        /// </summary>
        protected BatchPerfSet m_batchPerfSet = null;


        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nSeed">Specifies the random number generator seed.</param>
        /// <param name="bOutputTargetHistorical">Specifies to output the target historical data.</param>
        /// <param name="bOutputTime">Specifies to output the time matching each item in a separate blob.</param>
        /// <param name="bOutputMask">Specifies to output the mask (all 1's) matching each item in a separate blob.</param>
        /// <param name="db">Specifies the external database.</param>
        /// <param name="log">Specifies the output log.</param>
        public RawDirectData(uint? nSeed, bool bOutputTargetHistorical, bool bOutputTime, bool bOutputMask, IXTemporalDatabaseBase db, Log log) : base(nSeed, bOutputTargetHistorical, bOutputTime, bOutputMask)
        {
            m_db = db;
            m_log = log;
        }

        /// <summary>
        /// Loads all data values for the phase specified.
        /// </summary>
        /// <param name="phase">Specifies the phase for which the data is to be loaded (e.g., TRAIN, TEST)</param>
        /// <param name="strDataset">Specifies the name of the dataset.</param>
        /// <param name="bShuffleData">Specifies to shuffle the data.</param>
        /// <param name="nBatchSize">Specifies the batch size.</param>
        /// <param name="nHistoricalSteps">Specifies the number of historical steps (before current time).</param>
        /// <param name="nFutureSteps">Specifies the number of future steps (after current time).</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the event used to cancel loading.</param>
        /// <returns>True is returned if the data is loaded successfully, otherwise false.</returns>
        public override bool LoadData(Phase phase, string strDataset, bool bShuffleData, int nBatchSize, int nHistoricalSteps, int nFutureSteps, Log log, CancelEvent evtCancel)
        {
            m_phase = phase;

            if (m_db == null)
                throw new Exception("The direct load requires an already initialized database.");

            m_ds = m_db.GetDatasetByName(strDataset);
            if (m_ds == null)
            {
                m_log.WriteLine("ERROR: Could not find the dataset '" + strDataset + "'!");
                return false;
            }

            m_bShuffleData = bShuffleData;
            m_nBatchSize = nBatchSize;
            m_nHistoricalSteps = nHistoricalSteps;
            m_nFutureSteps = nFutureSteps;

            return true;
        }

        private float[] getBuffer(BlobCollection<T> col, int nIdx)
        {
            if (col.Count <= nIdx)
                return null;

            int nItemCount = col[nIdx].count();
            if (nItemCount == 0)
                return null;

            return new float[nItemCount];
        }

        private void setBuffer(BlobCollection<T> col, int nIdx, float[] rg)
        {
            if (rg == null)
                return;

            col[nIdx].mutable_cpu_data = Utility.ConvertVec<T>(rg);
        }

        /// <summary>
        /// Add the loss data for the batch into the performance data later used to select the worst cases.
        /// </summary>
        /// <param name="e">Specifies the loss data.</param>
        /// <param name="rgIdx">Specifies the selected indexes for the batch.</param>
        public override void Add(LossArgs e, int[,] rgIdx)
        {
            if (m_batchPerfSet == null)
                m_batchPerfSet = new BatchPerfSet(m_random, 0.25, (int)m_nBatchSize * 100, 2);

            m_batchPerfSet.Add(e, rgIdx);
        }

        /// <summary>
        /// Load a batch of data to feed into the network.
        /// </summary>
        /// <param name="phase">Specifies the phase being loaded (e.g., TRAIN, TEST).</param>
        /// <param name="nBatchSize">Specifies the batch size.</param>
        /// <param name="col">Specifies the collection of blobs to load.</param>
        /// <param name="bEnableDebug">Optionally, specifies to enable debug output (default = false).</param>
        /// <param name="strDebugPath">Optionally, specifies the debug path where debug images are placed when 'EnableDebug' = true.</param>
        /// <returns>The list of selected indexes is returned.</returns>
        public override int[,] LoadBatch(Phase phase, int nBatchSize, BlobCollection<T> col, bool bEnableDebug = false, string strDebugPath = null)
        {
            SourceDescriptor src = (phase == Phase.TRAIN) ? m_ds.TrainingSource : m_ds.TestingSource;
            DB_LABEL_SELECTION_METHOD itemSelection = (m_bShuffleData) ? DB_LABEL_SELECTION_METHOD.RANDOM : DB_LABEL_SELECTION_METHOD.NONE;
            DB_ITEM_SELECTION_METHOD valueSelection = (m_bShuffleData) ? DB_ITEM_SELECTION_METHOD.RANDOM : DB_ITEM_SELECTION_METHOD.NONE;

            if (m_rgStaticNum == null)
                m_rgStaticNum = getBuffer(col, 0);
            if (m_rgStaticCat == null)
                m_rgStaticCat = getBuffer(col, 1);
            if (m_rgHistoricalNum == null)
                m_rgHistoricalNum = getBuffer(col, 2);
            if (m_rgHistoricalCat == null)
                m_rgHistoricalCat = getBuffer(col, 3);
            if (m_rgFutureNum == null)
                m_rgFutureNum = getBuffer(col, 4);
            if (m_rgFutureCat == null)
                m_rgFutureCat = getBuffer(col, 5);
            if (m_rgTarget == null)
                m_rgTarget = getBuffer(col, 6);

            int nIdx = 7;

            if (m_bOutputTargetHistorical)
            {
                if (m_rgTargetHist == null)
                    m_rgTargetHist = getBuffer(col, nIdx);
                nIdx++;
            }

            if (m_bOutputTime)
            {
                if (m_rgTime == null)
                    m_rgTime = getBuffer(col, nIdx);
                nIdx++;
            }

            if (m_bOutputMask)
            {
                if (m_rgMask == null)
                    m_rgMask = getBuffer(col, nIdx);
            }

            if (m_rgIdx == null)
                m_rgIdx = new int[nBatchSize, 2];

            for (int i = 0; i < nBatchSize; i++)
            {
                int? nItemIdx = null;
                int? nValueIdx = null;

                // When using the batch performance set, the indexes are selected from the set,
                // seeking to select from the 25% worst performing items.
                if (m_batchPerfSet != null)
                    m_batchPerfSet.Select(ref nItemIdx, ref nValueIdx);

                SimpleTemporalDatumCollection rgData = m_db.QueryTemporalItem(i, src.ID, ref nItemIdx, ref nValueIdx, itemSelection, valueSelection, bEnableDebug, strDebugPath);
                if (rgData == null)
                    continue;

                m_rgIdx[i, 0] = nItemIdx.Value;
                m_rgIdx[i, 1] = nValueIdx.Value;

                SimpleTemporalDatum sdStatNum = rgData[0];
                SimpleTemporalDatum sdStatCat = rgData[1];
                SimpleTemporalDatum sdHistNum = rgData[2];
                SimpleTemporalDatum sdHistCat = rgData[3];
                SimpleTemporalDatum sdFutureNum = rgData[4];
                SimpleTemporalDatum sdFutureCat = rgData[5];
                SimpleTemporalDatum sdTarget = rgData[6];
                SimpleTemporalDatum sdTargetHist = null;
                SimpleTemporalDatum sdTime = null;
                SimpleTemporalDatum sdMask = null;

                nIdx = 7;
                if (m_bOutputTargetHistorical)
                {
                    sdTargetHist = rgData[nIdx];
                    nIdx++;
                }

                if (m_bOutputTime)
                {
                    sdTime = rgData[nIdx];
                    nIdx++;
                }

                if (m_bOutputMask)
                {
                    sdMask = rgData[nIdx];
                }

                // col[0] = STATIC_NUMERIC
                if (m_rgStaticNum != null)
                {
                    float[] rgRawData = sdStatNum.Data;
                    Array.Copy(rgRawData, 0, m_rgStaticNum, i * rgRawData.Length, rgRawData.Length);
                }

                // col[1] = STATIC_CATEGORICAL
                if (m_rgStaticCat != null)
                {
                    float[] rgRawData = sdStatCat.Data;
                    Array.Copy(rgRawData, 0, m_rgStaticCat, i * rgRawData.Length, rgRawData.Length);
                }

                // col[2] = HISTORICAL_NUMERIC
                if (m_rgHistoricalNum != null)
                {
                    float[] rgRawData = sdHistNum.Data;
                    Array.Copy(rgRawData, 0, m_rgHistoricalNum, i * rgRawData.Length, rgRawData.Length);
                }

                // col[3] = HISTORICAL_CATEGORICAL
                if (m_rgHistoricalCat != null)
                {
                    float[] rgRawData = sdHistCat.Data;
                    Array.Copy(rgRawData, 0, m_rgHistoricalCat, i * rgRawData.Length, rgRawData.Length);
                }

                // col[4] = FUTURE_NUMERIC
                if (m_rgFutureNum != null)
                {
                    float[] rgRawData = sdFutureNum.Data;
                    Array.Copy(rgRawData, 0, m_rgFutureNum, i * rgRawData.Length, rgRawData.Length);
                }

                // col[5] = FUTURE_CATEGORICAL
                if (m_rgFutureCat != null)
                {
                    float[] rgRawData = sdFutureCat.Data;
                    Array.Copy(rgRawData, 0, m_rgFutureCat, i * rgRawData.Length, rgRawData.Length);
                }

                // col[6] = TARGET
                if (m_rgTarget != null)
                {
                    float[] rgRawData = sdTarget.Data;
                    Array.Copy(rgRawData, 0, m_rgTarget, i * rgRawData.Length, rgRawData.Length);
                }

                // col[7] = Historical Target (optional)
                if (m_rgTargetHist != null)
                {
                    float[] rgRawData = sdTargetHist.Data;
                    Array.Copy(rgRawData, 0, m_rgTargetHist, i * rgRawData.Length, rgRawData.Length);
                }

                // col[8] = Time (optional)
                if (m_rgTime != null)
                {
                    float[] rgRawData = sdTime.Data;
                    Array.Copy(rgRawData, 0, m_rgTime, i * rgRawData.Length, rgRawData.Length);
                }

                // col[9] = Mask (optional)
                if (m_rgMask != null)
                {
                    float[] rgRawData = sdMask.Data;
                    Array.Copy(rgRawData, 0, m_rgMask, i * rgRawData.Length, rgRawData.Length);
                }
            }

            setBuffer(col, 0, m_rgStaticNum);
            setBuffer(col, 1, m_rgStaticCat);
            setBuffer(col, 2, m_rgHistoricalNum);
            setBuffer(col, 3, m_rgHistoricalCat);
            setBuffer(col, 4, m_rgFutureNum);
            setBuffer(col, 5, m_rgFutureCat);
            setBuffer(col, 6, m_rgTarget);

            nIdx = 7;

            if (m_bOutputTargetHistorical)
            {
                setBuffer(col, nIdx, m_rgTargetHist);
                nIdx++;
            }

            if (m_bOutputTime)
            {
                setBuffer(col, nIdx, m_rgTime);
                nIdx++;
            }

            if (m_bOutputMask)
            {
                setBuffer(col, nIdx, m_rgMask);
                nIdx++;
            }

            return m_rgIdx;
        }

        /// <summary>
        /// Return the total number of blocks available in the current phase.
        /// </summary>
        /// <returns>The total number of blocks is returned.</returns>
        public override int GetTotalSize()
        {
            return m_db.GetTotalSize(m_ds.ID, m_phase, m_nHistoricalSteps, m_nFutureSteps);
        }

        /// <summary>
        /// Return the shape of the OUTPUT_TYPE.
        /// </summary>
        /// <param name="ot">Specifies the output type.</param>
        /// <returns>The shape array is returned.</returns>
        public override int[] GetShape(DataNpy<T>.OUTPUT_TYPE ot)
        {
            int nStaticNumCount = 0;
            int nStaticCatCount = 0;
            int nObservedNumCount = 0;
            int nObservedCatCount = 0;
            int nKnownNumCount = 0;
            int nKnownCatCount = 0;

            foreach (ValueStreamDescriptor vsd in m_ds.TrainingSource.TemporalDescriptor.ValueStreamDescriptors)
            {
                if (vsd.ClassType == ValueStreamDescriptor.STREAM_CLASS_TYPE.STATIC)
                {
                    if (vsd.ValueType == ValueStreamDescriptor.STREAM_VALUE_TYPE.NUMERIC)
                        nStaticNumCount++;
                    else
                        nStaticCatCount++;
                }

                else if (vsd.ClassType == ValueStreamDescriptor.STREAM_CLASS_TYPE.OBSERVED)
                {
                    if (vsd.ValueType == ValueStreamDescriptor.STREAM_VALUE_TYPE.NUMERIC)
                        nObservedNumCount++;
                    else
                        nObservedCatCount++;
                }

                else if (vsd.ClassType == ValueStreamDescriptor.STREAM_CLASS_TYPE.KNOWN)
                {
                    if (vsd.ValueType == ValueStreamDescriptor.STREAM_VALUE_TYPE.NUMERIC)
                        nKnownNumCount++;
                    else
                        nKnownCatCount++;

                }
            }

            switch (ot)
            {
                case Data<T>.OUTPUT_TYPE.STATIC_CATEGORICAL:
                    if (nStaticCatCount == 0)
                        return null;
                    return new int[] { m_nBatchSize, nStaticCatCount };

                case Data<T>.OUTPUT_TYPE.STATIC_NUMERIC:
                    if (nStaticNumCount == 0)
                        return null;
                    return new int[] { m_nBatchSize, nStaticNumCount };

                case Data<T>.OUTPUT_TYPE.HISTORICAL_SYNC:
                    return new int[] { m_nBatchSize, m_nHistoricalSteps, 1 };

                case Data<T>.OUTPUT_TYPE.HISTORICAL_CATEGORICAL:
                    if (nKnownCatCount + nObservedCatCount == 0)
                        return null;
                    return new int[] { m_nBatchSize, m_nHistoricalSteps, nKnownCatCount + nObservedCatCount, 1 };

                case Data<T>.OUTPUT_TYPE.HISTORICAL_NUMERIC:
                    if (nKnownNumCount + nObservedNumCount == 0)
                        return null;
                    return new int[] { m_nBatchSize, m_nHistoricalSteps, nKnownNumCount + nObservedNumCount, 1 };

                case Data<T>.OUTPUT_TYPE.FUTURE_SYNC:
                    return new int[] { m_nBatchSize, m_nFutureSteps, 1 };

                case Data<T>.OUTPUT_TYPE.FUTURE_CATEGORICAL:
                    if (nKnownCatCount == 0)
                        return null;
                    return new int[] { m_nBatchSize, m_nFutureSteps, nKnownCatCount, 1 };

                case Data<T>.OUTPUT_TYPE.FUTURE_NUMERIC:
                    if (nKnownNumCount == 0)
                        return null;
                    return new int[] { m_nBatchSize, m_nFutureSteps, nKnownNumCount, 1 };

                case Data<T>.OUTPUT_TYPE.TARGET:
                    return new int[] { m_nBatchSize, m_nFutureSteps, 1, 1 };
            }

            return null;
        }
    }

    /// <summary>
    /// The RawSqlData class loads data from a database.
    /// </summary>
    /// <typeparam name="T">Specifies the base type.</typeparam>
    class RawSqlData<T> : RawDirectData<T>
    {
        int m_nDropRefreshReateInSec;
        double m_dfPctMaxLoad;
        uint m_nChunkCount;


        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nSeed">Specifies the random number generator seed.</param>
        /// <param name="bOutputTargetHistorical">Specifies to output the target historical data.</param>
        /// <param name="bOutputTime">Specifies to output the time matching each item in a separate blob.</param>
        /// <param name="bOutputMask">Specifies to output the mask (all 1's) matching each item in a separate blob.</param>
        /// <param name="db">Specifies the external database.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="nDripRefreshRateInSec">Specifies how often in seconds to refresh the data.</param>
        /// <param name="dfPctMaxLoad">Specifies the maximum percentage to load into memory.</param>
        /// <param name="nChunkCount">Specifies the number of chunks (blocks) to refresh.</param>
        public RawSqlData(uint? nSeed, bool bOutputTargetHistorical, bool bOutputTime, bool bOutputMask, IXTemporalDatabaseBase db, Log log, double dfPctMaxLoad, int nDripRefreshRateInSec, uint nChunkCount) : base(nSeed, bOutputTargetHistorical, bOutputTime, bOutputMask, db, log)
        {
            m_nDropRefreshReateInSec = nDripRefreshRateInSec;
            m_dfPctMaxLoad = dfPctMaxLoad;
            m_nChunkCount = nChunkCount;
        }

        /// <summary>
        /// Loads all data values for the phase specified.
        /// </summary>
        /// <param name="phase">Specifies the phase for which the data is to be loaded (e.g., TRAIN, TEST)</param>
        /// <param name="strDataset">Specifies the name of the dataset.</param>
        /// <param name="bShuffleData">Specifies to shuffle the data.</param>
        /// <param name="nBatchSize">Specifies the batch size.</param>
        /// <param name="nHistoricalSteps">Specifies the number of historical steps (before current time).</param>
        /// <param name="nFutureSteps">Specifies the number of future steps (after current time).</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the event used to cancel loading.</param>
        /// <returns>True is returned if the data is loaded successfully, otherwise false.</returns>
        public override bool LoadData(Phase phase, string strDataset, bool bShuffleData, int nBatchSize, int nHistoricalSteps, int nFutureSteps, Log log, CancelEvent evtCancel)
        {
            SettingsCaffe s = null;

            m_phase = phase;

            if (m_db == null)
            {
                s = new SettingsCaffe();
                s.DbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;
                s.DbLoadLimit = 0;

                PropertySet prop = new PropertySet();
                prop.SetProperty("NormalizedData", "True");
                prop.SetProperty("HistoricalSteps", nHistoricalSteps.ToString());
                prop.SetProperty("FutureSteps", nFutureSteps.ToString());

                m_db = new MyCaffeTemporalDatabase(m_log, prop);
            }

            m_ds = m_db.GetDatasetByName(strDataset);
            if (m_ds == null)
            {
                m_log.WriteLine("ERROR: Could not find the dataset '" + strDataset + "'!");
                return false;
            }

            if (s != null)
                m_db.InitializeWithDsName1(s, strDataset);

            m_bShuffleData = bShuffleData;
            m_nBatchSize = nBatchSize;
            m_nHistoricalSteps = nHistoricalSteps;
            m_nFutureSteps = nFutureSteps;

            return true;
        }
    }

    /// <summary>
    /// The RawFileData object is used to load raw NPY file data.
    /// </summary>
    /// <typeparam name="T">Specifies the base data type of 'float' or 'double'.</typeparam>
    class RawFileData<T> : RawData<T>
    {
        int m_nDropRefreshReateInSec;
        double m_dfPctMaxLoad;
        uint m_nChunkCount;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nSeed">Specifies the random number generator seed.</param>
        /// <param name="bOutputTargetHistorical">Specifies to output the target historical data.</param>
        /// <param name="bOutputTime">Specifies to output the time matching each item in a separate blob.</param>
        /// <param name="bOutputMask">Specifies to output the mask (all 1's) matching each item in a separate blob.</param>
        /// <param name="dfPctMaxLoad">Specifies the percent of total items to load in background (default = 1, or 100%).</param>
        /// <param name="nDripRefreshRateInSec">Specifies the rate in seconds to refresh the data.</param>
        /// <param name="nChunkCount">Specifies the number of items to load on each cycle.</param>
        public RawFileData(uint? nSeed, bool bOutputTargetHistorical, bool bOutputTime, bool bOutputMask, double dfPctMaxLoad, int nDripRefreshRateInSec, uint nChunkCount) : base(nSeed, bOutputTargetHistorical, bOutputTime, bOutputMask)
        {
            m_nDropRefreshReateInSec = nDripRefreshRateInSec;
            m_dfPctMaxLoad = dfPctMaxLoad;
            m_nChunkCount = nChunkCount;
        }

        /// <summary>
        /// Verify that the data files exist.
        /// </summary>
        /// <param name="phase">Specifies the phase.</param>
        /// <param name="strPath">Specifies the file path.</param>
        /// <exception cref="Exception">An exception is thrown if the file is missing.</exception>
        public void VerifyFiles(Phase phase, string strPath)
        {
            string strFile;
            string strType = "train";
            strPath = strPath.TrimEnd('\\', '/');
            strPath += "\\";

            if (phase == Phase.TEST)
                strType = "test";
            else if (phase == Phase.RUN)
                strType = "validation";

            strFile = strPath + strType + "_sync.npy";
            if (!File.Exists(strFile))
                throw new Exception("Could not find the data file '" + strFile + "'.  You may need to run the SignalPop AI Designer Dataset Creator.");

            strFile = strPath + strType + "_schema.xml";
            if (!File.Exists(strFile))
                throw new Exception("Could not find the schema file '" + strFile + "'.  You may need to run the SignalPop AI Designer Dataset Creator.");

            return;
        }

        /// <summary>
        /// Loads all data values for the phase specified.
        /// </summary>
        /// <param name="phase">Specifies the phase to load.</param>
        /// <param name="strPath">Specifies the base path for all data.</param>
        /// <param name="bShuffleData">Specifies to randomly select from the data.</param>
        /// <param name="nBatchSize">Specifies the batch size.</param>
        /// <param name="nHistoricalSteps">Specifies the number of historical steps.</param>
        /// <param name="nFutureSteps">Specifies the number of future steps.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        public override bool LoadData(Phase phase, string strPath, bool bShuffleData, int nBatchSize, int nHistoricalSteps, int nFutureSteps, Log log, CancelEvent evtCancel)
        {
            VerifyFiles(phase, strPath);
            m_data = new DataNpy<T>(m_random, log, nHistoricalSteps, nFutureSteps, bShuffleData, m_bOutputTargetHistorical);
            return base.LoadData(phase, strPath, bShuffleData, nBatchSize, nHistoricalSteps, nFutureSteps, log, evtCancel);
        }

        /// <summary>
        /// Get the data load parameters sent to the data load function thread.
        /// </summary>
        /// <param name="phase">Specifies the phase to load.</param>
        /// <param name="strPath">Specifies the base path for all data.</param>
        /// <param name="bShuffleData">Specifies to randomly select from the data.</param>
        /// <param name="nHistoricalSteps">Specifies the number of historical steps.</param>
        /// <param name="nFutureSteps">Specifies the number of future steps.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        /// <param name="evtReady">Specifies the event set when the data is ready.</param>
        /// <param name="evtDone">Specifies the event set when the data loading is done.</param>
        /// <returns>A new DataLoadParameters parameter is returned.</returns>
        protected override DataLoadParameters getDataLoadParameters(Phase phase, string strPath, bool bShuffleData, int nHistoricalSteps, int nFutureSteps, Log log, CancelEvent evtCancel, ManualResetEvent evtReady, ManualResetEvent evtDone)
        {
            return new DataLoadParameters(phase, strPath, nHistoricalSteps, nFutureSteps, m_dfPctMaxLoad, m_nDropRefreshReateInSec, m_nChunkCount, bShuffleData, log, evtCancel, evtReady, evtDone);
        }

        /// <summary>
        /// The loadDataFunction performs a background loading of the numpy data.
        /// </summary>
        /// <param name="obj">Specifies the DataLoadParameters.</param>
        protected override void loadDataFunction(object obj)
        {
            DataLoadParameters arg = obj as DataLoadParameters;
            string strPath = arg.Path;
            Phase phase = arg.Phase;
            Log log = arg.Log;
            double dfMaxLoadPct = arg.MaxLoadPercent;
            int nDripRefreshRateInSec = arg.DripRefreshRateInSec;
            CancelEvent evtCancel = arg.CancelEvent;
            ManualResetEvent evtReady = arg.ReadyEvent;
            ManualResetEvent evtDone = arg.DoneEvent;
            DataNpy<T> dataChunk = null;
            int nIteration = 0;

            try
            {
                string strType = "train";
                strPath = strPath.TrimEnd('\\', '/');
                strPath += "\\";

                if (phase == Phase.TEST)
                    strType = "test";
                else if (phase == Phase.RUN)
                    strType = "validation";

                dataChunk = new DataNpy<T>(m_data);
                dataChunk.Open(strPath, strType, m_nBatchSize);

                int nRowIdx = 0;
                int nRowCount = dataChunk.RowCount;
                int nMaxLoadCount = (int)(nRowCount * dfMaxLoadPct);
                int nWaitCount = 0;

                Stopwatch sw = new Stopwatch();
                sw.Start();

                while (!evtCancel.WaitOne(0))
                {
                    bool bGoodData = false;

                    while (dataChunk.Load(nRowIdx, out bGoodData))
                    {
                        if (!bGoodData)
                        {
                            nRowIdx++;
                            continue;
                        }

                        bool bRefreshed = m_data.Add(dataChunk, nMaxLoadCount);

                        if (m_data.IsReady)
                            evtReady.Set();

                        nRowIdx++;

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            if (evtCancel.WaitOne(0))
                            {
                                log.WriteLine("Background data loading for '" + strType + "' aborted.");
                                break;
                            }

                            double dfPct = (double)nRowIdx / (double)nRowCount;
                            if (nMaxLoadCount > 0)
                            {
                                if (nRowIdx > nMaxLoadCount)
                                    dfPct = 1;
                                else
                                    dfPct = (double)nRowIdx / (double)nMaxLoadCount;
                            }

                            log.WriteLine("Background data loading '" + strType + "' data at " + dfPct.ToString("P") + "...");
                            sw.Restart();
                        }

                        if (bRefreshed)
                        {
                            log.WriteLine("Background data loading '" + strType + "' refreshed...");

                            // Wait roughly 5 minutes before refreshing the data;
                            nWaitCount = 0;
                            while (!evtCancel.WaitOne(1000))
                            {
                                Thread.Sleep(50);
                                nWaitCount++;

                                if (nWaitCount > nDripRefreshRateInSec)
                                    break;
                            }

                            if (nDripRefreshRateInSec == 0)
                                break;
                        }
                    }

                    if (nIteration == 0)
                        log.WriteLine("Background data load completed.");

                    if (nDripRefreshRateInSec <= 0)
                        break;

                    if (nIteration == 0)
                        log.WriteLine("Starting drip refresing...");

                    nIteration++;
                    nWaitCount = 0;
                    while (!evtCancel.WaitOne(1000))
                    {
                        Thread.Sleep(50);
                        nWaitCount++;

                        if (nWaitCount > nDripRefreshRateInSec)
                            break;
                    }

                    nRowIdx = 0;
                }
            }
            finally
            {
                dataChunk.Close();
                dataChunk.Dispose();
                evtDone.Set();
            }
        }
    }

#pragma warning disable 1591

    class DataLoadParameters /** @private */
    {
        Phase m_phase;
        string m_strPath;
        int m_nNumHistSteps;
        int m_nNumFutureSteps;
        double m_dfMaxLoadPct;
        int m_nDripRrefreshRateInSec;
        uint m_nChunkCount;
        bool m_bShuffleData;
        Log m_log;
        CancelEvent m_evtCancel;
        ManualResetEvent m_evtReady;
        ManualResetEvent m_evtDone;

        public DataLoadParameters(Phase phase, string strPath, int nNumHistSteps, int nNumFutureSteps, double dfMaxLoadPct, int nDripRefreshRateInSec, uint nChunkCount, bool bShuffleData, Log log, CancelEvent evtCancel, ManualResetEvent evtReady, ManualResetEvent evtDone)
        {
            m_phase = phase;
            m_strPath = strPath;
            m_nNumHistSteps = nNumHistSteps;
            m_nNumFutureSteps = nNumFutureSteps;
            m_dfMaxLoadPct = dfMaxLoadPct;
            m_nDripRrefreshRateInSec = nDripRefreshRateInSec;
            m_nChunkCount = nChunkCount;
            m_bShuffleData = bShuffleData;
            m_log = log;
            m_evtCancel = evtCancel;
            m_evtReady = evtReady;
            m_evtDone = evtDone;
        }

        public Phase Phase { get { return m_phase; } }
        public string Path { get { return m_strPath; } }
        public int HistoricalSteps {  get { return m_nNumHistSteps; } }
        public int FutureSteps { get { return m_nNumFutureSteps; } }
        public double MaxLoadPercent { get { return m_dfMaxLoadPct; } }
        public int DripRefreshRateInSec { get { return m_nDripRrefreshRateInSec; } }
        public uint ChunkCount { get { return m_nChunkCount; } }
        public bool ShuffleData { get { return m_bShuffleData; } }
        public Log Log { get { return m_log; } }
        public CancelEvent CancelEvent { get { return m_evtCancel; } }
        public ManualResetEvent ReadyEvent { get { return m_evtReady; } }
        public ManualResetEvent DoneEvent { get { return m_evtDone; } } 
    }

    abstract class Data<T> : IDisposable /** @private */
    {
        protected Random m_random;
        protected Log m_log;
        protected int m_nHistoricalSteps;
        protected int m_nFutureSteps;
        protected bool m_bShuffleData;
        protected bool m_bOutputTargetHistorical = false;
        protected object m_syncObj = new object();
        protected int m_nRows = 0;
        protected int m_nBatchSize = 0;
        protected int m_nTotalSize = 0;

        public enum DATA_TYPE
        {
            SYNC,
            STATIC_NUMERIC,
            STATIC_CATEGORICAL,
            OBSERVED_NUMERIC,
            OBSERVED_CATEGORICAL,
            KNOWN_NUMERIC,
            KNOWN_CATEGORICAL
        }

        public enum OUTPUT_TYPE
        {
            STATIC_NUMERIC,
            STATIC_CATEGORICAL,
            HISTORICAL_NUMERIC,
            HISTORICAL_CATEGORICAL,
            FUTURE_NUMERIC,
            FUTURE_CATEGORICAL,
            TARGET,
            HISTORICAL_SYNC,
            FUTURE_SYNC,
            HISTORICAL_TARGET
        }

        public Data(Random random, Log log, int nHistoricalSteps, int nFutureSteps, bool bShuffleData, bool bOutputTargetHistorical)
        {
            m_random = random;
            m_log = log;
            m_nHistoricalSteps = nHistoricalSteps;
            m_nFutureSteps = nFutureSteps;
            m_bShuffleData = bShuffleData;
            m_bOutputTargetHistorical = bOutputTargetHistorical;
        }

        public Data(Data<T> data)
        {
            m_random = data.m_random;
            m_log = data.m_log;
            m_nHistoricalSteps = data.m_nHistoricalSteps;
            m_nFutureSteps = data.m_nFutureSteps;
            m_bShuffleData = data.m_bShuffleData;
            m_bOutputTargetHistorical = data.m_bOutputTargetHistorical;
        }

        public void Dispose()
        {
            Close();
        }

        public int RowCount
        {
            get { return m_nRows; }
        }

        public int GetTotalSize()
        {
            return m_nTotalSize;
        }

        public bool IsReady
        {
            get { return GetTotalSize() >= m_nBatchSize; }
        }

        public abstract void Open(string strSrc, string strType, int nBatchSize);

        public abstract void Close();

        public abstract int[,] LoadBatch(int nBatchSize, BlobCollection<T> col, bool bEnableDebug, string strDebugPath);

        public abstract int[] GetShape(OUTPUT_TYPE ot);

        public abstract bool Add(DataNpy<T> data, int nMaxLoad);
    }

    class DataNpy<T> : Data<T> /** @private */
    {
        DataSchema m_schema;
        Lookup m_validRanges = new Lookup();
        Dictionary<DATA_TYPE, string> m_rgstrFiles = new Dictionary<DATA_TYPE, string>();
        Dictionary<DATA_TYPE, List<float[]>> m_rgNumData = new Dictionary<DATA_TYPE, List<float[]>>();
        Dictionary<DATA_TYPE, List<long[]>> m_rgCatData = new Dictionary<DATA_TYPE, List<long[]>>();
        Dictionary<DATA_TYPE, NumpyFile<float>> m_rgNumFiles = new Dictionary<DATA_TYPE, NumpyFile<float>>();
        Dictionary<DATA_TYPE, NumpyFile<long>> m_rgCatFiles = new Dictionary<DATA_TYPE, NumpyFile<long>>();
        Dictionary<DATA_TYPE, int> m_rgFields = new Dictionary<DATA_TYPE, int>();
        Dictionary<OUTPUT_TYPE, long[]> m_rgBatchSync = new Dictionary<OUTPUT_TYPE, long[]>();
        Dictionary<OUTPUT_TYPE, float[]> m_rgBatchBuffers = new Dictionary<OUTPUT_TYPE, float[]>();
        int m_nMaxRowIdx = -1;
        int m_nRowIdx = 0;
        int m_nColIdx = 0;
        int m_nTargetFieldIdx = 0;
        int m_nIteration = 0;

        public DataNpy(Random random, Log log, int nHistoricalSteps, int nFutureSteps, bool bShuffleData, bool bOutputTargetHistorical) 
            : base(random, log, nHistoricalSteps, nFutureSteps, bShuffleData, bOutputTargetHistorical)
        {
        }

        public DataNpy(Data<T> data) 
            : base(data)
        {
        }

        public override void Open(string strPath, string strType, int nBatchSize)
        {
            int nLen;
            m_schema = DataSchema.Load(strPath + "\\" + strType + "_schema.xml");
            m_nTargetFieldIdx = m_schema.Data.ObservedNum.FindFieldIndex(Field.INPUT_TYPE.TARGET);

            m_nIteration = 0;

            m_nBatchSize = nBatchSize;
            m_rgstrFiles.Add(DATA_TYPE.SYNC, strPath + "\\" + strType + "_sync.npy");
            m_rgstrFiles.Add(DATA_TYPE.STATIC_NUMERIC, strPath + "\\" + strType + "_static_num.npy");
            m_rgstrFiles.Add(DATA_TYPE.STATIC_CATEGORICAL, strPath + "\\" + strType + "_static_cat.npy");
            m_rgstrFiles.Add(DATA_TYPE.OBSERVED_NUMERIC, strPath + "\\" + strType + "_observed_num.npy");
            m_rgstrFiles.Add(DATA_TYPE.OBSERVED_CATEGORICAL, strPath + "\\" + strType + "_observed_cat.npy");
            m_rgstrFiles.Add(DATA_TYPE.KNOWN_NUMERIC, strPath + "\\" + strType + "_known_num.npy");
            m_rgstrFiles.Add(DATA_TYPE.KNOWN_CATEGORICAL, strPath + "\\" + strType + "_known_cat.npy");

            // Verify the required files.
            if (!File.Exists(m_rgstrFiles[DATA_TYPE.SYNC]))
                throw new Exception("Could not find the sync file '" + m_rgstrFiles[DATA_TYPE.SYNC] + "'.");

            NumpyFile<long> npySync = new NumpyFile<long>(null);
            npySync.OpenRead(m_rgstrFiles[DATA_TYPE.SYNC]);
            m_rgCatFiles.Add(DATA_TYPE.SYNC, npySync);
            m_rgCatData.Add(DATA_TYPE.SYNC, new List<long[]>());
            m_rgFields.Add(DATA_TYPE.SYNC, npySync.Fields);

            nLen = nBatchSize * m_nHistoricalSteps * m_rgCatFiles[DATA_TYPE.SYNC].Fields;
            m_rgBatchSync.Add(OUTPUT_TYPE.HISTORICAL_SYNC, new long[nLen]);

            nLen = nBatchSize * m_nFutureSteps * m_rgCatFiles[DATA_TYPE.SYNC].Fields;
            m_rgBatchSync.Add(OUTPUT_TYPE.FUTURE_SYNC, new long[nLen]);

            if (!File.Exists(m_rgstrFiles[DATA_TYPE.OBSERVED_NUMERIC]))
                throw new Exception("Could not find the sync file '" + m_rgstrFiles[DATA_TYPE.OBSERVED_NUMERIC] + "'.");

            NumpyFile<float> npyObsNum = new NumpyFile<float>(null);
            npyObsNum.OpenRead(m_rgstrFiles[DATA_TYPE.OBSERVED_NUMERIC]);
            m_rgNumFiles.Add(DATA_TYPE.OBSERVED_NUMERIC, npyObsNum);
            m_rgNumData.Add(DATA_TYPE.OBSERVED_NUMERIC, new List<float[]>());
            m_rgFields.Add(DATA_TYPE.OBSERVED_NUMERIC, npyObsNum.Fields);
            m_nRows = npyObsNum.Rows;

            int nNumObsFields = m_schema.Data.ObservedNumExplicitCount;
            if (nNumObsFields != m_rgNumFiles[DATA_TYPE.OBSERVED_NUMERIC].Fields && nNumObsFields != m_rgNumFiles[DATA_TYPE.OBSERVED_NUMERIC].Fields - 1)
                throw new Exception("The number of observed numeric fields in the schema does not match the number of fields in the observed numeric data file.");

            nLen = nBatchSize * m_nHistoricalSteps * nNumObsFields;
            m_rgBatchBuffers.Add(OUTPUT_TYPE.HISTORICAL_NUMERIC, new float[nLen]);
            // The future observed are the target values.
            nLen = nBatchSize * m_nFutureSteps * 1;
            m_rgBatchBuffers.Add(OUTPUT_TYPE.TARGET, new float[nLen]);

            if (m_bOutputTargetHistorical)
            {
                // The past observed are the target values historical.
                nLen = nBatchSize * m_nHistoricalSteps * 1;
                m_rgBatchBuffers.Add(OUTPUT_TYPE.HISTORICAL_TARGET, new float[nLen]);
            }

            if (File.Exists(m_rgstrFiles[DATA_TYPE.OBSERVED_CATEGORICAL]))
            {
                NumpyFile<long> npyObsCat = new NumpyFile<long>(null);
                npyObsCat.OpenRead(m_rgstrFiles[DATA_TYPE.OBSERVED_CATEGORICAL]);
                m_rgCatFiles.Add(DATA_TYPE.OBSERVED_CATEGORICAL, npyObsCat);
                m_rgCatData.Add(DATA_TYPE.OBSERVED_CATEGORICAL, new List<long[]>());
                m_rgFields.Add(DATA_TYPE.OBSERVED_CATEGORICAL, npyObsCat.Fields);

                nLen = nBatchSize * m_nHistoricalSteps * m_rgNumFiles[DATA_TYPE.OBSERVED_CATEGORICAL].Fields;
                m_rgBatchBuffers.Add(OUTPUT_TYPE.HISTORICAL_CATEGORICAL, new float[nLen]);
            }

            if (File.Exists(m_rgstrFiles[DATA_TYPE.KNOWN_NUMERIC]))
            {
                NumpyFile<float> npyKnownNum = new NumpyFile<float>(null);
                npyKnownNum.OpenRead(m_rgstrFiles[DATA_TYPE.KNOWN_NUMERIC]);
                m_rgNumFiles.Add(DATA_TYPE.KNOWN_NUMERIC, npyKnownNum);
                m_rgNumData.Add(DATA_TYPE.KNOWN_NUMERIC, new List<float[]>());
                m_rgFields.Add(DATA_TYPE.KNOWN_NUMERIC, npyKnownNum.Fields);

                // Observed numeric and known numeric are combined into a single buffer.
                nLen = nBatchSize * m_nHistoricalSteps * (m_rgNumFiles[DATA_TYPE.OBSERVED_NUMERIC].Fields + m_rgNumFiles[DATA_TYPE.KNOWN_NUMERIC].Fields);
                m_rgBatchBuffers[OUTPUT_TYPE.HISTORICAL_NUMERIC] = new float[nLen];

                nLen = nBatchSize * m_nFutureSteps * m_rgNumFiles[DATA_TYPE.KNOWN_NUMERIC].Fields;
                m_rgBatchBuffers.Add(OUTPUT_TYPE.FUTURE_NUMERIC, new float[nLen]);
            }

            if (File.Exists(m_rgstrFiles[DATA_TYPE.KNOWN_CATEGORICAL]))
            {
                NumpyFile<long> npyKnownCat = new NumpyFile<long>(null);
                npyKnownCat.OpenRead(m_rgstrFiles[DATA_TYPE.KNOWN_CATEGORICAL]);
                m_rgCatFiles.Add(DATA_TYPE.KNOWN_CATEGORICAL, npyKnownCat);
                m_rgCatData.Add(DATA_TYPE.KNOWN_CATEGORICAL, new List<long[]>());
                m_rgFields.Add(DATA_TYPE.KNOWN_CATEGORICAL, npyKnownCat.Fields);

                nLen = nBatchSize * m_nHistoricalSteps * m_rgCatFiles[DATA_TYPE.KNOWN_CATEGORICAL].Fields;
                m_rgBatchBuffers.Add(OUTPUT_TYPE.HISTORICAL_CATEGORICAL, new float[nLen]);
                nLen = nBatchSize * m_nFutureSteps * m_rgCatFiles[DATA_TYPE.KNOWN_CATEGORICAL].Fields;
                m_rgBatchBuffers.Add(OUTPUT_TYPE.FUTURE_CATEGORICAL, new float[nLen]);
            }

            if (File.Exists(m_rgstrFiles[DATA_TYPE.STATIC_NUMERIC]))
            {
                NumpyFile<float> npyStatNum = new NumpyFile<float>(null);
                npyStatNum.OpenRead(m_rgstrFiles[DATA_TYPE.STATIC_NUMERIC]);
                m_rgNumFiles.Add(DATA_TYPE.STATIC_NUMERIC, npyStatNum);
                m_rgNumData.Add(DATA_TYPE.STATIC_NUMERIC, new List<float[]>());
                m_rgFields.Add(DATA_TYPE.STATIC_NUMERIC, npyStatNum.Fields);

                nLen = nBatchSize * m_rgNumFiles[DATA_TYPE.STATIC_NUMERIC].Fields;
                m_rgBatchBuffers.Add(OUTPUT_TYPE.STATIC_NUMERIC, new float[nLen]);
            }

            if (File.Exists(m_rgstrFiles[DATA_TYPE.STATIC_CATEGORICAL]))
            {
                NumpyFile<long> npyStatCat = new NumpyFile<long>(null);
                npyStatCat.OpenRead(m_rgstrFiles[DATA_TYPE.STATIC_CATEGORICAL]);
                m_rgCatFiles.Add(DATA_TYPE.STATIC_CATEGORICAL, npyStatCat);
                m_rgCatData.Add(DATA_TYPE.STATIC_CATEGORICAL, new List<long[]>());
                m_rgFields.Add(DATA_TYPE.STATIC_CATEGORICAL, npyStatCat.Fields);

                nLen = nBatchSize * m_rgCatFiles[DATA_TYPE.STATIC_CATEGORICAL].Fields;
                m_rgBatchBuffers.Add(OUTPUT_TYPE.STATIC_CATEGORICAL, new float[nLen]);
            }
        }

        public override void Close()
        {
            foreach (KeyValuePair<DATA_TYPE, NumpyFile<long>> kvp in m_rgCatFiles)
            {
                kvp.Value.Close();
            }

            foreach (KeyValuePair<DATA_TYPE, NumpyFile<float>> kvp in m_rgNumFiles)
            {
                kvp.Value.Close();
            }

            m_rgCatFiles.Clear();
            m_rgNumFiles.Clear();
            m_rgCatData.Clear();
            m_rgNumData.Clear();
            m_rgBatchBuffers.Clear();
            m_rgBatchSync.Clear();
            m_rgFields.Clear();
        }

        private int getMaxRowIdx(int nBatchSize)
        {
            int nFields = m_rgFields[DATA_TYPE.SYNC];
            int nCount = nBatchSize;

            for (int i=m_rgCatData[DATA_TYPE.SYNC].Count-1; i>=0; i--)
            {
                nCount -= m_rgCatData[DATA_TYPE.SYNC][i].Length / nFields;
                if (nCount <= 0)
                    return i;
            }   

            return -1;
        }

        public bool Load(int nRowIdx, out bool bGoodData)
        {
            bGoodData = false;

            if (nRowIdx >= m_nRows)
                return false;

            int nStartIdx = m_schema.Lookups[0][nRowIdx].ValidRangeStartIndex;
            int nEndIdx = m_schema.Lookups[0][nRowIdx].ValidRangeEndIndex;
            int nFields = m_rgFields[DATA_TYPE.SYNC];
            if (nStartIdx < 0 || nEndIdx < 0 || (nEndIdx - nStartIdx) < (m_nHistoricalSteps + m_nFutureSteps))
                return true;
            
            Dictionary<DATA_TYPE, long[]> cat = new Dictionary<DATA_TYPE, long[]>();
            foreach (KeyValuePair<DATA_TYPE, NumpyFile<long>> kvp in m_rgCatFiles)
            {
                int nStartIdx1 = (kvp.Key == DATA_TYPE.STATIC_CATEGORICAL) ? 0 : nStartIdx;
                int nEndIdx1 = (kvp.Key == DATA_TYPE.STATIC_CATEGORICAL) ? 0 : nEndIdx;
                long[] rgBuffer = null;
                rgBuffer = kvp.Value.LoadRow(rgBuffer, nRowIdx, nStartIdx1, (nEndIdx1 - nStartIdx1) + 1);
                cat.Add(kvp.Key, rgBuffer);
                if (rgBuffer == null)
                    return true;
            }

            Dictionary<DATA_TYPE, float[]> num = new Dictionary<DATA_TYPE, float[]>();
            foreach (KeyValuePair<DATA_TYPE, NumpyFile<float>> kvp in m_rgNumFiles)
            {
                int nStartIdx1 = (kvp.Key == DATA_TYPE.STATIC_NUMERIC) ? 0 : nStartIdx;
                int nEndIdx1 = (kvp.Key == DATA_TYPE.STATIC_NUMERIC) ? 0 : nEndIdx;
                float[] rgBuffer = null;
                rgBuffer = kvp.Value.LoadRow(rgBuffer, nRowIdx, nStartIdx1, (nEndIdx1 - nStartIdx1) + 1);
                num.Add(kvp.Key, rgBuffer);
                if (rgBuffer == null)
                    return true;
            }

            foreach (KeyValuePair<DATA_TYPE, long[]> kvp in cat)
            {
                m_rgCatData[kvp.Key].Add(kvp.Value);
            }

            foreach (KeyValuePair<DATA_TYPE, float[]> kvp in num)
            {
                m_rgNumData[kvp.Key].Add(kvp.Value);
            }

            m_validRanges.Add(m_schema.Lookups[0][nRowIdx]);

            bGoodData = true;

            return true;
        }

        public override bool Add(DataNpy<T> data, int nMaxLoad)
        {
            bool bRefreshed = false;

            lock (m_syncObj)
            {
                foreach (KeyValuePair<DATA_TYPE, List<float[]>> kv in data.m_rgNumData)
                {
                    if (!m_rgNumData.ContainsKey(kv.Key))
                        m_rgNumData.Add(kv.Key, new List<float[]>());

                    m_rgNumData[kv.Key].AddRange(kv.Value);
                    data.m_rgNumData[kv.Key].Clear();

                    while (m_rgNumData[kv.Key].Count > nMaxLoad)
                    {
                        m_rgNumData[kv.Key].RemoveAt(0);
                        bRefreshed = true;
                    }
                }

                foreach (KeyValuePair<DATA_TYPE, List<long[]>> kv in data.m_rgCatData)
                {
                    if (!m_rgCatData.ContainsKey(kv.Key))
                        m_rgCatData.Add(kv.Key, new List<long[]>());

                    m_rgCatData[kv.Key].AddRange(kv.Value);
                    data.m_rgCatData[kv.Key].Clear();

                    while (m_rgCatData[kv.Key].Count > nMaxLoad)
                    {
                        m_rgCatData[kv.Key].RemoveAt(0);
                    }
                }

                foreach (KeyValuePair<DATA_TYPE, int> kv in data.m_rgFields)
                {
                    if (!m_rgFields.ContainsKey(kv.Key))
                        m_rgFields.Add(kv.Key, kv.Value);
                }

                foreach (KeyValuePair<OUTPUT_TYPE, long[]> kv in data.m_rgBatchSync)
                {
                    m_rgBatchSync.Add(kv.Key, kv.Value);
                }
                data.m_rgBatchSync.Clear();

                foreach (KeyValuePair<OUTPUT_TYPE, float[]> kv in data.m_rgBatchBuffers)
                {
                    m_rgBatchBuffers.Add(kv.Key, kv.Value);
                }
                data.m_rgBatchBuffers.Clear();

                m_validRanges.Add(data.m_validRanges);
                data.m_validRanges.Clear();

                m_schema = data.m_schema;
                m_nBatchSize = data.m_nBatchSize;
                m_nMaxRowIdx = getMaxRowIdx(m_nBatchSize);
                m_nRows = m_rgCatData[DATA_TYPE.SYNC].Count;
                m_nTargetFieldIdx = data.m_nTargetFieldIdx;
                int nFields = m_rgFields[DATA_TYPE.SYNC];
                m_nTotalSize = m_rgCatData[DATA_TYPE.SYNC].Sum(p => p.Length) / (m_nHistoricalSteps + m_nFutureSteps) * nFields;
            }

            return bRefreshed;
        }

        public override int[] GetShape(OUTPUT_TYPE ot)
        {
            int nFields = 0;

            switch (ot)
            {
                case OUTPUT_TYPE.STATIC_NUMERIC:
                    if (m_rgFields.ContainsKey(DATA_TYPE.STATIC_NUMERIC))
                        return new int[] { m_nBatchSize, m_rgFields[DATA_TYPE.STATIC_NUMERIC] };
                    break;

                case OUTPUT_TYPE.STATIC_CATEGORICAL:
                    if (m_rgFields.ContainsKey(DATA_TYPE.STATIC_CATEGORICAL))
                        return new int[] { m_nBatchSize, m_rgFields[DATA_TYPE.STATIC_CATEGORICAL] };
                    break;

                case OUTPUT_TYPE.HISTORICAL_NUMERIC:
                    nFields = 0;
                    if (m_rgFields.ContainsKey(DATA_TYPE.OBSERVED_NUMERIC))
                        nFields += m_schema.Data.ObservedNumExplicitCount;
                    if (m_rgFields.ContainsKey(DATA_TYPE.KNOWN_NUMERIC))
                        nFields += m_rgFields[DATA_TYPE.KNOWN_NUMERIC];
                    if (nFields > 0)
                        return new int[] { m_nBatchSize, m_nHistoricalSteps, nFields };
                    break;

                case OUTPUT_TYPE.HISTORICAL_CATEGORICAL:
                    nFields = 0;
                    if (m_rgFields.ContainsKey(DATA_TYPE.OBSERVED_CATEGORICAL))
                        nFields += m_rgFields[DATA_TYPE.OBSERVED_CATEGORICAL];
                    if (m_rgFields.ContainsKey(DATA_TYPE.KNOWN_CATEGORICAL))
                        nFields += m_rgFields[DATA_TYPE.KNOWN_CATEGORICAL];
                    if (nFields > 0)
                        return new int[] { m_nBatchSize, m_nHistoricalSteps, nFields };
                    break;

                case OUTPUT_TYPE.FUTURE_NUMERIC:
                    if (m_rgFields.ContainsKey(DATA_TYPE.KNOWN_NUMERIC))
                        return new int[] { m_nBatchSize, m_nFutureSteps, m_rgFields[DATA_TYPE.KNOWN_NUMERIC] };
                    break;

                case OUTPUT_TYPE.FUTURE_CATEGORICAL:
                    if (m_rgFields.ContainsKey(DATA_TYPE.KNOWN_CATEGORICAL))
                        return new int[] { m_nBatchSize, m_nFutureSteps, m_rgFields[DATA_TYPE.KNOWN_CATEGORICAL] };
                    break;

                case OUTPUT_TYPE.TARGET:
                    return new int[] { m_nBatchSize, m_nFutureSteps, 1 };

                case OUTPUT_TYPE.HISTORICAL_TARGET:
                    return new int[] { m_nBatchSize, m_nHistoricalSteps, 1 };

                default:
                    throw new Exception("Unknown output type '" + ot.ToString() + "'!");
            }

            return null;
        }

        private void stepNext()
        {
            if (m_bShuffleData)
            {
                m_nRowIdx = m_random.Next(m_validRanges.Count);

                int nValidRangeCount = m_validRanges[m_nRowIdx].ValidRangeCount;
                int nRetry = 0;
                while (nRetry < 5 && nValidRangeCount < (m_nHistoricalSteps + m_nFutureSteps))
                {
                    m_nRowIdx = m_random.Next(m_validRanges.Count);
                    nValidRangeCount = m_validRanges[m_nRowIdx].ValidRangeCount;
                    nRetry++;
                }

                if (nRetry == 5)
                    throw new Exception("Could not find a row with more than " + (m_nHistoricalSteps + m_nFutureSteps).ToString() + " valid ranges!");

                m_nColIdx = m_random.Next(nValidRangeCount - (m_nHistoricalSteps + m_nFutureSteps));
            }
            else
            {
                m_nColIdx++;
                int nValidRangeCount = m_validRanges[m_nRowIdx].ValidRangeCount;
                if (m_nColIdx + m_nHistoricalSteps + m_nFutureSteps > nValidRangeCount)
                {
                    m_nRowIdx++;
                    if (m_nRowIdx >= m_nMaxRowIdx)
                        m_nRowIdx = 0;

                    m_nColIdx = 0;
                }
            }
        }

        private float[] getBatch(OUTPUT_TYPE ot)
        {
            if (!m_rgBatchBuffers.ContainsKey(ot))
                return null;

            return m_rgBatchBuffers[ot];
        }

        private bool loadSyncBatch(int nIdx, long[] rg, int nStartIdx, int nCount)
        {
            if (rg == null)
                return false;

            int nStartIdx1 = m_nColIdx + nStartIdx;
            int nFields = m_rgFields[DATA_TYPE.SYNC];
            long[] rgSrc = m_rgCatData[DATA_TYPE.SYNC][m_nRowIdx];

            if (nStartIdx1 * nFields + nCount * nFields > rgSrc.Length)
                return false;

            Array.Copy(rgSrc, nStartIdx1 * nFields, rg, nIdx * nCount * nFields, nCount * nFields);

            return true;
        }

        private void loadStaticCatBatch(int nIdx, float[] rg, DATA_TYPE dt)
        {
            if (rg == null)
                return;

            int nFields = m_rgFields[dt];
            long[] rgSrc = m_rgCatData[dt][m_nRowIdx];

            Array.Copy(rgSrc, 0, rg, nIdx * nFields, nFields);
        }

        private void loadStaticNumBatch(int nIdx, float[] rg, DATA_TYPE dt)
        {
            if (rg == null)
                return;

            int nFields = m_rgFields[dt];
            float[] rgSrc = m_rgNumData[dt][m_nRowIdx];

            Array.Copy(rgSrc, 0, rg, nIdx * nFields, nFields);
        }

        private void loadCatBatch(int nIdx, float[] rg, int nStartIdx, int nCount, DATA_TYPE dt)
        {
            if (rg == null)
                return;

            int nStartIdx1 = m_nColIdx + nStartIdx;
            int nFields = m_rgFields[dt];
            long[] rgSrc = m_rgCatData[dt][m_nRowIdx];
            Array.Copy(rgSrc, nStartIdx1 * nFields, rg, nIdx * nCount * nFields, nCount * nFields);
        }

        private void loadCatBatch(int nIdx, float[] rg, int nStartIdx, int nCount, DATA_TYPE dt1, DATA_TYPE dt2)
        {
            if (rg == null)
                return;

            int nStartIdx1 = m_nColIdx + nStartIdx;
            int nFields1 = (m_rgFields.ContainsKey(dt1)) ? m_rgFields[dt1] : 0;
            long[] rgSrc1 = (m_rgFields.ContainsKey(dt1)) ? m_rgCatData[dt1][m_nRowIdx] : null;
            int nFields2 = (m_rgFields.ContainsKey(dt2)) ? m_rgFields[dt2] : 0;
            long[] rgSrc2 = (m_rgFields.ContainsKey(dt2)) ? m_rgCatData[dt2][m_nRowIdx] : null;
            int nFields = nFields1 + nFields2;

            for (int j = nStartIdx1; j < nStartIdx1 + nCount; j++)
            {
                for (int k = 0; k < nFields1; k++)
                {
                    int nSrcIdx = j * nFields1 + k;
                    int nDstIdx = nIdx * nCount * nFields + (j - nStartIdx1) * nFields + k;
                    rg[nDstIdx] = rgSrc1[nSrcIdx];
                }
                for (int k = 0; k < nFields2; k++)
                {
                    int nSrcIdx = j * nFields2 + k;
                    int nDstIdx = nIdx * nCount * nFields + (j - nStartIdx1) * nFields + k + nFields1;
                    rg[nDstIdx] = rgSrc2[nSrcIdx];
                }
            }
        }

        private void loadNumBatch(int nIdx, float[] rg, int nStartIdx, int nCount, DATA_TYPE dt)
        {
            if (rg == null)
                return;

            int nStartIdx1 = m_nColIdx + nStartIdx;
            int nFields = m_rgFields[dt];
            float[] rgSrc = m_rgNumData[dt][m_nRowIdx];           
            Array.Copy(rgSrc, nStartIdx1 * nFields, rg, nIdx * nCount * nFields, nCount * nFields);
        }

        private void loadNumBatch(int nIdx, float[] rg, int nStartIdx, int nCount, int nFieldIdx, DATA_TYPE dt)
        {
            if (rg == null)
                return;

            int nStartIdx1 = m_nColIdx + nStartIdx;
            int nFields = m_rgFields[dt];
            float[] rgSrc = m_rgNumData[dt][m_nRowIdx];

            for (int i = 0; i < nCount; i++)
            {
                int nSrcIdx = nStartIdx1 * nFields + i * nFields + nFieldIdx;
                int nDstIdx = nIdx * nCount + i;

                rg[nDstIdx] = rgSrc[nSrcIdx];
            }
        }

        private int getNumFields(DATA_TYPE dt)
        {
            if (dt != DATA_TYPE.OBSERVED_NUMERIC)
                return m_rgFields[dt];

            return m_schema.Data.ObservedNumExplicitCount;
        }

        private void loadNumBatch(int nIdx, float[] rg, int nStartIdx, int nCount, DATA_TYPE dt1, DATA_TYPE dt2)
        {
            if (rg == null)
                return;

            int nStartIdx1 = m_nColIdx + nStartIdx;
            int nFields1Explicit = m_rgFields.ContainsKey(dt1) ? getNumFields(dt1) : 0;
            int nFields1 = (m_rgFields.ContainsKey(dt1)) ? m_rgFields[dt1] : 0;
            float[] rgSrc1 = (m_rgFields.ContainsKey(dt1)) ? m_rgNumData[dt1][m_nRowIdx] : null;
            int nFields2 = (m_rgFields.ContainsKey(dt2)) ? m_rgFields[dt2] : 0;
            float[] rgSrc2 = (m_rgFields.ContainsKey(dt2)) ? m_rgNumData[dt2][m_nRowIdx] : null;
            int nFields = nFields1Explicit + nFields2;

            for (int j = nStartIdx1; j < nStartIdx1 + nCount; j++)
            {
                int nDstIdx = nIdx * nCount * nFields + (j - nStartIdx1) * nFields;
                int nDstIdx1 = nDstIdx;

                for (int k = 0; k < nFields1; k++)
                {
                    int nSrcIdx = j * nFields1 + k;

                    if (m_schema.Data.IsObservedNum(k))
                    {
                        rg[nDstIdx1] = rgSrc1[nSrcIdx];
                        nDstIdx1++;
                    }
                }

                for (int k = 0; k < nFields2; k++)
                {
                    int nSrcIdx = j * nFields2 + k;
                    nDstIdx = nIdx * nCount * nFields + (j - nStartIdx1) * nFields + nFields1Explicit + k;
                    rg[nDstIdx] = rgSrc2[nSrcIdx];
                }
            }
        }

        public override int[,] LoadBatch(int nBatchSize, BlobCollection<T> col, bool bEnableDebug, string strDebugPath)
        {
            lock (m_syncObj)
            {
                long[] rgHistSync = m_rgBatchSync[OUTPUT_TYPE.HISTORICAL_SYNC];
                long[] rgFutSync = m_rgBatchSync[OUTPUT_TYPE.FUTURE_SYNC];
                float[] rgStatCat = getBatch(OUTPUT_TYPE.STATIC_CATEGORICAL);
                float[] rgStatNum = getBatch(OUTPUT_TYPE.STATIC_NUMERIC);
                float[] rgHistCat = getBatch(OUTPUT_TYPE.HISTORICAL_CATEGORICAL);
                float[] rgHistNum = getBatch(OUTPUT_TYPE.HISTORICAL_NUMERIC);
                float[] rgFutCat = getBatch(OUTPUT_TYPE.FUTURE_CATEGORICAL);
                float[] rgFutNum = getBatch(OUTPUT_TYPE.FUTURE_NUMERIC);
                float[] rgTarget = getBatch(OUTPUT_TYPE.TARGET);
                float[] rgHistTarget = getBatch(OUTPUT_TYPE.HISTORICAL_TARGET);

                for (int i = 0; i < nBatchSize; i++)
                {
                    if (loadSyncBatch(i, rgHistSync, 0, m_nHistoricalSteps) &&
                        loadSyncBatch(i, rgFutSync, m_nHistoricalSteps, m_nFutureSteps))
                    {
                        loadStaticCatBatch(i, rgStatCat, DATA_TYPE.STATIC_CATEGORICAL);
                        loadStaticNumBatch(i, rgStatNum, DATA_TYPE.STATIC_NUMERIC);

                        loadCatBatch(i, rgHistCat, 0, m_nHistoricalSteps, DATA_TYPE.OBSERVED_CATEGORICAL, DATA_TYPE.KNOWN_CATEGORICAL);
                        loadNumBatch(i, rgHistNum, 0, m_nHistoricalSteps, DATA_TYPE.OBSERVED_NUMERIC, DATA_TYPE.KNOWN_NUMERIC);

                        loadCatBatch(i, rgFutCat, m_nHistoricalSteps, m_nFutureSteps, DATA_TYPE.KNOWN_CATEGORICAL);
                        loadNumBatch(i, rgFutNum, m_nHistoricalSteps, m_nFutureSteps, DATA_TYPE.KNOWN_NUMERIC);

                        loadNumBatch(i, rgHistTarget, 0, m_nHistoricalSteps, m_nTargetFieldIdx, DATA_TYPE.OBSERVED_NUMERIC);
                        loadNumBatch(i, rgTarget, m_nHistoricalSteps, m_nFutureSteps, m_nTargetFieldIdx, DATA_TYPE.OBSERVED_NUMERIC);
                    }

                    stepNext();
                }

                if (rgStatNum != null)
                    col[0].mutable_cpu_data = Utility.ConvertVec<T>(rgStatNum);

                if (rgStatCat != null)
                    col[1].mutable_cpu_data = Utility.ConvertVec<T>(rgStatCat);

                if (rgHistNum != null)
                    col[2].mutable_cpu_data = Utility.ConvertVec<T>(rgHistNum);

                if (rgHistCat != null)
                    col[3].mutable_cpu_data = Utility.ConvertVec<T>(rgHistCat);

                if (rgFutNum != null)
                    col[4].mutable_cpu_data = Utility.ConvertVec<T>(rgFutNum);

                if (rgFutCat != null)
                    col[5].mutable_cpu_data = Utility.ConvertVec<T>(rgFutCat);

                if (rgTarget != null)
                    col[6].mutable_cpu_data = Utility.ConvertVec<T>(rgTarget);

                if (rgHistTarget != null)
                    col[7].mutable_cpu_data = Utility.ConvertVec<T>(rgHistTarget);

                if (bEnableDebug)
                {
                    if (Directory.Exists(strDebugPath))
                    {
                        //debug(strDebugPath, col[0].shape(), rgStatNum, "stat_num");
                        //debug(strDebugPath, col[1].shape(), rgStatCat, "stat_cat");
                        //debug(strDebugPath, col[2].shape(), rgHistNum, "hist_num");
                        //debug(strDebugPath, col[3].shape(), rgHistCat, "hist_cat");
                        //debug(strDebugPath, col[4].shape(), rgFutNum, "fut_num");
                        //debug(strDebugPath, col[5].shape(), rgFutCat, "fut_cat");
                        //debug(strDebugPath, col[6].shape(), rgTarget, "target");

                        //if (col.Count > 7)
                        //    debug(strDebugPath, col[7].shape(), rgHistTarget, "hist_target");
                    }
                }

                m_nIteration++;

                return null;
            }
        }

        //private void debug(string strPath, List<int> rgShape, float[] rgData, string strName)
        //{
        //    if (rgData == null || rgData.Length == 0 || rgShape.Count <= 2)
        //        return;

        //    string strFile = strPath + "\\" + m_nIteration.ToString() + "_" + m_nRowIdx.ToString() + "_" + m_nColIdx.ToString() + "_" + strName;

        //    int nBatch = rgShape[0];
        //    int nSeq = rgShape[1];
        //    int nFields = rgShape[2];

        //    for (int i = 0; i < nFields; i++)
        //    {
        //        string strFile1 = strFile + "_field_" + i.ToString() + ".png";
        //        PlotCollectionSet set = new PlotCollectionSet();

        //        for (int j = 0; j < nBatch; j++)
        //        {
        //            PlotCollection plots = new PlotCollection(strName + "_batch_" + j.ToString());
        //            for (int k = 0; k < nSeq; k++)
        //            {
        //                int nIdx = j * nSeq * nFields + k * nFields + i;
        //                Plot plot = new Plot(k, rgData[nIdx]);
        //                plots.Add(plot);
        //            }

        //            set.Add(plots);
        //        }

        //        Image img = SimpleGraphingControl.QuickRender(set);
        //        img.Save(strFile1);
        //    }
        //}
    }

    class BatchPerfSet /** @private */
    {
        BatchPerf[] m_rgBatchPerf = new BatchPerf[2];
        int m_nSelectIdx = 0;
        int m_nLoadIdx = 0;
        int m_nSelectFrequency = 1;
        int m_nSelectCount = 0;
        Random m_random;
        double m_dfPctTopSelectionPct = 0.25;
        bool m_bActive = false;

        public BatchPerfSet(Random rand, double dfPctTopSelectionPct, int nMax, int nSelectFrequency)
        {
            m_rgBatchPerf[0] = new BatchPerf(nMax, dfPctTopSelectionPct);
            m_rgBatchPerf[1] = new BatchPerf(nMax, dfPctTopSelectionPct);
            m_nSelectFrequency = nSelectFrequency;
            m_dfPctTopSelectionPct = dfPctTopSelectionPct;
            m_random = rand;
        }

        public bool Add(LossArgs e, int[,] rg)
        {
            if (m_rgBatchPerf[m_nLoadIdx].Add(e, rg))
            {
                if (m_nLoadIdx == 0)
                {
                    m_rgBatchPerf[1].Clear();
                    m_nSelectIdx = 0;
                    m_nLoadIdx = 1;
                }
                else
                {
                    m_rgBatchPerf[0].Clear();
                    m_nLoadIdx = 0;
                    m_nSelectIdx = 1;
                }

                m_bActive = true;
            }
            else
            {
                m_bActive = false;
            }

            return m_bActive;
        }

        public bool IsActive
        {
            get { return m_bActive; }
        }

        public bool Select(ref int? nIdx1, ref int? nIdx2)
        {
            m_nSelectCount++;
            if (m_nSelectCount % m_nSelectFrequency == 0)
                return m_rgBatchPerf[m_nSelectIdx].Select(m_random, m_dfPctTopSelectionPct, ref nIdx1, ref nIdx2);

            return false;
        }
    }

    class BatchPerf /** @private */
    {
        int m_nMax;
        int m_nLastSortCount;
        double m_dfTopSelectionPct;
        List<Tuple<float, int, int>> m_rgPerformanceItems = new List<Tuple<float, int, int>>();

        public BatchPerf(int nMax, double dfPctTopSelectionPct)
        {
            m_rgPerformanceItems = new List<Tuple<float, int, int>>(nMax + 1);
            m_dfTopSelectionPct = dfPctTopSelectionPct;
            m_nMax = nMax;
            m_nLastSortCount = (int)(nMax * dfPctTopSelectionPct);
        }

        public bool Add(LossArgs e, int[,] rg)
        {
            bool bAtMax = false;

            for (int i = 0; i < e.Data.Length; i++)
            {
                if (rg[i,0] == -1 || rg[i, 1] == -1)
                    continue;

                m_rgPerformanceItems.Add(new Tuple<float, int, int>(e.Data[i], rg[i,0], rg[i,1]));
                m_nLastSortCount--;

                if (m_rgPerformanceItems.Count > m_nMax)
                {
                    m_rgPerformanceItems.RemoveAt(0);
                    bAtMax = true;
                }
            }

            return bAtMax;
        }

        public void Clear()
        {
            m_rgPerformanceItems.Clear();   
        }

        public void Sort()
        {
            m_rgPerformanceItems = m_rgPerformanceItems.OrderByDescending(p => p.Item1).ToList();
            m_nLastSortCount = (int)(m_nMax * m_dfTopSelectionPct);
        }

        public bool Select(Random rand, double dfPct, ref int? nIdx1, ref int? nIdx2)
        {
            if (m_rgPerformanceItems.Count < m_nMax)
                return false;

            if (m_nLastSortCount <= 0)
                Sort();

            int nCount = (int)(m_rgPerformanceItems.Count * dfPct);
            int nIdx = rand.Next(nCount);

            nIdx1 = m_rgPerformanceItems[nIdx].Item2;
            nIdx2 = m_rgPerformanceItems[nIdx].Item3;

            return true;
        }
    }
#pragma warning restore 1591
}
