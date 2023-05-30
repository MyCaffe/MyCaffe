using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Policy;
using System.Text;
using System.Threading;
using System.Xml.Linq;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.tft;

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
        List<int> m_rgShape = new List<int>(4);
        uint m_nBatchSize;
        uint m_nNumHistoricalSteps;
        uint m_nNumFutureSteps;
        RawFileData<T> m_data = null;
        CancelEvent m_evtCancel;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type DATA_TEMPORAL with parameter data_temporal_param</param>
        /// <param name="evtCancel">Specifies the cancel event used to cancel background data loading.</param>
        public DataTemporalLayer(CudaDnn<T> cuda, Log log, LayerParameter p, CancelEvent evtCancel)
            : base(cuda, log, p)
        {
            m_evtCancel = evtCancel;
            m_type = LayerParameter.LayerType.DATA_TEMPORAL;
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
        /// Returns the exact number of required top (output) Blobs: static_numeric, static_categorical, hist_numeric, hist_categorical, future_numeric, future_categorical, target
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 7; }
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
                m_data = new RawFileData<T>(m_param.data_temporal_param.seed);

            Phase phase = m_phase;
            if (m_param.data_temporal_param.forced_phase.HasValue)
            {
                m_log.WriteLine("INFO: Using forced phase = " + m_param.data_temporal_param.forced_phase.Value.ToString() + ".");
                phase = m_param.data_temporal_param.forced_phase.Value;
            }

            if (!m_data.LoadData(phase, m_param.data_temporal_param.source, m_param.data_temporal_param.shuffle_data, (int)m_param.data_temporal_param.batch_size, (int)m_nNumHistoricalSteps, (int)m_nNumFutureSteps, m_param.data_temporal_param.max_load_percent, m_param.data_temporal_param.drip_refresh_rate_in_sec, m_param.data_temporal_param.chunk_count, m_log, m_evtCancel))
                throw new Exception("DataTemporalLayer data loading aborted!");

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

            if ((rgShape = m_data.GetShape(Data<T>.OUTPUT_TYPE.STATIC_NUMERIC)) != null)
                colTop[0].Reshape(rgShape);

            if ((rgShape = m_data.GetShape(Data<T>.OUTPUT_TYPE.STATIC_CATEGORICAL)) != null)
                colTop[1].Reshape(rgShape);

            if ((rgShape = m_data.GetShape(Data<T>.OUTPUT_TYPE.HISTORICAL_NUMERIC)) != null)
                colTop[2].Reshape(rgShape);

            if ((rgShape = m_data.GetShape(Data<T>.OUTPUT_TYPE.HISTORICAL_CATEGORICAL)) != null)
                colTop[3].Reshape(rgShape);

            if ((rgShape = m_data.GetShape(Data<T>.OUTPUT_TYPE.FUTURE_NUMERIC)) != null)
                colTop[4].Reshape(rgShape);

            if ((rgShape = m_data.GetShape(Data<T>.OUTPUT_TYPE.FUTURE_CATEGORICAL)) != null)
                colTop[5].Reshape(rgShape);

            if (colTop.Count > 6)
            {
                if ((rgShape = m_data.GetShape(Data<T>.OUTPUT_TYPE.TARGET)) != null)
                    colTop[6].Reshape(rgShape);
            }
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
            m_data.LoadBatch((int)m_nBatchSize, colTop);
        }

        /// @brief Not implemented - data Layers do not perform backward.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }
    }

    /// <summary>
    /// The RawFileData object is used to load raw NPY file data.
    /// </summary>
    /// <typeparam name="T">Specifies the base data type of 'float' or 'double'.</typeparam>
    class RawFileData<T>
    {
        Data<T> m_data;
        Random m_random;
        int m_nBatchSize;


        /// <summary>
        /// The constructor.
        /// </summary>
        public RawFileData(uint? nSeed)
        {
            if (nSeed.HasValue)
                m_random = new Random((int)nSeed.Value);
            else
                m_random = new Random();
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
        /// <param name="dfPctMaxLoad">Specifies the percent of total items to load in background (default = 1, or 100%).</param>
        /// <param name="nDripRefreshRateInSec">Specifies the rate in seconds to refresh the data.</param>
        /// <param name="nChunkCount">Specifies the number of items to load on each cycle.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the cancel event.</param>
        public bool LoadData(Phase phase, string strPath, bool bShuffleData, int nBatchSize, int nHistoricalSteps, int nFutureSteps, double dfPctMaxLoad, int nDripRefreshRateInSec, uint nChunkCount, Log log, CancelEvent evtCancel)
        {
            m_nBatchSize = nBatchSize;
            m_data = new Data<T>(m_random, log, nHistoricalSteps, nFutureSteps, bShuffleData);

            VerifyFiles(phase, strPath);

            ManualResetEvent evtReady = new ManualResetEvent(false);
            ManualResetEvent evtDone = new ManualResetEvent(false);
            Thread threadLoad = new Thread(new ParameterizedThreadStart(loadDataFunction));
            threadLoad.Start(new DataLoadParameters(phase, strPath, nHistoricalSteps, nFutureSteps, dfPctMaxLoad, nDripRefreshRateInSec, nChunkCount, bShuffleData, log, evtCancel, evtReady, evtDone));

            while (!evtReady.WaitOne(1000))
            {
                if (evtCancel.WaitOne(0))
                    return false;

                Thread.Sleep(50);
            }

            return true;
        }

        private void loadDataFunction(object obj)
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
            Data<T> dataChunk = null;
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

                dataChunk = new Data<T>(m_data);
                dataChunk.Open(strPath, strType, m_nBatchSize);

                int nRowIdx = 0;
                int nRowCount = dataChunk.RowCount;
                int nMaxLoadCount = (int)(nRowCount * dfMaxLoadPct);
                int nWaitCount = 0;

                Stopwatch sw = new Stopwatch();
                sw.Start();

                while (!evtCancel.WaitOne(0))
                {
                    bool bEndOfData = false;

                    while (dataChunk.Load(nRowIdx, out bEndOfData) && !bEndOfData)
                    {
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

        /// <summary>
        /// Loads a batch of data items into the BlobCollection.
        /// </summary>
        /// <param name="nBatchSize">Specifies the batch size.</param>
        /// <param name="col">Specifies the blob collection to load the batch into.</param>
        public void LoadBatch(int nBatchSize, BlobCollection<T> col)
        {
            m_data.LoadBatch(nBatchSize, col);
        }

        /// <summary>
        /// Returns the total size of the data.
        /// </summary>
        /// <returns>The total size is returned.</returns>
        public int GetTotalSize()
        {
            return m_data.GetTotalSize();
        }

        /// <summary>
        /// Returns the shape of a given output type.
        /// </summary>
        /// <param name="ot">Specifies the output type.</param>
        /// <returns>The shape returned can be used to reshape the Blob used to store the data on the GPU.</returns>
        public int[] GetShape(Data<T>.OUTPUT_TYPE ot)
        {
            return m_data.GetShape(ot);
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

    class Data<T> : IDisposable /** @private */
    {
        Random m_random;
        Log m_log;
        DataSchema m_schema;
        int m_nHistoricalSteps;
        int m_nFutureSteps;
        bool m_bShuffleData;
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
        int m_nRows = 0;
        int m_nBatchSize = 0;
        int m_nTotalSize = 0;
        object m_syncObj = new object();

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
            FUTURE_SYNC
        }

        public Data(Random random, Log log, int nHistoricalSteps, int nFutureSteps, bool bShuffleData)
        {
            m_random = random;
            m_log = log;
            m_nHistoricalSteps = nHistoricalSteps;
            m_nFutureSteps = nFutureSteps;
            m_bShuffleData = bShuffleData;
        }

        public Data(Data<T> data)
        {
            m_random = data.m_random;
            m_log = data.m_log;
            m_nHistoricalSteps = data.m_nHistoricalSteps;
            m_nFutureSteps = data.m_nFutureSteps;
            m_bShuffleData = data.m_bShuffleData;
        }

        public void Dispose()
        {
            Close();
        }

        public void Open(string strPath, string strType, int nBatchSize)
        {
            int nLen;
            m_schema = DataSchema.Load(strPath + "\\" + strType + "_schema.xml");
            m_nTargetFieldIdx = m_schema.Data.ObservedNum.FindFieldIndex(Field.INPUT_TYPE.TARGET);

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

            nLen = nBatchSize * m_nHistoricalSteps * m_rgNumFiles[DATA_TYPE.OBSERVED_NUMERIC].Fields;
            m_rgBatchBuffers.Add(OUTPUT_TYPE.HISTORICAL_NUMERIC, new float[nLen]);
            // The future observed are the target values.
            nLen = nBatchSize * m_nFutureSteps * m_rgNumFiles[DATA_TYPE.OBSERVED_NUMERIC].Fields;
            m_rgBatchBuffers.Add(OUTPUT_TYPE.TARGET, new float[nLen]);

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

        public void Close()
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

        public int RowCount
        {
            get { return m_nRows; }
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

        public bool Load(int nRowIdx, out bool bEndOfData)
        {
            bEndOfData = false;

            if (nRowIdx >= m_nRows)
            {
                bEndOfData = true;
                return true;
            }

            int nStartIdx = m_schema.Lookups[0][nRowIdx].ValidRangeStartIndex;
            int nEndIdx = m_schema.Lookups[0][nRowIdx].ValidRangeEndIndex;
            int nFields = m_rgFields[DATA_TYPE.SYNC];
            if (nStartIdx < 0 || nEndIdx - nStartIdx < (m_nHistoricalSteps + m_nFutureSteps))
                return false;
            
            Dictionary<DATA_TYPE, long[]> cat = new Dictionary<DATA_TYPE, long[]>();
            foreach (KeyValuePair<DATA_TYPE, NumpyFile<long>> kvp in m_rgCatFiles)
            {
                int nStartIdx1 = (kvp.Key == DATA_TYPE.STATIC_CATEGORICAL) ? 0 : nStartIdx;
                int nEndIdx1 = (kvp.Key == DATA_TYPE.STATIC_CATEGORICAL) ? 0 : nEndIdx;
                long[] rgBuffer = null;
                rgBuffer = kvp.Value.LoadRow(rgBuffer, nRowIdx, nStartIdx1, (nEndIdx1 - nStartIdx1) + 1);
                cat.Add(kvp.Key, rgBuffer);
                if (rgBuffer == null)
                    bEndOfData = true;
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
                    bEndOfData = true;
            }

            if (bEndOfData)
                return true;

            foreach (KeyValuePair<DATA_TYPE, long[]> kvp in cat)
            {
                m_rgCatData[kvp.Key].Add(kvp.Value);
            }

            foreach (KeyValuePair<DATA_TYPE, float[]> kvp in num)
            {
                m_rgNumData[kvp.Key].Add(kvp.Value);
            }

            return true;
        }

        public bool Add(Data<T> data, int nMaxLoad)
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

        public int GetTotalSize()
        {
            return m_nTotalSize;
        }

        public int[] GetShape(OUTPUT_TYPE ot)
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
                        nFields += m_rgFields[DATA_TYPE.OBSERVED_NUMERIC];
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

                default:
                    throw new Exception("Unknown output type '" + ot.ToString() + "'!");
            }

            return null;
        }

        public bool IsReady
        {
            get { return GetTotalSize() >= m_nBatchSize; }
        }

        private void stepNext()
        {
            if (m_bShuffleData)
            {
                m_nRowIdx = m_random.Next(m_rgNumData[DATA_TYPE.OBSERVED_NUMERIC].Count());
                m_nColIdx = m_random.Next(m_rgNumData[DATA_TYPE.OBSERVED_NUMERIC][m_nRowIdx].Length - (m_nHistoricalSteps + m_nFutureSteps));
            }
            else
            {
                m_nColIdx++;
                if (m_nColIdx + m_nHistoricalSteps + m_nFutureSteps > m_schema.Lookups[0][m_nRowIdx].ValidRangeCount)
                {
                    m_nColIdx = 0;

                    m_nRowIdx++;
                    if (m_nRowIdx >= m_nMaxRowIdx)
                        m_nRowIdx = 0;
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
            {
                m_log.WriteLine("WARNING: Row " + m_nRowIdx.ToString() + ", Col " + m_nColIdx.ToString() + " does not have enough data to load sync batch at index " + nStartIdx1.ToString() + " with count " + nCount.ToString() + ".");
                return false;
            }

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

        private void loadNumBatch(int nIdx, float[] rg, int nStartIdx, int nCount, DATA_TYPE dt1, DATA_TYPE dt2)
        {
            if (rg == null)
                return;

            int nStartIdx1 = m_nColIdx + nStartIdx;
            int nFields1 = (m_rgFields.ContainsKey(dt1)) ? m_rgFields[dt1] : 0;
            float[] rgSrc1 = (m_rgFields.ContainsKey(dt1)) ? m_rgNumData[dt1][m_nRowIdx] : null;
            int nFields2 = (m_rgFields.ContainsKey(dt2)) ? m_rgFields[dt2] : 0;
            float[] rgSrc2 = (m_rgFields.ContainsKey(dt2)) ? m_rgNumData[dt2][m_nRowIdx] : null;
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

        public void LoadBatch(int nBatchSize, BlobCollection<T> col)
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
            }
        }
    }

#pragma warning restore 1591
}
