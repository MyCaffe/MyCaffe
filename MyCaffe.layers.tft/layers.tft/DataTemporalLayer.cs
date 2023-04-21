using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

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
        uint m_nNumStaticFeatsNumeric;
        uint m_nNumStaticFeatsCategorical;
        uint m_nNumHistoricalNumeric;
        uint m_nNumHistoricalCategorical;
        uint m_nNumFutureNumeric;
        uint m_nNumFutureCategorical;
        RawFileData<T> m_data = new RawFileData<T>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public DataTemporalLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
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

            m_data.LoadData(m_phase, m_param.data_temporal_param.source, m_log);

            int nTotalSize = m_data.Data[RawFileData<T>.DATA_TYPE.STATIC_FEAT_NUMERIC].Item2[0];
            m_log.CHECK_GT(nTotalSize, m_nBatchSize, "There must be enough items for at least one batch - items found = " + nTotalSize.ToString() + ", batch size = " + m_nBatchSize.ToString());            

            m_nNumStaticFeatsNumeric = (uint)m_data.Data[RawFileData<T>.DATA_TYPE.STATIC_FEAT_NUMERIC].Item2[1];
            m_log.CHECK_EQ(m_data.Data[RawFileData<T>.DATA_TYPE.STATIC_FEAT_NUMERIC].Item2[0], nTotalSize, "The batch sizes do not match!");

            m_nNumStaticFeatsCategorical = (uint)m_data.Data[RawFileData<T>.DATA_TYPE.STATIC_FEAT_CATEGORICAL].Item2[1];
            m_log.CHECK_EQ(m_data.Data[RawFileData<T>.DATA_TYPE.STATIC_FEAT_CATEGORICAL].Item2[0], nTotalSize, "The batch sizes do not match!");

            m_nNumHistoricalNumeric = (uint)m_data.Data[RawFileData<T>.DATA_TYPE.HISTORICAL_NUMERIC].Item2[2];
            m_log.CHECK_EQ(m_data.Data[RawFileData<T>.DATA_TYPE.HISTORICAL_NUMERIC].Item2[0], nTotalSize, "The batch sizes do not match!");
            m_log.CHECK_EQ(m_data.Data[RawFileData<T>.DATA_TYPE.HISTORICAL_NUMERIC].Item2[1], m_nNumHistoricalSteps, "The num historical steps do not match!");

            m_nNumHistoricalCategorical = (uint)m_data.Data[RawFileData<T>.DATA_TYPE.HISTORICAL_CATEGORICAL].Item2[2];
            m_log.CHECK_EQ(m_data.Data[RawFileData<T>.DATA_TYPE.HISTORICAL_CATEGORICAL].Item2[0], nTotalSize, "The batch sizes do not match!");
            m_log.CHECK_EQ(m_data.Data[RawFileData<T>.DATA_TYPE.HISTORICAL_CATEGORICAL].Item2[1], m_nNumHistoricalSteps, "The num historical steps do not match!");

            m_nNumFutureNumeric = (uint)m_data.Data[RawFileData<T>.DATA_TYPE.FUTURE_NUMERIC].Item2[2];
            m_log.CHECK_EQ(m_data.Data[RawFileData<T>.DATA_TYPE.FUTURE_NUMERIC].Item2[0], nTotalSize, "The batch sizes do not match!");
            m_log.CHECK_EQ(m_data.Data[RawFileData<T>.DATA_TYPE.FUTURE_NUMERIC].Item2[1], m_nNumFutureSteps, "The num future steps do not match!");

            m_nNumFutureCategorical = (uint)m_data.Data[RawFileData<T>.DATA_TYPE.FUTURE_CATEGORICAL].Item2[2];
            m_log.CHECK_EQ(m_data.Data[RawFileData<T>.DATA_TYPE.FUTURE_CATEGORICAL].Item2[0], nTotalSize, "The batch sizes do not match!");
            m_log.CHECK_EQ(m_data.Data[RawFileData<T>.DATA_TYPE.FUTURE_CATEGORICAL].Item2[1], m_nNumFutureSteps, "The num future steps do not match!");
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_rgShape.Clear();
            m_rgShape.Add((int)m_nBatchSize);
            m_rgShape.Add(0);

            if (m_nNumStaticFeatsNumeric > 0)
            {
                m_rgShape[1] = (int)m_nNumStaticFeatsNumeric;
                colTop[0].Reshape(m_rgShape);
            }

            if (m_nNumStaticFeatsCategorical > 0)
            {
                m_rgShape[1] = (int)m_nNumStaticFeatsCategorical;
                colTop[1].Reshape(m_rgShape);
            }

            m_rgShape[1] = (int)m_nNumHistoricalSteps;
            m_rgShape.Add(0);

            m_rgShape[2] = (int)m_nNumHistoricalNumeric;
            colTop[2].Reshape(m_rgShape);

            m_rgShape[2] = (int)m_nNumHistoricalCategorical;
            colTop[3].Reshape(m_rgShape);

            m_rgShape[1] = (int)m_nNumFutureSteps;

            m_rgShape[2] = (int)m_nNumFutureNumeric;
            colTop[4].Reshape(m_rgShape);

            m_rgShape[2] = (int)m_nNumFutureCategorical;
            colTop[5].Reshape(m_rgShape);

            if (colTop.Count > 6)
            {
                m_rgShape.Clear();
                m_rgShape.Add((int)m_nBatchSize);
                m_rgShape.Add((int)m_nNumFutureSteps);
                colTop[6].Reshape(m_rgShape);
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

    class RawFileData<T>
    {
        List<string> m_rgstrCombinationID;
        List<DateTime> m_rgTimestamps = new List<DateTime>();
        Random m_random = new Random();
        int m_nTotalCount = 0;
        Dictionary<DATA_TYPE, Tuple<List<float[]>, int[], List<string>>> m_rgData = new Dictionary<DATA_TYPE, Tuple<List<float[]>, int[], List<string>>>();
        int m_nStaticNumericCount = 0;
        float[] m_rgStaticNumericBatch = null;
        int m_nStaticCategoricalCount = 0;
        float[] m_rgStaticCategoricalBatch = null;
        int m_nHistoricalNumericCount = 0;
        float[] m_rgHistoricalNumericBatch = null;
        int m_nHistoricalCategoricalCount = 0;
        float[] m_rgHistoricalCategoricalBatch = null;
        int m_nFutureNumericCount = 0;
        float[] m_rgFutureNumericBatch = null;
        int m_nFutureCategoricalCount = 0;
        float[] m_rgFutureCategoricalBatch = null;
        int m_nTargetCount = 0;
        float[] m_rgTargetBatch = null;

        public enum DATA_TYPE
        {
            COMBINATION_ID,
            TIME_INDEX,
            STATIC_FEAT_NUMERIC,
            STATIC_FEAT_CATEGORICAL,
            HISTORICAL_NUMERIC,
            HISTORICAL_CATEGORICAL,
            FUTURE_NUMERIC,
            FUTURE_CATEGORICAL,
            TARGET
        }

        public RawFileData() 
        { 
        }

        public static DateTime UnixTimeStampToDateTime(double unixTimeStamp)
        {
            // Unix timestamp is seconds past epoch
            DateTime dateTime = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);
            dateTime = dateTime.AddSeconds(unixTimeStamp).ToLocalTime();
            return dateTime;
        }

        private int calculateCount(DATA_TYPE dtype)
        {
            int[] rg = m_rgData[dtype].Item2;

            if (rg == null || rg.Length < 2)
                return 0;

            int nCount = rg[1];

            for (int i = 2; i < rg.Length; i++)
            {
                nCount *= rg[i];
            }

            return nCount;
        }

        public void LoadData(Phase phase, string strPath, Log log)
        {
            string strFile;
            string strType = "train";
            strPath = strPath.TrimEnd('\\', '/');
            strPath += "\\";

            if (phase == Phase.TEST)
                strType = "validation";

            if (!m_rgData.ContainsKey(DATA_TYPE.COMBINATION_ID))
            {
                strFile = strPath + strType + "_combination_id.npy";
                m_rgData.Add(DATA_TYPE.COMBINATION_ID, Blob<float>.LoadFromNumpyEx(strFile, log));
                m_nTotalCount = m_rgData[DATA_TYPE.COMBINATION_ID].Item2[0];
                m_rgstrCombinationID = m_rgData[DATA_TYPE.COMBINATION_ID].Item3;                
            }

            if (!m_rgData.ContainsKey(DATA_TYPE.TIME_INDEX))
            {
                strFile = strPath + strType + "_time_index.npy";
                m_rgData.Add(DATA_TYPE.TIME_INDEX, Blob<float>.LoadFromNumpyEx(strFile, log));

                List<float[]> rgTime = m_rgData[DATA_TYPE.TIME_INDEX].Item1;
                foreach (float[] rgTime2 in rgTime)
                {
                    DateTime dt = UnixTimeStampToDateTime(rgTime2[0]);
                    m_rgTimestamps.Add(dt);
                }
            }

            if (!m_rgData.ContainsKey(DATA_TYPE.STATIC_FEAT_NUMERIC))
            {
                strFile = strPath + strType + "_static_feats_numeric.npy";
                m_rgData.Add(DATA_TYPE.STATIC_FEAT_NUMERIC, Blob<float>.LoadFromNumpyEx(strFile, log));
                m_nStaticNumericCount = calculateCount(DATA_TYPE.STATIC_FEAT_NUMERIC);
            }

            if (!m_rgData.ContainsKey(DATA_TYPE.STATIC_FEAT_CATEGORICAL))
            {
                strFile = strPath + strType + "_static_feats_categorical.npy";
                m_rgData.Add(DATA_TYPE.STATIC_FEAT_CATEGORICAL, Blob<float>.LoadFromNumpyEx(strFile, log));
                m_nStaticCategoricalCount = calculateCount(DATA_TYPE.STATIC_FEAT_CATEGORICAL);
            }

            if (!m_rgData.ContainsKey(DATA_TYPE.HISTORICAL_NUMERIC))
            {
                strFile = strPath + strType + "_historical_ts_numeric.npy";
                m_rgData.Add(DATA_TYPE.HISTORICAL_NUMERIC, Blob<float>.LoadFromNumpyEx(strFile, log));
                m_nHistoricalNumericCount = calculateCount(DATA_TYPE.HISTORICAL_NUMERIC);
            }

            if (!m_rgData.ContainsKey(DATA_TYPE.HISTORICAL_CATEGORICAL))
            {
                strFile = strPath + strType + "_historical_ts_categorical.npy";
                m_rgData.Add(DATA_TYPE.HISTORICAL_CATEGORICAL, Blob<float>.LoadFromNumpyEx(strFile, log));
                m_nHistoricalCategoricalCount = calculateCount(DATA_TYPE.HISTORICAL_CATEGORICAL);
            }

            if (!m_rgData.ContainsKey(DATA_TYPE.FUTURE_NUMERIC))
            {
                strFile = strPath + strType + "_future_ts_numeric.npy";
                m_rgData.Add(DATA_TYPE.FUTURE_NUMERIC, Blob<float>.LoadFromNumpyEx(strFile, log));
                m_nFutureNumericCount = calculateCount(DATA_TYPE.FUTURE_NUMERIC);
            }

            if (!m_rgData.ContainsKey(DATA_TYPE.FUTURE_CATEGORICAL))
            {
                strFile = strPath + strType + "_future_ts_categorical.npy";
                m_rgData.Add(DATA_TYPE.FUTURE_CATEGORICAL, Blob<float>.LoadFromNumpyEx(strFile, log));
                m_nFutureCategoricalCount = calculateCount(DATA_TYPE.FUTURE_CATEGORICAL);
            }

            if (!m_rgData.ContainsKey(DATA_TYPE.TARGET))
            {
                strFile = strPath + strType + "_target.npy";
                m_rgData.Add(DATA_TYPE.TARGET, Blob<float>.LoadFromNumpyEx(strFile, log));
                m_nTargetCount = calculateCount(DATA_TYPE.TARGET);
            }
        }

        public Dictionary<DATA_TYPE, Tuple<List<float[]>, int[], List<string>>> Data
        {
            get { return m_rgData; }
        }

        private float[] createBatchBuffer(float[] rg, int nBatchSize, int nCount)
        {
            if (nCount == 0)
                return rg;

            int nDim = nBatchSize * nCount;
            if (rg == null || rg.Length != nDim)
                return new float[nDim];

            return rg;
        }
        public void LoadBatch(int nBatchSize, BlobCollection<T> col)
        {
            List<float[]> rgStaticNumeric = m_rgData[DATA_TYPE.STATIC_FEAT_NUMERIC].Item1;
            List<float[]> rgStaticCategorical = m_rgData[DATA_TYPE.STATIC_FEAT_CATEGORICAL].Item1;
            List<float[]> rgHistoricalNumeric = m_rgData[DATA_TYPE.HISTORICAL_NUMERIC].Item1;
            List<float[]> rgHistoricalCategorical = m_rgData[DATA_TYPE.HISTORICAL_CATEGORICAL].Item1;
            List<float[]> rgFutureNumeric = m_rgData[DATA_TYPE.FUTURE_NUMERIC].Item1;
            List<float[]> rgFutureCategorical = m_rgData[DATA_TYPE.FUTURE_CATEGORICAL].Item1;
            List<float[]> rgTarget = m_rgData[DATA_TYPE.TARGET].Item1;

            m_rgStaticNumericBatch = createBatchBuffer(m_rgStaticNumericBatch, nBatchSize, m_nStaticNumericCount);
            m_rgStaticCategoricalBatch = createBatchBuffer(m_rgStaticCategoricalBatch, nBatchSize, m_nStaticCategoricalCount);
            m_rgHistoricalNumericBatch = createBatchBuffer(m_rgHistoricalNumericBatch, nBatchSize, m_nHistoricalNumericCount);
            m_rgHistoricalCategoricalBatch = createBatchBuffer(m_rgHistoricalCategoricalBatch, nBatchSize, m_nHistoricalCategoricalCount);
            m_rgFutureNumericBatch = createBatchBuffer(m_rgFutureNumericBatch, nBatchSize, m_nFutureNumericCount);
            m_rgFutureCategoricalBatch = createBatchBuffer(m_rgFutureCategoricalBatch, nBatchSize, m_nFutureCategoricalCount);
            m_rgTargetBatch = createBatchBuffer(m_rgTargetBatch, nBatchSize, m_nTargetCount);

            for (int i=0; i<nBatchSize; i++)
            {
                int nIdx = m_random.Next(m_nTotalCount);

                if (m_rgStaticNumericBatch != null)
                {
                    float[] rgStaticNumeric1 = rgStaticNumeric[nIdx];
                    Array.Copy(rgStaticNumeric1, 0, m_rgStaticNumericBatch, i * m_nStaticNumericCount, m_nStaticNumericCount);
                }

                if (m_rgStaticCategoricalBatch != null)
                {
                    float[] rgStaticCategorical1 = rgStaticCategorical[nIdx];
                    Array.Copy(rgStaticCategorical1, 0, m_rgStaticCategoricalBatch, i * m_nStaticCategoricalCount, m_nStaticCategoricalCount);
                }

                if (m_rgHistoricalNumericBatch != null)
                {
                    float[] rgHistoricalNumeric1 = rgHistoricalNumeric[nIdx];
                    Array.Copy(rgHistoricalNumeric1, 0, m_rgHistoricalNumericBatch, i * m_nHistoricalNumericCount, m_nHistoricalNumericCount);
                }

                if (m_rgHistoricalCategoricalBatch != null)
                {
                    float[] rgHistoricalCategorical1 = rgHistoricalCategorical[nIdx];
                    Array.Copy(rgHistoricalCategorical1, 0, m_rgHistoricalCategoricalBatch, i * m_nHistoricalCategoricalCount, m_nHistoricalCategoricalCount);
                }

                if (m_rgFutureNumericBatch != null)
                {
                    float[] rgFutureNumeric1 = rgFutureNumeric[nIdx];
                    Array.Copy(rgFutureNumeric1, 0, m_rgFutureNumericBatch, i * m_nFutureNumericCount, m_nFutureNumericCount);
                }

                if (m_rgFutureCategoricalBatch != null)
                {
                    float[] rgFutureCategorical1 = rgFutureCategorical[nIdx];
                    Array.Copy(rgFutureCategorical1, 0, m_rgFutureCategoricalBatch, i * m_nFutureCategoricalCount, m_nFutureCategoricalCount);
                }

                if (m_rgTargetBatch != null && col.Count > 6)
                {
                    float[] rgTarget1 = rgTarget[nIdx];
                    Array.Copy(rgTarget1, 0, m_rgTargetBatch, i * m_nTargetCount, m_nTargetCount);
                }
            }

            if (m_rgStaticNumericBatch != null)
                col[0].mutable_cpu_data = Utility.ConvertVec<T>(m_rgStaticNumericBatch);

            if (m_rgStaticCategoricalBatch != null)
                col[1].mutable_cpu_data = Utility.ConvertVec<T>(m_rgStaticCategoricalBatch);

            col[2].mutable_cpu_data = Utility.ConvertVec<T>(m_rgHistoricalNumericBatch);
            col[3].mutable_cpu_data = Utility.ConvertVec<T>(m_rgHistoricalCategoricalBatch);
            col[4].mutable_cpu_data = Utility.ConvertVec<T>(m_rgFutureNumericBatch);
            col[5].mutable_cpu_data = Utility.ConvertVec<T>(m_rgFutureCategoricalBatch);

            if (col.Count > 6)
                col[6].mutable_cpu_data = Utility.ConvertVec<T>(m_rgTargetBatch);
        }
    }
}
