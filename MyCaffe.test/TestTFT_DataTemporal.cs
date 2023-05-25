using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe.data;
using MyCaffe.layers.tft;
using System.Threading;
using System.Diagnostics;
using MyCaffe.param.tft;
using System.IO;
using System.Xml;
using System.ServiceModel.Security;
using System.Data.Entity.ModelConfiguration.Conventions;

/// <summary>
/// Testing the DataTemporal.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_DataTemporal
    {
        [TestMethod]
        public void TestForwardTrainElectricity2()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\electricity\\preprocessed";

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(Phase.TRAIN, DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE, strPath, 100);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTestElectricity2()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\electricity\\preprocessed";

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(Phase.TEST, DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE, strPath, 100);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardRunElectricity2()
        {
            DataTemporalTest test = new DataTemporalTest();
            string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\electricity\\preprocessed";

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward(Phase.RUN, DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE, strPath, 100);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBlobLoadNumpyPartial()
        {
            DataTemporalTest test = new DataTemporalTest();

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestBlobLoadNumpyPartial();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

    }

    interface IDataTemporalTest : ITest
    {
        void TestForward(Phase phase, DataTemporalParameter.SOURCE_TYPE srcType, string srcPath, int nBatchSize);
        void TestBlobLoadNumpyPartial();
    }

    class DataTemporalTest : TestBase
    {
        public DataTemporalTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TFT DataTemporal Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new DataTemporalTest<double>(strName, nDeviceID, engine);
            else
                return new DataTemporalTest<float>(strName, nDeviceID, engine);
        }
    }

    class DataTemporalTest<T> : TestEx<T>, IDataTemporalTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();

        public DataTemporalTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            m_colData.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        private string getTestDataPath()
        {
            return "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data2\\data\\electricity\\preprocessed";
        }

        private string getTestWtsPath()
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\static_enrichment_grn\\";
        }

        private string buildModel(int nNumSamples, int nNumHist, int nNumFuture, DataTemporalParameter.SOURCE_TYPE srcType, string strSrc)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";


            //---------------------------------
            //  Data Temporal Input
            //---------------------------------
            LayerParameter data = new LayerParameter(LayerParameter.LayerType.DATA_TEMPORAL, "data");
            data.data_temporal_param.batch_size = (uint)nNumSamples;
            data.data_temporal_param.num_historical_steps = (uint)nNumHist;
            data.data_temporal_param.num_future_steps = (uint)nNumFuture;
            data.data_temporal_param.source = strSrc;
            data.data_temporal_param.source_type = srcType;
            data.data_temporal_param.shuffle_data = false;
            data.data_temporal_param.chunk_count = 1024;
            data.data_temporal_param.drip_refresh_rate_in_sec = 0;
            data.data_temporal_param.max_load_percent = 1.0;
            data.data_temporal_param.seed = 1704;
            data.include.Add(new NetStateRule(Phase.TRAIN));
            data.top.Add("x_numeric_static");
            data.top.Add("x_categorical_static");
            data.top.Add("x_numeric_hist");
            data.top.Add("x_categorical_hist");
            data.top.Add("x_numeric_future");
            data.top.Add("x_categorical_future");
            data.top.Add("target");
            p.layer.Add(data);

            return p.ToProto("root").ToString();
        }

        private Tuple<long[], int[]> load(string strPath, string strType, string strFile)
        {
            string strFile1 = strPath + "\\" + strType + "_" + strFile;
            NumpyFile<long> npy = new NumpyFile<long>(m_log);
            npy.OpenRead(strFile1);

            long[] rgData = new long[npy.TotalCount];
            int nOffset = 0;
            long[] rgVal = null;

            for (int i = 0; i < npy.Rows; i++)
            {
                rgVal = npy.LoadRow(rgVal, i);
                Array.Copy(rgVal, 0, rgData, nOffset, rgVal.Length);
                nOffset += rgVal.Length;
            }

            npy.Dispose();

            return new Tuple<long[], int[]>(rgData, npy.Shape);
        }

        private void load(Blob<T> blob, string strPath, string strType, string strFile)
        {
            string strFile1 = strPath + "\\" + strType + "_" + strFile;
            NumpyFile<float> npy = new NumpyFile<float>(m_log);
            npy.OpenRead(strFile1);

            float[] rgData = new float[npy.TotalCount];
            int nOffset = 0;
            float[] rgVal = null;

            for (int i = 0; i < npy.Rows; i++)
            {
                rgVal = npy.LoadRow(rgVal, i);
                Array.Copy(rgVal, 0, rgData, nOffset, rgVal.Length);
                nOffset += rgVal.Length;
            }

            blob.Reshape(npy.Shape);
            blob.mutable_cpu_data = convert(rgData);

            npy.Dispose();
        }

        private void loadNum(ref int nRowIdx, ref int nColIdx, DataSchema schema, Blob<T> blobObsNum, Blob<T> blobKnownNum, Blob<T> blobHistNum, Blob<T> blobFutNum, Blob<T> blobTarget, int nNumHist, int nNumFut, int nBatchSize)
        {
            float[] rgObsNum = convertF(blobObsNum.mutable_cpu_data);
            float[] rgKnownNum = convertF(blobKnownNum.mutable_cpu_data);

            int nObsFields = schema.Data.ObservedNum.Count;
            int nKnownFields = schema.Data.KnownNum.Count;
            int nFields = nObsFields + nKnownFields;

            blobHistNum.Reshape(nBatchSize, nNumHist, nObsFields + nKnownFields, 1);
            blobFutNum.Reshape(nBatchSize, nNumFut, nKnownFields, 1);
            blobTarget.Reshape(nBatchSize, nNumFut, 1, 1);

            float[] rgHistNum = convertF(blobHistNum.mutable_cpu_data);
            float[] rgFutNum = convertF(blobFutNum.mutable_cpu_data);
            float[] rgTarget = convertF(blobTarget.mutable_cpu_data);

            int nTargetIndex = schema.Data.TargetIndex;
            if (nTargetIndex < 0)
                m_log.FAIL("Could not find the target index in the schema!");

            for (int i = 0; i < nBatchSize; i++)
            {
                int nRowOffset = nRowIdx * schema.Data.Columns;
                int nStartIdx = schema.Lookups[0][nRowIdx].ValidRangeStartIndex;
                int nStartIdx1 = nRowOffset + nStartIdx + nColIdx;

                for (int j = 0; j < nNumHist + nNumFut; j++)
                {
                    int nIdxSrc = nStartIdx1 + j;

                    if (j < nNumHist)
                    {
                        int nIdxDst = (i * nNumHist * nFields) + (j * nFields);

                        for (int k = 0; k < nObsFields; k++)
                        {
                            rgHistNum[nIdxDst + k] = rgObsNum[nIdxSrc * nObsFields + k];
                        }

                        for (int k = 0; k < nKnownFields; k++)
                        {
                            rgHistNum[nIdxDst + nObsFields + k] = rgKnownNum[nIdxSrc * nKnownFields + k];
                        }
                    }
                    else
                    {
                        int nLocalJ = j - nNumHist;
                        int nIdxDst = (i * nNumFut * nKnownFields) + (nLocalJ * nKnownFields);
                        int nIdxDst1 = (i * nNumFut) + nLocalJ;

                        rgTarget[nIdxDst1] = rgObsNum[nIdxSrc * nObsFields + nTargetIndex];

                        for (int k = 0; k < nKnownFields; k++)
                        {
                            rgFutNum[nIdxDst + k] = rgKnownNum[nIdxSrc * nKnownFields + k];
                        }
                    }
                }

                nColIdx++;
                if (nColIdx + nNumHist + nNumFut > schema.Lookups[0][nRowIdx].ValidRangeCount)
                {
                    nColIdx = 0;
                    nRowIdx++;

                    if (nRowIdx >= schema.Lookups[0].Count)
                        nRowIdx = 0;
                }
            }

            blobHistNum.mutable_cpu_data = convert(rgHistNum);
            blobFutNum.mutable_cpu_data = convert(rgFutNum);
            blobTarget.mutable_cpu_data = convert(rgTarget);
        }

        /// <summary>
        /// Test the forward pass for self attention
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run training.py on fresh 'test\iter_0' data (run just one iteration to save data)
        ///     with: 
        ///         debug = True
        ///         use_mycaffe = False
        ///         use_mycaffe_data = False
        ///         use_mycaffe_model_direct = False
        ///         use_mycaffe_model = False
        ///         lstm_use_mycaffe = False
        ///         tag = "tft.all"
        ///         test = False
        /// 
        /// Fresh test\iter_0 data generated by running:
        /// training.py with TemporalFusionTransformer options: debug=True, tag='tft', use_mycaffe=True
        /// </remarks>
        public void TestForward(Phase phase, DataTemporalParameter.SOURCE_TYPE srcType, string strSrcPath, int nBatchSize)
        {
            CancelEvent evtCancel = new CancelEvent();
            string strPath = getTestDataPath();
            string strPathWt = getTestWtsPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;
            Blob<T> blobObsNum = null;
            Blob<T> blobKnownNum = null;
            Blob<T> blobTarget = null;
            Blob<T> blobHistNum = null;
            Blob<T> blobFutNum = null;

            Net<T> net = null;
            int nNumSamples = nBatchSize;
            int nNumHist = 90;
            int nNumFuture = 30;
            int nMaxIter = 200;
            int nRowIdx = 0;
            int nColIdx = 0;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);
                blobObsNum = new Blob<T>(m_cuda, m_log);
                blobKnownNum = new Blob<T>(m_cuda, m_log);
                blobTarget = new Blob<T>(m_cuda, m_log);
                blobHistNum = new Blob<T>(m_cuda, m_log);
                blobFutNum = new Blob<T>(m_cuda, m_log);

                string strType = phase.ToString().ToLower();
                if (strType == "run")
                    strType = "validation";

                DataSchema schema = DataSchema.Load(strSrcPath + "\\" + strType + "_schema.xml");
                Tuple<long[], int[]> sync = load(strSrcPath, strType, "sync.npy");
                load(blobObsNum, strSrcPath, strType, "observed_num.npy");
                load(blobKnownNum, strSrcPath, strType, "known_num.npy");   

                string strModel = buildModel(nNumSamples, nNumHist, nNumFuture, srcType, strSrcPath);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, evtCancel, null, phase);

                for (int i = 0; i < nMaxIter; i++)
                {
                    BlobCollection<T> colRes = net.Forward();

                    loadNum(ref nRowIdx, ref nColIdx, schema, blobObsNum, blobKnownNum, blobHistNum, blobFutNum, blobTarget, nNumHist, nNumFuture, nBatchSize);

                    blob1 = net.FindBlob("x_numeric_static");
                    m_log.CHECK(blob1 != null, "Could not find the blob 'x_numeric_static'!");
                    m_log.CHECK(blob1.CompareShape(new List<int>() { 0 }), "The blob shape is different than expected");

                    blob1 = net.FindBlob("x_categorical_static");
                    m_log.CHECK(blob1 != null, "Could not find the blob 'x_categorical_static'!");
                    m_log.CHECK(blob1.CompareShape(new List<int>() { nBatchSize, 1 }), "The blob shape is different than expected");

                    blob1 = net.FindBlob("x_numeric_hist");
                    m_log.CHECK(blob1 != null, "Could not find the blob 'x_numeric_hist'!");
                    m_log.CHECK(blob1.CompareShape(new List<int>() { nBatchSize, 90, 3 }), "The blob shape is different than expected");
                    m_log.CHECK(blobHistNum.Compare(blob1, blobWork, false, 0), "The blob is different than expected.");

                    blob1 = net.FindBlob("x_categorical_hist");
                    m_log.CHECK(blob1 != null, "Could not find the blob 'x_categorical_hist'!");
                    m_log.CHECK(blob1.CompareShape(new List<int>() { 0 }), "The blob shape is different than expected");

                    blob1 = net.FindBlob("x_numeric_future");
                    m_log.CHECK(blob1 != null, "Could not find the blob 'x_numeric_future'!");
                    m_log.CHECK(blob1.CompareShape(new List<int>() { nBatchSize, 30, 2 }), "The blob shape is different than expected");
                    m_log.CHECK(blobFutNum.Compare(blob1, blobWork, false, 0), "The blob is different than expected.");

                    blob1 = net.FindBlob("x_categorical_future");
                    m_log.CHECK(blob1 != null, "Could not find the blob 'x_categorical_future'!");
                    m_log.CHECK(blob1.CompareShape(new List<int>() { 0 }), "The blob shape is different than expected");

                    blob1 = net.FindBlob("target");
                    m_log.CHECK(blob1 != null, "Could not find the blob 'target'!");
                    m_log.CHECK(blob1.CompareShape(new List<int>() { nBatchSize, 30 }), "The blob shape is different than expected");
                    m_log.CHECK(blobTarget.Compare(blob1, blobWork, false, 0), "The blob is different than expected.");

                    double dfPct = (double)i / nMaxIter;
                    m_log.WriteLine("Testing batch " + i.ToString() + " (" + dfPct.ToString("P") + ")");
                }
            }
            finally
            {
                evtCancel.Set();
                Thread.Sleep(1000);
                
                dispose(ref blobVal);
                dispose(ref blobWork);
                dispose(ref blobObsNum);
                dispose(ref blobTarget);
                dispose(ref blobHistNum);
                dispose(ref blobFutNum);

                if (net != null)
                    net.Dispose();
            }
        }

        public static DateTime UnixTimeStampToDateTime(double unixTimeStamp)
        {
            // Unix timestamp is seconds past epoch
            DateTime dateTime = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);
            dateTime = dateTime.AddSeconds(unixTimeStamp).ToLocalTime();
            return dateTime;
        }        

        public void TestBlobLoadNumpyPartial()
        {
            Blob<T> blobVal = new Blob<T>(m_cuda, m_log);
            Blob<T> blobData = new Blob<T>(m_cuda, m_log);
            Blob<T> blobWork = new Blob<T>(m_cuda, m_log);
            string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\";
            string strFile;

            strFile = strPath + "validation_static_feats_categorical.npy";
            blobVal.LoadFromNumpy(strFile);
            blobData.ReshapeLike(blobVal);

            int nStartIdx = 0;
            int nCount = 1024;

            float[] rgData = Utility.ConvertVecF<T>(blobVal.mutable_cpu_data);
            Array.Clear(rgData, 0, rgData.Length);

            while (nStartIdx < blobVal.num)
            {
                Tuple<List<float[]>, int[], List<string>> data = Blob<T>.LoadFromNumpy(strFile, null, int.MaxValue, nStartIdx, nCount);

                for (int i = 0; i < data.Item1.Count; i++)
                {
                    int nItems = data.Item2.Last();
                    int nIdx = (nStartIdx + i) * nItems;
                    
                    Array.Copy(data.Item1[i], 0, rgData, nIdx, nItems);
                }

                nStartIdx += nCount;
            }

            blobData.mutable_cpu_data = convert(rgData);

            double dfMin;
            double dfMax;
            double dfErr = 1e-12;
            m_log.CHECK(blobVal.CompareEx(blobData, blobWork, out dfMin, out dfMax, false, dfErr), "The blobs are not the same.");
        }
    }
}
