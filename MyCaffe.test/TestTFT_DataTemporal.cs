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

/// <summary>
/// Testing the DataTemporal.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_DataTemporal
    {
        [TestMethod]
        public void TestForward()
        {
            DataTemporalTest test = new DataTemporalTest();

            try
            {
                foreach (IDataTemporalTest t in test.Tests)
                {
                    t.TestForward();
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
        void TestForward();
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
            return "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\all\\iter_0\\data\\";
        }

        private string getTestWtsPath()
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\static_enrichment_grn\\";
        }

        private string buildModel(int nNumSamples, int nNumHist, int nNumFuture)
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
            data.data_temporal_param.source = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita";
            data.data_temporal_param.source_type = DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE;
            data.data_temporal_param.shuffle_data = false;
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
        public void TestForward()
        {
            CancelEvent evtCancel = new CancelEvent();
            string strPath = getTestDataPath();
            string strPathWt = getTestWtsPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nNumHist = 90;
            int nNumFuture = 30;
            double dfMin;
            double dfMax;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(nNumSamples, nNumHist, nNumFuture);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, evtCancel, null, Phase.TRAIN);

                BlobCollection<T> colRes = net.Forward();

                blob1 = net.FindBlob("x_numeric_static");
                m_log.CHECK(blob1 != null, "Could not find the blob 'x_numeric_static'!");
                m_log.CHECK(blob1.CompareShape(new List<int>() { 0 }), "The blob shape is different than expected");

                blob1 = net.FindBlob("x_categorical_static");
                m_log.CHECK(blob1 != null, "Could not find the blob 'x_categorical_static'!");
                m_log.CHECK(blob1.CompareShape(new List<int>() { 256, 9 }), "The blob shape is different than expected");

                blobVal.LoadFromNumpy(strPath + "0_static_feats_categorical.npy");
                m_log.CHECK(blobVal.CompareEx(blob1, blobWork, out dfMin, out dfMax), "The blobs are not the same.");
                m_log.CHECK_EQ(dfMin, 0, "The min value is not as expected.");
                m_log.CHECK_EQ(dfMax, 0, "The max value is not as expected.");

                blob1 = net.FindBlob("x_numeric_hist");
                m_log.CHECK(blob1 != null, "Could not find the blob 'x_numeric_hist'!");
                m_log.CHECK(blob1.CompareShape(new List<int>() { 256, 90, 4 }), "The blob shape is different than expected");

                blobVal.LoadFromNumpy(strPath + "0_historical_ts_numeric.npy");
                m_log.CHECK(blobVal.CompareEx(blob1, blobWork, out dfMin, out dfMax), "The blobs are not the same.");
                m_log.CHECK_EQ(dfMin, 0, "The min value is not as expected.");
                m_log.CHECK_EQ(dfMax, 0, "The max value is not as expected.");

                blob1 = net.FindBlob("x_categorical_hist");
                m_log.CHECK(blob1 != null, "Could not find the blob 'x_categorical_hist'!");
                m_log.CHECK(blob1.CompareShape(new List<int>() { 256, 90, 7 }), "The blob shape is different than expected");

                blobVal.LoadFromNumpy(strPath + "0_historical_ts_categorical.npy");
                m_log.CHECK(blobVal.CompareEx(blob1, blobWork, out dfMin, out dfMax), "The blobs are not the same.");
                m_log.CHECK_EQ(dfMin, 0, "The min value is not as expected.");
                m_log.CHECK_EQ(dfMax, 0, "The max value is not as expected.");

                blob1 = net.FindBlob("x_numeric_future");
                m_log.CHECK(blob1 != null, "Could not find the blob 'x_numeric_future'!");
                m_log.CHECK(blob1.CompareShape(new List<int>() { 256, 30, 1 }), "The blob shape is different than expected");

                blobVal.LoadFromNumpy(strPath + "0_future_ts_numeric.npy");
                m_log.CHECK(blobVal.CompareEx(blob1, blobWork, out dfMin, out dfMax), "The blobs are not the same.");
                m_log.CHECK_EQ(dfMin, 0, "The min value is not as expected.");
                m_log.CHECK_EQ(dfMax, 0, "The max value is not as expected.");

                blob1 = net.FindBlob("x_categorical_future");
                m_log.CHECK(blob1 != null, "Could not find the blob 'x_categorical_future'!");
                m_log.CHECK(blob1.CompareShape(new List<int>() { 256, 30, 7 }), "The blob shape is different than expected");

                blobVal.LoadFromNumpy(strPath + "0_future_ts_categorical.npy");
                m_log.CHECK(blobVal.CompareEx(blob1, blobWork, out dfMin, out dfMax), "The blobs are not the same.");
                m_log.CHECK_EQ(dfMin, 0, "The min value is not as expected.");
                m_log.CHECK_EQ(dfMax, 0, "The max value is not as expected.");

                blob1 = net.FindBlob("target");
                m_log.CHECK(blob1 != null, "Could not find the blob 'target'!");
                m_log.CHECK(blob1.CompareShape(new List<int>() { 256, 30 }), "The blob shape is different than expected");

                blobVal.LoadFromNumpy(strPath + "0_target.npy");
                m_log.CHECK(blobVal.CompareEx(blob1, blobWork, out dfMin, out dfMax), "The blobs are not the same.");
                m_log.CHECK_EQ(dfMin, 0, "The min value is not as expected.");
                m_log.CHECK_EQ(dfMax, 0, "The max value is not as expected.");
            }
            finally
            {
                evtCancel.Set();
                Thread.Sleep(1000);
                
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
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
                Tuple<List<float[]>, int[], List<string>> data = Blob<T>.LoadFromNumpyEx(strFile, null, int.MaxValue, nStartIdx, nCount);

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
