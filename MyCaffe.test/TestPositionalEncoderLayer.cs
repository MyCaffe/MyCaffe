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
using System.Diagnostics;
using System.IO;
using MyCaffe.param.gpt;
using System.Net;
using System.Threading;
using System.IO.Compression;

namespace MyCaffe.test
{
    [TestClass]
    public class TestPositionalEncoderLayer
    {
        [TestMethod]
        public void TestForward()
        {
            PositionalEncoderLayerTest test = new PositionalEncoderLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPositionalEncoderLayerTest t in test.Tests)
                {
                    t.TestForward(3, 8);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackward()
        {
            PositionalEncoderLayerTest test = new PositionalEncoderLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPositionalEncoderLayerTest t in test.Tests)
                {
                    t.TestBackward(3, 8);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient()
        {
            PositionalEncoderLayerTest test = new PositionalEncoderLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IPositionalEncoderLayerTest t in test.Tests)
                {
                    t.TestGradient(3, 8);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IPositionalEncoderLayerTest : ITest
    {
        void TestForward(int nBatch, int nHeads);
        void TestBackward(int nBatch, int nHeads);
        void TestGradient(int nBatch, int nHeads);
    }

    class PositionalEncoderLayerTest : TestBase
    {
        public PositionalEncoderLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Positional Encoder Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PositionalEncoderLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new PositionalEncoderLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class PositionalEncoderLayerTest2<T> : TestEx<T>, IPositionalEncoderLayerTest
    {
        Blob<T> m_blobY;
        Blob<T> m_blobQ;
        Blob<T> m_blobWork;
        Stopwatch m_swUpdateTimer = new Stopwatch();
        double m_dfLastProgress = 0;
        AutoResetEvent m_evtDownloadDone = new AutoResetEvent(false);

        public PositionalEncoderLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobY = new Blob<T>(m_cuda, m_log);
            m_blobQ = new Blob<T>(m_cuda, m_log);
            m_blobWork = new Blob<T>(m_cuda, m_log);
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        private void dispose1(ref Blob<T> b)
        {
            if (b != null)
            {
                b.Dispose();
                b = null;
            }
        }

        protected override void dispose()
        {
            dispose(ref m_blobY);
            dispose(ref m_blobQ);
            dispose(ref m_blobWork);
            base.dispose();
        }

        private string loadTestData()
        {
            string strTestDataFile = downloadTestData();
            string strPath = Path.GetDirectoryName(strTestDataFile);
            
            if (!File.Exists(strPath + "\\test\\pos.output.npy"))
                ZipFile.ExtractToDirectory(strTestDataFile, strPath);

            return strPath + "\\test\\";
        }

        private string downloadTestData()
        {
            string strTestDataPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\pos\\";
            if (!Directory.Exists(strTestDataPath))
                Directory.CreateDirectory(strTestDataPath);

            string strTestDataFile = strTestDataPath + "_posenc_test.zip";
            if (!File.Exists(strTestDataFile))
            {
                using (WebClient webClient = new WebClient())
                {
                    string strUrl = "https://signalpopcdn.blob.core.windows.net/mycaffesupport/_posenc_test.zip";
                    string strFile1 = "_posenc_test.zip";
                    string strFile = strTestDataPath + strFile1;

                    m_swUpdateTimer.Start();
                    m_dfLastProgress = 0;

                    webClient.DownloadProgressChanged += WebClient_DownloadProgressChanged;
                    webClient.DownloadFileCompleted += WebClient_DownloadFileCompleted;
                    webClient.DownloadFileAsync(new Uri(strUrl), strFile, strFile1);

                    m_evtDownloadDone.WaitOne();
                }
            }

            return strTestDataFile;
        }

        private void WebClient_DownloadFileCompleted(object sender, System.ComponentModel.AsyncCompletedEventArgs e)
        {
            bool bTraceEnabled = m_log.EnableTrace;
            m_log.EnableTrace = true;
            m_log.WriteLine("Downloading done.");
            m_log.EnableTrace = bTraceEnabled;

            m_evtDownloadDone.Set();
        }

        private void WebClient_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            if (m_swUpdateTimer.Elapsed.TotalMilliseconds >= 1000)
            {
                if (m_dfLastProgress != e.ProgressPercentage)
                {
                    m_dfLastProgress = e.ProgressPercentage;
                    string strFile = e.UserState.ToString();
                    bool bTraceEnabled = m_log.EnableTrace;
                    m_log.EnableTrace = true;

                    m_log.Progress = e.ProgressPercentage / 100.0;
                    m_log.WriteLine("Downloading '" + strFile + "' at " + m_log.Progress.ToString("P") + "...");
                    m_log.EnableTrace = bTraceEnabled;
                }

                m_swUpdateTimer.Restart();
            }
        }

        private void verify(Blob<T> b1, Blob<T> b1exp, bool bCompareDiff)
        {
            float[] rgExpected = (bCompareDiff) ? convertF(b1exp.mutable_cpu_diff) : convertF(b1exp.mutable_cpu_data);
            float[] rgActual = (bCompareDiff) ? convertF(b1.mutable_cpu_diff) : convertF(b1.mutable_cpu_data);

            for (int i = 0; i < rgExpected.Length; i++)
            {
                float fExpected = rgExpected[i];
                float fActual = rgActual[i];
                float fErr = 1e-6f;

                m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fErr, "The values are not as expected!");
            }

            bool bRes = b1.Compare(b1exp, m_blobWork, bCompareDiff, 1e-6f);
            if (!bRes)
                m_log.FAIL("The blobs are not equal!");
        }

        public void TestForward(int nBatch, int nHeads)
        {
            string strTestDataPath = loadTestData();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODER);
            p.positional_encoder_param.embed = 512;
            p.positional_encoder_param.block_size = 200;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.POSITIONAL_ENCODER, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataPath + "q0.npy");
                
                BottomVec.Clear();
                BottomVec.Add(m_blobQ);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_blobY.LoadFromNumpy(strTestDataPath + "pos.output.npy");

                // Now, check values
                verify(TopVec[0], m_blobY, false);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackward(int nBatch, int nHeads)
        {
            string strTestDataPath = loadTestData();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODER);
            p.positional_encoder_param.embed = 512;
            p.positional_encoder_param.block_size = 200;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.POSITIONAL_ENCODER, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataPath + "q0.npy");

                BottomVec.Clear();
                BottomVec.Add(m_blobQ);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // Load the inbound gradients.
                TopVec[0].LoadFromNumpy(strTestDataPath + "grad_pos.output.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                m_blobQ.LoadFromNumpy(strTestDataPath + "grad_pos.1_x.npy", true);

                // Now, check values
                verify(BottomVec[0], m_blobQ, true);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient(int nBatch, int nHeads)
        {
            string strTestDataPath = loadTestData();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODER);
            p.positional_encoder_param.embed = 512;
            p.positional_encoder_param.block_size = 200;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.POSITIONAL_ENCODER, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataPath + "q0.npy");

                BottomVec.Clear();
                BottomVec.Add(m_blobQ);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 0.01, 0.0001);
                checker.CheckGradient(layer, BottomVec, TopVec, -1, 100);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
