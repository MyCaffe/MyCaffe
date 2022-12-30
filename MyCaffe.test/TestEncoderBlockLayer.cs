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
    public class TestEncoderBlockLayer
    {
        [TestMethod]
        public void TestForward()
        {
            EncoderBlockLayerTest test = new EncoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IEncoderBlockLayerTest t in test.Tests)
                {
                    t.TestForward(3, 8, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardLnCuda()
        {
            EncoderBlockLayerTest test = new EncoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IEncoderBlockLayerTest t in test.Tests)
                {
                    t.TestForward(3, 8, true);
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
            EncoderBlockLayerTest test = new EncoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IEncoderBlockLayerTest t in test.Tests)
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
            EncoderBlockLayerTest test = new EncoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IEncoderBlockLayerTest t in test.Tests)
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

    interface IEncoderBlockLayerTest : ITest
    {
        void TestForward(int nBatch, int nHeads, bool bEnableLnCudaImpl);
        void TestBackward(int nBatch, int nHeads);
        void TestGradient(int nBatch, int nHeads);
    }

    class EncoderBlockLayerTest : TestBase
    {
        public EncoderBlockLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Causal Self Attention Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new EncoderBlockLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new EncoderBlockLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class EncoderBlockLayerTest2<T> : TestEx<T>, IEncoderBlockLayerTest
    {
        Blob<T> m_blobX;
        Blob<T> m_blobMask;
        Blob<T> m_blobInput;
        Blob<T> m_blobYexp;
        Blob<T> m_blobWork;
        Stopwatch m_swUpdateTimer = new Stopwatch();
        double m_dfLastProgress = 0;
        AutoResetEvent m_evtDownloadDone = new AutoResetEvent(false);

        public EncoderBlockLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobX = new Blob<T>(m_cuda, m_log);
            m_blobMask = new Blob<T>(m_cuda, m_log);
            m_blobInput = new Blob<T>(m_cuda, m_log);
            m_blobYexp = new Blob<T>(m_cuda, m_log);
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
            dispose(ref m_blobX);
            dispose(ref m_blobMask);
            dispose(ref m_blobInput);
            dispose(ref m_blobYexp);
            dispose(ref m_blobWork);
            base.dispose();
        }

        private string loadTestData()
        {
            string strTestDataFile = downloadTestData();
            string strPath = Path.GetDirectoryName(strTestDataFile);
            
            if (!File.Exists(strPath + "\\test\\13_out1.npy"))
                ZipFile.ExtractToDirectory(strTestDataFile, strPath);

            return strPath + "\\test\\";
        }

        private string downloadTestData()
        {
            string strTestDataPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\enc\\";
            if (!Directory.Exists(strTestDataPath))
                Directory.CreateDirectory(strTestDataPath);

            string strTestDataFile = strTestDataPath + "_encoder_test.zip";
            if (!File.Exists(strTestDataFile))
            {
                using (WebClient webClient = new WebClient())
                {
                    string strUrl = "https://signalpopcdn.blob.core.windows.net/mycaffesupport/_encoder_test.zip";
                    string strFile1 = "_encoder_test.zip";
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

        public void TestForward(int nBatch, int nHeads, bool bEnableLnCudaImpl)
        {
            string strTestDataPath = loadTestData();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.ENCODER;
            p.transformer_block_param.heads = nHeads;
            p.transformer_block_param.embed = 512;
            p.transformer_block_param.block_size = 200;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            p.transformer_block_param.enable_layernorm_cuda_impl = bEnableLnCudaImpl;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.TRANSFORMER_BLOCK, "The layer type is incorrect!");

                m_blobX.LoadFromNumpy(strTestDataPath + "enc_x0.npy");
                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");
                m_blobMask.ReshapeLike(m_blobInput);
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMask.mutable_gpu_data);
                
                BottomVec.Clear();
                BottomVec.Add(m_blobX);
                BottomVec.Add(m_blobMask);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strTestDataPath + "mh.w_q_weight.npy");    // multi-head query weight
                layer.blobs[1].LoadFromNumpy(strTestDataPath + "mh.w_q_bias.npy");      // multi-head query bias
                layer.blobs[2].LoadFromNumpy(strTestDataPath + "mh.w_k_weight.npy");    // multi-head key weight
                layer.blobs[3].LoadFromNumpy(strTestDataPath + "mh.w_k_bias.npy");      // multi-head key bias
                layer.blobs[4].LoadFromNumpy(strTestDataPath + "mh.w_v_weight.npy");    // multi-head value weight
                layer.blobs[5].LoadFromNumpy(strTestDataPath + "mh.w_v_bias.npy");      // multi-head value bias
                layer.blobs[6].LoadFromNumpy(strTestDataPath + "mh.w_o_weight.npy");    // multi-head output weight
                layer.blobs[7].LoadFromNumpy(strTestDataPath + "mh.w_o_bias.npy");      // multi-head output bias

                layer.blobs[8].LoadFromNumpy(strTestDataPath + "ff.w_1_weight.npy");    // fc
                layer.blobs[9].LoadFromNumpy(strTestDataPath + "ff.w_1_bias.npy");      // fc
                layer.blobs[10].LoadFromNumpy(strTestDataPath + "ff.w_2_weight.npy");   // proj
                layer.blobs[11].LoadFromNumpy(strTestDataPath + "ff.w_2_bias.npy");     // proj

                layer.Forward(BottomVec, TopVec);

                // Now, check values
                m_blobYexp.LoadFromNumpy("C:\\temp\\projects\\TransformerTranslator\\TransformerTranslator\\test\\" + "enc.12_output.npy");
                verify(TopVec[0], m_blobYexp, false);

                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (int i = 0; i < 100; i++)
                {
                    layer.Forward(BottomVec, TopVec);
                }

                sw.Stop();
                double dfTime = sw.Elapsed.TotalMilliseconds / 100;
                Trace.WriteLine("Encoder Forward time = " + dfTime.ToString("N6") + " ms.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        /// <summary>
        /// WORK IN PROGRESS
        /// </summary>
        public void TestBackward(int nBatch, int nHeads)
        {
            string strTestDataPath = loadTestData();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.ENCODER;
            p.transformer_block_param.heads = nHeads;
            p.transformer_block_param.embed = 512;
            p.transformer_block_param.block_size = 200;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.TRANSFORMER_BLOCK, "The layer type is incorrect!");

                m_blobX.LoadFromNumpy(strTestDataPath + "enc_x0.npy");
                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");
                m_blobMask.ReshapeLike(m_blobInput);
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMask.mutable_gpu_data);

                BottomVec.Clear();
                BottomVec.Add(m_blobX);
                BottomVec.Add(m_blobMask);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strTestDataPath + "mh.w_q_weight.npy");    // multi-head query weight
                layer.blobs[1].LoadFromNumpy(strTestDataPath + "mh.w_q_bias.npy");      // multi-head query bias
                layer.blobs[2].LoadFromNumpy(strTestDataPath + "mh.w_k_weight.npy");    // multi-head key weight
                layer.blobs[3].LoadFromNumpy(strTestDataPath + "mh.w_k_bias.npy");      // multi-head key bias
                layer.blobs[4].LoadFromNumpy(strTestDataPath + "mh.w_v_weight.npy");    // multi-head value weight
                layer.blobs[5].LoadFromNumpy(strTestDataPath + "mh.w_v_bias.npy");      // multi-head value bias
                layer.blobs[6].LoadFromNumpy(strTestDataPath + "mh.w_o_weight.npy");    // multi-head output weight
                layer.blobs[7].LoadFromNumpy(strTestDataPath + "mh.w_o_bias.npy");      // multi-head output bias

                layer.blobs[8].LoadFromNumpy(strTestDataPath + "ff.w_1_weight.npy");    // fc
                layer.blobs[9].LoadFromNumpy(strTestDataPath + "ff.w_1_bias.npy");      // fc
                layer.blobs[10].LoadFromNumpy(strTestDataPath + "ff.w_2_weight.npy");   // proj
                layer.blobs[11].LoadFromNumpy(strTestDataPath + "ff.w_2_bias.npy");     // proj

                layer.Forward(BottomVec, TopVec);

                m_blobYexp.LoadFromNumpy(strTestDataPath + "enc_x1.npy");
                
                // Now, check values from forward
                verify(TopVec[0], m_blobYexp, false);

                // Load the inbound gradients.
                TopVec[0].LoadFromNumpy(strTestDataPath + "grad_enc.7_xC.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                m_blobYexp.LoadFromNumpy(strTestDataPath + "grad_enc_1_x.npy", true);

                // Now, check values form backward
                verify(BottomVec[0], m_blobYexp, true);
            }
            finally
            {
                layer.Dispose();
            }
        }

        /// <summary>
        /// WORK IN PROGRESS
        /// </summary>
        public void TestGradient(int nBatch, int nHeads)
        {
            string strTestDataPath = loadTestData();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.ENCODER;
            p.transformer_block_param.heads = nHeads;
            p.transformer_block_param.embed = 512;
            p.transformer_block_param.block_size = 200;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION, "The layer type is incorrect!");

                m_blobX.LoadFromNumpy(strTestDataPath + "enc_x0.npy");
                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");
                m_blobMask.ReshapeLike(m_blobInput);
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMask.mutable_gpu_data);

                BottomVec.Clear();
                BottomVec.Add(m_blobX);
                BottomVec.Add(m_blobMask);

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
