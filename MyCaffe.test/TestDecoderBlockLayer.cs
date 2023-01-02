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
    public class TestDecoderBlockLayer
    {
        [TestMethod]
        public void TestForward()
        {
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
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
        public void TestForwardCuda()
        {
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
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
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    t.TestBackward(3, 8, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardCuda()
        {
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    t.TestBackward(3, 8, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IDecoderBlockLayerTest : ITest
    {
        void TestForward(int nBatch, int nHeads, bool bEnableCudaImpl);
        void TestBackward(int nBatch, int nHeads, bool bEnableCudaImpl);
    }

    class DecoderBlockLayerTest : TestBase
    {
        public DecoderBlockLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Causal Self Attention Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new DecoderBlockLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new DecoderBlockLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class DecoderBlockLayerTest2<T> : TestEx<T>, IDecoderBlockLayerTest
    {
        Blob<T> m_blobX;
        Blob<T> m_blobEncOut;
        Blob<T> m_blobMaskEnc;
        Blob<T> m_blobMaskDec;
        Blob<T> m_blobInput;
        Blob<T> m_blobYexp;
        Blob<T> m_blobWork;
        Stopwatch m_swUpdateTimer = new Stopwatch();
        double m_dfLastProgress = 0;
        AutoResetEvent m_evtDownloadDone = new AutoResetEvent(false);

        public DecoderBlockLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobX = new Blob<T>(m_cuda, m_log);
            m_blobEncOut = new Blob<T>(m_cuda, m_log);
            m_blobMaskEnc = new Blob<T>(m_cuda, m_log);
            m_blobMaskDec = new Blob<T>(m_cuda, m_log);
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
            dispose(ref m_blobEncOut);
            dispose(ref m_blobMaskEnc);
            dispose(ref m_blobMaskDec);
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
            string strTestDataPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\dec\\";
            if (!Directory.Exists(strTestDataPath))
                Directory.CreateDirectory(strTestDataPath);

            string strTestDataFile = strTestDataPath + "_decoder_test.zip";
            if (!File.Exists(strTestDataFile))
            {
                using (WebClient webClient = new WebClient())
                {
                    string strUrl = "https://signalpopcdn.blob.core.windows.net/mycaffesupport/_decoder_test.zip";
                    string strFile1 = "_decoder_test.zip";
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

        private void verify(Blob<T> b1, Blob<T> b1exp, bool bCompareDiff, double fErr = 1e-06)
        {
            float[] rgExpected = (bCompareDiff) ? convertF(b1exp.mutable_cpu_diff) : convertF(b1exp.mutable_cpu_data);
            float[] rgActual = (bCompareDiff) ? convertF(b1.mutable_cpu_diff) : convertF(b1.mutable_cpu_data);

            for (int i = 0; i < rgExpected.Length; i++)
            {
                float fExpected = rgExpected[i];
                float fActual = rgActual[i];

                m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fErr, "The values are not as expected!");
            }
            
            bool bRes = b1.Compare(b1exp, m_blobWork, bCompareDiff, fErr);
            if (!bRes)
                m_log.FAIL("The blobs are not equal!");
        }

        private void createDecoderMask(Blob<T> blobMask)
        {
            int nBatch = blobMask.num;
            int nDim = blobMask.count(1);

            float[] rgMask = convertF(blobMask.mutable_cpu_data);

            blobMask.Reshape(nBatch, nDim, nDim, 1);
            float[] rgMask1 = new float[nBatch * nDim * nDim];

            for (int n = 0; n < nBatch; n++)
            {
                for (int i = 0; i < nDim; i++)
                {
                    for (int j = 0; j < nDim; j++)
                    {
                        int nIdxSrc = n * nDim + j;
                        float fVal = rgMask[nIdxSrc];

                        if (fVal > 0 && j > i)
                            fVal = 0;

                        int nIdxDst = n * nDim * nDim + i * nDim + j;
                        rgMask1[nIdxDst] = fVal;
                    }
                }
            }

            blobMask.mutable_cpu_data = convert(rgMask1);
        }

        /// <summary>
        /// WORK IN PROGRESS
        /// </summary>
        public void TestForward(int nBatch, int nHeads, bool bEnableCudaImpl)
        {
            string strTestDataPath = "C:\\temp\\projects\\TransformerTranslator\\TransformerTranslator\\test\\"; // loadTestData();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.DECODER;
            p.transformer_block_param.heads = nHeads;
            p.transformer_block_param.embed = 512;
            p.transformer_block_param.block_size = 200;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            p.transformer_block_param.enable_layernorm_cuda_impl = bEnableCudaImpl;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.TRANSFORMER_BLOCK, "The layer type is incorrect!");

                m_blobX.LoadFromNumpy(strTestDataPath + "dec_in_x0.npy");
                m_blobX.Name = "dec_in_x0";
                m_blobEncOut.LoadFromNumpy(strTestDataPath + "enc_out_x1.npy");
                m_blobEncOut.Name = "enc_out_x1";
                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");
                m_blobMaskDec.ReshapeLike(m_blobInput);
                m_blobMaskDec.Name = "d_mask";
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMaskDec.mutable_gpu_data);
                createDecoderMask(m_blobMaskDec);
                m_blobMaskEnc.ReshapeLike(m_blobEncOut);
                m_blobMaskEnc.Name = "e_mask";
                m_cuda.sign(m_blobEncOut.count(), m_blobEncOut.gpu_data, m_blobMaskEnc.mutable_gpu_data);

                BottomVec.Clear();
                BottomVec.Add(m_blobX);
                BottomVec.Add(m_blobMaskDec);
                BottomVec.Add(m_blobEncOut);
                BottomVec.Add(m_blobMaskEnc);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strTestDataPath + "mh1.w_q_weight.npy");    // multi-head query weight
                layer.blobs[1].LoadFromNumpy(strTestDataPath + "mh1.w_q_bias.npy");      // multi-head query bias
                layer.blobs[2].LoadFromNumpy(strTestDataPath + "mh1.w_k_weight.npy");    // multi-head key weight
                layer.blobs[3].LoadFromNumpy(strTestDataPath + "mh1.w_k_bias.npy");      // multi-head key bias
                layer.blobs[4].LoadFromNumpy(strTestDataPath + "mh1.w_v_weight.npy");    // multi-head value weight
                layer.blobs[5].LoadFromNumpy(strTestDataPath + "mh1.w_v_bias.npy");      // multi-head value bias
                layer.blobs[6].LoadFromNumpy(strTestDataPath + "mh1.w_o_weight.npy");    // multi-head output weight
                layer.blobs[7].LoadFromNumpy(strTestDataPath + "mh1.w_o_bias.npy");      // multi-head output bias

                layer.blobs[8].LoadFromNumpy(strTestDataPath + "mh2.w_q_weight.npy");    // multi-head query weight
                layer.blobs[9].LoadFromNumpy(strTestDataPath + "mh2.w_q_bias.npy");      // multi-head query bias
                layer.blobs[10].LoadFromNumpy(strTestDataPath + "mh2.w_k_weight.npy");    // multi-head key weight
                layer.blobs[11].LoadFromNumpy(strTestDataPath + "mh2.w_k_bias.npy");      // multi-head key bias
                layer.blobs[12].LoadFromNumpy(strTestDataPath + "mh2.w_v_weight.npy");    // multi-head value weight
                layer.blobs[13].LoadFromNumpy(strTestDataPath + "mh2.w_v_bias.npy");      // multi-head value bias
                layer.blobs[14].LoadFromNumpy(strTestDataPath + "mh2.w_o_weight.npy");    // multi-head output weight
                layer.blobs[15].LoadFromNumpy(strTestDataPath + "mh2.w_o_bias.npy");      // multi-head output bias

                layer.blobs[16].LoadFromNumpy(strTestDataPath + "ff.w_1_weight.npy");    // fc
                layer.blobs[17].LoadFromNumpy(strTestDataPath + "ff.w_1_bias.npy");      // fc
                layer.blobs[18].LoadFromNumpy(strTestDataPath + "ff.w_2_weight.npy");   // proj
                layer.blobs[19].LoadFromNumpy(strTestDataPath + "ff.w_2_bias.npy");     // proj

                layer.Forward(BottomVec, TopVec);

                // Now, check values
                m_blobYexp.LoadFromNumpy(strTestDataPath + "dec.12_output.npy");
                verify(TopVec[0], m_blobYexp, false, 3e-06);

                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (int i = 0; i < 100; i++)
                {
                    layer.Forward(BottomVec, TopVec);
                }

                sw.Stop();
                double dfTime = sw.Elapsed.TotalMilliseconds / 100;
                Trace.WriteLine("Decoder Forward time = " + dfTime.ToString("N6") + " ms.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        /// <summary>
        /// WORK IN PROGRESS
        /// </summary>
        public void TestBackward(int nBatch, int nHeads, bool bEnableCudaImpl)
        {
            string strTestDataPath = "C:\\temp\\projects\\TransformerTranslator\\TransformerTranslator\\test\\"; // loadTestData();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.DECODER;
            p.transformer_block_param.heads = nHeads;
            p.transformer_block_param.embed = 512;
            p.transformer_block_param.block_size = 200;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            p.transformer_block_param.enable_layernorm_cuda_impl = bEnableCudaImpl;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.TRANSFORMER_BLOCK, "The layer type is incorrect!");

                m_blobX.LoadFromNumpy(strTestDataPath + "dec_in_x0.npy");
                m_blobX.Name = "dec_in_x0";
                m_blobEncOut.LoadFromNumpy(strTestDataPath + "enc_out_x1.npy");
                m_blobEncOut.Name = "enc_out_x1";
                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");
                m_blobMaskDec.ReshapeLike(m_blobInput);
                m_blobMaskDec.Name = "d_mask";
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMaskDec.mutable_gpu_data);
                createDecoderMask(m_blobMaskDec);
                m_blobMaskEnc.ReshapeLike(m_blobEncOut);
                m_blobMaskEnc.Name = "e_mask";
                m_cuda.sign(m_blobEncOut.count(), m_blobEncOut.gpu_data, m_blobMaskEnc.mutable_gpu_data);

                BottomVec.Clear();
                BottomVec.Add(m_blobX);
                BottomVec.Add(m_blobMaskDec);
                BottomVec.Add(m_blobEncOut);
                BottomVec.Add(m_blobMaskEnc);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strTestDataPath + "mh1.w_q_weight.npy");    // multi-head query weight
                layer.blobs[1].LoadFromNumpy(strTestDataPath + "mh1.w_q_bias.npy");      // multi-head query bias
                layer.blobs[2].LoadFromNumpy(strTestDataPath + "mh1.w_k_weight.npy");    // multi-head key weight
                layer.blobs[3].LoadFromNumpy(strTestDataPath + "mh1.w_k_bias.npy");      // multi-head key bias
                layer.blobs[4].LoadFromNumpy(strTestDataPath + "mh1.w_v_weight.npy");    // multi-head value weight
                layer.blobs[5].LoadFromNumpy(strTestDataPath + "mh1.w_v_bias.npy");      // multi-head value bias
                layer.blobs[6].LoadFromNumpy(strTestDataPath + "mh1.w_o_weight.npy");    // multi-head output weight
                layer.blobs[7].LoadFromNumpy(strTestDataPath + "mh1.w_o_bias.npy");      // multi-head output bias

                layer.blobs[8].LoadFromNumpy(strTestDataPath + "mh2.w_q_weight.npy");    // multi-head query weight
                layer.blobs[9].LoadFromNumpy(strTestDataPath + "mh2.w_q_bias.npy");      // multi-head query bias
                layer.blobs[10].LoadFromNumpy(strTestDataPath + "mh2.w_k_weight.npy");    // multi-head key weight
                layer.blobs[11].LoadFromNumpy(strTestDataPath + "mh2.w_k_bias.npy");      // multi-head key bias
                layer.blobs[12].LoadFromNumpy(strTestDataPath + "mh2.w_v_weight.npy");    // multi-head value weight
                layer.blobs[13].LoadFromNumpy(strTestDataPath + "mh2.w_v_bias.npy");      // multi-head value bias
                layer.blobs[14].LoadFromNumpy(strTestDataPath + "mh2.w_o_weight.npy");    // multi-head output weight
                layer.blobs[15].LoadFromNumpy(strTestDataPath + "mh2.w_o_bias.npy");      // multi-head output bias

                layer.blobs[16].LoadFromNumpy(strTestDataPath + "ff.w_1_weight.npy");    // fc
                layer.blobs[17].LoadFromNumpy(strTestDataPath + "ff.w_1_bias.npy");      // fc
                layer.blobs[18].LoadFromNumpy(strTestDataPath + "ff.w_2_weight.npy");   // proj
                layer.blobs[19].LoadFromNumpy(strTestDataPath + "ff.w_2_bias.npy");     // proj

                layer.Forward(BottomVec, TopVec);

                // Now, check values from forward
                m_blobYexp.LoadFromNumpy(strTestDataPath + "dec.12_output.npy");
                verify(TopVec[0], m_blobYexp, false, 3e-06);

                // Load the inbound gradients.
                TopVec[0].LoadFromNumpy(strTestDataPath + "grad_dec.12_output.npy", true);

                List<bool> rgProp = new List<bool>() { true };
                layer.Backward(TopVec, rgProp, BottomVec);

                // Now, check values form backward
                m_blobYexp.LoadFromNumpy(strTestDataPath + "grad_dec.1_x.npy", true);
                verify(BottomVec[0], m_blobYexp, true, 1e-08);

                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (int i = 0; i < 100; i++)
                {
                    layer.Backward(TopVec, rgProp, BottomVec);
                }

                sw.Stop();
                double dfTime = sw.Elapsed.TotalMilliseconds / 100;
                Trace.WriteLine("Decoder Backward time = " + dfTime.ToString("N6") + " ms.");
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
