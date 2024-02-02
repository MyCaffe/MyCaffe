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
using System.Collections.Concurrent;
using System.Text.RegularExpressions;

namespace MyCaffe.test
{
    [TestClass]
    public class TestGPT_TestMultiheadAttentionLayer
    {
        [TestMethod]
        public void TestForward()
        {
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
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
        public void TestForwardWithFlash()
        {
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
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
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
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
        public void TestBackwardWithFlash()
        {
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
                {
                    t.TestBackward(3, 8, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackward2()
        {
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
                {
                    t.TestBackward2(3, 8, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientEncoderDecoder()
        {
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
                {
                    t.TestGradient(3, 8, MultiheadAttentionParameter.WEIGHT_INIT.ENCODER_DECODER, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientGpt()
        {
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
                {
                    t.TestGradient(3, 8, MultiheadAttentionParameter.WEIGHT_INIT.GPT, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFlashAttentionForward()
        {
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
                {
                    t.TestFlashAttentionForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestFlashAttentionBackward()
        {
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
                {
                    t.TestFlashAttentionBackward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IMultiheadAttentionLayerTest : ITest
    {
        void TestForward(uint nBatch, uint nHeads, bool bEnableFlash);
        void TestBackward(uint nBatch, uint nHeads, bool bEnableFlash);
        void TestBackward2(uint nBatch, uint nHeads, bool bEnableFlash);
        void TestGradient(uint nBatch, uint nHeads, MultiheadAttentionParameter.WEIGHT_INIT init, bool bEnableFlash);
        void TestFlashAttentionForward();
        void TestFlashAttentionBackward();
    }

    class MultiheadAttentionLayerTest : TestBase
    {
        public MultiheadAttentionLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Multihead Attention Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MultiheadAttentionLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new MultiheadAttentionLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class MultiheadAttentionLayerTest2<T> : TestEx<T>, IMultiheadAttentionLayerTest
    {
        Blob<T> m_blobY;
        Blob<T> m_blobQ;
        Blob<T> m_blobK;
        Blob<T> m_blobV;
        Blob<T> m_blobMask;
        Blob<T> m_blobInput;
        Blob<T> m_blobQexp;
        Blob<T> m_blobKexp;
        Blob<T> m_blobVexp;
        BlobCollection<T> m_colTop;
        BlobCollection<T> m_colBtm;

        Blob<T> m_blobKt1;
        Blob<T> m_blobAttA;
        Blob<T> m_blobAttB;
        Layer<T> m_softmax;
        Layer<T> m_transposeQ;
        Layer<T> m_attn_dropout;


        public MultiheadAttentionLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobQ = new Blob<T>(m_cuda, m_log);
            m_blobK = new Blob<T>(m_cuda, m_log);
            m_blobV = new Blob<T>(m_cuda, m_log);
            m_blobMask = new Blob<T>(m_cuda, m_log);
            m_blobInput = new Blob<T>(m_cuda, m_log);
            m_blobY = new Blob<T>(m_cuda, m_log);
            m_blobQexp = new Blob<T>(m_cuda, m_log);
            m_blobKexp = new Blob<T>(m_cuda, m_log);
            m_blobVexp = new Blob<T>(m_cuda, m_log);
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
            dispose(ref m_blobQ);
            dispose(ref m_blobK);
            dispose(ref m_blobV);
            dispose(ref m_blobMask);
            dispose(ref m_blobInput);
            dispose(ref m_blobQexp);
            dispose(ref m_blobKexp);
            dispose(ref m_blobVexp);
            dispose(ref m_blobY);
            base.dispose();
        }

        private string getTestDataPath(string strSubPath, string strFile)
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\gpt\\test\\" + strSubPath + "\\iter_0\\";

            if (!File.Exists(strPath + strFile))
                throw new Exception("Could not find the test data file '" + strPath + strFile + "'.  You may need to run the 'Test|Download Test Data | GPT' menu item.");

            return strPath;
        }

        private string getTestDataBasePath(string strFile)
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\gpt\\test\\";

            if (!File.Exists(strPath + strFile))
                throw new Exception("Could not find the test data file '" + strPath + strFile + "'.  You may need to run the 'Test|Download Test Data | GPT' menu item.");

            return strPath;
        }

        public void TestForward(uint nBatch, uint nHeads, bool bEnableFlash)
        {
            string strTestDataBasePath = getTestDataBasePath("q0.npy");
            string strTestDataPath = getTestDataPath("mha", "15_loss.npy");

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION, "mh", Phase.TEST);
            p.multihead_attention_param.heads = nHeads;
            p.multihead_attention_param.embed = 512;
            p.multihead_attention_param.block_size = 200;
            p.multihead_attention_param.attn_dropout = 0.0;
            p.multihead_attention_param.resid_dropout = 0.0;
            p.multihead_attention_param.enable_flash_scaled_dot_product_attention = bEnableFlash;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataBasePath + "q0.npy");
                m_blobK.LoadFromNumpy(strTestDataBasePath + "k0.npy");
                m_blobV.LoadFromNumpy(strTestDataBasePath + "v0.npy");
                m_blobInput.LoadFromNumpy(strTestDataBasePath + "src_input.npy");
                m_blobMask.ReshapeLike(m_blobInput);
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMask.mutable_gpu_data);
                
                BottomVec.Clear();
                BottomVec.Add(m_blobQ);
                BottomVec.Add(m_blobK);
                BottomVec.Add(m_blobV);
                BottomVec.Add(m_blobMask);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strTestDataPath + "mh.w_q.weight.npy");
                layer.blobs[1].LoadFromNumpy(strTestDataPath + "mh.w_q.bias.npy");
                layer.blobs[2].LoadFromNumpy(strTestDataPath + "mh.w_k.weight.npy");
                layer.blobs[3].LoadFromNumpy(strTestDataPath + "mh.w_k.bias.npy");
                layer.blobs[4].LoadFromNumpy(strTestDataPath + "mh.w_v.weight.npy");
                layer.blobs[5].LoadFromNumpy(strTestDataPath + "mh.w_v.bias.npy");
                layer.blobs[6].LoadFromNumpy(strTestDataPath + "mh.w_o.weight.npy");
                layer.blobs[7].LoadFromNumpy(strTestDataPath + "mh.w_o.bias.npy");

                layer.Forward(BottomVec, TopVec);

                m_blobY.LoadFromNumpy(strTestDataPath + "mh.12_output.npy");
                m_log.CHECK(m_blobY.Compare(TopVec[0], m_blobWork, false, (typeof(T) == typeof(float)) ? 1e-12 : 3e-06), "The blobs are different.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackward(uint nBatch, uint nHeads, bool bEnableFlash)
        {
            string strTestDataBasePath = getTestDataBasePath("q0.npy");
            string strTestDataPath = getTestDataPath("mha", "15_loss.npy");

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION, "mh", Phase.TRAIN);
            p.multihead_attention_param.heads = nHeads;
            p.multihead_attention_param.embed = 512;
            p.multihead_attention_param.block_size = 200;
            p.multihead_attention_param.attn_dropout = 0.0;
            p.multihead_attention_param.resid_dropout = 0.0;
            p.multihead_attention_param.enable_flash_scaled_dot_product_attention = bEnableFlash;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataBasePath + "q0.npy");
                m_blobK.LoadFromNumpy(strTestDataBasePath + "k0.npy");
                m_blobV.LoadFromNumpy(strTestDataBasePath + "v0.npy");
                m_blobInput.LoadFromNumpy(strTestDataBasePath + "src_input.npy");
                m_blobMask.ReshapeLike(m_blobInput);
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMask.mutable_gpu_data);

                BottomVec.Clear();
                BottomVec.Add(m_blobQ);
                BottomVec.Add(m_blobK);
                BottomVec.Add(m_blobV);
                BottomVec.Add(m_blobMask);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strTestDataPath + "mh.w_q.weight.npy");
                layer.blobs[1].LoadFromNumpy(strTestDataPath + "mh.w_q.bias.npy");
                layer.blobs[2].LoadFromNumpy(strTestDataPath + "mh.w_k.weight.npy");
                layer.blobs[3].LoadFromNumpy(strTestDataPath + "mh.w_k.bias.npy");
                layer.blobs[4].LoadFromNumpy(strTestDataPath + "mh.w_v.weight.npy");
                layer.blobs[5].LoadFromNumpy(strTestDataPath + "mh.w_v.bias.npy");
                layer.blobs[6].LoadFromNumpy(strTestDataPath + "mh.w_o.weight.npy");
                layer.blobs[7].LoadFromNumpy(strTestDataPath + "mh.w_o.bias.npy");

                layer.Forward(BottomVec, TopVec);

                // Load the inbound gradients.
                TopVec[0].LoadFromNumpy(strTestDataPath + "grad_mh.12_output.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                m_blobQexp.LoadFromNumpy(strTestDataPath + "grad_mh.1_q.npy", true);
                m_blobKexp.LoadFromNumpy(strTestDataPath + "grad_mh.1_k.npy", true);
                m_blobVexp.LoadFromNumpy(strTestDataPath + "grad_mh.1_v.npy", true);

                // Now, check values
                m_log.CHECK(m_blobQexp.Compare(BottomVec[0], m_blobWork, true), "The grads are different.");
                m_log.CHECK(m_blobKexp.Compare(BottomVec[1], m_blobWork, true), "The grads are different.");
                m_log.CHECK(m_blobVexp.Compare(BottomVec[2], m_blobWork, true), "The grads are different.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackward2(uint nBatch, uint nHeads, bool bEnableFlash)
        {
            string strTestDataBasePath = getTestDataBasePath("q0.npy");
            string strTestDataPath = getTestDataPath("mha", "15_loss.npy");

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION, "mh", Phase.TRAIN);
            p.multihead_attention_param.heads = nHeads;
            p.multihead_attention_param.embed = 512;
            p.multihead_attention_param.block_size = 200;
            p.multihead_attention_param.attn_dropout = 0.0;
            p.multihead_attention_param.resid_dropout = 0.0;
            p.multihead_attention_param.enable_flash_scaled_dot_product_attention = bEnableFlash;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataBasePath + "q0.npy");
                m_blobK.LoadFromNumpy(strTestDataBasePath + "k0.npy");
                m_blobV.LoadFromNumpy(strTestDataBasePath + "v0.npy");
                m_blobInput.LoadFromNumpy(strTestDataBasePath + "src_input.npy");
                m_blobMask.ReshapeLike(m_blobInput);
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMask.mutable_gpu_data);

                BottomVec.Clear();
                BottomVec.Add(m_blobQ);
                BottomVec.Add(m_blobK);
                BottomVec.Add(m_blobV);
                BottomVec.Add(m_blobMask);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strTestDataPath + "mh.w_q.weight.npy");
                layer.blobs[1].LoadFromNumpy(strTestDataPath + "mh.w_q.bias.npy");
                layer.blobs[2].LoadFromNumpy(strTestDataPath + "mh.w_k.weight.npy");
                layer.blobs[3].LoadFromNumpy(strTestDataPath + "mh.w_k.bias.npy");
                layer.blobs[4].LoadFromNumpy(strTestDataPath + "mh.w_v.weight.npy");
                layer.blobs[5].LoadFromNumpy(strTestDataPath + "mh.w_v.bias.npy");
                layer.blobs[6].LoadFromNumpy(strTestDataPath + "mh.w_o.weight.npy");
                layer.blobs[7].LoadFromNumpy(strTestDataPath + "mh.w_o.bias.npy");

                layer.Forward(BottomVec, TopVec);

                // Load the inbound gradients.
                TopVec[0].LoadFromNumpy(strTestDataPath + "grad_mh.12_output.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                m_blobQexp.LoadFromNumpy(strTestDataPath + "grad_mh.1_q.npy", true);
                m_blobKexp.LoadFromNumpy(strTestDataPath + "grad_mh.1_k.npy", true);
                m_blobVexp.LoadFromNumpy(strTestDataPath + "grad_mh.1_v.npy", true);

                // Now, check values
                m_log.CHECK(m_blobQexp.Compare(BottomVec[0], m_blobWork, true), "The grads are different.");
                m_log.CHECK(m_blobKexp.Compare(BottomVec[1], m_blobWork, true), "The grads are different.");
                m_log.CHECK(m_blobVexp.Compare(BottomVec[2], m_blobWork, true), "The grads are different.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient(uint nBatch, uint nHeads, MultiheadAttentionParameter.WEIGHT_INIT init, bool bEnableFlash)
        {
            string strTestDataBasePath = getTestDataBasePath("q0.npy");
            string strTestDataPath = getTestDataPath("mha", "15_loss.npy");

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION, "mh", Phase.TRAIN);
            p.multihead_attention_param.weight_init = init;
            p.multihead_attention_param.heads = nHeads;
            p.multihead_attention_param.embed = 512;
            p.multihead_attention_param.block_size = 200;
            p.multihead_attention_param.attn_dropout = 0.0;
            p.multihead_attention_param.resid_dropout = 0.0;
            p.multihead_attention_param.enable_flash_scaled_dot_product_attention = bEnableFlash;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataBasePath + "q0.npy");
                m_blobK.LoadFromNumpy(strTestDataBasePath + "k0.npy");
                m_blobV.LoadFromNumpy(strTestDataBasePath + "v0.npy");
                m_blobInput.LoadFromNumpy(strTestDataBasePath + "src_input.npy");
                m_blobMask.ReshapeLike(m_blobInput);
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMask.mutable_gpu_data);

                BottomVec.Clear();
                BottomVec.Add(m_blobQ);
                BottomVec.Add(m_blobK);
                BottomVec.Add(m_blobV);
                BottomVec.Add(m_blobMask);

                double dfPctStep = (typeof(T) == typeof(float)) ? 0.001 : 0.01;
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 0.01, 0.1);
                checker.CheckGradient(layer, BottomVec, TopVec, -1, 1, dfPctStep);
            }
            finally
            {
                layer.Dispose();
            }
        }

        private void addInternal(Blob<T> btm, Blob<T> top)
        {
            m_colBtm.Clear();
            m_colBtm.Add(btm);
            m_colTop.Clear();
            m_colTop.Add(top);
        }

        private void scaled_dot_product_setup(int nBatch, int nHeads, int nBlock, int nSize, float fDropout = 0, Phase phase = Phase.TRAIN)
        {
            LayerParameter transposeQ = new LayerParameter(LayerParameter.LayerType.TRANSPOSE, ".transQ", phase);
            transposeQ.transpose_param.dim[2] = 3;
            transposeQ.transpose_param.dim[3] = 2;
            m_transposeQ = Layer<T>.Create(m_cuda, m_log, transposeQ, null);

            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX, ".softmax", phase);
            softmax.softmax_param.axis = -1;
            softmax.softmax_param.engine = EngineParameter.Engine.CUDNN;
            m_softmax = Layer<T>.Create(m_cuda, m_log, softmax, null);

            if (fDropout > 0)
            {
                LayerParameter dropoutAttn = new LayerParameter(LayerParameter.LayerType.DROPOUT, ".drop.attn", phase);
                dropoutAttn.dropout_param.dropout_ratio = fDropout;
                m_attn_dropout = Layer<T>.Create(m_cuda, m_log, dropoutAttn, null);
            }

            m_blobQ.Reshape(nBatch, nHeads, nBlock, nSize);
            m_blobK.Reshape(nBatch, nHeads, nBlock, nSize);
            m_blobV.Reshape(nBatch, nHeads, nBlock, nSize);
            m_blobY.Reshape(nBatch, nHeads, nBlock, nSize);

            m_blobKt1 = new Blob<T>(m_cuda, m_log);
            m_blobAttA = new Blob<T>(m_cuda, m_log);
            m_blobAttB = new Blob<T>(m_cuda, m_log);
            m_blobKt1.Reshape(nBatch, nHeads, nSize, nBlock);
            m_blobAttA.Reshape(nBatch, nHeads, nBlock, nBlock);
            m_blobAttB.Reshape(nBatch, nHeads, nBlock, nBlock);

            m_colBtm = new BlobCollection<T>();
            m_colTop = new BlobCollection<T>();

            m_filler.Fill(m_blobQ);
            m_filler.Fill(m_blobK);
            m_filler.Fill(m_blobV);
            m_filler.Fill(m_blobY.count(), m_blobY.mutable_gpu_diff);

            m_blobMask.Reshape(nBatch, nBlock, 1, 1);
            m_blobMask.SetData(1.0);

            /// Setup
            addInternal(m_blobK, m_blobKt1);
            m_transposeQ.Setup(m_colBtm, m_colTop);
            addInternal(m_blobAttA, m_blobAttB);
            m_softmax.Setup(m_colBtm, m_colTop);

            if (m_attn_dropout != null)
            {
                addInternal(m_blobAttB, m_blobAttB);
                m_attn_dropout.Setup(m_colBtm, m_colTop);
            }
            m_blobWork.Reshape(m_blobV.num, m_blobV.channels, m_blobV.height, m_blobV.width);

            /// Reshape
            addInternal(m_blobK, m_blobKt1);
            m_transposeQ.Reshape(m_colBtm, m_colTop);
            addInternal(m_blobAttA, m_blobAttB);
            m_softmax.Reshape(m_colBtm, m_colTop);

            if (m_attn_dropout != null)
            {
                addInternal(m_blobAttB, m_blobAttB);
                m_attn_dropout.Reshape(m_colBtm, m_colTop);
            }
        }

        private void scaled_dot_product_cleanup()
        {
            dispose(ref m_blobKt1);
            dispose(ref m_blobAttA);
            dispose(ref m_blobAttB);
            dispose(ref m_softmax);
            dispose(ref m_attn_dropout);
        }

        private void scaled_dot_product_fwd(Blob<T> q, Blob<T> k, Blob<T> v, Blob<T> blobMask, Blob<T> y, int nSize)
        {
            bool bSaveDebugFiles = true;
            string strPath = "c:\\temp\\_debug\\";

            if (bSaveDebugFiles)
            {
                q.SaveToRawFile(strPath + "q.bin");
                k.SaveToRawFile(strPath + "k.bin");
                v.SaveToRawFile(strPath + "v.bin");
            }

            // Multiply query and key(T) matrices and scale
            // att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            addInternal(k, m_blobKt1);
            m_transposeQ.Forward(m_colBtm, m_colTop);

            if (bSaveDebugFiles)
                m_blobKt1.SaveToRawFile(strPath + "kt1.bin");

            Blob<T> blobTest = new Blob<T>(m_cuda, m_log);
            blobTest.CopyFromAndTransposeHeightWidth(k);
            m_log.CHECK(blobTest.Compare(m_blobKt1, m_blobWork), "The transposed blobs are different!");

            Blob<T> blobTest2 = new Blob<T>(m_cuda, m_log);
            blobTest2.CopyFromAndTransposeHeightWidth(m_blobKt1);
            m_log.CHECK(blobTest2.Compare(k, m_blobWork), "The transposed blobs are different!");

            blobTest.Dispose();
            blobTest2.Dispose();

            if (bSaveDebugFiles)
                m_log.CHECK(m_blobKt1.CompareToRawFile(strPath + "kt1.bin"), "The data has changed!");

            m_blobAttA.MatMul(q, m_blobKt1);
            double dfScale = 1.0 / Math.Sqrt(nSize);
            m_blobAttA.scale_data(dfScale);

            if (bSaveDebugFiles)
                m_blobAttA.SaveToRawFile(strPath + "attA.bin");

            // Apply mask to attention matrix
            // att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            float fInf = 1e+29f;
            m_cuda.mask_batch(m_blobAttA.count(), m_blobAttA.num, blobMask.count(), convert(0.0), convert(-1 * fInf), m_blobAttA.gpu_data, blobMask.gpu_data, m_blobAttA.mutable_gpu_data); // all masked items set to -inf.

            if (bSaveDebugFiles)
                m_blobAttA.SaveToRawFile(strPath + "attA_masked.bin");

            // Take softmax of attention along the last axis.
            // att = F.softmax(att, dim = -1)
            addInternal(m_blobAttA, m_blobAttB);
            m_softmax.Forward(m_colBtm, m_colTop);

            if (bSaveDebugFiles)
                m_blobAttB.SaveToRawFile(strPath + "attB.bin");

            // Apply attention dropout.
            // att = self.attn_dropout(att)
            if (m_attn_dropout != null)
            {
                addInternal(m_blobAttB, m_blobAttB);
                m_attn_dropout.Forward(m_colBtm, m_colTop);
            }

            if (bSaveDebugFiles)
                m_blobAttB.SaveToRawFile(strPath + "attB_dropout.bin");

            m_blobWork.Reshape(v.num, v.channels, v.height, v.width);

            // Multiply attention matrix with values
            // y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            m_blobWork.MatMul(m_blobAttB, v);
            y.CopyFrom(m_blobWork);

            if (bSaveDebugFiles)
                y.SaveToRawFile(strPath + "y.bin");
        }

        private void scaled_dot_product_bwd(Blob<T> q, Blob<T> k, Blob<T> v, Blob<T> blobMask, Blob<T> y, int nSize)
        {
            bool bDebug = false;
            string strPath = "c:\\temp\\_debug\\";
            List<bool> rgbPropagate = new List<bool>() { true, true };

            if (bDebug)
            {
                q.SaveToRawFile(strPath + "q.bin");
                k.SaveToRawFile(strPath + "k.bin");
                v.SaveToRawFile(strPath + "v.bin");
                y.SaveToRawFile(strPath + "y.grad.bin", true);
            }

            // Multiply attention matrix with values
            // y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            // Gradient with respect to att
            // att' = y' @ v^T 
            // Gradient with respect to vt
            // vt' = att^T @ y' 
            y.MatMulGrad(m_blobAttB, v, m_blobWork, 1);

            if (bDebug)
            {
                m_blobAttB.SaveToRawFile(strPath + "AttB.grad.bin", true);
                v.SaveToRawFile(strPath + "v.grad.bin", true);
            }

            // Apply attention dropout.
            // att = self.attn_dropout(att)
            if (m_attn_dropout != null)
            {
                addInternal(m_blobAttB, m_blobAttB);
                m_attn_dropout.Backward(m_colTop, rgbPropagate, m_colBtm);
            }

            if (bDebug)
                m_blobAttB.SaveToRawFile(strPath + "AttB_dropout.grad.bin", true);

            // Take softmax of attention along the last axis.
            // att = F.softmax(att, dim = -1)
            addInternal(m_blobAttA, m_blobAttB);
            m_softmax.Backward(m_colTop, rgbPropagate, m_colBtm);

            if (bDebug)
                m_blobAttA.SaveToRawFile(strPath + "AttA.grad.bin", true);

            // Multiply qt with kt^T to create attention matrix
            // att = qt @ kt^T
            // Gradient with respect to qt
            // qt' = att' @ kt
            // Gradient with respect to qt
            // qt' = att' @ kt

            if (bDebug)
            {
                m_blobKt1.SaveToRawFile(strPath + "kt1.bin");
                q.SaveToRawFile(strPath + "q_b.bin");
            }

            double dfScale = 1.0 / Math.Sqrt(nSize);
            m_blobAttA.MatMulGrad(q, m_blobKt1, m_blobWork, dfScale);

            if (bDebug)
            {
                q.SaveToRawFile(strPath + "q.grad.bin", true);
                m_blobKt1.SaveToRawFile(strPath + "kt1.grad.bin", true);
            }

            // Transpose Kt1 back to Kt
            k.CopyFromAndTransposeHeightWidth(m_blobKt1, true);

            if (bDebug)
                k.SaveToRawFile(strPath + "k.grad.bin", true);
        }

        public void TestFlashAttentionForward()
        {
            long hCuDnn = 0;
            long hAttn = 0;
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);

            try
            {
                int nBatch = 3;
                int nHeads = 8;
                int nBlock = 200;
                int nSize = 64;
                float fDropout = 0.0f;

                scaled_dot_product_setup(nBatch, nHeads, nBlock, nSize, fDropout);
                blobY.ReshapeLike(m_blobY);

                scaled_dot_product_fwd(m_blobQ, m_blobK, m_blobV, m_blobMask, m_blobY, nSize);

                hCuDnn = m_cuda.CreateCuDNN();
                hAttn = m_cuda.CreateAttn();
                m_cuda.SetAttn(hCuDnn, hAttn, 0, false, nBatch, nBlock, nHeads, nSize, fDropout);
                m_cuda.AttnScaledDotProductForward(hCuDnn, hAttn, m_blobQ.gpu_data, m_blobK.gpu_data, m_blobV.gpu_data, m_blobMask.gpu_data, blobY.mutable_gpu_data);

                m_log.CHECK(m_blobY.Compare(blobY, m_blobWork), "The Y blob data is different.");
            }
            finally
            {
                scaled_dot_product_cleanup();
                dispose(ref blobY);

                if (hAttn != 0)
                    m_cuda.FreeAttn(hAttn);

                if (hCuDnn != 0)
                    m_cuda.FreeCuDNN(hCuDnn);
            }
        }

        public void TestFlashAttentionBackward()
        {
            long hCuDnn = 0;
            long hAttn = 0;
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);
            Blob<T> blobQ = new Blob<T>(m_cuda, m_log);
            Blob<T> blobK = new Blob<T>(m_cuda, m_log);
            Blob<T> blobV = new Blob<T>(m_cuda, m_log);

            try
            {
                int nBatch = 3;
                int nHeads = 8;
                int nBlock = 200;
                int nSize = 64;
                float fDropout = 0.0f;

                scaled_dot_product_setup(nBatch, nHeads, nBlock, nSize, fDropout);
                blobY.CopyFrom(m_blobY, false, true);
                blobQ.CopyFrom(m_blobQ, false, true);
                blobK.CopyFrom(m_blobK, false, true);
                blobV.CopyFrom(m_blobV, false, true);

                scaled_dot_product_fwd(m_blobQ, m_blobK, m_blobV, m_blobMask, m_blobY, nSize);
                m_filler.Fill(m_blobY.count(), m_blobY.mutable_gpu_diff);

                scaled_dot_product_bwd(m_blobQ, m_blobK, m_blobV, m_blobMask, m_blobY, nSize);

                m_blobY.CompareToRawFile("c:\\temp\\_debug\\y.matmul1.a.grad.raw", true);

                hCuDnn = m_cuda.CreateCuDNN();
                hAttn = m_cuda.CreateAttn();
                m_cuda.SetAttn(hCuDnn, hAttn, 0, true, nBatch, nBlock, nHeads, nSize, fDropout);
                m_cuda.AttnScaledDotProductForward(hCuDnn, hAttn, blobQ.gpu_data, blobK.gpu_data, blobV.gpu_data, m_blobMask.gpu_data, blobY.mutable_gpu_data);
                blobY.CopyFrom(m_blobY, true);

                blobY.CompareToRawFile("c:\\temp\\_debug\\y.matmul1.a.grad.raw", true);

                Trace.WriteLine("Testing backward: hY = " + blobY.gpu_data.ToString() + " hdY = " + blobY.gpu_diff.ToString());
                m_cuda.AttnScaledDotProductBackward(hCuDnn, hAttn, blobQ.gpu_data, blobQ.mutable_gpu_diff, blobK.gpu_data, blobK.mutable_gpu_diff, blobV.gpu_data, blobV.mutable_gpu_diff, m_blobMask.gpu_data, blobY.gpu_data, blobY.gpu_diff);

                m_log.CHECK(m_blobQ.Compare(blobQ, m_blobWork, true), "The Q blob diffs are different.");
                m_log.CHECK(m_blobK.Compare(blobK, m_blobWork, true), "The K blob diffs are different.");
                m_log.CHECK(m_blobV.Compare(blobV, m_blobWork, true), "The V blob diffs are different.");
            }
            finally
            {
                scaled_dot_product_cleanup();
                dispose(ref blobY);

                if (hAttn != 0)
                    m_cuda.FreeAttn(hAttn);

                if (hCuDnn != 0)
                    m_cuda.FreeCuDNN(hCuDnn);
            }
        }
    }
}
