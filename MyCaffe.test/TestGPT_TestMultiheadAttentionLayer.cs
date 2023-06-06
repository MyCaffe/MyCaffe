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
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
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
        public void TestBackward2()
        {
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
                {
                    t.TestBackward2(3, 8);
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
                    t.TestGradient(3, 8, MultiheadAttentionParameter.WEIGHT_INIT.ENCODER_DECODER);
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
                    t.TestGradient(3, 8, MultiheadAttentionParameter.WEIGHT_INIT.GPT);
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
        void TestForward(uint nBatch, uint nHeads);
        void TestBackward(uint nBatch, uint nHeads);
        void TestBackward2(uint nBatch, uint nHeads);
        void TestGradient(uint nBatch, uint nHeads, MultiheadAttentionParameter.WEIGHT_INIT init);
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

        private string loadTestData1()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\mh\\";
            string strFileName = "_multihead_test.zip";
            string strTestPath = "test";
            string strTestFile = "iter_0\\mh.10_concat_output1.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }

        public void TestForward(uint nBatch, uint nHeads)
        {
            string strTestDataPath = loadTestData1();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION);
            p.multihead_attention_param.heads = nHeads;
            p.multihead_attention_param.embed = 512;
            p.multihead_attention_param.block_size = 200;
            p.multihead_attention_param.attn_dropout = 0.0;
            p.multihead_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataPath + "q0.npy");
                m_blobK.LoadFromNumpy(strTestDataPath + "k0.npy");
                m_blobV.LoadFromNumpy(strTestDataPath + "v0.npy");
                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");
                m_blobMask.ReshapeLike(m_blobInput);
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMask.mutable_gpu_data);
                
                BottomVec.Clear();
                BottomVec.Add(m_blobQ);
                BottomVec.Add(m_blobK);
                BottomVec.Add(m_blobV);
                BottomVec.Add(m_blobMask);

                layer.Setup(BottomVec, TopVec);

                strTestDataPath += "iter_0\\";

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

                // Now, check values
                verify(TopVec[0], m_blobY, false, (typeof(T) == typeof(float)) ? 1e-12 : 3e-06);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackward(uint nBatch, uint nHeads)
        {
            string strTestDataPath = loadTestData1();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION);
            p.multihead_attention_param.heads = nHeads;
            p.multihead_attention_param.embed = 512;
            p.multihead_attention_param.block_size = 200;
            p.multihead_attention_param.attn_dropout = 0.0;
            p.multihead_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataPath + "q0.npy");
                m_blobK.LoadFromNumpy(strTestDataPath + "k0.npy");
                m_blobV.LoadFromNumpy(strTestDataPath + "v0.npy");
                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");
                m_blobMask.ReshapeLike(m_blobInput);
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMask.mutable_gpu_data);

                BottomVec.Clear();
                BottomVec.Add(m_blobQ);
                BottomVec.Add(m_blobK);
                BottomVec.Add(m_blobV);
                BottomVec.Add(m_blobMask);

                layer.Setup(BottomVec, TopVec);

                strTestDataPath += "iter_0\\";

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
                verify(BottomVec[0], m_blobQexp, true);
                verify(BottomVec[1], m_blobKexp, true);
                verify(BottomVec[2], m_blobVexp, true);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackward2(uint nBatch, uint nHeads)
        {
            string strTestDataPath = loadTestData1();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION);
            p.multihead_attention_param.heads = nHeads;
            p.multihead_attention_param.embed = 512;
            p.multihead_attention_param.block_size = 200;
            p.multihead_attention_param.attn_dropout = 0.0;
            p.multihead_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataPath + "q0.npy");
                m_blobK.LoadFromNumpy(strTestDataPath + "k0.npy");
                m_blobV.LoadFromNumpy(strTestDataPath + "v0.npy");
                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");
                m_blobMask.ReshapeLike(m_blobInput);
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMask.mutable_gpu_data);

                BottomVec.Clear();
                BottomVec.Add(m_blobQ);
                BottomVec.Add(m_blobK);
                BottomVec.Add(m_blobV);
                BottomVec.Add(m_blobMask);

                layer.Setup(BottomVec, TopVec);

                strTestDataPath += "iter_0\\";

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
                verify(BottomVec[0], m_blobQexp, true);
                verify(BottomVec[1], m_blobKexp, true);
                verify(BottomVec[2], m_blobVexp, true);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient(uint nBatch, uint nHeads, MultiheadAttentionParameter.WEIGHT_INIT init)
        {
            string strTestDataPath = loadTestData1();
            
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION);
            p.multihead_attention_param.weight_init = init;
            p.multihead_attention_param.heads = nHeads;
            p.multihead_attention_param.embed = 512;
            p.multihead_attention_param.block_size = 200;
            p.multihead_attention_param.attn_dropout = 0.0;
            p.multihead_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION, "The layer type is incorrect!");

                m_blobQ.LoadFromNumpy(strTestDataPath + "q0.npy");
                m_blobK.LoadFromNumpy(strTestDataPath + "k0.npy");
                m_blobV.LoadFromNumpy(strTestDataPath + "v0.npy");
                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");
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
    }
}
