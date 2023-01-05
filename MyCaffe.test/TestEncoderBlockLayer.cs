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
                    t.TestForward(8, false);
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
            EncoderBlockLayerTest test = new EncoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IEncoderBlockLayerTest t in test.Tests)
                {
                    t.TestForward(8, true);
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
                    t.TestBackward(8, false);
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
            EncoderBlockLayerTest test = new EncoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IEncoderBlockLayerTest t in test.Tests)
                {
                    t.TestBackward(8, true);
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
        void TestForward(uint nHeads, bool bEnableCudaImpl);
        void TestBackward(uint nHeads, bool bEnableCudaImpl);
        void TestGradient(uint nHeads, bool bEnableCudaImpl);
    }

    class EncoderBlockLayerTest : TestBase
    {
        public EncoderBlockLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Encoder Transformer Block Test", TestBase.DEFAULT_DEVICE_ID, engine)
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

        public EncoderBlockLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobX = new Blob<T>(m_cuda, m_log);
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
            base.dispose();
        }

        private string loadTestData1()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\enc\\";
            string strFileName = "_encoder_test.zip";
            string strTestPath = "test";
            string strTestFile = "13_out1.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }

        public void TestForward(uint nHeads, bool bEnableCudaImpl)
        {
            string strTestDataPath = loadTestData1();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.ENCODER;
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

                m_blobX.LoadFromNumpy(strTestDataPath + "enc_in_x0.npy");
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
                m_blobYexp.LoadFromNumpy(strTestDataPath + "enc.12_output.npy");
                verify(TopVec[0], m_blobYexp, false, 3e-06);

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

        public void TestBackward(uint nHeads, bool bEnableCudaImpl)
        {
            string strTestDataPath = loadTestData1();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.ENCODER;
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

                m_blobX.LoadFromNumpy(strTestDataPath + "enc_in_x0.npy");
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

                // Now, check values from forward
                m_blobYexp.LoadFromNumpy(strTestDataPath + "enc.12_output.npy");
                verify(TopVec[0], m_blobYexp, false, 3e-06);

                // Load the inbound gradients.
                TopVec[0].LoadFromNumpy(strTestDataPath + "grad_enc.12_output.npy", true);

                List<bool> rgProp = new List<bool>() { true };
                layer.Backward(TopVec, rgProp, BottomVec);

                // Now, check values form backward
                m_blobYexp.LoadFromNumpy(strTestDataPath + "grad_enc.1_x.npy", true);
                verify(BottomVec[0], m_blobYexp, true, 1e-08);

                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (int i = 0; i < 100; i++)
                {
                    layer.Backward(TopVec, rgProp, BottomVec);
                }

                sw.Stop();
                double dfTime = sw.Elapsed.TotalMilliseconds / 100;
                Trace.WriteLine("Encoder Backward time = " + dfTime.ToString("N6") + " ms.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient(uint nHeads, bool bEnableCudaImpl)
        {
            string strTestDataPath = loadTestData1();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.ENCODER;
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

                m_blobX.LoadFromNumpy(strTestDataPath + "enc_in_x0.npy");
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
