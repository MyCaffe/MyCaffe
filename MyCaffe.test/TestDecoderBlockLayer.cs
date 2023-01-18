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
using static System.Net.Mime.MediaTypeNames;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.TrackBar;
using System.Drawing.Imaging;
using System.Runtime.CompilerServices;
using System.Windows.Forms;

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
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
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
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
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
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    t.TestBackward(8, true);
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
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    t.TestBackward2(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackward2Cuda()
        {
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    t.TestBackward2(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTraining()
        {
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestTraining();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestInference()
        {
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestInference();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFull()
        {
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    t.TestFull(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCudaFull()
        {
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    t.TestFull(true);
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
        void TestForward(uint nHeads, bool bEnableCudaImpl);
        void TestBackward(uint nHeads, bool bEnableCudaImpl);
        void TestBackward2(bool bEnableCudaImpl);
        void TestTraining();
        void TestInference();
        void TestFull(bool bEnableCudaImpl);
    }

    class DecoderBlockLayerTest : TestBase
    {
        public DecoderBlockLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Decoder Transformer Block Test", TestBase.DEFAULT_DEVICE_ID, engine)
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
        Blob<T> m_blobMaskEnc_exp;
        Blob<T> m_blobMaskDec_exp;
        Blob<T> m_blobInput;
        Blob<T> m_blobYexp;

        public DecoderBlockLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobX = new Blob<T>(m_cuda, m_log);
            m_blobEncOut = new Blob<T>(m_cuda, m_log);
            m_blobMaskEnc = new Blob<T>(m_cuda, m_log);
            m_blobMaskDec = new Blob<T>(m_cuda, m_log);
            m_blobMaskEnc_exp = new Blob<T>(m_cuda, m_log);
            m_blobMaskDec_exp = new Blob<T>(m_cuda, m_log);
            m_blobInput = new Blob<T>(m_cuda, m_log);
            m_blobYexp = new Blob<T>(m_cuda, m_log);
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
            dispose(ref m_blobMaskEnc_exp);
            dispose(ref m_blobMaskDec_exp);
            dispose(ref m_blobInput);
            dispose(ref m_blobYexp);
            base.dispose();
        }

        private string loadTestData1()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\dec\\";
            string strFileName = "_decoder_test.zip";
            string strTestPath = "test";
            string strTestFile = "13_out1.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }

        private string loadTestData2()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\trfb\\";
            string strFileName = "_transformer_test.zip";
            string strTestPath = "test";
            string strTestFile = "0_d_mask.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
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

        public void TestForward(uint nHeads, bool bEnableCudaImpl)
        {
            string strTestDataPath = loadTestData1();

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

                m_blobX.LoadFromNumpy(strTestDataPath + "dec.decin_x0.npy");
                m_blobX.Name = "dec.decin_x0";
                m_blobEncOut.LoadFromNumpy(strTestDataPath + "enc.encout_x1.npy");
                m_blobEncOut.Name = "enc.encout_x1";
                
                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");                
                m_blobMaskEnc.ReshapeLike(m_blobInput);
                m_blobMaskEnc.Name = "e_mask";
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMaskEnc.mutable_gpu_data);
                
                m_blobInput.LoadFromNumpy(strTestDataPath + "trg_input.npy");
                m_blobMaskDec.ReshapeLike(m_blobInput);
                m_blobMaskDec.Name = "d_mask";
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMaskDec.mutable_gpu_data);
                createDecoderMask(m_blobMaskDec);

                m_blobMaskDec_exp.LoadFromNumpy(strTestDataPath + "mh1.1_mask.npy");
                verify(m_blobMaskDec, m_blobMaskDec_exp, false, 1e-09);

                m_blobMaskEnc_exp.LoadFromNumpy(strTestDataPath + "mh2.1_mask.npy");
                verify(m_blobMaskEnc, m_blobMaskEnc_exp, false, 1e-09);

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
                layer.blobs[10].LoadFromNumpy(strTestDataPath + "mh2.w_k_weight.npy");   // multi-head key weight
                layer.blobs[11].LoadFromNumpy(strTestDataPath + "mh2.w_k_bias.npy");     // multi-head key bias
                layer.blobs[12].LoadFromNumpy(strTestDataPath + "mh2.w_v_weight.npy");   // multi-head value weight
                layer.blobs[13].LoadFromNumpy(strTestDataPath + "mh2.w_v_bias.npy");     // multi-head value bias
                layer.blobs[14].LoadFromNumpy(strTestDataPath + "mh2.w_o_weight.npy");   // multi-head output weight
                layer.blobs[15].LoadFromNumpy(strTestDataPath + "mh2.w_o_bias.npy");     // multi-head output bias

                layer.blobs[16].LoadFromNumpy(strTestDataPath + "ff.w_1_weight.npy");    // fc
                layer.blobs[17].LoadFromNumpy(strTestDataPath + "ff.w_1_bias.npy");      // fc
                layer.blobs[18].LoadFromNumpy(strTestDataPath + "ff.w_2_weight.npy");    // proj
                layer.blobs[19].LoadFromNumpy(strTestDataPath + "ff.w_2_bias.npy");      // proj

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

        public void TestBackward(uint nHeads, bool bEnableCudaImpl)
        {
            string strTestDataPath = loadTestData1();

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

                m_blobX.LoadFromNumpy(strTestDataPath + "dec.decin_x0.npy");
                m_blobX.Name = "dec.decin_x0";
                m_blobEncOut.LoadFromNumpy(strTestDataPath + "enc.encout_x1.npy");
                m_blobEncOut.Name = "enc.encout_x1";

                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");
                m_blobMaskEnc.ReshapeLike(m_blobInput);
                m_blobMaskEnc.Name = "e_mask";
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMaskEnc.mutable_gpu_data);

                m_blobInput.LoadFromNumpy(strTestDataPath + "trg_input.npy");
                m_blobMaskDec.ReshapeLike(m_blobInput);
                m_blobMaskDec.Name = "d_mask";
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMaskDec.mutable_gpu_data);
                createDecoderMask(m_blobMaskDec);

                m_blobMaskDec_exp.LoadFromNumpy(strTestDataPath + "mh1.1_mask.npy");
                verify(m_blobMaskDec, m_blobMaskDec_exp, false, 1e-09);

                m_blobMaskEnc_exp.LoadFromNumpy(strTestDataPath + "mh2.1_mask.npy");
                verify(m_blobMaskEnc, m_blobMaskEnc_exp, false, 1e-09);

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

        public void TestBackward2(bool bEnableCudaImpl)
        {
            string strPath = loadTestData1();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.DECODER;
            p.transformer_block_param.heads = 8;
            p.transformer_block_param.embed = 512;
            p.transformer_block_param.block_size = 200;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            p.transformer_block_param.enable_layernorm_cuda_impl = bEnableCudaImpl;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobDecIn = new Blob<T>(m_cuda, m_log);
            Blob<T> blobEncOut = new Blob<T>(m_cuda, m_log);
            Blob<T> blobDecMask = new Blob<T>(m_cuda, m_log);
            Blob<T> blobEncMask = new Blob<T>(m_cuda, m_log);
            Blob<T> blobVal = new Blob<T>(m_cuda, m_log);
            Blob<T> blobWork = new Blob<T>(m_cuda, m_log);

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.TRANSFORMER_BLOCK, "The layer type is incorrect!");
                
                blobDecIn.LoadFromNumpy(strPath + "dec.dec5.dec.1_x.npy");
                blobDecIn.Name = "dec.decin_x0";
                blobEncOut.LoadFromNumpy(strPath + "dec.dec5.dec.5_e_out.npy");
                blobEncOut.Name = "enc.encout_x1";

                blobEncMask.LoadFromNumpy(strPath + "dec.dec5.dec.1_e_mask.npy");
                blobEncMask.Name = "enc.mask";
                blobDecMask.LoadFromNumpy(strPath + "dec.dec5.dec.1_d_mask.npy");
                blobDecMask.Name = "dec.mask";

                BottomVec.Clear();
                BottomVec.Add(blobDecIn);
                BottomVec.Add(blobDecMask);
                BottomVec.Add(blobEncOut);
                BottomVec.Add(blobEncMask);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "dec.dec5.mh1.w_q.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "dec.dec5.mh1.w_q.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "dec.dec5.mh1.w_k.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "dec.dec5.mh1.w_k.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "dec.dec5.mh1.w_v.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "dec.dec5.mh1.w_v.bias.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "dec.dec5.mh1.w_o.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "dec.dec5.mh1.w_o.bias.npy");

                layer.blobs[8].LoadFromNumpy(strPath + "dec.dec5.mh2.w_q.weight.npy");
                layer.blobs[9].LoadFromNumpy(strPath + "dec.dec5.mh2.w_q.bias.npy");
                layer.blobs[10].LoadFromNumpy(strPath + "dec.dec5.mh2.w_k.weight.npy");
                layer.blobs[11].LoadFromNumpy(strPath + "dec.dec5.mh2.w_k.bias.npy");
                layer.blobs[12].LoadFromNumpy(strPath + "dec.dec5.mh2.w_v.weight.npy");
                layer.blobs[13].LoadFromNumpy(strPath + "dec.dec5.mh2.w_v.bias.npy");
                layer.blobs[14].LoadFromNumpy(strPath + "dec.dec5.mh2.w_o.weight.npy");
                layer.blobs[15].LoadFromNumpy(strPath + "dec.dec5.mh2.w_o.bias.npy");

                layer.blobs[16].LoadFromNumpy(strPath + "dec.dec5.ff.linear_1.weight.npy");
                layer.blobs[17].LoadFromNumpy(strPath + "dec.dec5.ff.linear_1.bias.npy");
                layer.blobs[18].LoadFromNumpy(strPath + "dec.dec5.ff.linear_2.weight.npy");
                layer.blobs[19].LoadFromNumpy(strPath + "dec.dec5.ff.linear_2.bias.npy");

                layer.Forward(BottomVec, TopVec);

                // Now, check values from forward
                blobVal.LoadFromNumpy(strPath + "dec.dec5.dec.10_xD.npy");
                Trace.Assert(blobVal.Compare(TopVec[0], blobWork, false, (typeof(T) == typeof(float)) ? 3e-06 : 4e-06));

                // Load the inbound gradients.
                TopVec[0].LoadFromNumpy(strPath + "grad_dec.dec5.dec.10_xD.npy", true);

                List<bool> rgProp = new List<bool>() { true };
                layer.Backward(TopVec, rgProp, BottomVec);

                // Now, check values form backward
                blobVal.LoadFromNumpy(strPath + "grad_dec.dec5.dec.1_x.npy", true);
                Trace.Assert(blobVal.Compare(BottomVec[0], blobWork, true, 3e-08));
                blobVal.LoadFromNumpy(strPath + "grad_dec.dec5.dec.5_e_out.npy", true);
                Trace.Assert(blobVal.Compare(BottomVec[2], blobWork, true, 1e-08));
            }
            finally
            {
                dispose(ref blobDecIn);
                dispose(ref blobEncOut);
                dispose(ref blobDecMask);
                dispose(ref blobEncMask);
                dispose(ref blobVal);
                dispose(ref blobWork);

                layer.Dispose();
            }
        }

        private Tuple<string, string, string, string, string, string> loadDataFiles1()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\encdec";
            string strFileName = "en_fr.zip";

            string strTestData = downloadTestData(strPath, strFileName);
            string strTestDataPath = Path.GetDirectoryName(strTestData);
            
            if (!File.Exists(strTestDataPath + "\\en_fr\\src\\train.txt"))
                ZipFile.ExtractToDirectory(strTestData, strPath);

            string strSrcTextT = strPath + "\\en_fr\\src\\train.txt";
            string strTrgTextT = strPath + "\\en_fr\\trg\\train.txt";
            string strSrcTextV = strPath + "\\en_fr\\src\\valid.txt";
            string strTrgTextV = strPath + "\\en_fr\\trg\\valid.txt";
            string strSrcVocab = strPath + "\\en_fr\\sp\\src_sp.vocab";
            string strTrgVocab = strPath + "\\en_fr\\sp\\trg_sp.vocab";

            return new Tuple<string, string, string, string, string, string>(strSrcTextT, strTrgTextT, strSrcTextV, strTrgTextV, strSrcVocab, strTrgVocab);
        }

        private string buildModel(string strSrcFileT, string strSrcFileV, string strSrcVocabFile, string strTrgFileT, string strTrgFileV, string strTrgVocabFile, uint nBatch, uint nBlockSize, uint nEmbed, uint nEncVocabSize, uint nDecVocabSize, double dfDropout)
        {
            NetParameter net = new NetParameter();
            net.name = "TranslatorNet";
            
            LayerParameter dataT = new LayerParameter(LayerParameter.LayerType.TOKENIZED_DATA_PAIRS);
            dataT.name = "tokdata1";
            dataT.tokenized_data_pairs_param.target = strTrgFileT;
            dataT.tokenized_data_pairs_param.target_vocab_file = strTrgVocabFile;
            dataT.tokenized_data_pairs_param.source = strSrcFileT;
            dataT.tokenized_data_pairs_param.source_vocab_file = strSrcVocabFile;
            dataT.tokenized_data_pairs_param.vocabulary_type = TokenizedDataParameter.VOCABULARY_TYPE.SENTENCEPIECE;
            dataT.tokenized_data_pairs_param.input_type = TokenizedDataParameter.INPUT_TYPE.TEXT_FILE;
            dataT.tokenized_data_pairs_param.batch_size = nBatch;
            dataT.tokenized_data_pairs_param.block_size = nBlockSize;
            dataT.top.Add("enc");
            dataT.top.Add("dec");
            dataT.top.Add("tgt");
            dataT.top.Add("emsk");
            dataT.top.Add("dmsk");
            dataT.include.Add(new NetStateRule(Phase.TRAIN));
            net.layer.Add(dataT);

            LayerParameter dataV = new LayerParameter(LayerParameter.LayerType.TOKENIZED_DATA_PAIRS);
            dataV.name = "tokdata1";
            dataV.tokenized_data_pairs_param.target = strTrgFileV;
            dataV.tokenized_data_pairs_param.target_vocab_file = strTrgVocabFile;
            dataV.tokenized_data_pairs_param.source = strSrcFileV;
            dataV.tokenized_data_pairs_param.source_vocab_file = strSrcVocabFile;
            dataV.tokenized_data_pairs_param.vocabulary_type = TokenizedDataParameter.VOCABULARY_TYPE.SENTENCEPIECE;
            dataV.tokenized_data_pairs_param.input_type = TokenizedDataParameter.INPUT_TYPE.TEXT_FILE;
            dataV.tokenized_data_pairs_param.batch_size = nBatch;
            dataV.tokenized_data_pairs_param.block_size = nBlockSize;
            dataV.top.Add("enc");
            dataV.top.Add("dec");
            dataV.top.Add("tgt");
            dataV.top.Add("emsk");
            dataV.top.Add("dmsk");
            dataV.include.Add(new NetStateRule(Phase.TEST));
            net.layer.Add(dataV);

            return buildModelEx(net, nBatch, nBlockSize, nEmbed, nEncVocabSize, nDecVocabSize, dfDropout);
        }

        private string buildModelEx(NetParameter net, uint nBatch, uint nBlockSize, uint nEmbed, uint nEncVocabSize, uint nDecVocabSize, double dfDropout, bool bAddInput = false, Phase phase = Phase.TRAIN)
        {
            if (bAddInput)
            {
                LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
                input.name = "input";
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize } });
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize } });
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize } });
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize } });
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize, (int)nBlockSize } });
                input.top.Add("enc");
                input.top.Add("dec");
                input.top.Add("tgt");
                input.top.Add("emsk");
                input.top.Add("dmsk");
                net.layer.Add(input);
            }

            LayerParameter emb1 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb1.name = "embed1";
            emb1.embed_param.bias_term = false;
            emb1.embed_param.input_dim = nEncVocabSize;
            emb1.embed_param.num_output = nEmbed;
            emb1.bottom.Add("enc");
            emb1.top.Add("emb1");
            net.layer.Add(emb1);

            LayerParameter emb2 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb2.name = "embed2";
            emb2.embed_param.bias_term = false;
            emb2.embed_param.input_dim = nDecVocabSize;
            emb2.embed_param.num_output = nEmbed;
            emb2.bottom.Add("dec");
            emb2.top.Add("emb2");
            net.layer.Add(emb2);

            LayerParameter pos1 = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODER);
            pos1.positional_encoder_param.block_size = nBlockSize;
            pos1.positional_encoder_param.embed = nEmbed;
            pos1.name = "posenc1";
            pos1.bottom.Add("emb1");
            pos1.top.Add("pos1");
            net.layer.Add(pos1);

            LayerParameter pos2 = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODER);
            pos2.positional_encoder_param.block_size = nBlockSize;
            pos2.positional_encoder_param.embed = nEmbed;
            pos2.name = "posenc2";
            pos2.bottom.Add("emb2");
            pos2.top.Add("pos2");
            net.layer.Add(pos2);

            string strEncBtm = "pos1";
            int nLayers = 6;
            for (int i = 0; i < nLayers; i++)
            {
                LayerParameter enc = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                enc.name = "enc" + (i + 1).ToString();
                enc.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.ENCODER;
                enc.transformer_block_param.heads = 8;
                enc.transformer_block_param.embed = nEmbed;
                enc.transformer_block_param.block_size = nBlockSize;
                enc.transformer_block_param.layers = (uint)nLayers;
                enc.transformer_block_param.activation = TransformerBlockParameter.ACTIVATION.RELU;
                enc.transformer_block_param.attn_dropout = dfDropout;
                enc.transformer_block_param.resid_dropout = dfDropout;
                enc.bottom.Add(strEncBtm);
                enc.bottom.Add("emsk");
                enc.top.Add(enc.name);
                net.layer.Add(enc);

                strEncBtm = enc.name;
            }

            LayerParameter ln1 = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            ln1.name = "ln1";
            ln1.layer_norm_param.enable_cuda_impl = false;
            ln1.bottom.Add(strEncBtm);
            ln1.top.Add("ln1");
            net.layer.Add(ln1);

            string strDecBtm = "pos2";
            for (int i = 0; i < nLayers; i++)
            {
                LayerParameter dec = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                dec.name = "dec" + (i + 1).ToString();
                dec.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.DECODER;
                dec.transformer_block_param.heads = 8;
                dec.transformer_block_param.embed = nEmbed;
                dec.transformer_block_param.block_size = nBlockSize;
                dec.transformer_block_param.layers = (uint)nLayers;
                dec.transformer_block_param.activation = TransformerBlockParameter.ACTIVATION.RELU;
                dec.transformer_block_param.attn_dropout = dfDropout;
                dec.transformer_block_param.resid_dropout = dfDropout;
                dec.bottom.Add(strDecBtm);
                dec.bottom.Add("dmsk");
                dec.bottom.Add("ln1");
                dec.bottom.Add("emsk");
                dec.top.Add(dec.name);
                net.layer.Add(dec);

                strDecBtm = dec.name;
            }

            LayerParameter ln2 = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            ln2.name = "ln2";
            ln2.layer_norm_param.enable_cuda_impl = false;
            ln2.bottom.Add(strDecBtm);
            ln2.top.Add("ln2");
            net.layer.Add(ln2);

            LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ip1.name = "ip1";
            ip1.inner_product_param.axis = 2;
            ip1.inner_product_param.num_output = nDecVocabSize;
            ip1.bottom.Add("ln2");
            ip1.top.Add("logits");
            net.layer.Add(ip1);

            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.LOG_SOFTMAX);
            softmax.name = "softmax";
            softmax.log_softmax_param.axis = 2;
            softmax.bottom.Add("logits");
            softmax.top.Add("prob");
            net.layer.Add(softmax);

            if (phase == Phase.TRAIN)
            {
                LayerParameter loss = new LayerParameter(LayerParameter.LayerType.NLL_LOSS);
                loss.name = "loss";
                loss.nll_loss_param.axis = 2;
                loss.loss_param.normalization = LossParameter.NormalizationMode.VALID;
                loss.bottom.Add("prob");
                loss.bottom.Add("tgt");
                loss.top.Add("loss");
                loss.include.Add(new NetStateRule(Phase.TRAIN));
                net.layer.Add(loss);
            }

            if (phase == Phase.TEST)
            {
                LayerParameter accuracy = new LayerParameter(LayerParameter.LayerType.ACCURACY);
                accuracy.name = "accuracy";
                accuracy.accuracy_param.axis = 2;
                accuracy.accuracy_param.ignore_labels.Add(0);
                accuracy.bottom.Add("prob");
                accuracy.bottom.Add("tgt");
                accuracy.top.Add("accuracy");
                accuracy.include.Add(new NetStateRule(Phase.TEST));
                net.layer.Add(accuracy);
            }

            return net.ToProto("root").ToString();
        }

        private string buildSolver()
        {
            SolverParameter solver = new SolverParameter();
            solver.base_lr = 1e-4;
            solver.type = SolverParameter.SolverType.ADAMW;
            solver.lr_policy = "fixed";
            solver.test_initialization = false;

            return solver.ToProto("root").ToString();
        }

        public void TestTraining()
        {
            Tuple<string, string, string, string, string, string> dataFiles = loadDataFiles1();
            string strSrcFileT = dataFiles.Item1;
            string strTrgFileT = dataFiles.Item2;
            string strSrcFileV = dataFiles.Item3;
            string strTrgFileV = dataFiles.Item4;
            string strSrcVocab = dataFiles.Item5;
            string strTrgVocab = dataFiles.Item6;

            string strModel = buildModel(strSrcFileT, strSrcFileV, strSrcVocab, strTrgFileT, strTrgFileV, strTrgVocab, 20, 200, 512, 14878, 14638, 0.1);
            string strSolver = buildSolver();

            SettingsCaffe s = new SettingsCaffe
            {
                GpuIds = "1"
            };
            CancelEvent evtCancel = new CancelEvent();
            MyCaffeControl<float> mycaffe = new MyCaffeControl<float>(s, m_log, evtCancel, null, null, null, null, "", true);
            mycaffe.OnTrainingIteration += Mycaffe_OnTrainingIteration;
            
            Blob<float> blobWork = null;
            
            try
            {
                mycaffe.Cuda.ReportMemory(m_log, "Pre-Model Load");
                SimpleDatum sdMean = new SimpleDatum(1, 1, 1);  // Dummy mean.
                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, true, sdMean);
                mycaffe.Cuda.ReportMemory(m_log, "Post-Model Load");
                
                Net<float> net = mycaffe.GetInternalNet(Phase.TRAIN);
                net.Forward();

                blobWork = mycaffe.CreateBlob("work");
                for (int i=0; i<net.parameters.Count; i++)
                {
                    Blob<float> blob = net.parameters[i];
                    blobWork.ReshapeLike(blob);

                    Tuple<double, double, double, double> data = blob.minmax_data(blobWork, true);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " min: " + data.Item1.ToString() + Environment.NewLine);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " max: " + data.Item2.ToString() + Environment.NewLine);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " nan: " + data.Item3.ToString() + Environment.NewLine);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " inf: " + data.Item4.ToString() + Environment.NewLine);
                    if (data.Item3 > 0 || data.Item4 > 0)
                        Trace.WriteLine("Nan or Inf detected!");
                    Trace.WriteLine("---------------");                    
                }

                for (int i = 0; i < net.blobs.Count; i++)
                {
                    Blob<float> blob = net.blobs[i];
                    blobWork.ReshapeLike(blob);

                    Tuple<double, double, double, double> data = blob.minmax_data(blobWork, true);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " min: " + data.Item1.ToString() + Environment.NewLine);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " max: " + data.Item2.ToString() + Environment.NewLine);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " nan: " + data.Item3.ToString() + Environment.NewLine);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " inf: " + data.Item4.ToString() + Environment.NewLine);
                    if (data.Item3 > 0 || data.Item4 > 0)
                        Trace.WriteLine("Nan or Inf detected!");
                    Trace.WriteLine("---------------");
                }

                mycaffe.Train(1);
                mycaffe.Test(1);
            }
            finally
            {
                if (blobWork != null)
                    blobWork.Dispose();

                mycaffe.Dispose();
            }
        }

        private void Mycaffe_OnTrainingIteration(object sender, TrainingIterationArgs<float> e)
        {
            Trace.WriteLine(e.Iteration.ToString() + " - " + e.Loss.ToString());
        }

        public void TestInference()
        {
            Tuple<string, string, string, string, string, string> dataFiles = loadDataFiles1();
            string strSrcFileT = dataFiles.Item1;
            string strTrgFileT = dataFiles.Item2;
            string strSrcFileV = dataFiles.Item3;
            string strTrgFileV = dataFiles.Item4;
            string strSrcVocab = dataFiles.Item5;
            string strTrgVocab = dataFiles.Item6;

            string strModel = File.ReadAllText("C:\\temp\\_models\\models\\enc_dec\\train_test.prototxt");
            byte[] rgWeights = File.ReadAllBytes("C:\\temp\\_models\\models\\enc_dec\\model.caffemodel");

            SettingsCaffe s = new SettingsCaffe
            {
                GpuIds = "1"
            };
            CancelEvent evtCancel = new CancelEvent();
            MyCaffeControl<float> mycaffe = new MyCaffeControl<float>(s, m_log, evtCancel, null, null, null, null, "", true);
            Blob<float> blobDec = null;

            try
            {
                mycaffe.Cuda.ReportMemory(m_log, "Pre-Model Load");
                mycaffe.LoadToRun(strModel, rgWeights, new BlobShape(new List<int>() { 1, 1, 1 }));
                mycaffe.Cuda.ReportMemory(m_log, "Post-Model Load");
                blobDec = mycaffe.CreateBlob("dec.decin");

                Net<float> net = mycaffe.GetInternalNet(Phase.RUN);

                string strInput = "When is the first session";
                PropertySet input = new PropertySet("InputData=" + strInput);
                BlobCollection<float> colBottom = net.layers[0].PreProcessInput(input);

                int nSeqLen = (int)net.layers[0].layer_param.tokenized_data_pairs_param.block_size;
                blobDec.Reshape(1, nSeqLen, 1, 1);
                blobDec.SetData(0);
                blobDec.SetData(1, 0); // BOS

                // Run up to the first decoder.
                net.input_blobs.Add(colBottom);
                net.input_blobs.Add(blobDec);

                BlobCollection<float> res = net.Forward();
                List<int> rgPredTokens = new List<int>();

                for (int i = 1; i < nSeqLen; i++)
                {
                    int nPredTok = argmax(res[0].mutable_cpu_data);

                    // Done, when EOS found.
                    if (nPredTok == 2)
                        break;

                    blobDec.SetData(nPredTok, i);

                    res = net.Forward();
                }

                string strResult = net.layers[0].PostProcessFullOutput(blobDec);
                Trace.WriteLine("Input: " + strInput);
                Trace.WriteLine("Output: " + strResult);
            }
            finally
            {
                mycaffe.Dispose();
            }
        }

        private int argmax(float[] rg)
        {
            float fMax = -float.MaxValue;
            int nIdx = -1;

            for (int i = 0; i < rg.Length; i++)
            {
                if (rg[i] > fMax)
                {
                    nIdx = i;
                    fMax = rg[i];
                }
            }

            return nIdx;
        }

        private void loadInitialState(Net<T> net, string strPath)
        {
            foreach (Layer<T> layer in net.layers)
            {
                switch (layer.layer_param.name)
                {
                    case "embed1":
                        layer.blobs[0].LoadFromNumpy(strPath + "transformer.src_emb.weight.npy");
                        break;

                    case "embed2":
                        layer.blobs[0].LoadFromNumpy(strPath + "transformer.trg_emb.weight.npy");
                        break;

                    case "enc1":
                        layer.blobs[0].LoadFromNumpy(strPath + "enc.enc0.mh.w_q.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "enc.enc0.mh.w_q.bias.npy");
                        layer.blobs[2].LoadFromNumpy(strPath + "enc.enc0.mh.w_k.weight.npy");
                        layer.blobs[3].LoadFromNumpy(strPath + "enc.enc0.mh.w_k.bias.npy");
                        layer.blobs[4].LoadFromNumpy(strPath + "enc.enc0.mh.w_v.weight.npy");
                        layer.blobs[5].LoadFromNumpy(strPath + "enc.enc0.mh.w_v.bias.npy");
                        layer.blobs[6].LoadFromNumpy(strPath + "enc.enc0.mh.w_o.weight.npy");
                        layer.blobs[7].LoadFromNumpy(strPath + "enc.enc0.mh.w_o.bias.npy");
                        
                        layer.blobs[8].LoadFromNumpy(strPath + "enc.enc0.ff.linear_1.weight.npy");
                        layer.blobs[9].LoadFromNumpy(strPath + "enc.enc0.ff.linear_1.bias.npy");
                        layer.blobs[10].LoadFromNumpy(strPath + "enc.enc0.ff.linear_2.weight.npy");
                        layer.blobs[11].LoadFromNumpy(strPath + "enc.enc0.ff.linear_2.bias.npy");
                        break;

                    case "enc2":
                        layer.blobs[0].LoadFromNumpy(strPath + "enc.enc1.mh.w_q.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "enc.enc1.mh.w_q.bias.npy");
                        layer.blobs[2].LoadFromNumpy(strPath + "enc.enc1.mh.w_k.weight.npy");
                        layer.blobs[3].LoadFromNumpy(strPath + "enc.enc1.mh.w_k.bias.npy");
                        layer.blobs[4].LoadFromNumpy(strPath + "enc.enc1.mh.w_v.weight.npy");
                        layer.blobs[5].LoadFromNumpy(strPath + "enc.enc1.mh.w_v.bias.npy");
                        layer.blobs[6].LoadFromNumpy(strPath + "enc.enc1.mh.w_o.weight.npy");
                        layer.blobs[7].LoadFromNumpy(strPath + "enc.enc1.mh.w_o.bias.npy");
                        
                        layer.blobs[8].LoadFromNumpy(strPath + "enc.enc1.ff.linear_1.weight.npy");
                        layer.blobs[9].LoadFromNumpy(strPath + "enc.enc1.ff.linear_1.bias.npy");
                        layer.blobs[10].LoadFromNumpy(strPath + "enc.enc1.ff.linear_2.weight.npy");
                        layer.blobs[11].LoadFromNumpy(strPath + "enc.enc1.ff.linear_2.bias.npy");
                        break;

                    case "enc3":
                        layer.blobs[0].LoadFromNumpy(strPath + "enc.enc2.mh.w_q.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "enc.enc2.mh.w_q.bias.npy");
                        layer.blobs[2].LoadFromNumpy(strPath + "enc.enc2.mh.w_k.weight.npy");
                        layer.blobs[3].LoadFromNumpy(strPath + "enc.enc2.mh.w_k.bias.npy");
                        layer.blobs[4].LoadFromNumpy(strPath + "enc.enc2.mh.w_v.weight.npy");
                        layer.blobs[5].LoadFromNumpy(strPath + "enc.enc2.mh.w_v.bias.npy");
                        layer.blobs[6].LoadFromNumpy(strPath + "enc.enc2.mh.w_o.weight.npy");
                        layer.blobs[7].LoadFromNumpy(strPath + "enc.enc2.mh.w_o.bias.npy");
                        
                        layer.blobs[8].LoadFromNumpy(strPath + "enc.enc2.ff.linear_1.weight.npy");
                        layer.blobs[9].LoadFromNumpy(strPath + "enc.enc2.ff.linear_1.bias.npy");
                        layer.blobs[10].LoadFromNumpy(strPath + "enc.enc2.ff.linear_2.weight.npy");
                        layer.blobs[11].LoadFromNumpy(strPath + "enc.enc2.ff.linear_2.bias.npy");
                        break;

                    case "enc4":
                        layer.blobs[0].LoadFromNumpy(strPath + "enc.enc3.mh.w_q.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "enc.enc3.mh.w_q.bias.npy");
                        layer.blobs[2].LoadFromNumpy(strPath + "enc.enc3.mh.w_k.weight.npy");
                        layer.blobs[3].LoadFromNumpy(strPath + "enc.enc3.mh.w_k.bias.npy");
                        layer.blobs[4].LoadFromNumpy(strPath + "enc.enc3.mh.w_v.weight.npy");
                        layer.blobs[5].LoadFromNumpy(strPath + "enc.enc3.mh.w_v.bias.npy");
                        layer.blobs[6].LoadFromNumpy(strPath + "enc.enc3.mh.w_o.weight.npy");
                        layer.blobs[7].LoadFromNumpy(strPath + "enc.enc3.mh.w_o.bias.npy");
                        
                        layer.blobs[8].LoadFromNumpy(strPath + "enc.enc3.ff.linear_1.weight.npy");
                        layer.blobs[9].LoadFromNumpy(strPath + "enc.enc3.ff.linear_1.bias.npy");
                        layer.blobs[10].LoadFromNumpy(strPath + "enc.enc3.ff.linear_2.weight.npy");
                        layer.blobs[11].LoadFromNumpy(strPath + "enc.enc3.ff.linear_2.bias.npy");
                        break;

                    case "enc5":
                        layer.blobs[0].LoadFromNumpy(strPath + "enc.enc4.mh.w_q.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "enc.enc4.mh.w_q.bias.npy");
                        layer.blobs[2].LoadFromNumpy(strPath + "enc.enc4.mh.w_k.weight.npy");
                        layer.blobs[3].LoadFromNumpy(strPath + "enc.enc4.mh.w_k.bias.npy");
                        layer.blobs[4].LoadFromNumpy(strPath + "enc.enc4.mh.w_v.weight.npy");
                        layer.blobs[5].LoadFromNumpy(strPath + "enc.enc4.mh.w_v.bias.npy");
                        layer.blobs[6].LoadFromNumpy(strPath + "enc.enc4.mh.w_o.weight.npy");
                        layer.blobs[7].LoadFromNumpy(strPath + "enc.enc4.mh.w_o.bias.npy");
                        
                        layer.blobs[8].LoadFromNumpy(strPath + "enc.enc4.ff.linear_1.weight.npy");
                        layer.blobs[9].LoadFromNumpy(strPath + "enc.enc4.ff.linear_1.bias.npy");
                        layer.blobs[10].LoadFromNumpy(strPath + "enc.enc4.ff.linear_2.weight.npy");
                        layer.blobs[11].LoadFromNumpy(strPath + "enc.enc4.ff.linear_2.bias.npy");
                        break;

                    case "enc6":
                        layer.blobs[0].LoadFromNumpy(strPath + "enc.enc5.mh.w_q.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "enc.enc5.mh.w_q.bias.npy");
                        layer.blobs[2].LoadFromNumpy(strPath + "enc.enc5.mh.w_k.weight.npy");
                        layer.blobs[3].LoadFromNumpy(strPath + "enc.enc5.mh.w_k.bias.npy");
                        layer.blobs[4].LoadFromNumpy(strPath + "enc.enc5.mh.w_v.weight.npy");
                        layer.blobs[5].LoadFromNumpy(strPath + "enc.enc5.mh.w_v.bias.npy");
                        layer.blobs[6].LoadFromNumpy(strPath + "enc.enc5.mh.w_o.weight.npy");
                        layer.blobs[7].LoadFromNumpy(strPath + "enc.enc5.mh.w_o.bias.npy");
                        
                        layer.blobs[8].LoadFromNumpy(strPath + "enc.enc5.ff.linear_1.weight.npy");
                        layer.blobs[9].LoadFromNumpy(strPath + "enc.enc5.ff.linear_1.bias.npy");
                        layer.blobs[10].LoadFromNumpy(strPath + "enc.enc5.ff.linear_2.weight.npy");
                        layer.blobs[11].LoadFromNumpy(strPath + "enc.enc5.ff.linear_2.bias.npy");
                        break;

                    case "dec1":
                        layer.blobs[0].LoadFromNumpy(strPath + "dec.dec0.mh1.w_q.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "dec.dec0.mh1.w_q.bias.npy");
                        layer.blobs[2].LoadFromNumpy(strPath + "dec.dec0.mh1.w_k.weight.npy");
                        layer.blobs[3].LoadFromNumpy(strPath + "dec.dec0.mh1.w_k.bias.npy");
                        layer.blobs[4].LoadFromNumpy(strPath + "dec.dec0.mh1.w_v.weight.npy");
                        layer.blobs[5].LoadFromNumpy(strPath + "dec.dec0.mh1.w_v.bias.npy");
                        layer.blobs[6].LoadFromNumpy(strPath + "dec.dec0.mh1.w_o.weight.npy");
                        layer.blobs[7].LoadFromNumpy(strPath + "dec.dec0.mh1.w_o.bias.npy");

                        layer.blobs[8].LoadFromNumpy(strPath + "dec.dec0.mh2.w_q.weight.npy");
                        layer.blobs[9].LoadFromNumpy(strPath + "dec.dec0.mh2.w_q.bias.npy");
                        layer.blobs[10].LoadFromNumpy(strPath + "dec.dec0.mh2.w_k.weight.npy");
                        layer.blobs[11].LoadFromNumpy(strPath + "dec.dec0.mh2.w_k.bias.npy");
                        layer.blobs[12].LoadFromNumpy(strPath + "dec.dec0.mh2.w_v.weight.npy");
                        layer.blobs[13].LoadFromNumpy(strPath + "dec.dec0.mh2.w_v.bias.npy");
                        layer.blobs[14].LoadFromNumpy(strPath + "dec.dec0.mh2.w_o.weight.npy");
                        layer.blobs[15].LoadFromNumpy(strPath + "dec.dec0.mh2.w_o.bias.npy");

                        layer.blobs[16].LoadFromNumpy(strPath + "dec.dec0.ff.linear_1.weight.npy");
                        layer.blobs[17].LoadFromNumpy(strPath + "dec.dec0.ff.linear_1.bias.npy");
                        layer.blobs[18].LoadFromNumpy(strPath + "dec.dec0.ff.linear_2.weight.npy");
                        layer.blobs[19].LoadFromNumpy(strPath + "dec.dec0.ff.linear_2.bias.npy");
                        break;

                    case "dec2":
                        layer.blobs[0].LoadFromNumpy(strPath + "dec.dec1.mh1.w_q.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "dec.dec1.mh1.w_q.bias.npy");
                        layer.blobs[2].LoadFromNumpy(strPath + "dec.dec1.mh1.w_k.weight.npy");
                        layer.blobs[3].LoadFromNumpy(strPath + "dec.dec1.mh1.w_k.bias.npy");
                        layer.blobs[4].LoadFromNumpy(strPath + "dec.dec1.mh1.w_v.weight.npy");
                        layer.blobs[5].LoadFromNumpy(strPath + "dec.dec1.mh1.w_v.bias.npy");
                        layer.blobs[6].LoadFromNumpy(strPath + "dec.dec1.mh1.w_o.weight.npy");
                        layer.blobs[7].LoadFromNumpy(strPath + "dec.dec1.mh1.w_o.bias.npy");

                        layer.blobs[8].LoadFromNumpy(strPath + "dec.dec1.mh2.w_q.weight.npy");
                        layer.blobs[9].LoadFromNumpy(strPath + "dec.dec1.mh2.w_q.bias.npy");
                        layer.blobs[10].LoadFromNumpy(strPath + "dec.dec1.mh2.w_k.weight.npy");
                        layer.blobs[11].LoadFromNumpy(strPath + "dec.dec1.mh2.w_k.bias.npy");
                        layer.blobs[12].LoadFromNumpy(strPath + "dec.dec1.mh2.w_v.weight.npy");
                        layer.blobs[13].LoadFromNumpy(strPath + "dec.dec1.mh2.w_v.bias.npy");
                        layer.blobs[14].LoadFromNumpy(strPath + "dec.dec1.mh2.w_o.weight.npy");
                        layer.blobs[15].LoadFromNumpy(strPath + "dec.dec1.mh2.w_o.bias.npy");

                        layer.blobs[16].LoadFromNumpy(strPath + "dec.dec1.ff.linear_1.weight.npy");
                        layer.blobs[17].LoadFromNumpy(strPath + "dec.dec1.ff.linear_1.bias.npy");
                        layer.blobs[18].LoadFromNumpy(strPath + "dec.dec1.ff.linear_2.weight.npy");
                        layer.blobs[19].LoadFromNumpy(strPath + "dec.dec1.ff.linear_2.bias.npy");
                        break;

                    case "dec3":
                        layer.blobs[0].LoadFromNumpy(strPath + "dec.dec2.mh1.w_q.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "dec.dec2.mh1.w_q.bias.npy");
                        layer.blobs[2].LoadFromNumpy(strPath + "dec.dec2.mh1.w_k.weight.npy");
                        layer.blobs[3].LoadFromNumpy(strPath + "dec.dec2.mh1.w_k.bias.npy");
                        layer.blobs[4].LoadFromNumpy(strPath + "dec.dec2.mh1.w_v.weight.npy");
                        layer.blobs[5].LoadFromNumpy(strPath + "dec.dec2.mh1.w_v.bias.npy");
                        layer.blobs[6].LoadFromNumpy(strPath + "dec.dec2.mh1.w_o.weight.npy");
                        layer.blobs[7].LoadFromNumpy(strPath + "dec.dec2.mh1.w_o.bias.npy");

                        layer.blobs[8].LoadFromNumpy(strPath + "dec.dec2.mh2.w_q.weight.npy");
                        layer.blobs[9].LoadFromNumpy(strPath + "dec.dec2.mh2.w_q.bias.npy");
                        layer.blobs[10].LoadFromNumpy(strPath + "dec.dec2.mh2.w_k.weight.npy");
                        layer.blobs[11].LoadFromNumpy(strPath + "dec.dec2.mh2.w_k.bias.npy");
                        layer.blobs[12].LoadFromNumpy(strPath + "dec.dec2.mh2.w_v.weight.npy");
                        layer.blobs[13].LoadFromNumpy(strPath + "dec.dec2.mh2.w_v.bias.npy");
                        layer.blobs[14].LoadFromNumpy(strPath + "dec.dec2.mh2.w_o.weight.npy");
                        layer.blobs[15].LoadFromNumpy(strPath + "dec.dec2.mh2.w_o.bias.npy");

                        layer.blobs[16].LoadFromNumpy(strPath + "dec.dec2.ff.linear_1.weight.npy");
                        layer.blobs[17].LoadFromNumpy(strPath + "dec.dec2.ff.linear_1.bias.npy");
                        layer.blobs[18].LoadFromNumpy(strPath + "dec.dec2.ff.linear_2.weight.npy");
                        layer.blobs[19].LoadFromNumpy(strPath + "dec.dec2.ff.linear_2.bias.npy");
                        break;

                    case "dec4":
                        layer.blobs[0].LoadFromNumpy(strPath + "dec.dec3.mh1.w_q.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "dec.dec3.mh1.w_q.bias.npy");
                        layer.blobs[2].LoadFromNumpy(strPath + "dec.dec3.mh1.w_k.weight.npy");
                        layer.blobs[3].LoadFromNumpy(strPath + "dec.dec3.mh1.w_k.bias.npy");
                        layer.blobs[4].LoadFromNumpy(strPath + "dec.dec3.mh1.w_v.weight.npy");
                        layer.blobs[5].LoadFromNumpy(strPath + "dec.dec3.mh1.w_v.bias.npy");
                        layer.blobs[6].LoadFromNumpy(strPath + "dec.dec3.mh1.w_o.weight.npy");
                        layer.blobs[7].LoadFromNumpy(strPath + "dec.dec3.mh1.w_o.bias.npy");

                        layer.blobs[8].LoadFromNumpy(strPath + "dec.dec3.mh2.w_q.weight.npy");
                        layer.blobs[9].LoadFromNumpy(strPath + "dec.dec3.mh2.w_q.bias.npy");
                        layer.blobs[10].LoadFromNumpy(strPath + "dec.dec3.mh2.w_k.weight.npy");
                        layer.blobs[11].LoadFromNumpy(strPath + "dec.dec3.mh2.w_k.bias.npy");
                        layer.blobs[12].LoadFromNumpy(strPath + "dec.dec3.mh2.w_v.weight.npy");
                        layer.blobs[13].LoadFromNumpy(strPath + "dec.dec3.mh2.w_v.bias.npy");
                        layer.blobs[14].LoadFromNumpy(strPath + "dec.dec3.mh2.w_o.weight.npy");
                        layer.blobs[15].LoadFromNumpy(strPath + "dec.dec3.mh2.w_o.bias.npy");

                        layer.blobs[16].LoadFromNumpy(strPath + "dec.dec3.ff.linear_1.weight.npy");
                        layer.blobs[17].LoadFromNumpy(strPath + "dec.dec3.ff.linear_1.bias.npy");
                        layer.blobs[18].LoadFromNumpy(strPath + "dec.dec3.ff.linear_2.weight.npy");
                        layer.blobs[19].LoadFromNumpy(strPath + "dec.dec3.ff.linear_2.bias.npy");
                        break;

                    case "dec5":
                        layer.blobs[0].LoadFromNumpy(strPath + "dec.dec4.mh1.w_q.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "dec.dec4.mh1.w_q.bias.npy");
                        layer.blobs[2].LoadFromNumpy(strPath + "dec.dec4.mh1.w_k.weight.npy");
                        layer.blobs[3].LoadFromNumpy(strPath + "dec.dec4.mh1.w_k.bias.npy");
                        layer.blobs[4].LoadFromNumpy(strPath + "dec.dec4.mh1.w_v.weight.npy");
                        layer.blobs[5].LoadFromNumpy(strPath + "dec.dec4.mh1.w_v.bias.npy");
                        layer.blobs[6].LoadFromNumpy(strPath + "dec.dec4.mh1.w_o.weight.npy");
                        layer.blobs[7].LoadFromNumpy(strPath + "dec.dec4.mh1.w_o.bias.npy");

                        layer.blobs[8].LoadFromNumpy(strPath + "dec.dec4.mh2.w_q.weight.npy");
                        layer.blobs[9].LoadFromNumpy(strPath + "dec.dec4.mh2.w_q.bias.npy");
                        layer.blobs[10].LoadFromNumpy(strPath + "dec.dec4.mh2.w_k.weight.npy");
                        layer.blobs[11].LoadFromNumpy(strPath + "dec.dec4.mh2.w_k.bias.npy");
                        layer.blobs[12].LoadFromNumpy(strPath + "dec.dec4.mh2.w_v.weight.npy");
                        layer.blobs[13].LoadFromNumpy(strPath + "dec.dec4.mh2.w_v.bias.npy");
                        layer.blobs[14].LoadFromNumpy(strPath + "dec.dec4.mh2.w_o.weight.npy");
                        layer.blobs[15].LoadFromNumpy(strPath + "dec.dec4.mh2.w_o.bias.npy");

                        layer.blobs[16].LoadFromNumpy(strPath + "dec.dec4.ff.linear_1.weight.npy");
                        layer.blobs[17].LoadFromNumpy(strPath + "dec.dec4.ff.linear_1.bias.npy");
                        layer.blobs[18].LoadFromNumpy(strPath + "dec.dec4.ff.linear_2.weight.npy");
                        layer.blobs[19].LoadFromNumpy(strPath + "dec.dec4.ff.linear_2.bias.npy");
                        break;

                    case "dec6":
                        layer.blobs[0].LoadFromNumpy(strPath + "dec.dec5.mh1.w_q.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "dec.dec5.mh1.w_q.bias.npy");
                        layer.blobs[2].LoadFromNumpy(strPath + "dec.dec5.mh1.w_k.weight.npy");
                        layer.blobs[3].LoadFromNumpy(strPath + "dec.dec5.mh1.w_k.bias.npy");
                        layer.blobs[4].LoadFromNumpy(strPath + "dec.dec5.mh1.w_v.weight.npy");
                        layer.blobs[5].LoadFromNumpy(strPath + "dec.dec5.mh1.w_v.bias.npy");
                        layer.blobs[6].LoadFromNumpy(strPath + "dec.dec5.mh1.w_o.weight.npy");
                        layer.blobs[7].LoadFromNumpy(strPath + "dec.dec5.mh1.w_o.bias.npy");

                        layer.blobs[8].LoadFromNumpy(strPath + "dec.dec5.mh2.w_q.weight.npy");
                        layer.blobs[9].LoadFromNumpy(strPath + "dec.dec5.mh2.w_q.bias.npy");
                        layer.blobs[10].LoadFromNumpy(strPath + "dec.dec5.mh2.w_k.weight.npy");
                        layer.blobs[11].LoadFromNumpy(strPath + "dec.dec5.mh2.w_k.bias.npy");
                        layer.blobs[12].LoadFromNumpy(strPath + "dec.dec5.mh2.w_v.weight.npy");
                        layer.blobs[13].LoadFromNumpy(strPath + "dec.dec5.mh2.w_v.bias.npy");
                        layer.blobs[14].LoadFromNumpy(strPath + "dec.dec5.mh2.w_o.weight.npy");
                        layer.blobs[15].LoadFromNumpy(strPath + "dec.dec5.mh2.w_o.bias.npy");

                        layer.blobs[16].LoadFromNumpy(strPath + "dec.dec5.ff.linear_1.weight.npy");
                        layer.blobs[17].LoadFromNumpy(strPath + "dec.dec5.ff.linear_1.bias.npy");
                        layer.blobs[18].LoadFromNumpy(strPath + "dec.dec5.ff.linear_2.weight.npy");
                        layer.blobs[19].LoadFromNumpy(strPath + "dec.dec5.ff.linear_2.bias.npy");
                        break;

                    case "ip1":
                        layer.blobs[0].LoadFromNumpy(strPath + "transformer.out_linear.weight.npy");
                        layer.blobs[1].LoadFromNumpy(strPath + "transformer.out_linear.bias.npy");
                        break;
                }
            }
        }

        public void TestFull(bool bEnableCudaImpl)
        {
            string strPath = loadTestData2();
            CancelEvent evtCancel = new CancelEvent();
            NetParameter net_param = new NetParameter();
            string strModel = buildModelEx(net_param, 3, 200, 512, 16000, 16000, 0.0, true, Phase.TRAIN);
            string strSolver = buildSolver();
            Blob<T> blobEncIn = null;
            Blob<T> blobDecIn = null;
            Blob<T> blobDecOut = null;
            Blob<T> blobEncMask = null;
            Blob<T> blobDecMask = null;
            Blob<T> blobLoss = null;
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            MyCaffeControl<T> mycaffe = null;

            try
            {                
                SettingsCaffe s = new SettingsCaffe() { GpuIds = "0" };
                mycaffe = new MyCaffeControl<T>(s, m_log, evtCancel);
                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);

                blobEncIn = mycaffe.CreateBlob("encin");
                blobDecIn = mycaffe.CreateBlob("decin");
                blobDecOut = mycaffe.CreateBlob("decout");
                blobEncMask = mycaffe.CreateBlob("e_mask");
                blobDecMask = mycaffe.CreateBlob("d_mask");
                blobLoss = mycaffe.CreateBlob("loss");
                blobVal = mycaffe.CreateBlob("val");
                blobWork = mycaffe.CreateBlob("work");

                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                loadInitialState(net, strPath + "iter_0\\");

                blobEncIn.LoadFromNumpy(strPath + "0_src_input.npy");
                blobDecIn.LoadFromNumpy(strPath + "0_trg_input.npy");
                blobDecOut.LoadFromNumpy(strPath + "0_trg_output.npy");
                blobEncMask.LoadFromNumpy(strPath + "0_e_mask.npy");
                blobDecMask.LoadFromNumpy(strPath + "0_d_mask.npy");

                blobEncMask.Reshape(blobEncMask.num, blobEncMask.height, 1, 1);

                BlobCollection<T> colBottom = new BlobCollection<T>();
                colBottom.Add(blobEncIn);
                colBottom.Add(blobDecIn);
                colBottom.Add(blobDecOut);
                colBottom.Add(blobEncMask);
                colBottom.Add(blobDecMask);

                strPath += "iter_0\\";

                //---------------------------------
                // Test the forward pass
                //---------------------------------
                double dfLoss;
                BlobCollection<T> colTop = net.Forward(colBottom, out dfLoss);

                int nEmb1LayerI = 3;
                blobVal.LoadFromNumpy(strPath + "transformer.src_input.npy");
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEmb1LayerI][0], blobWork, false, 1e-08));
                blobVal.LoadFromNumpy(strPath + "transformer.src_emb.weight.npy");
                Trace.Assert(blobVal.Compare(net.layers[nEmb1LayerI].blobs[0], blobWork, false, 1e-08));
                blobVal.LoadFromNumpy(strPath + "transformer.src_input_emb.npy");
                Trace.Assert(blobVal.Compare(net.top_vecs[nEmb1LayerI][0], blobWork, false, 1e-08));

                int nEmb2LayerI = 4;
                blobVal.LoadFromNumpy(strPath + "transformer.trg_input.npy");
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEmb2LayerI][0], blobWork, false, 1e-08));
                blobVal.LoadFromNumpy(strPath + "transformer.trg_emb.weight.npy");
                Trace.Assert(blobVal.Compare(net.layers[nEmb2LayerI].blobs[0], blobWork, false, 1e-08));
                blobVal.LoadFromNumpy(strPath + "transformer.trg_input_emb.npy");
                Trace.Assert(blobVal.Compare(net.top_vecs[nEmb2LayerI][0], blobWork, false, 1e-08));

                int nEmb1LayerO = 5;
                blobVal.LoadFromNumpy(strPath + "transformer.src_input_emb.npy");
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEmb1LayerO][0], blobWork, false, 1e-08));

                int nEmb2LayerO = 6;
                blobVal.LoadFromNumpy(strPath + "transformer.trg_input_emb.npy");
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEmb2LayerO][0], blobWork, false, 1e-08));

                int nPos1LayerO = 7;
                blobVal.LoadFromNumpy(strPath + "transformer.src_input_pos.npy");
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nPos1LayerO][0], blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-07));
                blobVal.LoadFromNumpy(strPath + "transformer.e_mask.npy");
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nPos1LayerO][1], blobWork, false, 1e-08));

                int nEnc1Layer = 8;
                blobVal.LoadFromNumpy(strPath + "enc.enc1.enc.1_x.npy");
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEnc1Layer][0], blobWork, false, 7e-06));

                int nEnc2Layer = 9;
                blobVal.LoadFromNumpy(strPath + "enc.enc2.enc.1_x.npy");
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEnc2Layer][0], blobWork, false, 8e-06));

                int nEnc3Layer = 10;
                blobVal.LoadFromNumpy(strPath + "enc.enc3.enc.1_x.npy");
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEnc3Layer][0], blobWork, false, 1e-05));

                int nEnc4Layer = 11;
                blobVal.LoadFromNumpy(strPath + "enc.enc4.enc.1_x.npy");
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEnc4Layer][0], blobWork, false, 1.2e-05));

                int nEnc5Layer = 12;
                blobVal.LoadFromNumpy(strPath + "enc.enc5.enc.1_x.npy");
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEnc5Layer][0], blobWork, false, 1.3e-05));

                int nEncOutLayerO = 13;
                blobVal.LoadFromNumpy(strPath + "enc.ln.out.npy");
                Trace.Assert(blobVal.Compare(net.top_vecs[nEncOutLayerO][0], blobWork, false, 5e-06));
                blobVal.LoadFromNumpy(strPath + "transformer.e_output.npy");
                Trace.Assert(blobVal.Compare(net.top_vecs[nEncOutLayerO][0], blobWork, false, 5e-06));

                int nPos2LayerO = 15;
                blobVal.LoadFromNumpy(strPath + "transformer.trg_input_pos.npy");
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nPos2LayerO][0], blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-07));

                int nDecOutLayerO = 21;
                blobVal.LoadFromNumpy(strPath + "transformer.d_output.npy");
                Trace.Assert(blobVal.Compare(net.top_vecs[nDecOutLayerO][0], blobWork, false, 5e-06));

                int nDecOutLayerO1 = 22;
                blobVal.LoadFromNumpy(strPath + "transformer.d_output1.npy");
                Trace.Assert(blobVal.Compare(net.top_vecs[nDecOutLayerO1][0], blobWork, false, 2e-06));

                blobLoss.LoadFromNumpy(strPath + "0_loss.npy");
                double dfActual = Utility.ConvertVal<T>(blobLoss.GetData(0));
                double dfErr = (typeof(T) == typeof(float)) ? 1e-08 : 1e-06;
                m_log.EXPECT_NEAR(dfLoss, dfActual, dfErr, "The loss does not match the actual.");

                //---------------------------------
                // Test the backward pass.
                //---------------------------------

                blobLoss.LoadFromNumpy(strPath + "grad_0_loss.npy", true);
                int nOutLayer = 24;
                net.top_vecs[nOutLayer][0].CopyFrom(blobLoss, true);

                net.Backward();

                blobVal.LoadFromNumpy(strPath + "grad_transformer.d_output1.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nDecOutLayerO1 + 1][0], blobWork, true, 1e-08));

                blobVal.LoadFromNumpy(strPath + "grad_transformer.d_output.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nDecOutLayerO1][0], blobWork, true, 1e-08));
                
                blobVal.LoadFromNumpy(strPath + "grad_dec.ln.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nDecOutLayerO - 0][0], blobWork, true, 1e-08));

                blobVal.LoadFromNumpy(strPath + "grad_dec.x5.out.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nDecOutLayerO - 0][0], blobWork, true, 3e-08));

                blobVal.LoadFromNumpy(strPath + "grad_dec.x5.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nDecOutLayerO - 1][0], blobWork, true, 3e-08));

                blobVal.LoadFromNumpy(strPath + "grad_dec.x4.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nDecOutLayerO - 2][0], blobWork, true, 3e-08));

                blobVal.LoadFromNumpy(strPath + "grad_dec.x3.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nDecOutLayerO - 3][0], blobWork, true, 4e-08));

                blobVal.LoadFromNumpy(strPath + "grad_dec.x2.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nDecOutLayerO - 4][0], blobWork, true, 5e-08));

                blobVal.LoadFromNumpy(strPath + "grad_dec.x1.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nDecOutLayerO - 5][0], blobWork, true, 2e-07));

                blobVal.LoadFromNumpy(strPath + "grad_dec.x0.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nDecOutLayerO - 6][0], blobWork, true, 3e-07));

                int nSplitLayer = 14;
                blobVal.LoadFromNumpy(strPath + "grad_transformer.e_output.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nSplitLayer][0], blobWork, true, 2e-08));

                int nLn1Layer = 13;
                blobVal.LoadFromNumpy(strPath + "grad_enc.ln.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nLn1Layer][0], blobWork, true, 1e-08));

                int nEncOutLayer = 12;
                blobVal.LoadFromNumpy(strPath + "grad_enc.x5.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEncOutLayer - 0][0], blobWork, true, 1e-08));

                blobVal.LoadFromNumpy(strPath + "grad_enc.x4.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEncOutLayer - 1][0], blobWork, true, 1e-08));

                blobVal.LoadFromNumpy(strPath + "grad_enc.x3.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEncOutLayer - 2][0], blobWork, true, 1e-08));

                blobVal.LoadFromNumpy(strPath + "grad_enc.x2.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEncOutLayer - 3][0], blobWork, true, 1e-08));

                blobVal.LoadFromNumpy(strPath + "grad_enc.x1.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEncOutLayer - 4][0], blobWork, true, 2e-08));

                blobVal.LoadFromNumpy(strPath + "grad_enc.x0.in.npy", true);
                Trace.Assert(blobVal.Compare(net.bottom_vecs[nEncOutLayer - 5][0], blobWork, true, 4e-08));

                int nDecPosLayer = 6;
                blobVal.LoadFromNumpy(strPath + "grad_transformer.trg_input_pos.npy", true);
                Trace.Assert(blobVal.Compare(net.top_vecs[nDecPosLayer][0], blobWork, true, 3e-07));

                int nEncPosLayer = 5;
                blobVal.LoadFromNumpy(strPath + "grad_transformer.src_input_pos.npy", true);
                Trace.Assert(blobVal.Compare(net.top_vecs[nEncPosLayer][0], blobWork, true, 4e-08));

                int nDecEmbLayer = 4;
                blobVal.LoadFromNumpy(strPath + "grad_transformer.trg_input_emb.npy", true);
                Trace.Assert(blobVal.Compare(net.top_vecs[nDecEmbLayer][0], blobWork, true, 5e-06));

                int nEncEmbLayer = 3;
                blobVal.LoadFromNumpy(strPath + "grad_transformer.src_input_emb.npy", true);
                Trace.Assert(blobVal.Compare(net.top_vecs[nEncEmbLayer][0], blobWork, true, 8e-07));

                // Train a few iterations.
                mycaffe.Train(5);

                
                //-------------------------------------
                // Verify training and testing weights
                //-------------------------------------

                Net<T> trainNet = mycaffe.GetInternalNet(Phase.TRAIN);
                Net<T> testNet = mycaffe.GetInternalNet(Phase.TEST);

                m_log.CHECK_EQ(trainNet.learnable_parameters.Count, testNet.learnable_parameters.Count, "The train and test nets should have an equal number of learnable parameters!");
                for (int i = 0; i < trainNet.learnable_parameters.Count; i++)
                {
                    Blob<T> blobTrain = trainNet.learnable_parameters[i];
                    Blob<T> blobTest = testNet.learnable_parameters[i];

                    long lGpuDataTrain = blobTrain.gpu_data;
                    long lGpuDiffTrain = blobTrain.gpu_diff;
                    long lGpuDataTest = blobTest.gpu_data;
                    long lGpuDiffTest = blobTest.gpu_diff;

                    m_log.CHECK_EQ(lGpuDataTrain, lGpuDataTest, "The gpu data for train and test should be sharing data!");
                    m_log.CHECK_EQ(lGpuDiffTrain, lGpuDiffTest, "The gpu diff for train and test should be sharing diff!");

                    Trace.Assert(blobTrain.Compare(blobTest, blobWork, false, 1e-10));
                }
            }
            finally
            {
                if (evtCancel != null)
                    evtCancel.Dispose();
                
                dispose(ref blobEncIn);
                dispose(ref blobDecIn);
                dispose(ref blobDecOut);
                dispose(ref blobEncMask);
                dispose(ref blobDecMask);
                dispose(ref blobLoss);
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (mycaffe != null)
                    mycaffe.Dispose();
            }
        }
    }
}
