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
using MyCaffe.layers.gpt;

namespace MyCaffe.test
{
    [TestClass]
    public class TestCausalSelfAttentionLayer
    {
        [TestMethod]
        public void TestForward()
        {
            CausalSelfAttentionLayerTest test = new CausalSelfAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ICausalSelfAttentionLayerTest t in test.Tests)
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
        public void TestBackward()
        {
            CausalSelfAttentionLayerTest test = new CausalSelfAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ICausalSelfAttentionLayerTest t in test.Tests)
                {
                    t.TestBackward();
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
            CausalSelfAttentionLayerTest test = new CausalSelfAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ICausalSelfAttentionLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient2()
        {
            CausalSelfAttentionLayerTest test = new CausalSelfAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ICausalSelfAttentionLayerTest t in test.Tests)
                {
                    t.TestGradient2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ICausalSelfAttentionLayerTest : ITest
    {
        void TestForward();
        void TestBackward();
        void TestGradient();
        void TestGradient2();
    }

    class CausalSelfAttentionLayerTest : TestBase
    {
        public CausalSelfAttentionLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Causal Self Attention Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new CausalSelfAttentionLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new CausalSelfAttentionLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    /// <summary>
    /// Auto test for CausalSelfAttentionLayer see remarks for generating a new set of data.
    /// </summary>
    /// <remarks>
    /// Pre generated data is downloaded for each test.  However, to re-generate the testing data follow
    /// these steps:
    /// 
    /// Test Project: minGPT
    /// 1.) constants.py, set 'mycaffe_softmax = False', 'mycaffe_innerproduct = False'
    /// 2.) run 'main.py' up to just past 'self.loss.backward()' on line 103 of 'trainer.py'
    /// 3.) constants.py, change 'mycaffe_softmax = True' 'mycaffe_innerproduct = True'
    /// 4.) run 'test_causalselfattention.py'
    /// </remarks>
    /// <typeparam name="T"></typeparam>
    class CausalSelfAttentionLayerTest2<T> : TestEx<T>, ICausalSelfAttentionLayerTest
    {
        Blob<T> m_blobY;

        public CausalSelfAttentionLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobY = new Blob<T>(m_cuda, m_log);
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        protected override void dispose()
        {
            dispose(ref m_blobY);
            base.dispose();
        }

        private string loadTestData1()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\csa\\";
            string strFileName = "_causalself_test.zip";
            string strTestPath = "test\\iter_0";
            string strTestFile = "1_x_emb.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }

        private void load_state(Layer<T> layer, string strPath)
        {
            layer.blobs[0].LoadFromNumpy(strPath + "blk0.attn.c_attn.weight.npy");
            layer.blobs[1].LoadFromNumpy(strPath + "blk0.attn.c_attn.bias.npy");
            layer.blobs[2].LoadFromNumpy(strPath + "blk0.attn.c_proj.weight.npy");
            layer.blobs[3].LoadFromNumpy(strPath + "blk0.attn.c_proj.bias.npy");
        }

        public void TestForward()
        {
            string strPath = loadTestData1();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CAUSAL_SELF_ATTENTION);
            p.causal_self_attention_param.heads = 6;
            p.causal_self_attention_param.embed = 192;
            p.causal_self_attention_param.block_size = 128;
            p.causal_self_attention_param.attn_dropout = 0.0;
            p.causal_self_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);
            Blob<T> blobVal = new Blob<T>(m_cuda, m_log);

            try
            {
                BlobCollection<T> colBtm = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();
                
                blobX.LoadFromNumpy(strPath + "1_x_emb.npy");
                colBtm.Add(blobX);
                colTop.Add(blobY);

                layer.Setup(colBtm, colTop);
                load_state(layer, strPath);

                layer.Forward(colBtm, colTop);

                blobVal.LoadFromNumpy(strPath + "12_out1.npy");
                verify(blobY, blobVal, false, 4e-09);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);
                dispose(ref blobVal);
                layer.Dispose();
            }
        }

        public void TestBackward()
        {
            string strPath = loadTestData1();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CAUSAL_SELF_ATTENTION);
            p.causal_self_attention_param.heads = 6;
            p.causal_self_attention_param.embed = 192;
            p.causal_self_attention_param.block_size = 128;
            p.causal_self_attention_param.attn_dropout = 0.0;
            p.causal_self_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);
            Blob<T> blobVal = new Blob<T>(m_cuda, m_log);

            try
            {
                BlobCollection<T> colBtm = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();

                blobX.LoadFromNumpy(strPath + "1_x_in.npy");
                colBtm.Add(blobX);
                colTop.Add(blobY);

                layer.Setup(colBtm, colTop);
                load_state(layer, strPath);

                layer.Forward(colBtm, colTop);

                blobVal.LoadFromNumpy(strPath + "12_out1.npy");
                verify(blobY, blobVal, false, 4e-09);

                colTop[0].LoadFromNumpy(strPath + "grad_12_out1.npy", true);
                layer.Backward(colTop, new List<bool>() { true }, colBtm);

                blobVal.LoadFromNumpy(strPath + "grad_1_x_in.npy", true);
                verify(colBtm[0], blobVal, true, 3e-12);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);
                dispose(ref blobVal);
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            string strPath = loadTestData1();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CAUSAL_SELF_ATTENTION);
            p.causal_self_attention_param.heads = 6;
            p.causal_self_attention_param.embed = 192;
            p.causal_self_attention_param.block_size = 128;
            p.causal_self_attention_param.attn_dropout = 0.0;
            p.causal_self_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);

            try
            {
                BlobCollection<T> colBtm = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();

                blobX.LoadFromNumpy(strPath + "1_x_in.npy");
                colBtm.Add(blobX);
                colTop.Add(blobY);

                layer.Setup(colBtm, colTop);
                load_state(layer, strPath);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradient(layer, colBtm, colTop, -1, 1, 0.05);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);
                layer.Dispose();
            }
        }

        public void TestGradient2()
        {
            string strPath = loadTestData1();
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CAUSAL_SELF_ATTENTION);
            p.causal_self_attention_param.heads = 6;
            p.causal_self_attention_param.embed = 192;
            p.causal_self_attention_param.block_size = 128;
            p.causal_self_attention_param.attn_dropout = 0.0;
            p.causal_self_attention_param.resid_dropout = 0.0;
            Layer<T> layer = new CausalSelfAttentionLayer2<T>(m_cuda, m_log, p);
            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);

            try
            {
                BlobCollection<T> colBtm = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();

                blobX.LoadFromNumpy(strPath + "1_x_in.npy");
                colBtm.Add(blobX);
                colTop.Add(blobY);

                layer.Setup(colBtm, colTop);
                load_state(layer, strPath);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradient(layer, colBtm, colTop, -1, 1, 0.05);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);
                layer.Dispose();
            }
        }
    }
}
