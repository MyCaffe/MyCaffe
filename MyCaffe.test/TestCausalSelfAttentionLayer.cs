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

namespace MyCaffe.test
{
    [TestClass]
    public class TestCausalSelfAttentionLayer
    {
        [TestMethod]
        public void TestForwardPico()
        {
            CausalSelfAttentionLayerTest test = new CausalSelfAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ICausalSelfAttentionLayerTest t in test.Tests)
                {
                    t.TestForwardPico(1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientPico()
        {
            CausalSelfAttentionLayerTest test = new CausalSelfAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ICausalSelfAttentionLayerTest t in test.Tests)
                {
                    t.TestGradientPico(1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardPico()
        {
            CausalSelfAttentionLayerTest test = new CausalSelfAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ICausalSelfAttentionLayerTest t in test.Tests)
                {
                    t.TestBackwardPico(1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardPico3()
        {
            CausalSelfAttentionLayerTest test = new CausalSelfAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ICausalSelfAttentionLayerTest t in test.Tests)
                {
                    t.TestForwardPico(3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientPico3()
        {
            CausalSelfAttentionLayerTest test = new CausalSelfAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ICausalSelfAttentionLayerTest t in test.Tests)
                {
                    t.TestGradientPico(3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestForwardMini()
        {
            CausalSelfAttentionLayerTest test = new CausalSelfAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ICausalSelfAttentionLayerTest t in test.Tests)
                {
                    t.TestForwardMini();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientMini()
        {
            CausalSelfAttentionLayerTest test = new CausalSelfAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ICausalSelfAttentionLayerTest t in test.Tests)
                {
                    t.TestGradientMini();
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
        void TestForwardPico(int nHeads);
        void TestGradientPico(int nHeads);
        void TestBackwardPico(int nHeads);
        void TestForwardMini();
        void TestGradientMini();
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
            base.dispose();
        }

        public Tuple<List<int>, float[]> Fill(string strGpt, string strName, Log log, CausalSelfAttentionParameter p)
        {
            string strFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\" + strGpt + "\\" + strName + ".txt";
            string[] rgstrLines = File.ReadAllLines(strFile);
            string strSize = rgstrLines[0].Trim('#', ' ', '(', ')', ',');
            string[] rgstrSize = strSize.Split(',');
            List<int> rgnShape = rgstrSize.Select(p1 => int.Parse(p1)).ToList();
            List<float> rgfVal = new List<float>();
            
            while (rgnShape.Count < 4)
            {
                rgnShape.Add(1);
            }

            int nCount = 1;
            foreach (int nDim in rgnShape)
            {
                nCount *= nDim;
            }

            for (int i = 1; i < rgstrLines.Length; i++)
            {
                string[] rgstrVals = rgstrLines[i].Split(' ');

                for (int j = 0; j < rgstrVals.Length; j++)
                {
                    string strVal = rgstrVals[j].Trim();

                    if (!string.IsNullOrEmpty(strVal))
                    {
                        float fVal = float.Parse(strVal);
                        rgfVal.Add(fVal);
                    }
                }
            }

            log.CHECK_EQ(rgfVal.Count, nCount, "The bottom count does not match the number of values read in!");

            float[] rgf = rgfVal.ToArray();

            return new Tuple<List<int>, float[]>(rgnShape, rgf);
        }

        public void TestForwardPico(int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CAUSAL_SELF_ATTENTION);
            p.causal_self_attention_param.heads = nHeads;
            p.causal_self_attention_param.embed = 3;
            p.causal_self_attention_param.block_size = 4;
            p.causal_self_attention_param.attn_dropout = 0.0;
            p.causal_self_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-pico";
                if (nHeads > 1)
                    strModel += nHeads.ToString();
                
                m_log.CHECK(layer.type == LayerParameter.LayerType.CAUSAL_SELF_ATTENTION, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill(strModel, "x", m_log, p.causal_self_attention_param);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);
                
                Tuple<List<int>, float[]> y = Fill(strModel, "y", m_log, p.causal_self_attention_param);
                m_blobY.Reshape(y.Item1);
                m_blobY.mutable_cpu_data = convert(y.Item2);
                
                Tuple<List<int>, float[]> attnBias = Fill(strModel, "attn_bias", m_log, p.causal_self_attention_param);
                Tuple<List<int>, float[]> attnWt = Fill(strModel, "attn_weight", m_log, p.causal_self_attention_param);
                Tuple<List<int>, float[]> projBias = Fill(strModel, "proj_bias", m_log, p.causal_self_attention_param);
                Tuple<List<int>, float[]> projWt = Fill(strModel, "proj_weight", m_log, p.causal_self_attention_param);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].mutable_cpu_data = convert(attnWt.Item2);
                layer.blobs[1].mutable_cpu_data = convert(attnBias.Item2);
                layer.blobs[2].mutable_cpu_data = convert(projWt.Item2);
                layer.blobs[3].mutable_cpu_data = convert(projBias.Item2);

                layer.Forward(BottomVec, TopVec);

                // Now, check values
                float[] rgExpected = convertF(m_blobY.mutable_cpu_data);
                float[] rgActual = convertF(m_blob_top.mutable_cpu_data);

                for (int i = 0; i < rgExpected.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
                    float fErr = 0.00000001f;

                    m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fErr, "The values are not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackwardPico(int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CAUSAL_SELF_ATTENTION);
            p.causal_self_attention_param.heads = nHeads;
            p.causal_self_attention_param.embed = 3;
            p.causal_self_attention_param.block_size = 4;
            p.causal_self_attention_param.attn_dropout = 0.0;
            p.causal_self_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-pico";
                if (nHeads > 1)
                    strModel += nHeads.ToString();

                m_log.CHECK(layer.type == LayerParameter.LayerType.CAUSAL_SELF_ATTENTION, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill(strModel, "x", m_log, p.causal_self_attention_param);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);

                Tuple<List<int>, float[]> y_grad = Fill(strModel, "1_grad_y", m_log, p.causal_self_attention_param);
                
                Tuple<List<int>, float[]> x_grad = Fill(strModel, "12_grad_x", m_log, p.causal_self_attention_param);
                Tuple<List<int>, float[]> attnBias = Fill(strModel, "attn_bias", m_log, p.causal_self_attention_param);
                Tuple<List<int>, float[]> attnWt = Fill(strModel, "attn_weight", m_log, p.causal_self_attention_param);
                Tuple<List<int>, float[]> projBias = Fill(strModel, "proj_bias", m_log, p.causal_self_attention_param);
                Tuple<List<int>, float[]> projWt = Fill(strModel, "proj_weight", m_log, p.causal_self_attention_param);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].mutable_cpu_data = convert(attnWt.Item2);
                layer.blobs[1].mutable_cpu_data = convert(attnBias.Item2);
                layer.blobs[2].mutable_cpu_data = convert(projWt.Item2);
                layer.blobs[3].mutable_cpu_data = convert(projBias.Item2);

                layer.Forward(BottomVec, TopVec);

                m_blob_top.mutable_cpu_diff = convert(y_grad.Item2);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);
                
                // Now, check values
                float[] rgExpected = x_grad.Item2;
                float[] rgActual = convertF(m_blob_bottom.mutable_cpu_diff);

                for (int i = 0; i < rgExpected.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
                    float fErr = 0.00000001f;

                    m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fErr, "The values are not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradientPico(int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CAUSAL_SELF_ATTENTION);
            p.causal_self_attention_param.heads = nHeads;
            p.causal_self_attention_param.embed = 3;
            p.causal_self_attention_param.block_size = 4;
            p.causal_self_attention_param.attn_dropout = 0.0;
            p.causal_self_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-pico";
                if (nHeads > 1)
                    strModel += nHeads.ToString();
                
                m_log.CHECK(layer.type == LayerParameter.LayerType.CAUSAL_SELF_ATTENTION, "The layer type is incorrect!");

                Tuple<List<int>, float[]> data = Fill(strModel, "x", m_log, p.causal_self_attention_param);
                m_blob_bottom.Reshape(data.Item1);
                m_blob_bottom.mutable_cpu_data = convert(data.Item2);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 0.01, 0.01);
                checker.CheckGradient(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardMini()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CAUSAL_SELF_ATTENTION);
            p.causal_self_attention_param.heads = 6;
            p.causal_self_attention_param.embed = 192;
            p.causal_self_attention_param.block_size = 128;
            p.causal_self_attention_param.attn_dropout = 0.0;
            p.causal_self_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.CAUSAL_SELF_ATTENTION, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill("gpt-mini", "x", m_log, p.causal_self_attention_param);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);

                Tuple<List<int>, float[]> y = Fill("gpt-mini", "y", m_log, p.causal_self_attention_param);
                m_blobY.Reshape(y.Item1);
                m_blobY.mutable_cpu_data = convert(y.Item2);

                Tuple<List<int>, float[]> attnBias = Fill("gpt-mini", "attn_bias", m_log, p.causal_self_attention_param);
                Tuple<List<int>, float[]> attnWt = Fill("gpt-mini", "attn_weight", m_log, p.causal_self_attention_param);
                Tuple<List<int>, float[]> projBias = Fill("gpt-mini", "proj_bias", m_log, p.causal_self_attention_param);
                Tuple<List<int>, float[]> projWt = Fill("gpt-mini", "proj_weight", m_log, p.causal_self_attention_param);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].mutable_cpu_data = convert(attnWt.Item2);
                layer.blobs[1].mutable_cpu_data = convert(attnBias.Item2);
                layer.blobs[2].mutable_cpu_data = convert(projWt.Item2);
                layer.blobs[3].mutable_cpu_data = convert(projBias.Item2);

                layer.Forward(BottomVec, TopVec);

                // Now, check values
                float[] rgExpected = convertF(m_blobY.mutable_cpu_data);
                float[] rgActual = convertF(m_blob_top.mutable_cpu_data);

                for (int i = 0; i < rgExpected.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
                    float fErr = 0.0000001f;

                    m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fErr, "The values are not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradientMini()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CAUSAL_SELF_ATTENTION);
            p.causal_self_attention_param.heads = 6;
            p.causal_self_attention_param.embed = 192;
            p.causal_self_attention_param.block_size = 128;
            p.causal_self_attention_param.attn_dropout = 0.0;
            p.causal_self_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.CAUSAL_SELF_ATTENTION, "The layer type is incorrect!");

                Tuple<List<int>, float[]> data = Fill("gpt-mini", "x", m_log, p.causal_self_attention_param);
                m_blob_bottom.Reshape(data.Item1);
                m_blob_bottom.mutable_cpu_data = convert(data.Item2);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 0.01, 0.01);
                checker.CheckGradient(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
