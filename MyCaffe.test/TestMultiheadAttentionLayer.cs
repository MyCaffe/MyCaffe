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

namespace MyCaffe.test
{
    [TestClass]
    public class TestMultiheadAttentionLayer
    {
        [TestMethod]
        public void TestForward()
        {
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
                {
                    t.TestForward(false, 1);
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
                    t.TestBackward(false, 1);
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
            MultiheadAttentionLayerTest test = new MultiheadAttentionLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultiheadAttentionLayerTest t in test.Tests)
                {
                    t.TestGradient(false, 1);
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
        void TestForward(bool bBatch, int nHeads);
        void TestBackward(bool bBatch, int nHeads);
        void TestGradient(bool bBatch, int nHeads);
    }

    class MultiheadAttentionLayerTest : TestBase
    {
        public MultiheadAttentionLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Causal Self Attention Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
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

        public MultiheadAttentionLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        public Tuple<List<int>, float[]> Fill(string strGpt, string strName, Log log, string strPass = "")
        {
            string strFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\gpt\\" + strGpt + "\\";

            if (!string.IsNullOrEmpty(strPass))
                strFile += strPass + "\\";

            strFile += strName + ".txt";

            string[] rgstrLines = File.ReadAllLines(strFile);
            string strSize = rgstrLines[0].Trim('#', ' ', '(', ')', ',');
            string[] rgstrSize = strSize.Split(',');
            List<int> rgnShape = new List<int>() { 1 };

            if (!string.IsNullOrEmpty(strSize))
                rgnShape = rgstrSize.Select(p1 => int.Parse(p1)).ToList();
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

        /// <summary>
        /// WORK IN PROGRESS
        /// </summary>
        public void TestForward(bool bBatch, int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION);
            p.multihead_attention_param.heads = nHeads;
            p.multihead_attention_param.embed = 3;
            p.multihead_attention_param.block_size = 4;
            p.multihead_attention_param.attn_dropout = 0.0;
            p.multihead_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-pico-csa";
                if (nHeads > 1)
                    strModel += nHeads.ToString();
                if (bBatch)
                    strModel += "B";

                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill(strModel, "x", m_log);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);
                
                Tuple<List<int>, float[]> y = Fill(strModel, "y", m_log);
                m_blobY.Reshape(y.Item1);
                m_blobY.mutable_cpu_data = convert(y.Item2);
                
                Tuple<List<int>, float[]> attnBias = Fill(strModel, "attn_bias", m_log);
                Tuple<List<int>, float[]> attnWt = Fill(strModel, "attn_weight", m_log);
                Tuple<List<int>, float[]> projBias = Fill(strModel, "attn_proj_bias", m_log);
                Tuple<List<int>, float[]> projWt = Fill(strModel, "attn_proj_weight", m_log);

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
                    float fErr = 1e-7f;

                    m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fErr, "The values are not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        /// <summary>
        /// WORK IN PROGRESS
        /// </summary>
        public void TestBackward(bool bBatch, int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION);
            p.multihead_attention_param.heads = nHeads;
            p.multihead_attention_param.embed = 3;
            p.multihead_attention_param.block_size = 4;
            p.multihead_attention_param.attn_dropout = 0.0;
            p.multihead_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-pico-csa";
                if (nHeads > 1)
                    strModel += nHeads.ToString();
                if (bBatch)
                    strModel += "B";

                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill(strModel, "x", m_log);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);

                Tuple<List<int>, float[]> y_grad = Fill(strModel, "grad_y", m_log, "iter_0");                
                Tuple<List<int>, float[]> x_grad = Fill(strModel, "grad_x", m_log, "iter_0");
                Tuple<List<int>, float[]> attnBias = Fill(strModel, "attn_bias", m_log);
                Tuple<List<int>, float[]> attnWt = Fill(strModel, "attn_weight", m_log);
                Tuple<List<int>, float[]> projBias = Fill(strModel, "attn_proj_bias", m_log);
                Tuple<List<int>, float[]> projWt = Fill(strModel, "attn_proj_weight", m_log);

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

        /// <summary>
        /// WORK IN PROGRESS
        /// </summary>
        public void TestGradient(bool bBatch, int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION);
            p.multihead_attention_param.heads = nHeads;
            p.multihead_attention_param.embed = 3;
            p.multihead_attention_param.block_size = 4;
            p.multihead_attention_param.attn_dropout = 0.0;
            p.multihead_attention_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-pico-csa";
                if (nHeads > 1)
                    strModel += nHeads.ToString();
                if (bBatch)
                    strModel += "B";

                m_log.CHECK(layer.type == LayerParameter.LayerType.CAUSAL_SELF_ATTENTION, "The layer type is incorrect!");

                Tuple<List<int>, float[]> data = Fill(strModel, "x", m_log);
                m_blob_bottom.Reshape(data.Item1);
                m_blob_bottom.mutable_cpu_data = convert(data.Item2);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 0.01, 0.001);
                checker.CheckGradient(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
