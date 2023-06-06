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
    public class TestGPT_TestGeluLayer
    {
        [TestMethod]
        public void TestForwardBert()
        {
            GeluLayerTest2 test = new GeluLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IGeluLayerTest2 t in test.Tests)
                {
                    t.TestForward(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardBert()
        {
            GeluLayerTest2 test = new GeluLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IGeluLayerTest2 t in test.Tests)
                {
                    t.TestBackward(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientBert()
        {
            GeluLayerTest2 test = new GeluLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IGeluLayerTest2 t in test.Tests)
                {
                    t.TestGradient(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientDefault()
        {
            GeluLayerTest2 test = new GeluLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IGeluLayerTest2 t in test.Tests)
                {
                    t.TestGradient(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardPico()
        {
            GeluLayerTest2 test = new GeluLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IGeluLayerTest2 t in test.Tests)
                {
                    t.TestForwardPico(false, 1);
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
            GeluLayerTest2 test = new GeluLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IGeluLayerTest2 t in test.Tests)
                {
                    t.TestBackwardPico(false, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IGeluLayerTest2 : ITest
    {
        void TestForward(bool bBertVersion);
        void TestBackward(bool bBertVersion);
        void TestGradient(bool bBertVersion);
        void TestForwardPico(bool bBatch, int nHeads);
        void TestBackwardPico(bool bBatch, int nHeads);
    }

    class GeluLayerTest2 : TestBase
    {
        public GeluLayerTest2(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("GELU Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new GeluLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new GeluLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class GeluLayerTest2<T> : TestEx<T>, IGeluLayerTest2
    {
        public GeluLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        protected override void dispose()
        {
            base.dispose();
        }

        /// <summary>
        /// Calculate the native GELU
        /// </summary>
        /// <param name="x">Specifies the input.</param>
        /// <returns>The calculated Gelu value is returned.</returns>
        /// <remarks>
        /// Computes the mish non-linearity @f$ y  = 0.5 * x * (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x^3))) @f$.
        /// @see [Github - Karpathy: NewGELU, line 21](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) by Karpathy, 2022.
        /// </remarks>
        protected double gelu_native(double x)
        {
            return 0.5 * x * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.Pow(x, 3))));
        }

        /// <summary>
        /// Calculate the native GELU gradient
        /// </summary>
        /// <param name="x">Specifies the input.</param>
        /// <returns>The calculated Gelu value is returned.</returns>
        /// <remarks>
        /// Computes the gelu non-linearity @f$ y  = 0.5 * x * (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x^3))) @f$.
        ///                                 @f$ y' = 0.5 * tanh(0.797885 * (x + 0.044715 * x^3)) + 
        ///                                          (0.0535161 * x^3 + 0.398942 * x) * sech^2(0.797885 * (x + 0.044715 * x^3)) + 0.5 @f$
        /// Note, see Wolfram Alpha with 'derivative of @f$ d/dx  = 0.5 * x * (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x^3))) @f$                                         
        protected double gelu_native_grad(double x)
        {
            double x3 = Math.Pow(x, 3);
            double tanh = Math.Tanh(0.797885 * (x + 0.044715 * x3));
            double sech = 1.0 / Math.Cosh(0.797885 * (x + 0.044715 * x3));

            return 0.5 * tanh + (0.0535161 * x3 + 0.398942 * x) * sech * sech + 0.5;
        }

        public void TestForward(bool bBertVersion, double dfFillerStd)
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = dfFillerStd;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GELU);
            p.gelu_param.enable_bert_version = bBertVersion;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.GELU, "The layer type is incorrect!");

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // Now, check values
                double[] rgBottomData = convert(Bottom.update_cpu_data());
                double[] rgTopData = convert(Top.update_cpu_data());
                double dfMinPrecision = 1e-5;

                for (int i = 0; i < Bottom.count(); i++)
                {
                    double dfExpectedValue = gelu_native(rgBottomData[i]);
                    double dfPrecision = Math.Max(Math.Abs(dfExpectedValue * 1e-4), dfMinPrecision);
                    m_log.EXPECT_NEAR(dfExpectedValue, rgTopData[i], dfPrecision);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackward(bool bBertVersion)
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = 1.0;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GELU);
            p.gelu_param.enable_bert_version = bBertVersion;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.GELU, "The layer type is incorrect!");

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_blob_top.SetDiff(1.0);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                // Now, check values
                double[] rgBottomData = convert(Bottom.update_cpu_data());
                double[] rgBottomDiff = convert(Bottom.update_cpu_diff());
                double dfMinPrecision = 1e-5;

                for (int i = 0; i < Bottom.count(); i++)
                {
                    double dfExpectedValue = gelu_native_grad(rgBottomData[i]);
                    double dfActualValue = rgBottomDiff[i];
                    double dfPrecision = Math.Max(Math.Abs(dfExpectedValue * 1e-4), dfMinPrecision);
                    m_log.EXPECT_NEAR(dfExpectedValue, dfActualValue, dfPrecision);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackward(bool bBertVersion, double dfFillerStd)
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = dfFillerStd;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GELU);
            p.gelu_param.enable_bert_version = bBertVersion;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.GELU, "The layer type is incorrect!");

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientEltwise(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward(bool bBertVersion)
        {
            TestForward(bBertVersion, 1.0);
        }

        public void TestGradient(bool bBertVersion)
        {
            TestBackward(bBertVersion, 1.0);
        }

        public Tuple<List<int>, float[]> Fill(string strGpt, string strName, Log log, CausalSelfAttentionParameter p)
        {
            string strFile = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\gpt\\" + strGpt + "\\" + strName + ".txt";
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

        public void TestForwardPico(bool bBatch, int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GELU);
            p.gelu_param.enable_bert_version = true;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobY = null;

            try
            {
                blobY = new Blob<T>(m_cuda, m_log);

                string strModel = "gpt-pico-gelu";
                if (nHeads > 1)
                    strModel += nHeads.ToString();
                if (bBatch)
                    strModel += "B";

                m_log.CHECK(layer.type == LayerParameter.LayerType.GELU, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill(strModel, "1_x", m_log, p.causal_self_attention_param);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);

                Tuple<List<int>, float[]> y = Fill(strModel, "2_y", m_log, p.causal_self_attention_param);
                blobY.Reshape(y.Item1);
                blobY.mutable_cpu_data = convert(y.Item2);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // Now, check values
                float[] rgExpected = convertF(blobY.mutable_cpu_data);
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
                if (blobY != null)
                    blobY.Dispose();

                layer.Dispose();
            }
        }

        public void TestBackwardPico(bool bBatch, int nHeads)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GELU);
            p.gelu_param.enable_bert_version = true;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                string strModel = "gpt-pico-gelu";
                if (nHeads > 1)
                    strModel += nHeads.ToString();
                if (bBatch)
                    strModel += "B";

                m_log.CHECK(layer.type == LayerParameter.LayerType.GELU, "The layer type is incorrect!");

                Tuple<List<int>, float[]> x = Fill(strModel, "1_x", m_log, p.causal_self_attention_param);
                m_blob_bottom.Reshape(x.Item1);
                m_blob_bottom.mutable_cpu_data = convert(x.Item2);

                Tuple<List<int>, float[]> y_grad = Fill(strModel, "grad_1_y", m_log, p.causal_self_attention_param);
                Tuple<List<int>, float[]> x_grad = Fill(strModel, "grad_2_x", m_log, p.causal_self_attention_param);

                layer.Setup(BottomVec, TopVec);
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
                    float fErr = 1e-7f;

                    m_log.EXPECT_NEAR_FLOAT(fExpected, fActual, fErr, "The values are not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
