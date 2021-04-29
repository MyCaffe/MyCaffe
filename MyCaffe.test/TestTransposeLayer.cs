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
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe.data;
using MyCaffe.layers.beta;

/// <summary>
/// Testing the transpose layer.
/// 
/// Transpose Layer - this layer permutes and transposes the input data based on the TransposeLayerParameter settings.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTransposeLayer
    {
        [TestMethod]
        public void TestForward()
        {
            TransposeLayerTest test = new TransposeLayerTest();

            try
            {
                foreach (ITransposeLayerTest t in test.Tests)
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
        public void TestForward2()
        {
            TransposeLayerTest test = new TransposeLayerTest();

            try
            {
                foreach (ITransposeLayerTest t in test.Tests)
                {
                    t.TestForward2();
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
            TransposeLayerTest test = new TransposeLayerTest();

            try
            {
                foreach (ITransposeLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ITransposeLayerTest : ITest
    {
        void TestForward();
        void TestForward2();
        void TestGradient();
    }

    class TransposeLayerTest : TestBase
    {
        public TransposeLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Transpose Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TransposeLayerTest<double>(strName, nDeviceID, engine);
            else
                return new TransposeLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class TransposeLayerTest<T> : TestEx<T>, ITransposeLayerTest
    {
        public TransposeLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        private void Fill(Blob<T> b)
        {
            List<int> rgShape = new List<int>() { 1, 2, 3 };
            b.Reshape(rgShape);

            List<float> rgF = new List<float>();
            rgF.Add(1.0f);
            rgF.Add(1.1f);
            rgF.Add(1.2f);

            rgF.Add(2.0f);
            rgF.Add(2.1f);
            rgF.Add(2.2f);

            b.mutable_cpu_data = convert(rgF.ToArray());
        }

        private void Fill2(Blob<T> b)
        {
            List<int> rgShape = new List<int>() { 2, 3, 3 };
            b.Reshape(rgShape);

            List<float> rgF = new List<float>();
            rgF.Add(1.10f);
            rgF.Add(1.11f);
            rgF.Add(1.12f);

            rgF.Add(1.20f);
            rgF.Add(1.21f);
            rgF.Add(1.22f);

            rgF.Add(1.30f);
            rgF.Add(1.31f);
            rgF.Add(1.32f);

            rgF.Add(2.10f);
            rgF.Add(2.11f);
            rgF.Add(2.12f);

            rgF.Add(2.20f);
            rgF.Add(2.21f);
            rgF.Add(2.22f);

            rgF.Add(2.30f);
            rgF.Add(2.31f);
            rgF.Add(2.32f);

            b.mutable_cpu_data = convert(rgF.ToArray());
        }

        public void TestForward()
        {
            List<int> rgDim = new List<int>() { 0, 2, 1 };
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSPOSE);
            p.transpose_param.dim = new List<int>(rgDim);

            Fill(m_blob_bottom);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_bottom.count(), m_blob_top.count(), "The top and bottom should have the same count!");
            m_log.CHECK_EQ(m_blob_bottom.num_axes, rgDim.Count, "The bottom must have the same number of axes as the rgDim!");
            m_log.CHECK_EQ(m_blob_top.num_axes, rgDim.Count, "The bottom must have the same number of axes as the rgDim!");

            for (int i = 0; i < rgDim.Count; i++)
            {
                int nAxis = rgDim[i];
                int nDim = m_blob_bottom.shape()[nAxis];

                m_log.CHECK_EQ(m_blob_top.shape()[i], nDim, "The top dimension at index " + i.ToString() + " is not correct!");
            }

            List<float> rgExpectedF = new List<float>();
            rgExpectedF.Add(1.0f);
            rgExpectedF.Add(2.0f);

            rgExpectedF.Add(1.1f);
            rgExpectedF.Add(2.1f);
           
            rgExpectedF.Add(1.2f);
            rgExpectedF.Add(2.2f);

            float[] rgF = convertF(m_blob_top.mutable_cpu_data);

            for (int i = 0; i < rgF.Length; i++)
            {
                m_log.EXPECT_EQUAL<float>(rgF[i], rgExpectedF[i], "The values do not match!");
            }
        }

        public void TestForward2()
        {
            List<int> rgDim = new List<int>() { 1, 0, 2 };
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSPOSE);
            p.transpose_param.dim = new List<int>(rgDim);

            Fill2(m_blob_bottom);
            float[] rgData = convertF(m_blob_bottom.mutable_cpu_data);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_bottom.count(), m_blob_top.count(), "The top and bottom should have the same count!");
            m_log.CHECK_EQ(m_blob_bottom.num_axes, rgDim.Count, "The bottom must have the same number of axes as the rgDim!");
            m_log.CHECK_EQ(m_blob_top.num_axes, rgDim.Count, "The bottom must have the same number of axes as the rgDim!");

            for (int i = 0; i < rgDim.Count; i++)
            {
                int nAxis = rgDim[i];
                int nDim = m_blob_bottom.shape()[nAxis];

                m_log.CHECK_EQ(m_blob_top.shape()[i], nDim, "The top dimension at index " + i.ToString() + " is not correct!");
            }

            float[] rgExpected = SimpleDatum.Transpose(rgData, m_blob_bottom.num, m_blob_bottom.channels, m_blob_bottom.count(2));
            float[] rgActual = convertF(m_blob_top.mutable_cpu_data);

            for (int i = 0; i < rgActual.Length; i++)
            {
                m_log.EXPECT_EQUAL<float>(rgActual[i], rgExpected[i], "The values do not match!");
            }
        }

        public void TestGradient()
        {
            List<int> rgDim = new List<int>() { 0, 2, 1 };
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSPOSE);
            p.transpose_param.dim = new List<int>(rgDim);

            Fill(m_blob_bottom);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_bottom.count(), m_blob_top.count(), "The top and bottom should have the same count!");
            m_log.CHECK_EQ(m_blob_bottom.num_axes, rgDim.Count, "The bottom must have the same number of axes as the rgDim!");
            m_log.CHECK_EQ(m_blob_top.num_axes, rgDim.Count, "The bottom must have the same number of axes as the rgDim!");

            for (int i = 0; i < rgDim.Count; i++)
            {
                int nAxis = rgDim[i];
                int nDim = m_blob_bottom.shape()[nAxis];

                m_log.CHECK_EQ(m_blob_top.shape()[i], nDim, "The top dimension at index " + i.ToString() + " is not correct!");
            }

            List<float> rgExpectedF = new List<float>();
            rgExpectedF.Add(1.0f);
            rgExpectedF.Add(2.0f);

            rgExpectedF.Add(1.1f);
            rgExpectedF.Add(2.1f);

            rgExpectedF.Add(1.2f);
            rgExpectedF.Add(2.2f);

            float[] rgF = convertF(m_blob_top.mutable_cpu_data);

            for (int i = 0; i < rgF.Length; i++)
            {
                m_log.EXPECT_EQUAL<float>(rgF[i], rgExpectedF[i], "The values do not match!");
            }

            m_cuda.copy(Top.count(), Top.gpu_data, Top.mutable_gpu_diff);
            layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

            float[] rgBtmData = convertF(Bottom.mutable_cpu_data);
            float[] rgBtmDiff = convertF(Bottom.mutable_cpu_diff);

            for (int i = 0; i < rgBtmData.Length; i++)
            {
                m_log.EXPECT_EQUAL<float>(rgBtmData[i], rgBtmDiff[i], "The bottom data and diff should be equal.");
            }
        }
    }
}
