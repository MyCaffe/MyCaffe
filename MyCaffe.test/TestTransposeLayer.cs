﻿using System;
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
        public void TestForwardEx()
        {
            TransposeLayerTest test = new TransposeLayerTest();

            try
            {
                foreach (ITransposeLayerTest t in test.Tests)
                {
                    t.TestForwardEx();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardBackward()
        {
            TransposeLayerTest test = new TransposeLayerTest();

            try
            {
                foreach (ITransposeLayerTest t in test.Tests)
                {
                    t.TestForwardBackward();
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
        void TestForwardEx();
        void TestForwardBackward();
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

            try
            {
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
            finally
            {
                layer.Dispose();
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

            try
            {
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
            finally
            {
                layer.Dispose();
            }
        }

        private string loadTestData2()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\trfb2\\";
            string strFileName = "_transformer_test2";

            strFileName += ".zip";

            string strTestPath = "test\\iter_0";
            string strTestFile = "1_x.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }

        public void TestForwardEx()
        {
            List<int> rgDim = new List<int>() { 0, 2, 1, 3 };
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSPOSE);
            p.transpose_param.dim = new List<int>(rgDim);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            string strPath = loadTestData2();
            Blob<T> blobVal = new Blob<T>(m_cuda, m_log);
            Blob<T> blobWork = new Blob<T>(m_cuda, m_log);
            Blob<T> blobYexp = new Blob<T>(m_cuda, m_log);

            try
            {
                m_blob_bottom.LoadFromNumpy(strPath + "blk0.csa.q1.npy");
                blobYexp.LoadFromNumpy(strPath + "blk0.csa.qt.npy");

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

                verify(blobYexp, m_blob_top, false);
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);
                dispose(ref blobYexp);

                layer.Dispose();
            }
        }

        public void TestForwardBackward()
        {
            float[] rgK = 
            {   
                 1.1544715e-08f,  1.10176614e-07f,  1.3792217e-08f,
                 1.0969044e-08f,  6.1579705e-08f,  -7.788405e-08f,
                -1.616798e-08f,  -7.6032656e-08f,   4.7443237e-07f,
                -6.3457657e-09f, -9.572371e-08f,   -4.1034065e-07f
            };
            float[] rgKt =
            {
                1.1544715e-08f,  1.0969044e-08f, -1.616798e-08f,  -6.3457657e-09f,
                1.10176614e-07f, 6.1579705e-08f, -7.6032656e-08f, -9.572371e-08f,
                1.3792217e-08f, -7.788405e-08f,   4.7443237e-07f, -4.1034065e-07f
            };

            List<int> rgDim = new List<int>() { 0, 2, 1, 3 };
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSPOSE);
            p.transpose_param.dim = new List<int>(rgDim);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_blob_bottom.Reshape(1, 4, 3, 1);
                List<int> rgExpectedTopShape = new List<int>() { 1, 3, 4, 1 };                

                layer.Setup(BottomVec, TopVec);

                // Test Forward
                m_blob_bottom.mutable_cpu_data = convert(rgK);
                layer.Forward(BottomVec, TopVec);                
              
                m_log.CHECK_EQ(m_blob_bottom.count(), m_blob_top.count(), "The top and bottom should have the same count!");
                m_log.CHECK_EQ(m_blob_bottom.num_axes, rgDim.Count, "The bottom must have the same number of axes as the rgDim!");
                m_log.CHECK_EQ(m_blob_top.num_axes, rgDim.Count, "The bottom must have the same number of axes as the rgDim!");

                for (int i = 0; i < rgExpectedTopShape.Count; i++)
                {
                    int nDim = m_blob_top.shape()[i];

                    m_log.CHECK_EQ(nDim, rgExpectedTopShape[i], "The top dimension at index " + i.ToString() + " is not correct!");
                }

                float[] rgExpected = rgKt;
                float[] rgActual = convertF(m_blob_top.mutable_cpu_data);

                for (int i = 0; i < rgActual.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
                    float fErr = 0.0000001f;

                    m_log.EXPECT_NEAR(fActual, fExpected, fErr, "The values do not match!");
                }

                // Test Backward
                m_blob_top.mutable_cpu_diff = convert(rgKt);
                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                rgExpected = rgK;
                rgActual = convertF(m_blob_bottom.mutable_cpu_diff);

                for (int i = 0; i < rgActual.Length; i++)
                {
                    float fExpected = rgExpected[i];
                    float fActual = rgActual[i];
                    float fErr = 0.0000001f;

                    m_log.EXPECT_NEAR(fActual, fExpected, fErr, "The values do not match!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            List<int> rgDim = new List<int>() { 0, 2, 1 };
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSPOSE);
            p.transpose_param.dim = new List<int>(rgDim);

            Fill(m_blob_bottom);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
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
            finally
            {
                layer.Dispose();
            }
        }
    }
}
