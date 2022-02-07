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
/// Testing the gather layer.
/// 
/// Gather Layer - this layer selects from the input data based on the GatherLayerParameter settings.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestGatherLayer
    {
        [TestMethod]
        public void TestForwardAxis0()
        {
            GatherLayerTest test = new GatherLayerTest();

            try
            {
                foreach (IGatherLayerTest t in test.Tests)
                {
                    t.TestForward(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAxis1()
        {
            GatherLayerTest test = new GatherLayerTest();

            try
            {
                foreach (IGatherLayerTest t in test.Tests)
                {
                    t.TestForward(1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAxis0()
        {
            GatherLayerTest test = new GatherLayerTest();

            try
            {
                foreach (IGatherLayerTest t in test.Tests)
                {
                    t.TestGradient(0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAxis1()
        {
            GatherLayerTest test = new GatherLayerTest();

            try
            {
                foreach (IGatherLayerTest t in test.Tests)
                {
                    t.TestGradient(1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IGatherLayerTest : ITest
    {
        void TestForward(int nAxis);
        void TestGradient(int nAxis);
    }

    class GatherLayerTest : TestBase
    {
        public GatherLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Gather Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new GatherLayerTest<double>(strName, nDeviceID, engine);
            else
                return new GatherLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class GatherLayerTest<T> : TestEx<T>, IGatherLayerTest
    {
        Blob<T> m_blobIndices;

        public GatherLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            m_blobIndices = new Blob<T>(m_cuda, m_log);
        }

        protected override void dispose()
        {
            m_blobIndices.Dispose();
            base.dispose();
        }

        public void TestForward(int nAxis)
        {
            switch (nAxis)
            {
                case 0:
                    testForwardAxis0();
                    break;

                case 1:
                    testForwardAxis1();
                    break;

                default:
                    throw new Exception("Currently tests exist for only axis 0 and 1.");
            }
        }

        /// <summary>
        /// Axis 0 test.
        /// </summary>
        /// <remarks>
        /// @see [https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather](ONNX Gather Axis=0 example)
        /// </remarks>
        private void testForwardAxis0()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GATHER);
            p.gather_param.axis = 0;
            GatherLayer<T> layer = new GatherLayer<T>(m_cuda, m_log, p);

            try
            {
                Bottom.Reshape(3, 2, 1, 1);
                float[] rgData = new float[] { 1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f };
                Bottom.mutable_cpu_data = convert(rgData);

                m_blobIndices.Reshape(2, 2, 1, 1);
                float[] rgIdx = new float[] { 0.0f, 1.0f, 1.0f, 2.0f };
                m_blobIndices.mutable_cpu_data = convert(rgIdx);

                BottomVec.Add(m_blobIndices);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, 2, "The top should have num = 2");
                m_log.CHECK_EQ(Top.channels, 2, "The top should have channels = 2");
                m_log.CHECK_EQ(Top.height, 2, "The top should have height = 2");
                m_log.CHECK_EQ(Top.width, 1, "The top should have width = 1");

                float[] rgResult = convertF(Top.mutable_cpu_data);
                float[] rgExpected = new float[] { 1.0f, 1.2f, 2.3f, 3.4f, 2.3f, 3.4f, 4.5f, 5.7f };
                m_log.CHECK_EQ(rgResult.Length, rgExpected.Length, "The result length is not as expected!");

                for (int i = 0; i < rgResult.Length; i++)
                {
                    m_log.CHECK_EQ(rgResult[i], rgExpected[i], "The item at index #" + i.ToString() + " is not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        /// <summary>
        /// Axis 1 test.
        /// </summary>
        /// <remarks>
        /// @see [https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather](ONNX Gather Axis=1 example)
        /// </remarks>
        private void testForwardAxis1()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GATHER);
            p.gather_param.axis = 1;
            GatherLayer<T> layer = new GatherLayer<T>(m_cuda, m_log, p);

            try
            {
                Bottom.Reshape(3, 3, 1, 1);
                float[] rgData = new float[] { 1.0f, 1.2f, 1.9f, 2.3f, 3.4f, 3.9f, 4.5f, 5.7f, 5.9f };
                Bottom.mutable_cpu_data = convert(rgData);

                m_blobIndices.Reshape(1, 2, 1, 1);
                float[] rgIdx = new float[] { 0.0f, 2.0f };
                m_blobIndices.mutable_cpu_data = convert(rgIdx);

                BottomVec.Add(m_blobIndices);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, 3, "The top should have num = 3");
                m_log.CHECK_EQ(Top.channels, 1, "The top should have channels = 1");
                m_log.CHECK_EQ(Top.height, 2, "The top should have height = 2");
                m_log.CHECK_EQ(Top.width, 1, "The top should have width = 1");

                float[] rgResult = convertF(Top.mutable_cpu_data);
                float[] rgExpected = new float[] { 1.0f, 1.9f, 2.3f, 3.9f, 4.5f, 5.9f };
                m_log.CHECK_EQ(rgResult.Length, rgExpected.Length, "The result length is not as expected!");

                for (int i = 0; i < rgResult.Length; i++)
                {
                    m_log.CHECK_EQ(rgResult[i], rgExpected[i], "The item at index #" + i.ToString() + " is not as expected!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }


        public void TestGradient(int nAxis)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GATHER);
            p.gather_param.axis = nAxis;
            GatherLayer<T> layer = new GatherLayer<T>(m_cuda, m_log, p);

            try
            {
                float[] rgInputData = null;
                float[] rgIdxF = null;

                if (nAxis == 0)
                {
                    Bottom.Reshape(3, 2, 1, 1);
                    rgInputData = new float[] { 1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f };
                    Bottom.mutable_cpu_data = convert(rgInputData);

                    m_blobIndices.Reshape(2, 2, 1, 1);
                    rgIdxF = new float[] { 0.0f, 1.0f, 1.0f, 2.0f };
                    m_blobIndices.mutable_cpu_data = convert(rgIdxF);

                    BottomVec.Add(m_blobIndices);
                }
                else
                {
                    Bottom.Reshape(3, 3, 1, 1);
                    rgInputData = new float[] { 1.0f, 1.2f, 1.9f, 2.3f, 3.4f, 3.9f, 4.5f, 5.7f, 5.9f };
                    Bottom.mutable_cpu_data = convert(rgInputData);

                    m_blobIndices.Reshape(1, 2, 1, 1);
                    rgIdxF = new float[] { 0.0f, 2.0f };
                    m_blobIndices.mutable_cpu_data = convert(rgIdxF);

                    BottomVec.Add(m_blobIndices);
                }

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_cuda.copy(Top.count(), Top.gpu_data, Top.mutable_gpu_diff);

                layer.Backward(TopVec, new List<bool> { true }, BottomVec);

                float[] rgOutputData = convertF(Bottom.mutable_cpu_diff);

                m_log.CHECK_EQ(rgOutputData.Length, rgInputData.Length, "The output and input data should have the same length.");

                List<int> rgIdx = new List<int>();
                foreach (float f in rgIdxF)
                {
                    rgIdx.Add((int)f);
                }

                for (int i = 0; i < rgInputData.Length; i++)
                {
                    int nIdx = i % Bottom.num;
                    if (!rgIdx.Contains(nIdx))
                        m_log.CHECK_EQ(0, rgOutputData[i], "The data at index #" + i.ToString() + " is not as expected.");
                    else
                        m_log.CHECK_EQ(rgInputData[i], rgOutputData[i], "The data at index #" + i.ToString() + " is not as expected.");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
