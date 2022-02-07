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
/// Testing the squeeze/unsqueeze layers.
/// 
/// Squeeze Layer - this layer removes dim=1 items from the shape.
/// Unsqueeze Layer = this layer adds dim=1 items to the shape at specified axes.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestSqueezeLayer
    {
        [TestMethod]
        public void TestSqueezeForward()
        {
            SqueezeLayerTest test = new SqueezeLayerTest();

            try
            {
                foreach (ISqueezeLayerTest t in test.Tests)
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
        public void TestSqueezeGradient()
        {
            SqueezeLayerTest test = new SqueezeLayerTest();

            try
            {
                foreach (ISqueezeLayerTest t in test.Tests)
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
        public void TestUnsqueezeForward()
        {
            UnsqueezeLayerTest test = new UnsqueezeLayerTest();

            try
            {
                foreach (ISqueezeLayerTest t in test.Tests)
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
        public void TestUnsqueezeGradient()
        {
            UnsqueezeLayerTest test = new UnsqueezeLayerTest();

            try
            {
                foreach (ISqueezeLayerTest t in test.Tests)
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

    interface ISqueezeLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class SqueezeLayerTest : TestBase
    {
        public SqueezeLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Squeeze Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SqueezeLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SqueezeLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SqueezeLayerTest<T> : TestEx<T>, ISqueezeLayerTest
    {
        public SqueezeLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SQUEEZE);
            List<int> rgShape = new List<int>() { 1, 2, 3, 1 };

            m_blob_bottom.Reshape(rgShape);

            List<float> rgData = new List<float>() { 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f };
            m_blob_bottom.mutable_cpu_data = convert(rgData.ToArray());

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_bottom.count(), m_blob_top.count(), "The top and bottom should have the same count!");
            m_log.CHECK_EQ(m_blob_top.num_axes, 2, "The top should have only 2 axes.");

            List<int> rgExpectedShape = new List<int>() { 2, 3 };

            for (int i = 0; i < m_blob_top.shape().Count; i++)
            {
                int nDim = m_blob_top.shape()[i];
                m_log.CHECK_EQ(nDim, rgExpectedShape[i], "The dimension is not as expected.");
            }

            float[] rgF = convertF(m_blob_top.mutable_cpu_data);
            m_log.CHECK_EQ(rgF.Length, rgData.Count, "The data counts do not match.");

            for (int i = 0; i < rgF.Length; i++)
            {
                m_log.CHECK_EQ(rgF[i], rgData[i], "The data items do not match!");
            }
        }


        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SQUEEZE);
            List<int> rgShape = new List<int>() { 1, 2, 3, 1 };

            m_blob_bottom.Reshape(rgShape);

            List<float> rgData = new List<float>() { 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f };
            m_blob_bottom.mutable_cpu_data = convert(rgData.ToArray());

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_bottom.count(), m_blob_top.count(), "The top and bottom should have the same count!");
            m_log.CHECK_EQ(m_blob_top.num_axes, 2, "The top should have only 2 axes.");

            List<int> rgExpectedShape = new List<int>() { 2, 3 };

            for (int i = 0; i < m_blob_top.shape().Count; i++)
            {
                int nDim = m_blob_top.shape()[i];
                m_log.CHECK_EQ(nDim, rgExpectedShape[i], "The dimension is not as expected.");
            }

            float[] rgF = convertF(m_blob_top.mutable_cpu_data);
            m_log.CHECK_EQ(rgF.Length, rgData.Count, "The data counts do not match.");

            for (int i = 0; i < rgF.Length; i++)
            {
                m_log.CHECK_EQ(rgF[i], rgData[i], "The data items do not match!");
            }

            m_cuda.copy(m_blob_top.count(), m_blob_top.gpu_data, m_blob_top.mutable_gpu_diff);

            layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

            m_log.CHECK_EQ(BottomVec[0].num_axes, rgShape.Count, "The number of axes does not match!");

            for (int i = 0; i < rgShape.Count; i++)
            {
                m_log.CHECK_EQ(BottomVec[0].shape()[i], rgShape[i], "The dimension at the axis does not match!");
            }

            rgF = convertF(m_blob_bottom.mutable_cpu_diff);

            for (int i = 0; i < rgF.Length; i++)
            {
                m_log.CHECK_EQ(rgF[i], rgData[i], "The data items do not match!");
            }
        }
    }

    class UnsqueezeLayerTest : TestBase
    {
        public UnsqueezeLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Unsqueeze Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new UnsqueezeLayerTest<double>(strName, nDeviceID, engine);
            else
                return new UnsqueezeLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class UnsqueezeLayerTest<T> : TestEx<T>, ISqueezeLayerTest
    {
        public UnsqueezeLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.UNSQUEEZE);
            p.squeeze_param.axes = new List<int>() { 0, 3 };
            List<int> rgShape = new List<int>() { 2, 3 };

            m_blob_bottom.Reshape(rgShape);

            List<float> rgData = new List<float>() { 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f };
            m_blob_bottom.mutable_cpu_data = convert(rgData.ToArray());

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_bottom.count(), m_blob_top.count(), "The top and bottom should have the same count!");
                m_log.CHECK_EQ(m_blob_top.num_axes, 4, "The top should have 4 axes.");

                List<int> rgExpectedShape = new List<int>() { 1, 2, 3, 1 };

                for (int i = 0; i < m_blob_top.shape().Count; i++)
                {
                    int nDim = m_blob_top.shape()[i];
                    m_log.CHECK_EQ(nDim, rgExpectedShape[i], "The dimension is not as expected.");
                }

                float[] rgF = convertF(m_blob_top.mutable_cpu_data);
                m_log.CHECK_EQ(rgF.Length, rgData.Count, "The data counts do not match.");

                for (int i = 0; i < rgF.Length; i++)
                {
                    m_log.CHECK_EQ(rgF[i], rgData[i], "The data items do not match!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }


        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.UNSQUEEZE);
            p.squeeze_param.axes = new List<int>() { 0, 3 };
            List<int> rgShape = new List<int>() { 2, 3 };

            m_blob_bottom.Reshape(rgShape);

            List<float> rgData = new List<float>() { 1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f };
            m_blob_bottom.mutable_cpu_data = convert(rgData.ToArray());

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_bottom.count(), m_blob_top.count(), "The top and bottom should have the same count!");
                m_log.CHECK_EQ(m_blob_top.num_axes, 4, "The top should have 4 axes.");

                List<int> rgExpectedShape = new List<int>() { 1, 2, 3, 1 };

                for (int i = 0; i < m_blob_top.shape().Count; i++)
                {
                    int nDim = m_blob_top.shape()[i];
                    m_log.CHECK_EQ(nDim, rgExpectedShape[i], "The dimension is not as expected.");
                }

                float[] rgF = convertF(m_blob_top.mutable_cpu_data);
                m_log.CHECK_EQ(rgF.Length, rgData.Count, "The data counts do not match.");

                for (int i = 0; i < rgF.Length; i++)
                {
                    m_log.CHECK_EQ(rgF[i], rgData[i], "The data items do not match!");
                }

                m_cuda.copy(m_blob_top.count(), m_blob_top.gpu_data, m_blob_top.mutable_gpu_diff);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                m_log.CHECK_EQ(BottomVec[0].num_axes, rgShape.Count, "The number of axes does not match!");

                for (int i = 0; i < rgShape.Count; i++)
                {
                    m_log.CHECK_EQ(BottomVec[0].shape()[i], rgShape[i], "The dimension at the axis does not match!");
                }

                rgF = convertF(m_blob_bottom.mutable_cpu_diff);

                for (int i = 0; i < rgF.Length; i++)
                {
                    m_log.CHECK_EQ(rgF[i], rgData[i], "The data items do not match!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
