using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestFlattenLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            FlattenLayerTest test = new FlattenLayerTest();

            try
            {
                foreach (IFlattenLayerTest t in test.Tests)
                {
                    t.TestSetup();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupWithAxis()
        {
            FlattenLayerTest test = new FlattenLayerTest();

            try
            {
                foreach (IFlattenLayerTest t in test.Tests)
                {
                    t.TestSetupWithAxis();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupWithEndAxis()
        {
            FlattenLayerTest test = new FlattenLayerTest();

            try
            {
                foreach (IFlattenLayerTest t in test.Tests)
                {
                    t.TestSetupWithEndAxis();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupWithStartAndEndAxis()
        {
            FlattenLayerTest test = new FlattenLayerTest();

            try
            {
                foreach (IFlattenLayerTest t in test.Tests)
                {
                    t.TestSetupWithStartAndEndAxis();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward()
        {
            FlattenLayerTest test = new FlattenLayerTest();

            try
            {
                foreach (IFlattenLayerTest t in test.Tests)
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
        public void TestGradient()
        {
            FlattenLayerTest test = new FlattenLayerTest();

            try
            {
                foreach (IFlattenLayerTest t in test.Tests)
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


    interface IFlattenLayerTest : ITest
    {
        void TestSetup();
        void TestSetupWithAxis();
        void TestSetupWithEndAxis();
        void TestSetupWithStartAndEndAxis();
        void TestForward();
        void TestGradient();
    }

    class FlattenLayerTest : TestBase
    {
        public FlattenLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Flatten Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new FlattenLayerTest<double>(strName, nDeviceID, engine);
            else
                return new FlattenLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class FlattenLayerTest<T> : TestEx<T>, IFlattenLayerTest
    {
        public FlattenLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 6, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.FLATTEN);
            FlattenLayer<T> layer = new FlattenLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(2, Top.num_axes, "The top should have 2 axes.");
                m_log.CHECK_EQ(2, Top.shape(0), "The top shape(0) should equal 2.");
                m_log.CHECK_EQ(3 * 6 * 5, Top.shape(1), "The top shape(1) should equal 3 * 6 * 5 = " + (3 * 6 * 5).ToString() + ".");
            }
            finally
            {
                layer.Dispose();
            }
        }


        public void TestSetupWithAxis()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.FLATTEN);
            p.flatten_param.axis = 2;
            FlattenLayer<T> layer = new FlattenLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(3, Top.num_axes, "The top should have 3 axes.");
                m_log.CHECK_EQ(2, Top.shape(0), "The top shape(0) should equal 2.");
                m_log.CHECK_EQ(3, Top.shape(1), "The top shape(1) should equal 3.");
                m_log.CHECK_EQ(6 * 5, Top.shape(2), "The top shape(2) should equal 6 * 5 = " + (6 * 5).ToString() + ".");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSetupWithEndAxis()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.FLATTEN);
            p.flatten_param.end_axis = -2;
            FlattenLayer<T> layer = new FlattenLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(3, Top.num_axes, "The top should have 3 axes.");
                m_log.CHECK_EQ(2, Top.shape(0), "The top shape(0) should equal 2.");
                m_log.CHECK_EQ(3 * 6, Top.shape(1), "The top shape(1) should equal 3 * 6 =" + (3 * 6).ToString() + ".");
                m_log.CHECK_EQ(5, Top.shape(2), "The top shape(2) should equal 5.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSetupWithStartAndEndAxis()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.FLATTEN);
            p.flatten_param.axis = 0;
            p.flatten_param.end_axis = -2;
            FlattenLayer<T> layer = new FlattenLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(2, Top.num_axes, "The top should have 2 axes.");
                m_log.CHECK_EQ(2 * 3 * 6, Top.shape(0), "The top shape(1) should equal 2 * 3 * 6 =" + (2 * 3 * 6).ToString() + ".");
                m_log.CHECK_EQ(5, Top.shape(1), "The top shape(2) should equal 5.");
            }
            finally
            {
                layer.Dispose();
            }
        }


        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.FLATTEN);
            FlattenLayer<T> layer = new FlattenLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // Now, check values
                for (int c = 0; c < 3 * 6 * 5; c++)
                {
                    double dfTop0 = convert(Top.data_at(0, c, 0, 0));
                    double dfBtm0 = convert(Bottom.data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
                    m_log.CHECK_EQ(dfTop0, dfBtm0, "The top and bottom values should be the same.");

                    double dfTop1 = convert(Top.data_at(1, c, 0, 0));
                    double dfBtm1 = convert(Bottom.data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
                    m_log.CHECK_EQ(dfTop1, dfBtm1, "The top and bottom values should be the same.");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.FLATTEN);
            FlattenLayer<T> layer = new FlattenLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
                checker.CheckGradientEltwise(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
