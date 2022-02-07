using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.layers.beta;

namespace MyCaffe.test
{
    [TestClass]
    public class TestInterpLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            InterpLayerTest test = new InterpLayerTest();

            try
            {
                foreach (IInterpLayerTest t in test.Tests)
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
        public void TestGradient()
        {
            InterpLayerTest test = new InterpLayerTest();

            try
            {
                foreach (IInterpLayerTest t in test.Tests)
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


    interface IInterpLayerTest : ITest
    {
        void TestSetup();
        void TestGradient();
    }

    class InterpLayerTest : TestBase
    {
        public InterpLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Interp Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new InterpLayerTest<double>(strName, nDeviceID, engine);
            else
                return new InterpLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class InterpLayerTest<T> : TestEx<T>, IInterpLayerTest
    {
        public InterpLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 6, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INTERP);
            p.interp_param.height = 13;
            p.interp_param.width = 11;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.INTERP, "The layer type is not correct for InterpLayer!");
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_top.num, 2, "The top(0) should have num = 2.");
                m_log.CHECK_EQ(m_blob_top.channels, 3, "The top(0) should have channels = 3.");
                m_log.CHECK_EQ(m_blob_top.height, 13, "The top(0) should have height = 13.");
                m_log.CHECK_EQ(m_blob_top.width, 11, "The top(0) should have width = 11.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INTERP);
            p.interp_param.height = 13;
            p.interp_param.width = 11;

            InterpLayer<T> layer = new InterpLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
