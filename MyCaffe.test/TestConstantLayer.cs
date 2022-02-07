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
using MyCaffe.layers.beta;

namespace MyCaffe.test
{
    [TestClass]
    public class TestConstantLayer
    {
        [TestMethod]
        public void TestForward()
        {
            ConstantLayerTest test = new ConstantLayerTest();

            try
            {
                foreach (IConstantLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IConstantLayerTest : ITest
    {
        void TestForward();
    }

    class ConstantLayerTest : TestBase
    {
        public ConstantLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Constant Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ConstantLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ConstantLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ConstantLayerTest<T> : TestEx<T>, IConstantLayerTest
    {
        public ConstantLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 10, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONSTANT);
            p.constant_param.output_shape.dim.Add(1);
            p.constant_param.output_shape.dim.Add(3);
            p.constant_param.output_shape.dim.Add(28);
            p.constant_param.output_shape.dim.Add(28);
            p.constant_param.values_f.Add(33.5f);
            ConstantLayer<T> layer = new ConstantLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, 1, "The top num should = 1");
                m_log.CHECK_EQ(Top.channels, 3, "The top num should = 3");
                m_log.CHECK_EQ(Top.height, 28, "The top num should = 28");
                m_log.CHECK_EQ(Top.width, 28, "The top num should = 28");

                double[] rgData = convert(Top.mutable_cpu_data);

                foreach (double df in rgData)
                {
                    m_log.EXPECT_NEAR_FLOAT(df, 33.5, 0.000001);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
