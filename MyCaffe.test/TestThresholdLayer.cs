using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestThresholdLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            ThresholdLayerTest test = new ThresholdLayerTest();

            try
            {
                foreach (IThresholdLayerTest t in test.Tests)
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
        public void Test()
        {
            ThresholdLayerTest test = new ThresholdLayerTest();

            try
            {
                foreach (IThresholdLayerTest t in test.Tests)
                {
                    t.Test();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void Test2()
        {
            ThresholdLayerTest test = new ThresholdLayerTest();

            try
            {
                foreach (IThresholdLayerTest t in test.Tests)
                {
                    t.Test2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IThresholdLayerTest : ITest
    {
        void TestSetup();
        void Test();
        void Test2();
    }

    class ThresholdLayerTest : TestBase
    {
        public ThresholdLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Threshold Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ThresholdLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ThresholdLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ThresholdLayerTest<T> : TestEx<T>, IThresholdLayerTest
    {
        public ThresholdLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 6, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.THRESHOLD);
            ThresholdLayer<T> layer = new ThresholdLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, Bottom.num, "The top and bottom should have an equal num.");
            m_log.CHECK_EQ(Top.channels, Bottom.channels, "The top and bottom should have equal channels.");
            m_log.CHECK_EQ(Top.height, Bottom.height, "The top and bottom should have an equal height.");
            m_log.CHECK_EQ(Top.width, Bottom.width, "The top and bottom should have an equal width.");
        }

        public void Test()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.THRESHOLD);
            ThresholdLayer<T> layer = new ThresholdLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBtm = convert(Bottom.update_cpu_data());
            double[] rgTop = convert(Top.update_cpu_data());
            double dfThreshold = p.threshold_param.threshold;

            for (int i = 0; i < Bottom.count(); i++)
            {
                m_log.CHECK_GE(rgTop[i], 0.0, "The top value should be greater than or equal to 0.0");
                m_log.CHECK_LE(rgTop[i], 1.0, "The top value should be less than or equal to 1.0");

                if (rgTop[i] == 0.0)
                    m_log.CHECK_LE(rgBtm[i], dfThreshold, "The bottom at " + i.ToString() + " should be less than or equal to the threshold of " + dfThreshold.ToString());
                else
                    m_log.CHECK_GT(rgBtm[i], dfThreshold, "The bottom at " + i.ToString() + " should be greater than the threshold of " + dfThreshold.ToString());
            }
        }

        public void Test2()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.THRESHOLD);
            p.threshold_param.threshold = 0.5;
            ThresholdLayer<T> layer = new ThresholdLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBtm = convert(Bottom.update_cpu_data());
            double[] rgTop = convert(Top.update_cpu_data());
            double dfThreshold = p.threshold_param.threshold;

            m_log.CHECK_EQ(dfThreshold, 0.5, "The threshold should equal 0.5");

            for (int i = 0; i < Bottom.count(); i++)
            {
                m_log.CHECK_GE(rgTop[i], 0.0, "The top value should be greater than or equal to 0.0");
                m_log.CHECK_LE(rgTop[i], 1.0, "The top value should be less than or equal to 1.0");

                if (rgTop[i] == 0.0)
                    m_log.CHECK_LE(rgBtm[i], dfThreshold, "The bottom at " + i.ToString() + " should be less than or equal to the threshold of " + dfThreshold.ToString());
                else
                    m_log.CHECK_GT(rgBtm[i], dfThreshold, "The bottom at " + i.ToString() + " should be greater than the threshold of " + dfThreshold.ToString());
            }
        }
    }
}
