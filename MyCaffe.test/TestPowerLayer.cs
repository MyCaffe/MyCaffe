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
    public class TestPowerLayer
    {
        [TestMethod]
        public void TestPower()
        {
            PowerLayerTest test = new PowerLayerTest();

            try
            {
                foreach (IPowerLayerTest t in test.Tests)
                {
                    t.TestForward(0.37, 0.83, -2.4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPowerGradient()
        {
            PowerLayerTest test = new PowerLayerTest();

            try
            {
                foreach (IPowerLayerTest t in test.Tests)
                {
                    t.TestForward(0.37, 0.83, -2.4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPowerShiftZero()
        {
            PowerLayerTest test = new PowerLayerTest();

            try
            {
                foreach (IPowerLayerTest t in test.Tests)
                {
                    t.TestForward(0.37, 0.83, 0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPowerGradientShiftZero()
        {
            PowerLayerTest test = new PowerLayerTest();

            try
            {
                foreach (IPowerLayerTest t in test.Tests)
                {
                    t.TestForward(0.37, 0.83, 0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPowerZero()
        {
            PowerLayerTest test = new PowerLayerTest();

            try
            {
                foreach (IPowerLayerTest t in test.Tests)
                {
                    t.TestForward(0.0, 0.83, -2.4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPowerZeroGradient()
        {
            PowerLayerTest test = new PowerLayerTest();

            try
            {
                foreach (IPowerLayerTest t in test.Tests)
                {
                    t.TestForward(0.0, 0.83, -2.4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPowerOne()
        {
            PowerLayerTest test = new PowerLayerTest();

            try
            {
                foreach (IPowerLayerTest t in test.Tests)
                {
                    t.TestForward(1.0, 0.83, -2.4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPowerOneGradient()
        {
            PowerLayerTest test = new PowerLayerTest();

            try
            {
                foreach (IPowerLayerTest t in test.Tests)
                {
                    t.TestForward(1.0, 0.83, -2.4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPowerTwo()
        {
            PowerLayerTest test = new PowerLayerTest();

            try
            {
                foreach (IPowerLayerTest t in test.Tests)
                {
                    t.TestForward(2.0, 0.83, -2.4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPowerTwoGradient()
        {
            PowerLayerTest test = new PowerLayerTest();

            try
            {
                foreach (IPowerLayerTest t in test.Tests)
                {
                    t.TestForward(2.0, 0.83, -2.4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPowerTwoScaleHalf()
        {
            PowerLayerTest test = new PowerLayerTest();

            try
            {
                foreach (IPowerLayerTest t in test.Tests)
                {
                    t.TestForward(2.0, 0.5, -2.4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPowerTwoScaleHalfGradient()
        {
            PowerLayerTest test = new PowerLayerTest();

            try
            {
                foreach (IPowerLayerTest t in test.Tests)
                {
                    t.TestForward(2.0, 0.5, -2.4);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IPowerLayerTest : ITest
    {
        void TestForward(double dfPower, double dfScale, double dfShift);
        void TestBackward(double dfPower, double dfScale, double dfShift);
    }

    class PowerLayerTest : TestBase
    {
        public PowerLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Power Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PowerLayerTest<double>(strName, nDeviceID, engine);
            else
                return new PowerLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class PowerLayerTest<T> : TestEx<T>, IPowerLayerTest
    {
        public PowerLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestForward(double dfPower, double dfScale, double dfShift)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POWER);
            p.power_param.power = dfPower;
            p.power_param.scale = dfScale;
            p.power_param.shift = dfShift;
            PowerLayer<T> layer = new PowerLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // Now, check values
                double[] rgBottom = convert(Bottom.update_cpu_data());
                double[] rgTop = convert(Top.update_cpu_data());
                double dfMinPrecision = 1e-5;

                for (int i = 0; i < Bottom.count(); i++)
                {
                    double dfBottom = rgBottom[i];
                    double dfTop = rgTop[i];
                    double dfExpected = Math.Pow(dfShift + dfScale * dfBottom, dfPower);

                    if (dfPower == 0 || dfPower == 1 || dfPower == 2)
                        m_log.CHECK(!double.IsNaN(dfTop), "The top value is NAN!");

                    if (double.IsNaN(dfExpected))
                    {
                        m_log.CHECK(double.IsNaN(dfTop), "The top value is not NAN when it should be!");
                    }
                    else
                    {
                        double dfPrecision = Math.Min(Math.Abs(dfExpected * 1e-4), dfMinPrecision);
                        m_log.EXPECT_NEAR(dfExpected, dfTop, dfPrecision);
                    }
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackward(double dfPower, double dfScale, double dfShift)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.POWER);
            p.power_param.power = dfPower;
            p.power_param.scale = dfScale;
            p.power_param.shift = dfShift;
            PowerLayer<T> layer = new PowerLayer<T>(m_cuda, m_log, p);

            try
            {
                if (dfPower != 0 && dfPower != 1 && dfPower != 2)
                {
                    // Avoid NaNs by forcing (dfShift + dfScale * x) >= 0
                    double[] rgBottom = convert(Bottom.mutable_cpu_data);
                    double dfMinVal = -dfShift / dfScale;

                    for (int i = 0; i < Bottom.count(); i++)
                    {
                        if (rgBottom[i] < dfMinVal)
                            rgBottom[i] = dfMinVal + (dfMinVal - rgBottom[i]);
                    }

                    Bottom.mutable_cpu_data = convert(rgBottom);
                }

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3, 1701, 0.0, 0.01);
                checker.CheckGradientEltwise(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
