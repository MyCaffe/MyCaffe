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
    public class TestNormalization1Layer
    {
        [TestMethod]
        public void TestForward()
        {
            Normalization1LayerTest test = new Normalization1LayerTest();

            try
            {
                foreach (INormalization1LayerTest t in test.Tests)
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
            Normalization1LayerTest test = new Normalization1LayerTest();

            try
            {
                foreach (INormalization1LayerTest t in test.Tests)
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

    interface INormalization1LayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class Normalization1LayerTest : TestBase
    {
        public Normalization1LayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Normalization1 Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new Normalization1LayerTest<double>(strName, nDeviceID, engine);
            else
                return new Normalization1LayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class Normalization1LayerTest<T> : TestEx<T>, INormalization1LayerTest
    {
        public Normalization1LayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 10, 4, 2, 3 }, nDeviceID)
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
            double dfPrecision = 1e-5;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NORMALIZATION1);
            Normalization1Layer<T> layer = new Normalization1Layer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                for (int i1 = 0; i1 < Bottom.num; i1++)
                {
                    double dfNormSqrBottom = 0;
                    double dfNormSqrTop = 0;

                    for (int i2 = 0; i2 < Top.channels; i2++)
                    {
                        for (int i3 = 0; i3 < Top.height; i3++)
                        {
                            for (int i4 = 0; i4 < Top.width; i4++)
                            {
                                double dfTop = convert(Top.data_at(i1, i2, i3, i4));
                                double dfBtm = convert(Bottom.data_at(i1, i2, i3, i4));

                                dfNormSqrTop += Math.Pow(dfTop, 2.0);
                                dfNormSqrBottom += Math.Pow(dfBtm, 2.0);
                            }
                        }
                    }

                    m_log.EXPECT_NEAR(dfNormSqrTop, 1, dfPrecision);
                    double dfC = Math.Pow(dfNormSqrBottom, -0.5);

                    for (int i2 = 0; i2 < Top.channels; i2++)
                    {
                        for (int i3 = 0; i3 < Top.height; i3++)
                        {
                            for (int i4 = 0; i4 < Top.width; i4++)
                            {
                                double dfTop = convert(Top.data_at(i1, i2, i3, i4));
                                double dfBtm = convert(Bottom.data_at(i1, i2, i3, i4)) * dfC;

                                m_log.EXPECT_NEAR(dfTop, dfBtm, dfPrecision);
                            }
                        }
                    }
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NORMALIZATION1);
            Normalization1Layer<T> layer = new Normalization1Layer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
