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
    public class TestGRNLayer
    {
        [TestMethod]
        public void TestForward()
        {
            GRNLayerTest test = new GRNLayerTest();

            try
            {
                foreach (IGRNLayerTest t in test.Tests)
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
            GRNLayerTest test = new GRNLayerTest();

            try
            {
                foreach (IGRNLayerTest t in test.Tests)
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

    interface IGRNLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class GRNLayerTest : TestBase
    {
        public GRNLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("GRN Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new GRNLayerTest<double>(strName, nDeviceID, engine);
            else
                return new GRNLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class GRNLayerTest<T> : TestEx<T>, IGRNLayerTest
    {
        public GRNLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRN);
            GRNLayer<T> layer = new GRNLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Test sum
            for (int i = 0; i < Bottom.num; i++)
            {
                for (int k = 0; k < Bottom.height; k++)
                {
                    for (int l=0; l<Bottom.width; l++)
                    {
                        double dfSum = 0;

                        for (int j=0; j<Top.channels; j++)
                        {
                            T fVal = Top.data_at(i, j, k, l);
                            double dfVal = convert(fVal);
                            dfSum += Math.Pow(dfVal, 2.0);
                        }

                        m_log.CHECK_GE(dfSum, 0.999, "The sum should be greater than or equal to 0.999.");
                        m_log.CHECK_LE(dfSum, 1.001, "The sum should be less than or equal to 1.001.");

                        // Test exact values
                        double dfScale = 0;
                        for (int j=0; j<Bottom.channels; j++)
                        {
                            T fVal = Bottom.data_at(i, j, k, l);
                            double dfVal = convert(fVal);
                            dfScale += Math.Pow(dfVal, 2.0);
                        }

                        for (int j=0; j<Bottom.channels; j++)
                        {
                            T fTop = Top.data_at(i, j, k, l);
                            T fBtm = Bottom.data_at(i, j, k, l);
                            double dfTop = convert(fTop);
                            double dfBtm = convert(fBtm);

                            m_log.CHECK_GE(dfTop + 1e-4, dfBtm / Math.Sqrt(dfScale), "The top and bottom at {" + i.ToString() + "," + j.ToString() + "," + k.ToString() + "," + l.ToString() + "} are not as expected.");
                            m_log.CHECK_LE(dfTop - 1e-4, dfBtm / Math.Sqrt(dfScale), "The top and bottom at {" + i.ToString() + "," + j.ToString() + "," + k.ToString() + "," + l.ToString() + "} are not as expected.");
                        }
                    }
                }
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRN);
            GRNLayer<T> layer = new GRNLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
