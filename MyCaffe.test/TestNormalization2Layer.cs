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
    public class TestNormalization2Layer
    {
        [TestMethod]
        public void TestForward()
        {
            Normalization2LayerTest test = new Normalization2LayerTest();

            try
            {
                foreach (INormalization2LayerTest t in test.Tests)
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
        public void TestForwardScale()
        {
            Normalization2LayerTest test = new Normalization2LayerTest();

            try
            {
                foreach (INormalization2LayerTest t in test.Tests)
                {
                    t.TestForward(10.0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardScaleChannel()
        {
            Normalization2LayerTest test = new Normalization2LayerTest();

            try
            {
                foreach (INormalization2LayerTest t in test.Tests)
                {
                    t.TestForward(10.0, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardEltwise()
        {
            Normalization2LayerTest test = new Normalization2LayerTest();

            try
            {
                foreach (INormalization2LayerTest t in test.Tests)
                {
                    t.TestForward(1.0, true, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardEltwiseScale()
        {
            Normalization2LayerTest test = new Normalization2LayerTest();

            try
            {
                foreach (INormalization2LayerTest t in test.Tests)
                {
                    t.TestForward(10.0, true, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardEltwiseScaleChannel()
        {
            Normalization2LayerTest test = new Normalization2LayerTest();

            try
            {
                foreach (INormalization2LayerTest t in test.Tests)
                {
                    t.TestForward(10.0, false, false);
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
            Normalization2LayerTest test = new Normalization2LayerTest();

            try
            {
                foreach (INormalization2LayerTest t in test.Tests)
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
        public void TestGradientScale()
        {
            Normalization2LayerTest test = new Normalization2LayerTest();

            try
            {
                foreach (INormalization2LayerTest t in test.Tests)
                {
                    t.TestGradient(3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestGradientScaleChannel()
        {
            Normalization2LayerTest test = new Normalization2LayerTest();

            try
            {
                foreach (INormalization2LayerTest t in test.Tests)
                {
                    t.TestGradient(3, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestGradientEltwise()
        {
            Normalization2LayerTest test = new Normalization2LayerTest();

            try
            {
                foreach (INormalization2LayerTest t in test.Tests)
                {
                    t.TestGradient(1.0, true, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestGradientEltwiseScale()
        {
            Normalization2LayerTest test = new Normalization2LayerTest();

            try
            {
                foreach (INormalization2LayerTest t in test.Tests)
                {
                    t.TestGradient(3, true, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestGradientEltwiseScaleChannel()
        {
            Normalization2LayerTest test = new Normalization2LayerTest();

            try
            {
                foreach (INormalization2LayerTest t in test.Tests)
                {
                    t.TestGradient(3, false, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface INormalization2LayerTest : ITest
    {
        void TestForward(double dfScale = 1.0, bool bChannelShared = true, bool bAcrossSpatial = true);
        void TestGradient(double dfScale = 1.0, bool bChannelShared = true, bool bAcrossSpatial = true);
    }

    class Normalization2LayerTest : TestBase
    {
        public Normalization2LayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Normalization2 Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new Normalization2LayerTest<double>(strName, nDeviceID, engine);
            else
                return new Normalization2LayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class Normalization2LayerTest<T> : TestEx<T>, INormalization2LayerTest
    {
        public Normalization2LayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 2, 3 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("constant", 1.0);
        }

        public void TestForward(double dfScale = 1.0, bool bChannelShared = true, bool bAcrossSpatial = true)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NORMALIZATION2);

            if (dfScale != 1.0)
                p.normalization2_param.scale_filler = new FillerParameter("constant", dfScale);

            p.normalization2_param.channel_shared = bChannelShared;
            p.normalization2_param.across_spatial = bAcrossSpatial;

            Normalization2Layer<T> layer = new Normalization2Layer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Test norm
            int nNum = m_blob_bottom.num;
            int nChannels = m_blob_bottom.channels;
            int nHeight = m_blob_bottom.height;
            int nWidth = m_blob_bottom.width;

            if (bAcrossSpatial)
            {
                for (int i = 0; i < nNum; i++)
                {
                    double dfNorm = 0;

                    for (int j = 0; j < nChannels; j++)
                    {
                        for (int k = 0; k < nHeight; k++)
                        {
                            for (int l = 0; l < nWidth; l++)
                            {
                                double dfData = Utility.ConvertVal<T>(m_blob_top.data_at(i, j, k, l));
                                dfNorm += dfData * dfData;
                            }
                        }
                    }

                    double kErrorBound = 1e-5;

                    // Expect unit norm.
                    double dfExpected = Math.Sqrt(dfNorm);
                    m_log.EXPECT_NEAR(dfScale, dfExpected, kErrorBound, "The values are not as expected for the norm.");
                }
            }
            else
            {
                for (int i = 0; i < nNum; i++)
                {
                    for (int k = 0; k < nHeight; k++)
                    {
                        for (int l = 0; l < nWidth; l++)
                        {
                            double dfNorm = 0;

                            for (int j = 0; j < nChannels; j++)
                            {
                                double dfData = Utility.ConvertVal<T>(m_blob_top.data_at(i, j, k, l));
                                dfNorm += dfData * dfData;
                            }

                            double kErrorBound = 1e-5;

                            // Expect unit norm.
                            double dfExpected = Math.Sqrt(dfNorm);
                            m_log.EXPECT_NEAR(dfScale, dfExpected, kErrorBound, "The values are not as expected for the norm.");
                        }
                    }
                }
            }
        }

        public void TestGradient(double dfScale = 1.0, bool bChannelShared = true, bool bAcrossSpatial = true)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NORMALIZATION2);

            if (dfScale != 1.0)
                p.normalization2_param.scale_filler = new FillerParameter("constant", dfScale);

            p.normalization2_param.channel_shared = bChannelShared;
            p.normalization2_param.across_spatial = bAcrossSpatial;

            Normalization2Layer<T> layer = new Normalization2Layer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
