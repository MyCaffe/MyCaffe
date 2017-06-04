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

namespace MyCaffe.test
{
    [TestClass]
    public class TestMVNLayer
    {
        [TestMethod]
        public void TestForward()
        {
            MVNLayerTest test = new MVNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMVNLayerTest t in test.Tests)
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
        public void TestForwardMeanOnly()
        {
            MVNLayerTest test = new MVNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMVNLayerTest t in test.Tests)
                {
                    t.TestForwardMeanOnly();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAcrossChannels()
        {
            MVNLayerTest test = new MVNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMVNLayerTest t in test.Tests)
                {
                    t.TestForwardAcrossChannels();
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
            MVNLayerTest test = new MVNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMVNLayerTest t in test.Tests)
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
        public void TestGradientMeanOnly()
        {
            MVNLayerTest test = new MVNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMVNLayerTest t in test.Tests)
                {
                    t.TestGradientMeanOnly();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAcrossChannels()
        {
            MVNLayerTest test = new MVNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMVNLayerTest t in test.Tests)
                {
                    t.TestGradientAcrossChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IMVNLayerTest : ITest
    {
        void TestForward();
        void TestForwardMeanOnly();
        void TestForwardAcrossChannels();
        void TestGradient();
        void TestGradientMeanOnly();
        void TestGradientAcrossChannels();
    }

    class MVNLayerTest : TestBase
    {
        public MVNLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("LRN Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MVNLayerTest<double>(strName, nDeviceID, engine);
            else
                return new MVNLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class MVNLayerTest<T> : TestEx<T>, IMVNLayerTest
    {
        public MVNLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
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
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MVN);
            MVNLayer<T> layer = new MVNLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Test mean
            int nNum = Bottom.num;
            int nChannels = Bottom.channels;
            int nHeight = Bottom.height;
            int nWidth = Bottom.width;

            for (int i=0; i<nNum; i++)
            {
                for (int j=0; j<nChannels; j++)
                {
                    double dfSum = 0;
                    double dfVar = 0;

                    for (int k = 0; k < nHeight; k++)
                    {
                        for (int l = 0; l < nWidth; l++)
                        {
                            T fVal = Top.data_at(i, j, k, l);
                            double dfVal = convert(fVal);
                            dfSum += dfVal;
                            dfVar += dfVal * dfVal;
                        }
                    }

                    dfSum /= nHeight * nWidth;
                    dfVar /= nHeight * nWidth;

                    double dfKErrorBound = 0.001;
                    // expect zero mean
                    m_log.EXPECT_NEAR(0, dfSum, dfKErrorBound);
                    // expect unit variance
                    m_log.EXPECT_NEAR(1, dfVar, dfKErrorBound);
                }
            }
        }

        public void TestForwardMeanOnly()
        {
            RawProto proto = RawProto.Parse("name: \"mvn\" type: \"MVN\" mvn_param { normalize_variance: false }");
            LayerParameter p = LayerParameter.FromProto(proto);
            MVNLayer<T> layer = new MVNLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Test mean
            int nNum = Bottom.num;
            int nChannels = Bottom.channels;
            int nHeight = Bottom.height;
            int nWidth = Bottom.width;

            for (int i = 0; i < nNum; i++)
            {
                for (int j = 0; j < nChannels; j++)
                {
                    double dfSum = 0;
                    double dfVar = 0;

                    for (int k = 0; k < nHeight; k++)
                    {
                        for (int l = 0; l < nWidth; l++)
                        {
                            T fVal = Top.data_at(i, j, k, l);
                            double dfVal = convert(fVal);
                            dfSum += dfVal;
                            dfVar += dfVal * dfVal;
                        }
                    }

                    dfSum /= nHeight * nWidth;

                    double dfKErrorBound = 0.001;
                    // expect zero mean
                    m_log.EXPECT_NEAR(0, dfSum, dfKErrorBound);
                }
            }
        }

        public void TestForwardAcrossChannels()
        {
            RawProto proto = RawProto.Parse("name: \"mvn\" type: \"MVN\" mvn_param { across_channels: true }");
            LayerParameter p = LayerParameter.FromProto(proto);
            MVNLayer<T> layer = new MVNLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Test mean
            int nNum = Bottom.num;
            int nChannels = Bottom.channels;
            int nHeight = Bottom.height;
            int nWidth = Bottom.width;

            for (int i = 0; i < nNum; i++)
            {
                double dfSum = 0;
                double dfVar = 0;

                for (int j = 0; j < nChannels; j++)
                {
                    for (int k = 0; k < nHeight; k++)
                    {
                        for (int l = 0; l < nWidth; l++)
                        {
                            T fVal = Top.data_at(i, j, k, l);
                            double dfVal = convert(fVal);
                            dfSum += dfVal;
                            dfVar += dfVal * dfVal;
                        }
                    }
                }

                dfSum /= nHeight * nWidth * nChannels;
                dfVar /= nHeight * nWidth * nChannels;

                double dfKErrorBound = 0.001;
                // expect zero mean
                m_log.EXPECT_NEAR(0, dfSum, dfKErrorBound);
                // expect unit variance
                m_log.EXPECT_NEAR(1, dfVar, dfKErrorBound);
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MVN);
            MVNLayer<T> layer = new MVNLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientMeanOnly()
        {
            RawProto proto = RawProto.Parse("name: \"mvn\" type: \"MVN\" mvn_param { normalize_variance: false }");
            LayerParameter p = LayerParameter.FromProto(proto);
            MVNLayer<T> layer = new MVNLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientAcrossChannels()
        {
            RawProto proto = RawProto.Parse("name: \"mvn\" type: \"MVN\" mvn_param { across_channels: true }");
            LayerParameter p = LayerParameter.FromProto(proto);
            MVNLayer<T> layer = new MVNLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
