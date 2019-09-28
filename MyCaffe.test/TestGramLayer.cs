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
/// Testing the Gram layer.
/// 
/// Gram Layer - layer calculates the Gram matrix used with Neural Style
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestGramLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            GramLayerTest test = new GramLayerTest();

            try
            {
                foreach (IGramLayerTest t in test.Tests)
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
        public void TestForward()
        {
            GramLayerTest test = new GramLayerTest();

            try
            {
                foreach (IGramLayerTest t in test.Tests)
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
            GramLayerTest test = new GramLayerTest();

            try
            {
                foreach (IGramLayerTest t in test.Tests)
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

    interface IGramLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
        void TestGradient();
    }

    class GramLayerTest : TestBase
    {
        public GramLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Gram Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new GramLayerTest<double>(strName, nDeviceID, engine);
            else
                return new GramLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class GramLayerTest<T> : TestEx<T>, IGramLayerTest
    {
        double[] kGramBottomData =
        {
            0.00,  0.01,  0.02,  0.03,  0.04,  0.05,  0.06,  0.07,  0.08,
            0.09,  0.10,  0.11,  0.12,  0.13,  0.14,  0.15,  0.16,  0.17,
            0.18,  0.19,  0.20,  0.21,  0.22,  0.23,  0.24,  0.25,  0.26,
            0.27,  0.28,  0.29,  0.30,  0.31,  0.32,  0.33,  0.34,  0.35,
            0.36,  0.37,  0.38,  0.39,  0.40,  0.41,  0.42,  0.43,  0.44,
            0.45,  0.46,  0.47,  0.48,  0.49,  0.50,  0.51,  0.52,  0.53,
            0.54,  0.55,  0.56,  0.57,  0.58,  0.59,  0.60,  0.61,  0.62,
            0.63,  0.64,  0.65,  0.66,  0.67,  0.68,  0.69,  0.70,  0.71,
            0.72,  0.73,  0.74,  0.75,  0.76,  0.77,  0.78,  0.79,  0.80,
            0.81,  0.82,  0.83,  0.84,  0.85,  0.86,  0.87,  0.88,  0.89,
            0.90,  0.91,  0.92,  0.93,  0.94,  0.95,  0.96,  0.97,  0.98,
            0.99,  1.00,  1.01,  1.02,  1.03,  1.04,  1.05,  1.06,  1.07,
            1.08,  1.09,  1.10,  1.11,  1.12,  1.13,  1.14,  1.15,  1.16,
            1.17,  1.18,  1.19
        };
        double[] kGramTopData =
        {
             0.247,   0.627,   1.007,   0.627,   1.807,   2.987,   1.007,
             2.987,   4.967,   9.727,  12.507,  15.287,  12.507,  16.087,
            19.667,  15.287,  19.667,  24.047
        };

        public GramLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
            m_blob_bottom.mutable_cpu_data = convert(kGramBottomData);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRAM);
            GramLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as GramLayer<T>;

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_top.shape().Count, 3, "The top should have 3 items in its shape.");
            m_log.CHECK_EQ(m_blob_top.shape(0), 2, "The top(0) should be 2");
            m_log.CHECK_EQ(m_blob_top.shape(1), 3, "The top(1) should be 3");
            m_log.CHECK_EQ(m_blob_top.shape(2), 3, "The top(2) should be 3");
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRAM);
            GramLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as GramLayer<T>;

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double[] rgTop = convert(m_blob_top.update_cpu_data());

            for (int i = 0; i < rgTop.Length; i++)
            {
                m_log.EXPECT_EQUAL<float>(rgTop[i], kGramTopData[i], "The top value at index " + i.ToString() + " is not as expected.");
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRAM);
            GramLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as GramLayer<T>;
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
