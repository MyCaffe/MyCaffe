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
using MyCaffe.layers.nt;

/// <summary>
/// Testing the TVLoss layer.
/// 
/// TVLoss Layer - layer calculates the TVLoss matrix used with Neural Style
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTVLossLayer
    {
        [TestMethod]
        public void TestForward()
        {
            TVLossLayerTest test = new TVLossLayerTest();

            try
            {
                foreach (ITVLossLayerTest t in test.Tests)
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
            TVLossLayerTest test = new TVLossLayerTest();

            try
            {
                foreach (ITVLossLayerTest t in test.Tests)
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

    interface ITVLossLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class TVLossLayerTest : TestBase
    {
        public TVLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TVLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TVLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new TVLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class TVLossLayerTest<T> : TestEx<T>, ITVLossLayerTest
    {
        double[] bottomData =
        {
            0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0,  1.1,
            1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.0,  2.1,  2.2,  2.3,
            2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.1,  3.2,  3.3,  3.4,  3.5,
            3.6,  3.7,  3.8,  3.9,  4.0,  4.1,  4.2,  4.3,  4.4,  4.5,  4.6,  4.7,
            4.8,  4.9,  5.0,  5.1,  5.2,  5.3,  5.4,  5.5,  5.6,  5.7,  5.8,  5.9,
            6.0,  6.1,  6.2,  6.3,  6.4,  6.5,  6.6,  6.7,  6.8,  6.9,  7.0,  7.1,
            7.2,  7.3,  7.4,  7.5,  7.6,  7.7,  7.8,  7.9,  8.0,  8.1,  8.2,  8.3,
            8.4,  8.5,  8.6,  8.7,  8.8,  8.9,  9.0,  9.1,  9.2,  9.3,  9.4,  9.5,
            9.6,  9.7,  9.8,  9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7,
            10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9
        };

        public TVLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 1, 3, 4, 10 }, nDeviceID)
        {
            m_engine = engine;
            m_blob_bottom.mutable_cpu_data = convert(bottomData);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TV_LOSS);
            p.tv_loss_param.beta = 2.5f;
            TVLossLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as TVLossLayer<T>;

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double dfLoss = convert(m_blob_top.GetData(0));
                m_log.EXPECT_EQUAL<float>(dfLoss, 82.0137624747, "The loss value is incorrect.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TV_LOSS);
            p.tv_loss_param.beta = 2.5f;
            TVLossLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as TVLossLayer<T>;

            try
            { 
                layer.Setup(BottomVec, TopVec);
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
