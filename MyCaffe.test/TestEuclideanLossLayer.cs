using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.fillers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestEuclideanLossLayer
    {
        [TestMethod]
        public void TestForward()
        {
            EuclideanLossLayerTest test = new EuclideanLossLayerTest();

            try
            {
                foreach (IEuclideanLossLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        /// <summary>
        /// This test fails.
        /// </summary>
        [TestMethod]
        public void TestGradient()
        {
            EuclideanLossLayerTest test = new EuclideanLossLayerTest();

            try
            {
                foreach (IEuclideanLossLayerTest t in test.Tests)
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

    interface IEuclideanLossLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class EuclideanLossLayerTest : TestBase
    {
        public EuclideanLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("EuclideanLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new EuclideanLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new EuclideanLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class EuclideanLossLayerTest<T> : TestEx<T>, IEuclideanLossLayerTest
    {
        Blob<T> m_blob_bottom_label;

        public EuclideanLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 10, 5, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blob_bottom_label = new Blob<T>(m_cuda, m_log, Bottom);

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, getFillerParam());
            filler.Fill(m_blob_bottom_label);

            BottomVec.Add(m_blob_bottom_label);
        }

        protected override void dispose()
        {
            m_blob_bottom_label.Dispose();
            base.dispose();
        }

        public void TestForward()
        {
            // Get the loss without a specified objective weight -- should be 
            // equivalent to explicitly specifying a weight of 1.
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.EUCLIDEAN_LOSS);            
            EuclideanLossLayer<T> layer_weight_1 = new EuclideanLossLayer<T>(m_cuda, m_log, p);

            layer_weight_1.Setup(BottomVec, TopVec);
            double dfLoss1 = layer_weight_1.Forward(BottomVec, TopVec);

            // Get the loss again with a different objective weight; check that it is
            // scaled appropriately.
            double kLossWeight = 3.7;
            p.loss_weight.Add(kLossWeight);
            EuclideanLossLayer<T> layer_weight_2 = new EuclideanLossLayer<T>(m_cuda, m_log, p);

            layer_weight_2.Setup(BottomVec, TopVec);
            double dfLoss2 = layer_weight_2.Forward(BottomVec, TopVec);

            double dfErrorMargin = 1e-5;

            m_log.EXPECT_NEAR(dfLoss1 * kLossWeight, dfLoss2, dfErrorMargin);
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.EUCLIDEAN_LOSS);
            double kLossWeight = 3.7;
            p.loss_weight.Add(kLossWeight);
            EuclideanLossLayer<T> layer = new EuclideanLossLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
