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
using MyCaffe.layers.alpha;

/// <summary>
/// Testing for simple triplet loss layer.
/// </summary>
/// <remarks>
/// See https://github.com/freesouls/caffe
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestSimpleTripletLossLayer
    {
        [TestMethod]
        public void TestForward()
        {
            SimpleTripletLossLayerTest test = new SimpleTripletLossLayerTest();

            try
            {
                foreach (ISimpleTripletLossLayerTest t in test.Tests)
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
            SimpleTripletLossLayerTest test = new SimpleTripletLossLayerTest();

            try
            {
                foreach (ISimpleTripletLossLayerTest t in test.Tests)
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

    interface ISimpleTripletLossLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class SimpleTripletLossLayerTest : TestBase
    {
        public SimpleTripletLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("SimpleTripletLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SimpleTripletLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SimpleTripletLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SimpleTripletLossLayerTest<T> : TestEx<T>, ISimpleTripletLossLayerTest
    {
        Random m_random;
        Blob<T> m_blobBottomLabel;

        public SimpleTripletLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 20, 10, 1, 1 }, nDeviceID)
        {
            m_random = new Random(1701);
            m_engine = engine;
            m_blobBottomLabel = new Blob<T>(m_cuda, m_log, 20, 1, 1, 1);

            double[] rgdfLabels = convert(m_blobBottomLabel.mutable_cpu_data);
            int nHalf = m_blobBottomLabel.count() / 2;

            for (int i=0; i<nHalf; i++)
            {
                rgdfLabels[i] = m_random.Next() % 10;
                rgdfLabels[i + nHalf] = rgdfLabels[i];
            }

            m_blobBottomLabel.mutable_cpu_data = convert(rgdfLabels);

            BottomVec.Add(m_blobBottomLabel);
        }

        protected override void dispose()
        {
            m_blobBottomLabel.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestForward()
        {
            // Get the loss without a specfied objective weight -- should be 
            // equivalent to explicitly specifying a weight of 1.
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRIPLET_LOSS_SIMPLE);
            TripletLossSimpleLayer<T> layer_weight_1 = new TripletLossSimpleLayer<T>(m_cuda, m_log, p);
            layer_weight_1.Setup(BottomVec, TopVec);
            double dfLoss1 = layer_weight_1.Forward(BottomVec, TopVec);

            // Get the loss again with a different objective weight; check that it is
            // scaled appropriately.
            double kLossWeight = 3.7;
            p.loss_weight.Add(kLossWeight);
            TripletLossSimpleLayer<T> layer_weight_2 = new TripletLossSimpleLayer<T>(m_cuda, m_log, p);
            layer_weight_2.Setup(BottomVec, TopVec);
            double dfLoss2 = layer_weight_2.Forward(BottomVec, TopVec);

            double dfErrorMargin = 1e-5;

            m_log.EXPECT_NEAR(dfLoss1 * kLossWeight, dfLoss2, dfErrorMargin);

            // Make sure loss is non-trivial
            double dfNonTrivialAbsThresh = 1e-2;
            m_log.CHECK_GE(Math.Abs(dfLoss1), dfNonTrivialAbsThresh, "The loss appears to be trivial.");
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRIPLET_LOSS_SIMPLE);
            double dfLossWeight = 3.7;
            p.loss_weight.Add(dfLossWeight);
            TripletLossSimpleLayer<T> layer = new TripletLossSimpleLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }
    }
}
