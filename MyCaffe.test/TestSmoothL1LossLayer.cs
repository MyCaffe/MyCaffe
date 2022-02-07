using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.test;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.fillers;
using MyCaffe.basecode;
using MyCaffe.layers.ssd;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSmoothL1LossLayer
    {
        [TestMethod]
        public void TestForward()
        {
            SmoothL1LossLayerTest test = new SmoothL1LossLayerTest();

            try
            {
                foreach (ISmoothL1LossLayerTest t in test.Tests)
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
            SmoothL1LossLayerTest test = new SmoothL1LossLayerTest();

            try
            {
                foreach (ISmoothL1LossLayerTest t in test.Tests)
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


    interface ISmoothL1LossLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class SmoothL1LossLayerTest : TestBase
    {
        public SmoothL1LossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("SmoothL1 Loss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SmoothL1LossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SmoothL1LossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SmoothL1LossLayerTest<T> : TestEx<T>, ISmoothL1LossLayerTest
    {
        Blob<T> m_blob_bottom_data;
        Blob<T> m_blob_bottom_label;
        Blob<T> m_blob_top_loss;

        public SmoothL1LossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            m_blob_bottom_data = new Blob<T>(m_cuda, m_log, 10, 5, 1, 1);
            m_blob_bottom_label = new Blob<T>(m_cuda, m_log, 10, 5, 1, 1);
            m_blob_top_loss = new Blob<T>(m_cuda, m_log);

            m_cuda.rng_setseed(1701);
            FillerParameter fp = new FillerParameter("gaussian");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(m_blob_bottom_data);
            filler.Fill(m_blob_bottom_label);

            BottomVec.Clear();
            BottomVec.Add(m_blob_bottom_data);
            BottomVec.Add(m_blob_bottom_label);
            TopVec.Clear();
            TopVec.Add(m_blob_top_loss);
        }

        protected override void dispose()
        {
            m_blob_bottom_data.Dispose();
            m_blob_bottom_label.Dispose();
            m_blob_top_loss.Dispose();
            base.dispose();
        }

        public void TestForward()
        {
            // Get the loss without specified objective weight -- should be
            // equivalent to explicitly specifying a weight of 1.
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SMOOTHL1_LOSS);
            SmoothL1LossLayer<T> layerWeight1 = Layer<T>.Create(m_cuda, m_log, p, null) as SmoothL1LossLayer<T>;
            SmoothL1LossLayer<T> layerWeight2 = null;

            try
            {
                layerWeight1.Setup(BottomVec, TopVec);
                double dfLoss1 = layerWeight1.Forward(BottomVec, TopVec);

                // Get the loss again with a different objective weight; check that it is
                // scaled appropriately.
                double kLossWeight = 3.7;
                p.loss_weight.Add(kLossWeight);
                layerWeight2 = Layer<T>.Create(m_cuda, m_log, p, null) as SmoothL1LossLayer<T>;

                layerWeight2.Setup(BottomVec, TopVec);
                double dfLoss2 = layerWeight2.Forward(BottomVec, TopVec);

                double kErrorMargin = 1e-5;
                m_log.EXPECT_NEAR(dfLoss1 * kLossWeight, dfLoss2, kErrorMargin, "The two weights should be near one another.");

                // Make sure the loss is non-trivial.
                double kNonTrivialAbsThres = 1e-1;
                m_log.CHECK_GE(Math.Abs(dfLoss1), kNonTrivialAbsThres, "The |loss1| should be >= the threshold.");
            }
            finally
            {
                layerWeight1.Dispose();

                if (layerWeight2 != null)  
                    layerWeight2.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SMOOTHL1_LOSS);
            double kLossWeight = 3.7;
            p.loss_weight.Add(kLossWeight);
            SmoothL1LossLayer<T> layer = new SmoothL1LossLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
