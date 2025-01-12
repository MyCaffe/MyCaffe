using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.layers.beta;
using MyCaffe.fillers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestPairwiseLossLayer
    {
        [TestMethod]
        public void TestForward()
        {
            PairwiseLossLayerTest test = new PairwiseLossLayerTest();

            try
            {
                foreach (IPairwiseLossLayerTest t in test.Tests)
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
            PairwiseLossLayerTest test = new PairwiseLossLayerTest();

            try
            {
                foreach (IPairwiseLossLayerTest t in test.Tests)
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

    interface IPairwiseLossLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class PairwiseLossLayerTest : TestBase
    {
        public PairwiseLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("PairwiseLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PairwiseLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new PairwiseLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class PairwiseLossLayerTest<T> : TestEx<T>, IPairwiseLossLayerTest
    {
        Blob<T> m_blob_bottom_target;
        const int BATCH_SIZE = 4;  // Small batch size for testing
        const double MARGIN = 1.0;

        public PairwiseLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { BATCH_SIZE, 1, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blob_bottom_target = new Blob<T>(m_cuda, m_log, Bottom);

            FillerParameter fp = new FillerParameter("uniform");
            fp.min = -2;  // Use wider range to ensure varied returns
            fp.max = 2;

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(m_blob_bottom);  // Fill predictions
            filler.Fill(m_blob_bottom_target);  // Fill target returns

            BottomVec.Add(m_blob_bottom_target);
        }

        protected override void dispose()
        {
            if (m_blob_bottom_target != null)
                m_blob_bottom_target.Dispose();
            base.dispose();
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PAIRWISE_LOSS);
            p.pairwise_loss_param.margin = MARGIN;
            p.loss_param.normalization = LossParameter.NormalizationMode.BATCH_SIZE;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                if (!(layer is PairwiseLossLayer<T>))
                    m_log.FAIL("The layer is not the expected PairwiseLoss type!");

                layer.Setup(BottomVec, TopVec);
                double dfLoss1 = layer.Forward(BottomVec, TopVec);
                double dfLoss = convert(TopVec[0].GetData(0));

                m_log.CHECK_EQ(dfLoss1, dfLoss, "The loss values should match!");

                // Calculate expected loss manually
                double[] rgPredicted = convert(m_blob_bottom.mutable_cpu_data);
                double[] rgTarget = convert(m_blob_bottom_target.mutable_cpu_data);
                double dfTotalLoss = 0;
                double dfTotalWeight = 0;

                // Compute pairwise comparisons
                for (int i = 0; i < BATCH_SIZE; i++)
                {
                    for (int j = 0; j < BATCH_SIZE; j++)
                    {
                        if (i == j) continue;

                        double dfTrueDiff = rgTarget[i] - rgTarget[j];
                        double dfPredDiff = rgPredicted[i] - rgPredicted[j];

                        if (Math.Abs(dfTrueDiff) > 1e-6)
                        {
                            double dfWeight = Math.Abs(dfTrueDiff) * (1.0 + Math.Abs(dfTrueDiff));
                            double dfPairLoss = dfWeight * Math.Max(0.0, MARGIN - Math.Sign(dfTrueDiff) * dfPredDiff);

                            dfTotalLoss += dfPairLoss;
                            dfTotalWeight += dfWeight;
                        }
                    }
                }

                double dfExpectedLoss = (dfTotalWeight > 0) ? dfTotalLoss / dfTotalWeight : 0;

                m_log.EXPECT_NEAR_FLOAT(dfExpectedLoss, dfLoss, 1e-4, "The computed loss does not match expected value!");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PAIRWISE_LOSS);
            p.pairwise_loss_param.margin = MARGIN;
            p.loss_param.normalization = LossParameter.NormalizationMode.BATCH_SIZE;
            p.loss_weight.Add(1.0);

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                layer.Setup(BottomVec, TopVec);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);  // Only check gradient w.r.t. predictions
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
