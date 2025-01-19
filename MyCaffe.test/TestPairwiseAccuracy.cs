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
    public class TestPairwiseAccuracyLayer
    {
        [TestMethod]
        public void TestForward()
        {
            PairwiseAccuracyLayerTest test = new PairwiseAccuracyLayerTest();

            try
            {
                foreach (IPairwiseAccuracyLayerTest t in test.Tests)
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
        public void TestPerfectRanking()
        {
            PairwiseAccuracyLayerTest test = new PairwiseAccuracyLayerTest();

            try
            {
                foreach (IPairwiseAccuracyLayerTest t in test.Tests)
                {
                    t.TestPerfectRanking();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestPositionWeighting()
        {
            PairwiseAccuracyLayerTest test = new PairwiseAccuracyLayerTest();

            try
            {
                foreach (IPairwiseAccuracyLayerTest t in test.Tests)
                {
                    t.TestPositionWeighting();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IPairwiseAccuracyLayerTest : ITest
    {
        void TestForward();
        void TestPerfectRanking();
        void TestPositionWeighting();  // Add new test method
    }

    class PairwiseAccuracyLayerTest : TestBase
    {
        public PairwiseAccuracyLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("PairwiseAccuracy Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PairwiseAccuracyLayerTest<double>(strName, nDeviceID, engine);
            else
                return new PairwiseAccuracyLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class PairwiseAccuracyLayerTest<T> : TestEx<T>, IPairwiseAccuracyLayerTest
    {
        Blob<T> m_blob_bottom_target;
        const int BATCH_SIZE = 4;  // Small batch size for testing

        public PairwiseAccuracyLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PAIRWISE_ACCURACY);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            try
            {
                if (!(layer is PairwiseAccuracyLayer<T>))
                    m_log.FAIL("The layer is not the expected PairwiseAccuracy type!");

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);
                double dfAccuracy = convert(TopVec[0].GetData(0));

                // Calculate expected accuracy manually
                double[] rgPredicted = convert(m_blob_bottom.mutable_cpu_data);
                double[] rgTarget = convert(m_blob_bottom_target.mutable_cpu_data);
                double dfTotalCorrect = 0;
                double dfTotalWeight = 0;

                // Compute pairwise comparisons with enhanced weighting
                for (int i = 0; i < BATCH_SIZE; i++)
                {
                    for (int j = 0; j < BATCH_SIZE; j++)
                    {
                        if (i == j) continue;

                        double dfTrueDiff = rgTarget[i] - rgTarget[j];
                        double dfPredDiff = rgPredicted[i] - rgPredicted[j];
                        double dfReturnDiffAbs = Math.Abs(dfTrueDiff);

                        // Position-based importance
                        double dfPositionWeight = 1.0;
                        bool isExtremePair = (i < BATCH_SIZE / 4 || i > 3 * BATCH_SIZE / 4) &&
                                           (j < BATCH_SIZE / 4 || j > 3 * BATCH_SIZE / 4);
                        if (isExtremePair)
                            dfPositionWeight = 2.0;

                        // Sigmoid-based magnitude weight
                        double dfMagnitudeWeight = 2.0 / (1.0 + Math.Exp(-2.0 * dfReturnDiffAbs)) - 1.0;

                        // Combine weights
                        double dfWeight = dfPositionWeight * dfMagnitudeWeight;

                        // Adaptive threshold
                        double dfValidThreshold = Math.Max(1e-6, 1e-5 * dfReturnDiffAbs);

                        if (dfReturnDiffAbs > dfValidThreshold)
                        {
                            bool bCorrect = Math.Sign(dfPredDiff) == Math.Sign(dfTrueDiff);
                            if (bCorrect)
                                dfTotalCorrect += dfWeight;
                            dfTotalWeight += dfWeight;
                        }
                    }
                }

                double dfExpectedAccuracy = (dfTotalWeight > 0) ? dfTotalCorrect / dfTotalWeight : 0;
                m_log.EXPECT_NEAR_FLOAT(dfExpectedAccuracy, dfAccuracy, 1e-4,
                    "The computed accuracy does not match expected value!");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestPerfectRanking()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PAIRWISE_ACCURACY);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            try
            {
                if (!(layer is PairwiseAccuracyLayer<T>))
                    m_log.FAIL("The layer is not the expected PairwiseAccuracy type!");

                // Set predictions to exactly match targets for perfect ranking
                m_cuda.copy(m_blob_bottom_target.count(), m_blob_bottom_target.gpu_data,
                    m_blob_bottom.mutable_gpu_data);
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);
                double dfAccuracy = convert(TopVec[0].GetData(0));

                // With perfect ranking, accuracy should still be 1.0 even with new weighting
                m_log.EXPECT_NEAR_FLOAT(1.0, dfAccuracy, 1e-4,
                    "Perfect ranking should give accuracy of 1.0!");

                // Test with scaled predictions (should still give perfect accuracy)
                m_cuda.scale(m_blob_bottom.count(), 2.0, m_blob_bottom_target.gpu_data,
                    m_blob_bottom.mutable_gpu_data);
                layer.Forward(BottomVec, TopVec);
                dfAccuracy = convert(TopVec[0].GetData(0));
                m_log.EXPECT_NEAR_FLOAT(1.0, dfAccuracy, 1e-4,
                    "Scaled perfect ranking should still give accuracy of 1.0!");

                // Additional test for extreme pairs
                m_cuda.copy(m_blob_bottom_target.count(), m_blob_bottom_target.gpu_data,
                    m_blob_bottom.mutable_gpu_data);
                // Modify some values to create extreme pairs
                double[] rgTarget = convert(m_blob_bottom_target.mutable_cpu_data);
                rgTarget[0] = 10.0;  // Very high value
                rgTarget[BATCH_SIZE - 1] = -10.0;  // Very low value
                layer.Forward(BottomVec, TopVec);
                dfAccuracy = convert(TopVec[0].GetData(0));
                m_log.EXPECT_NEAR_FLOAT(1.0, dfAccuracy, 1e-4,
                    "Perfect ranking with extreme values should give accuracy of 1.0!");
            }
            finally
            {
                layer.Dispose();
            }
        }

        // Add new test for position-based weighting
        public void TestPositionWeighting()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PAIRWISE_ACCURACY);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            try
            {
                if (!(layer is PairwiseAccuracyLayer<T>))
                    m_log.FAIL("The layer is not the expected PairwiseAccuracy type!");

                // Create a scenario where only extreme positions are correct
                double[] rgTarget = convert(m_blob_bottom_target.mutable_cpu_data);
                double[] rgPred = convert(m_blob_bottom.mutable_cpu_data);

                // Set up extreme positions correctly
                for (int i = 0; i < BATCH_SIZE / 4; i++)
                {
                    rgTarget[i] = 5.0 - i * 0.1;  // Top quartile
                    rgPred[i] = 5.0 - i * 0.1;
                }
                for (int i = 3 * BATCH_SIZE / 4; i < BATCH_SIZE; i++)
                {
                    rgTarget[i] = -5.0 - i * 0.1;  // Bottom quartile
                    rgPred[i] = -5.0 - i * 0.1;
                }

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);
                double dfAccuracy = convert(TopVec[0].GetData(0));

                // Accuracy should be higher than 0.5 due to position weighting
                m_log.CHECK_GT(dfAccuracy, 0.5,
                    "Accuracy with correct extreme positions should be higher than 0.5!");
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}