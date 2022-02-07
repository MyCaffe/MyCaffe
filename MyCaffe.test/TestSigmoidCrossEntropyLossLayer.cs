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
    public class TestSigmoidCrossEntropyLossLayer
    {
        [TestMethod]
        public void TestForward()
        {
            SigmoidCrossEntropyLossLayerTest test = new SigmoidCrossEntropyLossLayerTest();

            try
            {
                foreach (ISigmoidCrossEntropyLossLayerTest t in test.Tests)
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
            SigmoidCrossEntropyLossLayerTest test = new SigmoidCrossEntropyLossLayerTest();

            try
            {
                foreach (ISigmoidCrossEntropyLossLayerTest t in test.Tests)
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
        public void TestIgnoreGradient()
        {
            SigmoidCrossEntropyLossLayerTest test = new SigmoidCrossEntropyLossLayerTest();

            try
            {
                foreach (ISigmoidCrossEntropyLossLayerTest t in test.Tests)
                {
                    t.TestIgnoreGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface ISigmoidCrossEntropyLossLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
        void TestIgnoreGradient();
    }

    class SigmoidCrossEntropyLossLayerTest : TestBase
    {
        public SigmoidCrossEntropyLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("SigmoidCrossEntropyLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SigmoidCrossEntropyLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SigmoidCrossEntropyLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SigmoidCrossEntropyLossLayerTest<T> : TestEx<T>, ISigmoidCrossEntropyLossLayerTest
    {
        Blob<T> m_blob_bottom_targets;

        public SigmoidCrossEntropyLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 10, 5, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blob_bottom_targets = new Blob<T>(m_cuda, m_log, Bottom);

            // Fill the data vector.
            FillerParameter data_fp = new FillerParameter("gaussian");
            data_fp.std = 1;
            Filler<T> fillerData = Filler<T>.Create(m_cuda, m_log, data_fp);
            fillerData.Fill(Bottom);

            // Fill the targets vector.
            FillerParameter fp = new FillerParameter("uniform");
            fp.min = 0;
            fp.max = 1;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(m_blob_bottom_targets);

            BottomVec.Add(m_blob_bottom_targets);
        }

        protected override void dispose()
        {
            m_blob_bottom_targets.Dispose();
            base.dispose();
        }

        public Blob<T> BottomTargets
        {
            get { return m_blob_bottom_targets; }
        }

        public double SigmoidCrossEntropyLossReference(int nCount, int nNum, T[] rgInput, T[] rgTarget)
        {
            double[] rgdfInput = convert(rgInput);
            double[] rgdfTarget = convert(rgTarget);
            double dfLoss = 0;

            for (int i = 0; i < nCount; i++)
            {
                double dfInput = rgdfInput[i];
                double dfTarget = rgdfTarget[i];
                double dfPrediction = 1 / (1 + Math.Exp(-dfInput));

                double dfTargetEqualOne = (dfTarget == 1.0) ? 1.0 : 0.0;
                double dfTargetEqualZero = (dfTarget == 0.0) ? 1.0 : 0.0;

                m_log.CHECK_LE(dfPrediction, 1.0, "The prediction should be <= 1.0");
                m_log.CHECK_GE(dfPrediction, 0.0, "The preduction should be >= 0.0");
                m_log.CHECK_LE(dfTarget, 1.0, "The target should be <= 1.0");
                m_log.CHECK_GE(dfTarget, 0.0, "The target should be >= 0.0");
                dfLoss -= dfTarget * Math.Log(dfPrediction + dfTargetEqualZero);
                dfLoss -= (1 - dfTarget) * Math.Log(1 - dfPrediction + dfTargetEqualOne);
            }

            return dfLoss / nNum;
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SIGMOIDCROSSENTROPY_LOSS);
            double kLossWeight = 3.7;
            p.loss_weight.Add(kLossWeight);

            FillerParameter data_filler_param = new FillerParameter("gaussian");
            data_filler_param.std = 1.0;
            Filler<T> data_filler = Filler<T>.Create(m_cuda, m_log, data_filler_param);
            
            FillerParameter target_filler_param = new FillerParameter("uniform");
            target_filler_param.min = 0;
            target_filler_param.max = 1;
            Filler<T> target_filler = Filler<T>.Create(m_cuda, m_log, target_filler_param);

            double dfEps = 2e-2;

            for (int i = 0; i < 100; i++)
            {
                // Fill the data vector.
                data_filler.Fill(Bottom);
                // Fill the targets vector.
                target_filler.Fill(BottomTargets);

                SigmoidCrossEntropyLossLayer<T> layer = new SigmoidCrossEntropyLossLayer<T>(m_cuda, m_log, p);

                try
                {
                    layer.Setup(BottomVec, TopVec);

                    double dfLayerLoss = layer.Forward(BottomVec, TopVec);
                    int nCount = Bottom.count();
                    int nNum = Bottom.num;
                    T[] rgBottomData = Bottom.update_cpu_data();
                    T[] rgBottomTargets = BottomTargets.update_cpu_data();

                    double dfReferenceLoss = SigmoidCrossEntropyLossReference(nCount, nNum, rgBottomData, rgBottomTargets);
                    dfReferenceLoss *= kLossWeight;

                    m_log.EXPECT_NEAR(dfReferenceLoss, dfLayerLoss, dfEps, "Sigmoid cross entropy loss Forward - Debug: trial #" + i.ToString());
                }
                finally
                {
                    layer.Dispose();
                }
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SIGMOIDCROSSENTROPY_LOSS);
            double kLossWeight = 3.7;
            p.loss_weight.Add(kLossWeight);
            SigmoidCrossEntropyLossLayer<T> layer = new SigmoidCrossEntropyLossLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestIgnoreGradient()
        {
            FillerParameter data_filler_param = new FillerParameter("gaussian", 0, 0, 1);
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, data_filler_param);
            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SIGMOIDCROSSENTROPY_LOSS);
            p.loss_param.ignore_label = -1;

            long hTarget = BottomTargets.mutable_gpu_data;
            int nCount = BottomTargets.count();

            // Ignore half of targets, then check that diff of this half is zero,
            // while the other half is nonzero.
            m_cuda.set(nCount / 2, hTarget, -1);

            SigmoidCrossEntropyLossLayer<T> layer = new SigmoidCrossEntropyLossLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                List<bool> rgbPropagateDown = new List<bool>();
                rgbPropagateDown.Add(true);
                rgbPropagateDown.Add(false);

                layer.Backward(TopVec, rgbPropagateDown, BottomVec);

                double[] rgDiff = convert(Bottom.update_cpu_diff());

                for (int i = 0; i < nCount / 2; i++)
                {
                    double dfVal1 = rgDiff[i];
                    double dfVal2 = rgDiff[i + nCount / 2];

                    m_log.EXPECT_EQUAL<float>(dfVal1, 0, "The " + i.ToString() + "th value of the first half should be zero.");
                    m_log.CHECK_NE(dfVal2, 0, "The " + i.ToString() + "th value of the second half should not be zero.");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
