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
    public class TestMAELossLayer
    {
        [TestMethod]
        public void TestForward()
        {
            MAELossLayerTest test = new MAELossLayerTest();

            try
            {
                foreach (IMAELossLayerTest t in test.Tests)
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
            MAELossLayerTest test = new MAELossLayerTest();

            try
            {
                foreach (IMAELossLayerTest t in test.Tests)
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

    interface IMAELossLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class MAELossLayerTest : TestBase
    {
        public MAELossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("MAELoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MAELossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new MAELossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class MAELossLayerTest<T> : TestEx<T>, IMAELossLayerTest
    {
        Blob<T> m_blob_bottom_target;

        public MAELossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 10, 5, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blob_bottom_target = new Blob<T>(m_cuda, m_log, Bottom);

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, getFillerParam());
            filler.Fill(m_blob_bottom_target);

            BottomVec.Add(m_blob_bottom_target);
        }

        protected override void dispose()
        {
            m_blob_bottom_target.Dispose();
            base.dispose();
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MAE_LOSS);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            layer.Setup(BottomVec, TopVec);
            double dfLoss1 = layer.Forward(BottomVec, TopVec);
            double dfLoss = convert(TopVec[0].GetData(0));

            m_log.CHECK_EQ(dfLoss1, dfLoss, "The loss is incorrect!");

            double[] rgPredicted = convert(m_blob_bottom.mutable_cpu_data);
            double[] rgTarget = convert(m_blob_bottom_target.mutable_cpu_data);
            double dfSum = 0;

            for (int i = 0; i < rgPredicted.Length; i++)
            {
                double dfDiff = Math.Abs(rgTarget[i] - rgPredicted[i]);
                dfSum += dfDiff;
            }

            int nNum = m_blob_bottom.shape()[p.mae_loss_param.axis];
            double dfExpectedLoss = dfSum / nNum;

            m_log.EXPECT_NEAR_FLOAT(dfExpectedLoss, dfLoss, 0.000001, "The loss is incorrect!");
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MAE_LOSS);
            double kLossWeight = 3.7;
            p.loss_weight.Add(kLossWeight);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            layer.Setup(BottomVec, TopVec);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
