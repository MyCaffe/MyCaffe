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
    public class TestMeanErrorLossLayer
    {
        [TestMethod]
        public void TestForward_MAE()
        {
            MeanErrorLossLayerTest test = new MeanErrorLossLayerTest();

            try
            {
                foreach (IMeanErrorLossLayerTest t in test.Tests)
                {
                    t.TestForward(MEAN_ERROR.MAE);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient_MAE()
        {
            MeanErrorLossLayerTest test = new MeanErrorLossLayerTest();

            try
            {
                foreach (IMeanErrorLossLayerTest t in test.Tests)
                {
                    t.TestGradient(MEAN_ERROR.MAE);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward_MSE()
        {
            MeanErrorLossLayerTest test = new MeanErrorLossLayerTest();

            try
            {
                foreach (IMeanErrorLossLayerTest t in test.Tests)
                {
                    t.TestForward(MEAN_ERROR.MSE);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient_MSE()
        {
            MeanErrorLossLayerTest test = new MeanErrorLossLayerTest();

            try
            {
                foreach (IMeanErrorLossLayerTest t in test.Tests)
                {
                    t.TestGradient(MEAN_ERROR.MSE);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IMeanErrorLossLayerTest : ITest
    {
        void TestForward(MEAN_ERROR merr);
        void TestGradient(MEAN_ERROR merr);
    }

    class MeanErrorLossLayerTest : TestBase
    {
        public MeanErrorLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("MAELoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MeanErrorLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new MeanErrorLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class MeanErrorLossLayerTest<T> : TestEx<T>, IMeanErrorLossLayerTest
    {
        Blob<T> m_blob_bottom_target;

        public MeanErrorLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        public void TestForward(MEAN_ERROR merr)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MEAN_ERROR_LOSS);
            p.mean_error_loss_param.mean_error_type = merr;
            p.loss_param.normalization = LossParameter.NormalizationMode.BATCH_SIZE;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                layer.Setup(BottomVec, TopVec);
                double dfLoss1 = layer.Forward(BottomVec, TopVec);
                double dfLoss = convert(TopVec[0].GetData(0));

                m_log.CHECK_EQ(dfLoss1, dfLoss, "The loss is incorrect!");

                double[] rgPredicted = convert(m_blob_bottom.mutable_cpu_data);
                double[] rgTarget = convert(m_blob_bottom_target.mutable_cpu_data);
                double dfSum = 0;

                for (int i = 0; i < rgPredicted.Length; i++)
                {
                    double dfDiff = 0;

                    switch (merr)
                    {
                        case MEAN_ERROR.MSE:
                            dfDiff = Math.Pow(rgTarget[i] - rgPredicted[i], 2.0);
                            break;

                        case MEAN_ERROR.MAE:
                            dfDiff = Math.Abs(rgTarget[i] - rgPredicted[i]);
                            break;
                    }

                    dfSum += dfDiff;
                }
                
                int nNum = m_blob_bottom.num;
                double dfExpectedLoss = dfSum / nNum;

                m_log.EXPECT_NEAR_FLOAT(dfExpectedLoss, dfLoss, 0.000001, "The loss is incorrect!");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient(MEAN_ERROR merr)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MEAN_ERROR_LOSS);
            p.mean_error_loss_param.mean_error_type = merr;
            p.loss_param.normalization = LossParameter.NormalizationMode.BATCH_SIZE;
            double kLossWeight = 1.0;// 3.7;
            p.loss_weight.Add(kLossWeight);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

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
