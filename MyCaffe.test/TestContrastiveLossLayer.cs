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

namespace MyCaffe.test
{
    [TestClass]
    public class TestContrastiveLossLayer
    {
        [TestMethod]
        public void TestForward()
        {
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
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
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
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
        public void TestForwardLegacy()
        {
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
                {
                    t.TestForwardLegacy();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientLegacy()
        {
            ContrastiveLossLayerTest test = new ContrastiveLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IContrastiveLossLayerTest t in test.Tests)
                {
                    t.TestGradientLegacy();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IContrastiveLossLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
        void TestForwardLegacy();
        void TestGradientLegacy();
    }

    class ContrastiveLossLayerTest : TestBase
    {
        public ContrastiveLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Contrastive Loss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ContrastiveLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ContrastiveLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ContrastiveLossLayerTest<T> : TestEx<T>, IContrastiveLossLayerTest
    {
        CryptoRandom m_random = new CryptoRandom();
        Blob<T> m_blob_bottom_data_i;
        Blob<T> m_blob_bottom_data_j;
        Blob<T> m_blob_bottom_y;
        Blob<T> m_blob_top_loss;

        public ContrastiveLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            BottomVec.Clear();
            TopVec.Clear();

            m_blob_bottom_data_i = new Blob<T>(m_cuda, m_log, 5, 2, 1, 1);
            m_blob_bottom_data_j = new Blob<T>(m_cuda, m_log, 5, 2, 1, 1);
            m_blob_bottom_y = new Blob<T>(m_cuda, m_log, 5, 1, 1, 1);
            m_blob_top_loss = new Blob<T>(m_cuda, m_log);

            m_cuda.rng_setseed(1701);
            m_random = new CryptoRandom(CryptoRandom.METHOD.DEFAULT, 1701);

            // fill the values
            FillerParameter fp = new FillerParameter("uniform");
            fp.min = -1.0;
            fp.max = 1.0;   // distances ~=1.0 to test both sides of margin
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(m_blob_bottom_data_i);
            BottomVec.Add(m_blob_bottom_data_i);

            filler.Fill(m_blob_bottom_data_j);
            BottomVec.Add(m_blob_bottom_data_j);

            double[] rgdfY = convert(m_blob_bottom_y.mutable_cpu_data);

            for (int i = 0; i < m_blob_bottom_y.count(); i++)
            {
                rgdfY[i] = m_random.Next() % 2; // 0 or 1
            }

            m_blob_bottom_y.mutable_cpu_data = convert(rgdfY);
            BottomVec.Add(m_blob_bottom_y);
            TopVec.Add(m_blob_top_loss);
        }

        protected override void dispose()
        {
            m_blob_bottom_data_i.Dispose();
            m_blob_bottom_data_j.Dispose();
            m_blob_bottom_y.Dispose();
            m_blob_top_loss.Dispose();
            base.dispose();
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONTRASTIVE_LOSS);
            ContrastiveLossLayer<T> layer = new ContrastiveLossLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // manually compute to compare.
            double dfMargin = p.contrastive_loss_param.margin;
            int nNum = m_blob_bottom_data_i.num;
            int nChannels = m_blob_bottom_data_i.channels;
            double dfLoss = 0;

            double[] rgData_i = convert(m_blob_bottom_data_i.update_cpu_data());
            double[] rgData_j = convert(m_blob_bottom_data_j.update_cpu_data());
            double[] rgY = convert(m_blob_bottom_y.update_cpu_data());

            for (int i = 0; i < nNum; i++)
            {
                double dfDistSq = 0;

                for (int j = 0; j < nChannels; j++)
                {
                    int nIdx = i * nChannels + j;
                    double dfDiff = rgData_i[nIdx] - rgData_j[nIdx];
                    dfDistSq += dfDiff * dfDiff;
                }

                if (rgY[i] != 0)    // similar pairs
                {
                    dfLoss += dfDistSq;
                }
                else
                {
                    double dfDist = Math.Max(dfMargin - Math.Sqrt(dfDistSq), 0.0);
                    dfLoss += dfDist * dfDist;
                }
            }

            dfLoss /= nNum * 2.0;
            double dfTop = convert(m_blob_top_loss.GetData(0));

            m_log.EXPECT_NEAR(dfTop, dfLoss, 1e-6);
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONTRASTIVE_LOSS);
            ContrastiveLossLayer<T> layer = new ContrastiveLossLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);

            // check the gradient for the first two bottom layers
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 1);
        }

        public void TestForwardLegacy()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONTRASTIVE_LOSS);
            p.contrastive_loss_param.legacy_version = true;
            ContrastiveLossLayer<T> layer = new ContrastiveLossLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // manually compute to compare.
            double dfMargin = p.contrastive_loss_param.margin;
            int nNum = m_blob_bottom_data_i.num;
            int nChannels = m_blob_bottom_data_i.channels;
            double dfLoss = 0;

            double[] rgData_i = convert(m_blob_bottom_data_i.update_cpu_data());
            double[] rgData_j = convert(m_blob_bottom_data_j.update_cpu_data());
            double[] rgY = convert(m_blob_bottom_y.update_cpu_data());

            for (int i = 0; i < nNum; i++)
            {
                double dfDistSq = 0;

                for (int j = 0; j < nChannels; j++)
                {
                    int nIdx = i * nChannels + j;
                    double dfDiff = rgData_i[nIdx] - rgData_j[nIdx];
                    dfDistSq += dfDiff * dfDiff;
                }

                if (rgY[i] != 0)    // similar pairs
                    dfLoss += dfDistSq;
                else
                    dfLoss += Math.Max(dfMargin - dfDistSq, 0.0);
            }

            dfLoss /= (double)(nNum * 2.0);

            double dfTop = convert(m_blob_top_loss.GetData(0));
            m_log.EXPECT_NEAR(dfTop, dfLoss, 1e-6);
        }

        public void TestGradientLegacy()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONTRASTIVE_LOSS);
            p.contrastive_loss_param.legacy_version = true;
            ContrastiveLossLayer<T> layer = new ContrastiveLossLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);

            // check the gradient for the first two bottom layers
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 1);
        }
    }
}
