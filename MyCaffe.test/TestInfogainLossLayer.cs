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
    public class TestInfogainLossLayer
    {
        [TestMethod]
        public void TestInfogainLoss()
        {
            InfogainLossLayerTest test = new InfogainLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IInfogainLossLayerTest t in test.Tests)
                {
                    t.TestInfogainLoss();
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
            InfogainLossLayerTest test = new InfogainLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IInfogainLossLayerTest t in test.Tests)
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


    interface IInfogainLossLayerTest : ITest
    {
        void TestInfogainLoss();
        void TestGradient();
    }

    class InfogainLossLayerTest : TestBase
    {
        public InfogainLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Infogain Loss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new InfogainLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new InfogainLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class InfogainLossLayerTest<T> : TestEx<T>, IInfogainLossLayerTest
    {
        CryptoRandom m_random = new CryptoRandom();
        Blob<T> m_blob_bottom_data;
        Blob<T> m_blob_bottom_label;
        Blob<T> m_blob_bottom_infogain;
        Blob<T> m_blob_top_loss;
        Blob<T> m_blob_top_prob;
        int m_nInner = 2;
        int m_nOuter = 4 * 2;
        int m_nNumLabels = 5;

        public InfogainLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            BottomVec.Clear();
            TopVec.Clear();

            m_blob_bottom_data = new Blob<T>(m_cuda, m_log, 4, 2, 5, 2);
            m_blob_bottom_label = new Blob<T>(m_cuda, m_log, 4, 2, 1, 2);
            m_blob_bottom_infogain = new Blob<T>(m_cuda, m_log, 1, 1, 5, 5);
            m_blob_top_loss = new Blob<T>(m_cuda, m_log);
            m_blob_top_prob = new Blob<T>(m_cuda, m_log);

            m_cuda.rng_setseed(1701);
            m_random = new CryptoRandom(CryptoRandom.METHOD.DEFAULT, 1701);
            FillerParameter fp = new FillerParameter("positive_unitball");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(m_blob_bottom_data);
            BottomVec.Add(m_blob_bottom_data);

            double[] rgdfLabels = convert(m_blob_bottom_label.mutable_cpu_data);
            for (int i = 0; i < m_blob_bottom_label.count(); i++)
            {
                rgdfLabels[i] = m_random.Next() % 5;
            }

            m_blob_bottom_label.mutable_cpu_data = convert(rgdfLabels);
            BottomVec.Add(m_blob_bottom_label);

            fp = new FillerParameter("uniform");
            fp.min = 0.1;
            fp.max = 2.0;
            filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(m_blob_bottom_infogain);
            BottomVec.Add(m_blob_bottom_infogain);
            TopVec.Add(m_blob_top_loss);
            TopVec.Add(m_blob_top_prob);
        }

        protected override void dispose()
        {
            m_blob_bottom_data.Dispose();
            m_blob_bottom_label.Dispose();
            m_blob_bottom_infogain.Dispose();
            m_blob_top_loss.Dispose();
            m_blob_top_prob.Dispose();
            base.dispose();
        }

        public void TestInfogainLoss()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INFOGAIN_LOSS);
            p.infogain_loss_param.axis = 2;
            p.loss_weight.Clear();
            p.loss_weight.Add(1);
            p.loss_weight.Add(0);
            InfogainLossLayer<T> layer = new InfogainLossLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check the values.
            double[] rgData = convert(BottomVec[0].update_cpu_data());
            double[] rgProb = convert(TopVec[1].update_cpu_data());
            double[] rgLabels = convert(BottomVec[1].update_cpu_data());
            double[] rgH = convert(BottomVec[2].update_cpu_data());

            // first, test the prob top.
            m_log.CHECK_EQ(BottomVec[0].num_axes, TopVec[1].num_axes, "The prob top shape does not match the bottom data shape.");
            for (int ai = 0; ai < BottomVec[0].num; ai++)
            {
                m_log.CHECK_EQ(BottomVec[0].shape(ai), TopVec[1].shape(ai), "The prob shape does not match the bottom data shape.");
            }

            List<double> rgEstProb = Utility.Create<double>(m_nNumLabels, 0);
            for (int i = 0; i < m_nOuter; i++)
            {
                for (int j = 0; j < m_nInner; j++)
                {
                    double dfDen = 0;
                    for (int l = 0; l < m_nNumLabels; l++)
                    {
                        double dfVal = rgData[i * m_nNumLabels * m_nInner + l * m_nInner + j];
                        rgEstProb[l] = Math.Exp(dfVal);
                        dfDen += rgEstProb[l];
                    }
                    for (int l = 0; l < m_nNumLabels; l++)
                    {
                        double dfActualP = rgProb[i * m_nNumLabels * m_nInner + l * m_nInner + j];
                        double dfExpectedP = rgEstProb[l] / dfDen;

                        m_log.EXPECT_NEAR(dfActualP, dfExpectedP, 1e-6);
                    }
                }
            }

            double dfLoss = 0; // loss from prob top.
            for (int i = 0; i < m_nOuter; i++)
            {
                for (int j = 0; j < m_nInner; j++)
                {
                    int nGt = (int)rgLabels[i * m_nInner + j];

                    for (int l = 0; l < m_nNumLabels; l++)
                    {
                        double dfH = rgH[nGt * m_nNumLabels + l];
                        double dfProb = rgProb[i * m_nNumLabels * m_nInner + l * m_nInner + j];
                        double dfProbLog = Math.Log(Math.Max(dfProb, InfogainLossLayer<T>.kLOG_THRESHOLD));

                        dfLoss -= dfH * dfProbLog;
                    }
                }
            }

            double dfActual = convert(TopVec[0].GetData(0));
            double dfExpected = dfLoss / (m_nOuter * m_nInner);

            m_log.EXPECT_NEAR(dfActual, dfExpected, 1e-6);
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.INFOGAIN_LOSS);
            p.infogain_loss_param.axis = 2;
            InfogainLossLayer<T> layer = new InfogainLossLayer<T>(m_cuda, m_log, p);
            TopVec.Clear();
            TopVec.Add(m_blob_top_loss); // ignore prob top.
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-4, 2e-2, 1701); // no 'kink'
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }
    }
}
