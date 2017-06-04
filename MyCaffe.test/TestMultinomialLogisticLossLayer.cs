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
    public class TestMultinomialLogisticLossLayer
    {
        [TestMethod]
        public void TestGradient()
        {
            MultinomialLogisticLossLayerTest test = new MultinomialLogisticLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMultinomialLogisticLossLayerTest t in test.Tests)
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


    interface IMultinomialLogisticLossLayerTest : ITest
    {
        void TestGradient();
    }

    class MultinomialLogisticLossLayerTest : TestBase
    {
        public MultinomialLogisticLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("MultinomialLogistic Loss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MultinomialLogisticLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new MultinomialLogisticLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class MultinomialLogisticLossLayerTest<T> : TestEx<T>, IMultinomialLogisticLossLayerTest
    {
        CryptoRandom m_random = new CryptoRandom();
        Blob<T> m_blob_bottom_data;
        Blob<T> m_blob_bottom_label;
        Blob<T> m_blob_top_loss;

        public MultinomialLogisticLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            BottomVec.Clear();
            TopVec.Clear();

            m_blob_bottom_data = new Blob<T>(m_cuda, m_log, 10, 5, 1, 1);
            m_blob_bottom_label = new Blob<T>(m_cuda, m_log, 10, 1, 1, 1);
            m_blob_top_loss = new Blob<T>(m_cuda, m_log);

            m_cuda.rng_setseed(1701);
            m_random = new CryptoRandom(false, 1701);
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
            TopVec.Add(m_blob_top_loss);
        }

        protected override void dispose()
        {
            m_blob_bottom_data.Dispose();
            m_blob_bottom_label.Dispose();
            m_blob_top_loss.Dispose();
            base.dispose();
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTINOMIALLOGISTIC_LOSS);
            MultinomialLogisticLossLayer<T> layer = new MultinomialLogisticLossLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 2*1e-2, 1701, 0, 0.05);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }
    }
}
