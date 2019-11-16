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
    public class TestHingeLossLayer
    {
        [TestMethod]
        public void TestGradient1()
        {
            HingeLossLayerTest test = new HingeLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IHingeLossLayerTest t in test.Tests)
                {
                    t.TestGradient1();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient2()
        {
            HingeLossLayerTest test = new HingeLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IHingeLossLayerTest t in test.Tests)
                {
                    t.TestGradient2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IHingeLossLayerTest : ITest
    {
        void TestGradient1();
        void TestGradient2();
    }

    class HingeLossLayerTest : TestBase
    {
        public HingeLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("HingeLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new HingeLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new HingeLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class HingeLossLayerTest<T> : TestEx<T>, IHingeLossLayerTest
    {
        CryptoRandom m_random = new CryptoRandom();
        Blob<T> m_blob_bottom_data;
        Blob<T> m_blob_bottom_label;
        Blob<T> m_blob_top_loss;

        public HingeLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            BottomVec.Clear();
            TopVec.Clear();

            m_blob_bottom_data = new Blob<T>(m_cuda, m_log, 10, 5, 1, 1);
            m_blob_bottom_label = new Blob<T>(m_cuda, m_log, 10, 1, 1, 1);
            m_blob_top_loss = new Blob<T>(m_cuda, m_log);

            m_cuda.rng_setseed(1701);
            m_random = new CryptoRandom(CryptoRandom.METHOD.DEFAULT, 1701);
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = 10;
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

        public void TestGradient1()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.HINGE_LOSS);
            HingeLossLayer<T> layer = new HingeLossLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701, 1, 0.01);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }

        public void TestGradient2()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.HINGE_LOSS);
            // Set norm to L2
            p.hinge_loss_param.norm = HingeLossParameter.Norm.L2;
            HingeLossLayer<T> layer = new HingeLossLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }
    }
}
