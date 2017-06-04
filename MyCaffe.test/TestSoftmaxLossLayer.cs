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
    public class TestSoftmaxLossLayer
    {
        [TestMethod]
        public void TestGradient()
        {
            SoftmaxLossLayerTest test = new SoftmaxLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ISoftmaxLossLayerTest t in test.Tests)
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
        public void TestForwardIgnoreLabel()
        {
            SoftmaxLossLayerTest test = new SoftmaxLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ISoftmaxLossLayerTest t in test.Tests)
                {
                    t.TestForwardIgnoreLabel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientIgnoreLabel()
        {
            SoftmaxLossLayerTest test = new SoftmaxLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ISoftmaxLossLayerTest t in test.Tests)
                {
                    t.TestGradientIgnoreLabel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientUnnormalized()
        {
            SoftmaxLossLayerTest test = new SoftmaxLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ISoftmaxLossLayerTest t in test.Tests)
                {
                    t.TestGradientUnnormalized();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientCuDnn()
        {
            SoftmaxLossLayerTest test = new SoftmaxLossLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ISoftmaxLossLayerTest t in test.Tests)
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
        public void TestForwardIgnoreLabelCuDnn()
        {
            SoftmaxLossLayerTest test = new SoftmaxLossLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ISoftmaxLossLayerTest t in test.Tests)
                {
                    t.TestForwardIgnoreLabel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientIgnoreLabelCuDnn()
        {
            SoftmaxLossLayerTest test = new SoftmaxLossLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ISoftmaxLossLayerTest t in test.Tests)
                {
                    t.TestGradientIgnoreLabel();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientUnnormalizedCuDnn()
        {
            SoftmaxLossLayerTest test = new SoftmaxLossLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ISoftmaxLossLayerTest t in test.Tests)
                {
                    t.TestGradientUnnormalized();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface ISoftmaxLossLayerTest : ITest
    {
        void TestGradient();
        void TestForwardIgnoreLabel();
        void TestGradientIgnoreLabel();
        void TestGradientUnnormalized();
    }

    class SoftmaxLossLayerTest : TestBase
    {
        public SoftmaxLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("SoftmaxLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SoftmaxLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SoftmaxLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SoftmaxLossLayerTest<T> : TestEx<T>, ISoftmaxLossLayerTest
    {
        CryptoRandom m_random = new CryptoRandom();
        Blob<T> m_blob_bottom_data;
        Blob<T> m_blob_bottom_label;
        Blob<T> m_blob_top_loss;

        public SoftmaxLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            BottomVec.Clear();
            TopVec.Clear();

            m_blob_bottom_data = new Blob<T>(m_cuda, m_log, 10, 5, 2, 3);
            m_blob_bottom_label = new Blob<T>(m_cuda, m_log, 10, 1, 2, 3);
            m_blob_top_loss = new Blob<T>(m_cuda, m_log);

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, getFillerParam());
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
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXWITH_LOSS);
            p.softmax_param.engine = m_engine;
            SoftmaxLossLayer<T> layer = new SoftmaxLossLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }

        public void TestForwardIgnoreLabel()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXWITH_LOSS);
            p.softmax_param.engine = m_engine;
            p.loss_param.normalize = false;
            p.loss_param.normalization = LossParameter.NormalizationMode.NONE;
            SoftmaxLossLayer<T> layer = new SoftmaxLossLayer<T>(m_cuda, m_log, p);

            // First compute the loss with all labels.
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double dfFullLoss = convert(m_blob_top_loss.GetData(0));

            // Now, accumulate the loss, ignoring each label in { 0, .., 4} in turn.
            double dfAccumLoss = 0;
            double dfLocalLoss = 0;

            for (int nLabel = 0; nLabel < 5; nLabel++)
            {
                p.loss_param.ignore_label = nLabel;
                layer.Dispose();
                layer = new SoftmaxLossLayer<T>(m_cuda, m_log, p);
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                dfLocalLoss = convert(m_blob_top_loss.GetData(0));
                dfAccumLoss += dfLocalLoss;

                layer.Dispose();
            }

            // Check that each label was included all but once.
            m_log.EXPECT_NEAR(4 * dfFullLoss, dfAccumLoss, 1e-4);
        }

        public void TestGradientIgnoreLabel()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXWITH_LOSS);
            p.softmax_param.engine = m_engine;
            // lables are in {0, ..., 4}, so we'll ignore about a fifth of them.
            p.loss_param.ignore_label = 0;
            SoftmaxLossLayer<T> layer = new SoftmaxLossLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }

        public void TestGradientUnnormalized()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXWITH_LOSS);
            p.softmax_param.engine = m_engine;
            p.loss_param.normalize = false;
            SoftmaxLossLayer<T> layer = new SoftmaxLossLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }
    }
}
