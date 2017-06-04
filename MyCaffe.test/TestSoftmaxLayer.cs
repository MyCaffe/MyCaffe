using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSoftmaxLayer
    {
        [TestMethod]
        public void TestForward()
        {
            SoftmaxLayerTest test = new SoftmaxLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
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
            SoftmaxLayerTest test = new SoftmaxLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
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
        public void TestForwardCuDnn()
        {
            SoftmaxLayerTest test = new SoftmaxLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
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
        public void TestGradientCuDnn()
        {
            SoftmaxLayerTest test = new SoftmaxLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (INeuronLayerTest t in test.Tests)
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

    interface ISoftmaxLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class SoftmaxLayerTest : TestBase
    {
        public SoftmaxLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Softmax Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SoftmaxLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SoftmaxLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SoftmaxLayerTest<T> : TestEx<T>, INeuronLayerTest
    {
        public SoftmaxLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            p.softmax_param.engine = m_engine;
            SoftmaxLayer<T> layer = new SoftmaxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Test sum
            for (int i = 0; i < Bottom.num; i++)
            {
                for (int k = 0; k < Bottom.height; k++)
                {
                    for (int l = 0; l < Bottom.width; l++)
                    {
                        double dfSum = 0;

                        for (int j = 0; j < Top.channels; j++)
                        {
                            dfSum += convert(Top.data_at(i, j, k, l));
                        }

                        m_log.CHECK_GE(dfSum, 0.999, "The sum should be greater than equal to 0.999");
                        m_log.CHECK_LE(dfSum, 1.001, "The sum should be less than or equal to 1.001");

                        // Test exact values
                        double dfScale = 0;

                        for (int j = 0; j < Bottom.channels; j++)
                        {
                            dfScale += Math.Exp(convert(Bottom.data_at(i, j, k, l)));
                        }

                        for (int j = 0; j < Bottom.channels; j++)
                        {
                            double dfTop = convert(Top.data_at(i, j, k, l));
                            double dfBottom = convert(Bottom.data_at(i, j, k, l));

                            m_log.CHECK_GE(dfTop + 1e-4, Math.Exp(dfBottom) / dfScale, "The value is out of range at " + i.ToString() + ", " + j.ToString());
                            m_log.CHECK_LE(dfTop - 1e-4, Math.Exp(dfBottom) / dfScale, "The value is out of range at " + i.ToString() + ", " + j.ToString());
                        }
                    }
                }
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            p.softmax_param.engine = m_engine;
            SoftmaxLayer<T> layer = new SoftmaxLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
