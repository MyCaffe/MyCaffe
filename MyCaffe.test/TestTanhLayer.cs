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
    public class TestTanhLayer
    {
        [TestMethod]
        public void TestForward()
        {
            TanhLayerTest2 test = new TanhLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITanhLayerTest2 t in test.Tests)
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
            TanhLayerTest2 test = new TanhLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITanhLayerTest2 t in test.Tests)
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
        public void TestOverflow()
        {
            TanhLayerTest2 test = new TanhLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITanhLayerTest2 t in test.Tests)
                {
                    t.TestTanhOverflow();
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
            TanhLayerTest2 test = new TanhLayerTest2(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ITanhLayerTest2 t in test.Tests)
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
            TanhLayerTest2 test = new TanhLayerTest2(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ITanhLayerTest2 t in test.Tests)
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
        public void TestOverflowCuDnn()
        {
            TanhLayerTest2 test = new TanhLayerTest2(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ITanhLayerTest2 t in test.Tests)
                {
                    t.TestTanhOverflow();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ITanhLayerTest2 : ITest
    {
        void TestForward();
        void TestGradient();
        void TestTanhOverflow();
    }

    class TanhLayerTest2 : TestBase
    {
        public TanhLayerTest2(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Tanh Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TanhLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new TanhLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class TanhLayerTest2<T> : TestEx<T>, ITanhLayerTest2
    {
        public TanhLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected double tanh_native(double x)
        {
            // avoid negative overflow.
            if (x < -40)
                return -1;

            // avoid positive overflow.
            if (x > 40)
                return 1;

            double dfExp2x = Math.Exp(2 * x);
            return (dfExp2x - 1.0) / (dfExp2x + 1.0);
        }

        public void TestForward(double dfFillerStd)
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = dfFillerStd;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TANH);
            p.tanh_param.engine = m_engine;
            TanhLayer<T> layer = new TanhLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // Now, check values
                double[] rgBottomData = convert(Bottom.update_cpu_data());
                double[] rgTopData = convert(Top.update_cpu_data());
                double dfMinPrecision = 1e-5;

                for (int i = 0; i < Bottom.count(); i++)
                {
                    double dfExpectedValue = tanh_native(rgBottomData[i]);
                    double dfPrecision = Math.Max(Math.Abs(dfExpectedValue * 1e-4), dfMinPrecision);
                    m_log.EXPECT_NEAR(dfExpectedValue, rgTopData[i], dfPrecision);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackward(double dfFillerStd)
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = dfFillerStd;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TANH);
            p.tanh_param.engine = m_engine;
            TanhLayer<T> layer = new TanhLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientEltwise(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward()
        {
            TestForward(1.0);
        }

        public void TestGradient()
        {
            TestBackward(1.0);
        }

        public void TestTanhOverflow()
        {
            TestForward(10000.0);
        }
    }
}
