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
using System.Diagnostics;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSerfLayer
    {
        [TestMethod]
        public void TestForward()
        {
            SerfLayerTest2 test = new SerfLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ISerfLayerTest2 t in test.Tests)
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
            SerfLayerTest2 test = new SerfLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ISerfLayerTest2 t in test.Tests)
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
        public void TestGradient2()
        {
            SerfLayerTest2 test = new SerfLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ISerfLayerTest2 t in test.Tests)
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

    interface ISerfLayerTest2 : ITest
    {
        void TestForward();
        void TestGradient();
        void TestGradient2();
    }

    class SerfLayerTest2 : TestBase
    {
        public SerfLayerTest2(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Serf Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SerfLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new SerfLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class SerfLayerTest2<T> : TestEx<T>, ISerfLayerTest2
    {
        public SerfLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        protected double serf_native1(double x)
        {
            double dfVal = Math.Log(1 + Math.Exp(x));
            double dfErf = m_cuda.erf(dfVal);
            return dfErf;
        }

        /// <summary>
        /// Calculate the native SERF
        /// </summary>
        /// <param name="x">Specifies the input.</param>
        /// <returns>The calculated Mish value is returned.</returns>
        /// <remarks>
        /// Computes the serf non-linearity @f$ y  = x erf(\ln( 1 + \exp(x) )) @f$.
        /// with                            @f$ f(x)' = \text{erf}\left(\log \left(e^x+1\right)\right)+\frac{2 x e^{x-\log^2\left(e^x+1\right)}}{\sqrt{\pi } \left(e^x+1\right)} @f$
        /// @see [Serf: Towards better training of deep neural networks using log-Softplus ERror activation Function](https://arxiv.org/pdf/2108.09598.pdf) by Sayan Nag and Mayukh Bhattacharyya, 2021.
        /// </remarks>
        protected double serf_native(double x)
        {
            return x * serf_native1(x);
        }

        /// <summary>
        /// Calculate the native SERF gradient
        /// </summary>
        /// <param name="x">Specifies the input.</param>
        /// <returns>The calculated Serf value is returned.</returns>
        /// <remarks>
        /// Computes the serf non-linearity @f$ y  = x erf(\ln( 1 + \exp(x) )) @f$.
        /// with                            @f$ f(x)' = \text{erf}\left(\log \left(e^x+1\right)\right)+\frac{2 x e^{x-\log^2\left(e^x+1\right)}}{\sqrt{\pi } \left(e^x+1\right)} @f$
        /// @see [Serf: Towards better training of deep neural networks using log-Softplus ERror activation Function](https://arxiv.org/pdf/2108.09598.pdf) by Sayan Nag and Mayukh Bhattacharyya, 2021.
        protected double serf_native_grad(double x)
        {
            double dfSerf = serf_native1(x);

            double dfExpX = Math.Exp(x);
            double dfLogP = Math.Log(1 + dfExpX);
            double dfLogPSq = dfLogP * dfLogP;

            double dfNum = 2 * Math.Exp(x - dfLogPSq) * x;
            double dfDen = (1 + dfExpX)  * Math.Sqrt(Math.PI);

            return (dfNum / dfDen) + dfSerf;
        }

        public void TestForward(double dfFillerStd)
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = dfFillerStd;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SERF);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.SERF, "The layer type is incorrect!");

            layer.Setup(BottomVec, TopVec);
            m_cuda.debug();
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());
            double dfMinPrecision = 1e-5;

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfExpectedValue = serf_native(rgBottomData[i]);
                double dfPrecision = Math.Max(Math.Abs(dfExpectedValue * 1e-4), dfMinPrecision);
                m_log.EXPECT_NEAR(dfExpectedValue, rgTopData[i], dfPrecision);
            }
        }

        public void TestBackward(double dfFillerStd)
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = dfFillerStd;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SERF);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.SERF, "The layer type is incorrect!");

            layer.Setup(BottomVec, TopVec);
            m_cuda.debug();
            layer.Forward(BottomVec, TopVec);

            TopVec[0].SetDiff(1);
            layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

            // Now, check values
            double[] rgBottomDiff = convert(Bottom.update_cpu_diff());
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopDiff = convert(Top.update_cpu_diff());
            double[] rgTopData = convert(Top.update_cpu_data());
            double dfMinPrecision = 1e-5;

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfGrad = serf_native_grad(rgBottomData[i]);
                double dfExpectedValue = rgTopDiff[i] * dfGrad;

                double dfPrecision = Math.Max(Math.Abs(dfExpectedValue * 1e-4), dfMinPrecision);
                m_log.EXPECT_NEAR(dfExpectedValue, rgBottomDiff[i], dfPrecision);
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

        public void TestGradient2()
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = 1.0;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SERF);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.SERF, "The layer type is incorrect!");

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }
    }
}
