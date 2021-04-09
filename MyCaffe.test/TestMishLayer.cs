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
    public class TestMishLayer
    {
        [TestMethod]
        public void TestForward()
        {
            MishLayerTest2 test = new MishLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMishLayerTest2 t in test.Tests)
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
            MishLayerTest2 test = new MishLayerTest2(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IMishLayerTest2 t in test.Tests)
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

    interface IMishLayerTest2 : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class MishLayerTest2 : TestBase
    {
        public MishLayerTest2(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Mish Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MishLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new MishLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class MishLayerTest2<T> : TestEx<T>, IMishLayerTest2
    {
        public MishLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        /// <summary>
        /// Calculate the native MISH
        /// </summary>
        /// <param name="x">Specifies the input.</param>
        /// <returns>The calculated Mish value is returned.</returns>
        /// <remarks>
        /// Computes the mish non-linearity @f$ y  = x * tanh(log( 1 + exp(x) )) @f$.
        /// @see [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681v1) by Diganta Misra, 2019.
        /// </remarks>
        protected double mish_native(double x)
        {
            return x * Math.Tanh(Math.Log(1 + Math.Exp(x)));
        }

        /// <summary>
        /// Calculate the native MISH gradient
        /// </summary>
        /// <param name="x">Specifies the input.</param>
        /// <returns>The calculated Mish value is returned.</returns>
        /// <remarks>
        /// Computes the mish non-linearity @f$ y  = x * \tanh(\ln( 1 + \exp(x) )) @f$.
        /// with                            @f$ y' = \frac{\exp(x) * (4*\exp(x) * x + 4*x + 6*\exp(x) + 4*\exp(2x) + \exp(3x) + 4)}{(2*\exp(x) + \exp(2x) + 2)^2} @f$
        /// Note, see Wolfram Alpha with 'derivative of x * tanh(log(1 + e^x))'                                         
        protected double mish_native_grad(double x)
        {
            double dfExpx = Math.Exp(x);
            double dfExp2x = Math.Exp(2 * x);
            double dfExp3x = Math.Exp(3 * x);

            double dfVal1 = dfExpx * (4*dfExpx*x + 4*x + 6*dfExpx + 4*dfExp2x + dfExp3x + 4);
            double dfVal2a = 2 * dfExpx + dfExp2x + 2;
            double dfVal2 = dfVal2a * dfVal2a;

            if (dfVal2 == 0)
                return 0;

            return dfVal1 / dfVal2;
        }

        public void TestForward(double dfFillerStd)
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = dfFillerStd;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MISH);
            p.mish_param.engine = m_engine;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.MISH, "The layer type is incorrect!");

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgBottomData = convert(Bottom.update_cpu_data());
            double[] rgTopData = convert(Top.update_cpu_data());
            double dfMinPrecision = 1e-5;

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfExpectedValue = mish_native(rgBottomData[i]);
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

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MISH);
            p.mish_param.engine = m_engine;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.MISH, "The layer type is incorrect!");
                
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }

        public void TestForward()
        {
            TestForward(1.0);
        }

        public void TestGradient()
        {
            TestBackward(1.0);
        }
    }
}
