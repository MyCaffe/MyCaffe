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
        /// Computes the mish non-linearity @f$ y  = x * tanh(log( 1 + exp(x) )) @f$.
        /// with                            @f$ y' = ((2*exp(x) * x * (1 + exp(x))) / ((1 + exp(x)) + 1)) - 
        ///                                          ((2*exp(x) * x * ((1 + exp(x))^2 - 1) * (1 + exp(x))) / ((1 + exp(x))^2 - 1)^2) + 
        ///                                          (((1 + exp(x))^2 - 1) / ((1 + exp(x))^2 + 1)) @f$
        /// Note, see Wolfram Alpha with 'derivative of x * tanh(log(1 + e^x))'                                         
        protected double mish_native_grad(double x)
        {
            double dfExpx = Math.Exp(x);
            double dfTwoExpxX = 2 * dfExpx * x;
            double dfOne_p_expx = 1 + dfExpx;
            double dfOne_p_expx_sq = dfOne_p_expx * dfOne_p_expx;
            double dfOne_p_expx_sq_m_one = dfOne_p_expx_sq - 1;
            double dfOne_p_expx_sq_p_one = dfOne_p_expx_sq + 1;

            double dfVal1 = (dfTwoExpxX * dfOne_p_expx) / dfOne_p_expx_sq_p_one;
            double dfVal2 = (dfTwoExpxX * dfOne_p_expx_sq_m_one * dfOne_p_expx) / (dfOne_p_expx_sq_p_one * dfOne_p_expx_sq_p_one);
            double dfVal3 = (dfOne_p_expx_sq_m_one / dfOne_p_expx_sq_p_one);

            return dfVal1 - dfVal2 + dfVal3;
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

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double[] rgFwdBottomData = convert(Bottom.update_cpu_data());
            double[] rgFwdTopData = convert(Top.update_cpu_data());

            m_cuda.copy(TopVec[0].count(), TopVec[0].gpu_data, TopVec[0].mutable_gpu_diff);

            layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

            // Now, check values
            double[] rgBwdBottomDiff = convert(Bottom.update_cpu_diff());
            double dfMinPrecision = 1e-5;

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfExpectedValueFwd = mish_native(rgFwdBottomData[i]);
                double dfExpectedValueBwd = mish_native_grad(dfExpectedValueFwd) * dfExpectedValueFwd;
                double dfPrecision = Math.Max(Math.Abs(dfExpectedValueBwd * 1e-4), dfMinPrecision);
                double dfActual = rgBwdBottomDiff[i];

                m_log.EXPECT_NEAR(dfExpectedValueBwd, dfActual, dfPrecision);
            }
        }


        public void TestBackward2(double dfFillerStd)
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
