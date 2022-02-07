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
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe.data;

/// <summary>
/// Testing the Math layer.
/// 
/// Math Layer - this layer performs mathematical functions on the data as a neuron layer.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestMathLayer
    {
        [TestMethod]
        public void TestForwardAcos()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.ACOS);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        //[TestMethod]
        //public void TestGradientAcos()
        //{
        //    MathLayerTest test = new MathLayerTest();

        //    try
        //    {
        //        foreach (IMathLayerTest t in test.Tests)
        //        {
        //            t.TestGradient(MyCaffe.common.MATH_FUNCTION.ACOS);
        //        }
        //    }
        //    finally
        //    {
        //        test.Dispose();
        //    }
        //}

        [TestMethod]
        public void TestForwardAcosh()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.ACOSH);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        //[TestMethod]
        //public void TestGradientAcosh()
        //{
        //    MathLayerTest test = new MathLayerTest();

        //    try
        //    {
        //        foreach (IMathLayerTest t in test.Tests)
        //        {
        //            t.TestGradient(MyCaffe.common.MATH_FUNCTION.ACOSH);
        //        }
        //    }
        //    finally
        //    {
        //        test.Dispose();
        //    }
        //}

        [TestMethod]
        public void TestForwardCos()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.COS);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientCos()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestGradient(MyCaffe.common.MATH_FUNCTION.COS);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardCosh()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.COSH);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientCosh()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestGradient(MyCaffe.common.MATH_FUNCTION.COSH);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAsin()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.ASIN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        //[TestMethod]
        //public void TestGradientAsin()
        //{
        //    MathLayerTest test = new MathLayerTest();

        //    try
        //    {
        //        foreach (IMathLayerTest t in test.Tests)
        //        {
        //            t.TestGradient(MyCaffe.common.MATH_FUNCTION.ASIN);
        //        }
        //    }
        //    finally
        //    {
        //        test.Dispose();
        //    }
        //}

        [TestMethod]
        public void TestForwardAsinh()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.ASINH);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAsinh()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestGradient(MyCaffe.common.MATH_FUNCTION.ASINH);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardSin()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.SIN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientSin()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestGradient(MyCaffe.common.MATH_FUNCTION.SIN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardSinh()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.SINH);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientSinh()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestGradient(MyCaffe.common.MATH_FUNCTION.SINH);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAtan()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.ATAN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAtan()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestGradient(MyCaffe.common.MATH_FUNCTION.ATAN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAtanh()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.ATANH);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        //[TestMethod]
        //public void TestGradientAtanh()
        //{
        //    MathLayerTest test = new MathLayerTest();

        //    try
        //    {
        //        foreach (IMathLayerTest t in test.Tests)
        //        {
        //            t.TestGradient(MyCaffe.common.MATH_FUNCTION.ATANH);
        //        }
        //    }
        //    finally
        //    {
        //        test.Dispose();
        //    }
        //}

        [TestMethod]
        public void TestForwardTan()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.TAN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientTan()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestGradient(MyCaffe.common.MATH_FUNCTION.TAN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTanh()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.TANH);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientTanh()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestGradient(MyCaffe.common.MATH_FUNCTION.TANH);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardCeil()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.CEIL);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        //[TestMethod]
        //public void TestGradientCeil()
        //{
        //    MathLayerTest test = new MathLayerTest();

        //    try
        //    {
        //        foreach (IMathLayerTest t in test.Tests)
        //        {
        //            t.TestGradient(MyCaffe.common.MATH_FUNCTION.CEIL);
        //        }
        //    }
        //    finally
        //    {
        //        test.Dispose();
        //    }
        //}

        [TestMethod]
        public void TestForwardFloor()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.FLOOR);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        //[TestMethod]
        //public void TestGradientFloor()
        //{
        //    MathLayerTest test = new MathLayerTest();

        //    try
        //    {
        //        foreach (IMathLayerTest t in test.Tests)
        //        {
        //            t.TestGradient(MyCaffe.common.MATH_FUNCTION.FLOOR);
        //        }
        //    }
        //    finally
        //    {
        //        test.Dispose();
        //    }
        //}

        [TestMethod]
        public void TestForwardNeg()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.NEG);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        //[TestMethod]
        //public void TestGradientNeg()
        //{
        //    MathLayerTest test = new MathLayerTest();

        //    try
        //    {
        //        foreach (IMathLayerTest t in test.Tests)
        //        {
        //            t.TestGradient(MyCaffe.common.MATH_FUNCTION.NEG);
        //        }
        //    }
        //    finally
        //    {
        //        test.Dispose();
        //    }
        //}

        [TestMethod]
        public void TestForwardSign()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.SIGN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        //[TestMethod]
        //public void TestGradientSign()
        //{
        //    MathLayerTest test = new MathLayerTest();

        //    try
        //    {
        //        foreach (IMathLayerTest t in test.Tests)
        //        {
        //            t.TestGradient(MyCaffe.common.MATH_FUNCTION.SIGN);
        //        }
        //    }
        //    finally
        //    {
        //        test.Dispose();
        //    }
        //}

        [TestMethod]
        public void TestForwardSqrt()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestForward(MyCaffe.common.MATH_FUNCTION.SQRT);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientSqrt()
        {
            MathLayerTest test = new MathLayerTest();

            try
            {
                foreach (IMathLayerTest t in test.Tests)
                {
                    t.TestGradient(MyCaffe.common.MATH_FUNCTION.SQRT);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IMathLayerTest : ITest
    {
        void TestForward(MyCaffe.common.MATH_FUNCTION fn);
        void TestGradient(MyCaffe.common.MATH_FUNCTION fn);
    }

    class MathLayerTest : TestBase
    {
        public MathLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Math Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MathLayerTest<double>(strName, nDeviceID, engine);
            else
                return new MathLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class MathLayerTest<T> : TestEx<T>, IMathLayerTest
    {
        public MathLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestForward(MyCaffe.common.MATH_FUNCTION fn)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MATH);
            p.math_param.function = fn;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgBtmData = convert(Bottom.mutable_cpu_data);
                double[] rgTopData = convert(Top.mutable_cpu_data);

                for (int i = 0; i < rgTopData.Length; i++)
                {
                    double dfBtm = rgBtmData[i];
                    double dfTop = rgTopData[i];
                    double dfExpected = 0;

                    switch (fn)
                    {
                        case MATH_FUNCTION.ACOS:
                            dfExpected = Math.Acos(dfBtm);
                            break;

                        case MATH_FUNCTION.ACOSH:
                            dfExpected = MathHelper.HArccos(dfBtm);
                            break;

                        case MATH_FUNCTION.COS:
                            dfExpected = Math.Cos(dfBtm);
                            break;

                        case MATH_FUNCTION.COSH:
                            dfExpected = Math.Cosh(dfBtm);
                            break;

                        case MATH_FUNCTION.ASIN:
                            dfExpected = Math.Asin(dfBtm);
                            break;

                        case MATH_FUNCTION.ASINH:
                            dfExpected = MathHelper.HArcsin(dfBtm);
                            break;

                        case MATH_FUNCTION.SIN:
                            dfExpected = Math.Sin(dfBtm);
                            break;

                        case MATH_FUNCTION.SINH:
                            dfExpected = Math.Sinh(dfBtm);
                            break;

                        case MATH_FUNCTION.ATAN:
                            dfExpected = Math.Atan(dfBtm);
                            break;

                        case MATH_FUNCTION.ATANH:
                            dfExpected = MathHelper.HArctan(dfBtm);
                            break;

                        case MATH_FUNCTION.TAN:
                            dfExpected = Math.Tan(dfBtm);
                            break;

                        case MATH_FUNCTION.TANH:
                            dfExpected = Math.Tanh(dfBtm);
                            break;

                        case MATH_FUNCTION.CEIL:
                            dfExpected = Math.Ceiling(dfBtm);
                            break;

                        case MATH_FUNCTION.FLOOR:
                            dfExpected = Math.Floor(dfBtm);
                            break;

                        case MATH_FUNCTION.NEG:
                            dfExpected = dfBtm * -1;
                            break;

                        case MATH_FUNCTION.SIGN:
                            dfExpected = Math.Sign(dfBtm);
                            break;

                        case MATH_FUNCTION.SQRT:
                            dfExpected = Math.Sqrt(dfBtm);
                            break;
                    }

                    if (double.IsNaN(dfExpected) || double.IsInfinity(dfExpected))
                        dfExpected = 0;

                    m_log.EXPECT_NEAR_FLOAT(dfTop, dfExpected, 1e-4, "The value is not as expected for fn = " + fn.ToString());
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient(MyCaffe.common.MATH_FUNCTION fn)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MATH);
            p.math_param.function = fn;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 0.01, 0.01);
                checker.CheckGradientEltwise(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }

    /// <summary>
    /// Math helper functions.
    /// </summary>
    /// <remarks>
    /// @see [David Relihan's Formulas](https://stackoverflow.com/questions/2840798/c-sharp-math-class-question)
    /// </remarks>
    class MathHelper
    {
        // Secant 
        public static double Sec(double x)
        {
            return 1 / Math.Cos(x);
        }

        // Cosecant
        public static double Cosec(double x)
        {
            return 1 / Math.Sin(x);
        }

        // Cotangent 
        public static double Cotan(double x)
        {
            return 1 / Math.Tan(x);
        }

        // Inverse Sine 
        public static double Arcsin(double x)
        {
            return Math.Atan(x / Math.Sqrt(-x * x + 1));
        }

        // Inverse Cosine 
        public static double Arccos(double x)
        {
            return Math.Atan(-x / Math.Sqrt(-x * x + 1)) + 2 * Math.Atan(1);
        }


        // Inverse Secant 
        public static double Arcsec(double x)
        {
            return 2 * Math.Atan(1) - Math.Atan(Math.Sign(x) / Math.Sqrt(x * x - 1));
        }

        // Inverse Cosecant 
        public static double Arccosec(double x)
        {
            return Math.Atan(Math.Sign(x) / Math.Sqrt(x * x - 1));
        }

        // Inverse Cotangent 
        public static double Arccotan(double x)
        {
            return 2 * Math.Atan(1) - Math.Atan(x);
        }

        // Hyperbolic Sine 
        public static double HSin(double x)
        {
            return (Math.Exp(x) - Math.Exp(-x)) / 2;
        }

        // Hyperbolic Cosine 
        public static double HCos(double x)
        {
            return (Math.Exp(x) + Math.Exp(-x)) / 2;
        }

        // Hyperbolic Tangent 
        public static double HTan(double x)
        {
            return (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
        }

        // Hyperbolic Secant 
        public static double HSec(double x)
        {
            return 2 / (Math.Exp(x) + Math.Exp(-x));
        }

        // Hyperbolic Cosecant 
        public static double HCosec(double x)
        {
            return 2 / (Math.Exp(x) - Math.Exp(-x));
        }

        // Hyperbolic Cotangent 
        public static double HCotan(double x)
        {
            return (Math.Exp(x) + Math.Exp(-x)) / (Math.Exp(x) - Math.Exp(-x));
        }

        // Inverse Hyperbolic Sine 
        public static double HArcsin(double x)
        {
            return Math.Log(x + Math.Sqrt(x * x + 1));
        }

        // Inverse Hyperbolic Cosine 
        public static double HArccos(double x)
        {
            return Math.Log(x + Math.Sqrt(x * x - 1));
        }

        // Inverse Hyperbolic Tangent 
        public static double HArctan(double x)
        {
            return Math.Log((1 + x) / (1 - x)) / 2;
        }

        // Inverse Hyperbolic Secant 
        public static double HArcsec(double x)
        {
            return Math.Log((Math.Sqrt(-x * x + 1) + 1) / x);
        }

        // Inverse Hyperbolic Cosecant 
        public static double HArccosec(double x)
        {
            return Math.Log((Math.Sign(x) * Math.Sqrt(x * x + 1) + 1) / x);
        }

        // Inverse Hyperbolic Cotangent 
        public static double HArccotan(double x)
        {
            return Math.Log((x + 1) / (x - 1)) / 2;
        }

        // Logarithm to base N 
        public static double LogN(double x, double n)
        {
            return Math.Log(x) / Math.Log(n);
        }
    }
}
