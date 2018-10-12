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

namespace MyCaffe.test
{
    [TestClass]
    public class TestReductionLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestSetup();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupWithAxis1()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestSetup(1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupWithAxis2()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestSetup(2);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSum()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestForward(ReductionParameter.ReductionOp.SUM);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSumCoeff()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestForward(ReductionParameter.ReductionOp.SUM, 2.3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestSumCoeffAxis1()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestForward(ReductionParameter.ReductionOp.SUM, 2.3, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSumGradient()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestGradient(ReductionParameter.ReductionOp.SUM);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSumCoeffGradient()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestGradient(ReductionParameter.ReductionOp.SUM, 2.3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestSumCoeffAxis1Gradient()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestGradient(ReductionParameter.ReductionOp.SUM, 2.3, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMean()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestForward(ReductionParameter.ReductionOp.MEAN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMeanCoeff()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestForward(ReductionParameter.ReductionOp.MEAN, 2.3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestMeanCoeffAxis1()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestForward(ReductionParameter.ReductionOp.MEAN, 2.3, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMeanGradient()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestGradient(ReductionParameter.ReductionOp.MEAN);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMeanCoeffGradient()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestGradient(ReductionParameter.ReductionOp.MEAN, 2.3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestMeanCoeffAxis1Gradient()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestGradient(ReductionParameter.ReductionOp.MEAN, 2.3, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAbsSum()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestForward(ReductionParameter.ReductionOp.ASUM);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAbsSumCoeff()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestForward(ReductionParameter.ReductionOp.ASUM, 2.3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestAbsSumCoeffAxis1()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestForward(ReductionParameter.ReductionOp.ASUM, 2.3, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAbsSumGradient()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestGradient(ReductionParameter.ReductionOp.ASUM);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAbsSumCoeffGradient()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestGradient(ReductionParameter.ReductionOp.ASUM, 2.3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestAbsSumCoeffAxis1Gradient()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestGradient(ReductionParameter.ReductionOp.ASUM, 2.3, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSumOfSquares()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestForward(ReductionParameter.ReductionOp.SUMSQ);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSumOfSquaresCoeff()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestForward(ReductionParameter.ReductionOp.SUMSQ, 2.3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestSumOfSquaresCoeffAxis1()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestForward(ReductionParameter.ReductionOp.SUMSQ, 2.3, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSumOfSquaresGradient()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestGradient(ReductionParameter.ReductionOp.SUMSQ);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSumOfSquaresCoeffGradient()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestGradient(ReductionParameter.ReductionOp.SUMSQ, 2.3);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestSumOfSquaresCoeffAxis1Gradient()
        {
            ReductionLayerTest test = new ReductionLayerTest();

            try
            {
                foreach (IReductionLayerTest t in test.Tests)
                {
                    t.TestGradient(ReductionParameter.ReductionOp.SUMSQ, 2.3, 1);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IReductionLayerTest : ITest
    {
        void TestSetup(int nAxis = 0);
        void TestForward(ReductionParameter.ReductionOp op, double dfCoeff = 1, int nAxis = 0);
        void TestGradient(ReductionParameter.ReductionOp op, double dfCoeff = 1, int nAxis = 0);
    }

    class ReductionLayerTest : TestBase
    {
        public ReductionLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Reduction Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ReductionLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ReductionLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ReductionLayerTest<T> : TestEx<T>, IReductionLayerTest
    {
        public ReductionLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
            FillerParameter fp = new FillerParameter("uniform");

            //---------------------------------------------
            //  NOTE: The minimum random value is set to
            //  twice the step size (2e-3) used in the
            //  gradient check.  If a value for the gradient
            //  test is below the step size, the gradient
            //  estimated gradient fails to match up for
            //  the ABS Sum tests.  The reason for this is 
            //  that the Gradient Checker estimates the
            //  gradient by doing the following:
            //      add step size at item 'x'
            //      run forward
            //      sub 2*step size at item 'x'
            //      run forward
            //      subtract the results
            //
            //  This fails with values < step size * 2 for
            //  for the ASUM operations.
            //----------------------------------------------
            fp.min = 4e-3;
            fp.max = 1.0;
            return fp;
        }

        public void TestSetup(int nAxis = 0)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.REDUCTION);

            if (nAxis != 0)
                p.reduction_param.axis = nAxis;

            ReductionLayer<T> layer = new ReductionLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_top.num_axes, nAxis, "The top blob should have " + nAxis.ToString() + " axes.");

            if (nAxis > 0)
            {
                m_log.CHECK_EQ(m_blob_top.shape(0), 2, "The top blob shape(0) should have shape = 2.");

                if (nAxis > 1)
                    m_log.CHECK_EQ(m_blob_top.shape(1), 3, "The top blob shape(1) should have shape = 3.");
            }
        }

        public void TestForward(ReductionParameter.ReductionOp op, double dfCoeff = 1, int nAxis = 0)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.REDUCTION);
            p.reduction_param.operation = op;

            if (dfCoeff != 1.0)
                p.reduction_param.coeff = dfCoeff;

            if (nAxis != 0)
                p.reduction_param.axis = nAxis;

            ReductionLayer<T> layer = new ReductionLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double[] rgTopData = convert(m_blob_top.update_cpu_data());
            double[] rgInData = convert(m_blob_bottom.update_cpu_data());
            int nNum = m_blob_bottom.count(0, nAxis);
            int nDim = m_blob_bottom.count(nAxis);
            int nIdx = 0;

            for (int n = 0; n < nNum; n++)
            {
                double dfExpectedResult = 0;

                for (int d = 0; d < nDim; d++)
                {
                    switch (op)
                    {
                        case ReductionParameter.ReductionOp.SUM:
                            dfExpectedResult += rgInData[nIdx];
                            break;

                        case ReductionParameter.ReductionOp.MEAN:
                            dfExpectedResult += rgInData[nIdx] / nDim;
                            break;

                        case ReductionParameter.ReductionOp.ASUM:
                            dfExpectedResult += Math.Abs(rgInData[nIdx]);
                            break;

                        case ReductionParameter.ReductionOp.SUMSQ:
                            dfExpectedResult += (rgInData[nIdx] * rgInData[nIdx]);
                            break;

                        default:
                            m_log.FAIL("Unknown reduction op: " + op.ToString());
                            break;
                    }

                    nIdx++;
                }

                dfExpectedResult *= dfCoeff;
                double dfComputedResult = rgTopData[n];

                m_log.EXPECT_EQUAL<float>(dfExpectedResult, dfComputedResult, "Incorrect result computed with op " + op.ToString() + ", coeff " + dfCoeff.ToString() + " at n = " + n.ToString());
            }
        }

        public void TestGradient(ReductionParameter.ReductionOp op, double dfCoeff = 1, int nAxis = 0)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.REDUCTION);
            p.reduction_param.operation = op;
            p.reduction_param.coeff = dfCoeff;
            p.reduction_param.axis = nAxis;
            ReductionLayer<T> layer = new ReductionLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 2e-3);

            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
