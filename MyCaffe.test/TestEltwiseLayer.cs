using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.common;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.fillers;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestEltwiseLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
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
        public void TestProd()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestProd();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDiv()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestDiv();
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
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestSum();
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
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestSumCoeff();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSub()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestSub();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSubCoeff()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestSubCoeff();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestStableProdGradient()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestStableProdGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestUnstableProdGradient()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestUnstableProdGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDivGradient()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestDivGradient();
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
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestSumGradient();
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
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestSumCoeffGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSubGradient()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestSubGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSubCoeffGradient()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestSubCoeffGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMax()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestMax();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMaxGradient()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestMaxGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMin()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestMin();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMinGradient()
        {
            EltwiseLayerTest test = new EltwiseLayerTest();

            try
            {
                foreach (IEltwiseLayerTest t in test.Tests)
                {
                    t.TestMinGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IEltwiseLayerTest : ITest
    {
        void TestSetup();
        void TestProd();
        void TestDiv();
        void TestSum();
        void TestSumCoeff();
        void TestSub();
        void TestSubCoeff();
        void TestStableProdGradient();
        void TestDivGradient();
        void TestUnstableProdGradient();
        void TestSumGradient();
        void TestSumCoeffGradient();
        void TestSubGradient();
        void TestSubCoeffGradient();
        void TestMax();
        void TestMaxGradient();
        void TestMin();
        void TestMinGradient();
    }

    class EltwiseLayerTest : TestBase
    {
        public EltwiseLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Eltwise Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new EltwiseLayerTest<double>(strName, nDeviceID, engine);
            else
                return new EltwiseLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class EltwiseLayerTest<T> : TestEx<T>, IEltwiseLayerTest
    {
        Blob<T> m_blob_bottom_b;
        Blob<T> m_blob_bottom_c;

        public EltwiseLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;

            m_blob_bottom_b = new Blob<T>(m_cuda, m_log, Bottom);
            m_blob_bottom_c = new Blob<T>(m_cuda, m_log, Bottom);

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, getFillerParam());
            filler.Fill(m_blob_bottom_b);
            filler.Fill(m_blob_bottom_c);

            BottomVec.Add(m_blob_bottom_b);
            BottomVec.Add(m_blob_bottom_c);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public Blob<T> BottomB
        {
            get { return m_blob_bottom_b; }
        }

        public Blob<T> BottomC
        {
            get { return m_blob_bottom_c; }
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.PROD;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(2, Top.num, "Top num should equal 2.");
                m_log.CHECK_EQ(3, Top.channels, "Top channels should equal 3.");
                m_log.CHECK_EQ(4, Top.height, "Top height should equal 4.");
                m_log.CHECK_EQ(5, Top.width, "Top width should equal 5.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestProd()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.PROD;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(Top.update_cpu_data());
                int nCount = Top.count();
                double[] rgBottomA = convert(Bottom.update_cpu_data());
                double[] rgBottomB = convert(BottomB.update_cpu_data());
                double[] rgBottomC = convert(BottomC.update_cpu_data());

                for (int i = 0; i < nCount; i++)
                {
                    double dfTop = rgTop[i];
                    double dfBottomA = rgBottomA[i];
                    double dfBottomB = rgBottomB[i];
                    double dfBottomC = rgBottomC[i];
                    double dfExpected = dfBottomA * dfBottomB * dfBottomC;

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestDiv()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.DIV;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(Top.update_cpu_data());
                int nCount = Top.count();
                double[] rgBottomA = convert(Bottom.update_cpu_data());
                double[] rgBottomB = convert(BottomB.update_cpu_data());
                double[] rgBottomC = convert(BottomC.update_cpu_data());

                for (int i = 0; i < nCount; i++)
                {
                    double dfTop = rgTop[i];
                    double dfBottomA = rgBottomA[i];
                    double dfBottomB = rgBottomB[i];
                    double dfBottomC = rgBottomC[i];
                    double dfExpected = dfBottomA / dfBottomB / dfBottomC;

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSum()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUM;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(Top.update_cpu_data());
                int nCount = Top.count();
                double[] rgBottomA = convert(Bottom.update_cpu_data());
                double[] rgBottomB = convert(BottomB.update_cpu_data());
                double[] rgBottomC = convert(BottomC.update_cpu_data());

                for (int i = 0; i < nCount; i++)
                {
                    double dfTop = rgTop[i];
                    double dfBottomA = rgBottomA[i];
                    double dfBottomB = rgBottomB[i];
                    double dfBottomC = rgBottomC[i];
                    double dfExpected = dfBottomA + dfBottomB + dfBottomC;

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSumCoeff()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUM;
            p.eltwise_param.coeff.Add(1);
            p.eltwise_param.coeff.Add(-0.5);
            p.eltwise_param.coeff.Add(2);
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(Top.update_cpu_data());
                int nCount = Top.count();
                double[] rgBottomA = convert(Bottom.update_cpu_data());
                double[] rgBottomB = convert(BottomB.update_cpu_data());
                double[] rgBottomC = convert(BottomC.update_cpu_data());

                for (int i = 0; i < nCount; i++)
                {
                    double dfTop = rgTop[i];
                    double dfBottomA = rgBottomA[i];
                    double dfBottomB = rgBottomB[i];
                    double dfBottomC = rgBottomC[i];
                    double dfExpected = (1.0 * dfBottomA) + (-0.5 * dfBottomB) + (2.0 * dfBottomC);

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSub()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUB;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(Top.update_cpu_data());
                int nCount = Top.count();
                double[] rgBottomA = convert(Bottom.update_cpu_data());
                double[] rgBottomB = convert(BottomB.update_cpu_data());
                double[] rgBottomC = convert(BottomC.update_cpu_data());

                for (int i = 0; i < nCount; i++)
                {
                    double dfTop = rgTop[i];
                    double dfBottomA = rgBottomA[i];
                    double dfBottomB = rgBottomB[i];
                    double dfBottomC = rgBottomC[i];
                    double dfExpected = dfBottomA - dfBottomB - dfBottomC;

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSubCoeff()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUB;
            p.eltwise_param.coeff.Add(1);
            p.eltwise_param.coeff.Add(-0.5);
            p.eltwise_param.coeff.Add(2);
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(Top.update_cpu_data());
                int nCount = Top.count();
                double[] rgBottomA = convert(Bottom.update_cpu_data());
                double[] rgBottomB = convert(BottomB.update_cpu_data());
                double[] rgBottomC = convert(BottomC.update_cpu_data());

                for (int i = 0; i < nCount; i++)
                {
                    double dfTop = rgTop[i];
                    double dfBottomA = rgBottomA[i];
                    double dfBottomB = rgBottomB[i];
                    double dfBottomC = rgBottomC[i];
                    double dfExpected = (1.0 * dfBottomA) - (-0.5 * dfBottomB) - (2.0 * dfBottomC);

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestStableProdGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.PROD;
            p.eltwise_param.stable_prod_grad = true;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

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

        public void TestUnstableProdGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.PROD;
            p.eltwise_param.stable_prod_grad = false;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

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

        public void TestDivGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.DIV;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

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

        public void TestSumGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUM;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

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

        public void TestSumCoeffGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUM;
            p.eltwise_param.coeff.Add(1);
            p.eltwise_param.coeff.Add(-0.5);
            p.eltwise_param.coeff.Add(2);
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

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

        public void TestSubGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUB;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

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

        public void TestSubCoeffGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUB;
            p.eltwise_param.coeff.Add(1);
            p.eltwise_param.coeff.Add(-0.5);
            p.eltwise_param.coeff.Add(2);
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

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

        public void TestMax()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.MAX;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(Top.update_cpu_data());
                int nCount = Top.count();
                double[] rgBottomA = convert(Bottom.update_cpu_data());
                double[] rgBottomB = convert(BottomB.update_cpu_data());
                double[] rgBottomC = convert(BottomC.update_cpu_data());

                for (int i = 0; i < nCount; i++)
                {
                    double dfTop = rgTop[i];
                    double dfBottomA = rgBottomA[i];
                    double dfBottomB = rgBottomB[i];
                    double dfBottomC = rgBottomC[i];
                    double dfExpected = Math.Max(dfBottomA, Math.Max(dfBottomB, dfBottomC));

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestMaxGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.MAX;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-4);
                checker.CheckGradientEltwise(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestMin()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.MIN;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(Top.update_cpu_data());
                int nCount = Top.count();
                double[] rgBottomA = convert(Bottom.update_cpu_data());
                double[] rgBottomB = convert(BottomB.update_cpu_data());
                double[] rgBottomC = convert(BottomC.update_cpu_data());

                for (int i = 0; i < nCount; i++)
                {
                    double dfTop = rgTop[i];
                    double dfBottomA = rgBottomA[i];
                    double dfBottomB = rgBottomB[i];
                    double dfBottomC = rgBottomC[i];
                    double dfExpected = Math.Min(dfBottomA, Math.Min(dfBottomB, dfBottomC));

                    m_log.EXPECT_NEAR(dfTop, dfExpected, 1e-4);
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestMinGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            p.eltwise_param.operation = EltwiseParameter.EltwiseOp.MIN;
            EltwiseLayer<T> layer = new EltwiseLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-4);
                checker.CheckGradientEltwise(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
