using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestIm2ColLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            Im2ColLayerTest test = new Im2ColLayerTest();

            try
            {
                foreach (IIm2ColLayerTest t in test.Tests)
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
        public void TestForward()
        {
            Im2ColLayerTest test = new Im2ColLayerTest();

            try
            {
                foreach (IIm2ColLayerTest t in test.Tests)
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
            Im2ColLayerTest test = new Im2ColLayerTest();

            try
            {
                foreach (IIm2ColLayerTest t in test.Tests)
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
        public void TestDilatedGradient()
        {
            Im2ColLayerTest test = new Im2ColLayerTest();

            try
            {
                foreach (IIm2ColLayerTest t in test.Tests)
                {
                    t.TestDilatedGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientForceND()
        {
            Im2ColLayerTest test = new Im2ColLayerTest();

            try
            {
                foreach (IIm2ColLayerTest t in test.Tests)
                {
                    t.TestGradientForceND();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDilatedGradientForceND()
        {
            Im2ColLayerTest test = new Im2ColLayerTest();

            try
            {
                foreach (IIm2ColLayerTest t in test.Tests)
                {
                    t.TestDilatedGradientForceND();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRect()
        {
            Im2ColLayerTest test = new Im2ColLayerTest();

            try
            {
                foreach (IIm2ColLayerTest t in test.Tests)
                {
                    t.TestRect();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestRectGradient()
        {
            Im2ColLayerTest test = new Im2ColLayerTest();

            try
            {
                foreach (IIm2ColLayerTest t in test.Tests)
                {
                    t.TestRectGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    class Im2ColLayerTest : TestBase
    {
        public Im2ColLayerTest()
            : base("Im2Col Layer Test")
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            if (dt == common.DataType.DOUBLE)
                return new Im2ColLayerTest<double>(strName, nDeviceID);
            else
                return new Im2ColLayerTest<float>(strName, nDeviceID);
        }
    }

    interface IIm2ColLayerTest
    {
        void TestSetup();
        void TestForward();
        void TestGradient();
        void TestDilatedGradient();
        void TestGradientForceND();
        void TestDilatedGradientForceND();
        void TestRect();
        void TestRectGradient();
    }

    class Im2ColLayerTest<T> : TestEx<T>, IIm2ColLayerTest
    {
        public Im2ColLayerTest(string strName, int nDeviceID)
            : base(strName, new List<int>() { 2, 3, 6, 5 }, nDeviceID)
        {
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IM2COL);

            m_log.CHECK(p.convolution_param != null, "The convolution param should have been created.");
            List<int> rgBottomShape = new List<int>() { 2, 3, 10, 11 };

            Bottom.Reshape(rgBottomShape);

            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.dilation.Add(3);

            Im2colLayer<T> layer = new Im2colLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(2, Top.num, "The top should have num = 2.");
                m_log.CHECK_EQ(27, Top.channels, "The top should have channels = 27.");
                m_log.CHECK_EQ(2, Top.height, "The top should have height = 2.");
                m_log.CHECK_EQ(3, Top.width, "The top should have width = 3.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IM2COL);

            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);

            Im2colLayer<T> layer = new Im2colLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                T[] rgBottom = Bottom.update_cpu_data();
                T[] rgTop = Top.update_cpu_data();

                // We are lazy and will only check the top abnd left block.
                for (int c = 0; c < 27; c++)
                {
                    T fBottom = Bottom.data_at(0, (c / 9), (c / 3) % 3, c % 3);
                    double dfBottom = (double)Convert.ChangeType(fBottom, typeof(double));
                    T fTop = Top.data_at(0, c, 0, 0);
                    double dfTop = (double)Convert.ChangeType(fTop, typeof(double));

                    m_log.CHECK_EQ(dfBottom, dfTop, "The top and bottom at c = " + c.ToString() + " are not equal.");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IM2COL);

            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);

            Im2colLayer<T> layer = new Im2colLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestDilatedGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IM2COL);

            List<int> rgBottomShape = new List<int>() { 2, 3, 10, 9 };
            Bottom.Reshape(rgBottomShape);

            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.dilation.Add(3);

            Im2colLayer<T> layer = new Im2colLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradientForceND()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IM2COL);
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.force_nd_im2col = true;

            Im2colLayer<T> layer = new Im2colLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestDilatedGradientForceND()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IM2COL);
            p.convolution_param.kernel_size.Add(3);
            p.convolution_param.stride.Add(2);
            p.convolution_param.dilation.Add(3);
            p.convolution_param.force_nd_im2col = true;

            Im2colLayer<T> layer = new Im2colLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestRect()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IM2COL);
            p.convolution_param.kernel_h = 5;
            p.convolution_param.kernel_w = 3;
            p.convolution_param.stride.Add(2);

            Im2colLayer<T> layer = new Im2colLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);


                T[] rgBottom = Bottom.update_cpu_data();
                T[] rgTop = Top.update_cpu_data();

                // We are lazy and will only check the top abnd left block.
                for (int c = 0; c < 45; c++)
                {
                    T fBottom = Bottom.data_at(0, (c / 15), (c / 3) % 5, c % 3);
                    double dfBottom = (double)Convert.ChangeType(fBottom, typeof(double));
                    T fTop = Top.data_at(0, c, 0, 0);
                    double dfTop = (double)Convert.ChangeType(fTop, typeof(double));

                    m_log.CHECK_EQ(dfBottom, dfTop, "The top and bottom at c = " + c.ToString() + " are not equal.");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestRectGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IM2COL);
            p.convolution_param.kernel_h = 5;
            p.convolution_param.kernel_w = 3;
            p.convolution_param.stride.Add(2);

            Im2colLayer<T> layer = new Im2colLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
