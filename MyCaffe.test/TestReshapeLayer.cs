using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.fillers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestReshapeLayer
    {
        [TestMethod]
        public void TestFlattenOutputSizes()
        {
            ReshapeLayerTest test = new ReshapeLayerTest();

            try
            {
                foreach (IReshapeLayerTest t in test.Tests)
                {
                    t.TestFlattenOutputSizes();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFlattenValues()
        {
            ReshapeLayerTest test = new ReshapeLayerTest();

            try
            {
                foreach (IReshapeLayerTest t in test.Tests)
                {
                    t.TestFlattenValues();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCopyDimensions()
        {
            ReshapeLayerTest test = new ReshapeLayerTest();

            try
            {
                foreach (IReshapeLayerTest t in test.Tests)
                {
                    t.TestCopyDimensions();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestInferenceOfUnspecified()
        {
            ReshapeLayerTest test = new ReshapeLayerTest();

            try
            {
                foreach (IReshapeLayerTest t in test.Tests)
                {
                    t.TestInferenceOfUnspecified();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestInferenceOfUnspecifiedWithStartAxis()
        {
            ReshapeLayerTest test = new ReshapeLayerTest();

            try
            {
                foreach (IReshapeLayerTest t in test.Tests)
                {
                    t.TestInferenceOfUnspecifiedWithStartAxis();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestInsertSingletonAxisStart()
        {
            ReshapeLayerTest test = new ReshapeLayerTest();

            try
            {
                foreach (IReshapeLayerTest t in test.Tests)
                {
                    t.TestInsertSingletonAxisStart();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestInsertSingletonAxisMiddle()
        {
            ReshapeLayerTest test = new ReshapeLayerTest();

            try
            {
                foreach (IReshapeLayerTest t in test.Tests)
                {
                    t.TestInsertSingletonAxisMiddle();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestInsertSingletonAxisEnd()
        {
            ReshapeLayerTest test = new ReshapeLayerTest();

            try
            {
                foreach (IReshapeLayerTest t in test.Tests)
                {
                    t.TestInsertSingletonAxisEnd();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestFlattenMiddle()
        {
            ReshapeLayerTest test = new ReshapeLayerTest();

            try
            {
                foreach (IReshapeLayerTest t in test.Tests)
                {
                    t.TestFlattenMiddle();
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
            ReshapeLayerTest test = new ReshapeLayerTest();

            try
            {
                foreach (IReshapeLayerTest t in test.Tests)
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
        public void TestForwardAfterReshape()
        {
            ReshapeLayerTest test = new ReshapeLayerTest();

            try
            {
                foreach (IReshapeLayerTest t in test.Tests)
                {
                    t.TestForwardAfterReshape();
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
            ReshapeLayerTest test = new ReshapeLayerTest();

            try
            {
                foreach (IReshapeLayerTest t in test.Tests)
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


    interface IReshapeLayerTest : ITest
    {
        void TestFlattenOutputSizes();
        void TestFlattenValues();
        void TestCopyDimensions();
        void TestInferenceOfUnspecified();
        void TestInferenceOfUnspecifiedWithStartAxis();
        void TestInsertSingletonAxisStart();
        void TestInsertSingletonAxisMiddle();
        void TestInsertSingletonAxisEnd();
        void TestFlattenMiddle();
        void TestForward();
        void TestForwardAfterReshape();
        void TestGradient();
    }

    class ReshapeLayerTest : TestBase
    {
        public ReshapeLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Reshape Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ReshapeLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ReshapeLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ReshapeLayerTest<T> : TestEx<T>, IReshapeLayerTest
    {
        public ReshapeLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 6, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        public void TestFlattenOutputSizes()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RESHAPE);
            BlobShape blob_shape = p.reshape_param.shape;
            blob_shape.dim.Add(0);
            blob_shape.dim.Add(-1);
            blob_shape.dim.Add(1);
            blob_shape.dim.Add(1);

            ReshapeLayer<T> layer = new ReshapeLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(2, Top.num, "The top should have num = 2.");
                m_log.CHECK_EQ(3 * 6 * 5, Top.channels, "The top channels should equal 3 * 6 * 5 = " + (3 * 6 * 5).ToString() + ".");
                m_log.CHECK_EQ(1, Top.height, "The top height should equal 1.");
                m_log.CHECK_EQ(1, Top.width, "The top height should equal 1.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestFlattenValues()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RESHAPE);
            BlobShape blob_shape = p.reshape_param.shape;
            blob_shape.dim.Add(0);
            blob_shape.dim.Add(-1);
            blob_shape.dim.Add(1);
            blob_shape.dim.Add(1);

            ReshapeLayer<T> layer = new ReshapeLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                for (int c = 0; c < 3 * 6 * 5; c++)
                {
                    double dfTop0 = convert(Top.data_at(0, c, 0, 0));
                    double dfBtm0 = convert(Bottom.data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
                    m_log.CHECK_EQ(dfTop0, dfBtm0, "The top and bottom at 0 should be equal.");

                    double dfTop1 = convert(Top.data_at(1, c, 0, 0));
                    double dfBtm1 = convert(Bottom.data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
                    m_log.CHECK_EQ(dfTop0, dfBtm0, "The top and bottom at 1 should be equal.");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        // Test wether setting output dimensions to 0 either explicitly or implicitly
        // copies the respective dimension of the input layer.
        public void TestCopyDimensions()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RESHAPE);
            BlobShape blob_shape = p.reshape_param.shape;
            blob_shape.dim.Add(0);
            blob_shape.dim.Add(0);
            blob_shape.dim.Add(0);
            blob_shape.dim.Add(0);

            ReshapeLayer<T> layer = new ReshapeLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(2, Top.num, "The top should have num = 2.");
                m_log.CHECK_EQ(3, Top.channels, "The top channels should equal 3.");
                m_log.CHECK_EQ(6, Top.height, "The top height should equal 6.");
                m_log.CHECK_EQ(5, Top.width, "The top height should equal 7.");
            }
            finally
            {
                layer.Dispose();
            }
        }


        // When a dimension is set to -1, we should infer its value from the other
        // dimensions (including those that get copies from below).
        public void TestInferenceOfUnspecified()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RESHAPE);
            BlobShape blob_shape = p.reshape_param.shape;
            blob_shape.dim.Add(0);
            blob_shape.dim.Add(3);
            blob_shape.dim.Add(10);
            blob_shape.dim.Add(-1);

            // Count is 180, this height should be 180 / (2*3*10) = 3.

            ReshapeLayer<T> layer = new ReshapeLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(2, Top.num, "The top should have num = 2.");
                m_log.CHECK_EQ(3, Top.channels, "The top channels should equal 3.");
                m_log.CHECK_EQ(10, Top.height, "The top height should equal 10.");
                m_log.CHECK_EQ(3, Top.width, "The top height should equal 3.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestInferenceOfUnspecifiedWithStartAxis()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RESHAPE);
            p.reshape_param.axis = 1;
            BlobShape blob_shape = p.reshape_param.shape;
            blob_shape.dim.Add(3);
            blob_shape.dim.Add(10);
            blob_shape.dim.Add(-1);

            ReshapeLayer<T> layer = new ReshapeLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(4, Top.num_axes, "The top should have num_axes = 4.");
                m_log.CHECK_EQ(2, Top.num, "The top should have num = 2.");
                m_log.CHECK_EQ(3, Top.channels, "The top channels should equal 3.");
                m_log.CHECK_EQ(10, Top.height, "The top height should equal 10.");
                m_log.CHECK_EQ(3, Top.width, "The top height should equal 3.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestInsertSingletonAxisStart()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RESHAPE);
            p.reshape_param.axis = 0;
            p.reshape_param.num_axes = 0;
            BlobShape blob_shape = p.reshape_param.shape;
            blob_shape.dim.Add(1);
            blob_shape.dim.Add(1);
            blob_shape.dim.Add(1);

            ReshapeLayer<T> layer = new ReshapeLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(7, Top.num_axes, "The top should have num_axes = 7.");
                m_log.CHECK_EQ(1, Top.shape(0), "The top shape(0) should equal 1.");
                m_log.CHECK_EQ(1, Top.shape(1), "The top shape(1) should equal 1.");
                m_log.CHECK_EQ(1, Top.shape(2), "The top shape(2) should equal 1.");
                m_log.CHECK_EQ(2, Top.shape(3), "The top shape(3) should equal 2.");
                m_log.CHECK_EQ(3, Top.shape(4), "The top shape(4) should equal 3.");
                m_log.CHECK_EQ(6, Top.shape(5), "The top shape(5) should equal 6.");
                m_log.CHECK_EQ(5, Top.shape(6), "The top shape(6) should equal 5.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestInsertSingletonAxisMiddle()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RESHAPE);
            p.reshape_param.axis = 2;
            p.reshape_param.num_axes = 0;
            BlobShape blob_shape = p.reshape_param.shape;
            blob_shape.dim.Add(1);
            blob_shape.dim.Add(1);
            blob_shape.dim.Add(1);

            ReshapeLayer<T> layer = new ReshapeLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(7, Top.num_axes, "The top should have num_axes = 7.");
                m_log.CHECK_EQ(2, Top.shape(0), "The top shape(0) should equal 2.");
                m_log.CHECK_EQ(3, Top.shape(1), "The top shape(1) should equal 3.");
                m_log.CHECK_EQ(1, Top.shape(2), "The top shape(2) should equal 1.");
                m_log.CHECK_EQ(1, Top.shape(3), "The top shape(3) should equal 1.");
                m_log.CHECK_EQ(1, Top.shape(4), "The top shape(4) should equal 1.");
                m_log.CHECK_EQ(6, Top.shape(5), "The top shape(5) should equal 6.");
                m_log.CHECK_EQ(5, Top.shape(6), "The top shape(6) should equal 5.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestInsertSingletonAxisEnd()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RESHAPE);
            p.reshape_param.axis = -1;
            p.reshape_param.num_axes = 0;
            BlobShape blob_shape = p.reshape_param.shape;
            blob_shape.dim.Add(1);
            blob_shape.dim.Add(1);
            blob_shape.dim.Add(1);

            ReshapeLayer<T> layer = new ReshapeLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(7, Top.num_axes, "The top should have num_axes = 7.");
                m_log.CHECK_EQ(2, Top.shape(0), "The top shape(0) should equal 2.");
                m_log.CHECK_EQ(3, Top.shape(1), "The top shape(1) should equal 3.");
                m_log.CHECK_EQ(6, Top.shape(2), "The top shape(2) should equal 6.");
                m_log.CHECK_EQ(5, Top.shape(3), "The top shape(3) should equal 5.");
                m_log.CHECK_EQ(1, Top.shape(4), "The top shape(4) should equal 1.");
                m_log.CHECK_EQ(1, Top.shape(5), "The top shape(5) should equal 1.");
                m_log.CHECK_EQ(1, Top.shape(6), "The top shape(6) should equal 1.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestFlattenMiddle()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RESHAPE);
            p.reshape_param.axis = 1;
            p.reshape_param.num_axes = 2;
            BlobShape blob_shape = p.reshape_param.shape;
            blob_shape.dim.Add(-1);

            ReshapeLayer<T> layer = new ReshapeLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(3, Top.num_axes, "The top should have num_axes = 7.");
                m_log.CHECK_EQ(2, Top.shape(0), "The top shape(0) should equal 2.");
                m_log.CHECK_EQ(3 * 6, Top.shape(1), "The top shape(1) should equal 3 * 6 = 18.");
                m_log.CHECK_EQ(5, Top.shape(2), "The top shape(3) should equal 5.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RESHAPE);
            BlobShape blob_shape = p.reshape_param.shape;
            blob_shape.dim.Add(6);
            blob_shape.dim.Add(2);
            blob_shape.dim.Add(3);
            blob_shape.dim.Add(5);

            ReshapeLayer<T> layer = new ReshapeLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(Top.update_cpu_data());
                double[] rgBtm = convert(Bottom.update_cpu_data());

                for (int i = 0; i < Bottom.count(); i++)
                {
                    m_log.CHECK_EQ(rgTop[i], rgBtm[i], "The top and bottom values at " + i.ToString() + " should be equal.");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardAfterReshape()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RESHAPE);
            BlobShape blob_shape = p.reshape_param.shape;
            blob_shape.dim.Add(6);
            blob_shape.dim.Add(2);
            blob_shape.dim.Add(3);
            blob_shape.dim.Add(5);

            ReshapeLayer<T> layer = new ReshapeLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                // We know the above produced the corred result from TestForward.
                // Reshape the bottom and call layer.Reshape, then try again.

                List<int> rgNewBottomShape = new List<int>() { 2 * 3 * 6 * 5 };
                Bottom.Reshape(rgNewBottomShape);

                FillerParameter fp = new FillerParameter("gaussian");
                Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                filler.Fill(Bottom);

                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(Top.update_cpu_data());
                double[] rgBtm = convert(Bottom.update_cpu_data());

                for (int i = 0; i < Bottom.count(); i++)
                {
                    m_log.CHECK_EQ(rgTop[i], rgBtm[i], "The top and bottom values at " + i.ToString() + " should be equal.");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.RESHAPE);
            BlobShape blob_shape = p.reshape_param.shape;
            blob_shape.dim.Add(6);
            blob_shape.dim.Add(2);
            blob_shape.dim.Add(3);
            blob_shape.dim.Add(5);

            ReshapeLayer<T> layer = new ReshapeLayer<T>(m_cuda, m_log, p);

            try
            {
                GradientChecker<T> checker = new test.GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
                checker.CheckGradientEltwise(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
