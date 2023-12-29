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
using MyCaffe.layers.nt;

/// <summary>
/// Testing the SharpeLoss layer.
/// 
/// SharpeLoss Layer - layer calculates the SharpeLoss
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestSharpeLossLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            SharpeLossLayerTest test = new SharpeLossLayerTest();

            try
            {
                foreach (ISharpeLossLayerTest t in test.Tests)
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
            SharpeLossLayerTest test = new SharpeLossLayerTest();

            try
            {
                foreach (ISharpeLossLayerTest t in test.Tests)
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
            SharpeLossLayerTest test = new SharpeLossLayerTest();

            try
            {
                foreach (ISharpeLossLayerTest t in test.Tests)
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

    interface ISharpeLossLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
        void TestGradient();
    }

    class SharpeLossLayerTest : TestBase
    {
        public SharpeLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Gram Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SharpeLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SharpeLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SharpeLossLayerTest<T> : TestEx<T>, ISharpeLossLayerTest
    {
        Blob<T> m_blob_bottom2;

        public SharpeLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
            m_blob_bottom2 = new Blob<T>(m_cuda, m_log, 2, 3, 4, 5);   
        }

        protected override void dispose()
        {
            dispose(ref m_blob_bottom2);
            base.dispose();
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SHARPE_LOSS);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as SharpeLossLayer<T>;

            try
            {
                if (!(layer is SharpeLossLayer<T>))
                    m_log.FAIL("The layer is not a SharpeLossLayer!");

                float[] rgBtmTrue = new float[]
                {
                    0.3399f,
                    0.9907f,
                    0.7453f,
                    0.0616f,
                    0.7079f,
                    1.3399f,
                    1.9907f,
                    1.7453f,
                    1.0616f,
                    1.7079f
                };
                float[] rgBtmPred = new float[]
                {
                    0.2f,
                    0.2f,
                    0.2f,
                    0.3f,
                    0.3f,
                    0.5f,
                    0.5f,
                    0.5f,
                    0.5f,
                    0.5f
                };

                m_blob_bottom.Reshape(2, 5, 1, 1);
                m_blob_bottom2.ReshapeLike(m_blob_bottom);
                m_blob_bottom.mutable_cpu_data = convert(rgBtmPred);
                m_blob_bottom2.mutable_cpu_data = convert(rgBtmTrue);

                BottomVec.Clear();
                BottomVec.Add(m_blob_bottom);
                BottomVec.Add(m_blob_bottom2);

                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_top.shape().Count, 1, "The top should have 1 items in its shape.");
                m_log.CHECK_EQ(m_blob_top.count(), 1, "The top should have 1 item in its count.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SHARPE_LOSS);
            SharpeLossLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as SharpeLossLayer<T>;

            try
            {
                float[] rgBtmTrue = new float[] 
                { 
                    0.3399f,
                    0.9907f,
                    0.7453f,
                    0.0616f,
                    0.7079f,
                    1.3399f,
                    1.9907f,
                    1.7453f,
                    1.0616f,
                    1.7079f
                };
                float[] rgBtmPred = new float[]
                {
                    0.2f,
                    0.2f,
                    0.2f,
                    0.3f,
                    0.3f,
                    0.5f,
                    0.5f,
                    0.5f,
                    0.5f,
                    0.5f
                };

                m_blob_bottom.Reshape(2, 5, 1, 1);
                m_blob_bottom2.ReshapeLike(m_blob_bottom);
                m_blob_bottom.mutable_cpu_data = convert(rgBtmPred);
                m_blob_bottom2.mutable_cpu_data = convert(rgBtmTrue);

                BottomVec.Clear();
                BottomVec.Add(m_blob_bottom);
                BottomVec.Add(m_blob_bottom2);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(m_blob_top.update_cpu_data());
                float fExpected = -1.2994f;
                float fActual = (float)rgTop[0];

                m_log.EXPECT_EQUAL<float>(fActual, fExpected, "The loss is not as expected.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        /// <summary>
        /// Test the gradient.
        /// </summary>
        /// <remarks>
        /// Direct gradient calculation in python:
        ///    def custom_loss1_grad(y_true, weights, grad_y):
        ///        # Loss calculation
        ///        captured_returns = weights* y_true
        ///        mean_returns = torch.mean(captured_returns)
        ///        mean_returns_sq = torch.square(mean_returns)
        ///        captured_returns_sq = torch.square(captured_returns)
        ///        mean_captured_returns_sq = torch.mean(captured_returns_sq)
        ///        mean_captured_returns_sq_minus_mean_returns_sq = mean_captured_returns_sq - mean_returns_sq # + 1e-9
        ///        mean_captured_returns_sq_minus_mean_returns_sqrt = torch.sqrt(mean_captured_returns_sq_minus_mean_returns_sq)
        ///
        ///        loss1 = (mean_returns / mean_captured_returns_sq_minus_mean_returns_sqrt) # * torch.sqrt(twofiftytwo))
        ///        loss = loss1* -1
        ///
        ///        # Gradient calculation
        ///        loss1_grad = grad_y* -1
        ///        mean_captured_returns_sq_minus_mean_returns_sqrt_grad = -1 * mean_returns / mean_captured_returns_sq_minus_mean_returns_sqrt**2 * loss1_grad
        ///        mean_captured_returns_sq_minus_mean_returns_sq_grad = 0.5 * mean_captured_returns_sq_minus_mean_returns_sq * *-0.5 * mean_captured_returns_sq_minus_mean_returns_sqrt_grad
        ///        mean_captured_returns_sq_grad = mean_captured_returns_sq_minus_mean_returns_sq_grad # + 1
        ///        captured_returns_sq_grad = torch.ones_like(captured_returns) / captured_returns.numel() * mean_captured_returns_sq_grad
        ///        mean_returns_sq_grad = -1 * mean_captured_returns_sq_grad
        ///        mean_returns_grad = (2 * mean_returns * mean_returns_sq_grad) + (1 / mean_captured_returns_sq_minus_mean_returns_sqrt * loss1_grad)
        ///
        ///        captured_returns_grad_1 = mean_returns_grad / captured_returns.numel()
        ///        captured_returns_grad_2 = 2 * captured_returns * captured_returns_sq_grad
        ///        captured_returns_grad = captured_returns_grad_1 + captured_returns_grad_2
        ///
        ///        weights_grad_1 = y_true * captured_returns_grad_1
        ///        weights_grad_2 = y_true* captured_returns_grad_2
        ///        weights_grad = weights_grad_1 + weights_grad_2
        ///             
        ///        return weights_grad
        /// </remarks>
        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SHARPE_LOSS);
            SharpeLossLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as SharpeLossLayer<T>;

            try
            {
                float[] rgBtmTrue = new float[]
                {
                    0.3399f,
                    0.9907f,
                    0.7453f,
                    0.0616f,
                    0.7079f,
                    1.3399f,
                    1.9907f,
                    1.7453f,
                    1.0616f,
                    1.7079f
                };
                float[] rgBtmPred = new float[]
                {
                    0.2f,
                    0.2f,
                    0.2f,
                    0.3f,
                    0.3f,
                    0.5f,
                    0.5f,
                    0.5f,
                    0.5f,
                    0.5f
                };
                float[] rgExpectedGrad = new float[]
                {
                    -0.23560524f,
                    -0.55118006f,
                    -0.45309797f,
                    -0.04590359f,
                    -0.38325533f,
                    -0.08100017f,
                     0.560508f,
                     0.26632923f,
                    -0.21944097f,
                     0.22705352f
                };

                m_blob_bottom.Reshape(2, 5, 1, 1);
                m_blob_bottom2.ReshapeLike(m_blob_bottom);
                m_blob_bottom.mutable_cpu_data = convert(rgBtmPred);
                m_blob_bottom2.mutable_cpu_data = convert(rgBtmTrue);

                BottomVec.Clear();
                BottomVec.Add(m_blob_bottom);
                BottomVec.Add(m_blob_bottom2);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(m_blob_top.update_cpu_data());
                float fExpected = -1.2994f;
                float fActual = (float)rgTop[0];

                m_log.EXPECT_EQUAL<float>(fActual, fExpected, "The loss is not as expected.");

                m_blob_top.SetDiff(1.0);
                layer.Backward(TopVec, new List<bool>() { true, false }, BottomVec);

                double[] rgBtmGrad = convert(m_blob_bottom.update_cpu_diff());

                for (int i=0; i<rgBtmGrad.Length; i++)
                {
                    m_log.EXPECT_EQUAL<float>(rgBtmGrad[i], rgExpectedGrad[i], "The gradient is not as expected.");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestGradient2()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRAM);
            SharpeLossLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as SharpeLossLayer<T>;

            try
            { 
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
