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
using System.Security.Cryptography;

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
                float[] rgBtm = new float[] 
                { 
                    0.3399f,
                    0.9907f,
                    0.7453f,
                    0.0616f,
                    0.7079f
                };

                m_blob_bottom.Reshape(1, 5, 1, 1);
                m_blob_bottom2.ReshapeLike(m_blob_bottom);
                m_blob_bottom.mutable_cpu_data = convert(rgBtm);
                m_blob_bottom2.SetData(1.0);

                BottomVec.Clear();
                BottomVec.Add(m_blob_bottom);
                BottomVec.Add(m_blob_bottom2);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(m_blob_top.update_cpu_data());
                float fExpected = -1.5515f;
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
        /// Python code to duplicate the test.
        /// 
        /// x = np.array([0.3399, 0.9907, 0.7453, 0.0616, 0.7079])
        /// x = torch.from_numpy(x)
        /// x.requires_grad = True
        /// 
        /// y = -torch.mean(x)/ torch.std(x)
        /// print("y", y)
        /// y.backward()
        /// print("x", x)
        /// print("y loss grad", x.grad)
        /// 
        /// Output: y loss grad tensor([-1.2060,  0.6703, -0.0372, -2.0084, -0.1450], dtype=torch.float64)
        /// 
        /// Direct gradient calculation in python:
        /// def neg_mean_div_std_grad(x, mean, std, grad_output):
        ///    n = x.numel()
        ///    mean_grad = np.full(n, grad_output) / n
        ///    mean_grad = torch.from_numpy(mean_grad)
        ///    std_grad = (2.0 / (n - 1)) * grad_output / (std * 2) * (x - mean)
        ///    grad = ((mean_grad* std) - (mean* std_grad)) / std**2
        ///    return -1 * grad        
        /// </remarks>
        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SHARPE_LOSS);
            SharpeLossLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as SharpeLossLayer<T>;

            try
            {
                float[] rgBtm = new float[]
                {
                    0.3399f,
                    0.9907f,
                    0.7453f,
                    0.0616f,
                    0.7079f
                };

                float[] rgExpectedGrad = new float[]
                {
                    -1.2060f,
                     0.6703f,
                    -0.0372f,
                    -2.0084f,
                    -0.1450f
                };

                m_blob_bottom.Reshape(1, 5, 1, 1);
                m_blob_bottom2.ReshapeLike(m_blob_bottom);
                m_blob_bottom.mutable_cpu_data = convert(rgBtm);
                m_blob_bottom2.SetData(1.0);

                BottomVec.Clear();
                BottomVec.Add(m_blob_bottom);
                BottomVec.Add(m_blob_bottom2);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double[] rgTop = convert(m_blob_top.update_cpu_data());
                float fExpected = -1.5515f;
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
