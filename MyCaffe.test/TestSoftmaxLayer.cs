using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.common;
using MyCaffe.layers.gpt;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSoftmaxLayer
    {
        [TestMethod]
        public void TestForward()
        {
            SoftmaxLayerTest test = new SoftmaxLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ISoftmaxLayerTest t in test.Tests)
                {
                    t.TestForward(SOFTMAX_ALGORITHM.DEFAULT);
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
            SoftmaxLayerTest test = new SoftmaxLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ISoftmaxLayerTest t in test.Tests)
                {
                    t.TestGradient(SOFTMAX_ALGORITHM.DEFAULT);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardLog()
        {
            SoftmaxLayerTest test = new SoftmaxLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ISoftmaxLayerTest t in test.Tests)
                {
                    t.TestForward(SOFTMAX_ALGORITHM.LOG);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientLog()
        {
            SoftmaxLayerTest test = new SoftmaxLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ISoftmaxLayerTest t in test.Tests)
                {
                    t.TestGradient(SOFTMAX_ALGORITHM.LOG);
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestForwardCuDnn()
        {
            SoftmaxLayerTest test = new SoftmaxLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ISoftmaxLayerTest t in test.Tests)
                {
                    t.TestForward(SOFTMAX_ALGORITHM.DEFAULT);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientCuDnn()
        {
            SoftmaxLayerTest test = new SoftmaxLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ISoftmaxLayerTest t in test.Tests)
                {
                    t.TestGradient(SOFTMAX_ALGORITHM.DEFAULT);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardCuDnnLog()
        {
            SoftmaxLayerTest test = new SoftmaxLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ISoftmaxLayerTest t in test.Tests)
                {
                    t.TestForward(SOFTMAX_ALGORITHM.LOG);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientCuDnnLog()
        {
            SoftmaxLayerTest test = new SoftmaxLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ISoftmaxLayerTest t in test.Tests)
                {
                    t.TestGradient(SOFTMAX_ALGORITHM.LOG);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ISoftmaxLayerTest : ITest
    {
        void TestForward(SOFTMAX_ALGORITHM alg);
        void TestGradient(SOFTMAX_ALGORITHM alg);
    }

    class SoftmaxLayerTest : TestBase
    {
        public SoftmaxLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Softmax Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SoftmaxLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SoftmaxLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SoftmaxLayerTest<T> : TestEx<T>, ISoftmaxLayerTest
    {
        public SoftmaxLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
            FillerParameter p = new FillerParameter("uniform");
            return p;
        }

        public void TestForward(SOFTMAX_ALGORITHM alg)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            p.softmax_param.engine = m_engine;
            p.softmax_param.algorithm = alg;
            SoftmaxLayer<T> layer = new SoftmaxLayer<T>(m_cuda, m_log, p);

            try
            {
                if (alg == SOFTMAX_ALGORITHM.LOG)
                    test_fwd_log(layer, 1e-04);
                else
                    test_fwd_default(layer);
            }
            finally
            {
                layer.Dispose();
            }
        }

        /// <summary>
        /// Forward/backward test for LogSoftmaxLayer using data generated with Python script.
        /// </summary>
        /// <remarks>
        /// Code for generating the test data:
        /// <code>
        /// from torch import nn
        /// import os
        /// import torch
        /// import numpy as np
        /// 
        /// input_dict = { None : "" }
        /// 
        /// class DebugFunction(torch.autograd.Function):
        ///     out_path = "test/"
        /// 
        ///     @staticmethod
        ///     def set_output_path(i):
        ///         DebugFunction.out_path = "test/iter_%d/" % i
        ///         if not os.path.exists(DebugFunction.out_path):
        ///             os.makedirs(DebugFunction.out_path)
        /// 
        ///     @staticmethod
        ///     def trace(t, name):
        ///         if not os.path.exists(DebugFunction.out_path):
        ///             os.makedirs(DebugFunction.out_path)
        ///         input_dict.update({t: name})
        ///         np.save(DebugFunction.out_path + name + ".npy", t.detach().cpu().numpy())
        /// 
        ///     @staticmethod
        ///     def forward(ctx, input):
        ///         ctx.save_for_backward(input)       
        ///         return input
        /// 
        ///     @staticmethod
        ///     def backward(ctx, grad_output):
        ///         input, = ctx.saved_tensors
        ///         name = input_dict.get(input)
        ///         
        ///         if name == None:
        ///             name = "unknown";
        /// 
        ///         np.save(DebugFunction.out_path + "grad_" + name, grad_output.detach().cpu().numpy())
        ///         return grad_output
        /// 
        /// np.set_printoptions(precision=12, suppress=False)
        /// 
        /// class LogSoftmaxFunction(torch.autograd.Function):
        ///     @staticmethod
        ///     def forward(ctx, x):
        ///         m = x.max(axis=1, keepdims=True)[0]
        ///         xmu = x - m
        ///         expx = torch.exp(xmu)
        ///         sumexp = torch.sum(expx, axis=1, keepdims=True)   
        ///         log_z = m + torch.log(sumexp)
        ///         y = x - log_z
        ///         ctx.save_for_backward(y)
        ///         print(y.detach().cpu().numpy())
        ///         return y
        ///     
        ///     @staticmethod
        ///     def backward(ctx, grad_output):
        ///         gy = grad_output
        ///         y, = ctx.saved_tensors
        ///         sumgy = gy.sum(axis=1, keepdims=True)
        ///         expy = torch.exp(y)
        ///         grad = gy - expy * sumgy
        ///         print("grad -> input")
        ///         print(grad.detach().cpu().numpy())
        ///         return grad
        /// 
        /// class LogSoftMaxEx(nn.Module):
        ///     def __init__(self, axis):
        ///         super().__init__()
        ///         self.axis = axis
        ///         
        ///     def forward(self, x):
        ///         softmax = LogSoftmaxFunction.apply
        ///         return softmax(x)
        ///     
        /// torch.manual_seed(1701)
        /// input = torch.randn(3, 5, requires_grad=True)
        /// print("input")
        /// print(input.detach().cpu().numpy())
        /// 
        /// debug = DebugFunction.apply
        /// 
        /// nllgrad = torch.zeros(3, 5)
        /// target = torch.tensor([1, 0, 4])
        /// print("target")
        /// print(target.detach().cpu().numpy())
        /// loss = nn.NLLLoss()
        /// 
        /// softmaxEx = LogSoftMaxEx(axis=-1)
        /// print("sm")
        /// sm = softmaxEx(input)
        /// 
        /// DebugFunction.trace(sm, "sm")
        /// sm = debug(sm)
        /// 
        /// output = loss(sm, target)
        /// output.backward()
        /// 
        /// softmaxPy = nn.LogSoftmax(dim=-1)
        /// 
        /// DebugFunction.trace(input, "input")
        /// input = debug(input)
        /// 
        /// smpy = softmaxPy(input)
        /// print("smpy")
        /// print(smpy.detach().cpu().numpy())
        /// output = loss(smpy, target)
        /// print("smpy grad")
        /// output.backward()
        /// 
        /// print("done!")
        /// </code>
        /// </remarks>
        private void test_fwd_log(Layer<T> layer, double dfErr)
        {
            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);
            Blob<T> blobYexp = new Blob<T>(m_cuda, m_log);

            try
            {
                // data from 'x.npy'
                float[] rgX = new float[]
                {
                    0.4499141f,   0.7017462f,  -0.6518978f,   0.5561651f,  0.3701814f,
                    1.2739493f,   0.823534f,   -0.5920543f,   1.8774917f, -0.28897834f,
                   -0.41542238f, -0.39132163f, -1.0049742f,   1.5979314f, -0.69894683f
                };
                // data from 'sm.npy'
                float[] rgYexp = new float[]
                {
                   -1.5377513f,  -1.2859192f,   -2.639563f,   -1.4315003f,  -1.617484f,
                   -1.34292f,    -1.7933352f,   -3.2089236f,  -0.7393775f,  -2.9058475f,
                   -2.3814409f,  -2.35734f,     -2.9709928f,  -0.36808717f, -2.6649654f
                };
                // data from 'grad_sm.npy'
                float[] rgYgrad = new float[]
                {
                    0f,          -0.333333343f,  0f,          0f,           0f,
                   -0.333333343f, 0f,            0f,          0f,           0f,
                    0f,           0f,            0f,          0f,          -0.333333343f
                };
                // data from 'grad_x.npy'
                float[] rgXgradExp = new float[]
                {
                    0.07162124f, -0.2412012f,    0.023797486f,   0.07965005f,  0.06613242f,
                   -0.24630594f,  0.055468082f,  0.0134666925f,  0.15913701f,  0.018234137f,
                    0.030805774f, 0.031557236f,  0.017084135f,   0.23068562f, -0.31013274f
                };

                blobX.Reshape(3, 5, 1, 1);
                blobX.mutable_cpu_data = convert(rgX);
                blobYexp.ReshapeLike(blobX);
                blobYexp.mutable_cpu_data = convert(rgYexp);
                blobYexp.mutable_cpu_diff = convert(rgXgradExp);

                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);
                m_log.CHECK(blobX.CompareShape(blobY.shape()), "The shapes should be equal!");

                layer.Forward(BottomVec, TopVec);
                verify(blobY, blobYexp, false, dfErr);

                blobY.mutable_cpu_diff = convert(rgYgrad);
                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);
                verify(blobX, blobYexp, true, dfErr);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);
                dispose(ref blobYexp);
                layer.Dispose();
            }
        }

        private void test_fwd_default(Layer<T> layer)
        {
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Test sum
            for (int i = 0; i < Bottom.num; i++)
            {
                for (int k = 0; k < Bottom.height; k++)
                {
                    for (int l = 0; l < Bottom.width; l++)
                    {
                        double dfSum = 0;

                        for (int j = 0; j < Top.channels; j++)
                        {
                            dfSum += convert(Top.data_at(i, j, k, l));
                        }

                        m_log.CHECK_GE(dfSum, 0.999, "The sum should be greater than equal to 0.999");
                        m_log.CHECK_LE(dfSum, 1.001, "The sum should be less than or equal to 1.001");

                        // Test exact values
                        double dfScale = 0;

                        for (int j = 0; j < Bottom.channels; j++)
                        {
                            dfScale += Math.Exp(convert(Bottom.data_at(i, j, k, l)));
                        }

                        for (int j = 0; j < Bottom.channels; j++)
                        {
                            double dfTop = convert(Top.data_at(i, j, k, l));
                            double dfBottom = convert(Bottom.data_at(i, j, k, l));

                            m_log.CHECK_GE(dfTop + 1e-4, Math.Exp(dfBottom) / dfScale, "The value is out of range at " + i.ToString() + ", " + j.ToString());
                            m_log.CHECK_LE(dfTop - 1e-4, Math.Exp(dfBottom) / dfScale, "The value is out of range at " + i.ToString() + ", " + j.ToString());
                        }
                    }
                }
            }
        }

        public void TestGradient(SOFTMAX_ALGORITHM alg)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            p.softmax_param.engine = m_engine;
            p.softmax_param.algorithm = alg;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                m_log.CHECK(layer.layer_param.type == LayerParameter.LayerType.SOFTMAX, "The layer type should be SOFTMAX!");
                GradientChecker <T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
