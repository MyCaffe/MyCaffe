using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.layers.gpt;
using MyCaffe.common;

namespace MyCaffe.test
{
    [TestClass]
    public class TestLogSoftmaxLayer
    {
        [TestMethod]
        public void TestForwardBackward()
        {
            LogSoftmaxLayerTest test = new LogSoftmaxLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILogSoftmaxLayerTest t in test.Tests)
                {
                    t.TestForwardBackward();
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
            LogSoftmaxLayerTest test = new LogSoftmaxLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILogSoftmaxLayerTest t in test.Tests)
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

    interface ILogSoftmaxLayerTest : ITest
    {
        void TestForwardBackward();
        void TestGradient();
    }

    class LogSoftmaxLayerTest : TestBase
    {
        public LogSoftmaxLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("LogSoftmax Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new LogSoftmaxLayerTest<double>(strName, nDeviceID, engine);
            else
                return new LogSoftmaxLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class LogSoftmaxLayerTest<T> : TestEx<T>, ILogSoftmaxLayerTest
    {
        public LogSoftmaxLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
        public void TestForwardBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LOG_SOFTMAX);
            p.log_softmax_param.axis = 1;
            LogSoftmaxLayer<T> layer = new LogSoftmaxLayer<T>(m_cuda, m_log, p);
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

                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);
                m_log.CHECK(blobX.CompareShape(blobY.shape()), "The shapes should be equal!");

                layer.Forward(BottomVec, TopVec);
                float[] rgYexp1 = convertF(blobYexp.mutable_cpu_data);
                float[] rgYactual = convertF(blobY.mutable_cpu_data);
                for (int i = 0; i < rgYactual.Length; i++)
                {
                    float fActual = rgYactual[i];
                    float fExpected = rgYexp1[i];
                    float fDiff = Math.Abs(fActual - fExpected);
                    float fErr = (typeof(T) == typeof(float)) ? 1e-12f : 6e-08f;

                    if (fDiff > fErr)
                        m_log.FAIL("The error exeeds the expected value at i = " + i.ToString());
                }

                blobY.mutable_cpu_diff = convert(rgYgrad);
                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                float[] rgXgradActual = convertF(blobX.mutable_cpu_diff);
                blobYexp.mutable_cpu_diff = convert(rgXgradExp);
                float[] rgXgradExp1 = convertF(blobYexp.mutable_cpu_diff);
                for (int i = 0; i < rgXgradActual.Length; i++)
                {
                    float fActual = rgXgradActual[i];
                    float fExpected = rgXgradExp1[i];
                    float fDiff = Math.Abs(fActual - fExpected);
                    float fErr = 2e-08f;

                    if (fDiff > fErr)
                        m_log.FAIL("The error exeeds the expected value at i = " + i.ToString());
                }
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);
                dispose(ref blobYexp);
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LOG_SOFTMAX);
            p.log_softmax_param.axis = -1;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                m_log.CHECK(layer.layer_param.type == LayerParameter.LayerType.LOG_SOFTMAX, "The layer type should be LOG_SOFTMAX!");
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
