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
    }

    interface ILogSoftmaxLayerTest : ITest
    {
        void TestForwardBackward();
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
        /// class LogSoftMaxEx():
        ///     def log_softmax(self, x):
        ///         debug = DebugFunction.apply        
        ///         DebugFunction.trace(x, "x")
        ///         x = debug(x)
        ///         
        ///         c = torch.max(x, dim=-1)[0]
        ///         c = torch.reshape(c, (-1, 1))
        ///         DebugFunction.trace(c, "c")
        ///         c = debug(c)
        /// 
        ///         xm = x - c
        ///         DebugFunction.trace(xm, "xm")
        ///         xm = debug(xm)
        /// 
        ///         exp_x = torch.exp(xm)        
        ///         DebugFunction.trace(exp_x, "exp_x")
        ///         exp_x = debug(exp_x)
        ///         
        ///         exp_sum = exp_x.sum(axis=-1)
        ///         exp_sum = torch.reshape(exp_sum, (-1, 1))
        ///         DebugFunction.trace(exp_sum, "exp_sum")
        ///         exp_sum = debug(exp_sum)
        ///         
        ///         exp_log = torch.log(exp_sum)
        ///         DebugFunction.trace(exp_log, "exp_log")
        ///         exp_log = debug(exp_log)
        ///         
        ///         sm = xm - exp_log
        ///         DebugFunction.trace(sm, "sm")
        ///         sm = debug(sm)
        ///         return sm
        ///     
        /// torch.manual_seed(1701)
        /// input = torch.randn(3, 5, requires_grad=True)
        /// nllgrad = torch.zeros(3, 5)
        /// target = torch.tensor([1, 0, 4])
        /// loss = nn.NLLLoss()
        /// softmaxEx = LogSoftMaxEx()
        /// 
        /// sm = softmaxEx.log_softmax(input)
        /// 
        /// output = loss(sm, target)
        /// output.backward()
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
                    0.4499141f,   0.7017462f,   -0.6518978f,  0.5561651f,   0.3701814f,   
                    1.27394927f,  0.823534f,    -0.5920543f,  1.87749171f, -0.288978338f,    
                   -0.41542238f, -0.391321629f, -1.00497425f, 1.59793139f, -0.698946834f    
                };
                // data from 'sm.npy'
                float[] rgYexp = new float[]
                {
                   -1.53775132f, -1.28591919f,  -2.639563f,  -1.43150032f, -1.617484f,   
                   -1.34292f,    -1.7933352f,   -3.20892358f,-0.7393775f,  -2.90584755f, 
                   -2.381441f,   -2.35734034f,  -2.970993f,  -0.368087173f,-2.66496563f 
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
                    0.07162125f,  -0.2412012f,    0.0237974841f, 0.07965006f, 0.06613242f,  
                   -0.246305943f,  0.0554680862f, 0.0134666925f, 0.15913704f, 0.0182341356f,
                    0.0308057684f, 0.0315572321f, 0.0170841329f, 0.2306856f, -0.310132742f
                };

                blobX.Reshape(3, 5, 1, 1);
                blobX.mutable_cpu_data = convert(rgX);
                blobYexp.ReshapeLike(blobX);
                blobYexp.mutable_cpu_data = convert(rgYexp);
                //string strPath = "...\\PythonApplication4\\PythonApplication4\\test\\";
                // blobX.LoadFromNumpy(strPath + "x.npy");
                // blobYexp.LoadFromNumpy(strPath + "sm.npy");
                
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
                    float fErr = (typeof(T) == typeof(float)) ? 1e-08f : 2.5e-07f;

                    if (fDiff > fErr)
                        m_log.FAIL("The error exeeds the expected value at i = " + i.ToString());
                }
                
                blobY.mutable_cpu_diff = convert(rgYgrad);
                //blobY.LoadFromNumpy(strPath + "grad_sm.npy", true);
                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                float[] rgXgradActual = convertF(blobX.mutable_cpu_diff);
                //blobYexp.LoadFromNumpy(strPath + "grad_x.npy", true);
                blobYexp.mutable_cpu_diff = convert(rgXgradExp);
                float[] rgXgradExp1 = convertF(blobYexp.mutable_cpu_diff);
                for (int i = 0; i < rgXgradActual.Length; i++)
                {
                    float fActual = rgXgradActual[i];
                    float fExpected = rgXgradExp1[i];
                    float fDiff = Math.Abs(fActual - fExpected);
                    float fErr = 3e-08f;

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
    }
}
