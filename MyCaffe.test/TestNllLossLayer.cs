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
    public class TestNLLLossLayer
    {
        [TestMethod]
        public void TestForwardBackward()
        {
            NLLLossLayerTest test = new NLLLossLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INLLLossLayerTest t in test.Tests)
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

    interface INLLLossLayerTest : ITest
    {
        void TestForwardBackward();
    }

    class NLLLossLayerTest : TestBase
    {
        public NLLLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("NLLLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new NLLLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new NLLLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class NLLLossLayerTest<T> : TestEx<T>, INLLLossLayerTest
    {
        public NLLLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
        /// Forward/backward test for NLLLossLayer using data generated with Python script.
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
        /// debug = DebugFunction.apply        
        /// DebugFunction.trace(input, "input")
        /// input = debug(input);
        /// 
        /// sm1 = softmaxEx.log_softmax(input)
        /// 
        /// DebugFunction.trace(sm1, "sm1")
        /// sm1 = debug(sm1);
        /// DebugFunction.trace(target, "target")
        /// target = debug(target);
        /// 
        /// output = loss(sm1, target)
        /// 
        /// DebugFunction.trace(output, "output")
        /// output = debug(output);
        /// 
        /// output.backward()
        /// </code>
        /// </remarks>
        public void TestForwardBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NLL_LOSS);
            p.nll_loss_param.axis = 2;
            NLLLossLayer<T> layer = new NLLLossLayer<T>(m_cuda, m_log, p);
            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            Blob<T> blobTarget = new Blob<T>(m_cuda, m_log);
            Blob<T> blobLoss = new Blob<T>(m_cuda, m_log);
            Blob<T> blobLossExp = new Blob<T>(m_cuda, m_log);
            
            try
            {
                // data from 'sm1.npy'
                float[] rgX = new float[]
                {
                   -1.53775132f, -1.28591919f,  -2.639563f,  -1.43150032f, -1.617484f,   
                   -1.34292f,    -1.7933352f,   -3.20892358f,-0.7393775f,  -2.90584755f, 
                   -2.381441f,   -2.35734034f,  -2.970993f,  -0.368087173f,-2.66496563f 
                };
                // data from 'grad_sm1.npy'
                float[] rgYgradExp = new float[]
                {
                    0f,          -0.333333343f,  0f,          0f,           0f,
                   -0.333333343f, 0f,            0f,          0f,           0f,
                    0f,           0f,            0f,          0f,          -0.333333343f
                };
                // data from 'target.npy'
                float[] rgTarget = new float[]
                {
                    1, 0, 4
                };

                blobX.Reshape(1, 3, 5, 1);
                blobX.mutable_cpu_data = convert(rgX);
                blobTarget.Reshape(1, 3, 1, 1);
                blobTarget.mutable_cpu_data = convert(rgTarget);
                blobLossExp.Reshape(1, 1, 1, 1);
                blobLossExp.SetData(1.76460159);
                
                BottomVec.Clear();
                BottomVec.Add(blobX);
                BottomVec.Add(blobTarget);
                TopVec.Clear();
                TopVec.Add(blobLoss);

                layer.Setup(BottomVec, TopVec);
                m_log.CHECK_EQ(blobLoss.count(), 1, "The shapes should be equal!");

                layer.Forward(BottomVec, TopVec);
                float[] rgYexp1 = convertF(blobLossExp.mutable_cpu_data);
                float[] rgYactual = convertF(blobLoss.mutable_cpu_data);
                for (int i = 0; i < rgYactual.Length; i++)
                {
                    float fActual = rgYactual[i];
                    float fExpected = rgYexp1[i];
                    float fDiff = Math.Abs(fActual - fExpected);
                    float fErr = 1e-08f;

                    if (fDiff > fErr)
                        m_log.FAIL("The error exeeds the expected value at i = " + i.ToString());
                }
                
                blobLoss.SetDiff(1);
                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                float[] rgXgradActual = convertF(blobX.mutable_cpu_diff);
                blobLossExp.mutable_cpu_diff = convert(rgYgradExp);
                float[] rgXgradExp1 = convertF(blobLossExp.mutable_cpu_diff);
                for (int i = 0; i < rgXgradActual.Length; i++)
                {
                    float fActual = rgXgradActual[i];
                    float fExpected = rgXgradExp1[i];
                    float fDiff = Math.Abs(fActual - fExpected);
                    float fErr = 1e-08f;

                    if (fDiff > fErr)
                        m_log.FAIL("The error exeeds the expected value at i = " + i.ToString());
                }
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobTarget);
                dispose(ref blobLoss);
                dispose(ref blobLossExp);
                layer.Dispose();
            }
        }
    }
}
