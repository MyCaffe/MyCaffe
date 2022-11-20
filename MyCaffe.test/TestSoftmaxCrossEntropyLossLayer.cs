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
using static MyCaffe.param.beta.DecodeParameter;
using System.Security.Cryptography;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSoftmaxCrossEntropyLossLayer
    {
        [TestMethod]
        public void TestForward()
        {
            SoftmaxCrossEntropyLossLayerTest test = new SoftmaxCrossEntropyLossLayerTest();

            try
            {
                foreach (ISoftmaxCrossEntropyLossLayerTest t in test.Tests)
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
        public void TestForwardBatch()
        {
            SoftmaxCrossEntropyLossLayerTest test = new SoftmaxCrossEntropyLossLayerTest();

            try
            {
                foreach (ISoftmaxCrossEntropyLossLayerTest t in test.Tests)
                {
                    t.TestForwardBatch();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackward()
        {
            SoftmaxCrossEntropyLossLayerTest test = new SoftmaxCrossEntropyLossLayerTest();

            try
            {
                foreach (ISoftmaxCrossEntropyLossLayerTest t in test.Tests)
                {
                    t.TestBackward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardBatch()
        {
            SoftmaxCrossEntropyLossLayerTest test = new SoftmaxCrossEntropyLossLayerTest();

            try
            {
                foreach (ISoftmaxCrossEntropyLossLayerTest t in test.Tests)
                {
                    t.TestBackwardBatch();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface ISoftmaxCrossEntropyLossLayerTest : ITest
    {
        void TestForward();
        void TestForwardBatch();
        void TestBackward();
        void TestBackwardBatch();
    }

    class SoftmaxCrossEntropyLossLayerTest : TestBase
    {
        public SoftmaxCrossEntropyLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("SoftmaxCrossEntropyLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SoftmaxCrossEntropyLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SoftmaxCrossEntropyLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SoftmaxCrossEntropyLossLayerTest<T> : TestEx<T>, ISoftmaxCrossEntropyLossLayerTest
    {
        Blob<T> m_blob_bottom_targets;

        public SoftmaxCrossEntropyLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 4, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blob_bottom_targets = new Blob<T>(m_cuda, m_log, Bottom);

            // Fill the data vector.
            FillerParameter data_fp = new FillerParameter("gaussian");
            data_fp.std = 1;
            Filler<T> fillerData = Filler<T>.Create(m_cuda, m_log, data_fp);
            fillerData.Fill(Bottom);

            // Fill the targets vector.
            FillerParameter fp = new FillerParameter("uniform");
            fp.min = 0;
            fp.max = 1;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(m_blob_bottom_targets);

            BottomVec.Add(m_blob_bottom_targets);
        }

        protected override void dispose()
        {
            m_blob_bottom_targets.Dispose();
            base.dispose();
        }

        public Blob<T> BottomTargets
        {
            get { return m_blob_bottom_targets; }
        }

        /// <summary>
        /// Test function for forward with a single batch.
        /// </summary>
        /// <remarks>
        /// Test code used to compare results:
        /// <code>
        /// import torch
        /// import torch.nn as nn
        /// import torch
        /// torch.manual_seed(1)
        ///
        /// def NLLLoss(logs, targets, reduce= True):
        /// out = torch.zeros_like(targets, dtype=torch.float)
        /// for n in range(len(targets)) :
        ///    for i in range(len(targets[n])) :
        ///       out[n][i] = logs[n][i][targets[n][i]]
        /// outSum = out.sum()
        /// outLen = len(out) * len(out[0])
        /// loss = -(outSum/outLen)
        /// return loss if reduce else out
        ///
        /// x = torch.tensor([[[-0.1, 0.2, -0.3, 0.4],
        ///                    [0.5, -0.6, 0.7, -0.8],
        ///                    [-0.9, 0.1, -0.11, 0.12]]])
        /// y = torch.LongTensor([[0, 1, 2]])
        ///
        /// cross_entropy_loss = torch.nn.CrossEntropyLoss()
        /// log_softmax = torch.nn.LogSoftmax(dim = 2)
        ///
        /// x_log = log_softmax(x)
        /// print(x_log)
        ///
        /// torch.set_printoptions(precision=10)
        /// loss = NLLLoss(x_log, y)
        /// print("Custom NLL loss: ", loss)
        ///
        /// nll_loss = torch.nn.NLLLoss()
        /// print("Torch CrossEntropyLoss: ", cross_entropy_loss(x[0], y[0]))
        /// print("Torch NLL loss: ", nll_loss(x_log[0], y[0]))
        /// </code>
        /// </remarks>
        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXCROSSENTROPY2_LOSS);
            p.softmax_param.axis = 2;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);
            m_log.CHECK(layer.type == LayerParameter.LayerType.SOFTMAXCROSSENTROPY2_LOSS, "The layer type is incorrect.");

            m_blob_bottom.Reshape(1, 3, 4, 1);
            m_blob_bottom_targets.Reshape(1, 3, 1, 1);

            double[] rgInput =
            {
                -0.1, 0.2, -0.3, 0.4,
                0.5, -0.6, 0.7, -0.8,
                -0.9, 0.1, -0.11, 0.12
            };

            m_blob_bottom.mutable_cpu_data = convert(rgInput);

            double[] rgTarget =
            {
                0.0, 1.0, 2.0
            };

            m_blob_bottom_targets.mutable_cpu_data = convert(rgTarget);

            BottomVec.Clear();
            BottomVec.Add(m_blob_bottom);
            BottomVec.Add(m_blob_bottom_targets);
            TopVec.Clear();
            TopVec.Add(m_blob_top);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double dfExpectedLoss = 1.6946989298;        
            double dfActualLoss = convert(m_blob_top.GetData(0));
            double dfDiff = Math.Abs(dfExpectedLoss - dfActualLoss);
            double dfErr = 1e-7;

            if (dfDiff > dfErr)
                m_log.FAIL("The loss values do not match!");            
        }

        /// <summary>
        /// Test function for forward with a multi batch.
        /// </summary>
        /// <remarks>
        /// Test code used to compare results:
        /// <code>
        /// import torch
        /// import torch.nn as nn
        /// import torch
        /// torch.manual_seed(1)
        ///
        /// def NLLLoss(logs, targets, reduce= True):
        /// out = torch.zeros_like(targets, dtype=torch.float)
        /// for n in range(len(targets)) :
        ///    for i in range(len(targets[n])) :
        ///       out[n][i] = logs[n][i][targets[n][i]]
        /// outSum = out.sum()
        /// outLen = len(out) * len(out[0])
        /// loss = -(outSum/outLen)
        /// return loss if reduce else out
        ///
        /// x = torch.tensor([[[-0.1, 0.2, -0.3, 0.4],
        ///                     [0.5, -0.6, 0.7, -0.8],
        ///                     [-0.9, 0.1, -0.11, 0.12]],
        ///                    [[-1.1, 1.2, -1.3, 1.4],
        ///                     [1.5, -1.6, 1.7, -1.8],
        ///                     [-1.9, 1.1, -1.11, 1.12]],
        ///                    [[-2.1, 2.2, -2.3, 2.4],
        ///                     [2.5, -2.6, 2.7, -2.8],
        ///                     [-2.9, 2.1, -2.11, 2.12]]])
        /// y = torch.LongTensor([[0, 1, 2],
        ///                       [0, 1, 2],
        ///                       [0, 1, 2]])
        ///
        /// cross_entropy_loss = torch.nn.CrossEntropyLoss()
        /// log_softmax = torch.nn.LogSoftmax(dim = 2)
        ///
        /// x_log = log_softmax(x)
        /// print(x_log)
        ///
        /// torch.set_printoptions(precision=10)
        /// loss = NLLLoss(x_log, y)
        /// print("Custom NLL loss: ", loss)
        ///
        /// nll_loss = torch.nn.NLLLoss()
        /// print("Torch CrossEntropyLoss: ", cross_entropy_loss(x[0], y[0]))
        /// print("Torch NLL loss: ", nll_loss(x_log[0], y[0]))
        /// </code>
        /// </remarks>
        public void TestForwardBatch()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXCROSSENTROPY2_LOSS);
            p.softmax_param.axis = 2;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);
            m_log.CHECK(layer.type == LayerParameter.LayerType.SOFTMAXCROSSENTROPY2_LOSS, "The layer type is incorrect.");

            m_blob_bottom.Reshape(3, 3, 4, 1);
            m_blob_bottom_targets.Reshape(3, 3, 1, 1);

            double[] rgInput =
            {
                -0.1,  0.2, -0.3,  0.4,
                 0.5, -0.6,  0.7, -0.8,
                -0.9,  0.1, -0.11, 0.12,

                -1.1,  1.2, -1.3,  1.4,
                 1.5, -1.6,  1.7, -1.8,
                -1.9,  1.1, -1.11, 1.12,

                -2.1,  2.2, -2.3,  2.4,
                 2.5, -2.6,  2.7, -2.8,
                -2.9,  2.1, -2.11, 2.12
            };

            m_blob_bottom.mutable_cpu_data = convert(rgInput);

            double[] rgTarget =
            {
                0.0, 1.0, 2.0,
                0.0, 1.0, 2.0,
                0.0, 1.0, 2.0
            };

            m_blob_bottom_targets.mutable_cpu_data = convert(rgTarget);

            BottomVec.Clear();
            BottomVec.Add(m_blob_bottom);
            BottomVec.Add(m_blob_bottom_targets);
            TopVec.Clear();
            TopVec.Add(m_blob_top);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double dfExpectedLoss = 3.4578659534;
            double dfActualLoss = convert(m_blob_top.GetData(0));
            double dfDiff = Math.Abs(dfExpectedLoss - dfActualLoss);
            double dfErr = 1e-6;

            if (dfDiff > dfErr)
                m_log.FAIL("The loss values do not match!");
        }

        /// <summary>
        /// Test function for backward with a single batch.
        /// </summary>
        /// <remarks>
        /// Test code used to compare results:
        /// <code>
        /// import torch
        /// import torch.nn as nn
        /// import torch
        /// torch.manual_seed(1)
        ///
        /// class DebugFunction(torch.autograd.Function):
        ///    @staticmethod
        ///    def forward(ctx, input) :
        ///        ctx.save_for_backward(input)       
        ///        return input
        ///
        ///    @staticmethod
        ///    def backward(ctx, grad_output):
        ///        input, = ctx.saved_tensors
        ///        name = input_dict.get(input)
        ///        
        ///        if name == None:
        ///            name = "unknown";
        ///
        ///        b = grad_output.detach().cpu().numpy()
        ///        print(name, input.shape, grad_output)
        ///        return grad_output
        /// 
        /// def NLLLoss(logs, targets, reduce= True):
        /// out = torch.zeros_like(targets, dtype=torch.float)
        /// for n in range(len(targets)) :
        ///    for i in range(len(targets[n])) :
        ///       out[n][i] = logs[n][i][targets[n][i]]
        /// outSum = out.sum()
        /// outLen = len(out) * len(out[0])
        /// loss = -(outSum/outLen)
        /// return loss if reduce else out
        ///
        /// x = torch.tensor([[[-0.1, 0.2, -0.3, 0.4],
        ///                    [0.5, -0.6, 0.7, -0.8],
        ///                    [-0.9, 0.1, -0.11, 0.12]]])
        /// x.requires_grad = True
        /// y = torch.LongTensor([[0, 1, 2]])
        ///
        /// torch.set_printoptions(precision=10)
        /// debug = DebugFunction.apply
        ///
        /// cross_entropy_loss = torch.nn.CrossEntropyLoss()
        ///
        /// x = debug(x)
        /// input_dict.update({x: "x"})
        /// 
        /// x = x.view(-1, x.size(-1))
        /// y = y.view(-1)
        /// celoss = cross_entropy_loss(x, y)
        /// print("Torch CrossEntropyLoss: ", celoss)
        /// 
        /// celoss.backward()
        /// </code>
        /// </remarks>
        public void TestBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXCROSSENTROPY2_LOSS);
            p.softmax_param.axis = 2;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);
            m_log.CHECK(layer.type == LayerParameter.LayerType.SOFTMAXCROSSENTROPY2_LOSS, "The layer type is incorrect.");

            m_blob_bottom.Reshape(1, 3, 4, 1);
            m_blob_bottom_targets.Reshape(1, 3, 1, 1);

            double[] rgInput =
            {
                -0.1, 0.2, -0.3, 0.4,
                0.5, -0.6, 0.7, -0.8,
                -0.9, 0.1, -0.11, 0.12
            };

            m_blob_bottom.mutable_cpu_data = convert(rgInput);

            double[] rgTarget =
            {
                0.0, 1.0, 2.0
            };

            m_blob_bottom_targets.mutable_cpu_data = convert(rgTarget);

            BottomVec.Clear();
            BottomVec.Add(m_blob_bottom);
            BottomVec.Add(m_blob_bottom_targets);
            TopVec.Clear();
            TopVec.Add(m_blob_top);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double dfExpectedLoss = 1.6946989298;
            double dfActualLoss = convert(m_blob_top.GetData(0));
            double dfDiff = Math.Abs(dfExpectedLoss - dfActualLoss);
            double dfErr = 1e-7;

            if (dfDiff > dfErr)
                m_log.FAIL("The loss values do not match!");

            // Test the backward pass
            m_blob_top.SetDiff(1.0);
            layer.Backward(TopVec, new List<bool>() { true }, BottomVec);
            double[] rgGrad = convert(m_blob_bottom.mutable_cpu_diff);
            double[] rgExpectedGrad =
            {
                -0.2641384602,  0.0934033319,  0.0566519871,  0.1140830964,
                0.1179187223, -0.2940815985,  0.1440262496,  0.0321366042,
                0.0383367687,  0.1042101383, -0.2488622367,  0.1063153297
            };

            for (int i = 0; i < rgGrad.Length; i++)
            {
                double dfExpected = rgExpectedGrad[i];
                double dfActual = rgGrad[i];
                double dfDiffGrad = Math.Abs(dfExpected - dfActual);
                double dfErrGrad = 1e-7;

                if (dfDiffGrad > dfErrGrad)
                    m_log.FAIL("The gradient values do not match!");
            }
        }

        /// <summary>
        /// Test function for backward with a multi batch.
        /// </summary>
        /// <remarks>
        /// Test code used to compare results:
        /// <code>
        /// import torch
        /// import torch.nn as nn
        /// import torch
        /// torch.manual_seed(1)
        ///
        /// class DebugFunction(torch.autograd.Function):
        ///    @staticmethod
        ///    def forward(ctx, input) :
        ///        ctx.save_for_backward(input)       
        ///        return input
        ///
        ///    @staticmethod
        ///    def backward(ctx, grad_output):
        ///        input, = ctx.saved_tensors
        ///        name = input_dict.get(input)
        ///        
        ///        if name == None:
        ///            name = "unknown";
        ///
        ///        b = grad_output.detach().cpu().numpy()
        ///        print(name, input.shape, grad_output)
        ///        return grad_output
        /// 
        /// def NLLLoss(logs, targets, reduce= True):
        /// out = torch.zeros_like(targets, dtype=torch.float)
        /// for n in range(len(targets)) :
        ///    for i in range(len(targets[n])) :
        ///       out[n][i] = logs[n][i][targets[n][i]]
        /// outSum = out.sum()
        /// outLen = len(out) * len(out[0])
        /// loss = -(outSum/outLen)
        /// return loss if reduce else out
        ///
        /// x = torch.tensor([[[-0.1, 0.2, -0.3, 0.4],
        ///                     [0.5, -0.6, 0.7, -0.8],
        ///                     [-0.9, 0.1, -0.11, 0.12]],
        ///                    [[-1.1, 1.2, -1.3, 1.4],
        ///                     [1.5, -1.6, 1.7, -1.8],
        ///                     [-1.9, 1.1, -1.11, 1.12]],
        ///                    [[-2.1, 2.2, -2.3, 2.4],
        ///                     [2.5, -2.6, 2.7, -2.8],
        ///                     [-2.9, 2.1, -2.11, 2.12]]])
        /// x.requires_grad = True
        /// y = torch.LongTensor([[0, 1, 2],
        ///                       [0, 1, 2],
        ///                       [0, 1, 2]])
        ///
        /// torch.set_printoptions(precision=10)
        /// debug = DebugFunction.apply
        ///
        /// cross_entropy_loss = torch.nn.CrossEntropyLoss()
        ///
        /// x = debug(x)
        /// input_dict.update({x: "x"})
        /// 
        /// x = x.view(-1, x.size(-1))
        /// y = y.view(-1)
        /// celoss = cross_entropy_loss(x, y)
        /// print("Torch CrossEntropyLoss: ", celoss)
        /// 
        /// celoss.backward()
        /// </code>
        /// </remarks>
        public void TestBackwardBatch()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXCROSSENTROPY2_LOSS);
            p.softmax_param.axis = 2;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);
            m_log.CHECK(layer.type == LayerParameter.LayerType.SOFTMAXCROSSENTROPY2_LOSS, "The layer type is incorrect.");

            m_blob_bottom.Reshape(3, 3, 4, 1);
            m_blob_bottom_targets.Reshape(3, 3, 1, 1);

            double[] rgInput =
            {
                -0.1,  0.2, -0.3,  0.4,
                 0.5, -0.6,  0.7, -0.8,
                -0.9,  0.1, -0.11, 0.12,

                -1.1,  1.2, -1.3,  1.4,
                 1.5, -1.6,  1.7, -1.8,
                -1.9,  1.1, -1.11, 1.12,

                -2.1,  2.2, -2.3,  2.4,
                 2.5, -2.6,  2.7, -2.8,
                -2.9,  2.1, -2.11, 2.12
            };

            m_blob_bottom.mutable_cpu_data = convert(rgInput);

            double[] rgTarget =
            {
                0.0, 1.0, 2.0,
                0.0, 1.0, 2.0,
                0.0, 1.0, 2.0
            };

            m_blob_bottom_targets.mutable_cpu_data = convert(rgTarget);

            BottomVec.Clear();
            BottomVec.Add(m_blob_bottom);
            BottomVec.Add(m_blob_bottom_targets);
            TopVec.Clear();
            TopVec.Add(m_blob_top);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double dfExpectedLoss = 3.4578659534;
            double dfActualLoss = convert(m_blob_top.GetData(0));
            double dfDiff = Math.Abs(dfExpectedLoss - dfActualLoss);
            double dfErr = 1e-6;

            if (dfDiff > dfErr)
                m_log.FAIL("The loss values do not match!");

            // Test the backward pass
            m_blob_top.SetDiff(1.0);
            layer.Backward(TopVec, new List<bool>() { true }, BottomVec);
            double[] rgGrad = convert(m_blob_bottom.mutable_cpu_diff);
            double[] rgExpectedGrad =
            {
                -0.0880461335,  0.0311344452,  0.0188839938,  0.0380276814,
                0.0393062420, -0.0980271995,  0.0480087548,  0.0107122008,
                0.0127789229,  0.0347367115, -0.0829540789,  0.0354384296,

                -0.1064767316,  0.0462241359,  0.0037943083,  0.0564582832,
                0.0482392237, -0.1089379713,  0.0589195229,  0.0017792154,
                0.0025379297,  0.0509756766, -0.1055190340,  0.0520054512,

                -0.1104398817,  0.0494689010,  0.0005495498,  0.0604214482,
                0.0497700162, -0.1108076721,  0.0607892349,  0.0002484317,
                0.0003666696,  0.0544185899, -0.1103031859,  0.0555179194
            };

            for (int i = 0; i < rgGrad.Length; i++)
            {
                double dfExpected = rgExpectedGrad[i];
                double dfActual = rgGrad[i];
                double dfDiffGrad = Math.Abs(dfExpected - dfActual);
                double dfErrGrad = 1e-7;

                if (dfDiffGrad > dfErrGrad)
                    m_log.FAIL("The gradient values do not match!");
            }
        }
    }
}
