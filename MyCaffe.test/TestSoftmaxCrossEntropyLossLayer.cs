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

        /// <summary>
        /// This test fails.
        /// </summary>
        [TestMethod]
        public void TestGradient()
        {
            SoftmaxCrossEntropyLossLayerTest test = new SoftmaxCrossEntropyLossLayerTest();

            try
            {
                foreach (ISoftmaxCrossEntropyLossLayerTest t in test.Tests)
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
        public void TestIgnoreGradient()
        {
            SoftmaxCrossEntropyLossLayerTest test = new SoftmaxCrossEntropyLossLayerTest();

            try
            {
                foreach (ISoftmaxCrossEntropyLossLayerTest t in test.Tests)
                {
                    t.TestIgnoreGradient();
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
        void TestGradient();
        void TestIgnoreGradient();
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
        ///                        [0, 1, 2],
        ///                        [0, 1, 2]])
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
        /// WORK IN PROGRESS
        /// </summary>
        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXCROSSENTROPY2_LOSS);
            double kLossWeight = 3.7;
            p.loss_weight.Add(kLossWeight);
            SoftmaxCrossEntropyLossLayer<T> layer = new SoftmaxCrossEntropyLossLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            try
            {
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2, 1701);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
            }
            finally
            {
                layer.Dispose();
            }
        }

        /// <summary>
        /// WORK IN PROGRESS
        /// </summary>
        public void TestIgnoreGradient()
        {
            FillerParameter data_filler_param = new FillerParameter("gaussian", 0, 0, 1);
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, data_filler_param);
            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAXCROSSENTROPY2_LOSS);
            p.loss_param.ignore_label = -1;

            long hTarget = BottomTargets.mutable_gpu_data;
            int nCount = BottomTargets.count();

            // Ignore half of targets, then check that diff of this half is zero,
            // while the other half is nonzero.
            m_cuda.set(nCount / 2, hTarget, -1);

            SoftmaxCrossEntropyLossLayer<T> layer = new SoftmaxCrossEntropyLossLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                List<bool> rgbPropagateDown = new List<bool>();
                rgbPropagateDown.Add(true);
                rgbPropagateDown.Add(false);

                layer.Backward(TopVec, rgbPropagateDown, BottomVec);

                double[] rgDiff = convert(Bottom.update_cpu_diff());

                for (int i = 0; i < nCount / 2; i++)
                {
                    double dfVal1 = rgDiff[i];
                    double dfVal2 = rgDiff[i + nCount / 2];

                    m_log.EXPECT_EQUAL<float>(dfVal1, 0, "The " + i.ToString() + "th value of the first half should be zero.");
                    m_log.CHECK_NE(dfVal2, 0, "The " + i.ToString() + "th value of the second half should not be zero.");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
