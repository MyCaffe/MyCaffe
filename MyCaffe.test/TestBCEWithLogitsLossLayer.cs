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
    public class TestBCEWithLogitsLossLayer
    {
        [TestMethod]
        public void TestForward()
        {
            BCEWithLogitsLossLayerTest test = new BCEWithLogitsLossLayerTest();

            try
            {
                foreach (IBCEWithLogitsLossLayerTest t in test.Tests)
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
        public void TestBackward()
        {
            BCEWithLogitsLossLayerTest test = new BCEWithLogitsLossLayerTest();

            try
            {
                foreach (IBCEWithLogitsLossLayerTest t in test.Tests)
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
        public void TestForwardWeights()
        {
            BCEWithLogitsLossLayerTest test = new BCEWithLogitsLossLayerTest();

            try
            {
                foreach (IBCEWithLogitsLossLayerTest t in test.Tests)
                {
                    t.TestForwardWeights();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardWeights()
        {
            BCEWithLogitsLossLayerTest test = new BCEWithLogitsLossLayerTest();

            try
            {
                foreach (IBCEWithLogitsLossLayerTest t in test.Tests)
                {
                    t.TestBackwardWeights();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IBCEWithLogitsLossLayerTest : ITest
    {
        void TestForward();
        void TestBackward();
        void TestForwardWeights();
        void TestBackwardWeights();
    }

    class BCEWithLogitsLossLayerTest : TestBase
    {
        public BCEWithLogitsLossLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("BCEWithLogitsLoss Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new BCEWithLogitsLossLayerTest<double>(strName, nDeviceID, engine);
            else
                return new BCEWithLogitsLossLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class BCEWithLogitsLossLayerTest<T> : TestEx<T>, IBCEWithLogitsLossLayerTest
    {
        BCEWithLogitsLoss m_bce = new BCEWithLogitsLoss();
        Blob<T> m_blob_bottom_targets;

        public BCEWithLogitsLossLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS);
            p.loss_param.normalization = LossParameter.NormalizationMode.NONE;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS, "The layer type is incorrect.");

                double[] rgInput = new double[] { 0.2, -0.5, 1.3, 0.7, -1.2 };
                double[] rgTarget = new double[] { 1, 0, 1, 0, 1 };
                double[] rgExpectedLoss = m_bce.ComputeLoss(rgInput, rgTarget);
                double dfExpectedLoss = rgExpectedLoss[0];

                m_blob_bottom.Reshape(1, 5, 1, 1);
                m_blob_bottom_targets.Reshape(1, 5, 1, 1);

                m_blob_bottom.mutable_cpu_data = convert(rgInput);
                m_blob_bottom_targets.mutable_cpu_data = convert(rgTarget);

                BottomVec.Clear();
                BottomVec.Add(m_blob_bottom);
                BottomVec.Add(m_blob_bottom_targets);
                TopVec.Clear();
                TopVec.Add(m_blob_top);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double dfActualLoss = convert(m_blob_top.GetData(0));
                double dfDiff = Math.Abs(dfExpectedLoss - dfActualLoss);
                double dfErr = 1e-7;

                if (dfDiff > dfErr)
                    m_log.FAIL("The loss values do not match!");
            }
            finally
            {
                layer.Dispose();
            }
        }

        /// <summary>
        /// Test function for backward with a single batch.
        /// </summary>
        public void TestBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS);
            p.loss_param.normalization = LossParameter.NormalizationMode.NONE;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS, "The layer type is incorrect.");

                double[] rgInput = new double[] { 0.2, -0.5, 1.3, 0.7, -1.2 };
                double[] rgTarget = new double[] { 1, 0, 1, 0, 1 };
                double[] rgGrad = new double[] { 1.0 };
                double[] rgExpectedLoss = m_bce.ComputeLoss(rgInput, rgTarget);
                double dfExpectedLoss = rgExpectedLoss[0];
                double[] rgExpectedGrad = m_bce.ComputeGradient(rgGrad);

                m_blob_bottom.Reshape(1, 5, 1, 1);
                m_blob_bottom_targets.Reshape(1, 5, 1, 1);

                m_blob_bottom.mutable_cpu_data = convert(rgInput);
                m_blob_bottom_targets.mutable_cpu_data = convert(rgTarget);

                BottomVec.Clear();
                BottomVec.Add(m_blob_bottom);
                BottomVec.Add(m_blob_bottom_targets);
                TopVec.Clear();
                TopVec.Add(m_blob_top);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double dfActualLoss = convert(m_blob_top.GetData(0));
                double dfDiff = Math.Abs(dfExpectedLoss - dfActualLoss);
                double dfErr = 1e-7;

                if (dfDiff > dfErr)
                    m_log.FAIL("The loss values do not match!");

                m_blob_top.mutable_cpu_diff = convert(rgGrad);
                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                double[] rgActualGrad = convert(m_blob_bottom.mutable_cpu_diff);

                for (int i = 0; i < rgActualGrad.Length; i++)
                {
                    double dfExpected = rgExpectedGrad[i];
                    double dfActual = rgActualGrad[i];
                    double dfDiffGrad = Math.Abs(dfExpected - dfActual);
                    double dfErrGrad = 1e-7;

                    if (dfDiffGrad > dfErrGrad)
                        m_log.FAIL("The gradient values do not match!");
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        private List<float> getWeights(int n, int nOffset)
        {
            List<float> rgWts = new List<float>();

            for (int i=0; i<n; i++)
            {
                float fWt = 10.0f;
                if (i >= nOffset)
                    fWt += 2.0f;

                rgWts.Add(fWt);
            }

            return rgWts;
        }

        /// <summary>
        /// Test function for forward with a single batch.
        /// </summary>
        public void TestForwardWeights()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS);
            p.loss_param.normalization = LossParameter.NormalizationMode.NONE;
            p.bce_with_logits_loss_param.weights = getWeights(5, 3);
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS, "The layer type is incorrect.");
                m_bce.SetWeights(p.bce_with_logits_loss_param.weights);

                double[] rgInput = new double[] { 0.2, -0.5, 1.3, 0.7, -1.2 };
                double[] rgTarget = new double[] { 1, 0, 1, 0, 1 };
                double[] rgExpectedLoss = m_bce.ComputeLoss(rgInput, rgTarget);
                double dfExpectedLoss = rgExpectedLoss[0];

                m_blob_bottom.Reshape(1, 5, 1, 1);
                m_blob_bottom_targets.Reshape(1, 5, 1, 1);

                m_blob_bottom.mutable_cpu_data = convert(rgInput);
                m_blob_bottom_targets.mutable_cpu_data = convert(rgTarget);

                BottomVec.Clear();
                BottomVec.Add(m_blob_bottom);
                BottomVec.Add(m_blob_bottom_targets);
                TopVec.Clear();
                TopVec.Add(m_blob_top);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double dfActualLoss = convert(m_blob_top.GetData(0));
                double dfDiff = Math.Abs(dfExpectedLoss - dfActualLoss);
                double dfErr = 1e-6;

                if (dfDiff > dfErr)
                    m_log.FAIL("The loss values do not match!");
            }
            finally
            {
                layer.Dispose();
                m_bce.ClearWeights();
            }
        }

        /// <summary>
        /// Test function for backward with a single batch.
        /// </summary>
        public void TestBackwardWeights()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS);
            p.bce_with_logits_loss_param.weights = getWeights(5, 3);
            p.loss_param.normalization = LossParameter.NormalizationMode.NONE;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                m_bce.SetWeights(p.bce_with_logits_loss_param.weights);
                m_log.CHECK(layer.type == LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS, "The layer type is incorrect.");

                double[] rgInput = new double[] { 0.2, -0.5, 1.3, 0.7, -1.2 };
                double[] rgTarget = new double[] { 1, 0, 1, 0, 1 };
                double[] rgGrad = new double[] { 1.0 };
                double[] rgExpectedLoss = m_bce.ComputeLoss(rgInput, rgTarget);
                double dfExpectedLoss = rgExpectedLoss[0];
                double[] rgExpectedGrad = m_bce.ComputeGradient(rgGrad);

                m_blob_bottom.Reshape(1, 5, 1, 1);
                m_blob_bottom_targets.Reshape(1, 5, 1, 1);

                m_blob_bottom.mutable_cpu_data = convert(rgInput);
                m_blob_bottom_targets.mutable_cpu_data = convert(rgTarget);

                BottomVec.Clear();
                BottomVec.Add(m_blob_bottom);
                BottomVec.Add(m_blob_bottom_targets);
                TopVec.Clear();
                TopVec.Add(m_blob_top);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                double dfActualLoss = convert(m_blob_top.GetData(0));
                double dfDiff = Math.Abs(dfExpectedLoss - dfActualLoss);
                double dfErr = 1e-6;

                if (dfDiff > dfErr)
                    m_log.FAIL("The loss values do not match!");

                m_blob_top.mutable_cpu_diff = convert(rgGrad);
                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                double[] rgActualGrad = convert(m_blob_bottom.mutable_cpu_diff);

                for (int i = 0; i < rgActualGrad.Length; i++)
                {
                    double dfExpected = rgExpectedGrad[i];
                    double dfActual = rgActualGrad[i];
                    double dfDiffGrad = Math.Abs(dfExpected - dfActual);
                    double dfErrGrad = 1e-7;

                    if (dfDiffGrad > dfErrGrad)
                        m_log.FAIL("The gradient values do not match!");
                }
            }
            finally
            {
                layer.Dispose();
                m_bce.ClearWeights();
            }
        }
    }

    /// <summary>
    /// The BCEWithLogitsLoss class the BCEWithLogitsLoss function in C# without GPU support.  Note, this has been verified
    /// against the BCEWithLogitsLoss function in PyTorch.
    /// </summary>
    /// <remarks>
    /// @see [What does BCEWithLogitsLoss actually do?](https://kamilelukosiute.com/2022/04/14/bce-with-logits-loss/) by Kamile Lukosiute, 2022
    /// @see [How to Use PyTorch's BCEWithLogitsLoss Function](https://reason.town/pytorch-bcewithlogitsloss/) by joseph, 2022
    /// </remarks>
    /// <summary>
    /// A sclass that implements the BCEWithLogitsLoss function
    /// </summary>
    /// <remarks>
    /// @see [What does BCEWithLogitsLoss actually do?](https://kamilelukosiute.com/2022/04/14/bce-with-logits-loss/) by Kamile Lukosiute, 2022
    /// @see [How to Use PyTorch's BCEWithLogitsLoss Function](https://reason.town/pytorch-bcewithlogitsloss/) by joseph, 2022
    /// </remarks>
    public class BCEWithLogitsLoss
    {
        // A constructor that takes optional parameters for weight, reduction and pos_weight
        public BCEWithLogitsLoss(string reduction = "mean", double[] pos_weight = null, double[] weight = null)
        {
            this.m_reduction = reduction;
            this.m_pos_weight = pos_weight;
            this.m_weight = weight;
        }

        public void ClearWeights()
        {
            m_weight = null;
        }

        public void SetWeights(List<float> rgWts)
        {
            m_weight = rgWts.ToArray().Select(p => (double)p).ToArray();
        }

        public double[] Weight
        {
            get { return m_weight; }
            set { m_weight = value; }
        }

        // A method that computes the loss given the input and target tensors
        public double[] ComputeLoss(double[] input, double[] target)
        {
            this.m_input = input;
            this.m_target = target;

            // Check that the input and target have the same length
            if (input.Length != target.Length)
            {
                throw new ArgumentException("Input and target must have the same length");
            }

            // Initialize the loss variable
            double[] loss = new double[input.Length];

            // Loop over the input and target elements
            for (int i = 0; i < input.Length; i++)
            {
                // Clamp the input to avoid overflow
                double z = Math.Max(0, -input[i]);

                // Apply the pos_weight if defined
                double log_weight = 1;
                if (m_pos_weight != null)
                {
                    log_weight = (m_pos_weight[i] - 1) * target[i] + 1;
                }

                // Compute the loss element-wise
                loss[i] = (1 - target[i]) * input[i] + log_weight * (z + Math.Log(Math.Exp(-z) + Math.Exp(-input[i] - z)));
            }

            // Apply the weight if defined
            if (m_weight != null)
            {
                for (int i = 0; i < loss.Length; i++)
                {
                    loss[i] *= m_weight[i];
                }
            }

            double dfLoss = 0;

            // Apply the reduction if defined
            if (m_reduction == "mean")
            {
                dfLoss = loss.Sum();
                dfLoss /= input.Length;
            }
            else if (m_reduction == "sum")
            {
                dfLoss = loss.Sum();
            }
            else if (m_reduction == "none")
            {
                return loss;
            }
            else
            {
                throw new ArgumentException("Invalid reduction option");
            }

            // Return the loss as a scalar
            return new double[] { dfLoss };
        }

        // A method that computes the gradient of the loss with respect to the input tensor
        public double[] ComputeGradient(double[] grad)
        {
            double[] input = m_input;
            double[] target = m_target;

            // Check that the input and target have the same length
            if (input.Length != target.Length)
            {
                throw new ArgumentException("Input and target must have the same length");
            }

            // Initialize the gradient variable
            double[] gradient = new double[input.Length];

            // Calculate the gradient
            // compute the sigmoid of the input
            double[] input_sigmoid = input.Select(x => 1 / (1 + Math.Exp(-x))).ToArray();

            // compute the gradient of the binary cross entropy loss with respect to the input
            for (int i = 0; i < input.Length; i++)
            {
                gradient[i] = -target[i] * (1 - input_sigmoid[i]) + (1 - target[i]) * input_sigmoid[i];
            }

            // apply the posittive weight if defined
            if (m_pos_weight != null)
            {
                for (int i = 0; i < gradient.Length; i++)
                {
                    gradient[i] *= (m_pos_weight[i] * target[i] + (1 - target[i]));
                }
            }

            // Apply chain rule.
            for (int i = 0; i < gradient.Length; i++)
            {
                gradient[i] *= grad[0];
            }

            // Apply the reduction if defined
            if (m_reduction == "mean")
            {
                for (int i = 0; i < gradient.Length; i++)
                {
                    gradient[i] /= input.Length;
                }
            }
            else if (m_reduction == "sum")
            {
                // Do nothing, the gradient is already summed
            }
            else if (m_reduction == "none")
            {
                // Return an array of gradients instead of a scalar
                return gradient;
            }
            else
            {
                throw new ArgumentException("Invalid reduction option");
            }

            // Return the gradient as a scalar
            return gradient;
        }

        // The weight, reduction and pos_weight parameters
        private double[] m_weight;
        private string m_reduction;
        private double[] m_pos_weight;
        private double[] m_target;
        private double[] m_input;
    }
}
