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
        BCEWithLogitsLoss<T> m_bce;
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

        /// <summary>
        /// Assuming batch size = 21
        /// </summary>
        /// <returns>Returns the weights.</returns>
        private List<float> getWeights()
        {
            double[] rgWts = new double[21]
            {
                11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
                10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
            };
            return rgWts.Select(p => (float)p).ToList();
        }

        /// <summary>
        /// Assumes shape = (21, 1), batch size = 21.
        /// </summary>
        /// <param name="rgInput">Specifies the input data.</param>
        /// <param name="rgTarget">Specifies the target data.</param>
        public void getData(out double[] rgInput, out double[] rgTarget)
        {
            rgInput = new double[21]
            {
                0.2066, -0.1471, -0.0830, -0.0881, -0.0279, 0.1623, 0.0013, -0.0695,  0.0622, -0.0600,
                0.1123,  0.0305,  0.1389, -0.0661,  0.3031, 0.0825, 0.0655, -0.0051, -0.0726, -0.0868, -0.0136
            };
            rgTarget = new double[21]
            {
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };
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
                m_bce = new BCEWithLogitsLoss<T>(21, 1);
                m_log.CHECK(layer.type == LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS, "The layer type is incorrect.");

                double[] rgInput;
                double[] rgTarget;
                getData(out rgInput, out rgTarget);
                double[] rgExpectedLoss = m_bce.ComputeLoss(rgInput, rgTarget);
                double dfExpectedLoss = rgExpectedLoss[0];

                m_blob_bottom.Reshape(21, 1, 1, 1);
                m_blob_bottom_targets.Reshape(21, 1, 1, 1);

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
                m_bce = new BCEWithLogitsLoss<T>(21, 1);

                double[] rgInput;
                double[] rgTarget;
                getData(out rgInput, out rgTarget);
                double[] rgGrad = new double[] { 1.0 };
                double[] rgExpectedLoss = m_bce.ComputeLoss(rgInput, rgTarget);
                double dfExpectedLoss = rgExpectedLoss[0];
                double[] rgExpectedGrad = m_bce.ComputeGradient(rgGrad);

                m_blob_bottom.Reshape(21, 1, 1, 1);
                m_blob_bottom_targets.Reshape(21, 1, 1, 1);

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

        /// <summary>
        /// Test function for forward with a single batch.
        /// </summary>
        public void TestForwardWeights()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS);
            p.loss_param.normalization = LossParameter.NormalizationMode.NONE;
            p.bce_with_logits_loss_param.weights = getWeights();
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS, "The layer type is incorrect.");
                m_bce = new BCEWithLogitsLoss<T>(21, 1, null, p.bce_with_logits_loss_param.weights);

                double[] rgInput;
                double[] rgTarget;
                getData(out rgInput, out rgTarget);
                double[] rgExpectedLoss = m_bce.ComputeLoss(rgInput, rgTarget);
                double dfExpectedLoss = rgExpectedLoss[0];

                m_blob_bottom.Reshape(21, 1, 1, 1);
                m_blob_bottom_targets.Reshape(21, 1, 1, 1);

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
            p.bce_with_logits_loss_param.weights = getWeights();
            p.loss_param.normalization = LossParameter.NormalizationMode.NONE;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS, "The layer type is incorrect.");
                m_bce = new BCEWithLogitsLoss<T>(21, 1, null, p.bce_with_logits_loss_param.weights);

                double[] rgInput;
                double[] rgTarget;
                getData(out rgInput, out rgTarget);
                double[] rgGrad = new double[] { 1.0 };
                double[] rgExpectedLoss = m_bce.ComputeLoss(rgInput, rgTarget);
                double dfExpectedLoss = rgExpectedLoss[0];
                double[] rgExpectedGrad = m_bce.ComputeGradient(rgGrad);

                m_blob_bottom.Reshape(21, 1, 1, 1);
                m_blob_bottom_targets.Reshape(21, 1, 1, 1);

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
    public class BCEWithLogitsLoss<T>
    {
        // A constructor that takes optional parameters for weight, reduction and pos_weight
        public BCEWithLogitsLoss(int nN, int nC, T[] pos_weight = null, List<float> weight = null, string reduction="mean")
        {
            this.m_reduction = reduction;
            this.m_pos_weight = pos_weight;
            this.m_weight = (weight != null) ? weight.Select(p => (T)Convert.ChangeType(p, typeof(T))).ToArray() : null;
            this.m_nN = nN;
            this.m_nC = nC;

            if (pos_weight != null)
            {
                if (pos_weight.Length != nC)
                    throw new ArgumentException("pos_weight must have the same length as the input channel");
            }

            if (weight != null)
            {
                if (weight.Count != nN)
                    throw new ArgumentException("weight must have the same length as the input num");
            }
        }

        public void SetWeights(List<float> rgW)
        {
            m_weight = new T[rgW.Count];

            for (int i=0; i<rgW.Count; i++)
            {
                m_weight[i] = (T)Convert.ChangeType(rgW[i], typeof(T));
            }
        }

        public void ClearWeights()
        {
            m_weight = null;
        }

        public double[] ComputeLoss(double[] input, double[] target)
        {
            this.m_input = input.Select(p => (T)Convert.ChangeType(p, typeof(T))).ToArray();
            this.m_target = target.Select(p => (T)Convert.ChangeType(p, typeof(T))).ToArray();

            // Check that the input and target have the same length
            if (input.Length != target.Length)
            {
                throw new ArgumentException("Input and target must have the same length");
            }

            if (typeof(T) == typeof(double))
            {
                double[] rgInput = input;
                double[] rgTarget = target;
                double[] rgPosWt = (m_pos_weight != null) ? m_pos_weight.Select(p => (double)Convert.ChangeType(p, typeof(double))).ToArray() : null;
                double[] rgWt = (m_weight != null) ? m_weight.Select(p => (double)Convert.ChangeType(p, typeof(double))).ToArray() : null;
                double[] rgLoss = ComputeLossD(rgInput, rgTarget, rgPosWt, rgWt);
                return rgLoss;
            }
            else
            {
                float[] rgInput = input.Select(p => (float)Convert.ChangeType(p, typeof(float))).ToArray();
                float[] rgTarget = target.Select(p => (float)Convert.ChangeType(p, typeof(float))).ToArray();
                float[] rgPosWt = (m_pos_weight != null) ? m_pos_weight.Select(p => (float)Convert.ChangeType(p, typeof(float))).ToArray() : null;
                float[] rgWt = (m_weight != null) ? m_weight.Select(p => (float)Convert.ChangeType(p, typeof(float))).ToArray() : null;
                float[] rgLoss = ComputeLossF(rgInput, rgTarget, rgPosWt, rgWt);
                return rgLoss.Select(p => (double)Convert.ChangeType(p, typeof(double))).ToArray();
            }
        }

        // A method that computes the loss given the input and target tensors
        public double[] ComputeLossD(double[] input, double[] target, double[] pos_weight, double[] weight)
        {
            // Initialize the loss variable
            double[] loss = new double[input.Length];

            // Loop over the input and target elements
            for (int i = 0; i < input.Length; i++)
            {
                // Compute the sigmoid of the input
                double input_sigmoid = 1 / (1 + Math.Exp(-input[i]));
                double dfY = target[i];

                if (m_pos_weight != null)
                {
                    dfY *= pos_weight[i / m_nN];
                }

                // Compute the binary cross entropy loss
                double dfLogSigmoid = Math.Log(input_sigmoid);
                double dfLogOneMinusSigmoid = Math.Log(1 - input_sigmoid);
                loss[i] = -dfY * dfLogSigmoid - (1 - target[i]) * dfLogOneMinusSigmoid;

                if (pos_weight != null)
                {
                    loss[i] *= (pos_weight[i] * target[i] + (1 - target[i]));
                }
            }

            // Apply the weight if defined
            if (m_weight != null)
            {
                for (int i = 0; i < loss.Length; i++)
                {
                    loss[i] *= weight[i / m_nC];
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

        public float[] ComputeLossF(float[] input, float[] target, float[] pos_weight, float[] weight)
        {
            // Initialize the loss variable
            float[] loss = new float[input.Length];

            // Loop over the input and target elements
            for (int i = 0; i < input.Length; i++)
            {
                // Compute the sigmoid of the input
                float input_sigmoid = 1 / (1 + (float)Math.Exp(-input[i]));
                float fY = target[i];

                if (m_pos_weight != null)
                {
                    fY *= pos_weight[i / m_nN];
                }

                // Compute the binary cross entropy loss
                float fLogSigmoid = (float)Math.Log(input_sigmoid);
                float fLogOneMinusSigmoid = (float)Math.Log(1 - input_sigmoid);
                loss[i] = -fY * fLogSigmoid - (1 - target[i]) * fLogOneMinusSigmoid;

                if (pos_weight != null)
                {
                    loss[i] *= (pos_weight[i] * target[i] + (1 - target[i]));
                }
            }

            // Apply the weight if defined
            if (m_weight != null)
            {
                for (int i = 0; i < loss.Length; i++)
                {
                    loss[i] *= weight[i / m_nC];
                }
            }

            float fLoss = 0;

            // Apply the reduction if defined
            if (m_reduction == "mean")
            {
                fLoss = loss.Sum();
                fLoss /= input.Length;
            }
            else if (m_reduction == "sum")
            {
                fLoss = loss.Sum();
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
            return new float[] { fLoss };
        }

        // A method that computes the gradient of the loss with respect to the input tensor
        public double[] ComputeGradient(double[] grad)
        {
            T[] input = m_input;
            T[] target = m_target;

            // Check that the input and target have the same length
            if (input.Length != target.Length)
            {
                throw new ArgumentException("Input and target must have the same length");
            }

            if (typeof(T) == typeof(double))
            {
                double[] rgInput = input.Select(p => (double)Convert.ChangeType(p, typeof(double))).ToArray();
                double[] rgTarget = target.Select(p => (double)Convert.ChangeType(p, typeof(double))).ToArray();
                double[] rgPosWt = (m_pos_weight != null) ? m_pos_weight.Select(p => (double)Convert.ChangeType(p, typeof(double))).ToArray() : null;
                double[] rgWt = (m_weight != null) ? m_weight.Select(p => (double)Convert.ChangeType(p, typeof(double))).ToArray() : null;
                double[] rgGrad = grad;
                double[] rgGradOut = ComputeGradientD(rgInput, rgTarget, rgPosWt, rgWt, rgGrad);
                return rgGradOut;
            }
            else
            {
                float[] rgGrad = grad.Select(p => (float)Convert.ChangeType(p, typeof(float))).ToArray();
                float[] rgInput = input.Select(p => (float)Convert.ChangeType(p, typeof(float))).ToArray();
                float[] rgTarget = target.Select(p => (float)Convert.ChangeType(p, typeof(float))).ToArray();
                float[] rgPosWt = (m_pos_weight != null) ? m_pos_weight.Select(p => (float)Convert.ChangeType(p, typeof(float))).ToArray() : null;
                float[] rgWt = (m_weight != null) ? m_weight.Select(p => (float)Convert.ChangeType(p, typeof(float))).ToArray() : null;
                float[] rgGradOut = ComputeGradientF(rgInput, rgTarget, rgPosWt, rgWt, rgGrad);
                return rgGradOut.Select(p => (double)Convert.ChangeType(p, typeof(double))).ToArray();    
            }
        }

        public double[] ComputeGradientD(double[] input, double[] target, double[] pos_weight, double[] weight, double[] rgGrad)
        {

            // Initialize the gradient variable
            double[] gradient = new double[input.Length];

            // Loop over the input and target elements
            for (int i = 0; i < input.Length; i++)
            {
                // Compute the sigmoid of the input
                double input_sigmoid = 1 / (1 + Math.Exp(-input[i]));
                double dfY = target[i];

                if (pos_weight != null)
                {
                    dfY *= pos_weight[i / m_nN];
                }

                // Compute the derivative of the binary cross entropy loss with respect to the input
                gradient[i] = -dfY * (1 - input_sigmoid) + (1 - target[i]) * input_sigmoid;

                if (m_pos_weight != null)
                {
                    gradient[i] *= (pos_weight[i] * target[i] + (1 - target[i]));
                }
            }

            // Apply the weight if defined
            if (m_weight != null)
            {
                for (int i = 0; i < gradient.Length; i++)
                {
                    gradient[i] *= weight[i / m_nC];
                }
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
                // Do nothing
            }
            else if (m_reduction == "none")
            {
                return gradient;
            }
            else
            {
                throw new ArgumentException("Invalid reduction option");
            }

            // Apply chain rule.
            for (int i = 0; i < gradient.Length; i++)
            {
                gradient[i] *= rgGrad[0];
            }

            // Return the gradient as an array
            return gradient;
        }

        public float[] ComputeGradientF(float[] input, float[] target, float[] pos_weight, float[] weight, float[] rgGrad)
        {
            // Initialize the gradient variable
            float[] gradient = new float[input.Length];

            // Loop over the input and target elements
            for (int i = 0; i < input.Length; i++)
            {
                // Compute the sigmoid of the input
                float input_sigmoid = 1 / (1 + (float)Math.Exp(-input[i]));
                float fY = target[i];

                if (pos_weight != null)
                {
                    fY *= pos_weight[i / m_nN];
                }

                // Compute the derivative of the binary cross entropy loss with respect to the input
                gradient[i] = -fY * (1 - input_sigmoid) + (1 - target[i]) * input_sigmoid;

                if (m_pos_weight != null)
                {
                    gradient[i] *= (pos_weight[i] * target[i] + (1 - target[i]));
                }
            }

            // Apply the weight if defined
            if (m_weight != null)
            {
                for (int i = 0; i < gradient.Length; i++)
                {
                    gradient[i] *= weight[i / m_nC];
                }
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
                // Do nothing
            }
            else if (m_reduction == "none")
            {
                return gradient;
            }
            else
            {
                throw new ArgumentException("Invalid reduction option");
            }

            // Apply chain rule.
            for (int i = 0; i < gradient.Length; i++)
            {
                gradient[i] *= rgGrad[0];
            }

            // Return the gradient as an array
            return gradient;
        }

        // The weight, reduction and pos_weight parameters
        private T[] m_weight;
        private string m_reduction;
        private T[] m_pos_weight;
        private T[] m_target;
        private T[] m_input;
        private int m_nN;
        private int m_nC;
    }
}
