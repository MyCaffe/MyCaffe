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
using MyCaffe.layers.lnn;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.Tab;
using System.IO;
using System.Diagnostics;
using MyCaffe.gym.python;
using MyCaffe.param.tft;
using MyCaffe.solvers;
using MyCaffe.gym;
using System.Drawing;
using static System.Windows.Forms.AxHost;

/// <summary>
/// Testing the Cfc layer.
/// 
/// Cfc Layer - layer calculating closed form continuous time.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestLNN_CfcLayer
    {
        [TestMethod]
        public void TestForward()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestForward(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward_NoGate()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestForward(true);
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
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestBackward(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackward_NoGate()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestBackward(true);
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
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestGradient(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient_NoGate()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestGradient(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingBatch()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestTrainingBatch(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRealTime()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestTrainingRealTime(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ICfcLayerTest : ITest
    {
        void TestForward(bool bNoGate);
        void TestBackward(bool bNoGate);
        void TestGradient(bool bNoGate);
        void TestTrainingBatch(bool bNoGate);
        void TestTrainingRealTime(bool bNoGate);
    }

    class CfcLayerTest : TestBase
    {
        public CfcLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Cfc Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new CfcLayerTest<double>(strName, nDeviceID, engine);
            else
                return new CfcLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class CfcLayerTest<T> : TestEx<T>, ICfcLayerTest
    {
        public CfcLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        private string getTestDataPath(string strSubPath)
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\LNN\\test\\" + strSubPath + "\\iter_0\\";
        }

        private string getTestWtsPath(string strSubPath)
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\LNN\\test\\" + strSubPath + "\\iter_0\\weights\\";
        }

        private void verifyFileDownload(string strSubPath, string strFile)
        {
            string strPath = getTestDataPath(strSubPath);
            if (!File.Exists(strPath + strFile))
                throw new Exception("ERROR: You need to download the LNN test data by running the MyCaffe Test Application and selecting the 'Download Test Data | LNN' menu.");
        }

        private void load_weights(Layer<T> layer, string strPath)
        {
            int nNumLayes = layer.layer_param.cfc_unit_param.backbone_layers;
            int nIdx = 0;

            for (int i = 0; i < nNumLayes; i++)
            {
                layer.blobs[nIdx].LoadFromNumpy(strPath + "cfc.rnn_cell.bb_" + i.ToString() + ".weight.npy");
                nIdx++;
                layer.blobs[nIdx].LoadFromNumpy(strPath + "cfc.rnn_cell.bb_" + i.ToString() + ".bias.npy");
                nIdx++;
            }

            layer.blobs[nIdx].LoadFromNumpy(strPath + "cfc.rnn_cell.ff1.weight.npy");
            nIdx++;
            layer.blobs[nIdx].LoadFromNumpy(strPath + "cfc.rnn_cell.ff1.bias.npy");
            nIdx++;

            layer.blobs[nIdx].LoadFromNumpy(strPath + "cfc.rnn_cell.ff2.weight.npy");
            nIdx++;
            layer.blobs[nIdx].LoadFromNumpy(strPath + "cfc.rnn_cell.ff2.bias.npy");
            nIdx++;

            layer.blobs[nIdx].LoadFromNumpy(strPath + "cfc.rnn_cell.time_a.weight.npy");
            nIdx++;
            layer.blobs[nIdx].LoadFromNumpy(strPath + "cfc.rnn_cell.time_a.bias.npy");
            nIdx++;

            layer.blobs[nIdx].LoadFromNumpy(strPath + "cfc.rnn_cell.time_b.weight.npy");
            nIdx++;
            layer.blobs[nIdx].LoadFromNumpy(strPath + "cfc.rnn_cell.time_b.bias.npy");
            nIdx++;

            layer.blobs[nIdx].LoadFromNumpy(strPath + "cfc.fc.weight.npy");
            nIdx++;
            layer.blobs[nIdx].LoadFromNumpy(strPath + "cfc.fc.bias.npy");
            nIdx++;
        }

        /// <summary>
        /// Test Cfc forward
        /// </summary>
        /// <remarks>
        /// To generate the test data, run the following:
        /// Code: test_cfc.py
        /// Path: cfc
        /// </remarks>
        public void TestForward(bool bNoGate)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CFC);
            p.cfc_unit_param.input_size = 82;
            p.cfc_unit_param.hidden_size = 256;
            p.cfc_unit_param.backbone_activation = param.lnn.CfcUnitParameter.ACTIVATION.RELU;
            p.cfc_unit_param.backbone_dropout_ratio = 0.0f;
            p.cfc_unit_param.backbone_layers = 2;
            p.cfc_unit_param.backbone_units = 64;
            p.cfc_unit_param.no_gate = bNoGate;
            p.cfc_unit_param.minimal = false;
            p.cfc_param.input_features = 82;
            p.cfc_param.hidden_size = 256;
            p.cfc_param.output_features = 2;
            Layer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobTimeSpans = null;
            Blob<T> blobMask = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strSubPath = (bNoGate) ? "cfc_no_gate" : "cfc_gate";
            string strPath = getTestDataPath(strSubPath);
            string strPathWts = getTestWtsPath(strSubPath);

            verifyFileDownload(strSubPath, "cell_ff1.npy");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX = new Blob<T>(m_cuda, m_log);
                blobTimeSpans = new Blob<T>(m_cuda, m_log);
                blobMask = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CFC, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x.npy");
                blobTimeSpans.LoadFromNumpy(strPath + "timespans.npy");
                blobMask.LoadFromNumpy(strPath + "mask.npy");

                BottomVec.Clear();
                BottomVec.Add(blobX);
                BottomVec.Add(blobTimeSpans);
                BottomVec.Add(blobMask);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);
                load_weights(layer, strPathWts);

                layer.Forward(BottomVec, TopVec);

                blobYexp.LoadFromNumpy(strPath + "y.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp, blobWork, false, 1e-07), "The blobs do not match.");

                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (int i = 0; i < 100; i++)
                {
                    layer.Forward(BottomVec, TopVec);
                }

                sw.Stop();
                double dfTime = sw.Elapsed.TotalMilliseconds / 100;
                Trace.WriteLine("Ave time per forward = " + dfTime.ToString("N3") + " ms.");
            }
            finally
            {
                dispose(ref blobYexp);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobTimeSpans);
                dispose(ref blobMask);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }

        /// <summary>
        /// [WORK IN PROGRESS] Test Cfc backward
        /// </summary>
        /// <remarks>
        /// To generate the test data, run the following:
        /// Code: test_cfc.py
        /// Path: cfc
        /// </remarks>
        public void TestBackward(bool bNoGate)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CFC);
            p.cfc_unit_param.input_size = 82;
            p.cfc_unit_param.hidden_size = 256;
            p.cfc_unit_param.backbone_activation = param.lnn.CfcUnitParameter.ACTIVATION.RELU;
            p.cfc_unit_param.backbone_dropout_ratio = 0.0f;
            p.cfc_unit_param.backbone_layers = 2;
            p.cfc_unit_param.backbone_units = 64;
            p.cfc_unit_param.no_gate = bNoGate;
            p.cfc_unit_param.minimal = false;
            p.cfc_param.input_features = 82;
            p.cfc_param.hidden_size = 256;
            p.cfc_param.output_features = 2;
            Layer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobXgrad = null;
            Blob<T> blobTimeSpans = null;
            Blob<T> blobTimeSpansGrad = null;
            Blob<T> blobMask = null;
            Blob<T> blobMaskGrad = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strSubPath = (bNoGate) ? "cfc_no_gate" : "cfc_gate";
            string strPath = getTestDataPath(strSubPath);
            string strPathWts = getTestWtsPath(strSubPath);

            verifyFileDownload(strSubPath, "cell_ff1.npy");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX = new Blob<T>(m_cuda, m_log);
                blobXgrad = new Blob<T>(m_cuda, m_log);
                blobTimeSpans = new Blob<T>(m_cuda, m_log);
                blobTimeSpansGrad = new Blob<T>(m_cuda, m_log);
                blobMask = new Blob<T>(m_cuda, m_log);
                blobMaskGrad = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CFC, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x.npy");
                blobTimeSpans.LoadFromNumpy(strPath + "timespans.npy");
                blobMask.LoadFromNumpy(strPath + "mask.npy");

                BottomVec.Clear();
                BottomVec.Add(blobX);
                BottomVec.Add(blobTimeSpans);
                BottomVec.Add(blobMask);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);
                load_weights(layer, strPathWts);

                layer.Forward(BottomVec, TopVec);

                blobYexp.LoadFromNumpy(strPath + "y.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp, blobWork, false, 1e-07), "The blobs do not match.");

                //** BACKWARD **
                TopVec[0].LoadFromNumpy(strPath + "y.grad.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                blobXgrad.LoadFromNumpy(strPath + "x.grad.npy", true);
                m_log.CHECK(blobXgrad.Compare(blobX, blobWork, true), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobYexp);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobXgrad);
                dispose(ref blobTimeSpans);
                dispose(ref blobTimeSpansGrad);
                dispose(ref blobMask);
                dispose(ref blobMaskGrad);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }

        /// <summary>
        /// [WORK IN PROGRESS] Test Cfc gradient check
        /// </summary>
        public void TestGradient(bool bNoGate)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CFC);
            p.cfc_unit_param.input_size = 82;
            p.cfc_unit_param.hidden_size = 256;
            p.cfc_unit_param.backbone_activation = param.lnn.CfcUnitParameter.ACTIVATION.RELU;
            p.cfc_unit_param.backbone_dropout_ratio = 0.0f;
            p.cfc_unit_param.backbone_layers = 2;
            p.cfc_unit_param.backbone_units = 64;
            p.cfc_unit_param.no_gate = bNoGate;
            p.cfc_unit_param.minimal = false;
            p.cfc_param.input_features = 82;
            p.cfc_param.hidden_size = 256;
            p.cfc_param.output_features = 2;
            Layer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobTimeSpans = null;
            Blob<T> blobMask = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strSubPath = (bNoGate) ? "cfc_no_gate" : "cfc_gate";
            string strPath = getTestDataPath(strSubPath);
            string strPathWts = getTestWtsPath(strSubPath);

            verifyFileDownload(strSubPath, "cell_ff1.npy");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX = new Blob<T>(m_cuda, m_log);
                blobTimeSpans = new Blob<T>(m_cuda, m_log);
                blobMask = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CFC, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x.npy");
                blobTimeSpans.LoadFromNumpy(strPath + "timespans.npy");
                blobMask.LoadFromNumpy(strPath + "mask.npy");

                BottomVec.Clear();
                BottomVec.Add(blobX);
                BottomVec.Add(blobTimeSpans);
                BottomVec.Add(blobMask);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);
                load_weights(layer, strPathWts);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 0.01, 0.01);
                checker.CheckGradient(layer, BottomVec, TopVec, -1, 1, 0.01);
            }
            finally
            {
                dispose(ref blobYexp);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobTimeSpans);
                dispose(ref blobMask);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }

        /// <summary>
        /// Create a simple CFC model with MSE loss.
        /// </summary>
        /// <param name="nBatchSize">Specifies the batch size.</param>
        /// <param name="nInputSize">Specifies the input size (number of time steps)</param>
        /// <param name="bNoGate">Specifies whether no-gating is used.</param>
        /// <param name="nHiddenSize">Specifies the hidden size.</param>
        /// <param name="fDropout">Specifies dropout ratio.</param>
        /// <param name="nLayers">Specifies the number of backbone layers used.</param>
        /// <param name="nUnits">Specifies the number of backbone units used.</param>
        /// <param name="nOutputSize">Specifies the number of outputs.</param>
        /// <returns></returns>
        private string buildModel(int nBatchSize, int nInputSize, bool bNoGate, int nHiddenSize, float fDropout, int nLayers, int nUnits, int nOutputSize)
        {
            NetParameter p = new NetParameter();
            p.name = "cfc_net";

            //---------------------------------
            //  Data Temporal Input
            //---------------------------------
            LayerParameter data = new LayerParameter(LayerParameter.LayerType.INPUT);
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize, 1 }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize, 1 }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nOutputSize }));
            data.top.Add("x");
            data.top.Add("tt");
            data.top.Add("mask");
            data.top.Add("target");
            data.include.Add(new NetStateRule(Phase.TRAIN));
            p.layer.Add(data);

            data = new LayerParameter(LayerParameter.LayerType.INPUT);
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize, 1 }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize, 1 }));
            data.top.Add("x");
            data.top.Add("tt");
            data.top.Add("mask");
            data.include.Add(new NetStateRule(Phase.TEST));
            p.layer.Add(data);

            //---------------------------------
            //  CFC Layer (Closed form Continuous-time)
            //---------------------------------
            LayerParameter cfc = new LayerParameter(LayerParameter.LayerType.CFC);
            cfc.cfc_unit_param.input_size = nInputSize;
            cfc.cfc_unit_param.hidden_size = nHiddenSize;
            cfc.cfc_unit_param.backbone_activation = param.lnn.CfcUnitParameter.ACTIVATION.RELU;
            cfc.cfc_unit_param.backbone_dropout_ratio = fDropout;
            cfc.cfc_unit_param.backbone_layers = nLayers;
            cfc.cfc_unit_param.backbone_units = nUnits;
            cfc.cfc_unit_param.no_gate = bNoGate;
            cfc.cfc_unit_param.minimal = false;
            cfc.cfc_param.input_features = nInputSize;
            cfc.cfc_param.hidden_size = nHiddenSize;
            cfc.cfc_param.output_features = nOutputSize;
            cfc.bottom.Add("x");
            cfc.bottom.Add("tt");
            cfc.bottom.Add("mask");
            cfc.top.Add("x_hat");
            p.layer.Add(cfc);

            //---------------------------------
            //  MSE Loss
            //---------------------------------
            LayerParameter loss = new LayerParameter(LayerParameter.LayerType.MEAN_ERROR_LOSS, "loss");
            loss.mean_error_loss_param.axis = 1;
            loss.mean_error_loss_param.mean_error_type = MEAN_ERROR.MSE;
            loss.loss_weight.Add(1); // for loss
            loss.loss_param.normalization = LossParameter.NormalizationMode.NONE;
            loss.bottom.Add("x_hat");
            loss.bottom.Add("target");
            loss.top.Add("loss");
            loss.include.Add(new NetStateRule(Phase.TRAIN));
            p.layer.Add(loss);

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Create the solver using the ADAMW solver.
        /// </summary>
        /// <param name="fLearningRate">Specifies the learning rate.</param>
        /// <returns></returns>
        private string buildSolver(float fLearningRate)
        {
            SolverParameter solverParam = new SolverParameter();
            solverParam.base_lr = fLearningRate;
            solverParam.type = SolverParameter.SolverType.ADAMW;
            solverParam.test_initialization = false;
            solverParam.test_interval = 10000;
            solverParam.display = 10;
            solverParam.test_iter.Add(1);
            solverParam.weight_decay = 0;
            solverParam.momentum = 0.9;
            solverParam.momentum2 = 0.999;
            solverParam.adamw_decay = 0;
            solverParam.lr_policy = "fixed";

            return solverParam.ToProto("root").ToString();
        }

        /// <summary>
        /// Test training with batches of input data.
        /// </summary>
        /// <param name="bNoGate">Specifies whether the no-gate mode is used.</param>
        public void TestTrainingBatch(bool bNoGate)
        {
            int nBatchSize = 128;
            int nInputSize = 82;
            int nOutputSize = 1;
            int nHiddenSize = 256;
            int nBackboneLayers = 2;
            int nBackboneUnits = 64;
            string strSolver = buildSolver(0.01f);
            string strModel = buildModel(nBatchSize, nInputSize, false, nHiddenSize, 0.0f, nBackboneLayers, nBackboneUnits, nOutputSize);

            //---------------------------------------------------
            // Setup MyCaffe and load the model.
            //---------------------------------------------------
            m_log.EnableTrace = true;
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, new CancelEvent());

            mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);
            Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
            Solver<T> solver = mycaffe.GetInternalSolver();

            Blob<T> blobX = net.FindBlob("x");
            Blob<T> blobTt = net.FindBlob("tt");
            Blob<T> blobMask = net.FindBlob("mask");
            Blob<T> blobY = net.FindBlob("x_hat");

            // Setup the curve gym for the data.
            MyCaffePythonGym gym = new MyCaffePythonGym();
            Random random = new Random();

            // 0 = Sin, 1 = Cos, 2 = Random
            gym.Initialize("Curve", "CurveType=0");

            string strName = gym.Name;
            Assert.AreEqual(strName, "MyCaffe Curve");

            int[] rgActions = gym.Actions;
            Assert.AreEqual(rgActions.Length, 2);
            Assert.AreEqual(rgActions[0], 0);
            Assert.AreEqual(rgActions[1], 1);

            PropertySet propTrain = new PropertySet();
            propTrain.SetProperty("Training", "True");
            Stopwatch sw = new Stopwatch();
            sw.Start();

            float[] rgInputBatch = new float[nBatchSize * nInputSize];
            float[] rgTargetBatch = new float[nBatchSize * nOutputSize];
            float[] rgMaskBatch = new float[nBatchSize * nInputSize];
            float[] rgTtBatch = new float[nBatchSize * nInputSize];

            //---------------------------------------------------
            // Train the model
            //---------------------------------------------------
            for (int i = 0; i < 100; i++)
            {
                // Load a batch of data using data generated by the Gym.
                for (int k = 0; k < nBatchSize; k++)
                {
                    float fStart = (float)(random.NextDouble() * 2.0 * Math.PI);
                    propTrain.SetProperty("TrainingStart", fStart.ToString());
                    gym.Reset(propTrain);
                    CurrentState state1 = null;

                    for (int j = 0; j < nInputSize; j++)
                    {
                        state1 = gym.Step(0, 1, propTrain);
                    }

                    List<DataPoint> rgHistory = state1.GymState.History;

                    float[] rgInput1 = rgHistory.Select(p => p.Inputs[0]).ToArray();
                    float[] rgTimeStamps1 = rgHistory.Select(p => p.Time).ToArray();
                    float[] rgMask1 = rgHistory.Select(p => p.Mask[0]).ToArray();
                    float[] rgTarget1 = rgHistory.Select(p => p.Target).ToArray();

                    Array.Copy(rgInput1, 0, rgInputBatch, k * nInputSize, nInputSize);
                    Array.Copy(rgTarget1, 0, rgTargetBatch, k * nOutputSize, nOutputSize);
                    Array.Copy(rgMask1, 0, rgMaskBatch, k * nInputSize, nInputSize);
                    Array.Copy(rgTimeStamps1, 0, rgTtBatch, k * nInputSize, nInputSize);
                }

                blobX.mutable_cpu_data = convert(rgInputBatch);
                blobTt.mutable_cpu_data = convert(rgTtBatch);
                blobMask.mutable_cpu_data = convert(rgMaskBatch);
                blobY.mutable_cpu_data = convert(rgTargetBatch);

                // Run the forward and backward pass.
                net.Forward();
                net.Backward();

                // Run the solver to perform the weight update.
                solver.Step(1);

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    double dfPct = (double)i / 1000.0;
                    m_log.WriteLine("Training " + dfPct.ToString("P") + " complete.");
                    sw.Restart();
                }
            }

            //IXPersist<T> persist = new common.PersistCaffe<T>(m_log, false);
            //byte[] rgWts = net.SaveWeights(persist, false);

            //---------------------------------------------------
            // Run the trained model
            //---------------------------------------------------
            gym.OpenUi();

            float fPredictedY = 0;
            PropertySet propTest = new PropertySet();

            propTest.SetProperty("get_input_data", "True");
            CurrentState state = gym.Step(0, 1, propTest);
            propTest = new PropertySet();

            // Use the test net for running the model.
            Net<T> netTest = mycaffe.GetInternalNet(Phase.TEST);
            //netTest.LoadWeights(rgWts, persist);

            BlobCollection<T> colInputs = new BlobCollection<T>();
            Blob<T> blobX1 = mycaffe.CreateBlob("x");
            blobX1.Reshape(1, blobX.channels, 1, 1);

            Blob<T> blobTt1 = mycaffe.CreateBlob("tt");
            blobTt1.Reshape(1, blobTt.channels, 1, 1);

            Blob<T> blobMask1 = mycaffe.CreateBlob("mask");
            blobMask1.Reshape(1, blobMask.channels, 1, 1);

            colInputs.Add(blobX1);
            colInputs.Add(blobTt1);
            colInputs.Add(blobMask1);

            float[] rgInput = new float[nInputSize];
            float[] rgTimeSteps = new float[nInputSize];
            float[] rgMask = new float[nInputSize];

            for (int i = 0; i < 1000; i++)
            {
                List<DataPoint> rgHistory = state.GymState.History;

                if (rgHistory.Count >= nInputSize)
                {
                    // Load the input data.
                    for (int j = 0; j < nInputSize; j++)
                    {
                        int nIdx = rgHistory.Count - nInputSize + j;
                        rgInput[j] = rgHistory[nIdx].Inputs[0];
                        rgTimeSteps[j] = rgHistory[nIdx].Time;
                        rgMask[j] = rgHistory[nIdx].Mask[0];
                    }

                    blobX1.mutable_cpu_data = convert(rgInput);
                    blobTt1.mutable_cpu_data = convert(rgTimeSteps);
                    blobMask1.mutable_cpu_data = convert(rgMask);

                    // Run the forward pass.
                    double dfLoss;
                    BlobCollection<T> colPred = netTest.Forward(colInputs, out dfLoss, true);
                    float[] rgOutput = convertF(colPred[0].update_cpu_data());
                    fPredictedY = rgOutput[0];

                    propTest.SetProperty("override_prediction", fPredictedY.ToString());
                }

                state = gym.Step(0, 1, propTest);
            }

            gym.CloseUi();

            colInputs.Dispose();
            mycaffe.Dispose();
        }

        /// <summary>
        /// Test the training using real-time data (with batch = 1).
        /// </summary>
        /// <param name="bNoGate">Specifies the whether the no-gate mode is used.</param>
        public void TestTrainingRealTime(bool bNoGate)
        {
            int nBatchSize = 1;
            int nInputSize = 82;
            int nOutputSize = 1;
            int nHiddenSize = 256;
            int nBackboneLayers = 2;
            int nBackboneUnits = 64;
            string strSolver = buildSolver(0.01f);
            string strModel = buildModel(nBatchSize, nInputSize, false, nHiddenSize, 0.0f, nBackboneLayers, nBackboneUnits, nOutputSize);

            // Setup MyCaffe and load the model.
            m_log.EnableTrace = true;
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, new CancelEvent());

            mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);
            Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
            Solver<T> solver = mycaffe.GetInternalSolver();

            // Setup the curve gym for the data.
            MyCaffePythonGym gym = new MyCaffePythonGym();
            Random random = new Random();

            // 0 = Sin, 1 = Cos, 2 = Random
            gym.Initialize("Curve", "CurveType=0");

            // Run the trained model
            gym.OpenUi();

            float fPredictedY = 0;
            PropertySet propTest = new PropertySet();

            propTest.SetProperty("get_input_data", "True");
            CurrentState state = gym.Step(0, 1, propTest);

            propTest = new PropertySet();

            Blob<T> blobX = net.FindBlob("x");
            Blob<T> blobTt = net.FindBlob("tt");
            Blob<T> blobMask = net.FindBlob("mask");
            Blob<T> blobY = net.FindBlob("target");
            Blob<T> blobXhat = net.FindBlob("x_hat");

            float[] rgInput = new float[nInputSize];
            float[] rgTimeSteps = new float[nInputSize];
            float[] rgMask = new float[nInputSize];
            float[] rgTarget = new float[nOutputSize];

            for (int i = 0; i < 2000; i++)
            {
                List<DataPoint> rgHistory = state.GymState.History;

                if (rgHistory.Count >= nInputSize)
                {
                    for (int j = 0; j < nInputSize; j++)
                    {
                        int nIdx = rgHistory.Count - nInputSize + j;
                        rgInput[j] = rgHistory[nIdx].Inputs[0];
                        rgTimeSteps[j] = rgHistory[nIdx].Time;
                        rgMask[j] = rgHistory[nIdx].Mask[0];
                        rgTarget[0] = rgHistory[nIdx].Target;
                    }

                    blobX.mutable_cpu_data = convert(rgInput);
                    blobTt.mutable_cpu_data = convert(rgTimeSteps);
                    blobMask.mutable_cpu_data = convert(rgMask);
                    blobY.mutable_cpu_data = convert(rgTarget);

                    net.Forward();
                    net.Backward();

                    solver.Step(1);

                    float[] rgOutput = convertF(blobXhat.mutable_cpu_data);
                    fPredictedY = rgOutput[0];

                    propTest.SetProperty("override_prediction", fPredictedY.ToString());
                }

                state = gym.Step(0, 1, propTest);
            }

            gym.CloseUi();

            mycaffe.Dispose();
        }
    }
}
