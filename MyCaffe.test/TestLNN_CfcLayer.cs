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
using System.IO;
using System.Diagnostics;
using MyCaffe.gym.python;
using MyCaffe.param.tft;
using MyCaffe.solvers;
using MyCaffe.gym;
using System.Drawing;
using MyCaffe.param.lnn;
using System.Threading;
using SimpleGraphing;
using static MyCaffe.param.lnn.CfcParameter;
using System.Windows.Forms;
using MyCaffe.db.temporal;

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
        public void TestTrainingBatch_CFC()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestTrainingBatch(false, false, CfcParameter.CELL_TYPE.CFC);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRealTime_CFC()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    int nStepsForward = -1; // default = -1 for the present value.
                    int nFutureSteps = 1;  // default = 1 for the present value.
                    bool bEnableUI = false;
                    //int nStepsForward = 0; // default = -1 for the present value.
                    //int nFutureSteps = 25;  // default = 1 for the present value.
                    //bool bEnableUI = true;

                    t.TestTrainingRealTime(false, bEnableUI, CfcParameter.CELL_TYPE.CFC, nStepsForward, nFutureSteps);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingBatch_LTC()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestTrainingBatch(false, false, CfcParameter.CELL_TYPE.LTC);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRealTime_LTC()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestTrainingRealTime(false, false, CfcParameter.CELL_TYPE.LTC);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRealTimeMultiValue_CFC()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    int nStepsForward = 0; // default = -1 for the present value.
                    int nFutureSteps = 25;  // default = 1 for the present value.
                    bool bEnableUI = false;
                    int nCurveType = 0; // default = 0, SIN
                    bool bRecord = false;
                    CfcUnitParameter.ACTIVATION activation = CfcUnitParameter.ACTIVATION.TANH;

                    t.TestTrainingRealTimeMultiValue(false, bEnableUI, CfcParameter.CELL_TYPE.CFC, activation, nStepsForward, nFutureSteps, nCurveType, bRecord);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRealTimeFutureBatch_CFC()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    int nStepsForward = 0; // default = -1 for the present value.
                    int nFutureSteps = 25;  // default = 1 for the present value.
                    bool bEnableUI = false;
                    int nCurveType = 0; // default = 0, SIN
                    bool bRecord = false;
                    bool bShuffle = true;
                    bool bLiquidInference = true;
                    int nBatchIdxLock = -1; // Values >= 0 lock on that item within the dataset;
                    LayerParameter.LayerType layerType = LayerParameter.LayerType.INNERPRODUCT;
                    int nTrainIter = 12000;

                    t.TestTrainingRealTimeFutureBatch(false, bEnableUI, layerType, nStepsForward, nFutureSteps, nCurveType, bRecord, bShuffle, bLiquidInference, nBatchIdxLock, nTrainIter);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRealTime_Combo()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestTrainingRealTimeCombo(false, true, true, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainingRealTime_ComboNet()
        {
            CfcLayerTest test = new CfcLayerTest();

            try
            {
                foreach (ICfcLayerTest t in test.Tests)
                {
                    t.TestTrainingRealTimeComboNets(false, true, false, false);
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
        void TestTrainingBatch(bool bNoGate, bool bEnableUI, CfcParameter.CELL_TYPE cell_type);
        void TestTrainingRealTime(bool bNoGate, bool bEnableUI, CfcParameter.CELL_TYPE cell_type, int nStepsForward = -1, int nFutureSteps = 1, bool bRecord = false);
        void TestTrainingRealTimeMultiValue(bool bNoGate, bool bEnableUI, CfcParameter.CELL_TYPE cell_type, CfcUnitParameter.ACTIVATION activation, int nStepsForward = -1, int nFutureSteps = 1, int nCurveType = 0, bool bRecord = false);
        void TestTrainingRealTimeFutureBatch(bool bNoGate, bool bEnableUI, LayerParameter.LayerType layerType, int nStepsForward = -1, int nFutureSteps = 1, int nCurveType = 0, bool bRecord = false, bool bShuffle = false, bool bLiquidInference = true, int nBatchIdxLock = -1, int nTrainIter = 1000);
        void TestTrainingRealTimeCombo(bool bEnableUI, bool bEmphasizeCfcNoGateF, bool bEmphasizeCfcNoGateT, bool bEmphasizeLtc, bool bRecord = false);
        void TestTrainingRealTimeComboNets(bool bEnableUI, bool bEmphasizeCfc, bool bEmphasizeLstm, bool bEmphasizeLinear, bool bRecord = false);
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
        Random m_random = new Random();
        ManualResetEvent m_evtCancel = new ManualResetEvent(false);
        PlotCollection m_plotsLossCurve = new PlotCollection();
        List<Tuple<float[], float[], float[], float[]>> m_rgBatch = new List<Tuple<float[], float[], float[], float[]>>();
        string m_strLossCurveFile = "";
        int m_nLossCurveFiles = 0;
        int m_nBatchIdx = 0;
        bool m_bShuffleData = false;
        int m_nBatchIdxLock = -1;

        public CfcLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            m_evtCancel.Set();
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
            p.cfc_param.cell_type = param.lnn.CfcParameter.CELL_TYPE.CFC;
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
        /// Test Cfc backward
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
            p.cfc_param.cell_type = param.lnn.CfcParameter.CELL_TYPE.CFC;
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
        /// Test Cfc gradient check
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
        /// <param name="cell_type">Optionally, specifies the cell type (default = CFC)</param>
        /// <param name="activation">Optionally, specifies the activation type (default = RELU).</param>
        /// <param name="bFuturePrediction">Optionally, specifies future prediction (default = false).</param>
        /// <returns></returns>
        private string buildModel(int nBatchSize, int nInputSize, bool bNoGate, int nHiddenSize, float fDropout, int nLayers, int nUnits, int nOutputSize, CfcParameter.CELL_TYPE cell_type, CfcUnitParameter.ACTIVATION activation = CfcUnitParameter.ACTIVATION.RELU, bool bFuturePrediction = false)
        {
            NetParameter p = new NetParameter();
            p.name = "cfc_net";

            int nFutureInputSize = (bFuturePrediction) ? nOutputSize : 0;

            //---------------------------------
            //  Data Temporal Input
            //---------------------------------
            LayerParameter data = new LayerParameter(LayerParameter.LayerType.INPUT);
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize + nFutureInputSize, 1 }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize + nFutureInputSize }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize + nFutureInputSize, 1 }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nOutputSize }));
            data.top.Add("x");
            data.top.Add("tt");
            data.top.Add("mask");
            data.top.Add("target");
            data.include.Add(new NetStateRule(Phase.TRAIN));
            p.layer.Add(data);

            data = new LayerParameter(LayerParameter.LayerType.INPUT);
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize + nFutureInputSize, 1 }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize + nFutureInputSize }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize + nFutureInputSize, 1 }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nOutputSize }));
            data.top.Add("x");
            data.top.Add("tt");
            data.top.Add("mask");
            data.top.Add("target");
            data.include.Add(new NetStateRule(Phase.TEST));
            p.layer.Add(data);

            //---------------------------------
            //  CFC Layer (Closed form Continuous-time)
            //---------------------------------
            LayerParameter cfc = new LayerParameter(LayerParameter.LayerType.CFC);

            if (cell_type == CfcParameter.CELL_TYPE.LTC)
            {
                cfc.ltc_unit_param.input_size = nInputSize + nFutureInputSize;
                cfc.ltc_unit_param.hidden_size = nHiddenSize;
                cfc.ltc_unit_param.ode_unfolds = 6;
            }
            else
            {
                cfc.cfc_unit_param.input_size = nInputSize + nFutureInputSize;
                cfc.cfc_unit_param.hidden_size = nHiddenSize;
                cfc.cfc_unit_param.backbone_activation = activation;
                cfc.cfc_unit_param.backbone_dropout_ratio = fDropout;
                cfc.cfc_unit_param.backbone_layers = nLayers;
                cfc.cfc_unit_param.backbone_units = nUnits;
                cfc.cfc_unit_param.no_gate = bNoGate;
                cfc.cfc_unit_param.minimal = false;
            }

            cfc.cfc_param.input_features = nInputSize + nFutureInputSize;
            cfc.cfc_param.hidden_size = nHiddenSize;
            cfc.cfc_param.output_features = nOutputSize;
            cfc.cfc_param.cell_type = cell_type;
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
            //loss.include.Add(new NetStateRule(Phase.TRAIN));
            p.layer.Add(loss);

            return p.ToProto("root").ToString();
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
        /// <param name="layerType">Specifies the layer type: CFC, LSTM or INNERPRODUCT</param>
        /// <param name="bUseDataLayer">Specifies to use a DataTemporal layer instead of an input layer (default = false).</param>
        /// <param name="strDataSource">Optionally, specifies the data source (default = "").</param>
        /// <returns></returns>
        private string buildModel(int nBatchSize, int nInputSize, bool bNoGate, int nHiddenSize, float fDropout, int nLayers, int nUnits, int nOutputSize, LayerParameter.LayerType layerType, bool bUseDataLayer = false, string strDataSource = "")
        {
            NetParameter p = new NetParameter();
            p.name = "test_net";

            //---------------------------------
            //  Data Temporal Input
            //---------------------------------
            if (bUseDataLayer)
            {
                LayerParameter data = new LayerParameter(LayerParameter.LayerType.DATA_TEMPORAL);
                data.data_temporal_param.batch_size = (uint)nBatchSize;
                data.data_temporal_param.num_historical_steps = (uint)nInputSize;
                data.data_temporal_param.num_future_steps = (uint)nOutputSize;
                data.data_temporal_param.shuffle_data = true;
                data.data_temporal_param.source_type = DataTemporalParameter.SOURCE_TYPE.DIRECT;
                data.data_temporal_param.source = strDataSource;
                data.data_temporal_param.enable_debug_output = false;
                data.data_temporal_param.output_target_historical = true;
                data.data_temporal_param.output_time = true;
                data.data_temporal_param.output_mask = true;
                data.top.Add("ns");
                data.top.Add("cs");
                data.top.Add("x"); // nh
                data.top.Add("ch");
                data.top.Add("nf");
                data.top.Add("cf");
                data.top.Add("target");
                data.top.Add("trg_hist");
                data.top.Add("tt");
                data.top.Add("mask");
                data.include.Add(new NetStateRule(Phase.TRAIN));
                p.layer.Add(data);

                data = new LayerParameter(LayerParameter.LayerType.DATA_TEMPORAL);
                data.data_temporal_param.batch_size = (uint)nBatchSize;
                data.data_temporal_param.num_historical_steps = (uint)nInputSize;
                data.data_temporal_param.num_future_steps = (uint)nOutputSize;
                data.data_temporal_param.shuffle_data = true;
                data.data_temporal_param.source_type = DataTemporalParameter.SOURCE_TYPE.DIRECT;
                data.data_temporal_param.source = strDataSource;
                data.data_temporal_param.enable_debug_output = false;
                data.data_temporal_param.output_target_historical = true;
                data.data_temporal_param.output_time = true;
                data.data_temporal_param.output_mask = true;
                data.top.Add("ns");
                data.top.Add("cs");
                data.top.Add("x"); // nh
                data.top.Add("ch");
                data.top.Add("nf");
                data.top.Add("cf");
                data.top.Add("target");
                data.top.Add("trg_hist");
                data.top.Add("tt");
                data.top.Add("mask");
                data.include.Add(new NetStateRule(Phase.TEST));
                p.layer.Add(data);

                LayerParameter silence = new LayerParameter(LayerParameter.LayerType.SILENCE);
                silence.bottom.Add("ns");
                p.layer.Add(silence);

                silence = new LayerParameter(LayerParameter.LayerType.SILENCE);
                silence.bottom.Add("cs");
                p.layer.Add(silence);

                silence = new LayerParameter(LayerParameter.LayerType.SILENCE);
                silence.bottom.Add("ch");
                p.layer.Add(silence);

                silence = new LayerParameter(LayerParameter.LayerType.SILENCE);
                silence.bottom.Add("nf");
                p.layer.Add(silence);

                silence = new LayerParameter(LayerParameter.LayerType.SILENCE);
                silence.bottom.Add("cf");
                p.layer.Add(silence);
            }
            else
            {
                LayerParameter data = new LayerParameter(LayerParameter.LayerType.INPUT);
                data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize, 1, 1 }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize, 1, 1 }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nOutputSize }));
                data.top.Add("x");
                data.top.Add("tt");
                data.top.Add("mask");
                data.top.Add("target");
                data.include.Add(new NetStateRule(Phase.TRAIN));
                p.layer.Add(data);

                data = new LayerParameter(LayerParameter.LayerType.INPUT);
                data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize, 1, 1 }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nBatchSize, nInputSize, 1, 1 }));
                data.top.Add("x");
                data.top.Add("tt");
                data.top.Add("mask");
                data.include.Add(new NetStateRule(Phase.TEST));
                p.layer.Add(data);
            }

            //---------------------------------
            //  CFC Layer (Closed form Continuous-time)
            //---------------------------------

            switch (layerType)
            {
                case LayerParameter.LayerType.CFC:
                    LayerParameter cfc = new LayerParameter(LayerParameter.LayerType.CFC);
                    cfc.cfc_unit_param.input_size = nInputSize;
                    cfc.cfc_unit_param.hidden_size = nHiddenSize;
                    cfc.cfc_unit_param.backbone_activation = param.lnn.CfcUnitParameter.ACTIVATION.TANH;
                    cfc.cfc_unit_param.backbone_dropout_ratio = fDropout;
                    cfc.cfc_unit_param.backbone_layers = nLayers;
                    cfc.cfc_unit_param.backbone_units = nUnits;
                    cfc.cfc_unit_param.no_gate = bNoGate;
                    cfc.cfc_unit_param.minimal = false;
                    cfc.cfc_param.input_features = nInputSize;
                    cfc.cfc_param.hidden_size = nHiddenSize;
                    cfc.cfc_param.output_features = nOutputSize;
                    cfc.cfc_param.cell_type = CELL_TYPE.CFC;
                    cfc.bottom.Add("x");
                    cfc.bottom.Add("tt");
                    cfc.bottom.Add("mask");
                    cfc.top.Add("x_hat");
                    p.layer.Add(cfc);
                    break;

                case LayerParameter.LayerType.LSTM:
                    LayerParameter silence = new LayerParameter(LayerParameter.LayerType.SILENCE);
                    silence.bottom.Add("tt");
                    p.layer.Add(silence);

                    LayerParameter lstm1 = new LayerParameter(LayerParameter.LayerType.LSTM);
                    lstm1.recurrent_param.dropout_ratio = fDropout;
                    lstm1.recurrent_param.engine = EngineParameter.Engine.CUDNN;
                    lstm1.recurrent_param.use_cudnn_rnn8_if_supported = true;
                    lstm1.recurrent_param.batch_first = true;
                    lstm1.recurrent_param.num_layers = (uint)nLayers;
                    lstm1.recurrent_param.num_output = (uint)nHiddenSize;
                    lstm1.recurrent_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.5);
                    lstm1.recurrent_param.bias_filler = new FillerParameter("constant", 0);
                    lstm1.bottom.Add("x");
                    lstm1.bottom.Add("mask");
                    lstm1.top.Add("lstm1");
                    p.layer.Add(lstm1);

                    LayerParameter slice = new LayerParameter(LayerParameter.LayerType.SLICE);
                    slice.slice_param.axis = 1;
                    slice.slice_param.slice_point.Add((uint)(nInputSize - 1));
                    slice.bottom.Add("lstm1");
                    slice.top.Add("lstmx");
                    slice.top.Add("lstm");
                    p.layer.Add(slice);

                    LayerParameter silence_lstmx = new LayerParameter(LayerParameter.LayerType.SILENCE);
                    silence_lstmx.bottom.Add("lstmx");
                    p.layer.Add(silence_lstmx);

                    LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
                    ip1.inner_product_param.num_output = (uint)nOutputSize;
                    ip1.inner_product_param.axis = 1;
                    ip1.inner_product_param.bias_term = true;
                    ip1.bottom.Add("lstm");
                    ip1.top.Add("x_hat");
                    p.layer.Add(ip1);
                    break;

                case LayerParameter.LayerType.INNERPRODUCT:
                    LayerParameter silence_tt = new LayerParameter(LayerParameter.LayerType.SILENCE);
                    silence_tt.bottom.Add("tt");
                    p.layer.Add(silence_tt);
                    LayerParameter silence_mask = new LayerParameter(LayerParameter.LayerType.SILENCE);
                    silence_mask.bottom.Add("mask");
                    p.layer.Add(silence_mask);

                    LayerParameter ip = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
                    ip.inner_product_param.num_output = (uint)nOutputSize;
                    ip.inner_product_param.axis = 1;
                    ip.inner_product_param.bias_term = true;
                    ip.bottom.Add("x");
                    ip.top.Add("x_hat");
                    p.layer.Add(ip);
                    break;
            }


            //---------------------------------
            //  MSE Loss
            //---------------------------------
            LayerParameter loss = new LayerParameter(LayerParameter.LayerType.MEAN_ERROR_LOSS, "loss");
            loss.mean_error_loss_param.axis = 2;
            loss.mean_error_loss_param.mean_error_type = MEAN_ERROR.MSE;
            loss.loss_weight.Add(1); // for loss
            loss.loss_param.normalization = LossParameter.NormalizationMode.VALID;
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
        private string buildSolver(float fLearningRate, int nIterSize = 1, bool bVerbose = true)
        {
            SolverParameter solverParam = new SolverParameter();
            solverParam.base_lr = fLearningRate;
            solverParam.type = SolverParameter.SolverType.ADAMW;
            solverParam.test_initialization = false;
            solverParam.test_interval = 100000;
            solverParam.display = 10;
            solverParam.test_iter.Add(1);
            solverParam.weight_decay = 0.0;
            solverParam.momentum = 0.9;
            solverParam.momentum2 = 0.999;
            solverParam.adamw_decay = 0.01;
            solverParam.lr_policy = "fixed";
            solverParam.verbose_optimization_output = bVerbose;
            solverParam.iter_size = nIterSize;

            return solverParam.ToProto("root").ToString();
        }

        /// <summary>
        /// Test training with batches of input data.
        /// </summary>
        /// <param name="bNoGate">Specifies whether the no-gate mode is used.</param>
        /// <param name="bEnableUI">Specifies to turn on the UI display.</param>
        /// <param name="cell_type">Specifies the cell type.</param>
        public void TestTrainingBatch(bool bNoGate, bool bEnableUI, CfcParameter.CELL_TYPE cell_type)
        {
            int nBatchSize = (cell_type == CfcParameter.CELL_TYPE.CFC) ? 128 : (typeof(T) == typeof(float)) ? 8 : 4;
            int nInputSize = 82;
            int nOutputSize = 1;
            int nHiddenSize = 256;
            int nBackboneLayers = 2;
            int nBackboneUnits = 64;
            string strSolver = buildSolver(0.01f);
            string strModel = buildModel(nBatchSize, nInputSize, false, nHiddenSize, 0.0f, nBackboneLayers, nBackboneUnits, nOutputSize, cell_type);

            //---------------------------------------------------
            // Setup MyCaffe and load the model.
            //---------------------------------------------------
            m_log.EnableTrace = true;
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, new CancelEvent());
            BlobCollection<T> colInputs = new BlobCollection<T>();

            try
            {
                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);
                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Solver<T> solver = mycaffe.GetInternalSolver();

                Blob<T> blobX = net.FindBlob("x");
                Blob<T> blobTt = net.FindBlob("tt");
                Blob<T> blobMask = net.FindBlob("mask");
                Blob<T> blobY = net.FindBlob("x_hat");
                Blob<T> blobTarget = net.FindBlob("target");

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

                int nMax = 100;
                if (bEnableUI)
                    nMax = 1000;

                for (int i = 0; i < nMax; i++)
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

                        rgTargetBatch[k] = rgHistory[rgHistory.Count - 1].Target;
                        Array.Copy(rgInput1, 0, rgInputBatch, k * nInputSize, nInputSize);
                        Array.Copy(rgMask1, 0, rgMaskBatch, k * nInputSize, nInputSize);
                        Array.Copy(rgTimeStamps1, 0, rgTtBatch, k * nInputSize, nInputSize);
                    }

                    blobX.mutable_cpu_data = convert(rgInputBatch);
                    blobTt.mutable_cpu_data = convert(rgTtBatch);
                    blobMask.mutable_cpu_data = convert(rgMaskBatch);
                    blobTarget.mutable_cpu_data = convert(rgTargetBatch);

                    // Run the solver to perform the forward/backward and weight update.
                    solver.Step(1);

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        double dfPct = (double)i / nMax;
                        m_log.WriteLine("Training " + dfPct.ToString("P") + " complete.");
                        sw.Restart();
                    }
                }

                //---------------------------------------------------
                // Run the trained model
                //---------------------------------------------------
                if (bEnableUI)
                    gym.OpenUi();
                gym.Reset();

                PropertySet propTest = new PropertySet();
                CurrentState state = gym.Step(0, 1, propTest);

                // Use the test net for running the model.
                Net<T> netTest = mycaffe.GetInternalNet(Phase.TEST);

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

                for (int i = 0; i < nMax; i++)
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
                        float fPredictedY = rgOutput[0];

                        propTest.SetProperty("override_prediction", fPredictedY.ToString());
                    }

                    state = gym.Step(0, 1, propTest);
                }

                if (bEnableUI)
                    gym.CloseUi();
            }
            finally
            {
                if (colInputs != null)
                    colInputs.Dispose();

                if (mycaffe != null)
                    mycaffe.Dispose();
            }
        }

        /// <summary>
        /// Test the training using real-time data (with batch = 1).
        /// </summary>
        /// <param name="bNoGate">Specifies the whether the no-gate mode is used.</param>
        /// <param name="bEnableUI">Specifies to turn on the UI display.</param>
        /// <param name="cell_type">Specifies the cell type.</param>
        /// <param name="nFutureSteps">Optionally, specifies the number of steps forward into the future to predict (default = 0, the present)</param>
        /// <param name="nStepsForward">Optionally, specifies the number of future steps to predict (default = 1).  Note, nFutureSteps must be less than or equal to nStepsForward + 1.</param>
        /// <param name="bRecord">Optionally, specifies to record the run (default = false).</param>
        public void TestTrainingRealTime(bool bNoGate, bool bEnableUI, CfcParameter.CELL_TYPE cell_type, int nStepsForward = -1, int nFutureSteps = 1, bool bRecord = false)
        {
            int nBatchSize = 1;
            int nInputSize = 82;
            int nOutputSize = nFutureSteps;
            int nHiddenSize = 256;
            int nBackboneLayers = 2;
            int nBackboneUnits = 64;
            string strSolver = buildSolver(0.01f);
            string strModel = buildModel(nBatchSize, nInputSize, false, nHiddenSize, 0.0f, nBackboneLayers, nBackboneUnits, nOutputSize, cell_type);

            // Setup MyCaffe and load the model.
            m_log.EnableTrace = true;
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, new CancelEvent());

            try
            {
                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);
                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Solver<T> solver = mycaffe.GetInternalSolver();

                // Setup the curve gym for the data.
                MyCaffePythonGym gym = new MyCaffePythonGym();
                Random random = new Random();

                // 0 = Sin, 1 = Cos, 2 = Random
                gym.Initialize("Curve", "CurveType=0");

                // Run the trained model
                if (bEnableUI)
                    gym.OpenUi(bRecord);

                PropertySet propTest = new PropertySet();
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

                int nMax = 100;
                if (bEnableUI)
                    nMax = 2000;

                for (int i = 0; i < nMax; i++)
                {
                    List<DataPoint> rgHistory = state.GymState.History;

                    if (rgHistory.Count >= (nInputSize + nOutputSize + nStepsForward))
                    {
                        for (int j = 0; j < nInputSize; j++)
                        {
                            int nIdx = rgHistory.Count - (nInputSize + nOutputSize + nStepsForward) + j;
                            rgInput[j] = rgHistory[nIdx].Inputs[0];
                            rgTimeSteps[j] = rgHistory[nIdx].Time;
                            rgMask[j] = rgHistory[nIdx].Mask[0];
                        }

                        for (int j=nInputSize + nStepsForward; j<nInputSize + nStepsForward + nFutureSteps; j++)
                        {
                            int nSrcIdx = rgHistory.Count - (nInputSize + nStepsForward + nFutureSteps) + j;
                            int nDstIdx = j - (nInputSize + nStepsForward);

                            rgTarget[nDstIdx] = rgHistory[nSrcIdx].Target;
                        }

                        blobX.mutable_cpu_data = convert(rgInput);
                        blobTt.mutable_cpu_data = convert(rgTimeSteps);
                        blobMask.mutable_cpu_data = convert(rgMask);
                        blobY.mutable_cpu_data = convert(rgTarget);

                        // Performs forward, backward pass and applies weights.
                        solver.Step(1);

                        float[] rgOutput1 = convertF(blobXhat.mutable_cpu_data);

                        if (nStepsForward >= 0)
                        {
                            for (int j = 0; j < nInputSize; j++)
                            {
                                int nIdx = rgHistory.Count - nInputSize + j;
                                rgInput[j] = rgHistory[nIdx].Inputs[0];
                                rgTimeSteps[j] = rgHistory[nIdx].Time;
                                rgMask[j] = rgHistory[nIdx].Mask[0];
                            }

                            net.Forward();
                        }

                        float[] rgOutput = convertF(blobXhat.mutable_cpu_data);
                        float fPredictedY = rgOutput[rgOutput.Length - 1];

                        propTest.SetProperty("override_prediction", fPredictedY.ToString());
                        
                        if (nFutureSteps > 1)
                        {
                            propTest.SetProperty("override_future_predictions", nFutureSteps.ToString());
                            propTest.SetProperty("override_future_prediction_start", (nStepsForward + 1).ToString());

                            for (int j=0; j<nFutureSteps; j++)
                            {
                                propTest.SetProperty("override_future_prediction" + j.ToString(), rgOutput[j].ToString());
                            }
                        }
                    }

                    state = gym.Step(0, 1, propTest);
                }

                if (bEnableUI)
                    gym.CloseUi();
            }
            finally
            {
                if (mycaffe != null)
                    mycaffe.Dispose();
            }
        }

        /// <summary>
        /// Test the training using real-time combo data (with batch = 1).
        /// </summary>
        /// <param name="bEnableUI">Specifies to turn on the UI display.</param>
        /// <param name="bEmphasizeCfcNoGateF">Specifies to emphasize the CfcNoGate=F</param>
        /// <param name="bEmphasizeCfcNoGateT">Specifies to emphasize the CfcNoGate=T</param>
        /// <param name="bEmphasizeLtc">Specifies to emphasize the LTC.</param>
        /// <param name="bRecord">Optionally, specifies to record the run (default = false).</param>
        public void TestTrainingRealTimeCombo(bool bEnableUI, bool bEmphasizeCfcNoGateF, bool bEmphasizeCfcNoGateT, bool bEmphasizeLtc, bool bRecord = false)
        {
            if (m_evtCancel.WaitOne(0))
                return;

            int nBatchSize = 1;
            int nInputSize = 82;
            int nOutputSize = 1;
            int nHiddenSize = 256;
            int nBackboneLayers = 2;
            int nBackboneUnits = 64;
            string strSolver = buildSolver(0.01f);
            string strModelCfcNoGateF = buildModel(nBatchSize, nInputSize, false, nHiddenSize, 0.0f, nBackboneLayers, nBackboneUnits, nOutputSize, CfcParameter.CELL_TYPE.CFC);
            string strModelCfcNoGateT = buildModel(nBatchSize, nInputSize, true, nHiddenSize, 0.0f, nBackboneLayers, nBackboneUnits, nOutputSize, CfcParameter.CELL_TYPE.CFC);
            string strModelLtc = buildModel(nBatchSize, nInputSize, false, nHiddenSize, 0.0f, nBackboneLayers, nBackboneUnits, nOutputSize, CfcParameter.CELL_TYPE.LTC);

            m_log.EnableTrace = true;

            // Setup MyCaffe and load the model.
            MyCaffeOperation<T> mycaffeOp_cfcNoGateF = new MyCaffeOperation<T>("CFC No Gate (F)");
            MyCaffeOperation<T> mycaffeOp_cfcNoGateT = new MyCaffeOperation<T>("CFC No Gate (T)");
            MyCaffeOperation<T> mycaffeOp_ltc = new MyCaffeOperation<T>("LTC");

            try
            {
                EventWaitHandle evtGlobalCancel = new EventWaitHandle(false, EventResetMode.ManualReset, "__GRADIENT_CHECKER_CancelEvent__");

                if (!mycaffeOp_cfcNoGateF.Initialize(evtGlobalCancel, m_evtCancel, m_log, 0, strModelCfcNoGateF, strSolver, nInputSize, nOutputSize))
                    throw new Exception("Could not initialize the CFC No Gate (F) model!");

                if (!mycaffeOp_cfcNoGateT.Initialize(evtGlobalCancel, m_evtCancel, m_log, 0, strModelCfcNoGateT, strSolver, nInputSize, nOutputSize))
                    throw new Exception("Could not initialize the CFC No Gate (T) model!");

                if (!mycaffeOp_ltc.Initialize(evtGlobalCancel, m_evtCancel, m_log, 1, strModelLtc, strSolver, nInputSize, nOutputSize))
                    throw new Exception("Could not initialize the LTC model!");

                // Setup the curve gym for the data.
                MyCaffePythonGym gym = new MyCaffePythonGym();
                Random random = new Random();

                // 0 = Sin, 1 = Cos, 2 = Random
                gym.Initialize("Curve", "CurveType=0");

                // Run the trained model
                if (bEnableUI)
                    gym.OpenUi(bRecord);
               
                PropertySet propTest = new PropertySet();
                propTest.SetProperty("override_predictions", "3");
                propTest.SetProperty("override_prediction0", "0");
                propTest.SetProperty("override_prediction1", "0");
                propTest.SetProperty("override_prediction2", "0");
                propTest.SetProperty("override_prediction0_name", "CFC no_gate=F");
                propTest.SetProperty("override_prediction0_emphasize", bEmphasizeCfcNoGateF.ToString());
                propTest.SetProperty("override_prediction1_name", "CFC no_gate=T");
                propTest.SetProperty("override_prediction1_emphasize", bEmphasizeCfcNoGateT.ToString());
                propTest.SetProperty("override_prediction2_name", "LTC");
                propTest.SetProperty("override_prediction2_emphasize", bEmphasizeLtc.ToString());

                CurrentState state = gym.Step(0, 1, propTest);

                float[] rgInput = new float[nInputSize];
                float[] rgTimeSteps = new float[nInputSize];
                float[] rgMask = new float[nInputSize];
                float[] rgTarget = new float[nOutputSize];

                WaitHandle[] rgWait = new WaitHandle[3];

                CalculationArray caCfcNoGateF = new CalculationArray(200);
                CalculationArray caCfcNoGateT = new CalculationArray(200);
                CalculationArray caLtc = new CalculationArray(200);

                Stopwatch sw = new Stopwatch();
                sw.Start();

                int nMax = 100;
                if (bEnableUI)
                    nMax = 2000;

                for (int i = 0; i < nMax; i++)
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

                        rgWait[0] = mycaffeOp_cfcNoGateF.RunCycleAsync(rgInput, rgTimeSteps, rgMask, rgTarget);
                        rgWait[1] = mycaffeOp_cfcNoGateT.RunCycleAsync(rgInput, rgTimeSteps, rgMask, rgTarget);
                        rgWait[2] = mycaffeOp_ltc.RunCycleAsync(rgInput, rgTimeSteps, rgMask, rgTarget);

                        while (!WaitHandle.WaitAll(rgWait, 10))
                        {
                            Thread.Sleep(1);
                            if (m_evtCancel.WaitOne(i))
                                break;

                            if (evtGlobalCancel.WaitOne(0))
                                break;
                        }

                        if (m_evtCancel.WaitOne(0))
                            break;

                        if (evtGlobalCancel.WaitOne(0))
                            break;

                        caCfcNoGateF.Add(mycaffeOp_cfcNoGateF.TotalMilliseconds);
                        caCfcNoGateT.Add(mycaffeOp_cfcNoGateT.TotalMilliseconds);
                        caLtc.Add(mycaffeOp_ltc.TotalMilliseconds);

                        float fPredicted_cfcNoGate = mycaffeOp_cfcNoGateF.Output[0];
                        float fPredicted_cfcGate = mycaffeOp_cfcNoGateT.Output[0];
                        float fPredicted_ltc = mycaffeOp_ltc.Output[0];

                        propTest.SetProperty("override_prediction0", fPredicted_cfcNoGate.ToString());
                        propTest.SetProperty("override_prediction1", fPredicted_cfcGate.ToString());
                        propTest.SetProperty("override_prediction2", fPredicted_ltc.ToString());

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            sw.Restart();

                            m_log.WriteLine("CFC No Gate (F): " + caCfcNoGateF.Average.ToString("N3") + " ms.");
                            m_log.WriteLine("CFC No Gate (T): " + caCfcNoGateT.Average.ToString("N3") + " ms.");
                            m_log.WriteLine("LTC: " + caLtc.Average.ToString("N3") + " ms.");
                            m_log.WriteLine("---------------------------------");
                        }
                    }

                    state = gym.Step(0, 1, propTest);
                }

                if (bEnableUI)
                    gym.CloseUi();
            }
            finally
            {
                if (mycaffeOp_cfcNoGateF != null)
                    mycaffeOp_cfcNoGateF.Dispose();

                if (mycaffeOp_cfcNoGateT != null)
                    mycaffeOp_cfcNoGateT.Dispose();

                if (mycaffeOp_ltc != null)
                    mycaffeOp_ltc.Dispose();
            }
        }

        /// <summary>
        /// Test the training using real-time combo data (with batch = 1).
        /// </summary>
        /// <param name="bEnableUI">Specifies to turn on the UI display.</param>
        /// <param name="bEmphasizeCfc">Specifies the emphasize the Cfc model.</param>
        /// <param name="bEmphasizeLstm">Specifies to emphasize the LSTM model.</param>
        /// <param name="bEmphasizeLinear">Specifies to emphasize the Linear model.</param>
        /// <param name="bRecord">Optionally, specifies to record the run (default = false).</param>
        public void TestTrainingRealTimeComboNets(bool bEnableUI, bool bEmphasizeCfc, bool bEmphasizeLstm, bool bEmphasizeLinear, bool bRecord = false)
        {
            if (m_evtCancel.WaitOne(0))
                return;

            int nBatchSize = 1;
            int nInputSize = 82;
            int nOutputSize = 1;
            int nHiddenSize = 256;
            int nBackboneLayers = 2;
            int nBackboneUnits = 64;
            string strSolver = buildSolver(0.01f);
            string strModelCfc = buildModel(nBatchSize, nInputSize, false, nHiddenSize, 0.0f, nBackboneLayers, nBackboneUnits, nOutputSize, LayerParameter.LayerType.CFC);
            string strModelLstm = buildModel(nBatchSize, nInputSize, true, nHiddenSize, 0.0f, nBackboneLayers, nBackboneUnits, nOutputSize, LayerParameter.LayerType.LSTM);
            string strModelLinear = buildModel(nBatchSize, nInputSize, false, nHiddenSize, 0.0f, nBackboneLayers, nBackboneUnits, nOutputSize, LayerParameter.LayerType.INNERPRODUCT);

            m_log.EnableTrace = true;

            // Setup MyCaffe and load the model.
            MyCaffeOperation<T> mycaffeOp_cfc= new MyCaffeOperation<T>("CFC (TANH)");
            MyCaffeOperation<T> mycaffeOp_lstm = new MyCaffeOperation<T>("LSTM");
            MyCaffeOperation<T> mycaffeOp_linear = new MyCaffeOperation<T>("Linear");

            try
            {
                EventWaitHandle evtGlobalCancel = new EventWaitHandle(false, EventResetMode.ManualReset, "__GRADIENT_CHECKER_CancelEvent__");

                if (!mycaffeOp_cfc.Initialize(evtGlobalCancel, m_evtCancel, m_log, 0, strModelCfc, strSolver, nInputSize, nOutputSize, bEmphasizeCfc))
                    throw new Exception("Could not initialize the CFC (TANH) model!");

                if (!mycaffeOp_lstm.Initialize(evtGlobalCancel, m_evtCancel, m_log, 0, strModelLstm, strSolver, nInputSize, nOutputSize, bEmphasizeLstm))
                    throw new Exception("Could not initialize the LSTM model!");

                if (!mycaffeOp_linear.Initialize(evtGlobalCancel, m_evtCancel, m_log, 1, strModelLinear, strSolver, nInputSize, nOutputSize, bEmphasizeLinear))
                    throw new Exception("Could not initialize the Linear model!");

                // Setup the curve gym for the data.
                MyCaffePythonGym gym = new MyCaffePythonGym();
                Random random = new Random();

                // 0 = Sin, 1 = Cos, 2 = Random
                gym.Initialize("Curve", "CurveType=0");

                // Run the trained model
                if (bEnableUI)
                    gym.OpenUi(bRecord);

                PropertySet propTest = new PropertySet();
                propTest.SetProperty("override_predictions", "3");
                propTest.SetProperty("override_prediction0", "0");
                propTest.SetProperty("override_prediction1", "0");
                propTest.SetProperty("override_prediction2", "0");
                propTest.SetProperty("override_prediction0_name", "CFC (TANH)");
                propTest.SetProperty("override_prediction0_emphasize", bEmphasizeCfc.ToString());
                propTest.SetProperty("override_prediction1_name", "LSTM");
                propTest.SetProperty("override_prediction1_emphasize", bEmphasizeLstm.ToString());
                propTest.SetProperty("override_prediction2_name", "Linear");
                propTest.SetProperty("override_prediction2_emphasize", bEmphasizeLinear.ToString());

                CurrentState state = gym.Step(0, 1, propTest);

                float[] rgInput = new float[nInputSize];
                float[] rgTimeSteps = new float[nInputSize];
                float[] rgMask = new float[nInputSize];
                float[] rgTarget = new float[nOutputSize];

                WaitHandle[] rgWait = new WaitHandle[3];

                CalculationArray caCfc = new CalculationArray(200);
                CalculationArray caLstm = new CalculationArray(200);
                CalculationArray caLinear = new CalculationArray(200);

                Stopwatch sw = new Stopwatch();
                sw.Start();

                int nMax = 100;
                if (bEnableUI)
                    nMax = 2000;

                for (int i = 0; i < nMax; i++)
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

                        rgWait[0] = mycaffeOp_cfc.RunCycleAsync(rgInput, rgTimeSteps, rgMask, rgTarget);
                        rgWait[1] = mycaffeOp_lstm.RunCycleAsync(rgInput, rgTimeSteps, rgMask, rgTarget);
                        rgWait[2] = mycaffeOp_linear.RunCycleAsync(rgInput, rgTimeSteps, rgMask, rgTarget);

                        while (!WaitHandle.WaitAll(rgWait, 10))
                        {
                            Thread.Sleep(1);
                            if (m_evtCancel.WaitOne(i))
                                break;

                            if (evtGlobalCancel.WaitOne(0))
                                break;
                        }

                        if (m_evtCancel.WaitOne(0))
                            break;

                        if (evtGlobalCancel.WaitOne(0))
                            break;

                        caCfc.Add(mycaffeOp_cfc.TotalMilliseconds);
                        caLstm.Add(mycaffeOp_lstm.TotalMilliseconds);
                        caLinear.Add(mycaffeOp_linear.TotalMilliseconds);

                        float fPredicted_cfc = mycaffeOp_cfc.Output[0];
                        float fPredicted_lstm = mycaffeOp_lstm.Output[0];
                        float fPredicted_linear = mycaffeOp_linear.Output[0];

                        propTest.SetProperty("override_prediction0", fPredicted_cfc.ToString());
                        propTest.SetProperty("override_prediction1", fPredicted_lstm.ToString());
                        propTest.SetProperty("override_prediction2", fPredicted_linear.ToString());

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            sw.Restart();

                            m_log.WriteLine("CFC (TANH): " + caCfc.Average.ToString("N3") + " ms.");
                            m_log.WriteLine("LSTM: " + caLstm.Average.ToString("N3") + " ms.");
                            m_log.WriteLine("Linear: " + caLinear.Average.ToString("N3") + " ms.");
                            m_log.WriteLine("---------------------------------");
                        }
                    }

                    state = gym.Step(0, 1, propTest);
                }

                if (bEnableUI)
                    gym.CloseUi();
            }
            finally
            {
                if (mycaffeOp_cfc != null)
                    mycaffeOp_cfc.Dispose();

                if (mycaffeOp_lstm != null)
                    mycaffeOp_lstm.Dispose();

                if (mycaffeOp_linear != null)
                    mycaffeOp_linear.Dispose();
            }
        }

        /// <summary>
        /// Test the training using real-time data (with batch = 1).
        /// </summary>
        /// <param name="bNoGate">Specifies the whether the no-gate mode is used.</param>
        /// <param name="bEnableUI">Specifies to turn on the UI display.</param>
        /// <param name="cell_type">Specifies the cell type.</param>
        /// <param name="activation">Specifies the activation type to use (only applies when cell_type=CFC)</param>
        /// <param name="nFutureSteps">Optionally, specifies the number of steps forward into the future to predict (default = 0, the present)</param>
        /// <param name="nStepsForward">Optionally, specifies the number of future steps to predict (default = 1).  Note, nFutureSteps must be less than or equal to nStepsForward + 1.</param>
        /// <param name="nCurveType">Optionally, specifies the curve type where nCurveType = 0 for SIN, nCurveType = 1 for COS and nCurveType = 2 for RANDOM.</param>
        /// <param name="bRecord">Optionally, specifies to record the run (default = false).</param>
        public void TestTrainingRealTimeMultiValue(bool bNoGate, bool bEnableUI, CfcParameter.CELL_TYPE cell_type, CfcUnitParameter.ACTIVATION activation, int nStepsForward = -1, int nFutureSteps = 1, int nCurveType = 0, bool bRecord = false)
        {
            int nBatchSize = 1;
            int nInputSize = 82;
            int nOutputSize = nFutureSteps;
            int nHiddenSize = 256;
            int nBackboneLayers = 2;
            int nBackboneUnits = 64;
            string strSolver = buildSolver(0.01f);
            string strModel = buildModel(nBatchSize, nInputSize, false, nHiddenSize, 0.0f, nBackboneLayers, nBackboneUnits, nOutputSize, cell_type, activation);

            // Setup MyCaffe and load the model.
            m_log.EnableTrace = true;
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, new CancelEvent());

            try
            {
                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);
                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Solver<T> solver = mycaffe.GetInternalSolver();

                // Setup the curve gym for the data.
                MyCaffePythonGym gym = new MyCaffePythonGym();
                Random random = new Random();

                // 0 = Sin, 1 = Cos, 2 = Random
                gym.Initialize("Curve", "CurveType=" + nCurveType.ToString());

                // Run the trained model
                if (bEnableUI)
                    gym.OpenUi(bRecord);

                PropertySet propTest = new PropertySet();
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

                int nMax = 100;
                if (bEnableUI)
                    nMax = 2000;

                for (int i = 0; i < nMax; i++)
                {
                    List<DataPoint> rgHistory = state.GymState.History;

                    if (rgHistory.Count >= nInputSize + nOutputSize)
                    {
                        for (int j = 0; j < nInputSize; j++)
                        {
                            int nIdx = rgHistory.Count - (nInputSize + nOutputSize) + j;
                            rgInput[j] = rgHistory[nIdx].Inputs[0];
                            rgTimeSteps[j] = rgHistory[nIdx].Time;
                            rgMask[j] = rgHistory[nIdx].Mask[0];
                        }

                        for (int j = 0; j < nOutputSize; j++)
                        {
                            int nIdx = rgHistory.Count - nOutputSize + j;
                            rgTarget[j] = rgHistory[nIdx].Target;
                        }

                        blobX.mutable_cpu_data = convert(rgInput);
                        blobTt.mutable_cpu_data = convert(rgTimeSteps);
                        blobMask.mutable_cpu_data = convert(rgMask);
                        blobY.mutable_cpu_data = convert(rgTarget);

                        // Performs forward, backward pass and applies weights.
                        solver.Step(1);

                        float[] rgOutput1 = convertF(blobXhat.mutable_cpu_data);
                        float fPredictedY = rgOutput1[rgOutput1.Length - 1];

                        propTest.SetProperty("override_prediction", fPredictedY.ToString());

                        if (rgOutput1.Length > 1)
                        {
                            propTest.SetProperty("override_future_prediction_box_offset", (-nOutputSize).ToString());
                            propTest.SetProperty("override_future_predictions", rgOutput1.Length.ToString());
                            propTest.SetProperty("override_future_predictions_start", (-nOutputSize + 1).ToString());

                            for (int j = 0; j < rgOutput1.Length; j++)
                            {
                                propTest.SetProperty("override_future_prediction" + j.ToString(), rgOutput1[j].ToString());
                            }
                        }
                    }

                    state = gym.Step(0, 1, propTest);
                }

                if (bEnableUI)
                    gym.CloseUi();
            }
            finally
            {
                if (mycaffe != null)
                    mycaffe.Dispose();
            }
        }

        private Tuple<DatasetDescriptor, MyCaffeTemporalDatabase, PlotCollectionSet> loadTemporalDatabaseSet(SettingsCaffe s, bool bNormalizedData, int nHistSteps, int nFutureSteps, int nCurveType)
        {
            MyCaffePythonGym gym = new MyCaffePythonGym();

            // 0 = Sin, 1 = Cos, 2 = Random
            gym.Initialize("Curve", "CurveType=" + nCurveType.ToString());

            PropertySet propTest = new PropertySet();

            propTest.SetProperty("Training", "True");
            gym.Reset(propTest);
            CurrentState state = gym.Step(0, 1, propTest);

            // Create sine curve data.
            PlotCollection plots = new PlotCollection("SineCurve");

            plots.SetParameter("StreamCount", 1);
            plots.SetParameter("StreamTargetIdx", 0); // Specifies target index.
            plots.SetParameter("StreamTargetOverlapsNum", 1); // Keep target ans inputs separate.
            plots.ParametersEx.Add("Stream0", "Input");

            DateTime dt = DateTime.Now - TimeSpan.FromSeconds(1000);

            for (int i = 0; i < 360; i++)
            {
                state = gym.Step(0, 1, propTest);

                List<DataPoint> rgHistory = state.GymState.History;
                DataPoint pt = rgHistory[rgHistory.Count - 1];

                float[] rgf = new float[] { pt.Target };
                Plot plot = new Plot(dt.ToFileTime(), rgf);
                plot.Tag = dt;
                plots.Add(plot);
                dt += TimeSpan.FromSeconds(1);
            }

            PlotCollectionSet set = new PlotCollectionSet();
            set.Add(plots);
            set.Add(plots);
            set.Add(plots);
            set.Add(plots);
            set.Add(plots);

            // Create in-memory database.
            PropertySet prop = new PropertySet();
            prop.SetProperty("NormalizedData", bNormalizedData.ToString());
            prop.SetProperty("HistoricalSteps", nHistSteps.ToString());
            prop.SetProperty("FutureSteps", nFutureSteps.ToString());
            MyCaffeTemporalDatabase db = new MyCaffeTemporalDatabase(m_log, prop);

            // Create simple, single direct stream.
            Tuple<DatasetDescriptor, int[], int[]> dsd = db.CreateSimpleDirectStream("Direct", "SineCurve", s, prop, set);

            return new Tuple<DatasetDescriptor, MyCaffeTemporalDatabase, PlotCollectionSet>(dsd.Item1, db, set);
        }

        private Tuple<DatasetDescriptor, MyCaffeTemporalDatabase, PlotCollection> loadTemporalDatabasePlots(SettingsCaffe s, bool bNormalizedData, int nHistSteps, int nFutureSteps, int nCurveType)
        {
            MyCaffePythonGym gym = new MyCaffePythonGym();

            // 0 = Sin, 1 = Cos, 2 = Random
            gym.Initialize("Curve", "CurveType=" + nCurveType.ToString());

            PropertySet propTest = new PropertySet();

            propTest.SetProperty("Training", "True");
            gym.Reset(propTest);
            CurrentState state = gym.Step(0, 1, propTest);

            // Create sine curve data.
            PlotCollection plots = new PlotCollection("SineCurve");

            plots.SetParameter("StreamCount", 1);
            plots.SetParameter("StreamTargetIdx", 0); // Specifies target index.
            plots.SetParameter("StreamTargetOverlapsNum", 1); // Keep target ans inputs separate.
            plots.ParametersEx.Add("Stream0", "Input");

            DateTime dt = DateTime.Now - TimeSpan.FromSeconds(1000);

            for (int i = 0; i < 1000; i++)
            {
                state = gym.Step(0, 1, propTest);

                List<DataPoint> rgHistory = state.GymState.History;
                DataPoint pt = rgHistory[rgHistory.Count - 1];

                float[] rgf = new float[] { pt.Target };
                Plot plot = new Plot(dt.ToFileTime(), rgf);
                plot.Tag = dt;
                plots.Add(plot);
                dt += TimeSpan.FromSeconds(1);
            }

            // Create in-memory database.
            PropertySet prop = new PropertySet();
            prop.SetProperty("NormalizedData", bNormalizedData.ToString());
            prop.SetProperty("HistoricalSteps", nHistSteps.ToString());
            prop.SetProperty("FutureSteps", nFutureSteps.ToString());
            MyCaffeTemporalDatabase db = new MyCaffeTemporalDatabase(m_log, prop);

            // Create simple, single direct stream.
            Tuple<DatasetDescriptor, int, int> dsd = db.CreateSimpleDirectStream("Direct", "SineCurve", s, prop, plots);

            return new Tuple<DatasetDescriptor, MyCaffeTemporalDatabase, PlotCollection>(dsd.Item1, db, plots);
        }

        /// <summary>
        /// Test the training using real-time data to test prediction of unseen future values (with batch = 16).
        /// </summary>
        /// <param name="bNoGate">Specifies the whether the no-gate mode is used.</param>
        /// <param name="bEnableUI">Specifies to turn on the UI display.</param>
        /// <param name="layerType">Specifies the type of network to use.</param>
        /// <param name="nFutureSteps">Optionally, specifies the number of steps forward into the future to predict (default = 0, the present)</param>
        /// <param name="nStepsForward">Optionally, specifies the number of future steps to predict (default = 1).  Note, nFutureSteps must be less than or equal to nStepsForward + 1.</param>
        /// <param name="nCurveType">Optionally, specifies the curve type where nCurveType = 0 for SIN, nCurveType = 1 for COS and nCurveType = 2 for RANDOM.</param>
        /// <param name="bRecord">Optionally, specifies to record the run (default = false).</param>
        /// <param name="bShuffle">Optionally, specifies to shuffle the data loaded into each batch (default = false).</param>
        /// <param name="bLiquidInference">Optionally, specifies to enable the liquid inferencing where a single step of learning occurs at each inference step (default = true).</param>
        /// <remarks>WORK IN PROGRESS</remarks>
        public void TestTrainingRealTimeFutureBatch(bool bNoGate, bool bEnableUI, LayerParameter.LayerType layerType, int nStepsForward = -1, int nFutureSteps = 1, int nCurveType = 0, bool bRecord = false, bool bShuffle = false, bool bLiquidInference = true, int nBatchIdxLock = -1, int nTrainIter = 1000)
        {
            int nBatchSize = 64;
            int nInputSize = 82;
            int nOutputSize = nFutureSteps;
            int nHiddenSize = 256;
            int nBackboneLayers = 1;
            int nBackboneUnits = 64;
            string strSolver = buildSolver(0.01f, 1, false);
            string strModel = buildModel(nBatchSize, nInputSize, false, nHiddenSize, 0.0f, nBackboneLayers, nBackboneUnits, nOutputSize, layerType, true, "Direct");

            m_bShuffleData = bShuffle;
            m_nBatchIdxLock = nBatchIdxLock;

            // Setup MyCaffe and load the model.
            m_log.EnableTrace = true;
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<T> mycaffe = new MyCaffeControl<T>(s, m_log, new CancelEvent());

            try
            {
                Tuple<DatasetDescriptor, MyCaffeTemporalDatabase, PlotCollection> db1 = loadTemporalDatabasePlots(s, false, nInputSize, nFutureSteps, nCurveType);
                DatasetDescriptor ds = db1.Item1;
                MyCaffeTemporalDatabase db = db1.Item2;
                PlotCollection plots = db1.Item3;

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false, null, null, false, db);
                mycaffe.OnTrainingIteration += Mycaffe_OnTrainingIteration;
                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Solver<T> solver = mycaffe.GetInternalSolver();

                // Setup the curve gym for the data.
                MyCaffePythonGym gym = new MyCaffePythonGym();
                Random random = new Random();

                // 0 = Sin, 1 = Cos, 2 = Random
                gym.Initialize("Curve", "CurveType=" + nCurveType.ToString());

                PropertySet propTest = new PropertySet();

                propTest.SetProperty("Training", "True");
                gym.Reset(propTest);
                CurrentState state = gym.Step(0, 1, propTest);

                //-------------------------------------------------------------
                // Pre-train the model.
                //-------------------------------------------------------------
                if (nTrainIter > 0)
                    mycaffe.Train(nTrainIter);

                // Display the loss curve then the gym.
                if (bEnableUI)
                {
                    if (!string.IsNullOrEmpty(m_strLossCurveFile))
                    {
                        Process process = new Process();
                        process.StartInfo = new ProcessStartInfo(m_strLossCurveFile);
                        process.Start();
                    }

                    // Show the user interface.
                    gym.OpenUi(bRecord);
                }


                //-------------------------------------------------------------
                // Next run the trained model to predict the future values
                // using a pseudo inferencing.  Note when using a curve
                // Gym like the Sine curve, the values will have been 'seen'
                // by the model during training for the same curve is created
                // each time the Gym runs.
                //-------------------------------------------------------------
                int nMax = 100;
                if (bEnableUI)
                    nMax = 2000;

                Blob<T> blobX = net.FindBlob("x");
                Blob<T> blobTt = net.FindBlob("tt");
                Blob<T> blobMask = net.FindBlob("mask");
                Blob<T> blobY = net.FindBlob("target");
                Blob<T> blobXhat = net.FindBlob("x_hat");

                // Reshape the blobs and net to a single batch.
                // Note we are using the training net here on purpose.
                net.layers[0].layer_param.data_temporal_param.batch_size = 1;
                if (!bLiquidInference)
                    net.Reshape();

                propTest.SetProperty("status_text", "running 'pseudo' inference...");
                propTest.SetProperty("Training", "False");
                propTest.SetProperty("override_future_predictions", nOutputSize.ToString());
                propTest.SetProperty("override_future_prediction_box_offset", "0");
                propTest.SetProperty("override_future_predictions_start", "0");
                propTest.SetProperty("override_future_predictions_name", "pred");

                gym.Reset(propTest);
                state = gym.Step(0, 1, propTest);

                List<Tuple<List<float>, List<float>>> rgInputQueue = new List<Tuple<List<float>, List<float>>>();
                List<float> rgInput1 = new List<float>();
                List<float> rgTimeSteps1 = new List<float>();
                List<float> rgMask1 = new List<float>();
                List<float> rgTarget1 = new List<float>();
                float[] rgOutput1 = null;
                float fPredictedY = 0;

                // Disable the loadBatch used in training.
                solver.OnStart -= Solver_OnStart;
                int nIdx = 0;

                List<DataPoint> rgHistory = state.GymState.History;

                for (int i = 0; i < nMax; i++)
                {
                    if (rgHistory.Count > nInputSize)
                    {
                        // Load the initial input from the history.
                        if (rgInput1.Count < nInputSize)
                        {
                            for (int j = rgInput1.Count; j < nInputSize; j++)
                            {
                                nIdx = rgHistory.Count - nInputSize + j;

                                rgInput1.Add(rgHistory[nIdx].Target);
                                rgTimeSteps1.Add(rgHistory[nIdx].Time);
                                rgMask1.Add(rgHistory[nIdx].Mask[0]);
                            }

                            List<float> rgInputFull = new List<float>(rgInput1);
                            List<float> rgTimeStepsFull = new List<float>(rgTimeSteps1);
                            List<float> rgMaskFull = new List<float>(rgMask1);
                            float fTimeStep = rgTimeStepsFull[rgTimeStepsFull.Count - 1];
                            float fTimeStepInc = rgTimeStepsFull[rgTimeStepsFull.Count - 1] - rgTimeStepsFull[rgTimeStepsFull.Count - 2];

                            blobX.mutable_cpu_data = convert(rgInputFull.ToArray());
                            blobTt.mutable_cpu_data = convert(rgTimeStepsFull.ToArray());
                            blobMask.mutable_cpu_data = convert(rgMaskFull.ToArray());
                        }

                        // Perform forward pass only for inferencing, skipping the data layer
                        // so that the blob values set above are used.
                        net.ForwardFromTo(1);

                        // Get the predicted output.
                        rgOutput1 = convertF(blobXhat.mutable_cpu_data);
                        fPredictedY = rgOutput1[0];

                        propTest.SetProperty("override_prediction", fPredictedY.ToString());

                        if (rgOutput1.Length > 1)
                        {
                            for (int j = 0; j < nOutputSize; j++)
                            {
                                propTest.SetProperty("override_future_prediction" + j.ToString(), rgOutput1[j].ToString());
                            }
                        }
                    }

                    // Step to the next step in the curve for the next cycle.
                    state = gym.Step(0, 1, propTest);
                    rgHistory = state.GymState.History;

                    // Update the input with the next step, and the
                    // target with the predicted values, then replace
                    // the first target value with the new input value
                    // from this step.
                    if (rgInput1.Count == nInputSize && rgHistory.Count > nInputSize)
                    {
                        // Remove the old input, time and mask values.
                        rgInput1.RemoveAt(0);
                        rgTimeSteps1.RemoveAt(0);
                        rgMask1.RemoveAt(0);

                        // Add the new input, time step, mask and target.
                        rgInput1.Add(rgHistory[rgHistory.Count - 1].Target);
                        rgTimeSteps1.Add(rgHistory[rgHistory.Count - 1].Time);
                        rgMask1.Add(rgHistory[rgHistory.Count - 1].Mask[0]);

                        // Set the target to all of the predicted values.
                        rgTarget1.Add(rgHistory[rgHistory.Count - 1].Target);
                        rgInputQueue.Add(new Tuple<List<float>, List<float>>(new List<float>(rgInput1), new List<float>(rgTimeSteps1)));

                        // Once the target fills up, perform a single training cycle.
                        if (bLiquidInference && rgTarget1.Count == nOutputSize)
                        {
                            blobX.mutable_cpu_data = convert(rgInputQueue[0].Item1.ToArray());
                            blobTt.mutable_cpu_data = convert(rgInputQueue[0].Item2.ToArray());
                            blobY.mutable_cpu_data = convert(rgTarget1.ToArray());

                            // And perform a single forward/backward/update pass.
                            solver.Step(1);

                            rgInputQueue.RemoveAt(0);
                            rgTarget1.RemoveAt(0);
                        }

                        // Set the blob data to the current input.
                        blobX.mutable_cpu_data = convert(rgInput1.ToArray());
                        blobTt.mutable_cpu_data = convert(rgTimeSteps1.ToArray());
                        blobMask.mutable_cpu_data = convert(rgMask1.ToArray());
                    }
                }

                if (bEnableUI)
                    gym.CloseUi();
            }
            finally
            {
                if (mycaffe != null)
                    mycaffe.Dispose();
            }
        }

        private void debug(List<Tuple<float[], float[], float[], float[]>> rgData, int nBatchSize = 64, int nIdx = -1)
        {
            string strLossCurvePath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData) + "\\MyCaffe\\test\\images\\debug\\";
            if (!Directory.Exists(strLossCurvePath))
                Directory.CreateDirectory(strLossCurvePath);

            if (nIdx >= 0)
            {
                int nBatchStartIdx = (nIdx * nBatchSize) % rgData.Count;

                for (int i=0; i < nBatchSize; i++)
                {
                    save(strLossCurvePath, rgData, nBatchStartIdx);
                    nBatchStartIdx++;
                    if (nBatchStartIdx >= rgData.Count)
                        nBatchStartIdx = 0;
                }
            }
            else
            {
                for (int i = 0; i < rgData.Count; i++)
                {
                    save(strLossCurvePath, rgData, i);
                }
            }
        }

        private void save(string strLossCurvePath, List<Tuple<float[], float[], float[], float[]>> rgData, int nIdx)
        {
            PlotCollection plot1 = new PlotCollection("Input");
            PlotCollection plot2 = new PlotCollection("TimeSteps");
            PlotCollection plot3 = new PlotCollection("Mask");
            PlotCollection plot4 = new PlotCollection("Target");

            for (int j = 0; j < rgData[nIdx].Item1.Length; j++)
            {
                plot1.Add(new Plot(j, rgData[nIdx].Item1[j]));
                plot2.Add(new Plot(j, rgData[nIdx].Item2[j]));
                plot3.Add(new Plot(j, rgData[nIdx].Item3[j]));

                if (j < rgData[nIdx].Item4.Length)
                    plot4.Add(new Plot(j, rgData[nIdx].Item4[j]));
            }

            save(strLossCurvePath, nIdx, "input", plot1);
            save(strLossCurvePath, nIdx, "timesteps", plot2);
            save(strLossCurvePath, nIdx, "mask", plot3);
            save(strLossCurvePath, nIdx, "target", plot4);
        }

        private void save(string strPath, int nIdx, string strName, PlotCollection plot)
        {
            plot.Name = strName;
            Image img = SimpleGraphingControl.QuickRender(plot, 600, 600);
            string strFile = strPath + nIdx.ToString() + "_" + strName + ".png";
            img.Save(strFile);
        }

        private void loadBatch(Solver<T> solver)
        {
            Net<T> net = solver.net;

            Blob<T> blobX = net.FindBlob("x");
            Blob<T> blobTt = net.FindBlob("tt");
            Blob<T> blobMask = net.FindBlob("mask");
            Blob<T> blobY = net.FindBlob("target");

            int nInputSize = blobX.count(1);
            int nOutputSize = blobY.count(1);
            int nBatchSize = blobX.num;

            float[] rgInputBatch = new float[nInputSize * nBatchSize];
            float[] rgTimeStepsBatch = new float[nInputSize * nBatchSize];
            float[] rgMaskBatch = new float[nInputSize * nBatchSize];
            float[] rgTargetBatch = new float[nOutputSize * nBatchSize];

            for (int b = 0; b < nBatchSize; b++)
            {
                int nIdx = (m_nBatchIdxLock >= 0) ? m_nBatchIdxLock : m_nBatchIdx;

                if (m_nBatchIdxLock < 0 && m_bShuffleData)
                    nIdx = m_random.Next(0, m_rgBatch.Count - (nInputSize + nOutputSize));

                Array.Copy(m_rgBatch[nIdx].Item1, 0, rgInputBatch, b * nInputSize, nInputSize);
                Array.Copy(m_rgBatch[nIdx].Item2, 0, rgTimeStepsBatch, b * nInputSize, nInputSize);
                Array.Copy(m_rgBatch[nIdx].Item3, 0, rgMaskBatch, b * nInputSize, nInputSize);
                Array.Copy(m_rgBatch[nIdx].Item4, 0, rgTargetBatch, b * nOutputSize, nOutputSize);
                m_nBatchIdx++;

                if (m_nBatchIdx + nBatchSize >= m_rgBatch.Count)
                    m_nBatchIdx = 0;
            }

            blobX.mutable_cpu_data = convert(rgInputBatch);
            blobTt.mutable_cpu_data = convert(rgTimeStepsBatch);
            blobMask.mutable_cpu_data = convert(rgMaskBatch);
            blobY.mutable_cpu_data = convert(rgTargetBatch);
        }

        private void saveLoss(int nIteration, double dfLoss)
        {
            Trace.WriteLine("Iteration " + nIteration.ToString() + ", Smoothed Loss = " + dfLoss.ToString());

            Plot plot = new Plot(nIteration, dfLoss);
            m_plotsLossCurve.Add(plot);

            if (m_plotsLossCurve.Count > 100)
            {
                // Save the loss curve.
                m_plotsLossCurve.Name = "Loss Curve (" + nIteration.ToString() + " iterations)";
                Image imgLoss = SimpleGraphingControl.QuickRender(m_plotsLossCurve, 1200, 600);
                string strLossCurvePath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData) + "\\MyCaffe\\test\\images\\";
                if (!Directory.Exists(strLossCurvePath))
                    Directory.CreateDirectory(strLossCurvePath);

                string strLossCurveFile = strLossCurvePath + "loss_curve";
                if (m_nLossCurveFiles > 0)
                    strLossCurveFile += "_" + m_nLossCurveFiles.ToString();
                strLossCurveFile += ".png";

                imgLoss.Save(strLossCurveFile);

                if (string.IsNullOrEmpty(m_strLossCurveFile))
                    m_strLossCurveFile = strLossCurveFile;

                m_plotsLossCurve.Clear();
                m_nLossCurveFiles++;
            }
        }

        private void Solver_OnStart(object sender, EventArgs e)
        {
            loadBatch(sender as Solver<T>);
        }

        private void Mycaffe_OnTrainingIteration(object sender, TrainingIterationArgs<T> e)
        {
            saveLoss(e.Iteration, e.SmoothedLoss);
        }
    }

    internal class MyCaffeOperation<T> : IDisposable
    {
        Log m_log;
        ManualResetEvent m_evtCancel = new ManualResetEvent(false);
        ManualResetEvent m_evtInitalized = new ManualResetEvent(false);
        AutoResetEvent m_evtReady = new AutoResetEvent(false);
        AutoResetEvent m_evtDone = new AutoResetEvent(false);
        string m_strName;
        string m_strModel;
        string m_strSolver;
        int m_nGpuID = 0;
        float[] m_rgInput = null;
        float[] m_rgTimeSteps = null;
        float[] m_rgMask = null;
        float[] m_rgTarget = null;
        float[] m_rgOutput = null;
        Exception m_err = null;
        Stopwatch m_sw = new Stopwatch();
        EventWaitHandle m_evtGlobalCancel;
        bool m_bEmphasize = false;

        public MyCaffeOperation(string strName)
        {
            m_strName = strName;
        }

        public bool Initialize(EventWaitHandle evtGlobalCancel, ManualResetEvent evtCancel, Log log, int nGpuID, string strModel, string strSolver, int nInputSize, int nOutputSize, bool bEmphasize = true)
        {
            m_log = log;
            m_nGpuID = nGpuID;
            m_strModel = strModel;
            m_strSolver = strSolver;
            m_bEmphasize = bEmphasize;

            m_evtCancel = evtCancel;
            m_evtGlobalCancel = evtGlobalCancel;
            m_rgInput = new float[nInputSize];
            m_rgTimeSteps = new float[nInputSize];
            m_rgMask = new float[nInputSize];
            m_rgTarget = new float[nOutputSize];

            Thread th = new Thread(new ThreadStart(operationThread));
            th.Start();

            WaitHandle[] rgWait = new WaitHandle[2];
            rgWait[0] = m_evtCancel;
            rgWait[1] = m_evtInitalized;

            if (WaitHandle.WaitAny(rgWait) == 0)
                return false;

            return true;
        }

        public void Dispose()
        {
            m_evtCancel.Set();
        }

        public WaitHandle RunCycleAsync(float[] rgInput, float[] rgTimeSteps, float[] rgMask, float[] rgTarget)
        {
            m_rgInput = rgInput;
            m_rgTimeSteps = rgTimeSteps;
            m_rgMask = rgMask;
            m_rgTarget = rgTarget;

            m_sw.Restart();
            m_evtReady.Set();

            return m_evtDone;
        }

        protected float[] convertF(T[] rg)
        {
            return Utility.ConvertVecF<T>(rg);
        }

        protected T[] convert(float[] rg)
        {
            return Utility.ConvertVec<T>(rg);
        }

        public WaitHandle DoneEvent
        {
            get { return m_evtDone; }
        }

        public float[] Output
        {
            get { return m_rgOutput; }
        }

        public double TotalMilliseconds
        {
            get { return m_sw.Elapsed.TotalMilliseconds; }
        }

        private void operationThread()
        {
            MyCaffeControl<T> mycaffe = null;

            try
            {
                m_log.EnableTrace = true;
                SettingsCaffe s = new SettingsCaffe();
                s.GpuIds = m_nGpuID.ToString();

                mycaffe = new MyCaffeControl<T>(s, m_log, new CancelEvent());
                mycaffe.LoadLite(Phase.TRAIN, m_strSolver, m_strModel, null, false, false);
                Net<T> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Solver<T> solver = mycaffe.GetInternalSolver();

                Blob<T> blobX = net.FindBlob("x");
                Blob<T> blobTt = net.FindBlob("tt");
                Blob<T> blobMask = net.FindBlob("mask");
                Blob<T> blobY = net.FindBlob("target");
                Blob<T> blobXhat = net.FindBlob("x_hat");

                m_evtInitalized.Set();

                WaitHandle[] rgWait = new WaitHandle[3];

                rgWait[0] = m_evtGlobalCancel;
                rgWait[1] = m_evtCancel;
                rgWait[2] = m_evtReady;

                while (WaitHandle.WaitAny(rgWait) > 1)
                {
                    if (m_bEmphasize)
                    {
                        blobX.mutable_cpu_data = convert(m_rgInput);
                        blobTt.mutable_cpu_data = convert(m_rgTimeSteps);
                        blobMask.mutable_cpu_data = convert(m_rgMask);
                        blobY.mutable_cpu_data = convert(m_rgTarget);

                        solver.Step(1);
                    }

                    m_rgOutput = convertF(blobXhat.mutable_cpu_data);

                    m_evtDone.Set();
                    m_sw.Stop();
                }
            }
            catch (Exception excpt)
            {
                m_err = excpt;
                m_log.WriteError(excpt);
            }
            finally
            {
                m_evtCancel.Set();
                if (mycaffe != null)
                    mycaffe.Dispose();
            }
        }
    }
}
