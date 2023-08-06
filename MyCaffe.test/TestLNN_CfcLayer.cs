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
    }

    interface ICfcLayerTest : ITest
    {
        void TestForward(bool bNoGate);
        void TestBackward(bool bNoGate);
        void TestGradient(bool bNoGate);
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
            return "C:\\temp\\projects\\LNN\\PythonApplication2\\PythonApplication2\\test\\" + strSubPath + "\\iter_0\\";
            //return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\LNN\\test\\" + strSubPath + "\\iter_0\\";
        }

        private string getTestWtsPath(string strSubPath)
        {
            return "C:\\temp\\projects\\LNN\\PythonApplication2\\PythonApplication2\\test\\" + strSubPath + "\\iter_0\\weights\\";
            //return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\LNN\\test\\" + strSubPath + "\\iter_0\\weights\\";
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
                blobTimeSpansGrad.LoadFromNumpy(strPath + "timespans.grad.npy", true);
                m_log.CHECK(blobTimeSpansGrad.Compare(blobTimeSpans, blobWork, true), "The blobs do not match.");
                blobMaskGrad.LoadFromNumpy(strPath + "cell_ts.grad.npy", true);
                m_log.CHECK(blobMaskGrad.Compare(blobMask, blobWork, true), "The blobs do not match.");
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
    }
}
