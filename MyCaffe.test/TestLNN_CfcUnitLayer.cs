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

/// <summary>
/// Testing the CfcCell layer.
/// 
/// CfcCell Layer - layer calculate gated linear unit.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestLNN_CfcUnitLayer
    {
        [TestMethod]
        public void TestForward()
        {
            CfcCellLayerTest test = new CfcCellLayerTest();

            try
            {
                foreach (ICfcCellLayerTest t in test.Tests)
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
            CfcCellLayerTest test = new CfcCellLayerTest();

            try
            {
                foreach (ICfcCellLayerTest t in test.Tests)
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
            CfcCellLayerTest test = new CfcCellLayerTest();

            try
            {
                foreach (ICfcCellLayerTest t in test.Tests)
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
            CfcCellLayerTest test = new CfcCellLayerTest();

            try
            {
                foreach (ICfcCellLayerTest t in test.Tests)
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
            CfcCellLayerTest test = new CfcCellLayerTest();

            try
            {
                foreach (ICfcCellLayerTest t in test.Tests)
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
            CfcCellLayerTest test = new CfcCellLayerTest();

            try
            {
                foreach (ICfcCellLayerTest t in test.Tests)
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

    interface ICfcCellLayerTest : ITest
    {
        void TestForward(bool bNoGate);
        void TestBackward(bool bNoGate);
        void TestGradient(bool bNoGate);
    }

    class CfcCellLayerTest : TestBase
    {
        public CfcCellLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("CfcCell Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new CfcCellLayerTest<double>(strName, nDeviceID, engine);
            else
                return new CfcCellLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class CfcCellLayerTest<T> : TestEx<T>, ICfcCellLayerTest
    {
        public CfcCellLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        private void load_weights(Layer<T> layer, string strPath, string strTag = "")
        {
            int nNumLayes = layer.layer_param.cfc_unit_param.backbone_layers;
            int nIdx = 0;

            for (int i = 0; i < nNumLayes; i++)
            {
                layer.blobs[nIdx].LoadFromNumpy(strPath + strTag + "bb_" + i.ToString() + ".weight.npy");
                nIdx++;
                layer.blobs[nIdx].LoadFromNumpy(strPath + strTag + "bb_" + i.ToString() + ".bias.npy");
                nIdx++;
            }

            layer.blobs[nIdx].LoadFromNumpy(strPath + strTag + "ff1.weight.npy");
            nIdx++;
            layer.blobs[nIdx].LoadFromNumpy(strPath + strTag + "ff1.bias.npy");
            nIdx++;

            layer.blobs[nIdx].LoadFromNumpy(strPath + strTag + "ff2.weight.npy");
            nIdx++;
            layer.blobs[nIdx].LoadFromNumpy(strPath + strTag + "ff2.bias.npy");
            nIdx++;

            layer.blobs[nIdx].LoadFromNumpy(strPath + strTag + "time_a.weight.npy");
            nIdx++;
            layer.blobs[nIdx].LoadFromNumpy(strPath + strTag + "time_a.bias.npy");
            nIdx++;

            layer.blobs[nIdx].LoadFromNumpy(strPath + strTag + "time_b.weight.npy");
            nIdx++;
            layer.blobs[nIdx].LoadFromNumpy(strPath + strTag + "time_b.bias.npy");
            nIdx++;
        }

        /// <summary>
        /// Test CfcUnit forward
        /// </summary>
        /// <remarks>
        /// To generate the test data, run the following:
        /// Code: test_cfc_cell.py
        /// Path: cfc_cell
        /// </remarks>
        public void TestForward(bool bNoGate)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CFC_UNIT);
            p.cfc_unit_param.input_size = 82;
            p.cfc_unit_param.hidden_size = 256;
            p.cfc_unit_param.backbone_activation = param.lnn.CfcUnitParameter.ACTIVATION.SILU;
            p.cfc_unit_param.backbone_dropout_ratio = 0.0f;
            p.cfc_unit_param.backbone_layers = 2;
            p.cfc_unit_param.backbone_units = 64;
            p.cfc_unit_param.no_gate = bNoGate;
            p.cfc_unit_param.minimal = false;
            Layer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobHx = null;
            Blob<T> blobTs = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strSubPath = (bNoGate) ? "cfc_cell_no_gate" : "cfc_cell_gate";
            string strPath = getTestDataPath(strSubPath);
            string strPathWts = getTestWtsPath(strSubPath);

            verifyFileDownload(strSubPath, "cell_ff1.npy");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX = new Blob<T>(m_cuda, m_log);
                blobHx = new Blob<T>(m_cuda, m_log);
                blobTs = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CFC_UNIT, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x.npy");
                blobHx.LoadFromNumpy(strPath + "hx.npy");
                blobTs.LoadFromNumpy(strPath + "ts.npy");

                BottomVec.Clear();
                BottomVec.Add(blobX);
                BottomVec.Add(blobHx);
                BottomVec.Add(blobTs);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);
                load_weights(layer, strPathWts);

                layer.Forward(BottomVec, TopVec);

                blobYexp.LoadFromNumpy(strPath + "h_state.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp, blobWork, false, 1e-07), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobYexp);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobHx);
                dispose(ref blobTs);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }

        /// <summary>
        /// Test CfcUnit backward
        /// </summary>
        /// <remarks>
        /// To generate the test data, run the following:
        /// Code: test_cfc_cell.py
        /// Path: cfc_cell
        /// </remarks>
        public void TestBackward(bool bNoGate)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CFC_UNIT);
            p.cfc_unit_param.input_size = 82;
            p.cfc_unit_param.hidden_size = 256;
            p.cfc_unit_param.backbone_activation = param.lnn.CfcUnitParameter.ACTIVATION.SILU;
            p.cfc_unit_param.backbone_dropout_ratio = 0.0f;
            p.cfc_unit_param.backbone_layers = 2;
            p.cfc_unit_param.backbone_units = 64;
            p.cfc_unit_param.no_gate = bNoGate;
            p.cfc_unit_param.minimal = false;
            Layer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobHx = null;
            Blob<T> blobTs = null;
            Blob<T> blobXgrad = null;
            Blob<T> blobHxgrad = null;
            Blob<T> blobTsgrad = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strSubPath = (bNoGate) ? "cfc_cell_no_gate" : "cfc_cell_gate";
            string strTag = "";
            string strVerifyFile = "cell_ff1.npy";

            string strPath = getTestDataPath(strSubPath);
            string strPathWts = getTestWtsPath(strSubPath);

            verifyFileDownload(strSubPath, strVerifyFile);

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX = new Blob<T>(m_cuda, m_log);
                blobHx = new Blob<T>(m_cuda, m_log);
                blobTs = new Blob<T>(m_cuda, m_log);
                blobXgrad = new Blob<T>(m_cuda, m_log);
                blobHxgrad = new Blob<T>(m_cuda, m_log);
                blobTsgrad = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CFC_UNIT, "The layer type is incorrect.");

                string strXname = "x.npy";
                string strHxName = "hx.npy";
                string strTsName = "ts.npy";
                string strXGradName = "cell_input.grad.npy";
                string strHxGradName = "cell_hx.grad.npy";
                string strTsGradName = "cell_ts.grad.npy";
                string strHStateName = "h_state.npy";
                string strHStateGradName = "h_state.grad.npy";

                blobX.LoadFromNumpy(strPath + strXname);
                blobHx.LoadFromNumpy(strPath + strHxName);
                blobTs.LoadFromNumpy(strPath + strTsName);

                BottomVec.Clear();
                BottomVec.Add(blobX);
                BottomVec.Add(blobHx);
                BottomVec.Add(blobTs);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);
                load_weights(layer, strPathWts, strTag);

                layer.Forward(BottomVec, TopVec);

                blobYexp.LoadFromNumpy(strPath + strHStateName);
                m_log.CHECK(TopVec[0].Compare(blobYexp, blobWork, false, 2e-07), "The blobs do not match.");

                //** BACKWARD **

                TopVec[0].LoadFromNumpy(strPath + strHStateGradName, true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                blobXgrad.LoadFromNumpy(strPath + strXGradName, true);
                m_log.CHECK(blobXgrad.Compare(blobX, blobWork, true, 1e-07), "The blobs do not match.");

                blobHxgrad.LoadFromNumpy(strPath + strHxGradName, true);
                m_log.CHECK(blobHxgrad.Compare(blobHx, blobWork, true, 1e-07), "The blobs do not match.");
                blobTsgrad.LoadFromNumpy(strPath + strTsGradName, true);
                m_log.CHECK(blobTsgrad.Compare(blobTs, blobWork, true, 1e-07), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobYexp);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobHx);
                dispose(ref blobTs);
                dispose(ref blobXgrad);
                dispose(ref blobHxgrad);
                dispose(ref blobTsgrad);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }

        /// <summary>
        /// Test CfcCell gradient check
        /// </summary>
        public void TestGradient(bool bNoGate)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CFC_UNIT);
            p.cfc_unit_param.input_size = 82;
            p.cfc_unit_param.hidden_size = 256;
            p.cfc_unit_param.backbone_activation = param.lnn.CfcUnitParameter.ACTIVATION.SILU;
            p.cfc_unit_param.backbone_dropout_ratio = 0.0f;
            p.cfc_unit_param.backbone_layers = 2;
            p.cfc_unit_param.backbone_units = 64;
            p.cfc_unit_param.no_gate = bNoGate;
            p.cfc_unit_param.minimal = false;
            Layer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobHx = null;
            Blob<T> blobTs = null;
            Blob<T> blobY = null;
            string strSubPath = (bNoGate) ? "cfc_cell_no_gate" : "cfc_cell_gate";
            string strPath = getTestDataPath(strSubPath);
            string strPathWts = getTestWtsPath(strSubPath);

            verifyFileDownload(strSubPath, "cell_ff1.npy");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX = new Blob<T>(m_cuda, m_log);
                blobHx = new Blob<T>(m_cuda, m_log);
                blobTs = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CFC_UNIT, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "x.npy");
                blobHx.LoadFromNumpy(strPath + "hx.npy");
                blobTs.LoadFromNumpy(strPath + "ts.npy");

                BottomVec.Clear();
                BottomVec.Add(blobX);
                BottomVec.Add(blobHx);
                BottomVec.Add(blobTs);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);
                load_weights(layer, strPathWts);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 0.01, 0.01);
                checker.CheckGradient(layer, BottomVec, TopVec, -1, 1, 0.01);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobHx);
                dispose(ref blobTs);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }
    }
}
