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
using MyCaffe.layers.tft;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.Tab;

/// <summary>
/// Testing the Grn layer.
/// 
/// Grn Layer - layer calculate gated residual layer.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_GrnLayer
    {
        [TestMethod]
        public void TestForward()
        {
            GrnLayerTest test = new GrnLayerTest();

            try
            {
                foreach (IGrnLayerTest t in test.Tests)
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
            GrnLayerTest test = new GrnLayerTest();

            try
            {
                foreach (IGrnLayerTest t in test.Tests)
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
        public void TestGradient()
        {
            GrnLayerTest test = new GrnLayerTest();

            try
            {
                foreach (IGrnLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IGrnLayerTest : ITest
    {
        void TestForward();
        void TestBackward();
        void TestGradient();
    }

    class GrnLayerTest : TestBase
    {
        public GrnLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Grn Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new GrnLayerTest<double>(strName, nDeviceID, engine);
            else
                return new GrnLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class GrnLayerTest<T> : TestEx<T>, IGrnLayerTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();
        int m_nNumOutput = 3;
        int m_nBatchSize;
        int m_nVectorDim;

        public GrnLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            m_colData.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        private string getTestDataPath()
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\iter_0\\";
        }

        private string getTestWtsPath()
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\hist_ts_transform\\";
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRN);
            p.grn_param.input_dim = 576;
            p.grn_param.output_dim = 9;
            p.grn_param.context_dim = null;
            p.grn_param.hidden_dim = 64;
            p.grn_param.dropout_ratio = 0.0f;
            p.grn_param.batch_first = true;
            GrnLayer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as GrnLayer<T>;
                blobX = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.GRN, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "grn_x.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "test_grn_skip_layer.module.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "test_grn_skip_layer.module.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "test_grn_fc1.module.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "test_grn_fc1.module.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "test_grn_fc2.module.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "test_grn_fc2.module.bias.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "test_grn_gate.module.fc1.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "test_grn_gate.module.fc1.bias.npy");
                layer.blobs[8].LoadFromNumpy(strPath + "test_grn_gate.module.fc2.weight.npy");
                layer.blobs[9].LoadFromNumpy(strPath + "test_grn_gate.module.fc2.bias.npy");

                layer.Forward(BottomVec, TopVec);

                blobYexp.LoadFromNumpy(strPath + "test_grn_y.npy");
                double dfErr = (typeof(T) == typeof(float)) ? 3e-07 : 8e-07;
                m_log.CHECK(TopVec[0].Compare(blobYexp, blobWork, false, dfErr), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobYexp);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRN);
            p.grn_param.input_dim = 576;
            p.grn_param.output_dim = 9;
            p.grn_param.context_dim = null;
            p.grn_param.hidden_dim = 64;
            p.grn_param.dropout_ratio = 0.0f;
            p.grn_param.batch_first = true;
            GrnLayer<T> layer = null;
            Blob<T> blobGradExp = null;
            Blob<T> blobX = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as GrnLayer<T>;
                blobGradExp = new Blob<T>(m_cuda, m_log);
                blobX = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.GRN, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "test_grn_x.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "test_grn_skip_layer.module.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "test_grn_skip_layer.module.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "test_grn_fc1.module.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "test_grn_fc1.module.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "test_grn_fc2.module.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "test_grn_fc2.module.bias.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "test_grn_gate.module.fc1.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "test_grn_gate.module.fc1.bias.npy");
                layer.blobs[8].LoadFromNumpy(strPath + "test_grn_gate.module.fc2.weight.npy");
                layer.blobs[9].LoadFromNumpy(strPath + "test_grn_gate.module.fc2.bias.npy");

                layer.Forward(BottomVec, TopVec);

                blobYexp.LoadFromNumpy(strPath + "test_grn_y.npy");
                double dfErr = (typeof(T) == typeof(float)) ? 3e-07 : 8e-07;
                m_log.CHECK(TopVec[0].Compare(blobYexp, blobWork, false, dfErr), "The blobs do not match.");

                TopVec[0].LoadFromNumpy(strPath + "test_grn_y.grad.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                blobGradExp.LoadFromNumpy(strPath + "test_grn_x.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(blobX, blobWork, true, 5.5e-08), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "test_grn.skip_layer.module.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[0], blobWork, true), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "test_grn.skip_layer.module.bias.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[1], blobWork, true), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "test_grn.fc1.module.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[2], blobWork, true), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "test_grn.fc1.module.bias.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[3], blobWork, true), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "test_grn.fc2.module.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[4], blobWork, true), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "test_grn.fc2.module.bias.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[5], blobWork, true), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "test_grn.gate.module.fc1.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[6], blobWork, true), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "test_grn.gate.module.fc1.bias.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[7], blobWork, true), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "test_grn.gate.module.fc2.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[8], blobWork, true), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "test_grn.gate.module.fc2.bias.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[9], blobWork, true), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobGradExp);
                dispose(ref blobYexp);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRN);
            p.grn_param.input_dim = 576;
            p.grn_param.output_dim = 9;
            p.grn_param.context_dim = null;
            p.grn_param.hidden_dim = 64;
            p.grn_param.dropout_ratio = 0.0f;
            p.grn_param.batch_first = true;
            GrnLayer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as GrnLayer<T>;
                blobX = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.GRN, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "grn_x.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "test_grn_skip_layer.module.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "test_grn_skip_layer.module.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "test_grn_fc1.module.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "test_grn_fc1.module.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "test_grn_fc2.module.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "test_grn_fc2.module.bias.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "test_grn_gate.module.fc1.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "test_grn_gate.module.fc1.bias.npy");
                layer.blobs[8].LoadFromNumpy(strPath + "test_grn_gate.module.fc2.weight.npy");
                layer.blobs[9].LoadFromNumpy(strPath + "test_grn_gate.module.fc2.bias.npy");

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradient(layer, BottomVec, TopVec, -1, 1, 0.01);
            }
            finally
            {
                dispose(ref blobYexp);
                dispose(ref blobWork);
                dispose(ref blobX);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }
    }
}
