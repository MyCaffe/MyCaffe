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

/// <summary>
/// Testing the VarSelNet layer.
/// 
/// VarSelNet Layer - layer calculate variable selection network.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_VarSelNetLayer
    {
        [TestMethod]
        public void TestForward()
        {
            VarSelNetLayerTest test = new VarSelNetLayerTest();

            try
            {
                foreach (IVarSelNetLayerTest t in test.Tests)
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
            VarSelNetLayerTest test = new VarSelNetLayerTest();

            try
            {
                foreach (IVarSelNetLayerTest t in test.Tests)
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
            VarSelNetLayerTest test = new VarSelNetLayerTest();

            try
            {
                foreach (IVarSelNetLayerTest t in test.Tests)
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

    interface IVarSelNetLayerTest : ITest
    {
        void TestForward();
        void TestBackward();
        void TestGradient();
    }

    class VarSelNetLayerTest : TestBase
    {
        public VarSelNetLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("VarSelNet Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new VarSelNetLayerTest<double>(strName, nDeviceID, engine);
            else
                return new VarSelNetLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class VarSelNetLayerTest<T> : TestEx<T>, IVarSelNetLayerTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();

        public VarSelNetLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VARSELNET);
            p.varselnet_param.input_dim = 64;
            p.varselnet_param.hidden_dim = 64;
            p.varselnet_param.num_inputs = 8;
            p.varselnet_param.dropout = 0;
            p.varselnet_param.context_dim = 64;
            p.varselnet_param.axis = 1;
            Layer<T> layer = null;
            Blob<T> blobX1 = null;
            Blob<T> blobX2 = null;
            Blob<T> blobY1 = null;
            Blob<T> blobY2 = null;
            Blob<T> blobYexp1 = null;
            Blob<T> blobYexp2 = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX1 = new Blob<T>(m_cuda, m_log);
                blobX2 = new Blob<T>(m_cuda, m_log);
                blobY1 = new Blob<T>(m_cuda, m_log);
                blobY2 = new Blob<T>(m_cuda, m_log);
                blobYexp1 = new Blob<T>(m_cuda, m_log);
                blobYexp2 = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.VARSELNET, "The layer type is incorrect.");

                blobX1.LoadFromNumpy(strPath + "test_varsel_flattened_embedding.npy");
                blobX2.LoadFromNumpy(strPath + "test_varsel_context.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX1);
                BottomVec.Add(blobX2);
                TopVec.Clear();
                TopVec.Add(blobY1);
                TopVec.Add(blobY2);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.skip_layer.module.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.skip_layer.module.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.fc1.module.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.fc1.module.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.context_projection.module.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.fc2.module.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.fc2.module.bias.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.gate.module.fc1.weight.npy");
                layer.blobs[8].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.gate.module.fc1.bias.npy");
                layer.blobs[9].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.gate.module.fc2.weight.npy");
                layer.blobs[10].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.gate.module.fc2.bias.npy");

                int nIdx = 11;
                for (int i = 0; i < p.varselnet_param.num_inputs; i++)
                {
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }

                layer.Forward(BottomVec, TopVec);

                blobYexp1.LoadFromNumpy(strPath + "test_varsel_outputs_sum.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp1, blobWork, false, 2e-06), "The blobs do not match.");
                blobYexp2.LoadFromNumpy(strPath + "test_varsel_sparse_weights.npy");
                m_log.CHECK(TopVec[1].Compare(blobYexp2, blobWork, false, 3e-06), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobYexp1);
                dispose(ref blobYexp2);
                dispose(ref blobWork);
                dispose(ref blobX1);
                dispose(ref blobX2);
                dispose(ref blobY1);
                dispose(ref blobY2);

                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VARSELNET);
            p.varselnet_param.input_dim = 64;
            p.varselnet_param.hidden_dim = 64;
            p.varselnet_param.num_inputs = 8;
            p.varselnet_param.dropout = 0;
            p.varselnet_param.context_dim = 64;
            p.varselnet_param.axis = 1;
            Layer<T> layer = null;
            Blob<T> blobX1 = null;
            Blob<T> blobX2 = null;
            Blob<T> blobY1 = null;
            Blob<T> blobY2 = null;
            Blob<T> blobYexp1 = null;
            Blob<T> blobYexp2 = null;
            Blob<T> blobGradExp = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX1 = new Blob<T>(m_cuda, m_log);
                blobX2 = new Blob<T>(m_cuda, m_log);
                blobY1 = new Blob<T>(m_cuda, m_log);
                blobY2 = new Blob<T>(m_cuda, m_log);
                blobYexp1 = new Blob<T>(m_cuda, m_log);
                blobYexp2 = new Blob<T>(m_cuda, m_log);
                blobGradExp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.VARSELNET, "The layer type is incorrect.");

                blobX1.LoadFromNumpy(strPath + "test_varsel_flattened_embedding.npy");
                blobX2.LoadFromNumpy(strPath + "test_varsel_context.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX1);
                BottomVec.Add(blobX2);
                TopVec.Clear();
                TopVec.Add(blobY1);
                TopVec.Add(blobY2);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.skip_layer.module.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.skip_layer.module.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.fc1.module.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.fc1.module.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.context_projection.module.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.fc2.module.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.fc2.module.bias.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.gate.module.fc1.weight.npy");
                layer.blobs[8].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.gate.module.fc1.bias.npy");
                layer.blobs[9].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.gate.module.fc2.weight.npy");
                layer.blobs[10].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.gate.module.fc2.bias.npy");

                int nIdx = 11;
                for (int i = 0; i < p.varselnet_param.num_inputs; i++)
                {
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }

                layer.Forward(BottomVec, TopVec);

                blobYexp1.LoadFromNumpy(strPath + "test_varsel_outputs_sum.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp1, blobWork, false, 2e-06), "The blobs do not match.");
                blobYexp2.LoadFromNumpy(strPath + "test_varsel_sparse_weights.npy");
                m_log.CHECK(TopVec[1].Compare(blobYexp2, blobWork, false, 3e-06), "The blobs do not match.");

                TopVec[0].LoadFromNumpy(strPath + "test_varsel_outputs_sum.grad.npy", true);
                TopVec[1].LoadFromNumpy(strPath + "test_varsel_sparse_weights.grad.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                blobGradExp.LoadFromNumpy(strPath + "test_varsel_flattened_embedding.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(blobX1, blobWork, true, 3e-07), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "test_varsel_context.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(blobX2, blobWork, true, 2e-07), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobGradExp);
                dispose(ref blobYexp1);
                dispose(ref blobYexp2);
                dispose(ref blobWork);
                dispose(ref blobX1);
                dispose(ref blobX2);
                dispose(ref blobY1);
                dispose(ref blobY2);

                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VARSELNET);
            p.varselnet_param.input_dim = 64;
            p.varselnet_param.hidden_dim = 64;
            p.varselnet_param.num_inputs = 8;
            p.varselnet_param.dropout = 0;
            p.varselnet_param.context_dim = 64;
            p.varselnet_param.axis = 1;
            Layer<T> layer = null;
            Blob<T> blobX1 = null;
            Blob<T> blobX2 = null;
            Blob<T> blobY1 = null;
            Blob<T> blobY2 = null;
            Blob<T> blobYexp1 = null;
            Blob<T> blobYexp2 = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX1 = new Blob<T>(m_cuda, m_log);
                blobX2 = new Blob<T>(m_cuda, m_log);
                blobY1 = new Blob<T>(m_cuda, m_log);
                blobY2 = new Blob<T>(m_cuda, m_log);
                blobYexp1 = new Blob<T>(m_cuda, m_log);
                blobYexp2 = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.VARSELNET, "The layer type is incorrect.");

                blobX1.LoadFromNumpy(strPath + "test_varsel_flattened_embedding.npy");
                blobX2.LoadFromNumpy(strPath + "test_varsel_context.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX1);
                BottomVec.Add(blobX2);
                TopVec.Clear();
                TopVec.Add(blobY1);
                TopVec.Add(blobY2);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.skip_layer.module.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.skip_layer.module.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.fc1.module.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.fc1.module.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.context_projection.module.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.fc2.module.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.fc2.module.bias.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.gate.module.fc1.weight.npy");
                layer.blobs[8].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.gate.module.fc1.bias.npy");
                layer.blobs[9].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.gate.module.fc2.weight.npy");
                layer.blobs[10].LoadFromNumpy(strPath + "test_futuresel_flattened_grn.gate.module.fc2.bias.npy");

                int nIdx = 11;
                for (int i = 0; i < p.varselnet_param.num_inputs; i++)
                {
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "test_futuresel_single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradient(layer, BottomVec, TopVec, -1, 1, 0.01);
            }
            finally
            {
                dispose(ref blobYexp1);
                dispose(ref blobYexp2);
                dispose(ref blobWork);
                dispose(ref blobX1);
                dispose(ref blobX2);
                dispose(ref blobY1);
                dispose(ref blobY2);

                if (layer != null)
                    layer.Dispose();
            }
        }
    }
}
