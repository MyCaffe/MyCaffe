using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.param;
using System;
using System.Collections.Generic;

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

        [TestMethod]
        public void TestForwardTemporal()
        {
            VarSelNetLayerTest test = new VarSelNetLayerTest();

            try
            {
                foreach (IVarSelNetLayerTest t in test.Tests)
                {
                    t.TestForwardTemporal();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardTemporal()
        {
            VarSelNetLayerTest test = new VarSelNetLayerTest();

            try
            {
                foreach (IVarSelNetLayerTest t in test.Tests)
                {
                    t.TestBackwardTemporal();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardStatic()
        {
            VarSelNetLayerTest test = new VarSelNetLayerTest();

            try
            {
                foreach (IVarSelNetLayerTest t in test.Tests)
                {
                    t.TestForwardStatic();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardStatic()
        {
            VarSelNetLayerTest test = new VarSelNetLayerTest();

            try
            {
                foreach (IVarSelNetLayerTest t in test.Tests)
                {
                    t.TestBackwardStatic();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardHistorical()
        {
            VarSelNetLayerTest test = new VarSelNetLayerTest();

            try
            {
                foreach (IVarSelNetLayerTest t in test.Tests)
                {
                    t.TestForwardHistorical();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardHistorical()
        {
            VarSelNetLayerTest test = new VarSelNetLayerTest();

            try
            {
                foreach (IVarSelNetLayerTest t in test.Tests)
                {
                    t.TestBackwardHistorical();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardFuture()
        {
            VarSelNetLayerTest test = new VarSelNetLayerTest();

            try
            {
                foreach (IVarSelNetLayerTest t in test.Tests)
                {
                    t.TestForwardFuture();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardFuturel()
        {
            VarSelNetLayerTest test = new VarSelNetLayerTest();

            try
            {
                foreach (IVarSelNetLayerTest t in test.Tests)
                {
                    t.TestBackwardFuture();
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
        void TestForwardTemporal();
        void TestBackwardTemporal();
        void TestForwardStatic();
        void TestBackwardStatic();
        void TestForwardHistorical();
        void TestBackwardHistorical();
        void TestForwardFuture();
        void TestBackwardFuture();
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

        public void TestForwardTemporal()
        {
            int nNumHistNumeric = 4;
            int nNumHistCategorical = 7;
            int nStateSize = 64;
            LayerParameter reshape_before_param = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL);
            reshape_before_param.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.BEFORE;
            Layer<T> layerReshapeBefore = null;
            LayerParameter reshape_after_param = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL);
            reshape_after_param.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.AFTER;
            Layer<T> layerReshapeAfter = null;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VARSELNET);
            p.varselnet_param.input_dim = nStateSize;
            p.varselnet_param.hidden_dim = nStateSize;
            p.varselnet_param.num_inputs = nNumHistNumeric + nNumHistCategorical;
            p.varselnet_param.dropout = 0;
            p.varselnet_param.context_dim = nStateSize;
            p.varselnet_param.axis = 1;
            Layer<T> layer = null;
            Blob<T> blobX1a = null;
            Blob<T> blobX2a = null;
            Blob<T> blobX1b = null;
            Blob<T> blobX2b = null;
            Blob<T> blobX1c = null;
            Blob<T> blobX2c = null;
            Blob<T> blobY1 = null;
            Blob<T> blobY2 = null;
            Blob<T> blobYexp1 = null;
            Blob<T> blobYexp2 = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layerReshapeBefore = Layer<T>.Create(m_cuda, m_log, reshape_before_param, null);
                layerReshapeAfter = Layer<T>.Create(m_cuda, m_log, reshape_after_param, null);

                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX1a = new Blob<T>(m_cuda, m_log);
                blobX2a = new Blob<T>(m_cuda, m_log);
                blobX1b = new Blob<T>(m_cuda, m_log);
                blobX2b = new Blob<T>(m_cuda, m_log);
                blobX1c = new Blob<T>(m_cuda, m_log);
                blobX2c = new Blob<T>(m_cuda, m_log);
                blobY1 = new Blob<T>(m_cuda, m_log);
                blobY2 = new Blob<T>(m_cuda, m_log);
                blobYexp1 = new Blob<T>(m_cuda, m_log);
                blobYexp2 = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.VARSELNET, "The layer type is incorrect.");

                blobX1a.LoadFromNumpy(strPath + "tft.hist.temporal_representation.npy");
                blobX1a.Name = "temporal_respresentation";
                blobX2a.LoadFromNumpy(strPath + "tft.static_selection_signal.npy");
                blobX2a.Name = "static_selection_signal";
                
                BottomVec.Clear();
                BottomVec.Add(blobX1a);
                BottomVec.Add(blobX2a);
                TopVec.Clear();
                TopVec.Add(blobX1b);
                TopVec.Add(blobX2b);
                layerReshapeBefore.Setup(BottomVec, TopVec);

                BottomVec[0] = TopVec[0];
                BottomVec[1] = TopVec[1];
                TopVec[0] = blobX1c;
                TopVec[1] = blobX2c;
                layer.Setup(BottomVec, TopVec);

                BottomVec[0] = TopVec[0];
                BottomVec[1] = TopVec[1];
                TopVec[0] = blobY1;
                TopVec[1] = blobY2;
                layerReshapeAfter.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.skip_layer.module.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.skip_layer.module.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.fc1.module.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.fc1.module.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.context_projection.module.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.fc2.module.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.fc2.module.bias.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.gate.module.fc1.weight.npy");
                layer.blobs[8].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.gate.module.fc1.bias.npy");
                layer.blobs[9].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.gate.module.fc2.weight.npy");
                layer.blobs[10].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.gate.module.fc2.bias.npy");

                int nIdx = 11;
                for (int i = 0; i < p.varselnet_param.num_inputs; i++)
                {
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }

                BottomVec.Clear();
                BottomVec.Add(blobX1a);
                BottomVec.Add(blobX2a);
                TopVec.Clear();
                TopVec.Add(blobX1b);
                TopVec.Add(blobX2b);

                layerReshapeBefore.Forward(BottomVec, TopVec);

                blobYexp1.LoadFromNumpy(strPath + "test.tft.temporal_flattened_embedding.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp1, blobWork), "The blobs do not match.");
                blobYexp1.LoadFromNumpy(strPath + "test.tft.time_distributed_context.npy");
                m_log.CHECK(TopVec[1].Compare(blobYexp1, blobWork), "The blobs do not match.");

                BottomVec[0] = TopVec[0];
                BottomVec[1] = TopVec[1];
                TopVec[0] = blobX1c;
                TopVec[1] = blobX2c;
                layer.Forward(BottomVec, TopVec);

                blobYexp1.LoadFromNumpy(strPath + "test.tft.temporal_selection_output.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp1, blobWork, false, (typeof(T) == typeof(float)) ? 8e-07 : 2e-06), "The blobs do not match.");
                blobYexp1.LoadFromNumpy(strPath + "test.tft.temporal_selection_weights.npy");
                m_log.CHECK(TopVec[1].Compare(blobYexp1, blobWork, false, (typeof(T) == typeof(float)) ? 8e-07 : 2e-06), "The blobs do not match.");

                BottomVec[0] = TopVec[0];
                BottomVec[1] = TopVec[1];
                TopVec[0] = blobY1;
                TopVec[1] = blobY2;
                layerReshapeAfter.Forward(BottomVec, TopVec);

                blobYexp1.LoadFromNumpy(strPath + "test.tft.temporal_selection_output2.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp1, blobWork, false, (typeof(T) == typeof(float)) ? 5e-07 : 2e-06), "The blobs do not match.");
                blobYexp2.LoadFromNumpy(strPath + "test.tft.temporal_selection_weights2.npy");
                m_log.CHECK(TopVec[1].Compare(blobYexp2, blobWork, false, (typeof(T) == typeof(float)) ? 5e-07 : 2e-06), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobYexp1);
                dispose(ref blobYexp2);
                dispose(ref blobWork);
                dispose(ref blobX1a);
                dispose(ref blobX2a);
                dispose(ref blobX1b);
                dispose(ref blobX2b);
                dispose(ref blobX1c);
                dispose(ref blobX2c);
                dispose(ref blobY1);
                dispose(ref blobY2);

                if (layer != null)
                    layer.Dispose();
                if (layerReshapeBefore != null)
                    layerReshapeBefore.Dispose();
                if (layerReshapeAfter != null)
                    layerReshapeAfter.Dispose();
            }
        }

        public void TestBackwardTemporal()
        {
            int nNumHistNumeric = 4;
            int nNumHistCategorical = 7;
            int nStateSize = 64;
            LayerParameter reshape_before_param = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL);
            reshape_before_param.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.BEFORE;
            Layer<T> layerReshapeBefore = null;
            LayerParameter reshape_after_param = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL);
            reshape_after_param.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.AFTER;
            Layer<T> layerReshapeAfter = null;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VARSELNET);
            p.varselnet_param.input_dim = nStateSize;
            p.varselnet_param.hidden_dim = nStateSize;
            p.varselnet_param.num_inputs = nNumHistNumeric + nNumHistCategorical;
            p.varselnet_param.dropout = 0;
            p.varselnet_param.context_dim = nStateSize;
            p.varselnet_param.axis = 1;
            Layer<T> layer = null;
            Blob<T> blobX1a = null;
            Blob<T> blobX2a = null;
            Blob<T> blobX1b = null;
            Blob<T> blobX2b = null;
            Blob<T> blobX1c = null;
            Blob<T> blobX2c = null;
            Blob<T> blobY1 = null;
            Blob<T> blobY2 = null;
            Blob<T> blobYexp1 = null;
            Blob<T> blobYexp2 = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layerReshapeBefore = Layer<T>.Create(m_cuda, m_log, reshape_before_param, null);
                layerReshapeAfter = Layer<T>.Create(m_cuda, m_log, reshape_after_param, null);

                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX1a = new Blob<T>(m_cuda, m_log);
                blobX2a = new Blob<T>(m_cuda, m_log);
                blobX1b = new Blob<T>(m_cuda, m_log);
                blobX2b = new Blob<T>(m_cuda, m_log);
                blobX1c = new Blob<T>(m_cuda, m_log);
                blobX2c = new Blob<T>(m_cuda, m_log);
                blobY1 = new Blob<T>(m_cuda, m_log);
                blobY2 = new Blob<T>(m_cuda, m_log);
                blobYexp1 = new Blob<T>(m_cuda, m_log);
                blobYexp2 = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.VARSELNET, "The layer type is incorrect.");

                blobX1a.LoadFromNumpy(strPath + "tft.hist.temporal_representation.npy");
                blobX1a.Name = "temporal_respresentation";
                blobX2a.LoadFromNumpy(strPath + "tft.static_selection_signal.npy");
                blobX2a.Name = "static_selection_signal";

                BottomVec.Clear();
                BottomVec.Add(blobX1a);
                BottomVec.Add(blobX2a);
                TopVec.Clear();
                TopVec.Add(blobX1b);
                TopVec.Add(blobX2b);
                layerReshapeBefore.Setup(BottomVec, TopVec);

                BottomVec[0] = TopVec[0];
                BottomVec[1] = TopVec[1];
                TopVec[0] = blobX1c;
                TopVec[1] = blobX2c;
                layer.Setup(BottomVec, TopVec);

                BottomVec[0] = TopVec[0];
                BottomVec[1] = TopVec[1];
                TopVec[0] = blobY1;
                TopVec[1] = blobY2;
                layerReshapeAfter.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.skip_layer.module.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.skip_layer.module.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.fc1.module.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.fc1.module.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.context_projection.module.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.fc2.module.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.fc2.module.bias.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.gate.module.fc1.weight.npy");
                layer.blobs[8].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.gate.module.fc1.bias.npy");
                layer.blobs[9].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.gate.module.fc2.weight.npy");
                layer.blobs[10].LoadFromNumpy(strPath + "tft.hist_ts_selection.flattened_grn.gate.module.fc2.bias.npy");

                int nIdx = 11;
                for (int i = 0; i < p.varselnet_param.num_inputs; i++)
                {
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.hist_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }

                BottomVec.Clear();
                BottomVec.Add(blobX1a);
                BottomVec.Add(blobX2a);
                TopVec.Clear();
                TopVec.Add(blobX1b);
                TopVec.Add(blobX2b);

                layerReshapeBefore.Forward(BottomVec, TopVec);

                blobYexp1.LoadFromNumpy(strPath + "test.tft.temporal_flattened_embedding.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp1, blobWork), "The blobs do not match.");
                blobYexp1.LoadFromNumpy(strPath + "test.tft.time_distributed_context.npy");
                m_log.CHECK(TopVec[1].Compare(blobYexp1, blobWork), "The blobs do not match.");

                BottomVec[0] = TopVec[0];
                BottomVec[1] = TopVec[1];
                TopVec[0] = blobX1c;
                TopVec[1] = blobX2c;
                layer.Forward(BottomVec, TopVec);

                blobYexp1.LoadFromNumpy(strPath + "test.tft.temporal_selection_output.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp1, blobWork, false, (typeof(T) == typeof(float)) ? 8e-07 : 2e-06), "The blobs do not match.");
                blobYexp1.LoadFromNumpy(strPath + "test.tft.temporal_selection_weights.npy");
                m_log.CHECK(TopVec[1].Compare(blobYexp1, blobWork, false, (typeof(T) == typeof(float)) ? 8e-07 : 2e-06), "The blobs do not match.");

                BottomVec[0] = TopVec[0];
                BottomVec[1] = TopVec[1];
                TopVec[0] = blobY1;
                TopVec[1] = blobY2;
                layerReshapeAfter.Forward(BottomVec, TopVec);

                blobYexp1.LoadFromNumpy(strPath + "test.tft.temporal_selection_output2.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp1, blobWork, false, (typeof(T) == typeof(float)) ? 8e-07 : 2e-06), "The blobs do not match.");
                blobYexp2.LoadFromNumpy(strPath + "test.tft.temporal_selection_weights2.npy");
                m_log.CHECK(TopVec[1].Compare(blobYexp2, blobWork, false, (typeof(T) == typeof(float)) ? 8e-07 : 2e-06), "The blobs do not match.");

                //*** Backward ***

                TopVec[0].LoadFromNumpy(strPath + "test.tft.temporal_selection_output2.grad.npy", true);
                layerReshapeAfter.Backward(TopVec, new List<bool> { true }, BottomVec);

                blobYexp1.LoadFromNumpy(strPath + "test.tft.temporal_selection_output.grad.npy", true);
                m_log.CHECK(BottomVec[0].Compare(blobYexp1, blobWork, true), "The blobs do not match.");

                TopVec[0] = BottomVec[0];
                TopVec[1] = BottomVec[1];
                BottomVec[0] = blobX1b;
                BottomVec[1] = blobX2b;
                layer.Backward(TopVec, new List<bool> { true }, BottomVec);

                blobYexp1.LoadFromNumpy(strPath + "test.tft.temporal_flattened_embedding.grad.npy", true);
                m_log.CHECK(BottomVec[0].Compare(blobYexp1, blobWork, true, 2e-07), "The blobs do not match.");
                blobYexp1.LoadFromNumpy(strPath + "test.tft.time_distributed_context.grad.npy", true);
                m_log.CHECK(BottomVec[1].Compare(blobYexp1, blobWork, true, 1e-07), "The blobs do not match.");

                TopVec[0] = BottomVec[0];
                TopVec[1] = BottomVec[1];
                BottomVec[0] = blobX1a;
                BottomVec[1] = blobX2a;
                layerReshapeBefore.Backward(TopVec, new List<bool> { true }, BottomVec);

                blobYexp1.LoadFromNumpy(strPath + "test.tft.temporal_representation1.grad.npy", true);
                m_log.CHECK(BottomVec[0].Compare(blobYexp1, blobWork, true, 2e-07), "The blobs do not match.");
                blobYexp1.LoadFromNumpy(strPath + "test.tft.static_selection_signal1.grad.npy", true);
                m_log.CHECK(BottomVec[1].Compare(blobYexp1, blobWork, true, 2e-06), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobYexp1);
                dispose(ref blobYexp2);
                dispose(ref blobWork);
                dispose(ref blobX1a);
                dispose(ref blobX2a);
                dispose(ref blobX1b);
                dispose(ref blobX2b);
                dispose(ref blobX1c);
                dispose(ref blobX2c);
                dispose(ref blobY1);
                dispose(ref blobY2);

                if (layer != null)
                    layer.Dispose();
                if (layerReshapeBefore != null)
                    layerReshapeBefore.Dispose();
                if (layerReshapeAfter != null)
                    layerReshapeAfter.Dispose();
            }
        }


        private string buildModel_static(int nNumSamples, float fDropout, int nStateSize,
            int nNumStaticNumeric, int nNumStaticCategorical)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";

            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumStaticCategorical * nStateSize }));         
            input.top.Add("static_rep");
            p.layer.Add(input);


            //---------------------------------
            //  Variable Selection Networks - Static
            //---------------------------------
            LayerParameter static_vsn = new LayerParameter(LayerParameter.LayerType.VARSELNET, "static_vsn");
            static_vsn.varselnet_param.input_dim = nStateSize;
            static_vsn.varselnet_param.num_inputs = nNumStaticNumeric + nNumStaticCategorical;
            static_vsn.varselnet_param.hidden_dim = nStateSize;
            static_vsn.varselnet_param.dropout = fDropout;
            static_vsn.bottom.Add("static_rep");
            static_vsn.top.Add("selected_static");
            static_vsn.top.Add("static_wts");
            p.layer.Add(static_vsn);

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Test the forward pass for sequence processing
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run test_2_variableselectionnetwork_stat.py on fresh 'test\iter_0' data
        /// 
        /// Fresh test\iter_0 data generated by running:
        /// training.py with TemporalFusionTransformer options: debug=True, tag='tft', use_mycaffe=True
        /// </remarks>
        public void TestForwardStatic()
        {
            string strPath = getTestDataPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nStateSize = 64;
            int nNumStaticNumeric = 0;
            int nNumStaticCategorical = 9;
            float fDropout = 0;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel_static(nNumSamples, fDropout, nStateSize, nNumStaticNumeric, nNumStaticCategorical);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, null, null);

                int nIdx = 0;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.skip_layer.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.skip_layer.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.fc1.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.fc1.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.fc2.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.fc2.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.gate.module.fc1.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.gate.module.fc1.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.gate.module.fc2.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.gate.module.fc2.bias.npy");
                nIdx++;

                for (int i = 0; i < nNumStaticCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }

                blob1 = net.FindBlob("static_rep");
                blob1.LoadFromNumpy(strPath + "tft.static_rep.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.vsn.selected_static.npy");
                blob1 = net.FindBlob("selected_static");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 3e-07 : 4e-07), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.vsn.selected_static_wts.npy");
                blob1 = net.FindBlob("static_wts");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 3e-07 : 4e-07), "The blobs are different!");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }

        /// <summary>
        /// Test the backward pass for sequence processing
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run test_2_variableselectionnetwork_stat.py on fresh 'test\iter_0' data
        /// 
        /// Fresh test\iter_0 data generated by running:
        /// training.py with TemporalFusionTransformer options: debug=True, tag='tft', use_mycaffe=True
        /// </remarks>
        public void TestBackwardStatic()
        {
            string strPath = getTestDataPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nStateSize = 64;
            int nNumStaticNumeric = 0;
            int nNumStaticCategorical = 9;
            float fDropout = 0;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel_static(nNumSamples, fDropout, nStateSize, nNumStaticNumeric, nNumStaticCategorical);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);
                param.force_backward = true;

                net = new Net<T>(m_cuda, m_log, param, null, null);

                int nIdx = 0;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.skip_layer.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.skip_layer.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.fc1.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.fc1.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.fc2.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.fc2.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.gate.module.fc1.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.gate.module.fc1.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.gate.module.fc2.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.flattened_grn.gate.module.fc2.bias.npy");
                nIdx++;

                for (int i = 0; i < nNumStaticCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.static.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }

                blob1 = net.FindBlob("static_rep");
                blob1.LoadFromNumpy(strPath + "tft.static_rep.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.vsn.selected_static.npy");
                blob1 = net.FindBlob("selected_static");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 3e-07 : 4e-07), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.vsn.selected_static_wts.npy");
                blob1 = net.FindBlob("static_wts");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 3e-07 : 4e-07), "The blobs are different!");

                //*** BACKWARD ***

                blob1 = net.FindBlob("selected_static");
                blob1.LoadFromNumpy(strPath + "tft.vsn.selected_static.grad.npy", true);

                net.Backward();

                blobVal.LoadFromNumpy(strPath + "tft.vsn.static_rep1.grad.npy", true);
                blob1 = net.FindBlob("static_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grads are different!");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }


        private string buildModel_historical(int nNumSamples, float fDropout, int nStateSize, int nNumHistSteps,
            int nNumHistoricalNumeric, int nNumHistoricalCategorical)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";

            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHistSteps, (nNumHistoricalNumeric + nNumHistoricalCategorical) * nStateSize }));
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nStateSize }));
            input.top.Add("hist_ts_rep");
            input.top.Add("c_selection");
            p.layer.Add(input);

            //---------------------------------
            //  Variable Selection Networks - Temporal
            //---------------------------------
            LayerParameter hist_vsh_reshape_before = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_hist_b");
            hist_vsh_reshape_before.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.BEFORE;
            hist_vsh_reshape_before.bottom.Add("hist_ts_rep");
            hist_vsh_reshape_before.bottom.Add("c_selection");
            hist_vsh_reshape_before.top.Add("hist_ts_rep1");
            hist_vsh_reshape_before.top.Add("c_selection1");
            p.layer.Add(hist_vsh_reshape_before);

            LayerParameter hist_vsn = new LayerParameter(LayerParameter.LayerType.VARSELNET, "hist_vsn");
            hist_vsn.varselnet_param.input_dim = nStateSize;
            hist_vsn.varselnet_param.num_inputs = nNumHistoricalNumeric + nNumHistoricalCategorical;
            hist_vsn.varselnet_param.hidden_dim = nStateSize;
            hist_vsn.varselnet_param.dropout = fDropout;
            hist_vsn.varselnet_param.context_dim = nStateSize;
            hist_vsn.bottom.Add("hist_ts_rep1");
            hist_vsn.bottom.Add("c_selection1");
            hist_vsn.top.Add("selected_hist1");
            hist_vsn.top.Add("hist_wts");
            p.layer.Add(hist_vsn);

            LayerParameter hist_vsh_reshape_after = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_hist_a");
            hist_vsh_reshape_after.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.AFTER;
            hist_vsh_reshape_after.reshape_temporal_param.enable_clip_output = true;
            hist_vsh_reshape_after.bottom.Add("selected_hist1");
            hist_vsh_reshape_after.top.Add("selected_hist");
            hist_vsh_reshape_after.top.Add("selected_hist_clip");
            p.layer.Add(hist_vsh_reshape_after);

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Test the forward pass for sequence processing
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run test_2_variableselectionnetwork_hist2.py on fresh 'test\iter_0' data
        /// 
        /// Fresh test\iter_0 data generated by running:
        /// training.py with TemporalFusionTransformer options: debug=True, tag='tft', use_mycaffe=True
        /// </remarks>
        public void TestForwardHistorical()
        {
            string strPath = getTestDataPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nStateSize = 64;
            int nNumHistNumeric = 4;
            int nNumHistCategorical = 7;
            int nNumHistSteps = 90;
            float fDropout = 0;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel_historical(nNumSamples, fDropout, nStateSize, nNumHistSteps, nNumHistNumeric, nNumHistCategorical);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, null, null);

                int nIdx = 0;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.skip_layer.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.skip_layer.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.fc1.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.fc1.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.context_projection.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.fc2.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.fc2.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.gate.module.fc1.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.gate.module.fc1.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.gate.module.fc2.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.gate.module.fc2.bias.npy");
                nIdx++;

                for (int i = 0; i < nNumHistNumeric + nNumHistCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }

                blob1 = net.FindBlob("hist_ts_rep");
                blob1.LoadFromNumpy(strPath + "tft.historical_ts_rep.npy");
                blob1 = net.FindBlob("c_selection");
                blob1.LoadFromNumpy(strPath + "tft.c_selection.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.vsn.selected_historical.npy");
                blob1 = net.FindBlob("selected_hist");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 8e-07 : 1e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.vsn.selected_historical_wts.npy");
                blob1 = net.FindBlob("hist_wts");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 5e-07 : 1e-06), "The blobs are different!");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }

        /// <summary>
        /// Test the backward pass for sequence processing
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run test_2_variableselectionnetwork_hist2.py on fresh 'test\iter_0' data
        /// 
        /// Fresh test\iter_0 data generated by running:
        /// training.py with TemporalFusionTransformer options: debug=True, tag='tft', use_mycaffe=True
        /// </remarks>
        public void TestBackwardHistorical()
        {
            string strPath = getTestDataPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nStateSize = 64;
            int nNumHistNumeric = 4;
            int nNumHistCategorical = 7;
            int nNumHistSteps = 90;
            float fDropout = 0;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel_historical(nNumSamples, fDropout, nStateSize, nNumHistSteps, nNumHistNumeric, nNumHistCategorical);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);
                param.force_backward = true;

                net = new Net<T>(m_cuda, m_log, param, null, null);

                int nIdx = 0;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.skip_layer.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.skip_layer.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.fc1.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.fc1.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.context_projection.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.fc2.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.fc2.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.gate.module.fc1.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.gate.module.fc1.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.gate.module.fc2.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.flattened_grn.gate.module.fc2.bias.npy");
                nIdx++;

                for (int i = 0; i < nNumHistNumeric + nNumHistCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.hist.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }

                blob1 = net.FindBlob("hist_ts_rep");
                blob1.LoadFromNumpy(strPath + "tft.historical_ts_rep.npy");
                blob1 = net.FindBlob("c_selection");
                blob1.LoadFromNumpy(strPath + "tft.c_selection.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.vsn.selected_historical.npy");
                blob1 = net.FindBlob("selected_hist");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 8e-07 : 1e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.vsn.selected_historical_wts.npy");
                blob1 = net.FindBlob("hist_wts");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 5e-07 : 1e-06), "The blobs are different!");

                //*** BACKWARD ***

                blob1 = net.FindBlob("selected_hist");
                blob1.LoadFromNumpy(strPath + "tft.selected_historical.grad.npy", true);

                net.Backward();

                blob1 = net.FindBlob("hist_ts_rep");
                blobVal.LoadFromNumpy(strPath + "tft.vsn.historical_rep1.grad.npy", true);
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, 3e-7), "The grads are different!");

                blob1 = net.FindBlob("c_selection");
                blobVal.LoadFromNumpy(strPath + "tft.vsn.c_selection1.grad.npy", true);
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, 2e-05), "The grads are different!");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }

        private string buildModel_future(int nNumSamples, float fDropout, int nStateSize, int nNumFutureSteps,
            int nNumFutureNumeric, int nNumFutureCategorical)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";

            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFutureSteps, (nNumFutureNumeric + nNumFutureCategorical) * nStateSize }));
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nStateSize }));
            input.top.Add("future_ts_rep");
            input.top.Add("c_selection");
            p.layer.Add(input);

            //---------------------------------
            //  Variable Selection Networks - Temporal
            //---------------------------------
            LayerParameter future_vsh_reshape_before = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_future_b");
            future_vsh_reshape_before.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.BEFORE;
            future_vsh_reshape_before.bottom.Add("future_ts_rep");
            future_vsh_reshape_before.bottom.Add("c_selection");
            future_vsh_reshape_before.top.Add("future_ts_rep1");
            future_vsh_reshape_before.top.Add("c_selection1");
            p.layer.Add(future_vsh_reshape_before);

            LayerParameter future_vsn = new LayerParameter(LayerParameter.LayerType.VARSELNET, "future_vsn");
            future_vsn.varselnet_param.input_dim = nStateSize;
            future_vsn.varselnet_param.num_inputs = nNumFutureNumeric + nNumFutureCategorical;
            future_vsn.varselnet_param.hidden_dim = nStateSize;
            future_vsn.varselnet_param.dropout = fDropout;
            future_vsn.varselnet_param.context_dim = nStateSize;
            future_vsn.bottom.Add("future_ts_rep1");
            future_vsn.bottom.Add("c_selection1");
            future_vsn.top.Add("selected_future1");
            future_vsn.top.Add("future_wts");
            p.layer.Add(future_vsn);

            LayerParameter future_vsh_reshape_after = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_future_a");
            future_vsh_reshape_after.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.AFTER;
            future_vsh_reshape_after.reshape_temporal_param.enable_clip_output = true;
            future_vsh_reshape_after.bottom.Add("selected_future1");
            future_vsh_reshape_after.top.Add("selected_future");
            future_vsh_reshape_after.top.Add("selected_future_clip");
            p.layer.Add(future_vsh_reshape_after);

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Test the forward pass for sequence processing
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run test_2_variableselectionnetwork_future2.py on fresh 'test\iter_0' data
        /// 
        /// Fresh test\iter_0 data generated by running:
        /// training.py with TemporalFusionTransformer options: debug=True, tag='tft', use_mycaffe=True
        /// </remarks>
        public void TestForwardFuture()
        {
            string strPath = getTestDataPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nStateSize = 64;
            int nNumFutureNumeric = 1;
            int nNumFutureCategorical = 7;
            int nNumFutureSteps = 30;
            float fDropout = 0;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel_future(nNumSamples, fDropout, nStateSize, nNumFutureSteps, nNumFutureNumeric, nNumFutureCategorical);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, null, null);

                int nIdx = 0;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.skip_layer.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.skip_layer.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc1.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc1.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.context_projection.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc2.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc2.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc1.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc1.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc2.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc2.bias.npy");
                nIdx++;

                for (int i = 0; i < nNumFutureNumeric + nNumFutureCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }

                blob1 = net.FindBlob("future_ts_rep");
                blob1.LoadFromNumpy(strPath + "tft.future_ts_rep.npy");
                blob1 = net.FindBlob("c_selection");
                blob1.LoadFromNumpy(strPath + "tft.c_selection.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.vsn.selected_future.npy");
                blob1 = net.FindBlob("selected_future");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 8e-07 : 1e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.vsn.selected_future_wts.npy");
                blob1 = net.FindBlob("future_wts");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 6e-07 : 1e-06), "The blobs are different!");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }

        /// <summary>
        /// Test the backward pass for sequence processing
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run test_2_variableselectionnetwork_future2.py on fresh 'test\iter_0' data
        /// 
        /// Fresh test\iter_0 data generated by running:
        /// training.py with TemporalFusionTransformer options: debug=True, tag='tft', use_mycaffe=True
        /// </remarks>
        public void TestBackwardFuture()
        {
            string strPath = getTestDataPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nStateSize = 64;
            int nNumFutureNumeric = 1;
            int nNumFutureCategorical = 7;
            int nNumFutureSteps = 30;
            float fDropout = 0;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel_future(nNumSamples, fDropout, nStateSize, nNumFutureSteps, nNumFutureNumeric, nNumFutureCategorical);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, null, null);

                int nIdx = 0;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.skip_layer.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.skip_layer.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc1.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc1.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.context_projection.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc2.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc2.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc1.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc1.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc2.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc2.bias.npy");
                nIdx++;

                for (int i = 0; i < nNumFutureNumeric + nNumFutureCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }

                blob1 = net.FindBlob("future_ts_rep");
                blob1.LoadFromNumpy(strPath + "tft.future_ts_rep.npy");
                blob1 = net.FindBlob("c_selection");
                blob1.LoadFromNumpy(strPath + "tft.c_selection.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.vsn.selected_future.npy");
                blob1 = net.FindBlob("selected_future");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 8e-07 : 1e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.vsn.selected_future_wts.npy");
                blob1 = net.FindBlob("future_wts");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 5e-07 : 1e-06), "The blobs are different!");

                //*** BACKWARD ***

                blob1 = net.FindBlob("selected_future");
                blob1.LoadFromNumpy(strPath + "tft.selected_future.grad.npy", true);

                net.Backward();

                blob1 = net.FindBlob("future_ts_rep");
                blobVal.LoadFromNumpy(strPath + "tft.vsn.future_rep1.grad.npy", true);
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, 3e-7), "The grads are different!");

                blob1 = net.FindBlob("c_selection");
                blobVal.LoadFromNumpy(strPath + "tft.vsn.c_selection1.grad.npy", true);
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, 2e-05), "The grads are different!");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }
    }
}
