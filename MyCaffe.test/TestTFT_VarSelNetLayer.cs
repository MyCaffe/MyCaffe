using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.param;
using MyCaffe.param.tft;
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
        public void TestBackwardFuture()
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

        private string getTestDataPathBase()
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\test\\iter_0.base_set\\";
            //return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\iter_0.base_set\\";
        }

        private string getTestDataPath(string strSubPath)
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\test\\" + strSubPath + "\\iter_0\\";
            //return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\" + strSubPath + "\\iter_0\\";
        }

        private string getTestWtsPath(string strSubPath)
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\favorita\\weights\\" + strSubPath + "\\";
            //return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\" + strSubPath + "\\";
        }

        /// <summary>
        /// Test VarSelNet future focused forward pass.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_2_variableselectionnetwork_fut.py
        /// Target Dir: var_fut
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VARSELNET);
            p.varselnet_param.input_dim = 64;
            p.varselnet_param.hidden_dim = 64;
            p.varselnet_param.num_inputs = 8;
            p.varselnet_param.dropout_ratio = 0;
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
            string strPath = getTestDataPath("vsn_fut");
            string strPathWts = getTestWtsPath("future_ts_selection");

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

                blobX1.LoadFromNumpy(strPath + "tft.vsn.future_varsel_flattened_embedding.npy");
                blobX2.LoadFromNumpy(strPath + "tft.vsn.future_varsel_context.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX1);
                BottomVec.Add(blobX2);
                TopVec.Clear();
                TopVec.Add(blobY1);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.skip_layer.module.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.skip_layer.module.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc1.module.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc1.module.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.context_projection.module.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc2.module.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc2.module.bias.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc1.weight.npy");
                layer.blobs[8].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc1.bias.npy");
                layer.blobs[9].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc2.weight.npy");
                layer.blobs[10].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc2.bias.npy");

                int nIdx = 11;
                for (int i = 0; i < p.varselnet_param.num_inputs; i++)
                {
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }                

                layer.Forward(BottomVec, TopVec);

                blobYexp1.LoadFromNumpy(strPath + "tft.vsn.future_varsel_outputs_sum.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp1, blobWork, false, (typeof(T) == typeof(float)) ? 4e-07 : 4e-06), "The blobs do not match.");
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

        /// <summary>
        /// Test VarSelNet future focused backward pass.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_2_variableselectionnetwork_fut.py
        /// Target Dir: var_fut
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VARSELNET);
            p.varselnet_param.input_dim = 64;
            p.varselnet_param.hidden_dim = 64;
            p.varselnet_param.num_inputs = 8;
            p.varselnet_param.dropout_ratio = 0;
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
            string strPath = getTestDataPath("vsn_fut");
            string strPathWts = getTestWtsPath("");

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

                blobX1.LoadFromNumpy(strPath + "tft.vsn.future_varsel_flattened_embedding.npy");
                blobX2.LoadFromNumpy(strPath + "tft.vsn.future_varsel_context.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX1);
                BottomVec.Add(blobX2);
                TopVec.Clear();
                TopVec.Add(blobY1);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.skip_layer.module.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.skip_layer.module.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc1.module.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc1.module.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.context_projection.module.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc2.module.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc2.module.bias.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc1.weight.npy");
                layer.blobs[8].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc1.bias.npy");
                layer.blobs[9].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc2.weight.npy");
                layer.blobs[10].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc2.bias.npy");

                int nIdx = 11;
                for (int i = 0; i < p.varselnet_param.num_inputs; i++)
                {
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }

                layer.Forward(BottomVec, TopVec);

                blobYexp1.LoadFromNumpy(strPath + "tft.vsn.future_varsel_outputs_sum.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp1, blobWork, false, (typeof(T) == typeof(float)) ? 4e-07 : 4e-06), "The blobs do not match.");

                //** BACKWARD **

                TopVec[0].LoadFromNumpy(strPath + "tft.vsn.future_varsel_outputs_sum.grad.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                blobGradExp.LoadFromNumpy(strPath + "tft.vsn.future_varsel_flattened_embedding.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(blobX1, blobWork, true, (typeof(T) == typeof(float)) ? 3e-08 : 3e-06), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "tft.vsn.future_varsel_context.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(blobX2, blobWork, true), "The blobs do not match.");
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

        /// <summary>
        /// Test VarSelNet future focused gradient check.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_2_variableselectionnetwork_fut.py
        /// Target Dir: var_fut
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VARSELNET);
            p.varselnet_param.input_dim = 64;
            p.varselnet_param.hidden_dim = 64;
            p.varselnet_param.num_inputs = 8;
            p.varselnet_param.dropout_ratio = 0;
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
            string strPath = getTestDataPath("vsn_fut");
            string strPathWts = getTestWtsPath("");

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

                blobX1.LoadFromNumpy(strPath + "tft.vsn.future_varsel_flattened_embedding.npy");
                blobX2.LoadFromNumpy(strPath + "tft.vsn.future_varsel_context.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX1);
                BottomVec.Add(blobX2);
                TopVec.Clear();
                TopVec.Add(blobY1);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.skip_layer.module.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.skip_layer.module.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc1.module.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc1.module.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.context_projection.module.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc2.module.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.fc2.module.bias.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc1.weight.npy");
                layer.blobs[8].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc1.bias.npy");
                layer.blobs[9].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc2.weight.npy");
                layer.blobs[10].LoadFromNumpy(strPath + "tft.vsn.future.flattened_grn.gate.module.fc2.bias.npy");

                int nIdx = 11;
                for (int i = 0; i < p.varselnet_param.num_inputs; i++)
                {
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    layer.blobs[nIdx].LoadFromNumpy(strPath + "tft.vsn.future.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
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
            static_vsn.varselnet_param.dropout_ratio = fDropout;
            static_vsn.bottom.Add("static_rep");
            static_vsn.top.Add("selected_static");
            p.layer.Add(static_vsn);

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Test VarSelNet historical focused forward pass.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_2_variableselectionnetwork_stat.py
        /// Target Dir: var_fut
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestForwardStatic()
        {
            string strPath = getTestDataPath("vsn_stat");
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
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.static_rep.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "ZZZ.vsn.selected_static.npy");
                blob1 = net.FindBlob("selected_static");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 3e-07 : 5e-07), "The blobs are different!");
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
        /// Test VarSelNet historical focused backward pass.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_2_variableselectionnetwork_stat.py
        /// Target Dir: var_fut
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestBackwardStatic()
        {
            string strPath = getTestDataPath("vsn_stat");
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
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.static_rep.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "ZZZ.vsn.selected_static.npy");
                blob1 = net.FindBlob("selected_static");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 3e-07 : 5e-07), "The blobs are different!");

                //*** BACKWARD ***

                blob1 = net.FindBlob("selected_static");
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.selected_static.grad.npy", true);

                net.Backward();

                blobVal.LoadFromNumpy(strPath + "ZZZ.vsn.static_rep.grad.npy", true);
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
            int nNumHistNumeric, int nNumHistCategorical)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";

            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHistSteps, (nNumHistNumeric + nNumHistCategorical) * nStateSize }));
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nStateSize }));
            input.top.Add("hist_ts_rep");
            input.top.Add("c_selection_h");
            p.layer.Add(input);

            //---------------------------------
            //  Variable Selection Networks - Temporal
            //---------------------------------
            LayerParameter hist_vsh_reshape_before = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_hist_b");
            hist_vsh_reshape_before.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.BEFORE;
            hist_vsh_reshape_before.bottom.Add("hist_ts_rep");
            hist_vsh_reshape_before.bottom.Add("c_selection_h");
            hist_vsh_reshape_before.top.Add("hist_ts_rep1");
            hist_vsh_reshape_before.top.Add("c_selection1h");
            p.layer.Add(hist_vsh_reshape_before);

            LayerParameter hist_vsn = new LayerParameter(LayerParameter.LayerType.VARSELNET, "hist_vsn");
            hist_vsn.varselnet_param.input_dim = nStateSize;
            hist_vsn.varselnet_param.num_inputs = nNumHistNumeric + nNumHistCategorical;
            hist_vsn.varselnet_param.hidden_dim = nStateSize;
            hist_vsn.varselnet_param.dropout_ratio = fDropout;
            hist_vsn.varselnet_param.context_dim = nStateSize;
            hist_vsn.bottom.Add("hist_ts_rep1");
            hist_vsn.bottom.Add("c_selection1h");
            hist_vsn.top.Add("selected_hist1");
            p.layer.Add(hist_vsn);

            LayerParameter hist_vsh_reshape_after = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_hist_a");
            hist_vsh_reshape_after.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.AFTER;
            hist_vsh_reshape_after.reshape_temporal_param.enable_clip_output = true;
            hist_vsh_reshape_after.bottom.Add("selected_hist1");
            hist_vsh_reshape_after.top.Add("selected_hist");
            hist_vsh_reshape_after.top.Add("selected_hist_clip");
            p.layer.Add(hist_vsh_reshape_after);

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Test VarSelNet historical focused forward pass.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_2_variableselectionnetwork_hist.py
        /// Target Dir: var_fut
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestForwardHistorical()
        {
            string strPath = getTestDataPath("vsn_hist");
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
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.historical.temporal_representation.npy");
                blob1 = net.FindBlob("c_selection_h");
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.historical.static_selection_signal.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "ZZZ.vsn.historical.temporal_selection_output.npy");
                blob1 = net.FindBlob("selected_hist");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 4e-07 : 4e-05), "The blobs are different!");
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
        /// Test VarSelNet historical focused backward pass.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_2_variableselectionnetwork_hist.py
        /// Target Dir: var_fut
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestBackwardHistorical()
        {
            string strPath = getTestDataPath("vsn_hist");
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
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.historical.temporal_representation.npy");
                blob1 = net.FindBlob("c_selection_h");
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.historical.static_selection_signal.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "ZZZ.vsn.historical.temporal_selection_output.npy");
                blob1 = net.FindBlob("selected_hist");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 4e-07 : 4e-05), "The blobs are different!");

                //*** BACKWARD ***

                blob1 = net.FindBlob("selected_hist");
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.historical.temporal_selection_output.grad.npy", true);

                net.Backward();

                blob1 = net.FindBlob("hist_ts_rep");
                blobVal.LoadFromNumpy(strPath + "ZZZ.vsn.historical.temporal_representation.grad.npy", true);
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, (typeof(T) == typeof(float)) ? 1e-08 : 1e-07), "The grads are different!");

                blob1 = net.FindBlob("c_selection_h");
                blobVal.LoadFromNumpy(strPath + "ZZZ.vsn.historical.static_selection_signal.grad.npy", true);
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

        private string buildModel_future(int nNumSamples, float fDropout, int nStateSize, int nNumFutureSteps,
            int nNumFutureNumeric, int nNumFutureCategorical)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";

            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFutureSteps, (nNumFutureNumeric + nNumFutureCategorical) * nStateSize }));
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nStateSize }));
            input.top.Add("future_ts_rep");
            input.top.Add("c_selection_f");
            p.layer.Add(input);

            //---------------------------------
            //  Variable Selection Networks - Temporal
            //---------------------------------
            LayerParameter future_vsh_reshape_before = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_fut_b");
            future_vsh_reshape_before.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.BEFORE;
            future_vsh_reshape_before.bottom.Add("future_ts_rep");
            future_vsh_reshape_before.bottom.Add("c_selection_f");
            future_vsh_reshape_before.top.Add("future_ts_rep1");
            future_vsh_reshape_before.top.Add("c_selection1f");
            p.layer.Add(future_vsh_reshape_before);

            LayerParameter fut_vsn = new LayerParameter(LayerParameter.LayerType.VARSELNET, "future_vsn");
            fut_vsn.varselnet_param.input_dim = nStateSize;
            fut_vsn.varselnet_param.num_inputs = nNumFutureNumeric + nNumFutureCategorical;
            fut_vsn.varselnet_param.hidden_dim = nStateSize;
            fut_vsn.varselnet_param.dropout_ratio = fDropout;
            fut_vsn.varselnet_param.context_dim = nStateSize;
            fut_vsn.bottom.Add("future_ts_rep1");
            fut_vsn.bottom.Add("c_selection1f");
            fut_vsn.top.Add("selected_fut1");
            p.layer.Add(fut_vsn);

            LayerParameter future_vsh_reshape_after = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_fut_a");
            future_vsh_reshape_after.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.AFTER;
            future_vsh_reshape_after.reshape_temporal_param.enable_clip_output = true;
            future_vsh_reshape_after.bottom.Add("selected_fut1");
            future_vsh_reshape_after.top.Add("selected_fut");
            future_vsh_reshape_after.top.Add("selected_fut_clip");
            p.layer.Add(future_vsh_reshape_after);

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Test VarSelNet future focused forward pass.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_2_variableselectionnetwork_fut.py
        /// Target Dir: var_fut
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestForwardFuture()
        {
            string strPath = getTestDataPath("vsn_fut");
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
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.future.temporal_representation.npy");
                blob1 = net.FindBlob("c_selection_f");
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.future.static_selection_signal.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "ZZZ.vsn.future.temporal_selection_output.npy");
                blob1 = net.FindBlob("selected_fut");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 4e-07 : 4e-06), "The blobs are different!");
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
        /// Test VarSelNet future focused backward pass.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_2_variableselectionnetwork_fut.py
        /// Target Dir: var_fut
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestBackwardFuture()
        {
            string strPath = getTestDataPath("vsn_fut");
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
                param.force_backward = true;

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
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.future.temporal_representation.npy");
                blob1 = net.FindBlob("c_selection_f");
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.future.static_selection_signal.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "ZZZ.vsn.future.temporal_selection_output.npy");
                blob1 = net.FindBlob("selected_fut");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 4e-07 : 4e-06), "The blobs are different!");

                //*** BACKWARD ***

                blob1 = net.FindBlob("selected_fut");
                blob1.LoadFromNumpy(strPath + "ZZZ.vsn.future.temporal_selection_output.grad.npy", true);

                net.Backward();

                blob1 = net.FindBlob("future_ts_rep");
                blobVal.LoadFromNumpy(strPath + "ZZZ.vsn.future.temporal_representation.grad.npy", true);
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, (typeof(T) == typeof(float)) ? 3e-08 : 2e-06), "The grads are different!");

                blob1 = net.FindBlob("c_selection_f");
                blobVal.LoadFromNumpy(strPath + "ZZZ.vsn.future.static_selection_signal.grad.npy", true);
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
    }
}
