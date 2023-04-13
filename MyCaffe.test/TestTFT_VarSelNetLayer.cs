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

        // WORK IN PROGRESS
        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VARSELNET);
            p.varselnet_param.input_dim = 64;
            p.varselnet_param.axis = 1;
            Layer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.VARSELNET, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "VarSelNet_x.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "test1_VarSelNet.fc1.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "test1_VarSelNet.fc1.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "test1_VarSelNet.fc2.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "test1_VarSelNet.fc2.bias.npy");

                layer.Forward(BottomVec, TopVec);

                blobYexp.LoadFromNumpy(strPath + "test1_VarSelNet_y.npy");
                double dfErr = (typeof(T) == typeof(float)) ? 4e-07 : 4.5e-07;
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

        // WORK IN PROGRESS
        public void TestBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VARSELNET);
            p.varselnet_param.input_dim = 64;
            p.varselnet_param.axis = 1;
            Layer<T> layer = null;
            Blob<T> blobGradExp = null;
            Blob<T> blobX = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobGradExp = new Blob<T>(m_cuda, m_log);
                blobX = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.VARSELNET, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "VarSelNet_x.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "test1_VarSelNet.fc1.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "test1_VarSelNet.fc1.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "test1_VarSelNet.fc2.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "test1_VarSelNet.fc2.bias.npy");

                layer.Forward(BottomVec, TopVec);

                blobYexp.LoadFromNumpy(strPath + "test1_VarSelNet_y.npy");
                double dfErr = (typeof(T) == typeof(float)) ? 4e-07 : 4.5e-07;
                m_log.CHECK(TopVec[0].Compare(blobYexp, blobWork, false, dfErr), "The blobs do not match.");

                TopVec[0].LoadFromNumpy(strPath + "test1_VarSelNet_y.grad.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                blobGradExp.LoadFromNumpy(strPath + "test1_VarSelNet_x.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(blobX, blobWork, true, dfErr), "The blobs do not match.");

                if (typeof(T) == typeof(double))
                    dfErr = 0.03;

                blobGradExp.LoadFromNumpy(strPath + "test1_VarSelNet.fc1.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[0], blobWork, true, 1e-06), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "test1_VarSelNet.fc1.bias.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[1], blobWork, true, 2e-06), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "test1_VarSelNet.fc2.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[2], blobWork, true, (typeof(T) == typeof(float)) ? 1e-8 : 2.5e-06), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "test1_VarSelNet.fc2.bias.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[3], blobWork, true, 2e-05), "The blobs do not match.");
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

        // WORK IN PROGRESS
        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VARSELNET);
            p.varselnet_param.input_dim = 64;
            p.varselnet_param.axis = 1;
            Layer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobX = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.VARSELNET, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "VarSelNet_x.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "test1_VarSelNet.fc1.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "test1_VarSelNet.fc1.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "test1_VarSelNet.fc2.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "test1_VarSelNet.fc2.bias.npy");

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
