﻿using System;
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
using System.IO;

/// <summary>
/// Testing the Glu layer.
/// 
/// Glu Layer - layer calculate gated linear unit.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_GluLayer
    {
        [TestMethod]
        public void TestForward()
        {
            GluLayerTest test = new GluLayerTest();

            try
            {
                foreach (IGluLayerTest t in test.Tests)
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
            GluLayerTest test = new GluLayerTest();

            try
            {
                foreach (IGluLayerTest t in test.Tests)
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
            GluLayerTest test = new GluLayerTest();

            try
            {
                foreach (IGluLayerTest t in test.Tests)
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

    interface IGluLayerTest : ITest
    {
        void TestForward();
        void TestBackward();
        void TestGradient();
    }

    class GluLayerTest : TestBase
    {
        public GluLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Glu Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new GluLayerTest<double>(strName, nDeviceID, engine);
            else
                return new GluLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class GluLayerTest<T> : TestEx<T>, IGluLayerTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();
        int m_nNumOutput = 3;
        int m_nBatchSize;
        int m_nVectorDim;

        public GluLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        private string getTestDataPath(string strSubPath)
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\test\\" + strSubPath + "\\iter_0\\";
            //return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\" + strSubPath + "\\iter_0\\";
        }

        private string getTestWtsPath(string strSubPath)
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\favorita\\weights\\" + strSubPath + "\\";
            //return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\hist_ts_transform\\";
        }

        private void verifyFileDownload(string strSubPath, string strFile)
        {
            string strPath = getTestDataPath(strSubPath);
            if (!File.Exists(strPath + strFile))
                throw new Exception("ERROR: You need to download the TFT test data by running the MyCaffe Test Application and selecting the 'Download Test Data | TFT' menu.");
        }

        /// <summary>
        /// Test GLU foward
        /// </summary>
        /// <remarks>
        /// To generate the test data, run the following:
        /// 
        /// Code: test_8a_glu_imha.py
        /// Path: imha
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GLU);
            p.glu_param.input_dim = 64;
            p.glu_param.axis = 1;
            GluLayer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath("imha");
            string strPathWts = getTestWtsPath("hist_ts_transform");

            verifyFileDownload("imha", "tft.asa.gan_gan_glu_x.npy");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as GluLayer<T>;
                blobX = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.GLU, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "tft.asa.gan_gan_glu_x.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc1.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc1.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc2.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc2.bias.npy");

                layer.Forward(BottomVec, TopVec);

                blobYexp.LoadFromNumpy(strPath + "tft.asa.gan_gan_glu_y.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs do not match.");
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

        /// <summary>
        /// Test GLU backward
        /// </summary>
        /// <remarks>
        /// To generate the test data, run the following:
        /// 
        /// Code: test_8a_glu_imha.py
        /// Path: imha
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GLU);
            p.glu_param.input_dim = 64;
            p.glu_param.axis = 1;
            GluLayer<T> layer = null;
            Blob<T> blobGradExp = null;
            Blob<T> blobX = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath("imha");
            string strPathWts = getTestWtsPath("hist_ts_transform");

            verifyFileDownload("imha", "tft.asa.gan_gan_glu_x.npy");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as GluLayer<T>;
                blobGradExp = new Blob<T>(m_cuda, m_log);
                blobX = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.GLU, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "tft.asa.gan_gan_glu_x.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc1.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc1.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc2.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc2.bias.npy");

                layer.Forward(BottomVec, TopVec);

                blobYexp.LoadFromNumpy(strPath + "tft.asa.gan_gan_glu_y.npy");
                m_log.CHECK(TopVec[0].Compare(blobYexp, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs do not match.");

                //** BACKWARD **

                TopVec[0].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu_y.grad.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                blobGradExp.LoadFromNumpy(strPath + "test1_glu_x.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(blobX, blobWork, true, 2e-08), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc1.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[0], blobWork, true, (typeof(T) == typeof(float)) ? 1e-06 : 3e-05), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc1.bias.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[1], blobWork, true, 5e-06), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc2.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[2], blobWork, true, (typeof(T) == typeof(float)) ? 1e-8 : 7e-05), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc2.bias.grad.npy", true);
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

        /// <summary>
        /// Test GLU gradient check
        /// </summary>
        /// <remarks>
        /// To generate the test data, run the following:
        /// 
        /// Code: test_8a_glu_imha.py
        /// Path: imha
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GLU);
            p.glu_param.input_dim = 64;
            p.glu_param.axis = 1;
            GluLayer<T> layer = null;
            Blob<T> blobX = null;
            Blob<T> blobY = null;
            Blob<T> blobYexp = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath("imha");
            string strPathWts = getTestWtsPath("hist_ts_transform");

            verifyFileDownload("imha", "tft.asa.gan_gan_glu_x.npy");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as GluLayer<T>;
                blobX = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobYexp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.GLU, "The layer type is incorrect.");

                blobX.LoadFromNumpy(strPath + "tft.asa.gan_gan_glu_x.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc1.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc1.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc2.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "tft.asa.gan_gan_glu.internal.fc2.bias.npy");

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
