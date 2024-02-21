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
using MyCaffe.param.beta;
using System.IO;
using MyCaffe.param.gpt;
using System.IO.Compression;
using System.Net;
using System.Diagnostics;
using System.Threading;

namespace MyCaffe.test
{
    [TestClass]
    public class TestGPT_TestRMSNORMLayer
    {
        [TestMethod]
        public void TestGradient()
        {
            RmsNormLayerTest test = new RmsNormLayerTest();

            try
            {
                foreach (IRmsNormLayerTest t in test.Tests)
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
        public void TestForwardEx()
        {
            RmsNormLayerTest test = new RmsNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRmsNormLayerTest t in test.Tests)
                {
                    t.TestForwardEx(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardExCuda()
        {
            RmsNormLayerTest test = new RmsNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRmsNormLayerTest t in test.Tests)
                {
                    t.TestForwardEx(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardEx()
        {
            RmsNormLayerTest test = new RmsNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRmsNormLayerTest t in test.Tests)
                {
                    t.TestBackwardEx(false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardExCuda()
        {
            RmsNormLayerTest test = new RmsNormLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IRmsNormLayerTest t in test.Tests)
                {
                    t.TestBackwardEx(true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IRmsNormLayerTest : ITest
    {
        void TestGradient();

        void TestForwardEx(bool bUseCuda);
        void TestBackwardEx(bool bUseCuda);
    }

    class RmsNormLayerTest : TestBase
    {
        public RmsNormLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("RMSNORM Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new RmsNormLayerTest<double>(strName, nDeviceID, engine);
            else
                return new RmsNormLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class RmsNormLayerTest<T> : TestEx<T>, IRmsNormLayerTest
    {
        Blob<T> m_blobVal;

        public RmsNormLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 3, 1 }, nDeviceID)
        {
            m_engine = engine;

            m_blobWork = new Blob<T>(m_cuda, m_log);
            m_blobVal = new Blob<T>(m_cuda, m_log);
        }

        protected override void dispose()
        {
            dispose(ref m_blobWork);
            dispose(ref m_blobVal);
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestGradient()
        {
            Layer<T> layer = null;

            try
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.RMSNORM);
                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);

                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                if (layer != null)
                    layer.Dispose();
            }
        }


        private string getTestDataPath(string strSubPath, string strFile)
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\gpt\\test\\" + strSubPath + "\\";

            if (!File.Exists(strPath + strFile))
                throw new Exception("Could not find the test data file '" + strPath + strFile + "'.  You may need to run the 'Test|Download Test Data | GPT' menu item.");

            return strPath;
        }

        private string getTestDataBasePath(string strFile)
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\gpt\\test\\";

            if (!File.Exists(strPath + strFile))
                throw new Exception("Could not find the test data file '" + strPath + strFile + "'.  You may need to run the 'Test|Download Test Data | GPT' menu item.");

            return strPath;
        }

        public void TestForwardEx(bool bUseCuda)
        {
            string strTestDataPath = getTestDataPath("rmsnorm", "g.npy");
            Layer<T> layer = null;

            int nBatchSize = 3;
            int nSeqLen = 3;
            int nInputSize = 4;

            try
            {
                Stopwatch sw = new Stopwatch();

                LayerParameter p = new LayerParameter(LayerParameter.LayerType.RMSNORM);
                p.rms_norm_param.epsilon = 1e-5;
                //p.rms_norm_param.enable_cuda_impl = bUseCuda;
                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

                m_log.CHECK(layer.type == LayerParameter.LayerType.RMSNORM, "The layer type is incorrect!");

                m_blob_bottom.LoadFromNumpy(strTestDataPath + "input.npy");
                m_blob_top.SetData(0);

                m_log.CHECK_EQ(m_blob_bottom.num, nBatchSize, "The num should be " + nBatchSize.ToString());
                m_log.CHECK_EQ(m_blob_bottom.channels, nSeqLen, "The channels should be " + nSeqLen.ToString());
                m_log.CHECK_EQ(m_blob_bottom.height, nInputSize, "The height should be " + nInputSize.ToString());
                m_log.CHECK_EQ(m_blob_bottom.width, 1, "The width should be 1");

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_top.num, m_blob_bottom.num, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.channels, m_blob_bottom.channels, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.height, m_blob_bottom.height, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.width, m_blob_bottom.width, "The num does not match!");

                m_blobVal.LoadFromNumpy(strTestDataPath + "output.npy");
                m_log.CHECK(m_blobVal.Compare(m_blob_top, m_blobWork, false, 2e-07), "The blobs are different.");

                sw.Start();
                for (int i = 0; i < 100; i++)
                {
                    layer.Forward(BottomVec, TopVec);
                }
                sw.Stop();

                double dfTime = sw.Elapsed.TotalMilliseconds / 100.0;
                Trace.WriteLine("Time Per Forward = " + dfTime.ToString("N6") + " ms");
            }
            finally
            {
                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestBackwardEx(bool bUseCuda)
        {
            string strTestDataPath = getTestDataPath("rmsnorm", "g.npy");
            Layer<T> layer = null;

            int nBatchSize = 3;
            int nSeqLen = 3;
            int nInputSize = 4;

            try
            {
                Stopwatch sw = new Stopwatch();

                LayerParameter p = new LayerParameter(LayerParameter.LayerType.RMSNORM);
                p.rms_norm_param.epsilon = 1e-5;
                //p.rms_norm_param.enable_cuda_impl = bUseCuda;
                layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

                m_log.CHECK(layer.type == LayerParameter.LayerType.RMSNORM, "The layer type is incorrect!");

                m_blob_bottom.LoadFromNumpy(strTestDataPath + "input.npy");
                m_blob_top.SetData(0);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_blob_top.LoadFromNumpy(strTestDataPath + "output.grad.npy", true);

                List<bool> rgProp = new List<bool>() { true };
                layer.Backward(TopVec, rgProp, BottomVec);

                m_log.CHECK_EQ(m_blob_top.num, m_blob_bottom.num, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.channels, m_blob_bottom.channels, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.height, m_blob_bottom.height, "The num does not match!");
                m_log.CHECK_EQ(m_blob_top.width, m_blob_bottom.width, "The num does not match!");

                m_blobVal.LoadFromNumpy(strTestDataPath + "x.grad.npy", true);
                m_log.CHECK(m_blobVal.Compare(m_blob_bottom, m_blobWork, true, 6e-08), "The blobs are different.");

                sw.Start();
                for (int i = 0; i < 100; i++)
                {
                    layer.Backward(TopVec, rgProp, BottomVec);
                }
                sw.Stop();
                double dfTime = sw.Elapsed.TotalMilliseconds / 100.0;
                Trace.WriteLine("Time Per Backward = " + dfTime.ToString("N6") + " ms");
            }
            finally
            {
                if (layer != null)
                    layer.Dispose();
            }
        }
    }
}
