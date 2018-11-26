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
using System.Drawing;
using System.Collections;
using MyCaffe.data;
using System.Diagnostics;
using System.IO;
using System.Drawing.Imaging;
using MyCaffe.extras;

namespace MyCaffe.test
{
    [TestClass]
    public class TestNeuralStyleTransfer
    {
        [TestMethod]
        public void TestNeuralStyleTransfer1()
        {
            NeuralStyleTransferTest test = new NeuralStyleTransferTest();

            try
            {
                foreach (INeuralStyleTransferTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestNeuralStyleTransfer();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface INeuralStyleTransferTest : ITest
    {
        void TestNeuralStyleTransfer();
    }

    class NeuralStyleTransferTest : TestBase
    {
        public NeuralStyleTransferTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Neural Style Transfer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new NeuralStyleTransferTest<double>(strName, nDeviceID, engine);
            else
                return new NeuralStyleTransferTest<float>(strName, nDeviceID, engine);
        }
    }

    class NeuralStyleTransferTest<T> : TestEx<T>, INeuralStyleTransferTest
    {
        SettingsCaffe m_settings = new SettingsCaffe();
        CancelEvent m_evtCancel = new CancelEvent();
        MyCaffeControl<T> m_caffe = null;

        public NeuralStyleTransferTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            m_settings.GpuIds = nDeviceID.ToString();
            m_caffe = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel);
            m_log.EnableTrace = true;
        }

        protected override void dispose()
        {
            m_caffe.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        /// <summary>
        /// The NeuralStyleTransfer test is based on implementing the Neural Style Transfer algorithm
        /// from https://github.com/ftokarev/caffe-neural-style/blob/master/neural-style.py
        /// using MyCaffe.
        /// </summary>
        /// 
        public void TestNeuralStyleTransfer()
        {
            CancelEvent evtCancel = new CancelEvent();
            List<string> rgContentLayers = new List<string>() { "conv4_2" };
            List<string> rgStyleLayers = new List<string>() { "conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1" };
            string strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\vgg\\VGG_ILSVRC_19_layers_deploy.prototxt");
            string strWtsFile = getTestPath("\\MyCaffe\\test_data\\models\\vgg\\VGG_ILSVRC_19_layers.caffemodel");
            string strDataDir = getTestPath("\\MyCaffe\\test_data\\images\\", true, true, true);
            string strStyleImg = strDataDir + "style\\starry_night.jpg";
            string strContentImg = strDataDir + "content\\nanjing.jpg";
            string strResultDir = strDataDir + "result\\";
            byte[] rgWeights = null;
            string strModelDesc = "";

            if (File.Exists(strWtsFile))
            {
                using (FileStream fs = new FileStream(strWtsFile, FileMode.Open, FileAccess.Read))
                {
                    using (BinaryReader br = new BinaryReader(fs))
                    {
                        rgWeights = br.ReadBytes((int)fs.Length);
                    }
                }
            }

            using (StreamReader sr = new StreamReader(strModelFile))
            {
                strModelDesc = sr.ReadToEnd();
            }

            RawProto proto = RawProto.Parse(strModelDesc);
            NetParameter net_param = NetParameter.FromProto(proto);

            NeuralStyleTransfer<T> ns = new NeuralStyleTransfer<T>(m_cuda, m_log, net_param, rgContentLayers, rgStyleLayers, evtCancel);

            Bitmap bmpStyle = new Bitmap(strStyleImg);
            Bitmap bmpContent = new Bitmap(strContentImg);
            Bitmap bmpResult = ns.Process(bmpStyle, bmpContent, 1000);

            if (!Directory.Exists(strResultDir))
                Directory.CreateDirectory(strResultDir);

            bmpResult.Save(strResultDir + "result.png", ImageFormat.Png);
        }
    }
}
