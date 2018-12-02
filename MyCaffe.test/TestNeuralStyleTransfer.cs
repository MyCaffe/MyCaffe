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
                    if (t.DataType == DataType.DOUBLE)
                        t.TestNeuralStyleTransfer(1000, "vgg19");
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
        void TestNeuralStyleTransfer(int nIteration, string strName);
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

    public class NeuralStyleTransferTest<T> : TestEx<T>, INeuralStyleTransferTest
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

        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        private Tuple<string, string, string> getModel(string strName)
        {
            string strModelFile;
            string strWtsFile;

            switch (strName)
            {
                case "vgg19":
                    strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\vgg\\neuralstyle\\deploy.prototxt");
                    strWtsFile = getTestPath("\\MyCaffe\\test_data\\models\\vgg\\neuralstyle\\weights.caffemodel");
                    break;

                case "googlenet":
                    strModelFile = getTestPath("\\MyCaffe\\test_data\\models\\goognet\\neuralstyle\\train_val.prototxt");
                    strWtsFile = getTestPath("\\MyCaffe\\test_data\\models\\goognet\\neuralstyle\\weights.caffemodel");
                    break;

                default:
                    throw new Exception("Unknown model name '" + strName + "'");
            }

            return new Tuple<string, string, string>(strName, strModelFile, strWtsFile);
        }

        /// <summary>
        /// The NeuralStyleTransfer test is based on implementing the Neural Style Transfer algorithm
        /// from https://github.com/ftokarev/caffe-neural-style/blob/master/neural-style.py
        /// using MyCaffe.
        /// </summary>
        public void TestNeuralStyleTransfer(int nIterations, string strName)
        {
            CancelEvent evtCancel = new CancelEvent();
            Tuple<string, string, string> info = getModel(strName);
            string strModelName = info.Item1;
            string strModelFile = info.Item2;
            string strWeightFile = info.Item3;
            string strDataDir = getTestPath("\\MyCaffe\\test_data\\data\\images\\", true);
            string strStyleImg = strDataDir + "style\\style.png";
            string strContentImg = strDataDir + "content\\content.png";
            string strResultDir = strDataDir + "result\\";
            byte[] rgWeights = null;
            string strModelDesc = "";

            if (File.Exists(strWeightFile))
            {
                using (FileStream fs = new FileStream(strWeightFile, FileMode.Open, FileAccess.Read))
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

            NeuralStyleTransfer<T> ns = new NeuralStyleTransfer<T>(m_cuda, m_log, m_evtCancel, strModelName, strModelDesc, rgWeights, false);

            if (!Directory.Exists(strResultDir))
                Directory.CreateDirectory(strResultDir);

            Bitmap bmpStyle = new Bitmap(strStyleImg);
            Bitmap bmpContent = new Bitmap(strContentImg);

            Bitmap bmpResult = ns.Process(bmpStyle, bmpContent, nIterations, strResultDir, 100);

            string strResultFile = strResultDir + nIterations.ToString() + "_result.png";

            if (File.Exists(strResultFile))
                File.Delete(strResultFile);

            bmpResult.Save(strResultFile, ImageFormat.Png);
        }
    }
}
