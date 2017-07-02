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

namespace MyCaffe.test
{
    [TestClass]
    public class TestDeepDraw
    {
        [TestMethod]
        public void TestDeepDraw1()
        {
            DeepDrawTest test = new DeepDrawTest();

            try
            {
                foreach (IDeepDrawTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestDeepDraw1();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDeepDraw2()
        {
            DeepDrawTest test = new DeepDrawTest();

            try
            {
                foreach (IDeepDrawTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestDeepDraw2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IDeepDrawTest : ITest
    {
        void TestDeepDraw1();
        void TestDeepDraw2();
    }

    class DeepDrawTest : TestBase
    {
        public DeepDrawTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Deep Draw Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new DeepDrawTest<double>(strName, nDeviceID, engine);
            else
                return new DeepDrawTest<float>(strName, nDeviceID, engine);
        }
    }

    class DeepDrawTest<T> : TestEx<T>, IDeepDrawTest
    {
        public DeepDrawTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        /// <summary>
        /// The DeepDraw test is based on implementing the Deep Draw algorithms
        /// from https://github.com/kylemcdonald/deepdream/blob/master/dream.ipynb and
        /// from https://github.com/auduno/deepdraw/blob/master/deepdraw.ipynb and
        /// from http://www.kpkaiser.com/machine-learning/diving-deeper-into-deep-dreams/
        /// using MyCaffe.
        /// </summary>
        /// 
        public void TestDeepDraw1()
        {
            SettingsCaffe settings = new SettingsCaffe();
            Log log = new Log("Deep Draw");
            CancelEvent m_evtCancel = new CancelEvent();
            MyCaffeControl<T> caffe = new MyCaffeControl<T>(settings, log, m_evtCancel);

            log.EnableTrace = true;

            string strModelFile = getTestPath("\\test_data\\models\\bvlc_googlenet\\train_val.prototxt");
            string strFile = getTestPath("\\test_data\\models\\bvlc_googlenet\\bvlc_googlenet.caffemodel");
            byte[] rgWeights = null;
            string strModelDesc = "";

            using (FileStream fs = new FileStream(strFile, FileMode.Open))
            {
                using (BinaryReader br = new BinaryReader(fs))
                {
                    rgWeights = br.ReadBytes((int)fs.Length);
                }
            }

            using (StreamReader sr = new StreamReader(strModelFile))
            {
                strModelDesc = sr.ReadToEnd();
            }

            caffe.LoadToRun(strModelDesc, rgWeights, new BlobShape(1, 3, 300, 300), null, null, true);
            Net<T> net = caffe.GetInternalNet(Phase.RUN);
            DataTransformer<T> transformer = caffe.DataTransformer;
            // No cropping used.
            transformer.param.crop_size = 0;
            // Caffe weights use BGR color ordering (same as weight file)
            transformer.param.color_order = TransformationParameter.COLOR_ORDER.BGR;

            CancelEvent evtCancel = new CancelEvent();
            DeepDraw<T> deepDraw = new DeepDraw<T>(evtCancel, net, transformer, "data");

            // these octaves determine gradient ascent steps
            deepDraw.Add("loss3/classifier", 190, 2.5, 0.78, 11.0, 11.0, false);
            deepDraw.Add("loss3/classifier", 150, 0.78, 0.78, 6.0, 6.0, false);
            deepDraw.Add("loss2/classifier", 150, 0.78, 0.44, 6.0, 3.0, true);
            deepDraw.Add("loss1/classifier", 10, 0.44, 0.304, 3.0, 3.0, true);

            // Set the target output directory.
#warning TODO: Change image name to a file on your computer.
            string strVisualizeDir = createTargetDir("c:\\temp\\deepdraw");

            // the background color of the original image.
            Color clrBackground = Color.FromArgb(250, 250, 250);

            // generate the initial random image.
            Bitmap inputImg = deepDraw.CreateRandomImage(clrBackground);
            inputImg.Save(strVisualizeDir + "\\input.png");

            // which imagenet class to visualize
            List<int> rgImageNetClasses = new List<int>() { -1, 13, 20, 40, 90, 110, 140, 180, 200, 250, 300, 310, 370, 500, 600, 880 };

            for (int i = 0; i < rgImageNetClasses.Count; i++)
            {
                deepDraw.Render(inputImg, rgImageNetClasses[i], 0.25, strVisualizeDir);
            }

            deepDraw.Dispose();
        }

        /// <summary>
        /// The DeepDraw test is based on implementing the Deep Draw algorithms
        /// from https://github.com/kylemcdonald/deepdream/blob/master/dream.ipynb and
        /// from https://github.com/auduno/deepdraw/blob/master/deepdraw.ipynb and
        /// from http://www.kpkaiser.com/machine-learning/diving-deeper-into-deep-dreams/
        /// using MyCaffe.
        /// </summary>
        public void TestDeepDraw2()
        {
            SettingsCaffe settings = new SettingsCaffe();
            Log log = new Log("Deep Draw");
            CancelEvent m_evtCancel = new CancelEvent();
            MyCaffeControl<T> caffe = new MyCaffeControl<T>(settings, log, m_evtCancel);

            log.EnableTrace = true;

            string strModelFile = getTestPath("\\test_data\\models\\bvlc_googlenet\\train_val.prototxt");
            string strFile = getTestPath("\\test_data\\models\\bvlc_googlenet\\bvlc_googlenet.caffemodel");
            byte[] rgWeights = null;
            string strModelDesc = "";

            using (FileStream fs = new FileStream(strFile, FileMode.Open))
            {
                using (BinaryReader br = new BinaryReader(fs))
                {
                    rgWeights = br.ReadBytes((int)fs.Length);
                }
            }

            using (StreamReader sr = new StreamReader(strModelFile))
            {
                strModelDesc = sr.ReadToEnd();
            }


            // Set the target output directory.
#warning TODO: Change image name to a file on your computer.
            string strVisualizeDir = createTargetDir("c:\\temp\\deepdraw");

            // the background color of the original image.
            Color clrBackground = Color.FromArgb(250, 250, 250);

            // TODO: Change this file to an image of your own.
#warning TODO: Change image name to a file on your computer.
            Bitmap inputImg = new Bitmap("c:\\temp\\ocean.png");
            if (inputImg.Width > 600 || inputImg.Height > 600 || inputImg.Width != inputImg.Height)
                inputImg = ImageTools.ResizeImage(inputImg, 600, 600);
            inputImg.Save(strVisualizeDir + "\\input.png");

            caffe.LoadToRun(strModelDesc, rgWeights, new BlobShape(1, 3, inputImg.Height, inputImg.Width), null, null, true);
            Net<T> net = caffe.GetInternalNet(Phase.RUN);
            DataTransformer<T> transformer = caffe.DataTransformer;
            // No cropping used.
            transformer.param.crop_size = 0;
            // Caffe weights use BGR color ordering (same as weight file)
            transformer.param.color_order = TransformationParameter.COLOR_ORDER.BGR;

            CancelEvent evtCancel = new CancelEvent();
            DeepDraw<T> deepDraw = new DeepDraw<T>(evtCancel, net, transformer, "data");

            // these octaves determine gradient ascent steps
            deepDraw.Add("loss3/classifier", 50, 2.5, 0.78, 11.0, 11.0, false);
            deepDraw.Add("loss3/classifier", 30, 0.78, 0.78, 6.0, 6.0, false);
            deepDraw.Add("loss2/classifier", 20, 0.78, 0.44, 6.0, 3.0, true);
            deepDraw.Add("loss1/classifier", 10, 0.44, 0.304, 3.0, 1.5, true);

            // which imagenet class to visualize
            List<int> rgImageNetClasses = new List<int>() { -1 };

            for (int i = 0; i < rgImageNetClasses.Count; i++)
            {
                deepDraw.Render(inputImg, rgImageNetClasses[i], 0.5, strVisualizeDir);
            }

            deepDraw.Dispose();
        }


        private string createTargetDir(string strDir)
        {
            string strTargetDir = strDir;
            if (!Directory.Exists(strTargetDir))
                Directory.CreateDirectory(strTargetDir);

            string[] rgFiles = Directory.GetFiles(strTargetDir);
            foreach (string strFile in rgFiles)
            {
                File.Delete(strFile);
            }

            return strTargetDir;
        }
    }
}
