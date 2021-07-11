using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.data;
using MyCaffe.param.ssd;
using System.Net;
using System.IO;
using System.Drawing;
using System.Threading;

namespace MyCaffe.test
{
    [TestClass]
    public class TestImageTools
    {
        [TestMethod]
        public void TestAdjustContrast()
        {
            ImageToolsTest test = new ImageToolsTest();

            try
            {
                foreach (IImageToolsTest t in test.Tests)
                {
                    t.TestAdjustContrast();
                }
            }
            finally
            {
                test.Dispose();
            }
        }


        [TestMethod]
        public void TestAdjustContrastGpuNone()
        {
            ImageToolsTest test = new ImageToolsTest();

            try
            {
                foreach (IImageToolsTest t in test.Tests)
                {
                    t.TestAdjustContrastGpu(true, false, false, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAdjustContrastGpuBrightnessOnly()
        {
            ImageToolsTest test = new ImageToolsTest();

            try
            {
                foreach (IImageToolsTest t in test.Tests)
                {
                    t.TestAdjustContrastGpu(true, true, false, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAdjustContrastGpuContrastOnly()
        {
            ImageToolsTest test = new ImageToolsTest();

            try
            {
                foreach (IImageToolsTest t in test.Tests)
                {
                    t.TestAdjustContrastGpu(true, false, true, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAdjustContrastGpuSaturationOnly()
        {
            ImageToolsTest test = new ImageToolsTest();

            try
            {
                foreach (IImageToolsTest t in test.Tests)
                {
                    t.TestAdjustContrastGpu(true, false, false, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAdjustContrastGpuAll1()
        {
            ImageToolsTest test = new ImageToolsTest();

            try
            {
                foreach (IImageToolsTest t in test.Tests)
                {
                    t.TestAdjustContrastGpu(true, true, true, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestAdjustContrastGpuAll2()
        {
            ImageToolsTest test = new ImageToolsTest();

            try
            {
                foreach (IImageToolsTest t in test.Tests)
                {
                    t.TestAdjustContrastGpu(false, true, true, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IImageToolsTest : ITest
    {
        void TestAdjustContrast();
        void TestAdjustContrastGpu(bool bOrder, bool bBrightness, bool bContrast, bool bSaturation);
    }

    class ImageToolsTest : TestBase
    {
        public ImageToolsTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Image Tools Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ImageToolsTest<double>(strName, nDeviceID, engine);
            else
                return new ImageToolsTest<float>(strName, nDeviceID, engine);
        }
    }

    class ImageToolsTest<T> : TestEx<T>, IImageToolsTest
    {
        public ImageToolsTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 10, 1, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        private Bitmap downloadLenna(out string strFile)
        {
            string strDir = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\downloads\\";
            if (!Directory.Exists(strDir))
                Directory.CreateDirectory(strDir);

            strFile = strDir + "lenna.png";
            Bitmap bmp = null;

            if (File.Exists(strFile))
            {
                bmp = getBitmap(strFile);
                if (bmp != null)
                    return bmp;

                File.Delete(strFile);
            }

            string strUrl = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png";
            WebClient webClient = new WebClient();
            Thread.Sleep(10000);
            webClient.DownloadFile(strUrl, strFile);

            return getBitmap(strFile);
        }

        private Bitmap getBitmap(string strFile)
        {
            try
            {
                return new Bitmap(strFile);
            }
            catch (Exception excpt)
            {
                return null;
            }
        }

        private Bitmap downloadMandrill(out string strFile)
        {
            string strDir = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\downloads\\";
            if (!Directory.Exists(strDir))
                Directory.CreateDirectory(strDir);

            strFile = strDir + "mandrill.png";
            Bitmap bmp = null;

            if (File.Exists(strFile))
            {
                bmp = getBitmap(strFile);
                if (bmp != null)
                    return bmp;

                File.Delete(strFile);
            }

            string strUrl = "https://upload.wikimedia.org/wikipedia/commons/c/c1/Wikipedia-sipi-image-db-mandrill-4.2.03.png";
            WebClient webClient = new WebClient();
            Thread.Sleep(10000);
            webClient.DownloadFile(strUrl, strFile);

            return getBitmap(strFile);
        }

        public void TestAdjustContrast()
        {
            string strFile;
            string strLennaFile;
            Bitmap bmpMandrill = downloadMandrill(out strFile);
            Bitmap bmpLenna = downloadLenna(out strLennaFile);
            string strDir = Path.GetDirectoryName(strLennaFile) + "\\tests\\";

            if (!Directory.Exists(strDir))
                Directory.CreateDirectory(strDir);

            SimpleDatum sdLenna = ImageData.GetImageData(bmpLenna, new SimpleDatum(3, bmpLenna.Width, bmpLenna.Height));
            SimpleDatum sdMandrill = ImageData.GetImageData(bmpMandrill, new SimpleDatum(3, bmpMandrill.Width, bmpMandrill.Height));

            testBrightness("lenna", sdLenna, 32, strDir);
            testContrast("lenna", sdLenna, 1.5f, strDir);
            testGamma("lenna", sdLenna, 0.25f, strDir);
            testGamma("lenna", sdLenna, 2.0f, strDir);
            testGamma("mandrill", sdMandrill, 0.25f, strDir);
            testGamma("mandrill", sdMandrill, 2.0f, strDir);

            bmpLenna.Dispose();
            bmpMandrill.Dispose();
        }

        private void testBrightness(string strName, SimpleDatum sd, float fBrightness, string strDir)
        {
            SimpleDatum sd1 = new SimpleDatum(sd, true);

            ImageTools.AdjustContrast(sd1, fBrightness);
            Bitmap bmp = ImageData.GetImage(sd1);

            bmp.Save(strDir + strName + ".brightness_" + fBrightness.ToString() + ".png");
            bmp.Dispose();
        }

        private void testContrast(string strName, SimpleDatum sd, float fContrast, string strDir)
        {
            SimpleDatum sd1 = new SimpleDatum(sd, true);

            ImageTools.AdjustContrast(sd1, 0, fContrast);
            Bitmap bmp = ImageData.GetImage(sd1);

            bmp.Save(strDir + strName + ".contrast_" + fContrast.ToString() + ".png");
            bmp.Dispose();
        }

        private void testGamma(string strName, SimpleDatum sd, float fGamma, string strDir)
        {
            SimpleDatum sd1 = new SimpleDatum(sd, true);

            ImageTools.AdjustContrast(sd1, 0, 1, fGamma);
            Bitmap bmp = ImageData.GetImage(sd1);

            bmp.Save(strDir + strName + ".gamma_" + fGamma.ToString() + ".png");
            bmp.Dispose();
        }

        public void TestAdjustContrastGpu(bool bOrder, bool bBrightness, bool bContrast, bool bSaturation)
        {
            string strFile;
            string strLennaFile;
            Bitmap bmpMandrill = downloadMandrill(out strFile);
            Bitmap bmpLenna = downloadLenna(out strLennaFile);
            string strDir = Path.GetDirectoryName(strLennaFile) + "\\tests\\";

            if (!Directory.Exists(strDir))
                Directory.CreateDirectory(strDir);

            SimpleDatum sdLenna = ImageData.GetImageData(bmpLenna, new SimpleDatum(3, bmpLenna.Width, bmpLenna.Height));
            SimpleDatum sdMandrill = ImageData.GetImageData(bmpMandrill, new SimpleDatum(3, bmpMandrill.Width, bmpMandrill.Height));

            T[] rgData = new T[sdLenna.ItemCount + sdMandrill.ItemCount];
            Array.Copy(sdLenna.ByteData, rgData, sdLenna.ItemCount);
            Array.Copy(sdMandrill.ByteData, 0, rgData, sdLenna.ItemCount, sdMandrill.ItemCount);
            Blob<T> blob = new Blob<T>(m_cuda, m_log, 2, 3, bmpLenna.Height, bmpLenna.Width, false, false);

            blob.mutable_cpu_data = rgData;

            TransformationParameter tp = new TransformationParameter();
            tp.distortion_param = new DistortionParameter(true);
            tp.distortion_param.brightness_prob = (bBrightness) ? 1.0f : 0.0f;
            tp.distortion_param.brightness_delta = 32;
            tp.distortion_param.contrast_prob = (bContrast) ? 1.0f : 0.0f;
            tp.distortion_param.contrast_lower = 0.0f;
            tp.distortion_param.contrast_upper = 2.0f;
            tp.distortion_param.saturation_prob = (bSaturation) ? 1.0f : 0.0f;
            tp.distortion_param.saturation_lower = 0.0f;
            tp.distortion_param.saturation_upper = 2.0f;
            tp.distortion_param.random_order_prob = (bOrder) ? 1.0f : 0.0f;
            tp.distortion_param.random_seed = 1709;
            tp.distortion_param.use_gpu = true;

            DataTransformer<T> transform = new DataTransformer<T>(m_cuda, m_log, tp, Phase.TRAIN, 3, bmpLenna.Height, bmpLenna.Width);
            transform.DistortImage(blob);

            rgData = blob.update_cpu_data();
            float[] rgfData = Utility.ConvertVecF<T>(rgData);

            for (int i = 0; i < sdLenna.ItemCount; i++)
            {
                sdLenna.ByteData[i] = (byte)rgfData[i];
            }

            for (int i = 0; i < sdMandrill.ItemCount; i++)
            {
                sdMandrill.ByteData[i] = (byte)rgfData[i + sdLenna.ItemCount];
            }

            bmpLenna.Dispose();
            bmpMandrill.Dispose();

            bmpLenna = ImageData.GetImage(sdLenna);
            bmpMandrill = ImageData.GetImage(sdMandrill);

            string strType = ((int)tp.distortion_param.random_order_prob).ToString() + "_" + ((int)tp.distortion_param.brightness_prob).ToString() + "_" + ((int)tp.distortion_param.contrast_prob).ToString() + "_" + ((int)tp.distortion_param.saturation_prob).ToString();

            bmpLenna.Save(strDir + "lenna.gpu_distort." + strType + ".png");
            bmpMandrill.Save(strDir + "mandrill.gpu_distort." + strType + ".png");

            transform.Dispose();
            blob.Dispose();
        }
    }
}
