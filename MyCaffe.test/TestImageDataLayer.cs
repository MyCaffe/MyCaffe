using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.db.image;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using System.Threading;
using System.Diagnostics;
using MyCaffe.basecode.descriptors;
using MyCaffe.param.ssd;
using System.Drawing;
using System.IO;

namespace MyCaffe.test
{
    [TestClass]
    public class TestImageDataLayer
    {
        [TestMethod]
        public void TestRead()
        {
            ImageDataLayerTest test = new ImageDataLayerTest();

            try
            {
                foreach (IImageDataLayerTest t in test.Tests)
                {
                    t.TestRead();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestResize()
        {
            ImageDataLayerTest test = new ImageDataLayerTest();

            try
            {
                foreach (IImageDataLayerTest t in test.Tests)
                {
                    t.TestResize();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReshape()
        {
            ImageDataLayerTest test = new ImageDataLayerTest();

            try
            {
                foreach (IImageDataLayerTest t in test.Tests)
                {
                    t.TestReshape();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestShuffle()
        {
            ImageDataLayerTest test = new ImageDataLayerTest();

            try
            {
                foreach (IImageDataLayerTest t in test.Tests)
                {
                    t.TestShuffle();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSpace()
        {
            ImageDataLayerTest test = new ImageDataLayerTest();

            try
            {
                foreach (IImageDataLayerTest t in test.Tests)
                {
                    t.TestSpace();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    class ImageDataLayerTest : TestBase
    {
        CancelEvent m_evtCancel = new CancelEvent();

        public ImageDataLayerTest()
            : base("ImageData Layer Test")
        {
        }

        protected override ITest create(DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            string strPath = TestBase.CudaPath;

            if (dt == DataType.DOUBLE)
            {
                CudaDnn<double>.SetDefaultCudaPath(strPath);
                return new ImageDataLayerTest<double>(strName, nDeviceID, this);
            }
            else
            {
                CudaDnn<float>.SetDefaultCudaPath(strPath);
                return new ImageDataLayerTest<float>(strName, nDeviceID, this);
            }
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
        }
    }

    interface IImageDataLayerTest 
    {
        DataType Type { get; }
        void TestRead();
        void TestResize();
        void TestReshape();
        void TestShuffle();
        void TestSpace();
    }

    class ImageDataLayerTest<T> : TestEx<T>, IImageDataLayerTest
    {
        ImageDataLayerTest m_parent;
        string m_strFileName;
        string m_strFileNameReshape;
        string m_strFileNameSpace;
        Blob<T> m_blobTopLabel;

        public ImageDataLayerTest(string strName, int nDeviceID, ImageDataLayerTest parent, List<int> rgBottomShape = null)
            : base(strName, rgBottomShape, nDeviceID)
        {
            m_parent = parent;

            string strDataDir = getTestPath("\\MyCaffe\\test_data\\data\\images\\content\\", true);

            // Create test input file.
            m_strFileName = Path.GetTempFileName();
            m_log.WriteLine("Using temporary file '" + m_strFileName + "'");
            using (StreamWriter sw = new StreamWriter(m_strFileName))
            {
                for (int i = 0; i < 5; i++)
                {
                    string strFile = strDataDir + "cat.jpg";
                    sw.WriteLine(strFile + " " + i.ToString());
                }
            }

            // Create test input file for images of distinct sizes.
            m_strFileNameReshape = Path.GetTempFileName();
            m_log.WriteLine("Using temporary file '" + m_strFileNameReshape + "'");
            using (StreamWriter sw = new StreamWriter(m_strFileNameReshape))
            {
                sw.WriteLine(strDataDir + "cat.jpg 0");
                sw.WriteLine(strDataDir + "fish-bike.jpg  1");
            }

            // Create test input file for images with space in names.
            m_strFileNameSpace = Path.GetTempFileName();
            m_log.WriteLine("Using temporary file '" + m_strFileNameSpace + "'");
            using (StreamWriter sw = new StreamWriter(m_strFileNameSpace))
            {
                sw.WriteLine(strDataDir + "cat.jpg 0");
                sw.WriteLine(strDataDir + "cat gray.jpg 1");
            }

            m_blobTopLabel = new Blob<T>(m_cuda, m_log);
            TopVec.Add(m_blobTopLabel);
            BottomVec.Clear();
        }

        protected override void dispose()
        {
            if (File.Exists(m_strFileName))
                File.Delete(m_strFileName);

            if (File.Exists(m_strFileNameReshape))
                File.Delete(m_strFileNameReshape);

            if (File.Exists(m_strFileNameSpace))
                File.Delete(m_strFileNameSpace);

            if (m_blobTopLabel != null)
            {
                m_blobTopLabel.Dispose();
                m_blobTopLabel = null;
            }

            base.dispose();
        }

        public DataType Type
        {
            get { return m_dt; }
        }

        public Blob<T> TopLabel
        {
            get { return m_blobTopLabel; }
        }

        public void TestRead()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IMAGE_DATA);
            int nBatchSize = 5;
            p.image_data_param.batch_size = (uint)nBatchSize;
            p.image_data_param.source = m_strFileName;
            p.image_data_param.shuffle = false;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, nBatchSize, "The top num should = the batch size of 5.");
                m_log.CHECK_EQ(Top.channels, 3, "The top channels should = 3.");
                m_log.CHECK_EQ(Top.width, 480, "The top width should = 480");
                m_log.CHECK_EQ(Top.height, 360, "The top height should = 360");
                m_log.CHECK_EQ(TopLabel.num, nBatchSize, "The top num should = the batch size of 5.");
                m_log.CHECK_EQ(TopLabel.channels, 1, "The top channels should = 1.");
                m_log.CHECK_EQ(TopLabel.width, 1, "The top width should = 1");
                m_log.CHECK_EQ(TopLabel.height, 1, "The top height should = 1");

                // Go through the data twice.
                for (int iter = 0; iter < 2; iter++)
                {
                    layer.Forward(BottomVec, TopVec);
                    double[] rgLabel = convert(TopLabel.mutable_cpu_data);

                    for (int i = 0; i < 5; i++)
                    {
                        int nLabel = (int)rgLabel[i];
                        m_log.CHECK_EQ(nLabel, i, "The label is not as expected!");
                    }
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestResize()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IMAGE_DATA);
            int nBatchSize = 5;
            p.image_data_param.batch_size = (uint)nBatchSize;
            p.image_data_param.source = m_strFileName;
            p.image_data_param.shuffle = false;
            p.image_data_param.new_height = 256;
            p.image_data_param.new_width = 256;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, nBatchSize, "The top num should = the batch size of 5.");
                m_log.CHECK_EQ(Top.channels, 3, "The top channels should = 3.");
                m_log.CHECK_EQ(Top.width, 256, "The top width should = 256");
                m_log.CHECK_EQ(Top.height, 256, "The top height should = 256");
                m_log.CHECK_EQ(TopLabel.num, nBatchSize, "The top num should = the batch size of 5.");
                m_log.CHECK_EQ(TopLabel.channels, 1, "The top channels should = 1.");
                m_log.CHECK_EQ(TopLabel.width, 1, "The top width should = 1");
                m_log.CHECK_EQ(TopLabel.height, 1, "The top height should = 1");

                // Go through the data twice.
                for (int iter = 0; iter < 2; iter++)
                {
                    layer.Forward(BottomVec, TopVec);
                    double[] rgLabel = convert(TopLabel.mutable_cpu_data);

                    for (int i = 0; i < 5; i++)
                    {
                        int nLabel = (int)rgLabel[i];
                        m_log.CHECK_EQ(nLabel, i, "The label is not as expected!");
                    }
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestReshape()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IMAGE_DATA);
            p.image_data_param.batch_size = 1;
            p.image_data_param.source = m_strFileNameReshape;
            p.image_data_param.shuffle = false;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent);

            try
            {
                layer.Setup(BottomVec, TopVec);
                m_log.CHECK_EQ(TopLabel.num, 1, "The top num should = the batch size of 1.");
                m_log.CHECK_EQ(TopLabel.channels, 1, "The top channels should = 1.");
                m_log.CHECK_EQ(TopLabel.width, 1, "The top width should = 1");
                m_log.CHECK_EQ(TopLabel.height, 1, "The top height should = 1");

                // cat.jpg
                layer.Forward(BottomVec, TopVec);
                m_log.CHECK_EQ(Top.num, 1, "The top num should = the batch size of 1.");
                m_log.CHECK_EQ(Top.channels, 3, "The top channels should = 3.");
                m_log.CHECK_EQ(Top.width, 480, "The top width should = 480");
                m_log.CHECK_EQ(Top.height, 360, "The top height should = 360");

                // fish-bike.jpg
                layer.Forward(BottomVec, TopVec);
                m_log.CHECK_EQ(Top.num, 1, "The top num should = the batch size of 1.");
                m_log.CHECK_EQ(Top.channels, 3, "The top channels should = 3.");
                m_log.CHECK_EQ(Top.width, 481, "The top width should = 481");
                m_log.CHECK_EQ(Top.height, 323, "The top height should = 323");
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestShuffle()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IMAGE_DATA);
            int nBatchSize = 5;
            p.image_data_param.batch_size = (uint)nBatchSize;
            p.image_data_param.source = m_strFileName;
            p.image_data_param.shuffle = true;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, nBatchSize, "The top num should = the batch size of 5.");
                m_log.CHECK_EQ(Top.channels, 3, "The top channels should = 3.");
                m_log.CHECK_EQ(Top.width, 480, "The top width should = 480");
                m_log.CHECK_EQ(Top.height, 360, "The top height should = 360");
                m_log.CHECK_EQ(TopLabel.num, nBatchSize, "The top num should = the batch size of 5.");
                m_log.CHECK_EQ(TopLabel.channels, 1, "The top channels should = 1.");
                m_log.CHECK_EQ(TopLabel.width, 1, "The top width should = 1");
                m_log.CHECK_EQ(TopLabel.height, 1, "The top height should = 1");

                // Go through the data twice.
                for (int iter = 0; iter < 2; iter++)
                {
                    layer.Forward(BottomVec, TopVec);
                    Dictionary<double, int> rgValueToIndices = new Dictionary<double, int>();
                    int nNumInOrder = 0;
                    double[] rgLabel = convert(TopLabel.mutable_cpu_data);

                    for (int i = 0; i < 5; i++)
                    {
                        double dfLabel = rgLabel[i];
                        //Check that the value has not been seen already (no duplicates);
                        m_log.CHECK(!rgValueToIndices.ContainsKey(dfLabel), "Duplicate found!");
                        rgValueToIndices.Add(dfLabel, i);
                        nNumInOrder += (dfLabel == (double)i) ? 1 : 0;
                    }

                    m_log.CHECK_EQ(5, rgValueToIndices.Count, "The value to indices count is incorrect!");
                    m_log.CHECK_GT(5, nNumInOrder, "The number in order is incorrect.");
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestSpace()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.IMAGE_DATA);
            p.image_data_param.batch_size = 1;
            p.image_data_param.source = m_strFileNameSpace;
            p.image_data_param.shuffle = false;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent);

            try
            {
                layer.Setup(BottomVec, TopVec);
                m_log.CHECK_EQ(TopLabel.num, 1, "The top num should = the batch size of 1.");
                m_log.CHECK_EQ(TopLabel.channels, 1, "The top channels should = 1.");
                m_log.CHECK_EQ(TopLabel.width, 1, "The top width should = 1");
                m_log.CHECK_EQ(TopLabel.height, 1, "The top height should = 1");

                // cat.jpg
                layer.Forward(BottomVec, TopVec);
                m_log.CHECK_EQ(Top.num, 1, "The top num should = the batch size of 1.");
                m_log.CHECK_EQ(Top.channels, 3, "The top channels should = 3.");
                m_log.CHECK_EQ(Top.width, 480, "The top width should = 480");
                m_log.CHECK_EQ(Top.height, 360, "The top height should = 360");

                double[] rg = convert(TopLabel.mutable_cpu_data);
                int nLabel = (int)rg[0];
                m_log.CHECK_EQ(nLabel, 0, "The label is incorrect!");

                // fish-bike.jpg
                layer.Forward(BottomVec, TopVec);
                m_log.CHECK_EQ(Top.num, 1, "The top num should = the batch size of 1.");
                m_log.CHECK_EQ(Top.channels, 3, "The top channels should = 3.");
                m_log.CHECK_EQ(Top.width, 480, "The top width should = 480");
                m_log.CHECK_EQ(Top.height, 360, "The top height should = 360");

                rg = convert(TopLabel.mutable_cpu_data);
                nLabel = (int)rg[0];
                m_log.CHECK_EQ(nLabel, 1, "The label is incorrect!");
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
