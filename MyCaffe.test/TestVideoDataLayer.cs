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
    public class TestVideoDataLayer
    {
        [TestMethod]
        public void TestSetupWebCam()
        {
            VideoDataLayerTest test = new VideoDataLayerTest(VideoDataParameter.VideoType.WEBCAM);

            try
            {
                foreach (IVideoDataLayerTest t in test.Tests)
                {
                    t.TestSetup();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardWebCam()
        {
            VideoDataLayerTest test = new VideoDataLayerTest(VideoDataParameter.VideoType.WEBCAM);

            try
            {
                foreach (IVideoDataLayerTest t in test.Tests)
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
        public void TestSetupVideoFile()
        {
            VideoDataLayerTest test = new VideoDataLayerTest(VideoDataParameter.VideoType.VIDEO);

            try
            {
                foreach (IVideoDataLayerTest t in test.Tests)
                {
                    t.TestSetup();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardVideoFile()
        {
            VideoDataLayerTest test = new VideoDataLayerTest(VideoDataParameter.VideoType.VIDEO);

            try
            {
                foreach (IVideoDataLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    class VideoDataLayerTest : TestBase
    {
        CancelEvent m_evtCancel = new CancelEvent();
        VideoDataParameter.VideoType m_videoType = VideoDataParameter.VideoType.WEBCAM;

        public VideoDataLayerTest(VideoDataParameter.VideoType videoType)
            : base("VideoData Layer Test")
        {
            m_videoType = videoType;
        }

        protected override ITest create(DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            string strPath = TestBase.CudaPath;

            if (dt == DataType.DOUBLE)
            {
                CudaDnn<double>.SetDefaultCudaPath(strPath);
                return new VideoDataLayerTest<double>(strName, nDeviceID, this);
            }
            else
            {
                CudaDnn<float>.SetDefaultCudaPath(strPath);
                return new VideoDataLayerTest<float>(strName, nDeviceID, this);
            }
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public VideoDataParameter.VideoType VideoType
        {
            get { return m_videoType; }
        }

        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
        }
    }

    interface IVideoDataLayerTest 
    {
        DataType Type { get; }
        void TestSetup();
        void TestForward();
    }

    class VideoDataLayerTest<T> : TestEx<T>, IVideoDataLayerTest
    {
        VideoDataLayerTest m_parent;

        public VideoDataLayerTest(string strName, int nDeviceID, VideoDataLayerTest parent, List<int> rgBottomShape = null)
            : base(strName, rgBottomShape, nDeviceID)
        {
            m_parent = parent;
            BottomVec.Clear();
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public DataType Type
        {
            get { return m_dt; }
        }

        public string VideoFile
        {
            get
            {
                string strDataDir = getTestPath("\\MyCaffe\\test_data\\data\\video\\", true);
                string strFile = strDataDir + "PET717_Var3.300x225.wmv";
                return strFile;
            }
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VIDEO_DATA);
            int nBatchSize = 8;
            p.data_param.batch_size = (uint)nBatchSize;
            p.video_data_param.video_type = m_parent.VideoType;
            p.video_data_param.video_file = VideoFile;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, nBatchSize, "The top num should = the batch size of 8.");
                m_log.CHECK_EQ(Top.channels, 3, "The top channels should = 3.");
                m_log.CHECK_EQ(Top.width, p.video_data_param.video_width, "The top width should = the video width of " + p.video_data_param.video_width.ToString());
                m_log.CHECK_EQ(Top.height, p.video_data_param.video_height, "The top height should = the video height of " + p.video_data_param.video_height.ToString());
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

        public void TestForward()
        {
            string strDataDir = getTestPath("\\MyCaffe\\test_data\\data\\images\\", true);
            string strResultDir = strDataDir + "video_test_result\\";

            if (!Directory.Exists(strResultDir))
                Directory.CreateDirectory(strResultDir);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.VIDEO_DATA);

            int nBatchSize = 8;
            p.data_param.batch_size = (uint)nBatchSize;
            p.video_data_param.video_type = m_parent.VideoType;
            p.video_data_param.video_file = VideoFile;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent);

            try
            {
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, nBatchSize, "The top num should = the batch size of 8.");
                m_log.CHECK_EQ(Top.channels, 3, "The top channels should = 3.");
                m_log.CHECK_EQ(Top.width, p.video_data_param.video_width, "The top width should = the video width of " + p.video_data_param.video_width.ToString());
                m_log.CHECK_EQ(Top.height, p.video_data_param.video_height, "The top height should = the video height of " + p.video_data_param.video_height.ToString());

                int nDim = Top.count(1);
                T[] rgTopData1 = new T[nDim];
                T[] rgTopData = Top.mutable_cpu_data;

                // Check the data.
                for (int i = 0; i < Top.num; i++)
                {
                    Array.Copy(rgTopData, nDim * i, rgTopData1, 0, nDim);
                    float[] rgf = convertF(rgTopData1);

                    SimpleDatum sd = new SimpleDatum(Top.channels, Top.width, Top.height, rgf, 0, nDim);
                    Bitmap bmp = ImageData.GetImage(sd);

                    bmp.Save(strResultDir + "bmp_" + i.ToString() + ".png");
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
    }
}
