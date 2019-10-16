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

namespace MyCaffe.test
{
    [TestClass]
    public class TestImageTransforms
    {
        [TestMethod]
        public void TestUpdateBBoxByResizePolicy()
        {
            ImageTransformTest test = new ImageTransformTest();

            try
            {
                foreach (IImageTransformTest t in test.Tests)
                {
                    t.TestUpdateBBoxByResizePolicy();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IImageTransformTest : ITest
    {
        void TestUpdateBBoxByResizePolicy();
    }

    class ImageTransformTest : TestBase
    {
        public ImageTransformTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Image Transforms Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ImageTransformTest<double>(strName, nDeviceID, engine);
            else
                return new ImageTransformTest<float>(strName, nDeviceID, engine);
        }
    }

    class ImageTransformTest<T> : TestEx<T>, IImageTransformTest
    {
        float m_fEps = 1e-6f;
        ImageTransforms<T> m_imgTrans;

        public ImageTransformTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 10, 1, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_imgTrans = new ImageTransforms<T>(m_cuda, m_log, new CryptoRandom());
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestUpdateBBoxByResizePolicy()
        {
            NormalizedBBox bbox = new NormalizedBBox(0.1f, 0.3f, 0.3f, 0.6f, 0, false, 0, 0);
            int img_height = 600;
            int img_width = 1000;
            ResizeParameter resize_param = new ResizeParameter(true);
            resize_param.height = 300;
            resize_param.width = 300;
            NormalizedBBox out_bbox;

            // Test warp
            resize_param.resize_mode = ResizeParameter.ResizeMode.WARP;
            out_bbox = m_imgTrans.UpdateBBoxByResizePolicy(resize_param, img_width, img_height, bbox);
            m_log.EXPECT_NEAR(out_bbox.xmin, 0.1, m_fEps, "The xmin is not as expected");
            m_log.EXPECT_NEAR(out_bbox.ymin, 0.3, m_fEps, "The xmin is not as expected");
            m_log.EXPECT_NEAR(out_bbox.xmax, 0.3, m_fEps, "The xmin is not as expected");
            m_log.EXPECT_NEAR(out_bbox.ymax, 0.6, m_fEps, "The xmin is not as expected");

            // Test fit small size.
            resize_param.resize_mode = ResizeParameter.ResizeMode.FIT_SMALL_SIZE;
            out_bbox = m_imgTrans.UpdateBBoxByResizePolicy(resize_param, img_width, img_height, bbox);
            m_log.EXPECT_NEAR(out_bbox.xmin, 0.1, m_fEps, "The xmin is not as expected");
            m_log.EXPECT_NEAR(out_bbox.ymin, 0.3, m_fEps, "The xmin is not as expected");
            m_log.EXPECT_NEAR(out_bbox.xmax, 0.3, m_fEps, "The xmin is not as expected");
            m_log.EXPECT_NEAR(out_bbox.ymax, 0.6, m_fEps, "The xmin is not as expected");

            // Test fit large size.
            resize_param.resize_mode = ResizeParameter.ResizeMode.FIT_LARGE_SIZE_AND_PAD;
            out_bbox = m_imgTrans.UpdateBBoxByResizePolicy(resize_param, img_width, img_height, bbox);
            m_log.EXPECT_NEAR(out_bbox.xmin, 0.1, m_fEps, "The xmin is not as expected");
            m_log.EXPECT_NEAR(out_bbox.ymin, (180 * 0.3 + 60) / 300, m_fEps, "The xmin is not as expected");
            m_log.EXPECT_NEAR(out_bbox.xmax, 0.3, m_fEps, "The xmin is not as expected");
            m_log.EXPECT_NEAR(out_bbox.ymax, (180 * 0.6 + 60) / 300, m_fEps, "The xmin is not as expected");

            // Reverse the image size
            img_height = 1000;
            img_width = 600;

            // Test fit large size.
            resize_param.resize_mode = ResizeParameter.ResizeMode.FIT_LARGE_SIZE_AND_PAD;
            out_bbox = m_imgTrans.UpdateBBoxByResizePolicy(resize_param, img_width, img_height, bbox);
            m_log.EXPECT_NEAR(out_bbox.xmin, (180 * 0.1 + 60) / 300, m_fEps, "The xmin is not as expected");
            m_log.EXPECT_NEAR(out_bbox.ymin, 0.3, m_fEps, "The xmin is not as expected");
            m_log.EXPECT_NEAR(out_bbox.xmax, (180 * 0.3 + 60) / 300, m_fEps, "The xmin is not as expected");
            m_log.EXPECT_NEAR(out_bbox.ymax, 0.6, m_fEps, "The xmin is not as expected");
        }
    }
}
