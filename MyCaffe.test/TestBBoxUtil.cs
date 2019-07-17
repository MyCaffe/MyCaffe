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
using System.Drawing;

namespace MyCaffe.test
{
    [TestClass]
    public class TestBBoxUtil
    {
        [TestMethod]
        public void TestIntersectBBox()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestIntersectBBox();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBBoxSize()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestBBoxSize();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestScaleBBox()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestScaleBBox();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestClipBBox()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestClipBBox();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestOutputBBox()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestOutputBBox();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestJaccardOverlap()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestJaccardOverlap();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestEncodeBBoxCorner()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestEncodeBBoxCorner();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestEncoderBBoxCenterSize()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestEncoderBBoxCenterSize();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDecodeBBoxCorner()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestDecodeBBoxCorner();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDecoderBBoxCenterSize()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestDecoderBBoxCenterSize();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDecodeBBoxesCorner()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestDecodeBBoxesCorner();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestDecodeBBoxesCenterSize()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestDecodeBBoxesCenterSize();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IBBoxUtilTest : ITest
    {
        void TestIntersectBBox();
        void TestBBoxSize();
        void TestScaleBBox();
        void TestClipBBox();
        void TestOutputBBox();
        void TestJaccardOverlap();
        void TestEncodeBBoxCorner();
        void TestEncoderBBoxCenterSize();
        void TestDecodeBBoxCorner();
        void TestDecoderBBoxCenterSize();
        void TestDecodeBBoxesCorner();
        void TestDecodeBBoxesCenterSize();
    }

    class BBoxUtilTest : TestBase
    {
        public BBoxUtilTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("BBox Util Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new BBoxUtilTest<double>(strName, nDeviceID, engine);
            else
                return new BBoxUtilTest<float>(strName, nDeviceID, engine);
        }
    }

    class BBoxUtilTest<T> : TestEx<T>, IBBoxUtilTest
    {
        float m_fEps = 1e-6f;
        BBoxUtility<T> m_util;

        public BBoxUtilTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 10, 1, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_util = new BBoxUtility<T>(m_cuda, m_log);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        private void fillBBoxes(List<NormalizedBBox> gt_bboxes, List<NormalizedBBox> pred_bboxes)
        {
            gt_bboxes.Clear();
            pred_bboxes.Clear();

            // Fill in ground truth bboxes.
            gt_bboxes.Add(new NormalizedBBox(0.1f, 0.1f, 0.3f, 0.1f, 1));
            gt_bboxes.Add(new NormalizedBBox(0.3f, 0.3f, 0.6f, 0.5f, 2));

            // Fill in prediction bboxes
            // 4/9 with label 1
            // 0 with label 2
            pred_bboxes.Add(new NormalizedBBox(0.1f, 0.0f, 0.4f, 0.3f));

            // 2/6 with label 1
            // 0 with label 2
            pred_bboxes.Add(new NormalizedBBox(0.0f, 0.1f, 0.2f, 0.3f));

            // 2/8 with label 1
            // 1/11 with label 2
            pred_bboxes.Add(new NormalizedBBox(0.2f, 0.1f, 0.4f, 0.4f));

            // 0 with label 1
            // 4/8 with label 2
            pred_bboxes.Add(new NormalizedBBox(0.4f, 0.3f, 0.7f, 0.5f));

            // 0 with label 1
            // 1/11 with label 2
            pred_bboxes.Add(new NormalizedBBox(0.5f, 0.4f, 0.7f, 0.7f));

            // 0 with label 1
            // 0 with label 2
            pred_bboxes.Add(new NormalizedBBox(0.7f, 0.7f, 0.8f, 0.8f));
        }

        private void checkBBox(NormalizedBBox bbox, float fxmin, float fymin, float fxmax, float fymax, float? fSize = null, float? fEps = null)
        {
            float fEps1 = fEps.GetValueOrDefault(m_fEps);

            m_log.EXPECT_NEAR(fxmin, bbox.xmin, fEps1, "xmin's do not match.");
            m_log.EXPECT_NEAR(fymin, bbox.ymin, fEps1, "ymin's do not match.");
            m_log.EXPECT_NEAR(fxmax, bbox.xmax, fEps1, "xmax's do not match.");
            m_log.EXPECT_NEAR(fymax, bbox.ymax, fEps1, "ymax's do not match.");

            if (fSize.HasValue)
                m_log.EXPECT_NEAR(bbox.size, fSize.Value, fEps1, "The sizes do not match.");
        }

        public void TestIntersectBBox()
        {
            NormalizedBBox bbox_ref = new NormalizedBBox(0.2f, 0.3f, 0.3f, 0.5f);
            NormalizedBBox bbox_test;
            NormalizedBBox bbox_intersect;

            // Partially overlapped
            bbox_test = new NormalizedBBox(0.1f, 0.1f, 0.3f, 0.4f);
            bbox_intersect = m_util.Intersect(bbox_ref, bbox_test);
            checkBBox(bbox_intersect, 0.2f, 0.3f, 0.3f, 0.4f);

            // Fully contain
            bbox_test = new NormalizedBBox(0.1f, 0.1f, 0.4f, 0.6f);
            bbox_intersect = m_util.Intersect(bbox_ref, bbox_test);
            checkBBox(bbox_intersect, 0.2f, 0.3f, 0.3f, 0.5f);

            // Outside
            bbox_test = new NormalizedBBox(0.0f, 0.0f, 0.1f, 0.1f);
            bbox_intersect = m_util.Intersect(bbox_ref, bbox_test);
            checkBBox(bbox_intersect, 0.0f, 0.0f, 0.0f, 0.0f);
        }

        public void TestBBoxSize()
        {
            NormalizedBBox bbox;
            float fSize;

            // Valid box.
            bbox = new NormalizedBBox(0.2f, 0.3f, 0.3f, 0.5f);
            fSize = m_util.Size(bbox);
            m_log.EXPECT_NEAR(fSize, 0.02, m_fEps, "The sizes do not match.");

            // A line.
            bbox = new NormalizedBBox(0.2f, 0.3f, 0.2f, 0.5f);
            fSize = m_util.Size(bbox);
            m_log.EXPECT_NEAR(fSize, 0.0, m_fEps, "The sizes do not match.");

            // Invalid box.
            bbox = new NormalizedBBox(0.2f, 0.3f, 0.1f, 0.5f);
            fSize = m_util.Size(bbox);
            m_log.EXPECT_NEAR(fSize, 0.0, m_fEps, "The sizes do not match.");
        }

        public void TestScaleBBox()
        {
            NormalizedBBox bbox = new NormalizedBBox(0.21f, 0.32f, 0.33f, 0.54f);
            NormalizedBBox scale_bbox;

            int height = 10;
            int width = 20;
            scale_bbox = m_util.Scale(bbox, height, width);
            checkBBox(scale_bbox, 4.2f, 3.2f, 6.6f, 5.4f, 10.88f, 1e-5f);

            height = 1;
            width = 1;
            scale_bbox = m_util.Scale(bbox, height, width);
            checkBBox(scale_bbox, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, 0.0264f, 1e-5f);
        }

        public void TestClipBBox()
        {
            NormalizedBBox bbox;
            NormalizedBBox clip_bbox;

            bbox = new NormalizedBBox(0.2f, 0.3f, 0.3f, 0.5f);
            clip_bbox = m_util.Clip(bbox);
            checkBBox(clip_bbox, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, 0.02f);

            bbox = new NormalizedBBox(-0.2f, -0.3f, 1.3f, 1.5f);
            clip_bbox = m_util.Clip(bbox);
            checkBBox(clip_bbox, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        }

        public void TestOutputBBox()
        {
            NormalizedBBox bbox = new NormalizedBBox(-0.1f, 0.3f, 0.3f, 0.5f);
            SizeF szImg = new SizeF(500, 300);
            ResizeParameter p = new ResizeParameter();
            p.height = 300;
            p.width = 300;
            NormalizedBBox out_bbox;

            out_bbox = m_util.Output(bbox, szImg, p);
            checkBBox(out_bbox, 0.0f, 90.0f, 150.0f, 150.0f);

            p.resize_mode = ResizeParameter.ResizeMode.WARP;
            out_bbox = m_util.Output(bbox, szImg, p);
            checkBBox(out_bbox, 0.0f, 90.0f, 150.0f, 150.0f);

            p.resize_mode = ResizeParameter.ResizeMode.FIT_SMALL_SIZE;
            out_bbox = m_util.Output(bbox, szImg, p);
            checkBBox(out_bbox, 0.0f, 90.0f, 150.0f, 150.0f);

            p.resize_mode = ResizeParameter.ResizeMode.FIT_SMALL_SIZE;
            p.height_scale = 300;
            p.width_scale = 300;
            out_bbox = m_util.Output(bbox, szImg, p);
            checkBBox(out_bbox, 0.0f, 90.0f, 90.0f, 150.0f);

            p.resize_mode = ResizeParameter.ResizeMode.FIT_LARGE_SIZE_AND_PAD;
            out_bbox = m_util.Output(bbox, szImg, p);
            checkBBox(out_bbox, 0.0f, 50.0f, 150.0f, 150.0f);

            szImg.Height = 500;
            szImg.Width = 300;
            out_bbox = m_util.Output(bbox, szImg, p);
            checkBBox(out_bbox, 0.0f, 150.0f, 50.0f, 250.0f);
        }

        public void TestJaccardOverlap()
        {
            NormalizedBBox bbox1 = new NormalizedBBox(0.2f, 0.3f, 0.3f, 0.5f);
            NormalizedBBox bbox2;
            float fOverlap;

            // Partially overlapped.
            bbox2 = new NormalizedBBox(0.1f, 0.1f, 0.3f, 0.4f);
            fOverlap = m_util.JaccardOverlap(bbox1, bbox2);
            m_log.EXPECT_NEAR(fOverlap, 1.0f / 7, m_fEps, "The overlap does not match.");

            // Fully contain.
            bbox2 = new NormalizedBBox(0.1f, 0.1f, 0.4f, 0.6f);
            fOverlap = m_util.JaccardOverlap(bbox1, bbox2);
            m_log.EXPECT_NEAR(fOverlap, 2.0f / 15, m_fEps, "The overlap does not match.");

            // Outside.
            bbox2 = new NormalizedBBox(0.0f, 0.0f, 0.1f, 0.1f);
            fOverlap = m_util.JaccardOverlap(bbox1, bbox2);
            m_log.EXPECT_NEAR(fOverlap, 0.0f, m_fEps, "The overlap does not match.");
        }

        public void TestEncodeBBoxCorner()
        {
            NormalizedBBox prior_bbox = new NormalizedBBox(0.1f, 0.1f, 0.3f, 0.3f);
            List<float> prior_variance = Utility.Create<float>(4, 0.1f);
            NormalizedBBox bbox = new NormalizedBBox(0.0f, 0.2f, 0.4f, 0.5f);
            PriorBoxParameter.CodeType code_type = PriorBoxParameter.CodeType.CORNER;
            bool bEncodeVarianceInTarget = true;
            NormalizedBBox encode_bbox;

            encode_bbox = m_util.Encode(prior_bbox, prior_variance, code_type, bEncodeVarianceInTarget, bbox);
            checkBBox(encode_bbox, -0.1f, 0.1f, 0.1f, 0.2f);

            bEncodeVarianceInTarget = false;
            encode_bbox = m_util.Encode(prior_bbox, prior_variance, code_type, bEncodeVarianceInTarget, bbox);
            checkBBox(encode_bbox, -1.0f, 1.0f, 1.0f, 2.0f);
        }

        public void TestEncoderBBoxCenterSize()
        {
            NormalizedBBox prior_bbox = new NormalizedBBox(0.1f, 0.1f, 0.3f, 0.3f);
            List<float> prior_variance = Utility.Create<float>(4, 0.1f);
            prior_variance[2] = 0.2f;
            prior_variance[3] = 0.2f;
            NormalizedBBox bbox = new NormalizedBBox(0.0f, 0.2f, 0.4f, 0.5f);
            PriorBoxParameter.CodeType code_type = PriorBoxParameter.CodeType.CENTER_SIZE;
            bool bEncodeVarianceInTarget = true;
            NormalizedBBox encode_bbox;

            encode_bbox = m_util.Encode(prior_bbox, prior_variance, code_type, bEncodeVarianceInTarget, bbox);
            checkBBox(encode_bbox, 0.0f, 0.75f, (float)Math.Log(2.0f), (float)Math.Log(3.0f / 2));

            bEncodeVarianceInTarget = false;
            encode_bbox = m_util.Encode(prior_bbox, prior_variance, code_type, bEncodeVarianceInTarget, bbox);
            checkBBox(encode_bbox, 0.0f / 0.1f, 0.75f / 0.1f, (float)Math.Log(2.0f) / 0.2f, (float)Math.Log(3.0f / 2) / 0.2f, null, 1e-5f);
        }

        public void TestDecodeBBoxCorner()
        {
            NormalizedBBox prior_bbox = new NormalizedBBox(0.1f, 0.1f, 0.3f, 0.3f);
            List<float> prior_variance = Utility.Create<float>(4, 0.1f);
            NormalizedBBox bbox = new NormalizedBBox(-1.0f, 1.0f, 1.0f, 2.0f);
            PriorBoxParameter.CodeType code_type = PriorBoxParameter.CodeType.CORNER;
            bool bEncodeVarianceInTarget = false;
            NormalizedBBox decode_bbox;

            decode_bbox = m_util.Decode(prior_bbox, prior_variance, code_type, bEncodeVarianceInTarget, false, bbox);
            checkBBox(decode_bbox, 0.0f, 0.2f, 0.4f, 0.5f);

            bEncodeVarianceInTarget = true;
            decode_bbox = m_util.Decode(prior_bbox, prior_variance, code_type, bEncodeVarianceInTarget, false, bbox);
            checkBBox(decode_bbox, -0.9f, 1.1f, 1.3f, 2.3f);
        }

        public void TestDecoderBBoxCenterSize()
        {
            NormalizedBBox prior_bbox = new NormalizedBBox(0.1f, 0.1f, 0.3f, 0.3f);
            List<float> prior_variance = Utility.Create<float>(4, 0.1f);
            prior_variance[2] = 0.2f;
            prior_variance[3] = 0.2f;
            NormalizedBBox bbox = new NormalizedBBox(0.0f, 0.75f, (float)Math.Log(2.0), (float)Math.Log(3.0/2));
            PriorBoxParameter.CodeType code_type = PriorBoxParameter.CodeType.CENTER_SIZE;
            bool bEncodeVarianceInTarget = true;
            NormalizedBBox decode_bbox;

            decode_bbox = m_util.Decode(prior_bbox, prior_variance, code_type, bEncodeVarianceInTarget, false, bbox);
            checkBBox(decode_bbox, 0.0f, 0.2f, 0.4f, 0.5f);

            bbox = new NormalizedBBox(0.0f, 7.5f, (float)Math.Log(2.0) * 5, (float)Math.Log(3.0 / 2) * 5);
            bEncodeVarianceInTarget = false;
            decode_bbox = m_util.Decode(prior_bbox, prior_variance, code_type, bEncodeVarianceInTarget, false, bbox);
            checkBBox(decode_bbox, 0.0f, 0.2f, 0.4f, 0.5f);
        }

        public void TestDecodeBBoxesCorner()
        {
            List<NormalizedBBox> rgPriorBoxes = new List<NormalizedBBox>();
            List<List<float>> rgrgPriorVariances = new List<List<float>>();
            List<NormalizedBBox> rgBboxes = new List<NormalizedBBox>();

            for (int i = 1; i < 5; i++)
            {
                rgPriorBoxes.Add(new NormalizedBBox(0.1f * i, 0.1f * i, 0.1f * i + 0.2f, 0.1f * i + 0.2f));
                rgrgPriorVariances.Add(Utility.Create<float>(4, 0.1f));
                rgBboxes.Add(new NormalizedBBox(-1 * (i % 2), (i + 1) % 2, (i + 1) % 2, i % 2));
            }

            PriorBoxParameter.CodeType code_type = PriorBoxParameter.CodeType.CORNER;
            List<NormalizedBBox> rgDecodeBboxes;
            bool bVarianceEncodeInTarget = false;

            rgDecodeBboxes = m_util.Decode(rgPriorBoxes, rgrgPriorVariances, code_type, bVarianceEncodeInTarget, false, rgBboxes);
            m_log.CHECK_EQ(rgDecodeBboxes.Count, 4, "There should be four decoded boxes!");

            for (int i = 1; i < 5; i++)
            {
                checkBBox(rgDecodeBboxes[i - 1], 0.1f * i + i % 2 * -0.1f,
                                               0.1f * i + (i + 1) % 2 * 0.1f,
                                               0.1f * i + 0.2f + (i + 1) % 2 * 0.1f,
                                               0.1f * i + 0.2f + i % 2 * 0.1f);
            }

            bVarianceEncodeInTarget = true;
            rgDecodeBboxes = m_util.Decode(rgPriorBoxes, rgrgPriorVariances, code_type, bVarianceEncodeInTarget, false, rgBboxes);
            m_log.CHECK_EQ(rgDecodeBboxes.Count, 4, "There should be four decoded boxes!");

            for (int i = 1; i < 5; i++)
            {
                checkBBox(rgDecodeBboxes[i - 1], 0.1f * i + i % 2 * -1.0f,
                                               0.1f * i + (i + 1) % 2,
                                               0.1f * i + 0.2f + (i + 1) % 2,
                                               0.1f * i + 0.2f + i % 2);
            }
        }

        public void TestDecodeBBoxesCenterSize()
        {
            List<NormalizedBBox> rgPriorBoxes = new List<NormalizedBBox>();
            List<List<float>> rgrgPriorVariances = new List<List<float>>();
            List<NormalizedBBox> rgBboxes = new List<NormalizedBBox>();

            for (int i = 1; i < 5; i++)
            {
                rgPriorBoxes.Add(new NormalizedBBox(0.1f * i, 0.1f * i, 0.1f * i + 0.2f, 0.1f * i + 0.2f));
                List<float> rgVariance = Utility.Create<float>(4, 0.1f);
                rgVariance[2] = 0.2f;
                rgVariance[3] = 0.2f;
                rgrgPriorVariances.Add(rgVariance);
                rgBboxes.Add(new NormalizedBBox(0.0f, 0.75f, (float)Math.Log(2.0), (float)Math.Log(3.0/2)));
            }

            PriorBoxParameter.CodeType code_type = PriorBoxParameter.CodeType.CENTER_SIZE;
            List<NormalizedBBox> rgDecodeBboxes;
            bool bVarianceEncodeInTarget = true;

            rgDecodeBboxes = m_util.Decode(rgPriorBoxes, rgrgPriorVariances, code_type, bVarianceEncodeInTarget, false, rgBboxes);
            m_log.CHECK_EQ(rgDecodeBboxes.Count, 4, "There should be four decoded boxes!");
            float fEps = 1e-5f;

            for (int i = 1; i < 5; i++)
            {
                checkBBox(rgDecodeBboxes[i - 1], 0 + (i - 1) * 0.1f,
                                                 0.2f + (i - 1) * 0.1f,
                                                 0.4f + (i - 1) * 0.1f,
                                                 0.5f + (i - 1) * 0.1f,
                                                 null, fEps);
            }

            bVarianceEncodeInTarget = false;
            for (int i = 0; i < 4; i++)
            {
                rgBboxes[i] = new NormalizedBBox(0.0f, 7.5f, (float)Math.Log(2.0) * 5, (float)Math.Log(3.0 / 2) * 5);
            }

            rgDecodeBboxes = m_util.Decode(rgPriorBoxes, rgrgPriorVariances, code_type, bVarianceEncodeInTarget, false, rgBboxes);
            m_log.CHECK_EQ(rgDecodeBboxes.Count, 4, "There should be four decoded boxes!");

            for (int i = 1; i < 5; i++)
            {
                checkBBox(rgDecodeBboxes[i-1], 0.0f + (i - 1) * 0.1f,
                                               0.2f + (i - 1) * 0.1f,
                                               0.4f + (i - 1) * 0.1f,
                                               0.5f + (i - 1) * 0.1f,
                                               null, fEps);
            }
        }

        public void TestMatchBBoxLabelOnBipartite()
        {
        }
    }
}
