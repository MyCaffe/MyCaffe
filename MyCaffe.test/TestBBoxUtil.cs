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
using MyCaffe.param.ssd;

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

        [TestMethod]
        public void TestMatchBBoxLabelOneBipartite()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestMatchBBoxLabelOneBipartite();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatchBBoxLabelAllBipartite()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestMatchBBoxLabelAllBipartite();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatchBBoxLabelOnePerPrediction()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestMatchBBoxLabelOnePerPrediction();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatchBBoxLabelAllPerPrediction()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestMatchBBoxLabelAllPerPrediction();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestMatchBBoxLabelAllPerPredictionEx()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestMatchBBoxLabelAllPerPredictionEx();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetGroundTruth()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestGetGroundTruth();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetGroundTruthLabelBBox()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestGetGroundTruthLabelBBox();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetLocPredictionsShared()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestGetLocPredictionsShared();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGetLocPredictionsUnshared()
        {
            BBoxUtilTest test = new BBoxUtilTest();

            try
            {
                foreach (IBBoxUtilTest t in test.Tests)
                {
                    t.TestGetLocPredictionsUnshared();
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
        void TestMatchBBoxLabelOneBipartite();
        void TestMatchBBoxLabelAllBipartite();
        void TestMatchBBoxLabelOnePerPrediction();
        void TestMatchBBoxLabelAllPerPrediction();
        void TestMatchBBoxLabelAllPerPredictionEx();
        void TestGetGroundTruth();
        void TestGetGroundTruthLabelBBox();
        void TestGetLocPredictionsShared();
        void TestGetLocPredictionsUnshared();
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
            gt_bboxes.Add(new NormalizedBBox(0.1f, 0.1f, 0.3f, 0.3f, 1));
            gt_bboxes.Add(new NormalizedBBox(0.3f, 0.3f, 0.6f, 0.5f, 2));

            // Fill in prediction bboxes
            // 4/9 with label 1
            // 0 with label 2
            pred_bboxes.Add(new NormalizedBBox(0.1f, 0.0f, 0.4f, 0.3f, 2));

            // 2/6 with label 1
            // 0 with label 2
            pred_bboxes.Add(new NormalizedBBox(0.0f, 0.1f, 0.2f, 0.3f, 2));

            // 2/8 with label 1
            // 1/11 with label 2
            pred_bboxes.Add(new NormalizedBBox(0.2f, 0.1f, 0.4f, 0.4f, 2));

            // 0 with label 1
            // 4/8 with label 2
            pred_bboxes.Add(new NormalizedBBox(0.4f, 0.3f, 0.7f, 0.5f, 2));

            // 0 with label 1
            // 1/11 with label 2
            pred_bboxes.Add(new NormalizedBBox(0.5f, 0.4f, 0.7f, 0.7f, 2));

            // 0 with label 1
            // 0 with label 2
            pred_bboxes.Add(new NormalizedBBox(0.7f, 0.7f, 0.8f, 0.8f, 2));
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

        public void TestMatchBBoxLabelOneBipartite()
        {
            List<NormalizedBBox> rgGtBboxes = new List<NormalizedBBox>();
            List<NormalizedBBox> rgPredBboxes = new List<NormalizedBBox>();

            fillBBoxes(rgGtBboxes, rgPredBboxes);

            int nLabel = 1;
            MultiBoxLossParameter.MatchType match_type = MultiBoxLossParameter.MatchType.BIPARTITE;
            float fOverlap = -1;

            List<int> rgMatchIndices;
            List<float> rgMatchOverlaps;
            m_util.Match(rgGtBboxes, rgPredBboxes, nLabel, match_type, fOverlap, true, out rgMatchIndices, out rgMatchOverlaps);

            m_log.CHECK_EQ(rgMatchIndices.Count, 6, "There should be 6 matches!");
            m_log.CHECK_EQ(rgMatchOverlaps.Count, 6, "There should be 6 matches!");

            m_log.CHECK_EQ(rgMatchIndices[0], 0, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[1], -1, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[2], -1, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[0], 4.0/9, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[1], 2.0/6, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[2], 2.0/8, m_fEps, "The index match is incorrect.");

            for (int i = 3; i < 6; i++)
            {
                m_log.CHECK_EQ(rgMatchIndices[i], -1, "The index match is incorrect.");
                m_log.EXPECT_NEAR(rgMatchOverlaps[i], 0, m_fEps, "The overlap is incorrect");
            }
        }

        public void TestMatchBBoxLabelAllBipartite()
        {
            List<NormalizedBBox> rgGtBboxes = new List<NormalizedBBox>();
            List<NormalizedBBox> rgPredBboxes = new List<NormalizedBBox>();

            fillBBoxes(rgGtBboxes, rgPredBboxes);

            int nLabel = -1;
            MultiBoxLossParameter.MatchType match_type = MultiBoxLossParameter.MatchType.BIPARTITE;
            float fOverlap = -1;

            List<int> rgMatchIndices;
            List<float> rgMatchOverlaps;
            m_util.Match(rgGtBboxes, rgPredBboxes, nLabel, match_type, fOverlap, true, out rgMatchIndices, out rgMatchOverlaps);

            m_log.CHECK_EQ(rgMatchIndices.Count, 6, "There should be 6 matches!");
            m_log.CHECK_EQ(rgMatchOverlaps.Count, 6, "There should be 6 matches!");

            m_log.CHECK_EQ(rgMatchIndices[0], 0, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[3], 1, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[0], 4.0 / 9, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[1], 2.0 / 6, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[2], 2.0 / 8, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[3], 4.0 / 8, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[4], 1.0 / 11, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[5], 0, m_fEps, "The index match is incorrect.");

            for (int i = 0; i < 6; i++)
            {
                if (i == 0 || i == 3)
                    continue;

                m_log.CHECK_EQ(rgMatchIndices[i], -1, "The index match is incorrect.");
            }
        }

        public void TestMatchBBoxLabelOnePerPrediction()
        {
            List<NormalizedBBox> rgGtBboxes = new List<NormalizedBBox>();
            List<NormalizedBBox> rgPredBboxes = new List<NormalizedBBox>();

            fillBBoxes(rgGtBboxes, rgPredBboxes);

            int nLabel = 1;
            MultiBoxLossParameter.MatchType match_type = MultiBoxLossParameter.MatchType.PER_PREDICTION;
            float fOverlap = 0.3f;

            List<int> rgMatchIndices;
            List<float> rgMatchOverlaps;
            m_util.Match(rgGtBboxes, rgPredBboxes, nLabel, match_type, fOverlap, true, out rgMatchIndices, out rgMatchOverlaps);

            m_log.CHECK_EQ(rgMatchIndices.Count, 6, "There should be 6 matches!");
            m_log.CHECK_EQ(rgMatchOverlaps.Count, 6, "There should be 6 matches!");

            m_log.CHECK_EQ(rgMatchIndices[0], 0, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[1], 0, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[2], -1, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[0], 4.0 / 9, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[1], 2.0 / 6, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[2], 2.0 / 8, m_fEps, "The index match is incorrect.");

            for (int i = 3; i < 6; i++)
            {
                m_log.CHECK_EQ(rgMatchIndices[i], -1, "The index match is incorrect.");
                m_log.EXPECT_NEAR(rgMatchOverlaps[i], 0, m_fEps, "The overlap is incorrect");
            }
        }

        public void TestMatchBBoxLabelAllPerPrediction()
        {
            List<NormalizedBBox> rgGtBboxes = new List<NormalizedBBox>();
            List<NormalizedBBox> rgPredBboxes = new List<NormalizedBBox>();

            fillBBoxes(rgGtBboxes, rgPredBboxes);

            int nLabel = -1;
            MultiBoxLossParameter.MatchType match_type = MultiBoxLossParameter.MatchType.PER_PREDICTION;
            float fOverlap = 0.3f;

            List<int> rgMatchIndices;
            List<float> rgMatchOverlaps;
            m_util.Match(rgGtBboxes, rgPredBboxes, nLabel, match_type, fOverlap, true, out rgMatchIndices, out rgMatchOverlaps);

            m_log.CHECK_EQ(rgMatchIndices.Count, 6, "There should be 6 matches!");
            m_log.CHECK_EQ(rgMatchOverlaps.Count, 6, "There should be 6 matches!");

            m_log.CHECK_EQ(rgMatchIndices[0], 0, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[1], 0, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[2], -1, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[3], 1, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[4], -1, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[5], -1, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[0], 4.0 / 9, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[1], 2.0 / 6, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[2], 2.0 / 8, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[3], 4.0 / 8, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[4], 1.0 / 11, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[5], 0, m_fEps, "The index match is incorrect.");
        }

        public void TestMatchBBoxLabelAllPerPredictionEx()
        {
            List<NormalizedBBox> rgGtBboxes = new List<NormalizedBBox>();
            List<NormalizedBBox> rgPredBboxes = new List<NormalizedBBox>();

            fillBBoxes(rgGtBboxes, rgPredBboxes);

            int nLabel = -1;
            MultiBoxLossParameter.MatchType match_type = MultiBoxLossParameter.MatchType.PER_PREDICTION;
            float fOverlap = 0.001f;

            List<int> rgMatchIndices;
            List<float> rgMatchOverlaps;
            m_util.Match(rgGtBboxes, rgPredBboxes, nLabel, match_type, fOverlap, true, out rgMatchIndices, out rgMatchOverlaps);

            m_log.CHECK_EQ(rgMatchIndices.Count, 6, "There should be 6 matches!");
            m_log.CHECK_EQ(rgMatchOverlaps.Count, 6, "There should be 6 matches!");

            m_log.CHECK_EQ(rgMatchIndices[0], 0, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[1], 0, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[2], 0, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[3], 1, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[4], 1, "The index match is incorrect.");
            m_log.CHECK_EQ(rgMatchIndices[5], -1, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[0], 4.0 / 9, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[1], 2.0 / 6, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[2], 2.0 / 8, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[3], 4.0 / 8, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[4], 1.0 / 11, m_fEps, "The index match is incorrect.");
            m_log.EXPECT_NEAR(rgMatchOverlaps[5], 0, m_fEps, "The index match is incorrect.");
        }

        public void TestGetGroundTruth()
        {
            int nNumGt = 4;
            Blob<T> blobGt = new Blob<T>(m_cuda, m_log, 1, 1, nNumGt, 8);
            double[] rgGt1 = convert(blobGt.mutable_cpu_data);
            float[] rgGt = rgGt1.Select(p => (float)p).ToArray();
            for (int i = 0; i < 4; i++)
            {
                int nImageId = (int)Math.Ceiling(i / 2.0);
                rgGt[i * 8 + 0] = nImageId;
                rgGt[i * 8 + 1] = i;
                rgGt[i * 8 + 2] = 0;
                rgGt[i * 8 + 3] = 0.1f;
                rgGt[i * 8 + 4] = 0.1f;
                rgGt[i * 8 + 5] = 0.3f;
                rgGt[i * 8 + 6] = 0.3f;
                rgGt[i * 8 + 7] = i % 2;
            }

            Dictionary<int, List<NormalizedBBox>> rgAllGtBboxes = m_util.GetGroundTruth(rgGt, nNumGt, -1, true);

            m_log.CHECK_EQ(rgAllGtBboxes.Count, 3, "There should be 3 ground truths.");

            List<KeyValuePair<int, List<NormalizedBBox>>> rgAllGtList = rgAllGtBboxes.ToList();
            m_log.CHECK_EQ(rgAllGtBboxes[0].Count, 1, "The ground truth at 0 should have 1 bbox.");
            m_log.CHECK_EQ(rgAllGtBboxes[0][0].label, 0, "The label should be 0.");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].xmin, 0.1, m_fEps, "The xmin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].ymin, 0.1, m_fEps, "The ymin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].xmax, 0.3, m_fEps, "The xmax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].ymax, 0.3, m_fEps, "The ymax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].size, 0.04, m_fEps, "The size should be 0.04");
            m_log.CHECK(rgAllGtBboxes[0][0].difficult == false, "The difficult should be false.");

            m_log.CHECK_EQ(rgAllGtBboxes[1].Count, 2, "The ground truth at 1 should have 2 bboxes.");
            for (int i = 1; i < 3; i++)
            {
                m_log.CHECK_EQ(rgAllGtBboxes[1][i-1].label, i, "The label should be 0.");
                m_log.EXPECT_NEAR(rgAllGtBboxes[1][i-1].xmin, 0.1, m_fEps, "The xmin should be 0.1");
                m_log.EXPECT_NEAR(rgAllGtBboxes[1][i-1].ymin, 0.1, m_fEps, "The ymin should be 0.1");
                m_log.EXPECT_NEAR(rgAllGtBboxes[1][i-1].xmax, 0.3, m_fEps, "The xmax should be 0.3");
                m_log.EXPECT_NEAR(rgAllGtBboxes[1][i-1].ymax, 0.3, m_fEps, "The ymax should be 0.3");
                m_log.EXPECT_NEAR(rgAllGtBboxes[1][i-1].size, 0.04, m_fEps, "The size should be 0.04");
                m_log.CHECK(rgAllGtBboxes[1][i - 1].difficult == (i % 2 == 1), "The difficult should be " + (i % 2 == 0).ToString() + ".");
            }

            m_log.CHECK_EQ(rgAllGtBboxes[2].Count, 1, "The ground truth at 0 should have 1 bbox.");
            m_log.CHECK_EQ(rgAllGtBboxes[2][0].label, 3, "The label should be 0.");
            m_log.EXPECT_NEAR(rgAllGtBboxes[2][0].xmin, 0.1, m_fEps, "The xmin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[2][0].ymin, 0.1, m_fEps, "The ymin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[2][0].xmax, 0.3, m_fEps, "The xmax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[2][0].ymax, 0.3, m_fEps, "The ymax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[2][0].size, 0.04, m_fEps, "The size should be 0.04");
            m_log.CHECK(rgAllGtBboxes[2][0].difficult == true, "The difficult should be true.");

            // Skip difficult ground truth.
            rgAllGtBboxes = m_util.GetGroundTruth(rgGt, nNumGt, -1, false);

            m_log.CHECK_EQ(rgAllGtBboxes.Count, 2, "There should be 3 ground truths.");

            rgAllGtList = rgAllGtBboxes.ToList();
            m_log.CHECK_EQ(rgAllGtBboxes[0].Count, 1, "The ground truth at 0 should have 1 bbox.");
            m_log.CHECK_EQ(rgAllGtBboxes[0][0].label, 0, "The label should be 0.");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].xmin, 0.1, m_fEps, "The xmin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].ymin, 0.1, m_fEps, "The ymin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].xmax, 0.3, m_fEps, "The xmax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].ymax, 0.3, m_fEps, "The ymax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].size, 0.04, m_fEps, "The size should be 0.04");
            m_log.CHECK(rgAllGtBboxes[0][0].difficult == false, "The difficult should be false.");

            m_log.CHECK_EQ(rgAllGtBboxes[1].Count, 1, "The ground truth at 0 should have 1 bbox.");
            m_log.CHECK_EQ(rgAllGtBboxes[1][0].label, 2, "The label should be 0.");
            m_log.EXPECT_NEAR(rgAllGtBboxes[1][0].xmin, 0.1, m_fEps, "The xmin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[1][0].ymin, 0.1, m_fEps, "The ymin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[1][0].xmax, 0.3, m_fEps, "The xmax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[1][0].ymax, 0.3, m_fEps, "The ymax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[1][0].size, 0.04, m_fEps, "The size should be 0.04");
            m_log.CHECK(rgAllGtBboxes[1][0].difficult == false, "The difficult should be true.");

            blobGt.Dispose();
        }

        public void TestGetGroundTruthLabelBBox()
        {
            int nNumGt = 4;
            Blob<T> blobGt = new Blob<T>(m_cuda, m_log, 1, 1, nNumGt, 8);
            double[] rgGt1 = convert(blobGt.mutable_cpu_data);
            float[] rgGt = rgGt1.Select(p => (float)p).ToArray();
            for (int i = 0; i < 4; i++)
            {
                int nImageId = (int)Math.Ceiling(i / 2.0);
                rgGt[i * 8 + 0] = nImageId;
                rgGt[i * 8 + 1] = i;
                rgGt[i * 8 + 2] = 0;
                rgGt[i * 8 + 3] = 0.1f;
                rgGt[i * 8 + 4] = 0.1f;
                rgGt[i * 8 + 5] = 0.3f;
                rgGt[i * 8 + 6] = 0.3f;
                rgGt[i * 8 + 7] = i % 2;
            }

            Dictionary<int, List<NormalizedBBox>> rgAllGtBboxes = m_util.GetGroundTruth(rgGt, nNumGt, -1, true);

            m_log.CHECK_EQ(rgAllGtBboxes.Count, 3, "There should be 3 ground truths.");

            List<KeyValuePair<int, List<NormalizedBBox>>> rgAllGtList = rgAllGtBboxes.ToList();
            m_log.CHECK_EQ(rgAllGtBboxes[0].Count, 1, "The ground truth at 0 should have 1 bbox.");
            m_log.CHECK_EQ(rgAllGtBboxes[0][0].label, 0, "The label should be 0.");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].xmin, 0.1, m_fEps, "The xmin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].ymin, 0.1, m_fEps, "The ymin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].xmax, 0.3, m_fEps, "The xmax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].ymax, 0.3, m_fEps, "The ymax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].size, 0.04, m_fEps, "The size should be 0.04");
            m_log.CHECK(rgAllGtBboxes[0][0].difficult == false, "The difficult should be false.");

            m_log.CHECK_EQ(rgAllGtBboxes[1].Count, 2, "The ground truth at 1 should have 2 bboxes.");
            for (int i = 1; i < 3; i++)
            {
                m_log.CHECK_EQ(rgAllGtBboxes[1][i - 1].label, i, "The label should be 0.");
                m_log.EXPECT_NEAR(rgAllGtBboxes[1][i - 1].xmin, 0.1, m_fEps, "The xmin should be 0.1");
                m_log.EXPECT_NEAR(rgAllGtBboxes[1][i - 1].ymin, 0.1, m_fEps, "The ymin should be 0.1");
                m_log.EXPECT_NEAR(rgAllGtBboxes[1][i - 1].xmax, 0.3, m_fEps, "The xmax should be 0.3");
                m_log.EXPECT_NEAR(rgAllGtBboxes[1][i - 1].ymax, 0.3, m_fEps, "The ymax should be 0.3");
                m_log.EXPECT_NEAR(rgAllGtBboxes[1][i - 1].size, 0.04, m_fEps, "The size should be 0.04");
                m_log.CHECK(rgAllGtBboxes[1][i - 1].difficult == (i % 2 == 1), "The difficult should be " + (i % 2 == 0).ToString() + ".");
            }

            m_log.CHECK_EQ(rgAllGtBboxes[2].Count, 1, "The ground truth at 0 should have 1 bbox.");
            m_log.CHECK_EQ(rgAllGtBboxes[2][0].label, 3, "The label should be 0.");
            m_log.EXPECT_NEAR(rgAllGtBboxes[2][0].xmin, 0.1, m_fEps, "The xmin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[2][0].ymin, 0.1, m_fEps, "The ymin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[2][0].xmax, 0.3, m_fEps, "The xmax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[2][0].ymax, 0.3, m_fEps, "The ymax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[2][0].size, 0.04, m_fEps, "The size should be 0.04");
            m_log.CHECK(rgAllGtBboxes[2][0].difficult == true, "The difficult should be true.");

            // Skip difficult ground truth.
            rgAllGtBboxes = m_util.GetGroundTruth(rgGt, nNumGt, -1, false);

            m_log.CHECK_EQ(rgAllGtBboxes.Count, 2, "There should be 3 ground truths.");

            rgAllGtList = rgAllGtBboxes.ToList();
            m_log.CHECK_EQ(rgAllGtBboxes[0].Count, 1, "The ground truth at 0 should have 1 bbox.");
            m_log.CHECK_EQ(rgAllGtBboxes[0][0].label, 0, "The label should be 0.");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].xmin, 0.1, m_fEps, "The xmin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].ymin, 0.1, m_fEps, "The ymin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].xmax, 0.3, m_fEps, "The xmax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].ymax, 0.3, m_fEps, "The ymax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[0][0].size, 0.04, m_fEps, "The size should be 0.04");
            m_log.CHECK(rgAllGtBboxes[0][0].difficult == false, "The difficult should be false.");

            m_log.CHECK_EQ(rgAllGtBboxes[1].Count, 1, "The ground truth at 0 should have 1 bbox.");
            m_log.CHECK_EQ(rgAllGtBboxes[1][0].label, 2, "The label should be 0.");
            m_log.EXPECT_NEAR(rgAllGtBboxes[1][0].xmin, 0.1, m_fEps, "The xmin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[1][0].ymin, 0.1, m_fEps, "The ymin should be 0.1");
            m_log.EXPECT_NEAR(rgAllGtBboxes[1][0].xmax, 0.3, m_fEps, "The xmax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[1][0].ymax, 0.3, m_fEps, "The ymax should be 0.3");
            m_log.EXPECT_NEAR(rgAllGtBboxes[1][0].size, 0.04, m_fEps, "The size should be 0.04");
            m_log.CHECK(rgAllGtBboxes[1][0].difficult == false, "The difficult should be true.");

            blobGt.Dispose();
        }

        public void TestGetLocPredictionsShared()
        {
            int nNum = 2;
            int nNumPredsPerClass = 2;
            int nNumLocClasses = 1;
            bool bShareLocation = true;
            int nDim = nNumPredsPerClass * nNumLocClasses * 4;
            Blob<T> loc_blob = new Blob<T>(m_cuda, m_log, nNum, nDim, 1, 1);
            double[] rgLoc1 = convert(loc_blob.mutable_cpu_data);
            float[] rgLoc = rgLoc1.Select(p => (float)p).ToArray();

            for (int i = 0; i < nNum; i++)
            {
                for (int j = 0; j < nNumPredsPerClass; j++)
                {
                    int nStartIdx = i * nDim + j * 4;
                    rgLoc[nStartIdx + 0] = i * nNumPredsPerClass * 0.1f + j * 0.1f;
                    rgLoc[nStartIdx + 1] = i * nNumPredsPerClass * 0.1f + j * 0.1f;
                    rgLoc[nStartIdx + 2] = i * nNumPredsPerClass * 0.1f + j * 0.1f + 0.2f;
                    rgLoc[nStartIdx + 3] = i * nNumPredsPerClass * 0.1f + j * 0.1f + 0.2f;
                }
            }

            List<LabelBBox> rgAllLocBBoxes = m_util.GetLocPredictions(rgLoc, nNum, nNumPredsPerClass, nNumLocClasses, bShareLocation);
            m_log.CHECK_EQ(rgAllLocBBoxes.Count, 2, "There should be only 2 label bboxes.");

            for (int i = 0; i < nNum; i++)
            {
                List<KeyValuePair<int, List<NormalizedBBox>>> rg = rgAllLocBBoxes[i].ToList();

                m_log.CHECK_EQ(rg.Count, 1, "There should be 1 Normalized box in the label BBox.");
                m_log.CHECK_EQ(rg[0].Key, -1, "The label should be -1.");

                List<NormalizedBBox> rgBBoxes = rg[0].Value;
                m_log.CHECK_EQ(rgBBoxes.Count, nNumPredsPerClass, "The number of Normalized bboxes should equal the number of predictions per class.");
                float fStartVal = i * nNumPredsPerClass * 0.1f;

                for (int j = 0; j < nNumPredsPerClass; j++)
                {
                    checkBBox(rgBBoxes[j], fStartVal + j * 0.1f, fStartVal + j * 0.1f, fStartVal + j * 0.1f + 0.2f, fStartVal + j * 0.1f + 0.2f);
                }
            }

            loc_blob.Dispose();
        }

        public void TestGetLocPredictionsUnshared()
        {
            int nNum = 2;
            int nNumPredsPerClass = 2;
            int nNumLocClasses = 1;
            bool bShareLocation = true;
            int nDim = nNumPredsPerClass * nNumLocClasses * 4;
            Blob<T> loc_blob = new Blob<T>(m_cuda, m_log, nNum, nDim, 1, 1);
            double[] rgLoc1 = convert(loc_blob.mutable_cpu_data);
            float[] rgLoc = rgLoc1.Select(p => (float)p).ToArray();

            for (int i = 0; i < nNum; i++)
            {
                for (int j = 0; j < nNumPredsPerClass; j++)
                {
                    int nStartIdx = i * nDim + j * 4;
                    rgLoc[nStartIdx + 0] = i * nNumPredsPerClass * 0.1f + j * 0.1f;
                    rgLoc[nStartIdx + 1] = i * nNumPredsPerClass * 0.1f + j * 0.1f;
                    rgLoc[nStartIdx + 2] = i * nNumPredsPerClass * 0.1f + j * 0.1f + 0.2f;
                    rgLoc[nStartIdx + 3] = i * nNumPredsPerClass * 0.1f + j * 0.1f + 0.2f;
                }
            }

            List<LabelBBox> rgAllLocBBoxes = m_util.GetLocPredictions(rgLoc, nNum, nNumPredsPerClass, nNumLocClasses, bShareLocation);
            m_log.CHECK_EQ(rgAllLocBBoxes.Count, 2, "There should be only 2 label bboxes.");

            for (int i = 0; i < nNum; i++)
            {
                List<KeyValuePair<int, List<NormalizedBBox>>> rg = rgAllLocBBoxes[i].ToList();

                m_log.CHECK_EQ(rg.Count, 1, "There should be 1 Normalized box in the label BBox.");
                m_log.CHECK_EQ(rg[0].Key, -1, "The label should be -1.");

                List<NormalizedBBox> rgBBoxes = rg[0].Value;
                m_log.CHECK_EQ(rgBBoxes.Count, nNumPredsPerClass, "The number of Normalized bboxes should equal the number of predictions per class.");
                float fStartVal = i * nNumPredsPerClass * 0.1f;

                for (int j = 0; j < nNumPredsPerClass; j++)
                {
                    checkBBox(rgBBoxes[j], fStartVal + j * 0.1f, fStartVal + j * 0.1f, fStartVal + j * 0.1f + 0.2f, fStartVal + j * 0.1f + 0.2f);
                }
            }

            loc_blob.Dispose();
        }
    }
}
