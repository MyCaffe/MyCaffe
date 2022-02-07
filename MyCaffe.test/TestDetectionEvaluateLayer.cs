using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestDetectionEvaluateLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            DetectionEvaluateLayerTest test = new DetectionEvaluateLayerTest();

            try
            {
                foreach (IDetectionEvaluateLayerTest t in test.Tests)
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
        public void TestForward()
        {
            DetectionEvaluateLayerTest test = new DetectionEvaluateLayerTest();

            try
            {
                foreach (IDetectionEvaluateLayerTest t in test.Tests)
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
        public void TestForwardSkipDifficult()
        {
            DetectionEvaluateLayerTest test = new DetectionEvaluateLayerTest();

            try
            {
                foreach (IDetectionEvaluateLayerTest t in test.Tests)
                {
                    t.TestForwardSkipDifficult();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IDetectionEvaluateLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
        void TestForwardSkipDifficult();
    }

    class DetectionEvaluateLayerTest : TestBase
    {
        public DetectionEvaluateLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("DetectionEvaluate Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new DetectionEvaluateLayerTest<double>(strName, nDeviceID, engine);
            else
                return new DetectionEvaluateLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class DetectionEvaluateLayerTest<T> : TestEx<T>, IDetectionEvaluateLayerTest
    {
        int m_nNumClasses = 3;
        int m_nBackgroundLabelId = 0;
        float m_fOverlapThreshold = 0.3f;
        Blob<T> m_blobBottomDet;
        Blob<T> m_blobBottomGt;
        double m_dfEps = 1e-6;

        public DetectionEvaluateLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            m_blobBottomDet = new Blob<T>(m_cuda, m_log, 1, 1, 8, 7);
            m_blobBottomGt = new Blob<T>(m_cuda, m_log, 1, 1, 4, 8);

            FillData();

            BottomVec.Clear();
            BottomVec.Add(m_blobBottomDet);
            BottomVec.Add(m_blobBottomGt);
        }

        protected override void dispose()
        {
            if (m_blobBottomDet != null)
            {
                m_blobBottomDet.Dispose();
                m_blobBottomDet = null;
            }

            if (m_blobBottomGt != null)
            {
                m_blobBottomGt.Dispose();
                m_blobBottomGt = null;
            }

            base.dispose();
        }

        public void FillData()
        {
            // Fill ground truth.
            bool bIsGt = true;
            FillItem(m_blobBottomGt, 0, "0 1 0 0.1 0.1 0.3 0.3 0", bIsGt);
            FillItem(m_blobBottomGt, 1, "0 1 0 0.6 0.6 0.8 0.8 1", bIsGt);
            FillItem(m_blobBottomGt, 2, "1 2 0 0.3 0.3 0.6 0.5 0", bIsGt);
            FillItem(m_blobBottomGt, 3, "1 1 0 0.7 0.1 0.9 0.3 0", bIsGt);

            // Fill the detections.
            bIsGt = false;
            FillItem(m_blobBottomDet, 0, "0 1 0.3 0.1 0.0 0.4 0.3", bIsGt);
            FillItem(m_blobBottomDet, 1, "0 1 0.7 0.0 0.1 0.2 0.3", bIsGt);
            FillItem(m_blobBottomDet, 2, "0 1 0.9 0.7 0.6 0.8 0.8", bIsGt);
            FillItem(m_blobBottomDet, 3, "1 2 0.8 0.2 0.1 0.4 0.4", bIsGt);
            FillItem(m_blobBottomDet, 4, "1 2 0.1 0.4 0.3 0.7 0.5", bIsGt);
            FillItem(m_blobBottomDet, 5, "1 1 0.2 0.8 0.1 1.0 0.3", bIsGt);
            FillItem(m_blobBottomDet, 6, "1 3 0.2 0.8 0.1 1.0 0.3", bIsGt);
            FillItem(m_blobBottomDet, 7, "2 1 0.2 0.8 0.1 1.0 0.3", bIsGt);
        }

        public void FillItem(Blob<T> blob, int nItem, string strVal, bool bIsGt)
        {
            m_log.CHECK_LT(nItem, blob.height, "The item must be less than the blob height");

            // Split values into vector of items.
            string[] rgstr = strVal.Split(' ');
            List<float> rgfItems = new List<float>();

            foreach (string str in rgstr)
            {
                if (!string.IsNullOrEmpty(str))
                    rgfItems.Add(BaseParameter.ParseFloat(str));
            }

            if (bIsGt)
                m_log.CHECK_EQ(rgfItems.Count, 8, "There should be 8 items for each ground truth.");
            else
                m_log.CHECK_EQ(rgfItems.Count, 7, "There should be 7 items for each non-ground truth.");

            int nNumItems = rgfItems.Count;

            // Fill the item.
            float[] rgfData = Utility.ConvertVecF<T>(blob.mutable_cpu_data);

            for (int i = 0; i < 2; i++)
            {
                rgfData[nItem * nNumItems + i] = (int)rgfItems[i];
            }

            for (int i = 2; i < 7; i++)
            {
                rgfData[nItem * nNumItems + i] = rgfItems[i];
            }

            if (bIsGt)
                rgfData[nItem * nNumItems + 7] = (int)rgfItems[7];

            blob.mutable_cpu_data = Utility.ConvertVec<T>(rgfData);
        }

        public void CheckEqual(Blob<T> blob, int nNum, string strVal)
        {
            // Split values into vector of items.
            string[] rgstr = strVal.Split(' ');
            List<float> rgfItems = new List<float>();

            foreach (string str in rgstr)
            {
                if (!string.IsNullOrEmpty(str))
                    rgfItems.Add(BaseParameter.ParseFloat(str));
            }

            m_log.CHECK_EQ(rgfItems.Count, 5, "There should be 5 items to check!");

            // Check data.
            float[] rgfData = Utility.ConvertVecF<T>(blob.mutable_cpu_data);

            for (int i = 0; i < 5; i++)
            {
                if (i == 2)
                    m_log.EXPECT_NEAR_FLOAT(rgfData[nNum * blob.width + i], rgfItems[i], m_dfEps);
                else
                    m_log.CHECK_EQ((int)rgfData[nNum * blob.width + i], (int)rgfItems[i], "The items are not equal!");
            }
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DETECTION_EVALUATE);
            p.detection_evaluate_param.num_classes = (uint)m_nNumClasses;
            p.detection_evaluate_param.background_label_id = (uint)m_nBackgroundLabelId;
            p.detection_evaluate_param.overlap_threshold = m_fOverlapThreshold;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);
            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, 1, "The top num should = 1.");
                m_log.CHECK_EQ(Top.channels, 1, "The top channels should = 1.");
                m_log.CHECK_EQ(Top.height, m_blobBottomDet.height + 2, "The top height should = " + (m_blobBottomDet.height + 2).ToString() + "!");
                m_log.CHECK_EQ(Top.width, 5, "The top width should = 5.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DETECTION_EVALUATE);
            p.detection_evaluate_param.num_classes = (uint)m_nNumClasses;
            p.detection_evaluate_param.background_label_id = (uint)m_nBackgroundLabelId;
            p.detection_evaluate_param.overlap_threshold = m_fOverlapThreshold;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                FillData();
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, 1, "The top num should = 1.");
                m_log.CHECK_EQ(Top.channels, 1, "The top channels should = 1.");
                m_log.CHECK_EQ(Top.height, m_blobBottomDet.height + 2, "The top height should = " + (m_blobBottomDet.height + 2).ToString() + "!");
                m_log.CHECK_EQ(Top.width, 5, "The top width should = 5.");

                CheckEqual(Top, 0, "-1 1 3 -1 -1");
                CheckEqual(Top, 1, "-1 2 1 -1 -1");
                CheckEqual(Top, 2, "0 1 0.9 1 0");
                CheckEqual(Top, 3, "0 1 0.7 1 0");
                CheckEqual(Top, 4, "0 1 0.3 0 1");
                CheckEqual(Top, 5, "1 1 0.2 1 0");
                CheckEqual(Top, 6, "1 2 0.8 0 1");
                CheckEqual(Top, 7, "1 2 0.1 1 0");
                CheckEqual(Top, 8, "1 3 0.2 0 1");
                CheckEqual(Top, 9, "2 1 0.2 0 1");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardSkipDifficult()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DETECTION_EVALUATE);
            p.detection_evaluate_param.num_classes = (uint)m_nNumClasses;
            p.detection_evaluate_param.background_label_id = (uint)m_nBackgroundLabelId;
            p.detection_evaluate_param.overlap_threshold = m_fOverlapThreshold;
            p.detection_evaluate_param.evaulte_difficult_gt = false;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            try
            {
                FillData();
                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.num, 1, "The top num should = 1.");
                m_log.CHECK_EQ(Top.channels, 1, "The top channels should = 1.");
                m_log.CHECK_EQ(Top.height, m_blobBottomDet.height + 2, "The top height should = " + (m_blobBottomDet.height + 2).ToString() + "!");
                m_log.CHECK_EQ(Top.width, 5, "The top width should = 5.");

                CheckEqual(Top, 0, "-1 1 2 -1 -1");
                CheckEqual(Top, 1, "-1 2 1 -1 -1");
                CheckEqual(Top, 2, "0 1 0.9 0 0");
                CheckEqual(Top, 3, "0 1 0.7 1 0");
                CheckEqual(Top, 4, "0 1 0.3 0 1");
                CheckEqual(Top, 5, "1 1 0.2 1 0");
                CheckEqual(Top, 6, "1 2 0.8 0 1");
                CheckEqual(Top, 7, "1 2 0.1 1 0");
                CheckEqual(Top, 8, "1 3 0.2 0 1");
                CheckEqual(Top, 9, "2 1 0.2 0 1");
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
