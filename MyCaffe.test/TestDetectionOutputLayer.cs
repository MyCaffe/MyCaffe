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
    public class TestDetectionOutputLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            DetectionOutputLayerTest test = new DetectionOutputLayerTest();

            try
            {
                foreach (IDetectionOutputLayerTest t in test.Tests)
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
        public void TestForwardSharedLocation()
        {
            DetectionOutputLayerTest test = new DetectionOutputLayerTest();

            try
            {
                foreach (IDetectionOutputLayerTest t in test.Tests)
                {
                    t.TestForwardSharedLocation();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardSharedLocationTopK()
        {
            DetectionOutputLayerTest test = new DetectionOutputLayerTest();

            try
            {
                foreach (IDetectionOutputLayerTest t in test.Tests)
                {
                    t.TestForwardSharedLocationTopK();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardNoSharedLocation()
        {
            DetectionOutputLayerTest test = new DetectionOutputLayerTest();

            try
            {
                foreach (IDetectionOutputLayerTest t in test.Tests)
                {
                    t.TestForwardNoSharedLocation();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardNoSharedLocationTopK()
        {
            DetectionOutputLayerTest test = new DetectionOutputLayerTest();

            try
            {
                foreach (IDetectionOutputLayerTest t in test.Tests)
                {
                    t.TestForwardNoSharedLocationTopK();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardNoSharedLocationNeg0()
        {
            DetectionOutputLayerTest test = new DetectionOutputLayerTest();

            try
            {
                foreach (IDetectionOutputLayerTest t in test.Tests)
                {
                    t.TestForwardNoSharedLocationNeg0();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardNoSharedLocationNeg0TopK()
        {
            DetectionOutputLayerTest test = new DetectionOutputLayerTest();

            try
            {
                foreach (IDetectionOutputLayerTest t in test.Tests)
                {
                    t.TestForwardNoSharedLocationNeg0TopK();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IDetectionOutputLayerTest : ITest
    {
        void TestSetup();
        void TestForwardSharedLocation();
        void TestForwardSharedLocationTopK();
        void TestForwardNoSharedLocation();
        void TestForwardNoSharedLocationTopK();
        void TestForwardNoSharedLocationNeg0();
        void TestForwardNoSharedLocationNeg0TopK();
    }

    class DetectionOutputLayerTest : TestBase
    {
        public DetectionOutputLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("DetectionOutput Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new DetectionOutputLayerTest<double>(strName, nDeviceID, engine);
            else
                return new DetectionOutputLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class DetectionOutputLayerTest<T> : TestEx<T>, IDetectionOutputLayerTest
    {
        int m_nNum = 2;
        int m_nNumPriors = 4;
        int m_nNumClasses = 2;
        bool m_bShareLocation = true;
        int m_nNumLocClasses = 1;
        float m_fNmsTheshold = 0.1f;
        int m_nTopK = 2;
        Blob<T> m_blobBottomLoc;
        Blob<T> m_blobBottomConf;
        Blob<T> m_blobBottomPrior;
        float m_fEps = 1e-6f;


        public DetectionOutputLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            m_nNumLocClasses = (m_bShareLocation) ? 1 : m_nNumClasses;
            m_blobBottomLoc = new Blob<T>(m_cuda, m_log, m_nNum, m_nNumPriors * m_nNumLocClasses * 4, 1, 1);
            m_blobBottomConf = new Blob<T>(m_cuda, m_log, m_nNum, m_nNumPriors * m_nNumClasses, 1, 1);
            m_blobBottomPrior = new Blob<T>(m_cuda, m_log, m_nNum, 2, m_nNumPriors * 4, 1);

            FillData();

            BottomVec.Clear();
            BottomVec.Add(m_blobBottomLoc);
            BottomVec.Add(m_blobBottomConf);
            BottomVec.Add(m_blobBottomPrior);
        }

        protected override void dispose()
        {
            if (m_blobBottomPrior != null)
            {
                m_blobBottomPrior.Dispose();
                m_blobBottomPrior = null;
            }

            if (m_blobBottomLoc != null)
            {
                m_blobBottomLoc.Dispose();
                m_blobBottomLoc = null;
            }

            if (m_blobBottomConf != null)
            {
                m_blobBottomConf.Dispose();
                m_blobBottomConf = null;
            }            

            base.dispose();
        }

        public void FillData()
        {
            // Fill Prior data first.
            float[] rgdfPrioData = convertF(m_blobBottomPrior.mutable_cpu_data);
            float fStep = 0.5f;
            float fBoxSize = 0.3f;
            int nIdx = 0;

            for (int h = 0; h < 2; h++)
            {
                float fCenterY = (h + 0.5f) * fStep;

                for (int w = 0; w < 2; w++)
                {
                    float fCenterX = (w + 0.5f) * fStep;

                    rgdfPrioData[nIdx] = (fCenterX - fBoxSize / 2);
                    nIdx++;
                    rgdfPrioData[nIdx] = (fCenterY - fBoxSize / 2);
                    nIdx++;
                    rgdfPrioData[nIdx] = (fCenterX + fBoxSize / 2);
                    nIdx++;
                    rgdfPrioData[nIdx] = (fCenterY + fBoxSize / 2);
                    nIdx++;
                }
            }

            for (int i = 0; i < nIdx; i++)
            {
                rgdfPrioData[nIdx + i] = 0.1f;
            }

            m_blobBottomPrior.mutable_cpu_data = convert(rgdfPrioData);

            // Fill confidences
            float[] rgfConfData = convertF(m_blobBottomConf.mutable_cpu_data);
            nIdx = 0;

            for (int i = 0; i < m_nNum; i++)
            {
                for (int j = 0; j < m_nNumPriors; j++)
                {
                    for (int c = 0; c < m_nNumClasses; c++)
                    {
                        if (i % 2 == c % 2)
                            rgfConfData[nIdx] = j * 0.2f;
                        else
                            rgfConfData[nIdx] = 1 - j * 0.2f;

                        nIdx++;
                    }
                }
            }

            m_blobBottomConf.mutable_cpu_data = convert(rgfConfData);
        }

        public void FillLocData(bool bShareLocation = true)
        {
            // Fill location offsets.
            int nNumLocClasses = (bShareLocation) ? 1 : m_nNumClasses;
            m_blobBottomLoc.Reshape(m_nNum, m_nNumPriors * nNumLocClasses * 4, 1, 1);

            float[] rgfLocData = convertF(m_blobBottomLoc.mutable_cpu_data);
            int nIdx = 0;

            for (int i = 0; i < m_nNum; i++)
            {
                for (int h = 0; h < 2; h++)
                {
                    for (int w = 0; w < 2; w++)
                    {
                        for (int c = 0; c < nNumLocClasses; c++)
                        {
                            int nW = (w % 2 == 1) ? -1 : 1;
                            int nH = (h % 2 == 1) ? -1 : 1;
                            float fVal = (i * 1 + c / 2.0f + 0.5f);

                            rgfLocData[nIdx] = nW * fVal;
                            nIdx++;
                            rgfLocData[nIdx] = nH * fVal;
                            nIdx++;
                            rgfLocData[nIdx] = nW * fVal;
                            nIdx++;
                            rgfLocData[nIdx] = nH * fVal;
                            nIdx++;
                        }
                    }
                }
            }

            m_blobBottomLoc.mutable_cpu_data = convert(rgfLocData);
        }

        public void CheckEqual(Blob<T> blob, int nNum, string strValues)
        {
            m_log.CHECK_LT(nNum, blob.height, "The blob height should be > the num '" + nNum.ToString() + "'!");

            // Split values into a vector of items.
            List<float> rgf = new List<float>();
            string[] rgstr = strValues.Split(' ');

            foreach (string str in rgstr)
            {
                rgf.Add(BaseParameter.ParseFloat(str));
            }

            // Check the data.
            float[] rgfData = convertF(blob.mutable_cpu_data);

            for (int i = 0; i < 2; i++)
            {
                float fVal = rgfData[nNum * blob.width + i];
                m_log.CHECK_EQ(fVal, rgf[i], "The data values are not equal.");
            }

            for (int i = 2; i < 7; i++)
            {
                float fVal = rgfData[nNum * blob.width + i];
                m_log.EXPECT_NEAR_FLOAT(fVal, rgf[i], m_fEps);
            }
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DETECTION_OUTPUT);
            p.detection_output_param.num_classes = (uint)m_nNumClasses;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, 1, "The top num should = 1.");
            m_log.CHECK_EQ(Top.channels, 1, "The top channels should = 1.");
            m_log.CHECK_EQ(Top.height, 1, "The top height should = 1.");
            m_log.CHECK_EQ(Top.width, 7, "The top width should = 7.");
        }

        public void TestForwardSharedLocation()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DETECTION_OUTPUT);
            p.detection_output_param.num_classes = (uint)m_nNumClasses;
            p.detection_output_param.share_location = true;
            p.detection_output_param.background_label_id = 0;
            p.detection_output_param.nms_param.nms_threshold = m_fNmsTheshold;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            FillLocData(true);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, 1, "The top num should = 1.");
            m_log.CHECK_EQ(Top.channels, 1, "The top channels should = 1.");
            m_log.CHECK_EQ(Top.height, 6, "The top height should = 6.");
            m_log.CHECK_EQ(Top.width, 7, "The top width should = 7.");

            CheckEqual(Top, 0, "0 1 1.0 0.15 0.15 0.45 0.45");
            CheckEqual(Top, 1, "0 1 0.8 0.55 0.15 0.85 0.45");
            CheckEqual(Top, 2, "0 1 0.6 0.15 0.55 0.45 0.85");
            CheckEqual(Top, 3, "0 1 0.4 0.55 0.55 0.85 0.85");
            CheckEqual(Top, 4, "1 1 0.6 0.45 0.45 0.75 0.75");
            CheckEqual(Top, 5, "1 1 0.0 0.25 0.25 0.55 0.55");
        }

        public void TestForwardSharedLocationTopK()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DETECTION_OUTPUT);
            p.detection_output_param.num_classes = (uint)m_nNumClasses;
            p.detection_output_param.share_location = true;
            p.detection_output_param.background_label_id = 0;
            p.detection_output_param.nms_param.nms_threshold = m_fNmsTheshold;
            p.detection_output_param.nms_param.top_k = m_nTopK;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            FillLocData(true);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, 1, "The top num should = 1.");
            m_log.CHECK_EQ(Top.channels, 1, "The top channels should = 1.");
            m_log.CHECK_EQ(Top.height, 3, "The top height should = 3.");
            m_log.CHECK_EQ(Top.width, 7, "The top width should = 7.");

            CheckEqual(Top, 0, "0 1 1.0 0.15 0.15 0.45 0.45");
            CheckEqual(Top, 1, "0 1 0.8 0.55 0.15 0.85 0.45");
            CheckEqual(Top, 2, "1 1 0.6 0.45 0.45 0.75 0.75");
        }

        public void TestForwardNoSharedLocation()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DETECTION_OUTPUT);
            p.detection_output_param.num_classes = (uint)m_nNumClasses;
            p.detection_output_param.share_location = false;
            p.detection_output_param.background_label_id = -1;
            p.detection_output_param.nms_param.nms_threshold = m_fNmsTheshold;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            FillLocData(false);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, 1, "The top num should = 1.");
            m_log.CHECK_EQ(Top.channels, 1, "The top channels should = 1.");
            m_log.CHECK_EQ(Top.height, 11, "The top height should = 11.");
            m_log.CHECK_EQ(Top.width, 7, "The top width should = 7.");

            CheckEqual(Top, 0, "0 0 0.6 0.55 0.55 0.85 0.85");
            CheckEqual(Top, 1, "0 0 0.4 0.15 0.55 0.45 0.85");
            CheckEqual(Top, 2, "0 0 0.2 0.55 0.15 0.85 0.45");
            CheckEqual(Top, 3, "0 0 0.0 0.15 0.15 0.45 0.45");
            CheckEqual(Top, 4, "0 1 1.0 0.20 0.20 0.50 0.50");
            CheckEqual(Top, 5, "0 1 0.8 0.50 0.20 0.80 0.50");
            CheckEqual(Top, 6, "0 1 0.6 0.20 0.50 0.50 0.80");
            CheckEqual(Top, 7, "0 1 0.4 0.50 0.50 0.80 0.80");
            CheckEqual(Top, 8, "1 0 1.0 0.25 0.25 0.55 0.55");
            CheckEqual(Top, 9, "1 0 0.4 0.45 0.45 0.75 0.75");
            CheckEqual(Top, 10, "1 1 0.6 0.40 0.40 0.70 0.70");
        }

        public void TestForwardNoSharedLocationTopK()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DETECTION_OUTPUT);
            p.detection_output_param.num_classes = (uint)m_nNumClasses;
            p.detection_output_param.share_location = false;
            p.detection_output_param.background_label_id = -1;
            p.detection_output_param.nms_param.nms_threshold = m_fNmsTheshold;
            p.detection_output_param.nms_param.top_k = m_nTopK;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            FillLocData(false);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, 1, "The top num should = 1.");
            m_log.CHECK_EQ(Top.channels, 1, "The top channels should = 1.");
            m_log.CHECK_EQ(Top.height, 6, "The top height should = 6.");
            m_log.CHECK_EQ(Top.width, 7, "The top width should = 7.");

            CheckEqual(Top, 0, "0 0 0.6 0.55 0.55 0.85 0.85");
            CheckEqual(Top, 1, "0 0 0.4 0.15 0.55 0.45 0.85");
            CheckEqual(Top, 2, "0 1 1.0 0.20 0.20 0.50 0.50");
            CheckEqual(Top, 3, "0 1 0.8 0.50 0.20 0.80 0.50");
            CheckEqual(Top, 4, "1 0 1.0 0.25 0.25 0.55 0.55");
            CheckEqual(Top, 5, "1 1 0.6 0.40 0.40 0.70 0.70");
        }

        public void TestForwardNoSharedLocationNeg0()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DETECTION_OUTPUT);
            p.detection_output_param.num_classes = (uint)m_nNumClasses;
            p.detection_output_param.share_location = false;
            p.detection_output_param.background_label_id = 0;
            p.detection_output_param.nms_param.nms_threshold = m_fNmsTheshold;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            FillLocData(false);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, 1, "The top num should = 1.");
            m_log.CHECK_EQ(Top.channels, 1, "The top channels should = 1.");
            m_log.CHECK_EQ(Top.height, 5, "The top height should = 5.");
            m_log.CHECK_EQ(Top.width, 7, "The top width should = 7.");

            CheckEqual(Top, 0, "0 1 1.0 0.20 0.20 0.50 0.50");
            CheckEqual(Top, 1, "0 1 0.8 0.50 0.20 0.80 0.50");
            CheckEqual(Top, 2, "0 1 0.6 0.20 0.50 0.50 0.80");
            CheckEqual(Top, 3, "0 1 0.4 0.50 0.50 0.80 0.80");
            CheckEqual(Top, 4, "1 1 0.6 0.40 0.40 0.70 0.70");
        }

        public void TestForwardNoSharedLocationNeg0TopK()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DETECTION_OUTPUT);
            p.detection_output_param.num_classes = (uint)m_nNumClasses;
            p.detection_output_param.share_location = false;
            p.detection_output_param.background_label_id = 0;
            p.detection_output_param.nms_param.nms_threshold = m_fNmsTheshold;
            p.detection_output_param.nms_param.top_k = m_nTopK;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null);

            FillLocData(false);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, 1, "The top num should = 1.");
            m_log.CHECK_EQ(Top.channels, 1, "The top channels should = 1.");
            m_log.CHECK_EQ(Top.height, 3, "The top height should = 3.");
            m_log.CHECK_EQ(Top.width, 7, "The top width should = 7.");

            CheckEqual(Top, 0, "0 1 1.0 0.20 0.20 0.50 0.50");
            CheckEqual(Top, 1, "0 1 0.8 0.50 0.20 0.80 0.50");
            CheckEqual(Top, 2, "1 1 0.6 0.40 0.40 0.70 0.70");
        }
    }
}
