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
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;

namespace MyCaffe.test
{
    [TestClass]
    public class TestAnnotatedDataLayer
    {
        [TestMethod]
        public void TestReadDb()
        {
            AnnotatedDataLayerTest test = new AnnotatedDataLayerTest();

            try
            {
                foreach (IAnnotatedDataLayerTest t in test.Tests)
                {
                    t.TestReadDb();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReshapeDb()
        {
            AnnotatedDataLayerTest test = new AnnotatedDataLayerTest();

            try
            {
                foreach (IAnnotatedDataLayerTest t in test.Tests)
                {
                    t.TestReshapeDb();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadCropTrainSequenceSeededDb()
        {
            AnnotatedDataLayerTest test = new AnnotatedDataLayerTest();

            try
            {
                foreach (IAnnotatedDataLayerTest t in test.Tests)
                {
                    t.TestReadCropTrainSequenceSeededDb();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadCropTrainSequenceUnseededDb()
        {
            AnnotatedDataLayerTest test = new AnnotatedDataLayerTest();

            try
            {
                foreach (IAnnotatedDataLayerTest t in test.Tests)
                {
                    t.TestReadCropTrainSequenceUnseededDb();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadCropTestDb()
        {
            AnnotatedDataLayerTest test = new AnnotatedDataLayerTest();

            try
            {
                foreach (IAnnotatedDataLayerTest t in test.Tests)
                {
                    t.TestReadCropTestDb();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IAnnotatedDataLayerTest : ITest
    {
        void TestReadDb();
        void TestReshapeDb();
        void TestReadCropTrainSequenceSeededDb();
        void TestReadCropTrainSequenceUnseededDb();
        void TestReadCropTestDb();
    }

    class AnnotatedDataLayerTest : TestBase
    {
        SettingsCaffe m_settings;
        MyCaffeImageDatabase m_db;
        CancelEvent m_evtCancel = new CancelEvent();

        public AnnotatedDataLayerTest(string strDs = null, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Annotated Data Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
            m_settings = new SettingsCaffe();
            m_settings.EnableLabelBalancing = false;
            m_settings.EnableLabelBoosting = false;
            m_settings.EnablePairInputSelection = false;
            m_settings.EnableRandomInputSelection = false;

            if (strDs != null && strDs.Length > 0)
            {
                m_db = new MyCaffeImageDatabase();
                m_db.InitializeWithDsName(m_settings, strDs);
            }
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            string strPath = TestBase.CudaPath;

            if (dt == common.DataType.DOUBLE)
            {
                CudaDnn<double>.SetDefaultCudaPath(strPath);
                return new AnnotatedDataLayerTest<double>(strName, nDeviceID, engine, this);
            }
            else
            {
                CudaDnn<float>.SetDefaultCudaPath(strPath);
                return new AnnotatedDataLayerTest<float>(strName, nDeviceID, engine, this);
            }
        }

        protected override void dispose()
        {
            if (m_db != null)
            {
                m_db.Dispose();
                m_db = null;
            }

            base.dispose();
        }

        public string SourceName
        {
            get { return "MNIST.training"; }
        }

        public MyCaffeImageDatabase db
        {
            get { return m_db; }
        }

        public SettingsCaffe Settings
        {
            get { return m_settings; }
        }

        public CancelEvent CancelEvent
        {
            get { return m_evtCancel; }
        }
    }

    class AnnotatedDataLayerTest<T> : TestEx<T>, IAnnotatedDataLayerTest
    {
        AnnotatedDataLayerTest m_parent;
        double m_dfEps = 1e-6;
        int m_nSpatialDim;
        int m_nSize;
        int m_nNum = 6;
        int m_nChannels = 2;
        int m_nHeight = 10;
        int m_nWidth = 10;
        DataParameter.DB m_backend = DataParameter.DB.IMAGEDB;
        bool m_bUseUniquePixel;
        bool m_bUseUniqueAnnotation;
        bool m_bUseRichAnnotation;
        SimpleDatum.ANNOTATION_TYPE m_annoType;
        string m_strSrc1 = "test_ssd_data";
        string m_strSrc2 = "test_ssd_data.t";
        int m_nSrcID1 = 0;
        int m_nSrcID2 = 0;
        int m_nDsID = 0;
        Blob<T> m_blobTopLabel;
        int m_kNumChoices = 2;
        bool[] m_rgkBoolChoices = new bool[] { false, true };

        public AnnotatedDataLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine, AnnotatedDataLayerTest parent)
            : base(strName, new List<int>() { 6, 2, 10, 10 }, nDeviceID)
        {
            m_parent = parent;
            m_engine = engine;
            m_blobTopLabel = new Blob<T>(m_cuda, m_log);
            Setup();
        }

        protected override void dispose()
        {
            if (m_blobTopLabel != null)
            {
                m_blobTopLabel.Dispose();
                m_blobTopLabel = null;
            }

            base.dispose();
        }

        public void Setup()
        {
            m_nSpatialDim = m_nHeight * m_nWidth;
            m_nSize = m_nChannels * m_nSpatialDim;
            TopVec.Clear();
            TopVec.Add(Top);
            TopVec.Add(m_blobTopLabel);
        }

        /// <summary>
        /// Fill the DB with data.
        /// </summary>
        /// <param name="backend">Can only be IMAGEDB.</param>
        /// <param name="bUniquePixel">Specifies whether or not each pixel is unique or all images are the same.</param>
        /// <param name="bUniqueAnnotation">Specifies whether each annotation ia group is unique but all groups are the same
        /// at the same positions; else each group is unique but all annotations within a group are the same.</param>
        /// <param name="bUseRichAnnotation">Specifies whether or not to use rich annoations, when <i>false</i> the datum.label is used instead.</param>
        /// <param name="type">Specifies the type of rich annoation to use.</param>
        /// <returns>The name of the training data source is returned.</returns>
        public string Fill(DataParameter.DB backend, bool bUniquePixel, bool bUniqueAnnotation, bool bUseRichAnnotation, SimpleDatum.ANNOTATION_TYPE type)
        {
            m_backend = backend;
            m_bUseUniquePixel = bUniquePixel;
            m_bUseUniqueAnnotation = bUniqueAnnotation;
            m_bUseRichAnnotation = bUseRichAnnotation;
            m_annoType = type;

            DatasetFactory factory = new DatasetFactory();

            m_log.WriteLine("Creating temporary dataset '" + m_strSrc1 + "'.");
            SourceDescriptor srcTrain = new SourceDescriptor(0, m_strSrc1, m_nWidth, m_nHeight, m_nChannels, false, true);
            srcTrain.ID = factory.AddSource(srcTrain);
            m_nSrcID1 = srcTrain.ID;
            SourceDescriptor srcTest = new SourceDescriptor(0, m_strSrc2, m_nWidth, m_nHeight, m_nChannels, false, true);
            srcTest.ID = factory.AddSource(srcTest);
            m_nSrcID2 = srcTest.ID;

            factory.DeleteSourceData(srcTrain.ID);
            factory.DeleteSourceData(srcTest.ID);

            List<SourceDescriptor> rgSrcId = new List<SourceDescriptor>() { srcTrain, srcTest };

            for (int k = 0; k < 2; k++)
            {
                factory.Open(rgSrcId[k]);
                factory.DeleteSourceData();

                int nCount = factory.GetImageCount();
                for (int i = nCount; i < nCount + m_nNum; i++)
                {
                    List<byte> rgData = new List<byte>();

                    for (int j = 0; j < m_nSize; j++)
                    {
                        int datum = bUniquePixel ? j : i;
                        rgData.Add((byte)datum);
                    }

                    SimpleDatum sd = new SimpleDatum(false, m_nChannels, m_nWidth, m_nHeight, i, DateTime.Today, rgData, null, 0, false, i);

                    // Fill the annotation.
                    if (bUseRichAnnotation)
                    {
                        sd.annotation_type = m_annoType;
                        sd.annotation_group = new List<AnnotationGroup>();

                        for (int g=0; g<i; g++)
                        {
                            AnnotationGroup anno_group = new AnnotationGroup(null, g);
                            sd.annotation_group.Add(anno_group);

                            for (int a = 0; a < g; a++)
                            {
                                Annotation anno = new Annotation(null, a);
                                anno_group.annotations.Add(anno);

                                if (type == SimpleDatum.ANNOTATION_TYPE.BBOX)
                                {
                                    int b = (bUniqueAnnotation) ? a: g;
                                    anno.bbox = new NormalizedBBox(b * 0.1f, b * 0.1f, Math.Min(b * 0.1f + 0.2f, 1.0f), Math.Min(b * 0.1f + 0.2f, 1.0f), 0, (a % 2 == 1) ? true : false);
                                }
                            }
                        }
                    }
                    
                    factory.PutRawImage(i, sd);
                }

                factory.Close();
            }

            DatasetDescriptor ds = new DatasetDescriptor(0, "test_data", null, null, srcTrain, srcTest, null, null);
            ds.ID = factory.AddDataset(ds);

            factory.UpdateDatasetCounts(ds.ID);
            m_nDsID = ds.ID;

            return m_strSrc1;
        }

        private int OneBBoxNum(int n)
        {
            int nSum = 0;

            for (int g = 0; g < n; g++)
            {
                nSum += g;
            }

            return nSum;
        }

        private int BBoxNum(int n)
        {
            int nSum = 0;

            for (int i = 0; i < n; i++)
            {
                for (int g = 0; g < i; g++)
                {
                    nSum += g;
                }
            }

            return nSum;
        }

        public void TestRead()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ANNOTATED_DATA);
            p.phase = Phase.TRAIN;
            p.data_param.batch_size = (uint)m_nNum;
            p.data_param.source = m_strSrc1;
            p.data_param.backend = m_backend;

            double dfScale = 3;
            p.transform_param.scale = dfScale;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, m_nNum, "The top num is incorrect.");
            m_log.CHECK_EQ(Top.channels, m_nChannels, "The top channels are incorrect.");
            m_log.CHECK_EQ(Top.height, m_nHeight, "The top height is incorrect.");
            m_log.CHECK_EQ(Top.width, m_nWidth, "The top width is incorrect.");

            if (m_bUseRichAnnotation)
            {
                switch (m_annoType)
                {
                    case SimpleDatum.ANNOTATION_TYPE.BBOX:
                        m_log.CHECK_EQ(m_blobTopLabel.num, 1, "The top label num is incorrect.");
                        m_log.CHECK_EQ(m_blobTopLabel.channels, 1, "The top channels are incorrect.");
                        m_log.CHECK_EQ(m_blobTopLabel.height, 1, "The top height is incorrect.");
                        m_log.CHECK_EQ(m_blobTopLabel.width, 8, "The top width is incorrect.");
                        break;

                    default:
                        m_log.FAIL("Unknown annotation type.");
                        break;
                }
            }
            else
            {
                m_log.CHECK_EQ(m_blobTopLabel.num, m_nNum, "The top label num is incorrect.");
                m_log.CHECK_EQ(m_blobTopLabel.channels, m_nChannels, "The top channels are incorrect.");
                m_log.CHECK_EQ(m_blobTopLabel.height, m_nHeight, "The top height is incorrect.");
                m_log.CHECK_EQ(m_blobTopLabel.width, m_nWidth, "The top width is incorrect.");
            }

            for (int n = 0; n < 5; n++)
            {
                layer.Forward(BottomVec, TopVec);

                // Check the label.
                double[] rgLabelData = convert(m_blobTopLabel.mutable_cpu_data);
                double[] rgTopData = convert(Top.mutable_cpu_data);
                int nCurBbox = 0;

                for (int i = 0; i < m_nNum; i++)
                {
                    if (m_bUseRichAnnotation)
                    {
                        if (m_annoType == SimpleDatum.ANNOTATION_TYPE.BBOX)
                        {
                            m_log.CHECK_EQ(m_blobTopLabel.num, 1, "The top label num is incorrect.");
                            m_log.CHECK_EQ(m_blobTopLabel.channels, 1, "The top channels are incorrect.");
                            m_log.CHECK_EQ(m_blobTopLabel.height, BBoxNum(i), "The top height is incorrect.");
                            m_log.CHECK_EQ(m_blobTopLabel.width, m_nWidth, "The top width is incorrect.");

                            for (int g = 0; g < i; g++)
                            {
                                for (int a = 0; a < g; a++)
                                {
                                    m_log.CHECK_EQ(i, rgLabelData[nCurBbox * 8 + 0], "The label data is incorrect.");
                                    m_log.CHECK_EQ(g, rgLabelData[nCurBbox * 8 + 1], "The label data is incorrect.");
                                    m_log.CHECK_EQ(a, rgLabelData[nCurBbox * 8 + 2], "The label data is incorrect.");
                                    int b = (m_bUseUniqueAnnotation) ? a : g;

                                    for (int p1 = 3; p1 < 5; p1++)
                                    {
                                        m_log.EXPECT_NEAR_FLOAT(Math.Min(b * 0.1f + 0.2f, 1.0f), rgLabelData[nCurBbox * 8 + p1], m_dfEps);
                                    }

                                    m_log.CHECK_EQ(a % 2, rgLabelData[nCurBbox * 8 + 7], "The label data is incorrect.");
                                    nCurBbox++;
                                }
                            }
                        }
                        else
                        {
                            m_log.FAIL("Unknown annoation type.");
                        }
                    }
                    else
                    {
                        m_log.CHECK_EQ(i, rgLabelData[i], "The label is incorrect.");
                    }
                }

                // Check data.
                for (int i = 0; i < m_nNum; i++)
                {
                    for (int j = 0; j < m_nSize; j++)
                    {
                        m_log.CHECK_EQ(dfScale * ((m_bUseUniquePixel) ? j : i), rgTopData[i * m_nSize + j], "debug: iter " + n.ToString() + " i " + i.ToString() + " j " + j.ToString());
                    }
                }
            }

            layer.Dispose();
        }

        public void TestReshape(bool bUniquePixel, bool bUniqueAnnotation, bool bUseRichAnnotation, SimpleDatum.ANNOTATION_TYPE type)
        {
            DatasetFactory factory = new DatasetFactory();

            m_log.WriteLine("Creating temporary dataset '" + m_strSrc2 + "'.");
            SourceDescriptor srcTest = new SourceDescriptor(0, m_strSrc2, m_nWidth, m_nHeight, m_nChannels, false, true);
            srcTest.ID = factory.AddSource(srcTest);
            m_nSrcID1 = srcTest.ID;

            factory.DeleteSourceData(srcTest.ID);
            factory.Open(srcTest.ID);

            for (int i = 0; i < m_nNum; i++)
            {
                SimpleDatum datum = new SimpleDatum(true, m_nChannels, i % 4 + 1, i % 2 + 1, 0, DateTime.Now);

                // Fill data.
                for (int j = 0; j < datum.ByteData.Length; j++)
                {
                    datum.RealData[j] = j;
                }

                // Fill annotation.
                if (bUseRichAnnotation)
                {
                    datum.annotation_type = type;
                    datum.annotation_group = new List<AnnotationGroup>();

                    for (int g = 0; g < i; g++)
                    {
                        AnnotationGroup anno_group = new AnnotationGroup(null, g);
                        datum.annotation_group.Add(anno_group);

                        for (int a = 0; a < g; a++)
                        {
                            Annotation anno = new Annotation(null, a);

                            if (type == SimpleDatum.ANNOTATION_TYPE.BBOX)
                            {
                                int b = (bUniqueAnnotation) ? a : g;
                                anno.bbox = new NormalizedBBox(b * 0.1f, b * 0.1f, Math.Min(b * 0.1f + 0.2f, 1.0f), Math.Min(b * 0.1f + 0.2f, 1.0f), 0, (a % 2 == 0) ? false : true);
                            }
                        }
                    }
                }
                else
                {
                    datum.SetLabel(i);
                }

                factory.PutRawImage(i, datum);
            }

            factory.Close();

            // Load and check data of various shapes.
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ANNOTATED_DATA);
            p.phase = Phase.TEST;
            p.data_param.batch_size = (uint)m_nNum;
            p.data_param.source = srcTest.Name;
            p.data_param.backend = DataParameter.DB.IMAGEDB;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, 1, "The top should have a num = 1.");
            m_log.CHECK_EQ(Top.channels, m_nChannels, "The top should have channels = " + m_nChannels.ToString() + ".");

            if (bUseRichAnnotation)
            {
                switch (type)
                {
                    case SimpleDatum.ANNOTATION_TYPE.BBOX:
                        m_log.CHECK_EQ(m_blobTopLabel.num, 1, "The top label should have num = 1.");
                        m_log.CHECK_EQ(m_blobTopLabel.channels, 1, "The top label should have channels = 1.");
                        m_log.CHECK_EQ(m_blobTopLabel.height, 1, "The top label should have height = 1.");
                        m_log.CHECK_EQ(m_blobTopLabel.width, 8, "The top label should have width = 8.");
                        break;

                    default:
                        m_log.FAIL("Unknown annotation type.");
                        break;
                }
            }
            else
            {
                m_log.CHECK_EQ(m_blobTopLabel.num, 1, "The top label should have num = 1.");
                m_log.CHECK_EQ(m_blobTopLabel.channels, 1, "The top label should have channels = 1.");
                m_log.CHECK_EQ(m_blobTopLabel.height, 1, "The top label should have height = 1.");
                m_log.CHECK_EQ(m_blobTopLabel.width, 1, "The top label should have width = 1.");
            }

            for (int i=0; i<3; i++)
            {
                layer.Forward(BottomVec, TopVec);
                m_log.CHECK_EQ(Top.height, i % 2 + 1, "The top height is incorrect.");
                m_log.CHECK_EQ(Top.width, i % 2 + 1, "The top width is incorrect.");

                // Check label.
                double[] rgLabelData = convert(m_blobTopLabel.mutable_cpu_data);

                if (bUseRichAnnotation)
                {
                    if (type == SimpleDatum.ANNOTATION_TYPE.BBOX)
                    {
                        if (i <= 1)
                        {
                            m_log.CHECK_EQ(m_blobTopLabel.num, 1, "The top label num should = 1.");
                            m_log.CHECK_EQ(m_blobTopLabel.channels, 1, "The top label channels should = 1.");
                            m_log.CHECK_EQ(m_blobTopLabel.height, 1, "The top label height should = 1.");
                            m_log.CHECK_EQ(m_blobTopLabel.width, 8, "The top label width should = 8.");

                            for (int k = 0; k < 8; k++)
                            {
                                m_log.EXPECT_NEAR_FLOAT(rgLabelData[k], -1, m_dfEps);
                            }
                        }
                        else
                        {
                            int nCurBox = 0;
                            m_log.CHECK_EQ(m_blobTopLabel.num, 1, "The top label num should = 1.");
                            m_log.CHECK_EQ(m_blobTopLabel.channels, 1, "The top label channels should = 1.");
                            m_log.CHECK_EQ(m_blobTopLabel.height, OneBBoxNum(i), "The top label height shoudl = " + OneBBoxNum(i).ToString() + ".");
                            m_log.CHECK_EQ(m_blobTopLabel.width, 8, "The top label width should = 8.");

                            for (int g = 0; g < i; g++)
                            {
                                for (int a = 0; a < g; a++)
                                {
                                    m_log.CHECK_EQ(0, rgLabelData[nCurBox * 8 + 0], "The label data is incorrect.");
                                    m_log.CHECK_EQ(g, rgLabelData[nCurBox * 8 + 1], "The label data is incorrect.");
                                    m_log.CHECK_EQ(a, rgLabelData[nCurBox * 8 + 1], "The label data is incorrect.");

                                    int b = (bUniqueAnnotation) ? a : g;
                                    for (int p1 = 3; p1 < 5; p1++)
                                    {
                                        m_log.EXPECT_NEAR_FLOAT(b * 0.1f, rgLabelData[nCurBox * 8 + p1], m_dfEps);
                                    }

                                    for (int p1 = 5; p1 < 7; p1++)
                                    {
                                        m_log.EXPECT_NEAR_FLOAT(Math.Min(b * 0.1f + 0.2f, 1.0f), rgLabelData[nCurBox * 8 + p1], m_dfEps);
                                    }

                                    m_log.CHECK_EQ(a % 2, rgLabelData[nCurBox * 8 + 7], "The label data is incorrect.");
                                    nCurBox++;
                                }
                            }
                        }
                    }
                    else
                    {
                        m_log.FAIL("Unknown annotation type.");
                    }
                }
                else
                {
                    m_log.CHECK_EQ(rgLabelData[0], i, "The label is incorrect.");
                }

                // Check the data.
                int nChannels = Bottom.channels;
                int nHeight = Bottom.height;
                int nWidth = Bottom.width;
                double[] rgTopData = convert(Top.mutable_cpu_data);

                for (int c = 0; c < nChannels; c++)
                {
                    for (int h = 0; h < nHeight; h++)
                    {
                        for (int w = 0; w < nWidth; w++)
                        {
                            int nIdx = (c * nHeight + h) * nWidth + w;
                            m_log.CHECK_EQ(nIdx, (int)rgTopData[nIdx], "debug: iter " + i.ToString() + " c " + c.ToString() + " h " + h.ToString() + " w " + w.ToString());
                        }
                    }
                }
            }

            layer.Dispose();
        }

        public void TestReadCrop(Phase phase)
        {
            double dfScale = 3;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ANNOTATED_DATA);
            p.phase = phase;
            p.data_param.batch_size = (uint)m_nNum;
            p.data_param.source = (phase == Phase.TRAIN) ? m_strSrc1 : m_strSrc2;
            p.data_param.backend = DataParameter.DB.IMAGEDB;

            p.transform_param.scale = dfScale;
            p.transform_param.crop_size = 1;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, m_nNum, "The top num is incorrect.");
            m_log.CHECK_EQ(Top.channels, m_nChannels, "The top channels is incorrect.");
            m_log.CHECK_EQ(Top.height, 1, "The top height should = 1.");
            m_log.CHECK_EQ(Top.width, 1, "The top width should = 1.");
            m_log.CHECK_EQ(m_blobTopLabel.num, m_nNum, "The top label num is incorrect.");
            m_log.CHECK_EQ(m_blobTopLabel.channels, 1, "The top label channels should = 1.");
            m_log.CHECK_EQ(m_blobTopLabel.height, 1, "The top label height should = 1.");
            m_log.CHECK_EQ(m_blobTopLabel.width, 1, "The top label width should = 1.");

            for (int iter = 0; iter < 5; iter++)
            {
                layer.Forward(BottomVec, TopVec);

                double[] rgTopLabel = convert(m_blobTopLabel.mutable_cpu_data);
                double[] rgTopData = convert(Top.mutable_cpu_data);

                for (int i = 0; i < m_nNum; i++)
                {
                    m_log.CHECK_EQ(i, rgTopLabel[i], "The top label is incorrect.");
                }

                int nNumWithCenterValue = 0;
                for (int i = 0; i < m_nNum; i++)
                {
                    for (int j = 0; j < m_nChannels; j++)
                    {
                        double dfCenterValue = dfScale * ((Math.Ceiling(m_nHeight / 2.0) - 1) * m_nWidth +
                                                          (Math.Ceiling(m_nWidth / 2.0) - 1) + j * m_nSpatialDim);
                        nNumWithCenterValue += (dfCenterValue == rgTopData[i * 2 + j]) ? 1 : 0;

                        // At TEST time, check that we always get center values.
                        if (phase == Phase.TEST)
                        {
                            m_log.CHECK_EQ(dfCenterValue, rgTopData[i * m_nChannels + j], "debug : iter " + iter.ToString() + " i " + i.ToString() + " j " + j.ToString());
                        }
                    }
                }

                // At TRAIN time, check that we did not get the center crop all 10 times.
                // (This check fails with probability 1-1/12^10 in a correct
                // implementation.
                if (phase == Phase.TRAIN)
                    m_log.CHECK_LT(nNumWithCenterValue, 10, "The center count is too large!");
            }

            layer.Dispose();
        }

        public void TestReadCropTrainSequenceSeeded()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ANNOTATED_DATA);
            p.phase = Phase.TRAIN;
            p.data_param.batch_size = (uint)m_nNum;
            p.data_param.source = m_strSrc1;
            p.data_param.backend = DataParameter.DB.IMAGEDB;

            p.transform_param.crop_size = 1;
            p.transform_param.mirror = true;
            
            // Get crop sequence.
            List<List<double>> rgrgCropSequence = new List<List<double>>();
            {
                Layer<T> layer1 = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db);
                layer1.Setup(BottomVec, TopVec);

                for (int iter = 0; iter < 2; iter++)
                {
                    layer1.Forward(BottomVec, TopVec);

                    double[] rgTopLabel = convert(m_blobTopLabel.mutable_cpu_data);
                    double[] rgTopData = convert(Top.mutable_cpu_data);

                    for (int i = 0; i < m_nNum; i++)
                    {
                        m_log.CHECK_EQ(i, rgTopLabel[i], "The top label is incorrect.");
                    }

                    List<double> rgIterCropSequence = new List<double>();

                    for (int i = 0; i < m_nNum; i++)
                    {
                        for (int j = 0; j < m_nChannels; j++)
                        {
                            double dfData = rgTopData[i * m_nChannels + j];
                            rgIterCropSequence.Add(dfData);
                        }
                    }

                    rgrgCropSequence.Add(rgIterCropSequence);
                }

                layer1.Dispose();
            }

            // Get crop sequence after reseeding.
            m_cuda.SetRandomSeed(m_lSeed);
            Layer<T> layer2 = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db);
            layer2.Setup(BottomVec, TopVec);

            for (int iter = 0; iter < 2; iter++)
            {
                layer2.Forward(BottomVec, TopVec);

                double[] rgTopLabel = convert(m_blobTopLabel.mutable_cpu_data);
                double[] rgTopData = convert(Top.mutable_cpu_data);

                for (int i = 0; i < m_nNum; i++)
                {
                    m_log.CHECK_EQ(i, rgTopLabel[i], "The top label is incorrect.");
                }

                for (int i = 0; i < m_nNum; i++)
                {
                    for (int j = 0; j < m_nChannels; j++)
                    {
                        double dfCrop = rgrgCropSequence[iter][i * m_nChannels + j];
                        double dfVal = rgTopData[i * m_nChannels + j];
                        m_log.CHECK_EQ(dfCrop, dfVal, "debug: iter " + iter.ToString() + " i " + i.ToString() + " j " + j.ToString());
                    }
                }
            }

            layer2.Dispose();
        }

        public void TestReadCropTrainSequenceUnseeded()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.ANNOTATED_DATA);
            p.phase = Phase.TRAIN;
            p.data_param.batch_size = (uint)m_nNum;
            p.data_param.source = m_strSrc1;
            p.data_param.backend = DataParameter.DB.IMAGEDB;

            p.transform_param.crop_size = 1;
            p.transform_param.mirror = true;

            // Get crop sequence.
            List<List<double>> rgrgCropSequence = new List<List<double>>();
            {
                Layer<T> layer1 = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db);
                layer1.Setup(BottomVec, TopVec);

                for (int iter = 0; iter < 2; iter++)
                {
                    layer1.Forward(BottomVec, TopVec);

                    double[] rgTopLabel = convert(m_blobTopLabel.mutable_cpu_data);
                    double[] rgTopData = convert(Top.mutable_cpu_data);

                    for (int i = 0; i < m_nNum; i++)
                    {
                        m_log.CHECK_EQ(i, rgTopLabel[i], "The top label is incorrect.");
                    }

                    List<double> rgIterCropSequence = new List<double>();

                    for (int i = 0; i < m_nNum; i++)
                    {
                        for (int j = 0; j < m_nChannels; j++)
                        {
                            double dfData = rgTopData[i * m_nChannels + j];
                            rgIterCropSequence.Add(dfData);
                        }
                    }

                    rgrgCropSequence.Add(rgIterCropSequence);
                }

                layer1.Dispose();
            }

            // Get crop sequence continuing from prevous RNG state; 
            // Check that the sequence differs from the original.
            Layer<T> layer2 = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db);
            layer2.Setup(BottomVec, TopVec);

            for (int iter = 0; iter < 2; iter++)
            {
                layer2.Forward(BottomVec, TopVec);

                double[] rgTopLabel = convert(m_blobTopLabel.mutable_cpu_data);
                double[] rgTopData = convert(Top.mutable_cpu_data);

                for (int i = 0; i < m_nNum; i++)
                {
                    m_log.CHECK_EQ(i, rgTopLabel[i], "The top label is incorrect.");
                }

                int nNumSequenceMatches = 0;

                for (int i = 0; i < m_nNum; i++)
                {
                    for (int j = 0; j < m_nChannels; j++)
                    {
                        double dfCrop = rgrgCropSequence[iter][i * m_nChannels + j];
                        double dfVal = rgTopData[i * m_nChannels + j];
                        nNumSequenceMatches = (dfCrop == dfVal) ? 1 : 0;
                    }
                }

                m_log.CHECK_LT(nNumSequenceMatches, m_nNum * m_nChannels, "The sequence matches is too high!");
            }

            layer2.Dispose();
        }

        public void TestReadDb()
        {
            SimpleDatum.ANNOTATION_TYPE type = SimpleDatum.ANNOTATION_TYPE.BBOX;

            for (int p = 0; p < m_kNumChoices; p++)
            {
                bool bUniquePixel = m_rgkBoolChoices[p];

                for (int r = 0; r < m_kNumChoices; r++)
                {
                    bool bUseRichAnnotation = m_rgkBoolChoices[r];

                    for (int a = 0; a < m_kNumChoices; a++)
                    {
                        if (!bUseRichAnnotation)
                            continue;

                        bool bUniqueAnnotation = m_rgkBoolChoices[a];
                        Fill(DataParameter.DB.IMAGEDB, bUniquePixel, bUniqueAnnotation, bUseRichAnnotation, type);
                        TestRead();
                    }
                }
            }
        }

        public void TestReshapeDb()
        {
            SimpleDatum.ANNOTATION_TYPE type = SimpleDatum.ANNOTATION_TYPE.BBOX;

            for (int p = 0; p < m_kNumChoices; p++)
            {
                bool bUniquePixel = m_rgkBoolChoices[p];

                for (int r = 0; r < m_kNumChoices; r++)
                {
                    bool bUseRichAnnotation = m_rgkBoolChoices[r];

                    for (int a = 0; a < m_kNumChoices; a++)
                    {
                        if (!bUseRichAnnotation)
                            continue;

                        bool bUniqueAnnotation = m_rgkBoolChoices[a];
                        TestReshape(bUniquePixel, bUniqueAnnotation, bUseRichAnnotation, type);
                    }
                }
            }
        }

        public void TestReadCropDb()
        {
            bool bUniquePixel = true; // all pixels the same; images different.
            bool bUniqueAnnotation = false; // all anno the same; groups different.
            bool bUseRichAnnotation = false;
            SimpleDatum.ANNOTATION_TYPE type = SimpleDatum.ANNOTATION_TYPE.BBOX;

            Fill(DataParameter.DB.IMAGEDB, bUniquePixel, bUniqueAnnotation, bUseRichAnnotation, type);
            TestReadCrop(Phase.TRAIN);
        }

        public void TestReadCropTrainSequenceSeededDb()
        {
            bool bUniquePixel = true; // all pixels the same; images different.
            bool bUniqueAnnotation = false; // all anno the same; groups different.
            bool bUseRichAnnotation = false;
            SimpleDatum.ANNOTATION_TYPE type = SimpleDatum.ANNOTATION_TYPE.BBOX;

            Fill(DataParameter.DB.IMAGEDB, bUniquePixel, bUniqueAnnotation, bUseRichAnnotation, type);
            TestReadCropTrainSequenceSeeded();
        }

        public void TestReadCropTrainSequenceUnseededDb()
        {
            bool bUniquePixel = true; // all pixels the same; images different.
            bool bUniqueAnnotation = false; // all anno the same; groups different.
            bool bUseRichAnnotation = false;
            SimpleDatum.ANNOTATION_TYPE type = SimpleDatum.ANNOTATION_TYPE.BBOX;

            Fill(DataParameter.DB.IMAGEDB, bUniquePixel, bUniqueAnnotation, bUseRichAnnotation, type);
            TestReadCropTrainSequenceUnseeded();
        }

        public void TestReadCropTestDb()
        {
            bool bUniquePixel = true; // all pixels the same; images different.
            bool bUniqueAnnotation = false; // all anno the same; groups different.
            bool bUseRichAnnotation = false;
            SimpleDatum.ANNOTATION_TYPE type = SimpleDatum.ANNOTATION_TYPE.BBOX;

            Fill(DataParameter.DB.IMAGEDB, bUniquePixel, bUniqueAnnotation, bUseRichAnnotation, type);
            TestReadCrop(Phase.TEST);
        }
    }
}
