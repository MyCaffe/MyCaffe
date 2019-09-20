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

        public AnnotatedDataLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine, AnnotatedDataLayerTest parent)
            : base(strName, new List<int>() { 6, 2, 10, 10 }, nDeviceID)
        {
            m_parent = parent;
            m_engine = engine;
            Setup();
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void Setup()
        {
            m_nSpatialDim = m_nHeight * m_nWidth;
            m_nSize = m_nChannels * m_nSpatialDim;
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

        public void TestRead()
        {
        }

        public void TestReshape()
        {
        }

        public void TestReadCrop()
        {
        }

        public void TestReadCropTrainSequenceSeeded()
        {
        }

        public void TestReadCropTrainSequenceUnseeded()
        {
        }

        public void TestReadDb()
        {
        }

        public void TestReshapeDb()
        {
        }

        public void TestReadCropDb()
        {
        }

        public void TestReadCropTrainSequenceSeededDb()
        {
        }

        public void TestReadCropTrainSequenceUnseededDb()
        {
        }

        public void TestReadCropTestDb()
        {
        }
    }
}
