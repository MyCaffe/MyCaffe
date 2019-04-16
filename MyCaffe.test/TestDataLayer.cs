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

namespace MyCaffe.test
{
    [TestClass]
    public class TestDataLayer
    {
        [TestMethod]
        public void TestInitialize()
        {
            DataLayerTest test = new DataLayerTest("MNIST");
                     
            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    t.TestInitialization(test.SourceName);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetup()
        {
            DataLayerTest test = new DataLayerTest("MNIST");

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    t.TestSetup(test.SourceName);
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
            DataLayerTest test = new DataLayerTest("MNIST");

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    t.TestForward(test.SourceName);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward2()
        {
            DataLayerTest test = new DataLayerTest("MNIST");

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    t.TestForward2(test.SourceName);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadLoadAll()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    bool unique_pixels = false;
                    t.Fill(unique_pixels);
                    t.TestRead(IMAGEDB_LOAD_METHOD.LOAD_ALL);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadLoadOnDemand()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    bool unique_pixels = false;
                    t.Fill(unique_pixels);
                    t.TestRead(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSkipLoadAll()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    t.Fill(false);
                    t.TestSkip(IMAGEDB_LOAD_METHOD.LOAD_ALL);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSkipLoadOnDemand()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    t.Fill(false);
                    t.TestSkip(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReshapeLoadAll()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    t.TestReshape(IMAGEDB_LOAD_METHOD.LOAD_ALL);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReshapeLoadOnDemand()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    t.TestReshape(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadCropTrainLoadAll()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    bool unique_pixels = true;

                    for (int i = 0; i < 2; i++)
                    {
                        t.Fill(unique_pixels);
                        t.TestReadCrop(Phase.TRAIN, IMAGEDB_LOAD_METHOD.LOAD_ALL);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadCropTrainLoadOnDemand()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    bool unique_pixels = true;

                    for (int i = 0; i < 2; i++)
                    {
                        t.Fill(unique_pixels);
                        t.TestReadCrop(Phase.TRAIN, IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadCropTrainSequenceSeededLoadAll()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    bool unique_pixels = true;
                    t.Fill(unique_pixels);
                    t.TestReadCropSequenceSeeded(IMAGEDB_LOAD_METHOD.LOAD_ALL);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadCropTrainSequenceSeededLoadOnDemand()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    bool unique_pixels = true;
                    t.Fill(unique_pixels);
                    t.TestReadCropSequenceSeeded(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadCropTrainSequenceUnseededLoadAll()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    bool unique_pixels = true;
                    t.Fill(unique_pixels);
                    t.TestReadCropSequenceSeeded(IMAGEDB_LOAD_METHOD.LOAD_ALL);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadCropTrainSequenceUnseededLoadOnDemand()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    bool unique_pixels = true;
                    t.Fill(unique_pixels);
                    t.TestReadCropSequenceSeeded(IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadCropTestLoadAll()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    bool unique_pixels = true;

                    for (int i = 0; i < 2; i++)
                    {
                        t.Fill(unique_pixels);
                        t.TestReadCrop(Phase.TEST, IMAGEDB_LOAD_METHOD.LOAD_ALL);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestReadCropTestLoadOnDemand()
        {
            DataLayerTest test = new DataLayerTest();

            try
            {
                foreach (IDataLayerTest t in test.Tests)
                {
                    bool unique_pixels = true;

                    for (int i = 0; i < 2; i++)
                    {
                        t.Fill(unique_pixels);
                        t.TestReadCrop(Phase.TEST, IMAGEDB_LOAD_METHOD.LOAD_ON_DEMAND);
                    }
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    class DataLayerTest : TestBase
    {
        SettingsCaffe m_settings;
        MyCaffeImageDatabase m_db;
        CancelEvent m_evtCancel = new CancelEvent();

        public DataLayerTest(string strDs = null)
            : base("Data Layer Test")
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

        protected override ITest create(DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            string strPath = TestBase.CudaPath;

            if (dt == DataType.DOUBLE)
            {
                CudaDnn<double>.SetDefaultCudaPath(strPath);
                return new DataLayerTest<double>(strName, nDeviceID, this);
            }
            else
            {
                CudaDnn<double>.SetDefaultCudaPath(strPath);
                return new DataLayerTest<float>(strName, nDeviceID, this);
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

    interface IDataLayerTest 
    {
        DataType Type { get; }
        void TestInitialization(string strSrc);
        void TestSetup(string strSrc);
        void TestForward(string strSrc);
        void TestForward2(string strSrc);
        string Fill(bool unique_pixels);
        string Fill2(int num_inputs);
        void TestRead(IMAGEDB_LOAD_METHOD loadMethod);
        void TestSkip(IMAGEDB_LOAD_METHOD loadMethod);
        void TestReshape(IMAGEDB_LOAD_METHOD loadMethod);
        void TestReadCrop(Phase phase, IMAGEDB_LOAD_METHOD loadMethod);
        void TestReadCropSequenceSeeded(IMAGEDB_LOAD_METHOD loadMethod);
        void TestReadCropSequenceUnSeeded(IMAGEDB_LOAD_METHOD loadMethod);
    }

    class DataLayerTest<T> : TestEx<T>, IDataLayerTest
    {
        Blob<T> m_blob_top_label;
        DataLayerTest m_parent;
        int m_nSrcID1 = 0;
        int m_nSrcID2 = 0;
        string m_strSrc1 = "test_data";
        string m_strSrc2 = "test_data.t";
        int m_nDsID = 0;

        public DataLayerTest(string strName, int nDeviceID, DataLayerTest parent, List<int> rgBottomShape = null)
            : base(strName, rgBottomShape, nDeviceID)
        {
            m_parent = parent;
            m_blob_top_label = new Blob<T>(m_cuda, m_log);

            TopVec.Add(m_blob_top_label);
            BottomVec.Clear();
        }

        protected override void dispose()
        {
            m_blob_top_label.Dispose();
            base.dispose();
        }

        public DataType Type
        {
            get { return m_dt; }
        }

        public Blob<T> TopLabel
        {
            get { return m_blob_top_label; }
        }

        public void TestInitialization(string strSrc)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA);

            m_log.CHECK(p.data_param != null, "The data_param is null!");
            m_log.CHECK(p.transform_param != null, "The transform_para is null!");

            p.data_param.batch_size = 1;
            p.data_param.source = strSrc;

            m_log.CHECK(p.data_param.enable_pair_selection == null, "Pair selection should be off by default, letting the image database decide.");
            m_log.CHECK(p.data_param.enable_random_selection == null, "Random selection should be off by default, letting the image database decide.");
            m_log.CHECK_GT(p.data_param.prefetch, 0, "Pre fetch should be greater than zero.");

            DataLayer<T> layer = new DataLayer<T>(m_cuda, m_log, p, m_parent.db, m_parent.CancelEvent);

            layer.Dispose();
        }

        public void TestSetup(string strSrc)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA);

            m_log.CHECK(p.data_param != null, "The data_param is null!");
            m_log.CHECK(p.transform_param != null, "The transform_para is null!");

            p.data_param.batch_size = 1;
            p.data_param.source = strSrc;
            p.data_param.enable_random_selection = false;
            p.data_param.enable_pair_selection = false;

            DataLayer<T> layer = new DataLayer<T>(m_cuda, m_log, p, m_parent.db, m_parent.CancelEvent);

            layer.LayerSetUp(BottomVec, TopVec);
            layer.Reshape(BottomVec, TopVec);

            Thread.Sleep(2000);

            layer.Dispose();

            m_parent.CancelEvent.Reset();
        }

        public string Fill(bool unique_pixels)
        {
            DatasetFactory factory = new DatasetFactory();

            m_log.WriteLine("Creating temporary dataset '" + m_strSrc1 + "'.");
            SourceDescriptor srcTrain = new SourceDescriptor(0, m_strSrc1, 2, 4, 3, false, true);
            srcTrain.ID = factory.AddSource(srcTrain);
            m_nSrcID1 = srcTrain.ID;
            SourceDescriptor srcTest = new SourceDescriptor(0, m_strSrc2, 2, 4, 3, false, true);
            srcTest.ID = factory.AddSource(srcTest);
            m_nSrcID2 = srcTest.ID;

            List<SourceDescriptor> rgSrcId = new List<SourceDescriptor>() { srcTrain, srcTest };

            for (int k = 0; k < 2; k++)
            {
                factory.Open(rgSrcId[k]);
                factory.DeleteSourceData();

                int nCount = factory.GetImageCount();
                for (int i = nCount; i < nCount + 5; i++)
                {
                    List<byte> rgData = new List<byte>();

                    for (int j = 0; j < 24; j++)
                    {
                        int datum = unique_pixels ? j : i;
                        rgData.Add((byte)datum);
                    }

                    SimpleDatum sd = new SimpleDatum(false, 2, 4, 3, i, DateTime.Today, rgData, null, 0, false, i);
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

        public string Fill2(int num_inputs)
        {
            DatasetFactory factory = new DatasetFactory();

            string strName = m_strSrc1 + ".x";
            m_log.WriteLine("Creating temporary dataset '" + m_strSrc1 + "'.");
            SourceDescriptor srcTrain = new SourceDescriptor(0, strName, 2, 4, 3, false, true);
            m_nSrcID1 = factory.AddSource(srcTrain);
            SourceDescriptor srcTest = new SourceDescriptor(0, strName + ".t", 2, 4, 3, false, true);
            m_nSrcID2 = factory.AddSource(srcTest);

            List<SourceDescriptor> rgSrcId = new List<SourceDescriptor>() { srcTrain, srcTest };

            for (int k = 0; k < 2; k++)
            {
                factory.Open(rgSrcId[k]);

                int nCount = factory.GetImageCount();
                for (int i = nCount; i < num_inputs; i++)
                {
                    List<byte> rgData = new List<byte>();
                    int nChannels = 2;
                    int nHeight = i % 2 + 1;
                    int nWidth = i % 4 + 1;
                    int nDataSize = nChannels * nHeight * nWidth;

                    for (int j = 0; j < nDataSize; j++)
                    {
                        rgData.Add((byte)j);
                    }

                    SimpleDatum sd = new SimpleDatum(false, nChannels, nWidth, nHeight, i, DateTime.Today, rgData, null, 0, false, i);
                    factory.PutRawImage(i, sd);
                }

                factory.Close();
            }

            DatasetDescriptor ds = new DatasetDescriptor(0, strName, null, null, srcTrain, srcTest, null, null);
            ds.ID = factory.AddDataset(ds);

            factory.UpdateDatasetCounts(ds.ID);
            m_nDsID = ds.ID;

            return strName;
        }

        public void TestRead(IMAGEDB_LOAD_METHOD loadMethod)
        {
            Assert.AreNotEqual(0, m_nSrcID1, "You must call 'Fill' first to set the source id!");
            double dfScale = 3;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA);
            p.phase = Phase.TRAIN;
            p.data_param.batch_size = 5;
            p.data_param.source = m_strSrc1;
            p.data_param.enable_random_selection = false;
            p.data_param.backend = DataParameter.DB.IMAGEDB;

            p.transform_param.scale = dfScale;

            MyCaffeImageDatabase m_imgDb = new MyCaffeImageDatabase();

            m_parent.Settings.ImageDbLoadMethod = loadMethod;
            m_imgDb.InitializeWithDsId(m_parent.Settings, m_nDsID);
            CancelEvent evtCancel = new CancelEvent();

            DataLayer<T> layer = new DataLayer<T>(m_cuda, m_log, p, m_imgDb, evtCancel);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, 5, "The top should have num = 5");
            m_log.CHECK_EQ(Top.channels, 2, "The top should have channels = 2");
            m_log.CHECK_EQ(Top.height, 3, "The top should have height = 3");
            m_log.CHECK_EQ(Top.width, 4, "The top should have width = 4");
            m_log.CHECK_EQ(TopLabel.num, 5, "The top label should have num = 5");
            m_log.CHECK_EQ(TopLabel.channels, 1, "The top label should have channels = 1");
            m_log.CHECK_EQ(TopLabel.height, 1, "The top label should have height = 1");
            m_log.CHECK_EQ(TopLabel.width, 1, "The top label should have width = 1");

            for (int iter = 0; iter < 100; iter++)
            {
                layer.Forward(BottomVec, TopVec);

                for (int i = 0; i < 5; i++)
                {
                    double dfTopLabel = convert(TopLabel.GetData(i));
                    m_log.CHECK_EQ(i, dfTopLabel, "The top label value at " + i.ToString() + " is not correct.");
                }

                for (int i = 0; i < 5; i++)
                {
                    for (int j = 0; j < 24; j++)
                    {
                        double dfValue = dfScale * i;
                        double dfTop = convert(Top.GetData(i * 24 + j));

                        m_log.CHECK_EQ(dfValue, dfTop, "debug : iter " + iter.ToString() + " i " + i.ToString() + " j " + j.ToString());
                    }
                }
            }

            layer.Dispose();
            m_parent.CancelEvent.Reset();
        }

        public void TestSkip(IMAGEDB_LOAD_METHOD loadMethod)
        {
            int nBatchSize = 5;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA);
            p.phase = Phase.TRAIN;
            p.data_param.batch_size = (uint)nBatchSize;
            p.data_param.source = m_strSrc1;
            p.data_param.enable_random_selection = false;
            p.data_param.backend = DataParameter.DB.IMAGEDB;

            MyCaffeImageDatabase m_imgDb = new MyCaffeImageDatabase();
            m_parent.Settings.ImageDbLoadMethod = loadMethod;
            m_imgDb.InitializeWithDsId(m_parent.Settings, m_nDsID);
            CancelEvent evtCancel = new CancelEvent();

            int nSolverCount = 8;
            p.solver_count = nSolverCount;

            for (int dev = 0; dev < nSolverCount; dev++)
            {
                int nSolverRank = dev;
                p.solver_rank = nSolverRank;

                DataLayer<T> layer = new DataLayer<T>(m_cuda, m_log, p, m_imgDb, evtCancel);
                layer.Setup(BottomVec, TopVec);

                int nLabel = dev;

                for (int iter = 0; iter < 10; iter++)
                {
                    layer.Forward(BottomVec, TopVec);

                    double[] rgTopLabel = convert(TopLabel.update_cpu_data());

                    for (int i=0; i<nBatchSize; i++)
                    {
                        m_log.CHECK_EQ(nLabel % nBatchSize, (int)rgTopLabel[i], "The label is not as expected at " + i.ToString());
                        nLabel += nSolverCount;
                    }
                }

                layer.Dispose();
                m_parent.CancelEvent.Reset();
            }
        }

        public void TestReshape(IMAGEDB_LOAD_METHOD loadMethod)
        {
            int num_inputs = 5;
            // Save data of varying shapes

            string strName = Fill2(num_inputs);
            m_log.WriteLine("Using temporary dataset '" + strName + "'.");

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA);
            p.phase = Phase.TEST;
            p.data_param.batch_size = 1;
            p.data_param.source = strName;
            p.data_param.enable_random_selection = false;
            p.data_param.backend = DataParameter.DB.IMAGEDB;

            MyCaffeImageDatabase m_imgDb = new MyCaffeImageDatabase();
            m_parent.Settings.ImageDbLoadMethod = loadMethod;
            m_imgDb.InitializeWithDsId(m_parent.Settings, m_nDsID);
            CancelEvent evtCancel = new CancelEvent();

            DataLayer<T> layer = new DataLayer<T>(m_cuda, m_log, p, m_imgDb, evtCancel);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, 1, "The top should have num = 1");
            m_log.CHECK_EQ(Top.channels, 2, "The top should have channels = 2");
            m_log.CHECK_EQ(TopLabel.num, 1, "The top label should have num = 1");
            m_log.CHECK_EQ(TopLabel.channels, 1, "The top label should have channels = 1");
            m_log.CHECK_EQ(TopLabel.height, 1, "The top label should have height = 1");
            m_log.CHECK_EQ(TopLabel.width, 1, "The top label should have width = 1");

            for (int iter = 0; iter < num_inputs; iter++)
            {
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(Top.height, iter % 2 + 1, "The top height is not as expected.");
                m_log.CHECK_EQ(Top.width, iter % 4 + 1, "The top width is not as expected.");
                m_log.CHECK_EQ(iter, convert(TopLabel.GetData(0)), "The top label is not as expected.");

                int nChannels = Top.channels;
                int nHeight = Top.height;
                int nWidth = Top.width;

                for (int c = 0; c < nChannels; c++)
                {
                    for (int h = 0; h < nHeight; h++)
                    {
                        for (int w = 0; w < nWidth; w++)
                        {
                            int nIdx = (c * nHeight + h) * nWidth + w;
                            m_log.CHECK_EQ(nIdx, (int)convert(Top.GetData(nIdx)), "dbug: iter " + iter.ToString() + " c " + c.ToString() + " h " + h.ToString() + " w " + w.ToString());
                        }
                    }
                }
            }

            layer.Dispose();
            m_parent.CancelEvent.Reset();
        }

        public void TestReadCrop(Phase phase, IMAGEDB_LOAD_METHOD loadMethod)
        {
            Assert.AreNotEqual(0, m_nSrcID1, "You must call 'Fill' first to set the source id!");
            m_log.WriteLine("Using temporary dataset '" + m_strSrc1 + "'.");
            double dfScale = 3;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA);
            p.phase = phase;
            p.data_param.batch_size = 5;
            p.data_param.source = m_strSrc1;
            p.data_param.enable_random_selection = false;
            p.data_param.backend = DataParameter.DB.IMAGEDB;

            p.transform_param.scale = dfScale;
            p.transform_param.crop_size = 1;
            p.transform_param.random_seed = 1701;

            MyCaffeImageDatabase m_imgDb = new MyCaffeImageDatabase();
            m_parent.Settings.ImageDbLoadMethod = loadMethod;
            m_imgDb.InitializeWithDsId(m_parent.Settings, m_nDsID);
            CancelEvent evtCancel = new CancelEvent();

            DataLayer<T> layer = new DataLayer<T>(m_cuda, m_log, p, m_imgDb, evtCancel);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, 5, "The top should have num = 5");
            m_log.CHECK_EQ(Top.channels, 2, "The top should have channels = 2");
            m_log.CHECK_EQ(Top.height, 1, "The top should have channels = 1");
            m_log.CHECK_EQ(Top.width, 1, "The top should have channels = 1");
            m_log.CHECK_EQ(TopLabel.num, 5, "The top label should have num = 5");
            m_log.CHECK_EQ(TopLabel.channels, 1, "The top label should have channels = 1");
            m_log.CHECK_EQ(TopLabel.height, 1, "The top label should have channels = 1");
            m_log.CHECK_EQ(TopLabel.width, 1, "The top label should have channels = 1");

            for (int iter = 0; iter < 2; iter++)
            {
                layer.Forward(BottomVec, TopVec);

                for (int i = 0; i < 5; i++)
                {
                    m_log.CHECK_EQ(i, convert(TopLabel.GetData(i)), "The top label value at " + i.ToString() + " is not correct.");
                }

                int num_with_center_value = 0;

                for (int i = 0; i < 5; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        double dfCenterValue = dfScale * (j != 0 ? 17 : 5);
                        double dfTop = convert(Top.GetData(i * 2 + j));

                        if (dfCenterValue == dfTop)
                            num_with_center_value++;

                        // At TEST time, check that we always get center value.
                        if (phase == Phase.TEST)
                            m_log.CHECK_EQ(dfCenterValue, dfTop, "The center value '" + dfCenterValue.ToString() + "' should equal the top value '" + dfTop.ToString() + "', debug : iter " + iter.ToString() + " i " + i.ToString() + " j " + j.ToString());
                    }
                }

                // At TRAIN time, check that we did not get the center crop all 10 times.
                // (This check fails with probability 1-1/12^10 in a correct
                // implementation, so we set the random_seed param value).
                if (phase == Phase.TRAIN)
                    m_log.CHECK_LT(num_with_center_value, 10, "The num_with_center_value should be less than 10");
            }

            layer.Dispose();
            m_parent.CancelEvent.Reset();
        }

        public void TestReadCropSequenceSeeded(IMAGEDB_LOAD_METHOD loadMethod)
        {
            Assert.AreNotEqual(0, m_nSrcID1, "You must call 'Fill' first to set the source id!");
            m_log.WriteLine("Using temporary dataset '" + m_strSrc1 + "'.");
            double dfScale = 3;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA);
            p.phase = Phase.TRAIN;
            p.data_param.batch_size = 5;
            p.data_param.source = m_strSrc1;
            p.data_param.enable_random_selection = false;
            p.data_param.backend = DataParameter.DB.IMAGEDB;

            p.transform_param.scale = dfScale;
            p.transform_param.crop_size = 1;
            p.transform_param.mirror = true;
            p.transform_param.random_seed = 1701;

            MyCaffeImageDatabase imgDb = new MyCaffeImageDatabase();
            m_parent.Settings.ImageDbLoadMethod = loadMethod;
            imgDb.InitializeWithDsId(m_parent.Settings, m_nDsID);
            CancelEvent evtCancel = new CancelEvent();

            // Get crop sequence with seed 1701.
            List<List<double>> crop_sequence = new List<List<double>>();
            {
                DataLayer<T> layer1 = new DataLayer<T>(m_cuda, m_log, p, imgDb, evtCancel);
                layer1.Setup(BottomVec, TopVec);

                for (int iter = 0; iter < 2; iter++)
                {
                    layer1.Forward(BottomVec, TopVec);

                    for (int i = 0; i < 5; i++)
                    {
                        m_log.CHECK_EQ(i, convert(TopLabel.GetData(i)), "The top label value at " + i.ToString() + " is not correct.");
                    }

                    List<double> iter_crop_sequence = new List<double>();

                    for (int i = 0; i < 5; i++)
                    {
                        for (int j = 0; j < 2; j++)
                        {
                            double dfTop = convert(Top.GetData(i * 2 + j));
                            iter_crop_sequence.Add(dfTop);
                        }
                    }

                    crop_sequence.Add(iter_crop_sequence);
                }

                layer1.Dispose();
            } // destroy 1st data layer and unlock the db.


            imgDb = new MyCaffeImageDatabase();
            imgDb.InitializeWithDsId(m_parent.Settings, m_nDsID);

            // Get crop sequence after reseeding caffe (done within the data transformer)
            p.transform_param.random_seed = 1701;
            DataLayer<T> layer2 = new DataLayer<T>(m_cuda, m_log, p, imgDb, evtCancel);

            layer2.Setup(BottomVec, TopVec);

            for (int iter = 0; iter < 2; iter++)
            {
                layer2.Forward(BottomVec, TopVec);

                for (int i = 0; i < 5; i++)
                {
                    m_log.CHECK_EQ(i, convert(TopLabel.GetData(i)), "The top label value at " + i.ToString() + " is not correct.");
                }

                for (int i = 0; i < 5; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        double dfValue = crop_sequence[iter][i * 2 + j];
                        double dfTop = convert(Top.GetData(i * 2 + j));
                        m_log.CHECK_EQ(dfValue, dfTop, "debug: iter " + iter.ToString() + " i " + i.ToString() + " j " + j.ToString());
                    }
                }
            }

            layer2.Dispose();
            m_parent.CancelEvent.Reset();
        }

        public void TestReadCropSequenceUnSeeded(IMAGEDB_LOAD_METHOD loadMethod)
        {
            Assert.AreNotEqual(0, m_nSrcID1, "You must call 'Fill' first to set the source id!");
            m_log.WriteLine("Using temporary dataset '" + m_strSrc1 + "'.");
            double dfScale = 3;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA);
            p.phase = Phase.TRAIN;
            p.data_param.batch_size = 5;
            p.data_param.source = m_strSrc1;
            p.data_param.enable_random_selection = false;
            p.data_param.backend = DataParameter.DB.IMAGEDB;

            p.transform_param.scale = dfScale;
            p.transform_param.crop_size = 1;
            p.transform_param.mirror = true;

            MyCaffeImageDatabase m_imgDb = new MyCaffeImageDatabase();
            m_parent.Settings.ImageDbLoadMethod = loadMethod;
            m_imgDb.InitializeWithDsId(m_parent.Settings, m_nDsID);
            CancelEvent evtCancel = new CancelEvent();

            // Get crop sequence without setting a seed.
            List<List<double>> crop_sequence = new List<List<double>>();
            {
                DataLayer<T> layer1 = new DataLayer<T>(m_cuda, m_log, p, m_imgDb, evtCancel);
                layer1.Setup(BottomVec, TopVec);

                for (int iter = 0; iter < 2; iter++)
                {
                    layer1.Forward(BottomVec, TopVec);

                    for (int i = 0; i < 5; i++)
                    {
                        m_log.CHECK_EQ(i, convert(TopLabel.GetData(i)), "The top label value at " + i.ToString() + " is not correct.");
                    }

                    List<double> iter_crop_sequence = new List<double>();

                    for (int i = 0; i < 5; i++)
                    {
                        for (int j = 0; j < 2; j++)
                        {
                            double dfTop = convert(Top.GetData(i * 2 + j));
                            iter_crop_sequence.Add(dfTop);
                        }
                    }

                    crop_sequence.Add(iter_crop_sequence);
                }

                layer1.Dispose();
            } // destroy 1st data layer and unlock the db.


            // Get crop sequence without reseeding.  Check that the
            // sequence differs from the original.
            DataLayer<T> layer2 = new DataLayer<T>(m_cuda, m_log, p, m_imgDb, evtCancel);

            layer2.Setup(BottomVec, TopVec);

            for (int iter = 0; iter < 2; iter++)
            {
                layer2.Forward(BottomVec, TopVec);

                for (int i = 0; i < 5; i++)
                {
                    m_log.CHECK_EQ(i, convert(TopLabel.GetData(i)), "The top label value at " + i.ToString() + " is not correct.");
                }

                int num_sequence_matches = 0;

                for (int i = 0; i < 5; i++)
                {
                    for (int j = 0; j < 2; j++)
                    {
                        double dfValue = crop_sequence[iter][i * 2 + j];
                        double dfTop = convert(Top.GetData(i * 2 + j));

                        if (dfValue == dfTop)
                            num_sequence_matches++;
                    }
                }

                m_log.CHECK_LT(num_sequence_matches, 10, "The number of sequence matches doesn't differ when it should.");
            }

            layer2.Dispose();
            m_parent.CancelEvent.Reset();
        }

        public void TestForward(string strSrc)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA);

            m_log.CHECK(p.data_param != null, "The data_param is null!");
            m_log.CHECK(p.transform_param != null, "The transform_para is null!");

            p.data_param.batch_size = 1;
            p.data_param.source = strSrc;
            p.data_param.enable_random_selection = false;
            p.data_param.enable_pair_selection = false;

            DataLayer<T> layer = new DataLayer<T>(m_cuda, m_log, p, m_parent.db, m_parent.CancelEvent);
            int nSrcID = m_parent.db.GetSourceID(strSrc);

            layer.LayerSetUp(BottomVec, TopVec);
            layer.Reshape(BottomVec, TopVec);

            int nCount = 1000;
            Stopwatch sw = new Stopwatch();
            double dfTotalTime = 0;

            for (int i = 0; i < nCount; i++)
            {
                sw.Start();
                layer.Forward(BottomVec, TopVec);
                dfTotalTime += sw.ElapsedMilliseconds;
                sw.Stop();
                sw.Reset();

                if (i == 3)
                    Thread.Sleep(1000);

                m_log.CHECK_EQ(TopVec.Count, 2, "The top vec should have one element.");
                T[] rgData = TopVec[0].update_cpu_data();

                SimpleDatum d = m_parent.db.QueryImage(nSrcID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                byte[] rgData2 = d.ByteData;

                m_log.CHECK_EQ(rgData2.Length, rgData.Length, "The data from the data forward should have the same length as the first item in the database for the source = " + strSrc);

                for (int j = 0; j < rgData.Length; j++)
                {
                    double dfVal1 = (double)Convert.ChangeType(rgData[j], typeof(double));
                    double dfVal2 = (double)Convert.ChangeType(rgData2[j], typeof(double));

                    m_log.CHECK_EQ(dfVal1, dfVal2, "The values at index " + j.ToString() + " for image at index " + i.ToString() + " in source = " + strSrc + " do not match!");
                }
            }

            string str = (dfTotalTime / (double)nCount).ToString() + " ms.";
            Trace.WriteLine("Average DataLayer Forward Time = " + str);

            layer.Dispose();
            m_parent.CancelEvent.Reset();
        }

        public void TestForward2(string strSrc)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA);

            m_log.CHECK(p.data_param != null, "The data_param is null!");
            m_log.CHECK(p.transform_param != null, "The transform_para is null!");

            p.data_param.batch_size = 1;
            p.data_param.source = strSrc;
            p.data_param.enable_random_selection = true;
            p.data_param.enable_pair_selection = false;

            DataLayer<T> layer = new DataLayer<T>(m_cuda, m_log, p, m_parent.db, m_parent.CancelEvent);
            int nSrcID = m_parent.db.GetSourceID(strSrc);

            layer.LayerSetUp(BottomVec, TopVec);
            layer.Reshape(BottomVec, TopVec);

            int nCount = 1000;
            Stopwatch sw = new Stopwatch();
            double dfTotalTime = 0;

            for (int i = 0; i < nCount; i++)
            {
                sw.Start();
                layer.Forward(BottomVec, TopVec);
                dfTotalTime += sw.ElapsedMilliseconds;
                sw.Stop();
                sw.Reset();

                m_log.CHECK_EQ(TopVec.Count, 2, "The top vec should have one element.");
                T[] rgData = TopVec[0].update_cpu_data();

                SimpleDatum d = m_parent.db.QueryImage(nSrcID, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.NONE);
                byte[] rgData2 = d.ByteData;

                m_log.CHECK_EQ(rgData2.Length, rgData.Length, "The data from the data forward should have the same length as the first item in the database for the source = " + strSrc);

                int nMatches = 0;

                for (int j = 0; j < rgData.Length; j++)
                {
                    double dfVal1 = (double)Convert.ChangeType(rgData[j], typeof(double));
                    double dfVal2 = (double)Convert.ChangeType(rgData2[j], typeof(double));

                    if (dfVal1 != 0 || dfVal2 != 0)
                    {
                        if (dfVal1 == dfVal2)
                            nMatches++;
                    }
                }

                m_log.CHECK_LE(nMatches, rgData.Length, "The images at index " + i.ToString() + " in source = " + strSrc + " should not match!");
            }

            string str = (dfTotalTime / (double)nCount).ToString() + " ms.";
            Trace.WriteLine("Average DataLayer Forward Time = " + str);

            layer.Dispose();
            m_parent.CancelEvent.Reset();
        }
    }
}
