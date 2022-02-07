using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe.data;
using System.Diagnostics;
using static MyCaffe.basecode.SimpleDatum;

/// <summary>
/// Testing the MemoryData layer.
/// 
/// MemoryData Layer - this layer facilitates a data layer of values stored within memory.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestMemoryDataLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            MemoryDataLayerTest test = new MemoryDataLayerTest();

            try
            {
                foreach (IMemoryDataLayerTest t in test.Tests)
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
            MemoryDataLayerTest test = new MemoryDataLayerTest();

            try
            {
                foreach (IMemoryDataLayerTest t in test.Tests)
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
        public void TestForwardAddVecSingle()
        {
            MemoryDataLayerTest test = new MemoryDataLayerTest();

            try
            {
                foreach (IMemoryDataLayerTest t in test.Tests)
                {
                    t.TestForwardAddVecSingle();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAddVecMultiple()
        {
            MemoryDataLayerTest test = new MemoryDataLayerTest();

            try
            {
                foreach (IMemoryDataLayerTest t in test.Tests)
                {
                    t.TestForwardAddVecMultiple();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAddVecMultiple2()
        {
            MemoryDataLayerTest test = new MemoryDataLayerTest();

            try
            {
                foreach (IMemoryDataLayerTest t in test.Tests)
                {
                    t.TestForwardAddVecMultiple2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAddVecMultiple3()
        {
            MemoryDataLayerTest test = new MemoryDataLayerTest();

            try
            {
                foreach (IMemoryDataLayerTest t in test.Tests)
                {
                    t.TestForwardAddVecMultiple3();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IMemoryDataLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
        void TestForwardAddVecSingle();
        void TestForwardAddVecMultiple();
        void TestForwardAddVecMultiple2();
        void TestForwardAddVecMultiple3();
    }

    class MemoryDataLayerTest : TestBase
    {
        public MemoryDataLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("MemoryData Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MemoryDataLayerTest<double>(strName, nDeviceID, engine);
            else
                return new MemoryDataLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class MemoryDataLayerTest<T> : TestEx<T>, IMemoryDataLayerTest
    {
        IXImageDatabaseBase m_db;
        int m_nSrcId;
        Blob<T> m_data;
        Blob<T> m_labels;
        Blob<T> m_dataBlob;
        Blob<T> m_labelBlob;
        int m_nBatchSize = 8;
        int m_nBatches = 12;
        int m_nChannels = 4;
        int m_nHeight = 7;
        int m_nWidth = 11;

        public MemoryDataLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;

            m_data = new Blob<T>(m_cuda, m_log);
            m_labels = new Blob<T>(m_cuda, m_log);
            m_dataBlob = new Blob<T>(m_cuda, m_log);
            m_labelBlob = new Blob<T>(m_cuda, m_log);
            TopVec.Clear();
            TopVec.Add(m_dataBlob);
            TopVec.Add(m_labelBlob);

            // pick random input data.
            FillerParameter fp = getFillerParam();
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            m_data.Reshape(m_nBatches * m_nBatchSize, m_nChannels, m_nHeight, m_nWidth);
            m_labels.Reshape(m_nBatches * m_nBatchSize, m_nChannels, m_nHeight, m_nWidth);
            filler.Fill(m_data);
            filler.Fill(m_labels);

            BottomVec.Clear();
        }

        protected override void dispose()
        {
            m_data.Dispose();
            m_dataBlob.Dispose();
            m_labels.Dispose();
            m_labelBlob.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MEMORYDATA);
            p.memory_data_param.batch_size = (uint)m_nBatchSize;
            p.memory_data_param.channels = (uint)m_nChannels;
            p.memory_data_param.height = (uint)m_nHeight;
            p.memory_data_param.width = (uint)m_nWidth;
            MemoryDataLayer<T> layer = new MemoryDataLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(m_dataBlob.num, m_nBatchSize, "The bottom[0] num should equal the batch size.");
                m_log.CHECK_EQ(m_dataBlob.channels, m_nChannels, "The bottom[0] channels is not correct.");
                m_log.CHECK_EQ(m_dataBlob.height, m_nHeight, "The bottom[0] height is not correct.");
                m_log.CHECK_EQ(m_dataBlob.width, m_nWidth, "The bottom[0] width is not correct.");

                m_log.CHECK_EQ(m_labelBlob.num, m_nBatchSize, "The bottom[1] num should equal the batch size.");
                m_log.CHECK_EQ(m_labelBlob.channels, 1, "The bottom[1] channels is not correct.");
                m_log.CHECK_EQ(m_labelBlob.height, 1, "The bottom[1] height is not correct.");
                m_log.CHECK_EQ(m_labelBlob.width, 1, "The bottom[1] width is not correct.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MEMORYDATA);
            p.memory_data_param.batch_size = (uint)m_nBatchSize;
            p.memory_data_param.channels = (uint)m_nChannels;
            p.memory_data_param.height = (uint)m_nHeight;
            p.memory_data_param.width = (uint)m_nWidth;
            MemoryDataLayer<T> layer = new MemoryDataLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.LayerSetUp(BottomVec, TopVec);
                layer.Reset(m_data, m_labels, m_data.num);

                double[] rgData = convert(m_data.update_cpu_data());

                for (int i = 0; i < m_nBatches * 6; i++)
                {
                    int nBatchNum = i % m_nBatches;

                    layer.Forward(BottomVec, TopVec);

                    double[] rgDataBlob = convert(m_dataBlob.update_cpu_data());

                    for (int j = 0; j < m_dataBlob.count(); j++)
                    {
                        double df1 = rgDataBlob[j];
                        int nIdx = m_data.offset(1) * m_nBatchSize * nBatchNum + j;
                        double df2 = rgData[nIdx];

                        m_log.CHECK_EQ(df1, df2, "The data items should match.");
                    }

                    for (int j = 0; j < m_labelBlob.count(); j++)
                    {
                        double df1 = convert(m_labelBlob.GetData(j));
                        int nIdx = m_nBatchSize * nBatchNum + j;
                        double df2 = convert(m_labels.GetData(nIdx));

                        m_log.CHECK_EQ(df1, df2, "The label items should match.");
                    }

                    double dfPct = (double)i / (double)(m_nBatches * 6);
                    Trace.WriteLine("testing at " + dfPct.ToString("P"));
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardAddVecSingle()
        {
            m_db = createImageDb(m_log);
            SettingsCaffe settings = new SettingsCaffe();
            Stopwatch sw = new Stopwatch();

            settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
            m_db.InitializeWithDsName1(settings, "MNIST");
            DatasetDescriptor ds = m_db.GetDatasetByName("MNIST");
            m_nSrcId = ds.TrainingSource.ID;

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MEMORYDATA);
            p.memory_data_param.batch_size = (uint)m_nBatchSize;
            p.memory_data_param.channels = (uint)ds.TrainingSource.ImageChannels;
            p.memory_data_param.height = (uint)ds.TrainingSource.ImageHeight;
            p.memory_data_param.width = (uint)ds.TrainingSource.ImageWidth;
            MemoryDataLayer<T> layer = new MemoryDataLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.OnGetData += MemoryDataLayer_OnGetDataSingleLabel;

                layer.LayerSetUp(BottomVec, TopVec);

                double[] rgData = convert(m_data.update_cpu_data());

                for (int i = 0; i < m_nBatches * 6; i++)
                {
                    int nBatchNum = i % m_nBatches;

                    layer.Forward(BottomVec, TopVec);

                    double[] rgDataBlob = convert(m_dataBlob.update_cpu_data());

                    for (int j = 0; j < m_dataBlob.count(); j++)
                    {
                        double df1 = rgDataBlob[j];
                        int nIdx = m_data.offset(1) * m_nBatchSize * nBatchNum + j;
                        double df2 = rgData[nIdx];

                        m_log.CHECK_EQ(df1, df2, "The data items should match.");
                    }

                    for (int j = 0; j < m_labelBlob.count(); j++)
                    {
                        double df1 = convert(m_labelBlob.GetData(j));
                        int nIdx = m_labels.offset(1) * m_nBatchSize * nBatchNum + j;
                        double df2 = convert(m_labels.GetData(nIdx));

                        m_log.CHECK_EQ(df1, df2, "The label items should match.");
                    }

                    double dfPct = (double)i / (double)(m_nBatches * 6);
                    Trace.WriteLine("testing at " + dfPct.ToString("P"));
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        private void MemoryDataLayer_OnGetDataSingleLabel(object sender, MemoryDataLayerGetDataArgs e)
        {
            if (!e.Initialization)
                return;

            List<Datum> rgData = new List<Datum>();

            for (int i = 0; i < m_nBatchSize * m_nBatches; i++)
            {
                SimpleDatum sd = m_db.QueryImage(m_nSrcId, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
                rgData.Add(new Datum(sd));              
            }

            MemoryDataLayer<T> layer = sender as MemoryDataLayer<T>;

            try
            {
                layer.AddDatumVector(rgData);

                m_data.Reshape(rgData.Count, rgData[0].channels, rgData[0].height, rgData[0].width);
                m_labels.Reshape(rgData.Count, 1, 1, 1);

                // Get the transformed data so that we can verify it later.
                layer.Transformer.Transform(rgData, m_data, m_cuda, m_log);
                List<T> rgLbl = new List<T>();

                for (int i = 0; i < rgData.Count; i++)
                {
                    rgLbl.Add((T)Convert.ChangeType(rgData[i].Label, typeof(T)));
                }

                m_labels.mutable_cpu_data = rgLbl.ToArray();
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardAddVecMultiple()
        {
            m_db = createImageDb(m_log);
            SettingsCaffe settings = new SettingsCaffe();
            Stopwatch sw = new Stopwatch();

            settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
            m_db.InitializeWithDsName1(settings, "MNIST");
            DatasetDescriptor ds = m_db.GetDatasetByName("MNIST");
            m_nSrcId = ds.TrainingSource.ID;

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MEMORYDATA);
            p.memory_data_param.batch_size = (uint)m_nBatchSize;
            p.memory_data_param.channels = (uint)ds.TrainingSource.ImageChannels;
            p.memory_data_param.height = (uint)ds.TrainingSource.ImageHeight;
            p.memory_data_param.width = (uint)ds.TrainingSource.ImageWidth;
            p.memory_data_param.label_channels = 3;
            p.memory_data_param.label_height = 1;
            p.memory_data_param.label_width = 1;
            p.memory_data_param.label_type = LayerParameterBase.LABEL_TYPE.MULTIPLE;
            MemoryDataLayer<T> layer = new MemoryDataLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.OnGetData += MemoryDataLayer_OnGetDataMultipleLabel;

                layer.LayerSetUp(BottomVec, TopVec);

                double[] rgData = convert(m_data.update_cpu_data());

                for (int i = 0; i < m_nBatches * 6; i++)
                {
                    int nBatchNum = i % m_nBatches;

                    layer.Forward(BottomVec, TopVec);

                    double[] rgDataBlob = convert(m_dataBlob.update_cpu_data());

                    for (int j = 0; j < m_dataBlob.count(); j++)
                    {
                        double df1 = rgDataBlob[j];
                        int nIdx = m_data.offset(1) * m_nBatchSize * nBatchNum + j;
                        double df2 = rgData[nIdx];

                        m_log.CHECK_EQ(df1, df2, "The data items should match.");
                    }

                    for (int j = 0; j < m_labelBlob.count(); j++)
                    {
                        double df1 = convert(m_labelBlob.GetData(j));
                        int nIdx = m_labels.offset(1) * m_nBatchSize * nBatchNum + j;
                        double df2 = convert(m_labels.GetData(nIdx));

                        m_log.CHECK_EQ(df1, df2, "The label items should match.");
                    }

                    double dfPct = (double)i / (double)(m_nBatches * 6);
                    Trace.WriteLine("testing at " + dfPct.ToString("P"));
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        private void MemoryDataLayer_OnGetDataMultipleLabel(object sender, MemoryDataLayerGetDataArgs e)
        {
            if (!e.Initialization)
                return;

            List<Datum> rgData = new List<Datum>();

            for (int i = 0; i < m_nBatchSize * m_nBatches; i++)
            {
                SimpleDatum sd = m_db.QueryImage(m_nSrcId, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
                DATA_FORMAT fmt;
                sd.DataCriteria = BinaryData.Pack(new List<float>() { sd.Label, sd.Label, sd.Label }, out fmt);
                sd.DataCriteriaFormat = fmt;

                rgData.Add(new Datum(sd));
            }

            MemoryDataLayer<T> layer = sender as MemoryDataLayer<T>;

            try
            {
                layer.AddDatumVector(rgData);

                m_data.Reshape(rgData.Count, rgData[0].channels, rgData[0].height, rgData[0].width);
                List<float> rgLbl1 = BinaryData.UnPackFloatList(rgData[0].DataCriteria, rgData[0].DataCriteriaFormat);
                m_labels.Reshape(rgData.Count, rgLbl1.Count, 1, 1);

                // Get the transformed data so that we can verify it later.
                layer.Transformer.Transform(rgData, m_data, m_cuda, m_log);
                List<T> rgLbl = new List<T>();

                for (int i = 0; i < rgData.Count; i++)
                {
                    rgLbl1 = BinaryData.UnPackFloatList(rgData[i].DataCriteria, rgData[i].DataCriteriaFormat);

                    for (int j = 0; j < rgLbl1.Count; j++)
                    {
                        rgLbl.Add((T)Convert.ChangeType(rgLbl1[j], typeof(T)));
                    }
                }

                m_labels.mutable_cpu_data = rgLbl.ToArray();
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardAddVecMultiple2()
        {
            m_db = createImageDb(m_log);
            SettingsCaffe settings = new SettingsCaffe();
            Stopwatch sw = new Stopwatch();

            settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
            m_db.InitializeWithDsName1(settings, "MNIST");
            DatasetDescriptor ds = m_db.GetDatasetByName("MNIST");
            m_nSrcId = ds.TrainingSource.ID;

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MEMORYDATA);
            p.memory_data_param.batch_size = (uint)m_nBatchSize;
            p.memory_data_param.channels = (uint)ds.TrainingSource.ImageChannels;
            p.memory_data_param.height = (uint)ds.TrainingSource.ImageHeight;
            p.memory_data_param.width = (uint)ds.TrainingSource.ImageWidth;
            p.memory_data_param.label_channels = 1;
            p.memory_data_param.label_height = 3;
            p.memory_data_param.label_width = 1;
            p.memory_data_param.label_type = LayerParameterBase.LABEL_TYPE.MULTIPLE;
            MemoryDataLayer<T> layer = new MemoryDataLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.OnGetData += MemoryDataLayer_OnGetDataMultipleLabel2;

                layer.LayerSetUp(BottomVec, TopVec);

                double[] rgData = convert(m_data.update_cpu_data());

                for (int i = 0; i < m_nBatches * 6; i++)
                {
                    int nBatchNum = i % m_nBatches;

                    layer.Forward(BottomVec, TopVec);

                    double[] rgDataBlob = convert(m_dataBlob.update_cpu_data());

                    for (int j = 0; j < m_dataBlob.count(); j++)
                    {
                        double df1 = rgDataBlob[j];
                        int nIdx = m_data.offset(1) * m_nBatchSize * nBatchNum + j;
                        double df2 = rgData[nIdx];

                        m_log.CHECK_EQ(df1, df2, "The data items should match.");
                    }

                    for (int j = 0; j < m_labelBlob.count(); j++)
                    {
                        double df1 = convert(m_labelBlob.GetData(j));
                        int nIdx = m_labels.offset(1) * m_nBatchSize * nBatchNum + j;
                        double df2 = convert(m_labels.GetData(nIdx));

                        m_log.CHECK_EQ(df1, df2, "The label items should match.");
                    }

                    double dfPct = (double)i / (double)(m_nBatches * 6);
                    Trace.WriteLine("testing at " + dfPct.ToString("P"));
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        private void MemoryDataLayer_OnGetDataMultipleLabel2(object sender, MemoryDataLayerGetDataArgs e)
        {
            if (!e.Initialization)
                return;

            List<Datum> rgData = new List<Datum>();

            for (int i = 0; i < m_nBatchSize * m_nBatches; i++)
            {
                SimpleDatum sd = m_db.QueryImage(m_nSrcId, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
                DATA_FORMAT fmt;
                sd.DataCriteria = BinaryData.Pack(new List<float>() { sd.Label, sd.Label, sd.Label }, out fmt);
                sd.DataCriteriaFormat = fmt;

                rgData.Add(new Datum(sd));
            }

            MemoryDataLayer<T> layer = sender as MemoryDataLayer<T>;

            try
            {
                layer.AddDatumVector(rgData, null, 2);

                m_data.Reshape(rgData.Count, rgData[0].channels, rgData[0].height, rgData[0].width);
                List<float> rgLbl1 = BinaryData.UnPackFloatList(rgData[0].DataCriteria, rgData[0].DataCriteriaFormat);
                m_labels.Reshape(rgData.Count, 1, rgLbl1.Count, 1);

                // Get the transformed data so that we can verify it later.
                layer.Transformer.Transform(rgData, m_data, m_cuda, m_log);
                List<T> rgLbl = new List<T>();

                for (int i = 0; i < rgData.Count; i++)
                {
                    rgLbl1 = BinaryData.UnPackFloatList(rgData[i].DataCriteria, rgData[i].DataCriteriaFormat);

                    for (int j = 0; j < rgLbl1.Count; j++)
                    {
                        rgLbl.Add((T)Convert.ChangeType(rgLbl1[j], typeof(T)));
                    }
                }

                m_labels.mutable_cpu_data = rgLbl.ToArray();
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForwardAddVecMultiple3()
        {
            m_db = createImageDb(m_log);
            SettingsCaffe settings = new SettingsCaffe();
            Stopwatch sw = new Stopwatch();

            settings.ImageDbLoadMethod = IMAGEDB_LOAD_METHOD.LOAD_ALL;
            m_db.InitializeWithDsName1(settings, "MNIST");
            DatasetDescriptor ds = m_db.GetDatasetByName("MNIST");
            m_nSrcId = ds.TrainingSource.ID;

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MEMORYDATA);
            p.memory_data_param.batch_size = (uint)m_nBatchSize;
            p.memory_data_param.channels = (uint)ds.TrainingSource.ImageChannels;
            p.memory_data_param.height = (uint)ds.TrainingSource.ImageHeight;
            p.memory_data_param.width = (uint)ds.TrainingSource.ImageWidth;
            p.memory_data_param.label_channels = 1;
            p.memory_data_param.label_height = 1;
            p.memory_data_param.label_width = 3;
            p.memory_data_param.label_type = LayerParameterBase.LABEL_TYPE.MULTIPLE;
            MemoryDataLayer<T> layer = new MemoryDataLayer<T>(m_cuda, m_log, p);

            try
            {
                layer.OnGetData += MemoryDataLayer_OnGetDataMultipleLabel3;

                layer.LayerSetUp(BottomVec, TopVec);

                double[] rgData = convert(m_data.update_cpu_data());

                for (int i = 0; i < m_nBatches * 6; i++)
                {
                    int nBatchNum = i % m_nBatches;

                    layer.Forward(BottomVec, TopVec);

                    double[] rgDataBlob = convert(m_dataBlob.update_cpu_data());

                    for (int j = 0; j < m_dataBlob.count(); j++)
                    {
                        double df1 = rgDataBlob[j];
                        int nIdx = m_data.offset(1) * m_nBatchSize * nBatchNum + j;
                        double df2 = rgData[nIdx];

                        m_log.CHECK_EQ(df1, df2, "The data items should match.");
                    }

                    for (int j = 0; j < m_labelBlob.count(); j++)
                    {
                        double df1 = convert(m_labelBlob.GetData(j));
                        int nIdx = m_labels.offset(1) * m_nBatchSize * nBatchNum + j;
                        double df2 = convert(m_labels.GetData(nIdx));

                        m_log.CHECK_EQ(df1, df2, "The label items should match.");
                    }

                    double dfPct = (double)i / (double)(m_nBatches * 6);
                    Trace.WriteLine("testing at " + dfPct.ToString("P"));
                }
            }
            finally
            {
                layer.Dispose();
            }
        }

        private void MemoryDataLayer_OnGetDataMultipleLabel3(object sender, MemoryDataLayerGetDataArgs e)
        {
            if (!e.Initialization)
                return;

            List<Datum> rgData = new List<Datum>();

            for (int i = 0; i < m_nBatchSize * m_nBatches; i++)
            {
                SimpleDatum sd = m_db.QueryImage(m_nSrcId, i, IMGDB_LABEL_SELECTION_METHOD.NONE, IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
                DATA_FORMAT fmt;
                sd.DataCriteria = BinaryData.Pack(new List<float>() { sd.Label, sd.Label, sd.Label }, out fmt);
                sd.DataCriteriaFormat = fmt;

                rgData.Add(new Datum(sd));
            }

            MemoryDataLayer<T> layer = sender as MemoryDataLayer<T>;

            try
            {
                layer.AddDatumVector(rgData, null, 3);

                m_data.Reshape(rgData.Count, rgData[0].channels, rgData[0].height, rgData[0].width);
                List<float> rgLbl1 = BinaryData.UnPackFloatList(rgData[0].DataCriteria, rgData[0].DataCriteriaFormat);
                m_labels.Reshape(rgData.Count, 1, 1, rgLbl1.Count);

                // Get the transformed data so that we can verify it later.
                layer.Transformer.Transform(rgData, m_data, m_cuda, m_log);
                List<T> rgLbl = new List<T>();

                for (int i = 0; i < rgData.Count; i++)
                {
                    rgLbl1 = BinaryData.UnPackFloatList(rgData[i].DataCriteria, rgData[i].DataCriteriaFormat);

                    for (int j = 0; j < rgLbl1.Count; j++)
                    {
                        rgLbl.Add((T)Convert.ChangeType(rgLbl1[j], typeof(T)));
                    }
                }

                m_labels.mutable_cpu_data = rgLbl.ToArray();
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
