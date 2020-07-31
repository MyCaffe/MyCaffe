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
using System.Drawing;
using System.IO;

namespace MyCaffe.test
{
    [TestClass]
    public class TestDataSequenceLayer
    {
        [TestMethod]
        public void TestInitialize()
        {
            DataSequenceLayerTest test = new DataSequenceLayerTest("MNIST");
                     
            try
            {
                foreach (IDataSequenceLayerTest t in test.Tests)
                {
                    t.TestInitialization(test.SourceName, 0, true);
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
            DataSequenceLayerTest test = new DataSequenceLayerTest("MNIST");

            try
            {
                foreach (IDataSequenceLayerTest t in test.Tests)
                {
                    t.TestSetup(test.SourceName, 0, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardK0Balance()
        {
            DataSequenceLayerTest test = new DataSequenceLayerTest("MNIST");

            try
            {
                foreach (IDataSequenceLayerTest t in test.Tests)
                {
                    t.TestForward(test.SourceName, 0, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardK0()
        {
            DataSequenceLayerTest test = new DataSequenceLayerTest("MNIST");

            try
            {
                foreach (IDataSequenceLayerTest t in test.Tests)
                {
                    t.TestForward(test.SourceName, 0, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardK1()
        {
            DataSequenceLayerTest test = new DataSequenceLayerTest("MNIST");

            try
            {
                foreach (IDataSequenceLayerTest t in test.Tests)
                {
                    t.TestForward(test.SourceName, 1, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardK5()
        {
            DataSequenceLayerTest test = new DataSequenceLayerTest("MNIST");

            try
            {
                foreach (IDataSequenceLayerTest t in test.Tests)
                {
                    t.TestForward(test.SourceName, 5, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    class DataSequenceLayerTest : TestBase
    {
        SettingsCaffe m_settings;
        IXImageDatabaseBase m_db;
        CancelEvent m_evtCancel = new CancelEvent();
        string m_strSrc = "MNIST.training";

        public DataSequenceLayerTest(string strDs = null)
            : base("Data Sequence Layer Test")
        {
            m_settings = new SettingsCaffe();
            m_settings.EnableLabelBalancing = false;
            m_settings.EnableLabelBoosting = false;
            m_settings.EnablePairInputSelection = false;
            m_settings.EnableRandomInputSelection = false;

            if (strDs != null && strDs.Length > 0)
            {
                m_db = createImageDb(null);
                m_db.InitializeWithDsName1(m_settings, strDs);

                DatasetDescriptor ds = m_db.GetDatasetByName(strDs);
                m_strSrc = ds.TrainingSourceName;
            }            
        }

        protected override ITest create(DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
        {
            string strPath = TestBase.CudaPath;

            if (dt == DataType.DOUBLE)
            {
                CudaDnn<double>.SetDefaultCudaPath(strPath);
                return new DataSequenceLayerTest<double>(strName, nDeviceID, this);
            }
            else
            {
                CudaDnn<float>.SetDefaultCudaPath(strPath);
                return new DataSequenceLayerTest<float>(strName, nDeviceID, this);
            }
        }

        protected override void dispose()
        {
            if (m_db != null)
            {
                ((IDisposable)m_db).Dispose();
                m_db = null;
            }

            base.dispose();
        }

        public string SourceName
        {
            get { return m_strSrc; }
        }

        public IXImageDatabaseBase db
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

    interface IDataSequenceLayerTest 
    {
        DataType Type { get; }
        void TestInitialization(string strSrc, int nK, bool bBalance);
        void TestSetup(string strSrc, int nK, bool bBalance);
        void TestForward(string strSrc, int nK, bool bBalance);
    }

    class DataSequenceLayerTest<T> : TestEx<T>, IDataSequenceLayerTest
    {
        Blob<T> m_blobCompare;
        Blob<T> m_blob_top_label;
        BlobCollection<T> m_colDataSeqTop = new BlobCollection<T>();

        DataSequenceLayerTest m_parent;
        Layer<T> m_dataLayer = null;
        Layer<T> m_dataSeqLayer = null;
        int m_kCacheSize = 10;

        public DataSequenceLayerTest(string strName, int nDeviceID, DataSequenceLayerTest parent, List<int> rgBottomShape = null)
            : base(strName, rgBottomShape, nDeviceID)
        {
            m_parent = parent;
            m_blob_top_label = new Blob<T>(m_cuda, m_log);

            TopVec.Add(m_blob_top_label);
            BottomVec.Clear();
        }

        protected override void dispose()
        {
            if (m_dataLayer != null)
            {
                m_dataLayer.Dispose();
                m_dataLayer = null;
            }

            if (m_dataSeqLayer != null)
            {
                m_dataSeqLayer.Dispose();
                m_dataSeqLayer = null;
            }

            Thread.Sleep(2000);
            m_parent.CancelEvent.Reset();

            if (m_blobCompare != null)
            {
                m_blobCompare.Dispose();
                m_blobCompare = null;
            }

            m_blob_top_label.Dispose();
            m_colDataSeqTop.Dispose();

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

        public BlobCollection<T> TopVecSeq
        {
            get { return m_colDataSeqTop; }
        }

        public void TestInitialization(string strSrc, int nK, bool bBalance)
        {
            DatasetFactory factory = new DatasetFactory();
            SourceDescriptor src = factory.LoadSource(strSrc);
            m_blobCompare = new Blob<T>(m_cuda, m_log, 1, src.ImageChannels, src.ImageHeight, src.ImageWidth);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DATA);
            p.data_param.batch_size = 64;
            p.data_param.source = strSrc;
            p.data_param.enable_random_selection = false;
            p.data_param.enable_pair_selection = false;
            p.data_param.enable_noise_for_nonmatch = false;
            p.data_param.enable_debug_output = false;
            m_dataLayer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent, m_parent.db);

            p = new LayerParameter(LayerParameter.LayerType.DATA_SEQUENCE);
            p.data_sequence_param.k = nK;
            p.data_sequence_param.balance_matches = (nK == 0) ? bBalance : false;
            p.data_sequence_param.cache_size = m_kCacheSize;
            p.data_sequence_param.output_labels = true;
            p.data_sequence_param.label_count = 10;
            p.data_sequence_param.label_start = 0;
            m_dataSeqLayer = Layer<T>.Create(m_cuda, m_log, p, m_parent.CancelEvent);

            int nTopCount = 2 + nK + 1;

            for (int i = 0; i < nTopCount; i++)
            {
                Blob<T> top = new Blob<T>(m_cuda, m_log);
                top.Name = "top_" + i.ToString();
                m_colDataSeqTop.Add(top);
            }
        }

        public void TestSetup(string strSrc, int nK, bool bBalance)
        {
            TestInitialization(strSrc, nK, bBalance);

            m_log.CHECK_EQ(m_dataSeqLayer.layer_param.data_sequence_param.k, nK, "The 'k' value is incorred!");
            m_log.CHECK_EQ(m_dataSeqLayer.layer_param.data_sequence_param.cache_size, m_kCacheSize, "The 'cache_size' value is incorred!");

            m_dataLayer.LayerSetUp(BottomVec, TopVec);
            m_dataLayer.Reshape(BottomVec, TopVec);
            m_dataSeqLayer.LayerSetUp(TopVec, TopVecSeq);
            m_dataSeqLayer.Reshape(TopVec, TopVecSeq);
        }

        public void TestForward(string strSrc, int nK, bool bBalance)
        {
            TestSetup(strSrc, nK, bBalance);

            for (int i = 0; i < 100; i++)
            {
                m_dataLayer.Forward(BottomVec, TopVec);
                m_log.CHECK_EQ(TopVec.Count, 2, "The data top vec should return data and label!");

                float[] rgLabels = convertF(TopVec[1].update_cpu_data());
                m_log.CHECK_EQ(rgLabels.Length, m_dataLayer.layer_param.data_param.batch_size, "The label count does not equal the batch size!");

                m_dataSeqLayer.Forward(TopVec, TopVecSeq);

                if (nK == 0)
                    m_log.CHECK_EQ(TopVecSeq.Count, 2 + 1, "The data seq top vec should return anchor and negative!");
                else
                    m_log.CHECK_EQ(TopVecSeq.Count, 2 + nK + 1, "The data seq top vec should return anchor, postive and " + nK.ToString() + " negatives!");

                // Verify anchor sequence - should be in same order as the batch.
                float[] rgSeqLabels = convertF(TopVecSeq[TopVecSeq.Count - 1].update_cpu_data());
                float[] rgSeqAnchor = convertF(TopVecSeq[0].update_cpu_data());
                int nTupletDim = (2 + nK);
                m_log.CHECK_EQ(rgSeqLabels.Length, rgLabels.Length * nTupletDim, "The output label count is incorrect!");

                for (int j = 0; j < rgLabels.Length; j++)
                {
                    int nLabel = (int)rgLabels[j];
                    int nLabelAnchor = (int)rgSeqLabels[j * nTupletDim];
                    m_log.CHECK_EQ(nLabel, nLabelAnchor, "The batch label and anchor label do not match!");
                }

                // Verify tuplet labels.
                if (nK > 0)
                {
                    for (int j = 0; j < rgLabels.Length; j++)
                    {
                        int nLabelAnchor = (int)rgSeqLabels[j * nTupletDim];

                        // The negative labels should NOT equal the anchor.
                        for (int k = 1; k <= nK; k++)
                        {
                            int nLabelNegative = (int)rgSeqLabels[j * nTupletDim + 1 + k];
                            m_log.CHECK_NE(nLabelAnchor, nLabelNegative, "The batch label and anchor label do not match!");
                        }

                        // The positive labels should equal the anchor, but the data should be different.
                        int nLabelPositive = (int)rgSeqLabels[j * nTupletDim + 1];
                        m_log.CHECK_EQ(nLabelAnchor, nLabelPositive, "The anchor does not equal the positive label!");
                    }
                }
                // When k = 0, only one negative label is produced and no positives, and when 'balance_matches' = true, the negative alternates between negative and positive.
                else
                {
                    for (int j = 0; j < rgLabels.Length; j++)
                    {
                        int nLabelAnchor = (int)rgSeqLabels[j * nTupletDim];
                        int nLabelNegative = (int)rgSeqLabels[j * nTupletDim + 1];

                        if (m_dataSeqLayer.layer_param.data_sequence_param.balance_matches && ((j % 2) == 0))
                            m_log.CHECK_EQ(nLabelAnchor, nLabelNegative, "The anchor label and negative (balance matching) label should match!");
                        else
                            m_log.CHECK_NE(nLabelAnchor, nLabelNegative, "The anchor label and negative label should not match!");
                    }
                }

                // Verify the data.
                int nDim = m_blobCompare.count();
                double dfAsum;

                if (nK > 0)
                {
                    Blob<T> anchor = TopVecSeq[0];
                    Blob<T> positive = TopVecSeq[1];
                    int nDuplicateCount = 0;

                    BlobCollection<T> negatives = new BlobCollection<T>();
                    for (int k = 0; k < nK; k++)
                    {
                        Blob<T> negative = TopVecSeq[2 + k];
                        negatives.Add(negative);
                    }

                    for (int j = 0; j < anchor.num; j++)
                    {
                        m_blobCompare.SetData(0);
                        m_cuda.sub(nDim, anchor.gpu_data, positive.gpu_data, m_blobCompare.mutable_gpu_data, j * nDim, j * nDim);
                        dfAsum = Utility.ConvertVal<T>(m_blobCompare.asum_data());
                        nDuplicateCount += ((dfAsum == 0) ? 1 : 0);

                        for (int k = 1; k <= nK; k++)
                        {
                            m_blobCompare.SetData(0);
                            m_cuda.sub(nDim, anchor.gpu_data, negatives[k-1].gpu_data, m_blobCompare.mutable_gpu_data, j * nDim, j * nDim);
                            dfAsum = Utility.ConvertVal<T>(m_blobCompare.asum_data());
                            m_log.CHECK_NE(dfAsum, 0, "The difference between anchor and negative '" + (k-1).ToString() + " data should not be zero!");
                        }
                    }

                    if (nDuplicateCount >= anchor.num / 2)
                        m_log.FAIL("Too many duplicates found!");
                }
                // When k = 0, only one negative data is produced and no positives.
                else
                {
                    Blob<T> anchor = TopVecSeq[0];
                    Blob<T> negative = TopVecSeq[1];
                    int nDuplicateCount = 0;

                    for (int j = 0; j < anchor.num; j++)
                    {
                        m_blobCompare.SetData(0);
                        m_cuda.sub(nDim, anchor.gpu_data, negative.gpu_data, m_blobCompare.mutable_gpu_data, j * nDim, j * nDim);
                        dfAsum = Utility.ConvertVal<T>(m_blobCompare.asum_data());

                        if (!m_dataSeqLayer.layer_param.data_sequence_param.balance_matches || ((j % 2) != 0))
                            m_log.CHECK_NE(dfAsum, 0, "The difference between anchor and negative data should not be zero!");
                        else
                            nDuplicateCount += ((dfAsum == 0) ? 1 : 0);
                    }

                    if (nDuplicateCount >= anchor.num / 2)
                        m_log.FAIL("Too many duplicates found!");
                }
            }
        }
    }
}
