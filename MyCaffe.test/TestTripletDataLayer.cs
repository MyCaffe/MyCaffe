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
using MyCaffe.imagedb;
using MyCaffe.basecode.descriptors;
using MyCaffe.data;
using MyCaffe.layers.alpha;

/// <summary>
/// Testing for simple triplet select layer.
/// 
/// TripletData Layer - this is the triplet dataq layer used to load the triplet tuples into the top data
/// where the top is ordered as follows:
/// 
///     colTop[0..nBatchSize/3-1] = anchors
///     colTop[nBatchSize/3..2*nBatchSize/3-1] = positives
///     colTop[2*nBatchSize/3..nBatchSize] = negatives
///     
/// Anchors are all the same image,
/// Positives are ordered randomly,
/// Negatives are ordered randomly.
///     
/// Where Anchors and Positives are from the same class and Negatives are from a different class.  In the basic algorithm,
/// the distance between AP and AN are determined and the learning occurs by shrinking the distance between AP and increasing
/// the distance between AN.
/// 
/// </summary>
/// <remarks>
/// * Initial Python code for TripletDataLayer/TripletDataionLayer/TripletDataLayer by luhaofang/tripletloss on github. 
/// See https://github.com/luhaofang/tripletloss - for general architecture
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTripletDataLayer
    {
        [TestMethod]
        public void TestForward()
        {
            TripletDataLayerTest test = new TripletDataLayerTest();

            try
            {
                foreach (ITripletDataLayerTest t in test.Tests)
                {
                    t.TestForward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ITripletDataLayerTest : ITest
    {
        void TestForward();
    }

    class TripletDataLayerTest : TestBase
    {
        public TripletDataLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TripletData Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TripletDataLayerTest<double>(strName, nDeviceID, engine);
            else
                return new TripletDataLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class TripletDataLayerTest<T> : TestEx<T>, ITripletDataLayerTest
    {
        Blob<T> m_blobTopAnchors;
        Blob<T> m_blobTopPositives;
        Blob<T> m_blobTopNegatives;
        Blob<T> m_blobTopLabels;
        CancelEvent m_evtCancel;
        DatasetDescriptor m_ds;
        MyCaffeImageDatabase m_db;

        public TripletDataLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            m_blobTopAnchors = new Blob<T>(m_cuda, m_log);
            m_blobTopPositives = new Blob<T>(m_cuda, m_log);
            m_blobTopNegatives = new Blob<T>(m_cuda, m_log);
            m_blobTopLabels = new Blob<T>(m_cuda, m_log);
            m_db = new MyCaffeImageDatabase();
            m_evtCancel = new CancelEvent();

            DatasetFactory factory = new DatasetFactory();
            m_ds = factory.LoadDataset("MNIST");

            SettingsCaffe s = new SettingsCaffe();
            s.ImageDbLoadMethod = SettingsCaffe.IMAGEDB_LOAD_METHOD.LOAD_ALL;
            m_db.Initialize(s, m_ds);

            BottomVec.Clear();
            TopVec.Add(m_blobTopLabels);
        }

        protected override void dispose()
        {
            m_blobTopAnchors.Dispose();
            m_blobTopPositives.Dispose();
            m_blobTopNegatives.Dispose();
            m_blobTopLabels.Dispose();
            m_evtCancel.Dispose();
            m_db.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestForward()
        {
            int nBatchSize = 30;
            int nSubBatchSize = nBatchSize / 3;
            int nItemCount = m_ds.TrainingSource.ImageChannels * m_ds.TrainingSource.ImageHeight * m_ds.TrainingSource.ImageWidth;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRIPLET_DATA);
            p.data_param.batch_size = (uint)nBatchSize;
            p.data_param.source = m_ds.TrainingSourceName;
            TripletDataLayer<T> layer = new TripletDataLayer<T>(m_cuda, m_log, p, m_db, m_evtCancel);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            m_log.CHECK_EQ(nBatchSize * nItemCount, Top.count(), "The top should have " + nBatchSize.ToString() + " elements.");
            m_log.CHECK_EQ(nBatchSize, m_blobTopLabels.count(), "The top labels should have " + nBatchSize.ToString() + " elements.");

            int nVectorDim = Top.count(1);
            List<int> rgShape = Utility.Clone<int>(Top.shape());
            rgShape[0] = nSubBatchSize;
            m_blobTopAnchors.Reshape(rgShape);
            m_blobTopPositives.Reshape(rgShape);
            m_blobTopNegatives.Reshape(rgShape);

            m_cuda.copy(m_blobTopAnchors.count(), Top.gpu_data, m_blobTopAnchors.mutable_gpu_data, 0, 0);
            m_cuda.copy(m_blobTopPositives.count(), Top.gpu_data, m_blobTopPositives.mutable_gpu_data, m_blobTopAnchors.count(), 0);
            m_cuda.copy(m_blobTopNegatives.count(), Top.gpu_data, m_blobTopNegatives.mutable_gpu_data, m_blobTopAnchors.count() + m_blobTopPositives.count(), 0);

            rgShape[0] = 1;
            Blob<T> blobAnchor = new Blob<T>(m_cuda, m_log, rgShape);
            Blob<T> blobPositive = new Blob<T>(m_cuda, m_log, rgShape);
            Blob<T> blobNegative = new Blob<T>(m_cuda, m_log, rgShape);
            Blob<T> blobItem = new Blob<T>(m_cuda, m_log, rgShape);
            Blob<T> blobCompare = new Blob<T>(m_cuda, m_log, rgShape);
            double[] rgLabels = convert(m_blobTopLabels.update_cpu_data());
            int nAnchorLabel = -1;
            int nItemLabel = -1;

            // Verify that anchors are all the same.
            for (int i = 0; i < nSubBatchSize; i++)
            {
                m_cuda.copy(blobItem.count(), m_blobTopAnchors.gpu_data, blobItem.mutable_gpu_data, i * nVectorDim, 0);
                nItemLabel = (int)rgLabels[i];

                if (i == 0)
                {
                    m_cuda.copy(blobAnchor.count(), blobItem.gpu_data, blobAnchor.mutable_gpu_data);
                    nAnchorLabel = nItemLabel;
                }
                else
                {
                    m_cuda.sub(blobCompare.count(), blobAnchor.gpu_data, blobItem.gpu_data, blobCompare.mutable_gpu_data);
                    double[] rgData = convert(blobCompare.update_cpu_data());
                    int nDiffCount = 0;

                    for (int j = 0; j < rgData.Length; j++)
                    {
                        if (rgData[j] != 0)
                            nDiffCount++;
                    }

                    m_log.CHECK_EQ(0, nDiffCount, "The anchors should be the same.");
                    m_log.CHECK_EQ(nItemLabel, nAnchorLabel, "The anchor labels should be the same.");
                }
            }


            // Verify that the positives are the same class as the anchor, but contain different data.
            for (int i=0; i<nSubBatchSize; i++)
            {
                m_cuda.copy(blobItem.count(), m_blobTopPositives.gpu_data, blobItem.mutable_gpu_data, i * nVectorDim, 0);
                nItemLabel = (int)rgLabels[i + nSubBatchSize];
               
                m_cuda.sub(blobCompare.count(), blobAnchor.gpu_data, blobItem.gpu_data, blobCompare.mutable_gpu_data);
                double[] rgData = convert(blobCompare.update_cpu_data());
                int nDiffCount = 0;

                for (int j = 0; j < rgData.Length; j++)
                {
                    if (rgData[j] != 0)
                        nDiffCount++;
                }

                m_log.CHECK_GT(nDiffCount, 0, "The anchor data should be different from each positive data.");
                m_log.CHECK_EQ(nItemLabel, nAnchorLabel, "The anchor labels should be the same as the positive label.");
            }


            // Verify that the negative are of a different class as the anchor, and contain different data.
            for (int i = 0; i < nSubBatchSize; i++)
            {
                m_cuda.copy(blobItem.count(), m_blobTopNegatives.gpu_data, blobItem.mutable_gpu_data, i * nVectorDim, 0);
                nItemLabel = (int)rgLabels[i + (2 * nSubBatchSize)];

                m_cuda.sub(blobCompare.count(), blobAnchor.gpu_data, blobItem.gpu_data, blobCompare.mutable_gpu_data);
                double[] rgData = convert(blobCompare.update_cpu_data());
                int nDiffCount = 0;

                for (int j = 0; j < rgData.Length; j++)
                {
                    if (rgData[j] != 0)
                        nDiffCount++;
                }

                m_log.CHECK_GT(nDiffCount, 0, "The anchor data should be different from each negative data.");
                m_log.CHECK_NE(nItemLabel, nAnchorLabel, "The anchor label should be different from the negative labels.");
            }

            // Cleanup
            blobAnchor.Dispose();
            blobPositive.Dispose();
            blobNegative.Dispose();
            blobItem.Dispose();
            blobCompare.Dispose();
        }
    }
}
