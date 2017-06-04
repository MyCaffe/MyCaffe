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
using System.Diagnostics;

/// <summary>
/// Testing the embed knn layer.
/// 
/// EmbedKnn Layer - this converts embeddings received into the nearest neighbor and outputs
/// the inverse sum of distances between the input and all previously received inputs.
/// 
/// For example, when using a 128 item embedding for a 10 class problem, the EmbedKnn layer
/// takes each input and calculates the distance between the input and all other inputs
/// collected for each class.  The resulting collection of distances are then summed for
/// each class.  At this point the class with the lowest sum is the nearest neighbor.
/// 
/// However, in order to work with the Accuracy, SoftmaxLoss and Softmax layers, the
/// summed values are normalized to the range between 0 and 1 and then inverted so that
/// the maximum value is accociated with the nearest neighbor class.
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
    }

    interface IMemoryDataLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
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

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MEMORYDATA);
            p.memory_data_param.batch_size = (uint)m_nBatchSize;
            p.memory_data_param.channels = (uint)m_nChannels;
            p.memory_data_param.height = (uint)m_nHeight;
            p.memory_data_param.width = (uint)m_nWidth;
            MemoryDataLayer<T> layer = new MemoryDataLayer<T>(m_cuda, m_log, p);

            layer.DataLayerSetUp(BottomVec, TopVec);
            layer.Reset(m_data, m_labels, m_data.num);

            for (int i = 0; i < m_nBatches * 6; i++)
            {
                int nBatchNum = i % m_nBatches;

                layer.Forward(BottomVec, TopVec);

                for (int j = 0; j < m_dataBlob.count(); j++)
                {
                    double df1 = convert(m_dataBlob.GetData(j));
                    int nIdx = m_data.offset(1) * m_nBatchSize * nBatchNum + j;
                    double df2 = convert(m_data.GetData(nIdx));

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
    }
}
