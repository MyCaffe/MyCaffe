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

/// <summary>
/// Testing the debug layer.
/// 
/// Debug Layer - layer merely stores, up to max_stored_batches, batches of input which
/// are then optionally used by various debug visualizers.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestDebugLayer
    {
        [TestMethod]
        public void TestForward()
        {
            DebugLayerTest test = new DebugLayerTest();

            try
            {
                foreach (IDebugLayerTest t in test.Tests)
                {
                    t.TestForward(10);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IDebugLayerTest : ITest
    {
        void TestForward(int k);
    }

    class DebugLayerTest : TestBase
    {
        public DebugLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Debug Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new DebugLayerTest<double>(strName, nDeviceID, engine);
            else
                return new DebugLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class DebugLayerTest<T> : TestEx<T>, IDebugLayerTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();
        int m_nNumOutput = 3;
        int m_nBatchSize;
        int m_nVectorDim;

        public DebugLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            Fill(10, 4, 12);
        }

        protected override void dispose()
        {
            m_colData.Dispose();
            base.dispose();
        }

        public void Fill(int nCount, int nBatchSize, int nVecDim)
        {
            Random random = new Random();
            double[] rgData = new double[nBatchSize * nVecDim];
            double[] rgLabel = new double[nBatchSize];

            m_nBatchSize = nBatchSize;
            m_nVectorDim = nVecDim;

            for (int i = 0; i < nCount; i++)
            {
                for (int j = 0; j < nBatchSize * nVecDim; j++)
                {
                    rgData[j] = random.NextDouble();

                    if (j < nBatchSize)
                        rgLabel[j] = random.Next() % m_nNumOutput;
                }

                Blob<T> blobData = new Blob<T>(m_cuda, m_log, nBatchSize, nVecDim, 1, 1);
                blobData.mutable_cpu_data = convert(rgData);

                Blob<T> blobLabel = new Blob<T>(m_cuda, m_log, nBatchSize, 1, 1, 1);
                blobLabel.mutable_cpu_data = convert(rgLabel);

                m_colData.Add(blobData);
                m_colLabels.Add(blobLabel);
            }

            m_blob_bottom.Reshape(new List<int>() { nBatchSize, nVecDim, 1, 1 });

            m_blobBottomLabels = new Blob<T>(m_cuda, m_log);
            m_blobBottomLabels.Reshape(new List<int>() { nBatchSize, 1, 1, 1 });

            BottomVec.Add(m_blobBottomLabels);
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestForward(int nK)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.DEBUG);
            p.debug_param.max_stored_batches = 5;
            DebugLayer<T> layer = new DebugLayer<T>(m_cuda, m_log, p);


            layer.Setup(BottomVec, TopVec);

            for (int i = 0; i < m_colData.Count; i++)
            {
                m_cuda.copy(m_blob_bottom.count(), m_colData[i].gpu_data, m_blob_bottom.mutable_gpu_data);
                m_cuda.copy(m_blobBottomLabels.count(), m_colLabels[i].gpu_data, m_blobBottomLabels.mutable_gpu_data);

                layer.Forward(BottomVec, TopVec);
            }

            m_log.CHECK_EQ(p.debug_param.max_stored_batches, layer.data.Count, "The data collection count should equal the 'max_stored_batches'");
            m_log.CHECK_EQ(p.debug_param.max_stored_batches, layer.labels.Count, "The label collection count should equal the 'max_stored_batches'");

            // Verify that the last 'max_stored_batches' are the last items.                        
            int nIdx = 0;
            for (int i = m_colData.Count - p.debug_param.max_stored_batches; i < m_colData.Count; i++)
            {
                double[] rgData1 = convert(m_colData[i].update_cpu_data());
                double[] rgLabel1 = convert(m_colLabels[i].update_cpu_data());
                double[] rgData2 = convert(layer.data[nIdx].update_cpu_data());
                double[] rgLabel2 = convert(layer.labels[nIdx].update_cpu_data());

                m_log.CHECK_EQ(rgData1.Length, rgData2.Length, "The data counts don't match.");
                m_log.CHECK_EQ(rgLabel1.Length, rgLabel2.Length, "The label counts don't match.");

                for (int j = 0; j < rgData1.Length; j++)
                {
                    m_log.CHECK_EQ(rgData1[j], rgData2[j], "The data elements don't match.");
                }

                for (int j = 0; j < rgLabel1.Length; j++)
                {
                    m_log.CHECK_EQ(rgLabel1[j], rgLabel2[j], "The label elements don't match.");
                }

                nIdx++;
            }
        }
    }
}
