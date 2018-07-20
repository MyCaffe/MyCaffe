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
using MyCaffe.layers.beta;

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
    public class TestKnnLayer
    {
        [TestMethod]
        public void TestForward()
        {
            KnnLayerTest test = new KnnLayerTest();

            try
            {
                foreach (IKnnLayerTest t in test.Tests)
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

    interface IKnnLayerTest : ITest
    {
        void TestForward(int k);
    }

    class KnnLayerTest : TestBase
    {
        public KnnLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Knn Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new KnnLayerTest<double>(strName, nDeviceID, engine);
            else
                return new KnnLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class KnnLayerTest<T> : TestEx<T>, IKnnLayerTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();
        int m_nNumOutput = 4;
        int m_nBatchSize;
        int m_nVectorDim;

        public KnnLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
            Fill();
        }

        protected override void dispose()
        {
            m_colData.Dispose();
            base.dispose();
        }

        public void Fill()
        {
            m_nBatchSize = 4;
            m_nVectorDim = 2;

            double[] rgData = new double[m_nBatchSize * m_nVectorDim];
            double[] rgLabel = new double[m_nBatchSize];

            rgLabel[0] = 0;
            rgData[0] = 2;
            rgData[1] = 7;

            rgLabel[1] = 3;
            rgData[2] = 4;
            rgData[3] = 24;

            rgLabel[2] = 2;
            rgData[4] = 21;
            rgData[5] = 21;

            rgLabel[3] = 1;
            rgData[6] = 28;
            rgData[7] = 17;

            Blob<T> blobData1 = new Blob<T>(m_cuda, m_log, m_nBatchSize, m_nVectorDim, 1, 1);
            Blob<T> blobLabel1 = new Blob<T>(m_cuda, m_log, m_nBatchSize, 1, 1, 1);
            blobData1.mutable_cpu_data = convert(rgData);
            blobLabel1.mutable_cpu_data = convert(rgLabel);

            m_colData.Add(blobData1);
            m_colLabels.Add(blobLabel1);


            rgLabel[0] = 3;
            rgData[0] = 6;
            rgData[1] = 27;

            rgLabel[1] = 2;
            rgData[2] = 21;
            rgData[3] = 26;

            rgLabel[2] = 0;
            rgData[4] = 5;
            rgData[5] = 9;

            rgLabel[3] = 1;
            rgData[6] = 25;
            rgData[7] = 9;

            Blob<T> blobData2 = new Blob<T>(m_cuda, m_log, m_nBatchSize, m_nVectorDim, 1, 1);
            Blob<T> blobLabel2 = new Blob<T>(m_cuda, m_log, m_nBatchSize, 1, 1, 1);
            blobData2.mutable_cpu_data = convert(rgData);
            blobLabel2.mutable_cpu_data = convert(rgLabel);

            m_colData.Add(blobData2);
            m_colLabels.Add(blobLabel2);

            rgLabel[0] = 1;
            rgData[0] = 22;
            rgData[1] = 10;

            rgLabel[1] = 0;
            rgData[2] = 7;
            rgData[3] = 14;

            rgLabel[2] = 3;
            rgData[4] = 11;
            rgData[5] = 30;

            rgLabel[3] = 2;
            rgData[6] = 26;
            rgData[7] = 24;

            Blob<T> blobData3 = new Blob<T>(m_cuda, m_log, m_nBatchSize, m_nVectorDim, 1, 1);
            Blob<T> blobLabel3 = new Blob<T>(m_cuda, m_log, m_nBatchSize, 1, 1, 1);
            blobData3.mutable_cpu_data = convert(rgData);
            blobLabel3.mutable_cpu_data = convert(rgLabel);

            m_colData.Add(blobData3);
            m_colLabels.Add(blobLabel3);

            m_blob_bottom.Reshape(new List<int>() { m_nBatchSize, m_nVectorDim, 1, 1 });

            m_blobBottomLabels = new Blob<T>(m_cuda, m_log);
            m_blobBottomLabels.Reshape(new List<int>() { m_nBatchSize, 1, 1, 1 });

            BottomVec.Add(m_blobBottomLabels);
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        public void TestForward(int nK)
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.KNN);
            p.knn_param.k = nK;
            p.knn_param.max_stored_batches = 5;
            p.knn_param.num_output = m_nNumOutput;
            KnnLayer<T> layer = new KnnLayer<T>(m_cuda, m_log, p);


            layer.Setup(BottomVec, TopVec);

            TestForwardTrain(layer);
            TestForwardTest(layer);
        }

        public void TestForwardTrain(KnnLayer<T> layer)
        {
            int nK = layer.layer_param.knn_param.k;

            layer.layer_param.phase = Phase.TRAIN;

            BottomVec.Clear();
            BottomVec.Add(m_blob_bottom);
            BottomVec.Add(m_blobBottomLabels);

            for (int i = 0; i < m_colData.Count; i++)
            {
                m_cuda.copy(m_blob_bottom.count(), m_colData[i].gpu_data, m_blob_bottom.mutable_gpu_data);
                m_cuda.copy(m_blobBottomLabels.count(), m_colLabels[i].gpu_data, m_blobBottomLabels.mutable_gpu_data);

                layer.Forward(BottomVec, TopVec);
            }
        }

        public void TestForwardTest(KnnLayer<T> layer)
        {
            layer.layer_param.phase = Phase.TEST;

            BottomVec.Clear();
            BottomVec.Add(m_blob_bottom);
            BottomVec.Add(m_blobBottomLabels);

            int nBatchIdx = m_colData.Count - 1;
            m_cuda.copy(m_blob_bottom.count(), m_colData[nBatchIdx].gpu_data, m_blob_bottom.mutable_gpu_data);

            layer.Forward(BottomVec, TopVec);

            // Verify the data.
            double[] rgTop = convert(TopVec[0].update_cpu_data());

            // Check the values.
            verify_top(rgTop);
        }

        private void verify_top(double[] rgTop)
        {
            m_log.CHECK_EQ(rgTop.Length, m_nNumOutput * m_nBatchSize, "The top should have 'num_output' * 'batch_size' items.");
            // Check the values.
            int nFindMax;
            int nDim = m_nNumOutput;

            nFindMax = find_max(rgTop, 0 * nDim, nDim);
            m_log.CHECK_EQ(1, nFindMax, "The class found is not correct.");

            nFindMax = find_max(rgTop, 1 * nDim, nDim);
            m_log.CHECK_EQ(0, nFindMax, "The class found is not correct.");

            nFindMax = find_max(rgTop, 2 * nDim, nDim);
            m_log.CHECK_EQ(3, nFindMax, "The class found is not correct.");

            nFindMax = find_max(rgTop, 3 * nDim, nDim);
            m_log.CHECK_EQ(2, nFindMax, "The class found is not correct.");
        }

        private int find_max(double[] rg, int nStartIdx, int nCount)
        {
            int nIdx = 0;
            double df = rg[nStartIdx];

            for (int i = 1; i < nCount; i++)
            {
                if (rg[nStartIdx + i] > df)
                {
                    df = rg[nStartIdx + i];
                    nIdx = i;
                }
            }

            return nIdx;
        }
    }

    class DataItem
    {
        double[] m_rgData;
        int m_nLabel;

        public DataItem(double[] rgData, int nLabel)
        {
            m_rgData = rgData;
            m_nLabel = nLabel;
        }

        public double[] Data
        {
            get { return m_rgData; }
        }

        public int Label
        {
            get { return m_nLabel; }
        }
    }

    class DistanceItem
    {
        DataItem m_item1;
        DataItem m_item2;
        double m_dfDistance;

        public DistanceItem(DataItem item1, DataItem item2)
        {
            List<double> rgSub = new List<double>();

            for (int i=0; i<item1.Data.Length; i++)
            {
                rgSub.Add(item1.Data[i] - item2.Data[i]);
            }

            for (int i = 0; i < rgSub.Count; i++)
            {
                m_dfDistance += (rgSub[i] * rgSub[i]);
            }

            m_item1 = item1;
            m_item2 = item2;
        }

        public DataItem Item1
        {
            get { return m_item1; }
        }

        public DataItem Item2
        {
            get { return m_item2; }
        }

        public double Distance
        {
            get { return m_dfDistance; }
        }

        public override string ToString()
        {
            return "[" + m_item1.Label.ToString() + "] -> [" + m_item2.Label.ToString() + "] = " + m_dfDistance.ToString();
        }
    }
}
