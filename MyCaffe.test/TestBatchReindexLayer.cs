using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.common;
using MyCaffe.fillers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestBatchReindexLayer
    {
        [TestMethod]
        public void TestForward()
        {
            BatchReindexLayerTest test = new BatchReindexLayerTest();

            try
            {
                foreach (IBatchReindexLayerTest t in test.Tests)
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
        public void TestGradient()
        {
            BatchReindexLayerTest test = new BatchReindexLayerTest();

            try
            {
                foreach (IBatchReindexLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IBatchReindexLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class BatchReindexLayerTest : TestBase
    {
        public BatchReindexLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("BatchReindex Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new BatchReindexLayerTest<double>(strName, nDeviceID, engine);
            else
                return new BatchReindexLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class BatchReindexLayerTest<T> : TestEx<T>, IBatchReindexLayerTest
    {
        Random m_random;
        Blob<T> m_blob_bottom_permut;

        public BatchReindexLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 5, 4, 3, 2 }, nDeviceID)
        {
            m_engine = engine;
            m_random = new Random(1701);
            m_blob_bottom_permut = new Blob<T>(m_cuda, m_log, new List<int>() { 6 });

            List<int> rgPermut = new List<int>() { 4, 0, 4, 0, 1, 2 };
            for (int i = 0; i < m_blob_bottom_permut.count(); i++)
            {
                m_blob_bottom_permut.SetData(rgPermut[i], i);
            }

            BottomVec.Add(m_blob_bottom_permut);
        }

        protected override void dispose()
        {
            if (m_blob_bottom_permut != null)
            {
                m_blob_bottom_permut.Dispose();
                m_blob_bottom_permut = null;
            }

            base.dispose();
        }

        public Blob<T> BottomPermute
        {
            get { return m_blob_bottom_permut; }
        }

        public void TestForward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BATCHREINDEX);

            List<int> rgSize = new List<int>() { 5, 4, 3, 2 };
            m_blob_bottom.Reshape(rgSize);

            for (int i = 0; i < m_blob_bottom.count(); i++)
            {
                m_blob_bottom.SetData(i, i);
            }

            List<int> rgPermSize = new List<int>() { 6 };
            m_blob_bottom_permut.Reshape(rgPermSize);
            List<int> rgPerm = new List<int>() { 4, 0, 4, 0, 1, 2 };

            for (int i = 0; i < m_blob_bottom_permut.count(); i++)
            {
                m_blob_bottom_permut.SetData(rgPerm[i], i);
            }

            BatchReindexLayer<T> layer = new BatchReindexLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, BottomPermute.num, "The top num should equal the bottom permute num.");
            m_log.CHECK_EQ(Top.channels, Bottom.channels, "The top channels should equal the bottom channels.");
            m_log.CHECK_EQ(Top.height, Bottom.height, "The top height should equal the bottom height.");
            m_log.CHECK_EQ(Top.width, Bottom.width, "The top width should equal the bottom width.");

            layer.Forward(BottomVec, TopVec);

            int nChannels = Top.channels;
            int nHeight = Top.height;
            int nWidth = Top.width;

            for (int i = 0; i < Top.count(); i++)
            {
                int n = i / (nChannels * nWidth * nHeight);
                int inner_idx = (i % (nChannels * nWidth * nHeight));

                int nIdx = rgPerm[n] * nChannels * nWidth * nHeight + inner_idx;
                double dfTop = convert(Top.GetData(i));
                double dfBtm = convert(Bottom.GetData(nIdx));

                m_log.CHECK_EQ(dfTop, dfBtm, "The top and bottom values should be equal.");
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.BATCHREINDEX);
            BatchReindexLayer<T> layer = new BatchReindexLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new test.GradientChecker<T>(m_cuda, m_log, 1e-4, 1e-2);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, 0);
        }
    }
}
