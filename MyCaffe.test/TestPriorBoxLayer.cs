using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.layers.ssd;
using MyCaffe.fillers;
using MyCaffe.param.ssd;

namespace MyCaffe.test
{
    [TestClass]
    public class TestPriorBoxLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
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
        public void TestSetupMultiSize()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestSetupMultiSize();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupNoMaxSize()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestSetupNoMaxSize();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupMultiSizeNoMaxSize()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestSetupMultiSizeNoMaxSize();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupAspectRatio1()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestSetupAspectRatio1();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupAspectRatioNoFlip()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestSetupAspectRatioNoFlip();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupAspectRatio()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestSetupAspectRatio();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupAspectRatioMultiSize()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestSetupAspectRatioMultiSize();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCpu()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestCpu();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCpuNoMaxSize()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestCpuNoMaxSize();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCpuVariance1()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestCpuVariance1();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCpuVarianceMulti()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestCpuVarianceMulti();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCpuAspectRatioNoFlip()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestCpuAspectRatioNoFlip();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCpuAspectRatio()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestCpuAspectRatio();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCpuAspectRatioMultiSize()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestCpuAspectRatioMultiSize();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestCpuFixStep()
        {
            PriorBoxLayerTest test = new PriorBoxLayerTest();

            try
            {
                foreach (IPriorBoxLayerTest t in test.Tests)
                {
                    t.TestCpuFixStep();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IPriorBoxLayerTest : ITest
    {
        void TestSetup();
        void TestSetupMultiSize();
        void TestSetupNoMaxSize();
        void TestSetupMultiSizeNoMaxSize();
        void TestSetupAspectRatio1();
        void TestSetupAspectRatioNoFlip();
        void TestSetupAspectRatio();
        void TestSetupAspectRatioMultiSize();
        void TestCpu();
        void TestCpuNoMaxSize();
        void TestCpuVariance1();
        void TestCpuVarianceMulti();
        void TestCpuAspectRatioNoFlip();
        void TestCpuAspectRatio();
        void TestCpuAspectRatioMultiSize();
        void TestCpuFixStep();
    }

    class PriorBoxLayerTest : TestBase
    {
        public PriorBoxLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("PriorBox Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new PriorBoxLayerTest<double>(strName, nDeviceID, engine);
            else
                return new PriorBoxLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class PriorBoxLayerTest<T> : TestEx<T>, IPriorBoxLayerTest
    {
        Blob<T> m_blobData;
        int m_nMinSize = 4;
        int m_nMaxSize = 9;

        public PriorBoxLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 10, 10, 10, 10 }, nDeviceID)
        {
            m_engine = engine;

            m_blobData = new Blob<T>(m_cuda, m_log, 10, 3, 100, 100);
            m_filler.Fill(m_blobData);
            BottomVec.Add(m_blobData);
        }

        protected override void dispose()
        {
            dispose(ref m_blobData);
            base.dispose();
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.max_size.Add(m_nMaxSize);
            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_top.num, 1, "The top should have num = 1.");
            m_log.CHECK_EQ(m_blob_top.channels, 2, "The top should have channels = 2.");
            m_log.CHECK_EQ(m_blob_top.height, 100 * 2 * 4, "The top should have height = " + (100 * 2 * 4).ToString() + ".");
            m_log.CHECK_EQ(m_blob_top.width, 1, "The top should have width = 1.");
        }

        public void TestSetupMultiSize()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.min_size.Add(m_nMinSize + 10);
            p.prior_box_param.max_size.Add(m_nMaxSize);
            p.prior_box_param.max_size.Add(m_nMaxSize + 10);
            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_top.num, 1, "The top should have num = 1.");
            m_log.CHECK_EQ(m_blob_top.channels, 2, "The top should have channels = 2.");
            m_log.CHECK_EQ(m_blob_top.height, 100 * 4 * 4, "The top should have height = " + (100 * 1 * 4).ToString() + ".");
            m_log.CHECK_EQ(m_blob_top.width, 1, "The top should have width = 1.");
        }

        public void TestSetupNoMaxSize()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_top.num, 1, "The top should have num = 1.");
            m_log.CHECK_EQ(m_blob_top.channels, 2, "The top should have channels = 2.");
            m_log.CHECK_EQ(m_blob_top.height, 100 * 1 * 4, "The top should have height = " + (100 * 4 * 4).ToString() + ".");
            m_log.CHECK_EQ(m_blob_top.width, 1, "The top should have width = 1.");
        }

        public void TestSetupMultiSizeNoMaxSize()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.min_size.Add(m_nMinSize + 10);
            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_top.num, 1, "The top should have num = 1.");
            m_log.CHECK_EQ(m_blob_top.channels, 2, "The top should have channels = 2.");
            m_log.CHECK_EQ(m_blob_top.height, 100 * 2 * 4, "The top should have height = " + (100 * 2 * 4).ToString() + ".");
            m_log.CHECK_EQ(m_blob_top.width, 1, "The top should have width = 1.");
        }

        public void TestSetupAspectRatio1()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.max_size.Add(m_nMaxSize);
            p.prior_box_param.aspect_ratio.Add(1.0f);
            p.prior_box_param.aspect_ratio.Add(2.0f);
            p.prior_box_param.flip = false;

            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_top.num, 1, "The top should have num = 1.");
            m_log.CHECK_EQ(m_blob_top.channels, 2, "The top should have channels = 2.");
            m_log.CHECK_EQ(m_blob_top.height, 100 * 3 * 4, "The top should have height = " + (100 * 3 * 4).ToString() + ".");
            m_log.CHECK_EQ(m_blob_top.width, 1, "The top should have width = 1.");
        }

        public void TestSetupAspectRatioNoFlip()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.max_size.Add(m_nMaxSize);
            p.prior_box_param.aspect_ratio.Add(2.0f);
            p.prior_box_param.aspect_ratio.Add(3.0f);
            p.prior_box_param.flip = false;

            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_top.num, 1, "The top should have num = 1.");
            m_log.CHECK_EQ(m_blob_top.channels, 2, "The top should have channels = 2.");
            m_log.CHECK_EQ(m_blob_top.height, 100 * 4 * 4, "The top should have height = " + (100 * 4 * 4).ToString() + ".");
            m_log.CHECK_EQ(m_blob_top.width, 1, "The top should have width = 1.");
        }

        public void TestSetupAspectRatio()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.max_size.Add(m_nMaxSize);
            p.prior_box_param.aspect_ratio.Add(2.0f);
            p.prior_box_param.aspect_ratio.Add(3.0f);

            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_top.num, 1, "The top should have num = 1.");
            m_log.CHECK_EQ(m_blob_top.channels, 2, "The top should have channels = 2.");
            m_log.CHECK_EQ(m_blob_top.height, 100 * 6 * 4, "The top should have height = " + (100 * 6 * 4).ToString() + ".");
            m_log.CHECK_EQ(m_blob_top.width, 1, "The top should have width = 1.");
        }

        public void TestSetupAspectRatioMultiSize()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.min_size.Add(m_nMinSize + 10);
            p.prior_box_param.max_size.Add(m_nMaxSize);
            p.prior_box_param.max_size.Add(m_nMaxSize + 10);
            p.prior_box_param.aspect_ratio.Add(2.0f);
            p.prior_box_param.aspect_ratio.Add(3.0f);

            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(m_blob_top.num, 1, "The top should have num = 1.");
            m_log.CHECK_EQ(m_blob_top.channels, 2, "The top should have channels = 2.");
            m_log.CHECK_EQ(m_blob_top.height, 100 * 12 * 4, "The top should have height = " + (100 * 12 * 4).ToString() + ".");
            m_log.CHECK_EQ(m_blob_top.width, 1, "The top should have width = 1.");
        }

        public void TestCpu()
        {
            double dfEps = 1e-6;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.max_size.Add(m_nMaxSize);

            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgdfTopData = convert(m_blob_top.mutable_cpu_data);
            int nTopDataOffset = 0;
            int nDim = m_blob_top.height;

            // Pick a few generated priors and compare against expected number.
            // first prior
            m_log.EXPECT_NEAR(rgdfTopData[0], 0.03, dfEps, "The value at 0 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[1], 0.03, dfEps, "The value at 1 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[2], 0.07, dfEps, "The value at 2 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[3], 0.07, dfEps, "The value at 3 is incorrect.");

            // second prior
            m_log.EXPECT_NEAR(rgdfTopData[4], 0.02, dfEps, "The value at 4 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[5], 0.02, dfEps, "The value at 5 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[6], 0.08, dfEps, "The value at 6 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[7], 0.08, dfEps, "The value at 7 is incorrect.");

            // prior in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 2 * 4 + 4 * 2 * 4 + 0], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 2 * 4 + 4 * 2 * 4 + 1], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 2 * 4 + 4 * 2 * 4 + 2], 0.47, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 2 * 4 + 4 * 2 * 4 + 3], 0.47, dfEps, "The value at is incorrect.");

            // Check variance.
            nTopDataOffset += nDim;
            for (int d = 0; d < nDim; d++)
            {
                m_log.EXPECT_NEAR(rgdfTopData[nTopDataOffset + d], 0.1, dfEps, "The variance at " + d.ToString() + " is incorrect.");
            }
        }

        public void TestCpuNoMaxSize()
        {
            double dfEps = 1e-6;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);

            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgdfTopData = convert(m_blob_top.mutable_cpu_data);
            int nTopDataOffset = 0;
            int nDim = m_blob_top.height;

            // Pick a few generated priors and compare against expected number.
            // first prior
            m_log.EXPECT_NEAR(rgdfTopData[0], 0.03, dfEps, "The value at 0 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[1], 0.03, dfEps, "The value at 1 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[2], 0.07, dfEps, "The value at 2 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[3], 0.07, dfEps, "The value at 3 is incorrect.");

            // prior in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 1 * 4 + 4 * 1 * 4 + 0], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 1 * 4 + 4 * 1 * 4 + 1], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 1 * 4 + 4 * 1 * 4 + 2], 0.47, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 1 * 4 + 4 * 1 * 4 + 3], 0.47, dfEps, "The value at is incorrect.");

            // Check variance.
            nTopDataOffset += nDim;
            for (int d = 0; d < nDim; d++)
            {
                m_log.EXPECT_NEAR(rgdfTopData[nTopDataOffset + d], 0.1, dfEps, "The variance at " + d.ToString() + " is incorrect.");
            }
        }

        public void TestCpuVariance1()
        {
            double dfEps = 1e-6;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.max_size.Add(m_nMaxSize);
            p.prior_box_param.variance.Add(1.0f);

            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgdfTopData = convert(m_blob_top.mutable_cpu_data);
            int nTopDataOffset = 0;
            int nDim = m_blob_top.height;

            // Pick a few generated priors and compare against expected number.
            // first prior
            m_log.EXPECT_NEAR(rgdfTopData[0], 0.03, dfEps, "The value at 0 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[1], 0.03, dfEps, "The value at 1 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[2], 0.07, dfEps, "The value at 2 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[3], 0.07, dfEps, "The value at 3 is incorrect.");

            // second prior
            m_log.EXPECT_NEAR(rgdfTopData[4], 0.02, dfEps, "The value at 4 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[5], 0.02, dfEps, "The value at 5 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[6], 0.08, dfEps, "The value at 6 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[7], 0.08, dfEps, "The value at 7 is incorrect.");

            // prior in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 2 * 4 + 4 * 2 * 4 + 0], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 2 * 4 + 4 * 2 * 4 + 1], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 2 * 4 + 4 * 2 * 4 + 2], 0.47, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 2 * 4 + 4 * 2 * 4 + 3], 0.47, dfEps, "The value at is incorrect.");

            // Check variance.
            nTopDataOffset += nDim;
            for (int d = 0; d < nDim; d++)
            {
                m_log.EXPECT_NEAR(rgdfTopData[nTopDataOffset + d], 1.0, dfEps, "The variance at " + d.ToString() + " is incorrect.");
            }
        }

        public void TestCpuVarianceMulti()
        {
            double dfEps = 1e-6;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.max_size.Add(m_nMaxSize);
            p.prior_box_param.variance.Add(0.1f);
            p.prior_box_param.variance.Add(0.2f);
            p.prior_box_param.variance.Add(0.3f);
            p.prior_box_param.variance.Add(0.4f);

            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgdfTopData = convert(m_blob_top.mutable_cpu_data);
            int nTopDataOffset = 0;
            int nDim = m_blob_top.height;

            // Pick a few generated priors and compare against expected number.
            // first prior
            m_log.EXPECT_NEAR(rgdfTopData[0], 0.03, dfEps, "The value at 0 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[1], 0.03, dfEps, "The value at 1 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[2], 0.07, dfEps, "The value at 2 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[3], 0.07, dfEps, "The value at 3 is incorrect.");

            // second prior
            m_log.EXPECT_NEAR(rgdfTopData[4], 0.02, dfEps, "The value at 4 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[5], 0.02, dfEps, "The value at 5 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[6], 0.08, dfEps, "The value at 6 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[7], 0.08, dfEps, "The value at 7 is incorrect.");

            // prior in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 2 * 4 + 4 * 2 * 4 + 0], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 2 * 4 + 4 * 2 * 4 + 1], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 2 * 4 + 4 * 2 * 4 + 2], 0.47, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 2 * 4 + 4 * 2 * 4 + 3], 0.47, dfEps, "The value at is incorrect.");

            // Check variance.
            nTopDataOffset += nDim;
            for (int d = 0; d < nDim; d++)
            {
                m_log.EXPECT_NEAR(rgdfTopData[nTopDataOffset + d], 0.1 * (d % 4 + 1), dfEps, "The variance at " + d.ToString() + " is incorrect.");
            }
        }

        public void TestCpuAspectRatioNoFlip()
        {
            double dfEps = 1e-6;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.max_size.Add(m_nMaxSize);
            p.prior_box_param.aspect_ratio.Add(2.0f);
            p.prior_box_param.flip = false;

            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgdfTopData = convert(m_blob_top.mutable_cpu_data);
            int nTopDataOffset = 0;
            int nDim = m_blob_top.height;

            // Pick a few generated priors and compare against expected number.
            // first prior
            m_log.EXPECT_NEAR(rgdfTopData[0], 0.03, dfEps, "The value at 0 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[1], 0.03, dfEps, "The value at 1 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[2], 0.07, dfEps, "The value at 2 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[3], 0.07, dfEps, "The value at 3 is incorrect.");

            // second prior
            m_log.EXPECT_NEAR(rgdfTopData[4], 0.02, dfEps, "The value at 4 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[5], 0.02, dfEps, "The value at 5 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[6], 0.08, dfEps, "The value at 6 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[7], 0.08, dfEps, "The value at 7 is incorrect.");

            // third prior
            m_log.EXPECT_NEAR(rgdfTopData[8], 0.05 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at 8 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[9], 0.05 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at 9 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[10], 0.05 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at 10 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[11], 0.05 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at 11 is incorrect.");

            // prior in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 3 * 4 + 4 * 3 * 4 + 0], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 3 * 4 + 4 * 3 * 4 + 1], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 3 * 4 + 4 * 3 * 4 + 2], 0.47, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 3 * 4 + 4 * 3 * 4 + 3], 0.47, dfEps, "The value at is incorrect.");

            // prior with ratio 1:2 in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 3 * 4 + 4 * 3 * 4 + 8], 0.45 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 3 * 4 + 4 * 3 * 4 + 9], 0.45 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 3 * 4 + 4 * 3 * 4 + 10], 0.45 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 3 * 4 + 4 * 3 * 4 + 11], 0.45 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");

            // Check variance.
            nTopDataOffset += nDim;
            for (int d = 0; d < nDim; d++)
            {
                m_log.EXPECT_NEAR(rgdfTopData[nTopDataOffset + d], 0.1, dfEps, "The variance at " + d.ToString() + " is incorrect.");
            }
        }

        public void TestCpuAspectRatio()
        {
            double dfEps = 1e-6;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.max_size.Add(m_nMaxSize);
            p.prior_box_param.aspect_ratio.Add(2.0f);

            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgdfTopData = convert(m_blob_top.mutable_cpu_data);
            int nTopDataOffset = 0;
            int nDim = m_blob_top.height;

            // Pick a few generated priors and compare against expected number.
            // first prior
            m_log.EXPECT_NEAR(rgdfTopData[0], 0.03, dfEps, "The value at 0 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[1], 0.03, dfEps, "The value at 1 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[2], 0.07, dfEps, "The value at 2 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[3], 0.07, dfEps, "The value at 3 is incorrect.");

            // second prior
            m_log.EXPECT_NEAR(rgdfTopData[4], 0.02, dfEps, "The value at 4 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[5], 0.02, dfEps, "The value at 5 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[6], 0.08, dfEps, "The value at 6 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[7], 0.08, dfEps, "The value at 7 is incorrect.");

            // third prior
            m_log.EXPECT_NEAR(rgdfTopData[8], 0.05 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at 8 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[9], 0.05 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at 9 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[10], 0.05 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at 10 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[11], 0.05 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at 11 is incorrect.");

            // fourth prior
            m_log.EXPECT_NEAR(rgdfTopData[12], 0.05 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at 12 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[13], 0.05 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at 13 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[14], 0.05 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at 14 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[15], 0.05 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at 15 is incorrect.");

            // prior in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 4 * 4 + 4 * 4 * 4 + 0], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 4 * 4 + 4 * 4 * 4 + 1], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 4 * 4 + 4 * 4 * 4 + 2], 0.47, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 4 * 4 + 4 * 4 * 4 + 3], 0.47, dfEps, "The value at is incorrect.");

            // prior with ratio 1:2 in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 4 * 4 + 4 * 4 * 4 + 8], 0.45 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 4 * 4 + 4 * 4 * 4 + 9], 0.45 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 4 * 4 + 4 * 4 * 4 + 10], 0.45 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 4 * 4 + 4 * 4 * 4 + 11], 0.45 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");

            // prior with ratio 2:1 in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 4 * 4 + 4 * 4 * 4 + 12], 0.45 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 4 * 4 + 4 * 4 * 4 + 13], 0.45 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 4 * 4 + 4 * 4 * 4 + 14], 0.45 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[4 * 10 * 4 * 4 + 4 * 4 * 4 + 15], 0.45 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");

            // Check variance.
            nTopDataOffset += nDim;
            for (int d = 0; d < nDim; d++)
            {
                m_log.EXPECT_NEAR(rgdfTopData[nTopDataOffset + d], 0.1, dfEps, "The variance at " + d.ToString() + " is incorrect.");
            }
        }

        public void TestCpuAspectRatioMultiSize()
        {
            double dfEps = 1e-6;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.min_size.Add(m_nMinSize + 4);
            p.prior_box_param.max_size.Add(m_nMaxSize);
            p.prior_box_param.max_size.Add(m_nMaxSize + 9);
            p.prior_box_param.aspect_ratio.Add(2.0f);
            p.prior_box_param.clip = true;

            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgdfTopData = convert(m_blob_top.mutable_cpu_data);
            int nTopDataOffset = 0;
            int nDim = m_blob_top.height;

            // Pick a few generated priors and compare against expected number.
            // first prior
            m_log.EXPECT_NEAR(rgdfTopData[0], 0.03, dfEps, "The value at 0 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[1], 0.03, dfEps, "The value at 1 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[2], 0.07, dfEps, "The value at 2 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[3], 0.07, dfEps, "The value at 3 is incorrect.");

            // second prior
            m_log.EXPECT_NEAR(rgdfTopData[4], 0.02, dfEps, "The value at 4 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[5], 0.02, dfEps, "The value at 5 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[6], 0.08, dfEps, "The value at 6 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[7], 0.08, dfEps, "The value at 7 is incorrect.");

            // third prior
            m_log.EXPECT_NEAR(rgdfTopData[8], 0.05 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at 8 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[9], 0.05 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at 9 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[10], 0.05 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at 10 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[11], 0.05 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at 11 is incorrect.");

            // fourth prior
            m_log.EXPECT_NEAR(rgdfTopData[12], 0.05 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at 4 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[13], 0.05 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at 5 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[14], 0.05 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at 6 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[15], 0.05 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at 7 is incorrect.");

            // fifth prior
            m_log.EXPECT_NEAR(rgdfTopData[16], 0.01, dfEps, "The value at 16 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[17], 0.01, dfEps, "The value at 17 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[18], 0.09, dfEps, "The value at 18 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[19], 0.09, dfEps, "The value at 19 is incorrect.");

            // sixth prior
            m_log.EXPECT_NEAR(rgdfTopData[20], 0.00, dfEps, "The value at 20 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[21], 0.00, dfEps, "The value at 21 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[22], 0.11, dfEps, "The value at 22 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[23], 0.11, dfEps, "The value at 23 is incorrect.");

            // seventh prior
            m_log.EXPECT_NEAR(rgdfTopData[24], 0.00, dfEps, "The value at 24 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[25], 0.05 - 0.04/Math.Sqrt(2.0), dfEps, "The value at 25 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[26], 0.05 + 0.04*Math.Sqrt(2.0), dfEps, "The value at 26 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[27], 0.05 + 0.04/Math.Sqrt(2.0), dfEps, "The value at 27 is incorrect.");

            // eight prior
            m_log.EXPECT_NEAR(rgdfTopData[28], 0.05 - 0.04 / Math.Sqrt(2.0), dfEps, "The value at 28 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[29], 0.00, dfEps, "The value at 29 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[30], 0.05 + 0.04 / Math.Sqrt(2.0), dfEps, "The value at 30 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[31], 0.05 + 0.04 * Math.Sqrt(2.0), dfEps, "The value at 31 is incorrect.");

            // prior in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[8 * 10 * 4 * 4 + 8 * 4 * 4 + 0], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[8 * 10 * 4 * 4 + 8 * 4 * 4 + 1], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[8 * 10 * 4 * 4 + 8 * 4 * 4 + 2], 0.47, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[8 * 10 * 4 * 4 + 8 * 4 * 4 + 3], 0.47, dfEps, "The value at is incorrect.");

            // prior with ratio 1:2 in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[8 * 10 * 4 * 4 + 8 * 4 * 4 + 8], 0.45 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[8 * 10 * 4 * 4 + 8 * 4 * 4 + 9], 0.45 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[8 * 10 * 4 * 4 + 8 * 4 * 4 + 10], 0.45 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[8 * 10 * 4 * 4 + 8 * 4 * 4 + 11], 0.45 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");

            // prior with ratio 2:1 in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[8 * 10 * 4 * 4 + 8 * 4 * 4 + 12], 0.45 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[8 * 10 * 4 * 4 + 8 * 4 * 4 + 13], 0.45 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[8 * 10 * 4 * 4 + 8 * 4 * 4 + 14], 0.45 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[8 * 10 * 4 * 4 + 8 * 4 * 4 + 15], 0.45 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");

            // Check variance.
            nTopDataOffset += nDim;
            for (int d = 0; d < nDim; d++)
            {
                m_log.EXPECT_NEAR(rgdfTopData[nTopDataOffset + d], 0.1, dfEps, "The variance at " + d.ToString() + " is incorrect.");
            }
        }

        public void TestCpuFixStep()
        {
            double dfEps = 1e-6;
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.PRIORBOX);
            p.prior_box_param.min_size.Add(m_nMinSize);
            p.prior_box_param.max_size.Add(m_nMaxSize);
            p.prior_box_param.aspect_ratio.Add(2.0f);
            p.prior_box_param.img_size = 100;
            p.prior_box_param.step = 10;

            List<int> rgOriginalBottomShape = Utility.Clone<int>(m_blob_bottom.shape());
            List<int> rgOriginaDataShape = Utility.Clone<int>(m_blobData.shape());
            List<int> rgShape = Utility.Create<int>(4, 10);
            rgShape[2] = 20;
            m_blob_bottom.Reshape(rgShape);
            rgShape[1] = 3;
            rgShape[2] = 200;
            rgShape[3] = 100;
            m_blobData.Reshape(rgShape);

            PriorBoxLayer<T> layer = new PriorBoxLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            // Now, check values
            double[] rgdfTopData = convert(m_blob_top.mutable_cpu_data);
            int nTopDataOffset = 0;
            int nDim = m_blob_top.height;

            // Pick a few generated priors and compare against expected number.
            // first prior
            m_log.EXPECT_NEAR(rgdfTopData[0], 0.03, dfEps, "The value at 0 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[1], 0.03, dfEps, "The value at 1 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[2], 0.07, dfEps, "The value at 2 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[3], 0.07, dfEps, "The value at 3 is incorrect.");

            // second prior
            m_log.EXPECT_NEAR(rgdfTopData[4], 0.02, dfEps, "The value at 4 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[5], 0.02, dfEps, "The value at 5 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[6], 0.08, dfEps, "The value at 6 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[7], 0.08, dfEps, "The value at 7 is incorrect.");

            // third prior
            m_log.EXPECT_NEAR(rgdfTopData[8], 0.05 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at 8 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[9], 0.05 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at 9 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[10], 0.05 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at 10 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[11], 0.05 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at 11 is incorrect.");

            // fourth prior
            m_log.EXPECT_NEAR(rgdfTopData[12], 0.05 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at 4 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[13], 0.05 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at 5 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[14], 0.05 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at 6 is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[15], 0.05 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at 7 is incorrect.");

            // prior in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[14 * 10 * 4 * 4 + 4 * 4 * 4 + 0], 0.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[14 * 10 * 4 * 4 + 4 * 4 * 4 + 1], 1.43, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[14 * 10 * 4 * 4 + 4 * 4 * 4 + 2], 0.47, dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[14 * 10 * 4 * 4 + 4 * 4 * 4 + 3], 1.47, dfEps, "The value at is incorrect.");

            // prior with ratio 1:2 in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[14 * 10 * 4 * 4 + 4 * 4 * 4 + 8], 0.45 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[14 * 10 * 4 * 4 + 4 * 4 * 4 + 9], 1.45 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[14 * 10 * 4 * 4 + 4 * 4 * 4 + 10], 0.45 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[14 * 10 * 4 * 4 + 4 * 4 * 4 + 11], 1.45 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");

            // prior with ratio 2:1 in the 5-th row and 5-th col
            m_log.EXPECT_NEAR(rgdfTopData[14 * 10 * 4 * 4 + 4 * 4 * 4 + 12], 0.45 - 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[14 * 10 * 4 * 4 + 4 * 4 * 4 + 13], 1.45 - 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[14 * 10 * 4 * 4 + 4 * 4 * 4 + 14], 0.45 + 0.01 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");
            m_log.EXPECT_NEAR(rgdfTopData[14 * 10 * 4 * 4 + 4 * 4 * 4 + 15], 1.45 + 0.02 * Math.Sqrt(2.0), dfEps, "The value at is incorrect.");

            // Check variance.
            nTopDataOffset += nDim;
            for (int d = 0; d < nDim; d++)
            {
                m_log.EXPECT_NEAR(rgdfTopData[nTopDataOffset + d], 0.1, dfEps, "The variance at " + d.ToString() + " is incorrect.");
            }

            m_blobData.Reshape(rgOriginaDataShape);
            m_blob_bottom.Reshape(rgOriginalBottomShape);
        }
    }
}
