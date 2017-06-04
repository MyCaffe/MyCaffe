using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSPPLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            SPPLayerTest test = new SPPLayerTest();

            try
            {
                foreach (ISPPLayerTest t in test.Tests)
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
        public void TestEqualOutputDims()
        {
            SPPLayerTest test = new SPPLayerTest();

            try
            {
                foreach (ISPPLayerTest t in test.Tests)
                {
                    t.TestEqualOutputDims();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestEqualOutputDims2()
        {
            SPPLayerTest test = new SPPLayerTest();

            try
            {
                foreach (ISPPLayerTest t in test.Tests)
                {
                    t.TestEqualOutputDims2();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardBackward()
        {
            SPPLayerTest test = new SPPLayerTest();

            try
            {
                foreach (ISPPLayerTest t in test.Tests)
                {
                    t.TestForwardBackward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        /// <summary>
        /// This test fails.
        /// </summary>
        [TestMethod]
        public void TestGradient()
        {
            SPPLayerTest test = new SPPLayerTest();

            try
            {
                foreach (ISPPLayerTest t in test.Tests)
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


    interface ISPPLayerTest : ITest
    {
        void TestSetup();
        void TestEqualOutputDims();
        void TestEqualOutputDims2();
        void TestForwardBackward();
        void TestGradient();
    }

    class SPPLayerTest : TestBase
    {
        public SPPLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("SPP Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SPPLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SPPLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SPPLayerTest<T> : TestEx<T>, ISPPLayerTest
    {
        Blob<T> m_blob_bottom_2;
        Blob<T> m_blob_bottom_3;
        BlobCollection<T> m_bottom_vec_2 = new BlobCollection<T>();
        BlobCollection<T> m_bottom_vec_3 = new BlobCollection<T>();

        public SPPLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 9, 8 }, nDeviceID)
        {
            m_engine = engine;
            m_blob_bottom_2 = new Blob<T>(m_cuda, m_log, 4, 3, 1024, 765);
            m_blob_bottom_3 = new Blob<T>(m_cuda, m_log, 10, 3, 7, 7);

            BottomVec2.Add(m_blob_bottom_2);
            BottomVec3.Add(m_blob_bottom_3);
        }

        protected override void dispose()
        {
            if (m_blob_bottom_2 != null)
            {
                m_blob_bottom_2.Dispose();
                m_blob_bottom_2 = null;
            }

            if (m_blob_bottom_3 != null)
            {
                m_blob_bottom_3.Dispose();
                m_blob_bottom_3 = null;
            }

            base.dispose();
        }

        public Blob<T> Bottom2
        {
            get { return m_blob_bottom_2; }
        }

        public Blob<T> Bottom3
        {
            get { return m_blob_bottom_3; }
        }

        public BlobCollection<T> BottomVec2
        {
            get { return m_bottom_vec_2; }
        }

        public BlobCollection<T> BottomVec3
        {
            get { return m_bottom_vec_3; }
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SPP);
            p.spp_param.pyramid_height = 3;
            SPPLayer<T> layer = new SPPLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            // expected number of pool results is geometric sum
            // (1 - r ** n)/(1 - r) where r = 4 and n = pyramid_height;
            // (1 - 4 ** 3)/(1 - 4) = 21
            // multiply bottom num_chyannels * expected_pool_results
            // to get expected num_channels (3 * 21 = 63)
            m_log.CHECK_EQ(Top.num, 2, "The top should have num = 2");
            m_log.CHECK_EQ(Top.channels, 63, "The top channels should equal 63");
            m_log.CHECK_EQ(Top.height, 1, "The top height should equal 1");
            m_log.CHECK_EQ(Top.width, 1, "The top width should equal 1");
        }

        public void TestEqualOutputDims()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SPP);
            p.spp_param.pyramid_height = 5;
            SPPLayer<T> layer = new SPPLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec2, TopVec);

            // expected number of pool results is geometric sum
            // (1 - r ** n)/(1 - r) where r = 4 and n = pyramid_height;
            // (1 - 4 ** 5)/(1 - 4) = 341
            // multiply bottom num_chyannels * expected_pool_results
            // to get expected num_channels (3 * 341 = 1023)
            m_log.CHECK_EQ(Top.num, 4, "The top should have num = 4");
            m_log.CHECK_EQ(Top.channels, 1023, "The top channels should equal 1023");
            m_log.CHECK_EQ(Top.height, 1, "The top height should equal 1");
            m_log.CHECK_EQ(Top.width, 1, "The top width should equal 1");
        }

        public void TestEqualOutputDims2()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SPP);
            p.spp_param.pyramid_height = 3;
            SPPLayer<T> layer = new SPPLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec3, TopVec);

            // expected number of pool results is geometric sum
            // (1 - r ** n)/(1 - r) where r = 4 and n = pyramid_height;
            // (1 - 4 ** 3)/(1 - 4) = 21
            // multiply bottom num_chyannels * expected_pool_results
            // to get expected num_channels (3 * 21 = 63)
            m_log.CHECK_EQ(Top.num, 10, "The top should have num = 10");
            m_log.CHECK_EQ(Top.channels, 63, "The top channels should equal 63");
            m_log.CHECK_EQ(Top.height, 1, "The top height should equal 1");
            m_log.CHECK_EQ(Top.width, 1, "The top width should equal 1");
        }

        public void TestForwardBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SPP);
            p.spp_param.pyramid_height = 3;
            SPPLayer<T> layer = new SPPLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);
            List<bool> rgbPopagateDown = Utility.Create<bool>(BottomVec.Count, true);
            layer.Backward(BottomVec, rgbPopagateDown, TopVec);
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SPP);
            p.spp_param.pyramid_height = 3;
            SPPLayer<T> layer = new SPPLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new test.GradientChecker<T>(m_cuda, m_log, 1e-4, 1e-2);
            layer.Setup(BottomVec, TopVec);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
