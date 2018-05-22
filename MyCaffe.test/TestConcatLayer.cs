using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.fillers;
using MyCaffe.common;

namespace MyCaffe.test
{
    [TestClass]
    public class TestConcatLayer
    {
        [TestMethod]
        public void TestSetupNum()
        {
            ConcatLayerTest test = new ConcatLayerTest();

            try
            {
                foreach (IConcatLayerTest t in test.Tests)
                {
                    t.TestSetupNum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupChannels()
        {
            ConcatLayerTest test = new ConcatLayerTest();

            try
            {
                foreach (IConcatLayerTest t in test.Tests)
                {
                    t.TestSetupChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupChannelsNegativeIndexing()
        {
            ConcatLayerTest test = new ConcatLayerTest();

            try
            {
                foreach (IConcatLayerTest t in test.Tests)
                {
                    t.TestSetupChannelsNegativeIndexing();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardTrivial()
        {
            ConcatLayerTest test = new ConcatLayerTest();

            try
            {
                foreach (IConcatLayerTest t in test.Tests)
                {
                    t.TestForwardTrivial();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardNum()
        {
            ConcatLayerTest test = new ConcatLayerTest();

            try
            {
                foreach (IConcatLayerTest t in test.Tests)
                {
                    t.TestForwardNum();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardChannels()
        {
            ConcatLayerTest test = new ConcatLayerTest();

            try
            {
                foreach (IConcatLayerTest t in test.Tests)
                {
                    t.TestForwardChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientTrivial()
        {
            ConcatLayerTest test = new ConcatLayerTest();

            try
            {
                foreach (IConcatLayerTest t in test.Tests)
                {
                    t.TestGradientTrivial();
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
        public void TestGradientNum()
        {
            ConcatLayerTest test = new ConcatLayerTest();

            try
            {
                foreach (IConcatLayerTest t in test.Tests)
                {
                    t.TestGradientNum();
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
        public void TestGradientChannels()
        {
            ConcatLayerTest test = new ConcatLayerTest();

            try
            {
                foreach (IConcatLayerTest t in test.Tests)
                {
                    t.TestGradientChannels();
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
        public void TestGradientChannelsBottomOneOnly()
        {
            ConcatLayerTest test = new ConcatLayerTest();

            try
            {
                foreach (IConcatLayerTest t in test.Tests)
                {
                    t.TestGradientChannelsBottomOneOnly();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IConcatLayerTest : ITest
    {
        void TestSetupNum();
        void TestSetupChannels();
        void TestSetupChannelsNegativeIndexing();
        void TestForwardTrivial();
        void TestForwardNum();
        void TestForwardChannels();
        void TestGradientTrivial();
        void TestGradientNum();
        void TestGradientChannels();
        void TestGradientChannelsBottomOneOnly();
    }

    class ConcatLayerTest : TestBase
    {
        public ConcatLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Concat Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ConcatLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ConcatLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ConcatLayerTest<T> : TestEx<T>, IConcatLayerTest
    {
        Blob<T> m_blob_bottom_1;
        Blob<T> m_blob_bottom_2;
        BlobCollection<T> m_colBottom1 = new BlobCollection<T>();

        public ConcatLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 6, 5 }, nDeviceID)
        {
            m_engine = engine;

            m_blob_bottom_1 = new Blob<T>(m_cuda, m_log, Bottom);
            m_blob_bottom_2 = new Blob<T>(m_cuda, m_log, Bottom);

            FillerParameter fp1 = getFillerParam();
            fp1.value = 2.0;
            Filler<T> filler1 = Filler<T>.Create(m_cuda, m_log, fp1);
            filler1.Fill(m_blob_bottom_1);

            FillerParameter fp2 = getFillerParam();
            fp2.value = 3.0;
            Filler<T> filler2 = Filler<T>.Create(m_cuda, m_log, fp2);
            filler2.Fill(m_blob_bottom_2);

            BottomVec.Add(m_blob_bottom_1);
            BottomVec1.Add(Bottom);
            BottomVec1.Add(m_blob_bottom_2);
        }

        protected override void dispose()
        {
            m_blob_bottom_1.Dispose();
            m_blob_bottom_2.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            FillerParameter fp = new FillerParameter();
            fp.value = 1.0;
            return fp;
        }

        public Blob<T> Bottom1
        {
            get { return m_blob_bottom_1; }
        }

        public Blob<T> Bottom2
        {
            get { return m_blob_bottom_2; }
        }

        public BlobCollection<T> BottomVec1
        {
            get { return m_colBottom1; }
        }

        public void TestSetupNum()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONCAT);
            p.concat_param.axis = 0;
            ConcatLayer<T> layer = new ConcatLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec1, TopVec);

            m_log.CHECK_EQ(Top.num, Bottom.num + Bottom2.num, "The top num should equal bottom0.num + bottom2.num.");
            m_log.CHECK_EQ(Top.channels, Bottom.channels, "The top channels should equal the bottom channels.");
            m_log.CHECK_EQ(Top.height, Bottom.height, "The top height should equal the bottom height.");
            m_log.CHECK_EQ(Top.width, Bottom.width, "The top width should equal the bottom width.");
        }

        public void TestSetupChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONCAT);
            ConcatLayer<T> layer = new ConcatLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, Bottom.num, "The top num should equal bottom num.");
            m_log.CHECK_EQ(Top.channels, Bottom.channels + Bottom1.channels, "The top channels should equal the bottom0.channels + bottom1.channels.");
            m_log.CHECK_EQ(Top.height, Bottom.height, "The top height should equal the bottom height.");
            m_log.CHECK_EQ(Top.width, Bottom.width, "The top width should equal the bottom width.");
        }

        public void TestSetupChannelsNegativeIndexing()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONCAT);
            // 'channels' index is the third one from the end -- test negative indexing
            // by setting axis to -3 and checking that we get the same results as above
            // in TestSetupChannels.
            p.concat_param.axis = -3;
            ConcatLayer<T> layer = new ConcatLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num, Bottom.num, "The top num should equal bottom num.");
            m_log.CHECK_EQ(Top.channels, Bottom.channels + Bottom1.channels, "The top channels should equal the bottom0.channels + bottom1.channels.");
            m_log.CHECK_EQ(Top.height, Bottom.height, "The top height should equal the bottom height.");
            m_log.CHECK_EQ(Top.width, Bottom.width, "The top width should equal the bottom width.");
        }

        public void TestForwardTrivial()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONCAT);
            ConcatLayer<T> layer = new ConcatLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            double[] rgTop = convert(Top.update_cpu_data());
            double[] rgBottom = convert(Bottom.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTop = rgTop[i];
                double dfBottom = rgBottom[i];

                m_log.CHECK_EQ(dfTop, dfTop, "The top and bottom should be equal.");
            }
        }

        public void TestForwardNum()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONCAT);
            p.concat_param.axis = 0;
            ConcatLayer<T> layer = new ConcatLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec1, TopVec);
            layer.Forward(BottomVec1, TopVec);

            for (int n = 0; n < BottomVec1[0].num; n++)
            {
                for (int c = 0; c < Top.channels; c++)
                {
                    for (int h = 0; h < Top.height; h++)
                    {
                        for (int w = 0; w < Top.width; w++)
                        {
                            double dfTop = convert(Top.data_at(n, c, h, w));
                            double dfBottom = convert(BottomVec1[0].data_at(n, c, h, w));

                            m_log.CHECK_EQ(dfTop, dfBottom, "The top and bottom should be equal at: " + n.ToString() + "," + c.ToString() + "," + h.ToString() + "," + w.ToString());
                        }
                    }
                }
            }

            for (int n = 0; n < BottomVec1[1].num; n++)
            {
                for (int c = 0; c < Top.channels; c++)
                {
                    for (int h = 0; h < Top.height; h++)
                    {
                        for (int w = 0; w < Top.width; w++)
                        {
                            double dfTop = convert(Top.data_at(n + 2, c, h, w));
                            double dfBottom = convert(BottomVec1[1].data_at(n, c, h, w));

                            m_log.CHECK_EQ(dfTop, dfBottom, "The top and bottom should be equal at: " + n.ToString() + "," + c.ToString() + "," + h.ToString() + "," + w.ToString());
                        }
                    }
                }
            }
        }

        public void TestForwardChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONCAT);
            ConcatLayer<T> layer = new ConcatLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            for (int n = 0; n < Top.num; n++)
            {
                for (int c = 0; c < Bottom.channels; c++)
                {
                    for (int h = 0; h < Top.height; h++)
                    {
                        for (int w = 0; w < Top.width; w++)
                        {
                            double dfTop = convert(Top.data_at(n, c, h, w));
                            double dfBottom = convert(BottomVec[0].data_at(n, c, h, w));

                            m_log.CHECK_EQ(dfTop, dfBottom, "The top and bottom should be equal at: " + n.ToString() + "," + c.ToString() + "," + h.ToString() + "," + w.ToString());
                        }
                    }
                }
                for (int c = 0; c < Bottom1.channels; c++)
                {
                    for (int h = 0; h < Top.height; h++)
                    {
                        for (int w = 0; w < Top.width; w++)
                        {
                            double dfTop = convert(Top.data_at(n, c + 3, h, w));
                            double dfBottom = convert(BottomVec[1].data_at(n, c, h, w));

                            m_log.CHECK_EQ(dfTop, dfBottom, "The top and bottom should be equal at: " + n.ToString() + "," + c.ToString() + "," + h.ToString() + "," + w.ToString());
                        }
                    }
                }
            }
        }

        public void TestGradientTrivial()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONCAT);
            ConcatLayer<T> layer = new ConcatLayer<T>(m_cuda, m_log, p);

            BottomVec.RemoveAt(1);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
            checker.CheckGradientEltwise(layer, BottomVec, TopVec);
        }

        /// <summary>
        /// This test fails
        /// </summary>
        public void TestGradientNum()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONCAT);
            p.concat_param.axis = 0;
            ConcatLayer<T> layer = new ConcatLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
            checker.CheckGradient(layer, BottomVec1, TopVec);
        }

        /// <summary>
        /// This test fails
        /// </summary>
        public void TestGradientChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONCAT);
            ConcatLayer<T> layer = new ConcatLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
            checker.CheckGradient(layer, BottomVec, TopVec);
        }

        /// <summary>
        /// This test fails
        /// </summary>
        public void TestGradientChannelsBottomOneOnly()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONCAT);
            ConcatLayer<T> layer = new ConcatLayer<T>(m_cuda, m_log, p);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
            checker.CheckGradient(layer, BottomVec, TopVec, 1);
        }
    }
}
