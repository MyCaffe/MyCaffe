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

namespace MyCaffe.test
{
    [TestClass]
    public class TestLRNLayer
    {
        [TestMethod]
        public void TestSetupAcrossChannels()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestSetupAcrossChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAcrossChannels()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestForwardAcrossChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAcrossChannelsLargeRegion()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestForwardAcrossChannelsLargeRegion();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAcrossChannels()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestGradientAcrossChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAcrossChannelsLargeRegion()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestGradientAcrossChannelsLargeRegion();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupWithinChannels()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestSetupWithinChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardWithinChannels()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestForwardWithinChannels();
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
        public void TestGradientWithinChannels()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestGradientWithinChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupAcrossChannelsCuDnn()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestSetupAcrossChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAcrossChannelsCuDnn()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestForwardAcrossChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardAcrossChannelsLargeRegionCuDnn()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestForwardAcrossChannelsLargeRegion();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientAcrossChannelsCuDnn()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestGradientAcrossChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetupWithinChannelsCuDnn()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestSetupWithinChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardWithinChannelsCuDnn()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestForwardWithinChannels();
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
        public void TestGradientWithinChannelsCuDnn()
        {
            LRNLayerTest test = new LRNLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (ILRNLayerTest t in test.Tests)
                {
                    t.TestGradientWithinChannels();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ILRNLayerTest : ITest
    {
        void TestSetupAcrossChannels();
        void TestForwardAcrossChannels();
        void TestForwardAcrossChannelsLargeRegion();
        void TestGradientAcrossChannels();
        void TestGradientAcrossChannelsLargeRegion();
        void TestSetupWithinChannels();
        void TestForwardWithinChannels();
        void TestGradientWithinChannels();
    }

    class LRNLayerTest : TestBase
    {
        public LRNLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("LRN Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new LRNLayerTest<double>(strName, nDeviceID, engine);
            else
                return new LRNLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class LRNLayerTest<T> : TestEx<T>, ILRNLayerTest
    {
        double m_dfEpsilon = 1e-5;

        public LRNLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 7, 3, 3 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public double Epsilon
        {
            get { return m_dfEpsilon; }
        }

        public void ReferenceLRNForward(Blob<T> blobBottom, LayerParameter p, Blob<T> blobTop)
        {
            blobTop.ReshapeLike(blobBottom);
            double[] rgTopData = convert(blobTop.mutable_cpu_data);
            double dfAlpha = p.lrn_param.alpha;
            double dfBeta = p.lrn_param.beta;
            int nSize = (int)p.lrn_param.local_size;

            switch (p.lrn_param.norm_region)
            {
                case LRNParameter.NormRegion.ACROSS_CHANNELS:
                    for (int n = 0; n < blobBottom.num; n++)
                    {
                        for (int c = 0; c < blobBottom.channels; c++)
                        {
                            for (int h = 0; h < blobBottom.height; h++)
                            {
                                for (int w = 0; w < blobBottom.width; w++)
                                {
                                    int c_start = c - (nSize - 1) / 2;
                                    int c_end = Math.Min(c_start + nSize, blobBottom.channels);

                                    c_start = Math.Max(c_start, 0);
                                    double dfScale = 1.0;

                                    for (int i = c_start; i < c_end; i++)
                                    {
                                        double dfVal = convert(blobBottom.data_at(n, i, h, w));
                                        dfScale += dfVal * dfVal * dfAlpha / nSize;
                                    }

                                    double dfVal2 = convert(blobBottom.data_at(n, c, h, w)) / Math.Pow(dfScale, dfBeta);
                                    int nIdx = blobTop.offset(n, c, h, w);
                                    rgTopData[nIdx] = dfVal2;
                                }
                            }
                        }
                    }
                    break;

                case LRNParameter.NormRegion.WITHIN_CHANNEL:
                    for (int n = 0; n < blobBottom.num; n++)
                    {
                        for (int c = 0; c < blobBottom.channels; c++)
                        {
                            for (int h = 0; h < blobBottom.height; h++)
                            {
                                int h_start = h - (nSize - 1) / 2;
                                int h_end = Math.Min(h_start + nSize, blobBottom.height);
                                h_start = Math.Max(h_start, 0);

                                for (int w = 0; w < blobBottom.width; w++)
                                {
                                    double dfScale = 1.0;

                                    int w_start = w - (nSize - 1) / 2;
                                    int w_end = Math.Min(w_start + nSize, blobBottom.width);
                                    w_start = Math.Max(w_start, 0);

                                    for (int nh = h_start; nh < h_end; nh++)
                                    {
                                        for (int nw = w_start; nw < w_end; nw++)
                                        {
                                            double dfVal = convert(blobBottom.data_at(n, c, nh, nw));
                                            dfScale += dfVal * dfVal * dfAlpha / (nSize * nSize);                                            
                                        }
                                    }

                                    double dfVal2 = convert(blobBottom.data_at(n, c, h, w)) / Math.Pow(dfScale, dfBeta);
                                    int nIdx = blobTop.offset(n, c, h, w);
                                    rgTopData[nIdx] = dfVal2;
                                }
                            }
                        }
                    }
                    break;

                default:
                    m_log.FAIL("Unknown normalization region.");
                    break;
            }

            blobTop.mutable_cpu_data = convert(rgTopData);
        }

        public void TestSetupAcrossChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LRN);
            p.lrn_param.engine = m_engine;
            LRNLayer<T> layer = new LRNLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(2, Top.num, "The top num should equal 2.");
            m_log.CHECK_EQ(7, Top.channels, "The top channels should equal 7.");
            m_log.CHECK_EQ(3, Top.height, "The top height should equal 3.");
            m_log.CHECK_EQ(3, Top.width, "The top width should equal 3.");
        }

        public void TestForwardAcrossChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LRN);
            p.lrn_param.engine = m_engine;
            // set to caffe defaults
            p.lrn_param.alpha = 1.0;
            p.lrn_param.beta = 0.75;
            p.lrn_param.k = 1.0;
            LRNLayer<T> layer = new LRNLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            Blob<T> top = new Blob<T>(m_cuda, m_log);
            ReferenceLRNForward(Bottom, p, top);

            double[] rgTop = convert(Top.update_cpu_data());
            double[] rgTopRef = convert(top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTop = rgTop[i];
                double dfTopRef = rgTopRef[i];

                m_log.EXPECT_NEAR(dfTop, dfTopRef, Epsilon);
            }

            top.Dispose();
        }

        public void TestForwardAcrossChannelsLargeRegion()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LRN);
            p.lrn_param.engine = m_engine;
            p.lrn_param.local_size = 15;
            // set to caffe defaults
            p.lrn_param.alpha = 1.0;
            p.lrn_param.beta = 0.75;
            p.lrn_param.k = 1.0;
            LRNLayer<T> layer = new LRNLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            Blob<T> top = new Blob<T>(m_cuda, m_log);
            ReferenceLRNForward(Bottom, p, top);

            double[] rgTop = convert(Top.update_cpu_data());
            double[] rgTopRef = convert(top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTop = rgTop[i];
                double dfTopRef = rgTopRef[i];

                m_log.EXPECT_NEAR(dfTop, dfTopRef, Epsilon);
            }

            top.Dispose();
        }

        public void TestGradientAcrossChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LRN);
            p.lrn_param.engine = m_engine;
            // set to caffe defaults
            p.lrn_param.alpha = 1.0;
            p.lrn_param.beta = 0.75;
            p.lrn_param.k = 1.0;
            LRNLayer<T> layer = new LRNLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            Top.SetDiff(1.0);

            List<bool> rgbPropagateDown = new List<bool>();

            for (int i = 0; i < BottomVec.Count; i++)
            {
                rgbPropagateDown.Add(true);
            }

            layer.Backward(TopVec, rgbPropagateDown, BottomVec);

            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestGradientAcrossChannelsLargeRegion()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LRN);
            p.lrn_param.engine = m_engine;
            p.lrn_param.local_size = 15;
            // set to caffe defaults
            p.lrn_param.alpha = 1.0;
            p.lrn_param.beta = 0.75;
            p.lrn_param.k = 1.0;
            LRNLayer<T> layer = new LRNLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            Top.SetDiff(1.0);

            List<bool> rgbPropagateDown = new List<bool>();

            for (int i = 0; i < BottomVec.Count; i++)
            {
                rgbPropagateDown.Add(true);
            }

            layer.Backward(TopVec, rgbPropagateDown, BottomVec);

            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }

        public void TestSetupWithinChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LRN);
            p.lrn_param.engine = m_engine;
            p.lrn_param.norm_region = LRNParameter.NormRegion.WITHIN_CHANNEL;
            p.lrn_param.local_size = 3;
            LRNLayer<T> layer = new LRNLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(2, Top.num, "The top num should equal 2.");
            m_log.CHECK_EQ(7, Top.channels, "The top channels should equal 7.");
            m_log.CHECK_EQ(3, Top.height, "The top height should equal 3.");
            m_log.CHECK_EQ(3, Top.width, "The top width should equal 3.");
        }

        public void TestForwardWithinChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LRN);
            p.lrn_param.engine = m_engine;
            p.lrn_param.norm_region = LRNParameter.NormRegion.WITHIN_CHANNEL;
            p.lrn_param.local_size = 3;
            LRNLayer<T> layer = new LRNLayer<T>(m_cuda, m_log, p);

            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            Blob<T> top = new Blob<T>(m_cuda, m_log);
            ReferenceLRNForward(Bottom, p, top);

            double[] rgTop = convert(Top.update_cpu_data());
            double[] rgTopRef = convert(top.update_cpu_data());

            for (int i = 0; i < Bottom.count(); i++)
            {
                double dfTop = rgTop[i];
                double dfTopRef = rgTopRef[i];

                m_log.EXPECT_NEAR(dfTop, dfTopRef, Epsilon);
            }

            top.Dispose();
        }

        /// <summary>
        /// This test fails
        /// </summary>
        public void TestGradientWithinChannels()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.LRN);
            p.lrn_param.engine = m_engine;
            p.lrn_param.norm_region = LRNParameter.NormRegion.WITHIN_CHANNEL;
            p.lrn_param.local_size = 3;
            LRNLayer<T> layer = new LRNLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);

            Top.SetDiff(1.0);

#warning TestLRNLayer.TestGradientWithinChannels test fails.
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
        }
    }
}
