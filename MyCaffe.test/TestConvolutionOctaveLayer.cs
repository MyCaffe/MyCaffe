using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.layers.beta;

namespace MyCaffe.test
{
    [TestClass]
    public class TestConvolutionOctaveLayer
    {
        [TestMethod]
        public void TestSetup_1btmA0()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestSetup(1, 0.0, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetup_1btmAp5()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestSetup(1, 0.5, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward_1btmA0()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestForward(1, 0.0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward_1btmAp5()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestForward(1, 0.5);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient_1btmA0()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestGradient(1, 0.0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient_1btmAp5()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestGradient(1, 0.5);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetup_2btmA0()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestSetup(2, 0.0, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetup_2btmAp5()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestSetup(2, 0.5, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward_2btmA0()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestForward(2, 0.0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward_2btmAp5()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestForward(2, 0.5);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient_2btmA0()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestGradient(2, 0.0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient_2btmAp5()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestGradient(2, 0.5);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetup_1btmA0_CUDNN()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestSetup(1, 0, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetup_1btmAp5_CUDNN()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestSetup(1, 0.5, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward_1btmA0_CUDNN()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestForward(1, 0.0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward_1btmAp5_CUDNN()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestForward(1, 0.5);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient_1btmA0_CUDNN()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestGradient(1, 0.0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient_1btmAp5_CUDNN()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestGradient(1, 0.5);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetup_2btmA0_CUDNN()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestSetup(2, 0, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestSetup_2btmAp5_CUDNN()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestSetup(2, 0.5, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward_2btmA0_CUDNN()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestForward(2, 0.0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward_2btmAp5_CUDNN()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestForward(2, 0.5);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient_2btmA0_CUDNN()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestGradient(2, 0.0);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradient_2btmAp5_CUDNN()
        {
            ConvolutionOctaveLayerTest test = new ConvolutionOctaveLayerTest(EngineParameter.Engine.CUDNN);

            try
            {
                foreach (IConvolutionOctaveLayerTest t in test.Tests)
                {
                    t.TestGradient(2, 0.5);
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }


    interface IConvolutionOctaveLayerTest : ITest
    {
        void TestSetup(int nBottoms, double dfAlpha, bool bUseTensorCores);
        void TestForward(int nBottoms, double dfAlpha);
        void TestGradient(int nBottoms, double dfAlpha);
    }

    class ConvolutionOctaveLayerTest : TestBase
    {
        public ConvolutionOctaveLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Interp Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ConvolutionOctaveLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ConvolutionOctaveLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ConvolutionOctaveLayerTest<T> : TestEx<T>, IConvolutionOctaveLayerTest
    {
        Blob<T> m_blob_bottom2;
        Blob<T> m_blob_top2;

        public ConvolutionOctaveLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 24, 24 }, nDeviceID)
        {
            m_engine = engine;

            m_blob_bottom2 = new Blob<T>(m_cuda, m_log);
            m_blob_bottom2.Reshape(2, 3, 12, 12); // downsampled of bottom.
            m_filler.Fill(m_blob_bottom2);

            m_blob_top2 = new Blob<T>(m_cuda, m_log);
        }

        protected override void dispose()
        {
            m_blob_bottom2.Dispose();
            m_blob_top2.Dispose();

            base.dispose();
        }

        public Blob<T> Bottom2
        {
            get { return m_blob_bottom2; }
        }

        public Blob<T> Top2
        {
            get { return m_blob_top2; }
        }

        public void TestSetup(int nBottoms, double dfAlpha, bool bUseTensorCores)
        {
            BottomVec.Clear();
            BottomVec.Add(Bottom);

            if (nBottoms == 2)
                BottomVec.Add(Bottom2);

            TopVec.Clear();
            TopVec.Add(Top);

            if (dfAlpha > 0)
                TopVec.Add(Top2);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION_OCTAVE);
            p.convolution_octave_param.engine = m_engine;
            p.convolution_octave_param.kernel_size.Add(3);
            p.convolution_octave_param.stride.Add(2);
            p.convolution_octave_param.num_output = 10;
            p.convolution_octave_param.alpha_out = dfAlpha;
            p.convolution_octave_param.cudnn_enable_tensor_cores = bUseTensorCores;
            p.convolution_octave_param.bias_term = false;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.CONVOLUTION_OCTAVE, "The layer type is not correct for ConvolutionOctaveLayer!");
            layer.Setup(BottomVec, TopVec);

            // h2h
            m_log.CHECK_EQ(2, TopVec[0].num, "The top[0] Blob<T> should have num = 2.");
            m_log.CHECK_EQ(5, TopVec[0].channels, "The top[0] Blob<T> should have channels = 5.");
            m_log.CHECK_EQ(10, TopVec[0].height, "The top[0] Blob<T> should have height = 10.");
            m_log.CHECK_EQ(10, TopVec[0].width, "The top[0] Blob<T> should have width = 10.");

            // h2l
            if (dfAlpha > 0)
            {
                m_log.CHECK_EQ(2, TopVec.Count, "The top vector should have 2 items.");
                m_log.CHECK_EQ(2, TopVec[1].num, "The top[0] Blob<T> should have num = 2.");
                m_log.CHECK_EQ(5, TopVec[1].channels, "The top[0] Blob<T> should have channels = 5.");
                m_log.CHECK_EQ(4, TopVec[1].height, "The top[0] Blob<T> should have height = 4.");
                m_log.CHECK_EQ(4, TopVec[1].width, "The top[0] Blob<T> should have width = 4.");
            }

            layer.Dispose();

            // setting group should not change the shape
            p.convolution_octave_param.group = 3;
            layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            layer.Setup(BottomVec, TopVec);

            // h2h
            m_log.CHECK_EQ(2, TopVec[0].num, "The top[0] Blob<T> should have num = 2.");
            m_log.CHECK_EQ(5, TopVec[0].channels, "The top[0] Blob<T> should have channels = 5.");
            m_log.CHECK_EQ(10, TopVec[0].height, "The top[0] Blob<T> should have height = 10.");
            m_log.CHECK_EQ(10, TopVec[0].width, "The top[0] Blob<T> should have width = 10.");

            // h2l
            if (dfAlpha > 0)
            {
                m_log.CHECK_EQ(2, TopVec.Count, "The top vector should have 2 items.");
                m_log.CHECK_EQ(2, TopVec[1].num, "The top[0] Blob<T> should have num = 2.");
                m_log.CHECK_EQ(5, TopVec[1].channels, "The top[0] Blob<T> should have channels = 5.");
                m_log.CHECK_EQ(4, TopVec[1].height, "The top[0] Blob<T> should have height = 4.");
                m_log.CHECK_EQ(4, TopVec[1].width, "The top[0] Blob<T> should have width = 4.");
            }
        }

        public void TestForward(int nBottoms, double dfAlpha)
        {
            BottomVec.Clear();
            BottomVec.Add(Bottom);

            if (nBottoms == 2)
                BottomVec.Add(Bottom2);

            TopVec.Clear();
            TopVec.Add(Top);

            if (dfAlpha > 0)
                TopVec.Add(Top2);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION_OCTAVE);
            p.convolution_octave_param.engine = m_engine;
            p.convolution_octave_param.kernel_size.Add(3);
            p.convolution_octave_param.stride.Add(2);
            p.convolution_octave_param.num_output = 10;
            p.convolution_octave_param.alpha_out = dfAlpha;
            p.convolution_octave_param.cudnn_enable_tensor_cores = false;
            p.convolution_octave_param.bias_term = false;

            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            m_log.CHECK(layer.type == LayerParameter.LayerType.CONVOLUTION_OCTAVE, "The layer type is not correct for ConvolutionOctaveLayer!");
            layer.Setup(BottomVec, TopVec);
            layer.Forward(BottomVec, TopVec);
            layer.Dispose();

            int nChannels = (dfAlpha == 0) ? 10 : (int)(dfAlpha * 10);

            // h2h
            m_log.CHECK_EQ(2, TopVec[0].num, "The top[0] Blob<T> should have num = 2.");
            m_log.CHECK_EQ(nChannels, TopVec[0].channels, "The top[0] Blob<T> should have channels = " + nChannels.ToString() + ".");
            m_log.CHECK_EQ(10, TopVec[0].height, "The top[0] Blob<T> should have height = 10.");
            m_log.CHECK_EQ(10, TopVec[0].width, "The top[0] Blob<T> should have width = 10.");

            // h2l
            if (dfAlpha > 0)
            {
                m_log.CHECK_EQ(2, TopVec.Count, "The top vector should have 2 items.");
                m_log.CHECK_EQ(2, TopVec[1].num, "The top[0] Blob<T> should have num = 2.");
                m_log.CHECK_EQ(5, TopVec[1].channels, "The top[0] Blob<T> should have channels = 5.");
                m_log.CHECK_EQ(4, TopVec[1].height, "The top[0] Blob<T> should have height = 4.");
                m_log.CHECK_EQ(4, TopVec[1].width, "The top[0] Blob<T> should have width = 4.");
            }
        }

        public void TestGradient(int nBottoms, double dfAlpha)
        {
            BottomVec.Clear();
            BottomVec.Add(Bottom);

            if (nBottoms == 2)
                BottomVec.Add(Bottom2);

            TopVec.Clear();
            TopVec.Add(Top);

            if (dfAlpha > 0)
                TopVec.Add(Top2);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CONVOLUTION_OCTAVE);
            p.convolution_octave_param.engine = m_engine;
            p.convolution_octave_param.kernel_size.Add(3);
            p.convolution_octave_param.stride.Add(2);
            p.convolution_octave_param.num_output = 10;
            p.convolution_octave_param.alpha_out = dfAlpha;
            p.convolution_octave_param.cudnn_enable_tensor_cores = false;
            p.convolution_octave_param.bias_term = false;

            ConvolutionOctaveLayer<T> layer = new ConvolutionOctaveLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-2);
            checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            layer.Dispose();
        }
    }
}
