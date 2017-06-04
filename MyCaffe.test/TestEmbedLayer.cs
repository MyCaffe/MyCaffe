using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.layers;
using MyCaffe.common;

namespace MyCaffe.test
{
    [TestClass]
    public class TestEmbedLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            EmbedLayerTest test = new EmbedLayerTest();

            try
            {
                foreach (IEmbedLayerTest t in test.Tests)
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
            EmbedLayerTest test = new EmbedLayerTest();

            try
            {
                foreach (IEmbedLayerTest t in test.Tests)
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
        public void TestForwardWithBias()
        {
            EmbedLayerTest test = new EmbedLayerTest();

            try
            {
                foreach (IEmbedLayerTest t in test.Tests)
                {
                    t.TestForwardWithBias();
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
            EmbedLayerTest test = new EmbedLayerTest();

            try
            {
                foreach (IEmbedLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientWithBias()
        {
            EmbedLayerTest test = new EmbedLayerTest();

            try
            {
                foreach (IEmbedLayerTest t in test.Tests)
                {
                    t.TestGradientWithBias();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IEmbedLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
        void TestForwardWithBias();
        void TestGradient();
        void TestGradientWithBias();
    }

    class EmbedLayerTest : TestBase
    {
        public EmbedLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Embed Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new EmbedLayerTest<double>(strName, nDeviceID, engine);
            else
                return new EmbedLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class EmbedLayerTest<T> : TestEx<T>, IEmbedLayerTest
    {
        public EmbedLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 4, 1, 1, 1 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("uniform");
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.EMBED);
            p.embed_param.num_output = 10;
            p.embed_param.input_dim = 5;
            EmbedLayer<T> layer = new EmbedLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(Top.num_axes, 5, "The top should have num_axes = 5");
            m_log.CHECK_EQ(Top.shape(0), 4, "The top.shape(0) should = 4");
            m_log.CHECK_EQ(Top.shape(1), 1, "The top.shape(1) should = 1");
            m_log.CHECK_EQ(Top.shape(2), 1, "The top.shape(2) should = 1");
            m_log.CHECK_EQ(Top.shape(3), 1, "The top.shape(3) should = 1");
            m_log.CHECK_EQ(Top.shape(4), 10, "The top.shape(4) should = 10");
        }

        public void TestForward()
        {
            Random rand = new Random(1701);
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.EMBED);
            int kNumOutput = 10;
            int kInputDim = 5;
            p.embed_param.num_output = (uint)kNumOutput;
            p.embed_param.input_dim = (uint)kInputDim;
            p.embed_param.weight_filler = new FillerParameter("uniform");
            p.embed_param.weight_filler.min = -10;
            p.embed_param.weight_filler.max = 10;
            p.embed_param.bias_term = false;
            EmbedLayer<T> layer = new EmbedLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(1, layer.blobs.Count, layer.type.ToString() + " should have blobs.Count = 1");

            List<int> rgWeightShape = new List<int>() { kInputDim, kNumOutput };
            m_log.CHECK(Utility.Compare<int>(rgWeightShape, layer.blobs[0].shape()), "The weight shape is not correct!");

            double[] rgBottomData = convert(Bottom.mutable_cpu_data);
            for (int i = 0; i < Bottom.count(); i++)
            {
                rgBottomData[i] = rand.Next() % kInputDim;
            }
            Bottom.mutable_cpu_data = convert(rgBottomData);

            layer.Forward(BottomVec, TopVec);

            List<int> rgWeightOffset = Utility.Create<int>(2, 0);
            List<int> rgTopOffset = Utility.Create<int>(5, 0);

            for (int i = 0; i < Bottom.count(); i++)
            {
                rgWeightOffset[0] = (int)convert(Bottom.GetData(i));
                rgWeightOffset[1] = 0;
                rgTopOffset[0] = i;
                rgTopOffset[4] = 0;

                for (int j = 0; j < kNumOutput; j++)
                {
                    double dfWt = convert(layer.blobs[0].data_at(rgWeightOffset));
                    double dfTop = convert(Top.data_at(rgTopOffset));

                    m_log.CHECK_EQ(dfWt, dfTop, "The top and weight values are not as expected!");

                    rgTopOffset[4]++;
                    rgWeightOffset[1]++;
                }
            }
        }

        public void TestForwardWithBias()
        {
            Random rand = new Random(1701);
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.EMBED);
            int kNumOutput = 10;
            int kInputDim = 5;
            p.embed_param.num_output = (uint)kNumOutput;
            p.embed_param.input_dim = (uint)kInputDim;
            p.embed_param.weight_filler = new FillerParameter("uniform");
            p.embed_param.weight_filler.min = -10;
            p.embed_param.weight_filler.max = 10;
            p.embed_param.bias_term = true;
            p.embed_param.bias_filler = p.embed_param.weight_filler.Clone();
            EmbedLayer<T> layer = new EmbedLayer<T>(m_cuda, m_log, p);
            layer.Setup(BottomVec, TopVec);

            m_log.CHECK_EQ(2, layer.blobs.Count, layer.type.ToString() + " should have blobs.Count = 2");

            List<int> rgWeightShape = new List<int>() { kInputDim, kNumOutput };
            m_log.CHECK(Utility.Compare<int>(rgWeightShape, layer.blobs[0].shape()), "The weight shape is not correct!");

            double[] rgBottomData = convert(Bottom.mutable_cpu_data);
            for (int i = 0; i < Bottom.count(); i++)
            {
                rgBottomData[i] = rand.Next() % kInputDim;
            }
            Bottom.mutable_cpu_data = convert(rgBottomData);

            layer.Forward(BottomVec, TopVec);

            List<int> rgBiasOffset = Utility.Create<int>(1, 0);
            List<int> rgWeightOffset = Utility.Create<int>(2, 0);
            List<int> rgTopOffset = Utility.Create<int>(5, 0);

            for (int i = 0; i < Bottom.count(); i++)
            {
                rgWeightOffset[0] = (int)convert(Bottom.GetData(i));
                rgWeightOffset[1] = 0;

                rgTopOffset[0] = i;
                rgTopOffset[4] = 0;

                rgBiasOffset[0] = 0;

                for (int j = 0; j < kNumOutput; j++)
                {
                    double dfWt = convert(layer.blobs[0].data_at(rgWeightOffset));
                    double dfBias = convert(layer.blobs[1].data_at(rgBiasOffset));

                    dfWt += dfBias;

                    double dfTop = convert(Top.data_at(rgTopOffset));

                    m_log.EXPECT_EQUAL<float>(dfWt, dfTop, "The top and weight values are not as expected!");

                    rgTopOffset[4]++;
                    rgWeightOffset[1]++;
                    rgBiasOffset[0]++;
                }
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.EMBED);
            int kNumOutput = 10;
            int kInputDim = 5;
            p.embed_param.num_output = (uint)kNumOutput;
            p.embed_param.input_dim = (uint)kInputDim;
            p.embed_param.bias_term = false;
            p.embed_param.weight_filler = new FillerParameter("uniform");
            p.embed_param.weight_filler.min = -10;
            p.embed_param.weight_filler.max = 10;
            EmbedLayer<T> layer = new EmbedLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);

            double[] rgBottomData = convert(Bottom.mutable_cpu_data);
            rgBottomData[0] = 4;
            rgBottomData[1] = 2;
            rgBottomData[2] = 2;
            rgBottomData[3] = 3;
            Bottom.mutable_cpu_data = convert(rgBottomData);

            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, -2);
        }

        public void TestGradientWithBias()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.EMBED);
            int kNumOutput = 10;
            int kInputDim = 5;
            p.embed_param.num_output = (uint)kNumOutput;
            p.embed_param.input_dim = (uint)kInputDim;
            p.embed_param.bias_term = true;
            p.embed_param.weight_filler = new FillerParameter("uniform");
            p.embed_param.weight_filler.min = -10;
            p.embed_param.weight_filler.max = 10;
            p.embed_param.bias_filler = p.embed_param.weight_filler.Clone();
            EmbedLayer<T> layer = new EmbedLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log, 1e-2, 1e-3);

            double[] rgBottomData = convert(Bottom.mutable_cpu_data);
            rgBottomData[0] = 4;
            rgBottomData[1] = 2;
            rgBottomData[2] = 2;
            rgBottomData[3] = 3;
            Bottom.mutable_cpu_data = convert(rgBottomData);

            checker.CheckGradientExhaustive(layer, BottomVec, TopVec, -2);
        }
    }
}
