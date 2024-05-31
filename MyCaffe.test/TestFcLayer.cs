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
using System.Diagnostics;

namespace MyCaffe.test
{
    [TestClass]
    public class TestFcLayer
    {
        [TestMethod]
        public void TestForward()
        {
            FcLayerTest test = new FcLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IFcLayerTest t in test.Tests)
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
            FcLayerTest test = new FcLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IFcLayerTest t in test.Tests)
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

    interface IFcLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class FcLayerTest : TestBase
    {
        public FcLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("FC Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new FcLayerTest<double>(strName, nDeviceID, engine);
            else
                return new FcLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class FcLayerTest<T> : TestEx<T>, IFcLayerTest
    {
        public FcLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 3, 4, 5 }, nDeviceID)
        {
            m_engine = engine;
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestForward(double dfFillerStd)
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = dfFillerStd;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.FC, "fc", Phase.TRAIN);
            p.fc_param.axis = 2;
            p.fc_param.num_output = 32;
            p.fc_param.bias_term = true;
            p.fc_param.activation = param.ts.FcParameter.ACTIVATION.RELU;
            p.fc_param.dropout_ratio = 0.1f;
            p.fc_param.enable_normalization = true;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.FC, "The layer type is incorrect!");

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_top.num, 2, "The bottom num should equal 2.");
                m_log.CHECK_EQ(m_blob_top.channels, 3, "The bottom num should equal 3.");
                m_log.CHECK_EQ(m_blob_top.height, 32, "The bottom height should equal 32.");

                // Now, check values
                double[] rgTopData = convert(Top.update_cpu_data());
                int nZeroCount = 0;

                for (int i = 0; i < rgTopData.Length; i++)
                {
                    if (rgTopData[i] == 0)
                        nZeroCount++;
                }

                double dfTopPct = (double)nZeroCount / rgTopData.Length;
                m_log.CHECK_GE(dfTopPct, 0.09, "The top data should have at least 10% non-zero values.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackward(double dfFillerStd, int nMethod)
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = dfFillerStd;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.FC, "fc", Phase.TRAIN);
            p.fc_param.axis = 2;
            p.fc_param.num_output = 32;
            p.fc_param.bias_term = false;
            p.fc_param.activation = param.ts.FcParameter.ACTIVATION.RELU;
            p.fc_param.dropout_ratio = 0.0f;
            p.fc_param.enable_normalization = false;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.FC, "The layer type is incorrect!");

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward()
        {
            TestForward(1.0);
        }

        public void TestGradient()
        {
            TestBackward(1.0, 0);
        }

        public void TestGradient2()
        {
            TestBackward(1.0, 1);
        }
    }
}
