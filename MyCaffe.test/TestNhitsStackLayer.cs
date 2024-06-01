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
    public class TestNhitsStackLayer
    {
        [TestMethod]
        public void TestForward()
        {
            NhitsStackLayerTest test = new NhitsStackLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INhitsStackLayerTest t in test.Tests)
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
            NhitsStackLayerTest test = new NhitsStackLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (INhitsStackLayerTest t in test.Tests)
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

    interface INhitsStackLayerTest : ITest
    {
        void TestForward();
        void TestGradient();
    }

    class NhitsStackLayerTest : TestBase
    {
        public NhitsStackLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("NhitsStack Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new NhitsStackLayerTest<double>(strName, nDeviceID, engine);
            else
                return new NhitsStackLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class NhitsStackLayerTest<T> : TestEx<T>, INhitsStackLayerTest
    {
        Blob<T> m_blobStackFc0;
        Blob<T> m_blobStackFc1;

        public NhitsStackLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 2, 10, 3, 1 }, nDeviceID)
        {
            m_engine = engine;

            m_blobStackFc0 = new Blob<T>(m_cuda, m_log);
            m_blobStackFc0.Name = "stack_fc0";
            m_blobStackFc1 = new Blob<T>(m_cuda, m_log);
            m_blobStackFc1.Name = "stack_fc1";
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        protected override void dispose()
        {
            dispose(ref m_blobStackFc0);
            dispose(ref m_blobStackFc1);
            base.dispose();
        }

        public void TestForward(double dfFillerStd)
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = dfFillerStd;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NHITS_STACK, "nhstk", Phase.TRAIN);
            p.pooling_param.kernel_h = 8;
            p.pooling_param.stride_h = 8;
            p.pooling_param.kernel_w = 1;
            p.pooling_param.stride_w = 1;
            p.fc_param.axis = 2;
            p.fc_param.num_output = 32;
            p.fc_param.bias_term = true;
            p.fc_param.activation = param.ts.FcParameter.ACTIVATION.RELU;
            p.fc_param.dropout_ratio = 0.1f;
            p.fc_param.enable_normalization = true;
            p.nhits_block_param.num_input_chunks = 10;
            p.nhits_block_param.num_output_chunks = 3;
            p.nhits_block_param.downsample_size = 2;
            p.nhits_block_param.num_layers = 2;
            p.nhits_stack_param.num_blocks = 1;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.NHITS_STACK, "The layer type is incorrect!");

                m_blobStackFc0.Reshape(2, 3, 3, 1);
                m_blobStackFc0.SetData(0);

                BottomVec.Add(m_blobStackFc0);
                TopVec.Add(m_blobStackFc1);

                layer.Setup(BottomVec, TopVec);
                layer.Forward(BottomVec, TopVec);

                m_log.CHECK_EQ(m_blob_top.num, 2, "The top(0) num should equal 2.");
                m_log.CHECK_EQ(m_blob_top.channels, 10, "The top(0) num should equal 10.");
                m_log.CHECK_EQ(m_blob_top.height, 3, "The top(0) height should equal 3.");
                m_log.CHECK_EQ(m_blob_top.width, 1, "The top(0) width should equal 1.");
                m_log.CHECK_EQ(m_blobStackFc1.num, 2, "The top(1) num should equal 2.");
                m_log.CHECK_EQ(m_blobStackFc1.channels, 3, "The top(1) num should equal 3.");
                m_log.CHECK_EQ(m_blobStackFc1.height, 3, "The top(1) height should equal 3.");
                m_log.CHECK_EQ(m_blobStackFc1.width, 1, "The top(1) width should equal 1.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackward(double dfFillerStd)
        {
            FillerParameter fp = new FillerParameter("gaussian");
            fp.std = dfFillerStd;
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(Bottom);

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.NHITS_STACK, "nhstk", Phase.TRAIN);
            p.pooling_param.kernel_h = 8;
            p.pooling_param.stride_h = 8;
            p.pooling_param.kernel_w = 1;
            p.pooling_param.stride_w = 1;
            p.fc_param.axis = 2;
            p.fc_param.num_output = 32;
            p.fc_param.bias_term = true;
            p.fc_param.activation = param.ts.FcParameter.ACTIVATION.RELU;
            p.fc_param.dropout_ratio = 0.1f;
            p.fc_param.enable_normalization = true;
            p.nhits_block_param.num_input_chunks = 10;
            p.nhits_block_param.num_output_chunks = 3;
            p.nhits_block_param.downsample_size = 2;
            p.nhits_block_param.num_layers = 2;
            p.nhits_stack_param.num_blocks = 1;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.NHITS_STACK, "The layer type is incorrect!");

                m_blobStackFc0.Reshape(2, 3, 3, 1);
                m_blobStackFc0.SetData(0);

                BottomVec.Add(m_blobStackFc0);
                TopVec.Add(m_blobStackFc1);

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
            TestBackward(1.0);
        }
    }
}
