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
using MyCaffe.layers.beta;

namespace MyCaffe.test
{
    [TestClass]
    public class TestSpatialAttentionLayer
    {
        [TestMethod]
        public void TestSetup()
        {
            SpatialAttentionLayerTest test = new SpatialAttentionLayerTest();

            try
            {
                foreach (ISpatialAttentionLayerTest t in test.Tests)
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
            SpatialAttentionLayerTest test = new SpatialAttentionLayerTest();

            try
            {
                foreach (ISpatialAttentionLayerTest t in test.Tests)
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
            SpatialAttentionLayerTest test = new SpatialAttentionLayerTest();

            try
            {
                foreach (ISpatialAttentionLayerTest t in test.Tests)
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

    interface ISpatialAttentionLayerTest : ITest
    {
        void TestSetup();
        void TestForward();
        void TestGradient();
    }

    class SpatialAttentionLayerTest : TestBase
    {
        public SpatialAttentionLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("SpatialAttention Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SpatialAttentionLayerTest<double>(strName, nDeviceID, engine);
            else
                return new SpatialAttentionLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class SpatialAttentionLayerTest<T> : TestEx<T>, ISpatialAttentionLayerTest
    {
        CryptoRandom m_random = new CryptoRandom();

        public SpatialAttentionLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_random = new CryptoRandom(CryptoRandom.METHOD.DEFAULT, 1701);
            m_engine = engine;

            m_blob_bottom.Reshape(2, 3, 64, 64);
            m_filler.Fill(m_blob_bottom);
        }

        protected override void dispose()
        {
            base.dispose();
        }

        public void TestSetup()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SPATIAL_ATTENTION);
            SpatialAttentionLayer<T> layer = Layer<T>.Create(m_cuda, m_log, p, null) as SpatialAttentionLayer<T>;

            try
            {
                m_log.CHECK(layer != null, "The Spatial Attention layer is null.");

                layer.Setup(BottomVec, TopVec);

                m_log.CHECK_EQ(Bottom.num, Top.num, "The top num should equal the bottom num.");
                m_log.CHECK_EQ(Bottom.channels, Top.channels, "The top channels should equal the bottom channels.");
                m_log.CHECK_EQ(Bottom.height, Top.height, "The top height should equal the bottom height.");
                m_log.CHECK_EQ(Bottom.width, Top.width, "The top width should equal bottom width.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestForward()
        {
            Blob<T> blobAveOut = new Blob<T>(m_cuda, m_log);
            Blob<T> blobMaxOut = new Blob<T>(m_cuda, m_log);
            Blob<T> blobFc1Out = new Blob<T>(m_cuda, m_log);
            Blob<T> blobExpOut = new Blob<T>(m_cuda, m_log);
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SPATIAL_ATTENTION);
            SpatialAttentionLayer<T> layer = new SpatialAttentionLayer<T>(m_cuda, m_log, p);

            try
            {
                m_blob_bottom.Reshape(1, 1, 3, 3);
                float[] rgf = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
                m_blob_bottom.mutable_cpu_data = convert(rgf);

                m_log.CHECK(layer != null, "The Spatial Attention layer is null.");
                layer.Setup(BottomVec, TopVec);

                int nWtCount = layer.blobs.Count / 4;

                for (int i = 0; i < layer.blobs.Count; i++)
                {
                    if (i < nWtCount)
                        layer.blobs[i].SetData(0.1);
                    else if (i < nWtCount * 2)
                        layer.blobs[i].SetData(0.1);
                    else if (i < nWtCount * 3)
                        layer.blobs[i].SetData(0.2);
                    else
                        layer.blobs[i].SetData(0.2);
                }

                layer.Forward(BottomVec, TopVec);

                blobAveOut.CopyFrom(m_blob_bottom, false, true);
                blobAveOut.scale_data(0.1);
                blobMaxOut.CopyFrom(m_blob_bottom, false, true);
                blobMaxOut.scale_data(0.1);
                blobFc1Out.ReshapeLike(blobAveOut);
                blobFc1Out = blobAveOut.MathAdd(blobMaxOut, Utility.ConvertVal<T>(1.0));
                blobFc1Out.scale_data(0.2);
                blobFc1Out.scale_data(0.2);
                m_cuda.sigmoid_fwd(blobFc1Out.count(), blobFc1Out.gpu_data, blobFc1Out.mutable_gpu_data);

                blobExpOut.ReshapeLike(blobFc1Out);
                m_cuda.muladd(blobExpOut.count(), m_blob_bottom.gpu_data, blobFc1Out.gpu_data, blobExpOut.mutable_gpu_data, DIR.FWD);

                float[] rgfActual = convertF(m_blob_top.update_cpu_data());
                float[] rgfExpected = convertF(blobExpOut.update_cpu_data());

                for (int i = 0; i < rgfActual.Length; i++)
                {
                    m_log.EXPECT_NEAR_FLOAT(rgfActual[i], rgfExpected[i], 1e-4);
                }                  
            }
            finally
            {
                blobAveOut.Dispose();
                blobMaxOut.Dispose();
                blobFc1Out.Dispose();
                blobExpOut.Dispose();
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SPATIAL_ATTENTION);
            SpatialAttentionLayer<T> layer = new SpatialAttentionLayer<T>(m_cuda, m_log, p);
            GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);

            try
            {
                m_blob_bottom.Reshape(64, 3, 128, 128);
                m_filler.Fill(m_blob_bottom);

                m_log.CHECK(layer != null, "The Spatial Attention layer is null.");
                layer.Setup(BottomVec, TopVec);

                checker.CheckGradientExhaustive(layer, BottomVec, TopVec);
            }
            finally
            {
                layer.Dispose();
            }
        }
    }
}
