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
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.SPATIAL_ATTENTION);
            p.spatial_attention_param.kernel_size = 3;
            p.spatial_attention_param.axis = 1;
            p.spatial_attention_param.activation = SpatialAttentionParameter.ACTIVATION.RELU;
            SpatialAttentionLayer<T> layer = new SpatialAttentionLayer<T>(m_cuda, m_log, p);

            try
            {
                // Setup input data (3x3x3 image with 3 channels)
                m_blob_bottom.Reshape(1, 3, 3, 3);
                float[] rgData = new float[] { 
                    // Channel 1
                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9,
                    // Channel 2
                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9,
                    // Channel 3
                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9
                };
                m_blob_bottom.mutable_cpu_data = convert(rgData);

                layer.Setup(BottomVec, TopVec);
                layer.Reshape(BottomVec, TopVec);

                // Initialize weights and biases with 0.1f for predictable results
                layer.blobs[0].SetData(0.1f);  // ave_conv weight
                layer.blobs[1].SetData(0.1f);  // ave_conv bias
                layer.blobs[2].SetData(0.1f);  // max_conv weight
                layer.blobs[3].SetData(0.1f);  // max_conv bias
                layer.blobs[4].SetData(0.1f);  // fc1 weight
                layer.blobs[5].SetData(0.1f);  // fc2 weight

                // Forward pass
                layer.Forward(BottomVec, TopVec);

                // Calculate expected output manually
                float[] expected = new float[27]; // Same size as input

                // For each position in the input
                for (int i = 0; i < rgData.Length; i++)
                {
                    // Calculate average path
                    float ave_conv = rgData[i] * 0.1f * 27 + 0.1f; // 27 = 3x3x3 kernel
                    float ave_fc1 = ave_conv * 0.1f * 27;
                    float ave_fc1_relu = Math.Max(0, ave_fc1);
                    float ave_fc2 = ave_fc1_relu * 0.1f * 27;

                    // Calculate max path
                    float max_conv = rgData[i] * 0.1f * 27 + 0.1f;
                    float max_fc1 = max_conv * 0.1f * 27;
                    float max_fc1_relu = Math.Max(0, max_fc1);
                    float max_fc2 = max_fc1_relu * 0.1f * 27;

                    // Combine paths and apply sigmoid
                    float fExp = (float)Math.Exp(-(ave_fc2 + max_fc2));
                    float attention = 2.0f / (1.0f + fExp);

                    // Apply attention to input
                    expected[i] = rgData[i] * attention;
                }

                // Compare results
                float[] actual = convertF(m_blob_top.update_cpu_data());

                m_log.WriteLine("Comparing actual vs expected values:");
                for (int i = 0; i < actual.Length; i++)
                {
                    m_log.WriteLine($"Index {i}: Input={rgData[i]:F6}, Actual={actual[i]:F6}, Expected={expected[i]:F6}");
                    m_log.EXPECT_NEAR_FLOAT(actual[i], expected[i], 1e-4f);
                }
            }
            finally
            {
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
                m_blob_bottom.Reshape(8, 3, 32, 32);
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
