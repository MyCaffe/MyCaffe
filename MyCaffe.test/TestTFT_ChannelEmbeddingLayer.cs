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
using MyCaffe.db.image;
using MyCaffe.basecode.descriptors;
using MyCaffe.data;
using MyCaffe.layers.tft;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.Tab;

/// <summary>
/// Testing the Channel Embedding layer.
/// 
/// ChannelEmbedding Layer - layer converts numerical and categorical inputs into embeddings.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_ChannelEmbeddingLayer
    {
        [TestMethod]
        public void TestForwardStat()
        {
            ChannelEmbeddingLayerTest test = new ChannelEmbeddingLayerTest();

            try
            {
                foreach (IChannelEmbeddingLayerTest t in test.Tests)
                {
                    t.TestForwardStat();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardStat()
        {
            ChannelEmbeddingLayerTest test = new ChannelEmbeddingLayerTest();

            try
            {
                foreach (IChannelEmbeddingLayerTest t in test.Tests)
                {
                    t.TestBackwardStat();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardHist()
        {
            ChannelEmbeddingLayerTest test = new ChannelEmbeddingLayerTest();

            try
            {
                foreach (IChannelEmbeddingLayerTest t in test.Tests)
                {
                    t.TestForwardHist();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardHist()
        {
            ChannelEmbeddingLayerTest test = new ChannelEmbeddingLayerTest();

            try
            {
                foreach (IChannelEmbeddingLayerTest t in test.Tests)
                {
                    t.TestBackwardHist();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestGradientHist()
        {
            ChannelEmbeddingLayerTest test = new ChannelEmbeddingLayerTest();

            try
            {
                foreach (IChannelEmbeddingLayerTest t in test.Tests)
                {
                    t.TestGradientHist();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IChannelEmbeddingLayerTest : ITest
    {
        void TestForwardStat();
        void TestBackwardStat();

        void TestForwardHist();
        void TestBackwardHist();
        void TestGradientHist();
    }

    class ChannelEmbeddingLayerTest : TestBase
    {
        public ChannelEmbeddingLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("ChannelEmbedding Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new ChannelEmbeddingLayerTest<double>(strName, nDeviceID, engine);
            else
                return new ChannelEmbeddingLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class ChannelEmbeddingLayerTest<T> : TestEx<T>, IChannelEmbeddingLayerTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();
        int m_nNumOutput = 3;
        int m_nBatchSize;
        int m_nVectorDim;

        public ChannelEmbeddingLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, null, nDeviceID)
        {
            m_engine = engine;
        }

        protected override void dispose()
        {
            m_colData.Dispose();
            base.dispose();
        }

        protected override FillerParameter getFillerParam()
        {
            return new FillerParameter("gaussian");
        }

        private string getTestDataPath(string strSubPath)
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\" + strSubPath + "\\iter_0\\";
        }

        private string getTestWtsPath(string strSubPath)
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\" + strSubPath + "\\";
        }

        /// <summary>
        /// Test ChannelEmbedding(hist) foward
        /// </summary>
        /// <remarks>
        /// To generate the test data, run the following:
        /// 
        /// Code: test_1_inputchannelembedding_stat.py
        /// Path: ice_hist
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestForwardStat()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CHANNEL_EMBEDDING);
            p.numeric_trans_param.num_input = 0;
            p.numeric_trans_param.state_size = 64;
            p.categorical_trans_param.num_input = 9;
            p.categorical_trans_param.cardinalities = new List<int> { 54, 3627, 23, 17, 6, 18, 33, 320, 3 };
            p.categorical_trans_param.state_size = 64;
            ChannelEmbeddingLayer<T> layer = null;
            Blob<T> blobX_numeric = null;
            Blob<T> blobX_categorical = null;
            Blob<T> blobY = null;
            Blob<T> blobProcessed = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath("ice_stat");
            string strPathWts = getTestWtsPath("static_transform");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as ChannelEmbeddingLayer<T>;
                blobX_numeric = new Blob<T>(m_cuda, m_log);
                blobX_categorical = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobProcessed = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CHANNEL_EMBEDDING, "The layer type is incorrect.");

                //blobX_numeric.LoadFromNumpy(strPath + "x_numeric.npy");
                blobX_categorical.LoadFromNumpy(strPath + "x_categorical.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX_numeric);
                BottomVec.Add(blobX_categorical);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.0.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.1.weight.npy");
                layer.blobs[2].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.2.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.3.weight.npy");
                layer.blobs[4].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.4.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.5.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.6.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.7.weight.npy");
                layer.blobs[8].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.8.weight.npy");

                layer.Forward(BottomVec, TopVec);

                blobProcessed.LoadFromNumpy(strPath + "stat_processed_input.npy");
                m_log.CHECK(TopVec[0].Compare(blobProcessed, blobWork), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobProcessed);
                dispose(ref blobWork);
                dispose(ref blobX_numeric);
                dispose(ref blobX_categorical);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }

        /// <summary>
        /// Test ChannelEmbedding(hist) backward
        /// </summary>
        /// <remarks>
        /// To generate the test data, run the following:
        /// 
        /// Code: test_1_inputchannelembedding_stat.py
        /// Path: ice_hist
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestBackwardStat()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CHANNEL_EMBEDDING);
            p.numeric_trans_param.num_input = 0;
            p.numeric_trans_param.state_size = 64;
            p.categorical_trans_param.num_input = 9;
            p.categorical_trans_param.cardinalities = new List<int> { 54, 3627, 23, 17, 6, 18, 33, 320, 3 };
            p.categorical_trans_param.state_size = 64;
            ChannelEmbeddingLayer<T> layer = null;
            Blob<T> blobGradExp = null;
            Blob<T> blobX_numeric = null;
            Blob<T> blobX_categorical = null;
            Blob<T> blobY = null;
            Blob<T> blobProcessed = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath("ice_stat");
            string strPathWts = getTestWtsPath("static_transform");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as ChannelEmbeddingLayer<T>;
                blobGradExp = new Blob<T>(m_cuda, m_log);
                blobX_numeric = new Blob<T>(m_cuda, m_log);
                blobX_categorical = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobProcessed = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CHANNEL_EMBEDDING, "The layer type is incorrect.");

                //blobX_numeric.LoadFromNumpy(strPath + "x_numeric.npy");
                blobX_categorical.LoadFromNumpy(strPath + "x_categorical.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX_numeric);
                BottomVec.Add(blobX_categorical);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.0.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.1.weight.npy");
                layer.blobs[2].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.2.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.3.weight.npy");
                layer.blobs[4].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.4.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.5.weight.npy");
                layer.blobs[6].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.6.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.7.weight.npy");
                layer.blobs[8].LoadFromNumpy(strPathWts + "categorical_transform.categorical_embedding_layers.8.weight.npy");

                layer.Forward(BottomVec, TopVec);

                blobProcessed.LoadFromNumpy(strPath + "stat_processed_input.npy");
                m_log.CHECK(TopVec[0].Compare(blobProcessed, blobWork), "The blobs do not match.");

                TopVec[0].LoadFromNumpy(strPath + "stat_processed_input.grad.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                // Data inputs do not have output grads.

                double dfErr = 1e-08;

                blobGradExp.LoadFromNumpy(strPath + "static_emb_layer.0.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[0], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "static_emb_layer.1.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[1], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "static_emb_layer.2.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[2], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "static_emb_layer.3.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[3], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "static_emb_layer.4.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[4], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "static_emb_layer.5.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[5], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "static_emb_layer.6.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[6], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "static_emb_layer.7.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[7], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "static_emb_layer.8.weight_grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[8], blobWork, true, dfErr), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobGradExp);
                dispose(ref blobProcessed);
                dispose(ref blobWork);
                dispose(ref blobX_numeric);
                dispose(ref blobX_categorical);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }

        /// <summary>
        /// Test ChannelEmbedding(hist) foward
        /// </summary>
        /// <remarks>
        /// To generate the test data, run the following:
        /// 
        /// Code: test_1_inputchannelembedding_hist.py
        /// Path: ice_hist
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestForwardHist()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CHANNEL_EMBEDDING);
            p.numeric_trans_param.num_input = 4;
            p.numeric_trans_param.state_size = 64;
            p.categorical_trans_param.num_input = 7;
            p.categorical_trans_param.cardinalities = new List<int> { 2, 3, 8, 13, 72, 6, 28 };
            p.categorical_trans_param.state_size = 64;
            ChannelEmbeddingLayer<T> layer = null;
            Blob<T> blobX_numeric = null;
            Blob<T> blobX_categorical = null;
            Blob<T> blobY = null;
            Blob<T> blobProcessed = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath("ice_hist");
            string strPathWts = getTestWtsPath("hist_ts_transform");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as ChannelEmbeddingLayer<T>;
                blobX_numeric = new Blob<T>(m_cuda, m_log);
                blobX_categorical = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobProcessed = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CHANNEL_EMBEDDING, "The layer type is incorrect.");

                blobX_numeric.LoadFromNumpy(strPath + "x_numeric.npy");
                blobX_categorical.LoadFromNumpy(strPath + "x_categorical.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX_numeric);
                BottomVec.Add(blobX_categorical);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.0.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.0.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.1.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.1.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.2.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.2.bias.npy");
                layer.blobs[6].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.3.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.3.bias.npy");

                layer.blobs[8].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.0.weight.npy");
                layer.blobs[9].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.1.weight.npy");
                layer.blobs[10].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.2.weight.npy");
                layer.blobs[11].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.3.weight.npy");
                layer.blobs[12].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.4.weight.npy");
                layer.blobs[13].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.5.weight.npy");
                layer.blobs[14].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.6.weight.npy");

                layer.Forward(BottomVec, TopVec);

                blobProcessed.LoadFromNumpy(strPath + "hist_processed_input.npy");
                double dfErr = (typeof(T) == typeof(float)) ? 1e-08 : 2e-07;
                m_log.CHECK(TopVec[0].Compare(blobProcessed, blobWork, false, dfErr), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobProcessed);
                dispose(ref blobWork);
                dispose(ref blobX_numeric);
                dispose(ref blobX_categorical);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }

        /// <summary>
        /// Test ChannelEmbedding(hist) backward
        /// </summary>
        /// <remarks>
        /// To generate the test data, run the following:
        /// 
        /// Code: test_1_inputchannelembedding_hist.py
        /// Path: ice_hist
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestBackwardHist()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CHANNEL_EMBEDDING);
            p.numeric_trans_param.num_input = 4;
            p.numeric_trans_param.state_size = 64;
            p.categorical_trans_param.num_input = 7;
            p.categorical_trans_param.cardinalities = new List<int> { 2, 3, 8, 13, 72, 6, 28 };
            p.categorical_trans_param.state_size = 64;
            ChannelEmbeddingLayer<T> layer = null;
            Blob<T> blobGradExp = null;
            Blob<T> blobX_numeric = null;
            Blob<T> blobX_categorical = null;
            Blob<T> blobY = null;
            Blob<T> blobProcessed = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath("ice_hist");
            string strPathWts = getTestWtsPath("hist_ts_transform");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as ChannelEmbeddingLayer<T>;
                blobGradExp = new Blob<T>(m_cuda, m_log);
                blobX_numeric = new Blob<T>(m_cuda, m_log);
                blobX_categorical = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobProcessed = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CHANNEL_EMBEDDING, "The layer type is incorrect.");

                blobX_numeric.LoadFromNumpy(strPath + "x_numeric.npy");
                blobX_categorical.LoadFromNumpy(strPath + "x_categorical.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX_numeric);
                BottomVec.Add(blobX_categorical);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.0.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.0.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.1.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.1.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.2.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.2.bias.npy");
                layer.blobs[6].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.3.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.3.bias.npy");

                layer.blobs[8].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.0.weight.npy");
                layer.blobs[9].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.1.weight.npy");
                layer.blobs[10].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.2.weight.npy");
                layer.blobs[11].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.3.weight.npy");
                layer.blobs[12].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.4.weight.npy");
                layer.blobs[13].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.5.weight.npy");
                layer.blobs[14].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.6.weight.npy");

                layer.Forward(BottomVec, TopVec);

                blobProcessed.LoadFromNumpy(strPath + "hist_processed_input.npy");
                double dfErr = (typeof(T) == typeof(float)) ? 1e-08 : 2e-07;
                m_log.CHECK(TopVec[0].Compare(blobProcessed, blobWork, false, dfErr), "The blobs do not match.");

                TopVec[0].LoadFromNumpy(strPath + "hist_processed_input.grad.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                // Data inputs do not have output grads.

                if (typeof(T) == typeof(double))
                    dfErr = 0.03;

                blobGradExp.LoadFromNumpy(strPath + "hist_proj_layer.0.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[0], blobWork, true, dfErr), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "hist_proj_layer.0.bias.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[1], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "hist_proj_layer.1.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[2], blobWork, true, dfErr), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "hist_proj_layer.1.bias.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[3], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "hist_proj_layer.2.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[4], blobWork, true, dfErr), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "hist_proj_layer.2.bias.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[5], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "hist_proj_layer.3.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[6], blobWork, true, dfErr), "The blobs do not match.");
                blobGradExp.LoadFromNumpy(strPath + "hist_proj_layer.3.bias.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[7], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "hist_emb_layer.0.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[8], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "hist_emb_layer.1.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[9], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "hist_emb_layer.2.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[10], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "hist_emb_layer.3.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[11], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "hist_emb_layer.4.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[12], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "hist_emb_layer.5.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[13], blobWork, true, dfErr), "The blobs do not match.");

                blobGradExp.LoadFromNumpy(strPath + "hist_emb_layer.6.weight.grad.npy", true);
                m_log.CHECK(blobGradExp.Compare(layer.blobs[14], blobWork, true, dfErr), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobGradExp);
                dispose(ref blobProcessed);
                dispose(ref blobWork);
                dispose(ref blobX_numeric);
                dispose(ref blobX_categorical);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }

        /// <summary>
        /// Test ChannelEmbedding(hist) gradient check
        /// </summary>
        /// <remarks>
        /// To generate the test data, run the following:
        /// 
        /// Code: test_1_inputchannelembedding_hist.py
        /// Path: ice_hist
        /// Base: iter_0.base_set
        /// </remarks>
        public void TestGradientHist()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.CHANNEL_EMBEDDING);
            p.numeric_trans_param.num_input = 4;
            p.numeric_trans_param.state_size = 64;
            p.categorical_trans_param.num_input = 7;
            p.categorical_trans_param.cardinalities = new List<int> { 2, 3, 8, 13, 72, 6, 28 };
            p.categorical_trans_param.state_size = 64;
            ChannelEmbeddingLayer<T> layer = null;
            Blob<T> blobX_numeric = null;
            Blob<T> blobX_categorical = null;
            Blob<T> blobY = null;
            Blob<T> blobProcessed = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath("ice_hist");
            string strPathWts = getTestWtsPath("hist_ts_transform");

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null) as ChannelEmbeddingLayer<T>;
                blobX_numeric = new Blob<T>(m_cuda, m_log);
                blobX_categorical = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobProcessed = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.CHANNEL_EMBEDDING, "The layer type is incorrect.");

                blobX_numeric.LoadFromNumpy(strPath + "x_numeric.npy");
                blobX_categorical.LoadFromNumpy(strPath + "x_categorical.npy");
                BottomVec.Clear();
                BottomVec.Add(blobX_numeric);
                BottomVec.Add(blobX_categorical);
                TopVec.Clear();
                TopVec.Add(blobY);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.0.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.0.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.1.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.1.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.2.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.2.bias.npy");
                layer.blobs[6].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.3.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPathWts + "numeric_transform.module.numeric_projection_layers.3.bias.npy");

                layer.blobs[8].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.0.weight.npy");
                layer.blobs[9].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.1.weight.npy");
                layer.blobs[10].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.2.weight.npy");
                layer.blobs[11].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.3.weight.npy");
                layer.blobs[12].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.4.weight.npy");
                layer.blobs[13].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.5.weight.npy");
                layer.blobs[14].LoadFromNumpy(strPathWts + "categorical_transform.module.categorical_embedding_layers.6.weight.npy");

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradient(layer, BottomVec, TopVec, -1, 1, 0.5);
            }
            finally
            {
                dispose(ref blobProcessed);
                dispose(ref blobWork);
                dispose(ref blobX_numeric);
                dispose(ref blobX_categorical);
                dispose(ref blobY);

                if (layer != null)
                    layer.Dispose();
            }
        }
    }
}
