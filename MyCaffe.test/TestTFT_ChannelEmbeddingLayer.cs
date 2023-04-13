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
        public void TestForward()
        {
            ChannelEmbeddingLayerTest test = new ChannelEmbeddingLayerTest();

            try
            {
                foreach (IChannelEmbeddingLayerTest t in test.Tests)
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
        public void TestBackward()
        {
            ChannelEmbeddingLayerTest test = new ChannelEmbeddingLayerTest();

            try
            {
                foreach (IChannelEmbeddingLayerTest t in test.Tests)
                {
                    t.TestBackward();
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
            ChannelEmbeddingLayerTest test = new ChannelEmbeddingLayerTest();

            try
            {
                foreach (IChannelEmbeddingLayerTest t in test.Tests)
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

    interface IChannelEmbeddingLayerTest : ITest
    {
        void TestForward();
        void TestBackward();
        void TestGradient();
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

        private string getTestDataPath()
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\iter_0\\";
        }

        private string getTestWtsPath()
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\hist_ts_transform\\";
        }

        public void TestForward()
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
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

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
                double dfErr = (typeof(T) == typeof(float)) ? 1e-08 : 2.5e-07;
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

        public void TestBackward()
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
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

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
                double dfErr = (typeof(T) == typeof(float)) ? 1e-08 : 2.5e-07;
                m_log.CHECK(TopVec[0].Compare(blobProcessed, blobWork, false, dfErr), "The blobs do not match.");

                TopVec[0].LoadFromNumpy(strPath + "hist_processed_input.grad.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                //blobGradExp.LoadFromNumpy(strPath + "grad_x_categorical.npy", true);
                //m_log.CHECK(blobGradExp.Compare(blobX, blobWork, true, dfErr), "The blobs do not match.");

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

        public void TestGradient()
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
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

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
