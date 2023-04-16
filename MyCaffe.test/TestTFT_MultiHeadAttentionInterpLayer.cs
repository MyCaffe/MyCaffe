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

/// <summary>
/// Testing the MultiHeadAttentionInterp layer.
/// 
/// MultiHeadAttentionInterp Layer - layer calculate multi-headed interpretable attention.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_MultiHeadAttentionInterpLayer
    {
        [TestMethod]
        public void TestForward()
        {
            MultiHeadAttentionInterpLayerTest test = new MultiHeadAttentionInterpLayerTest();

            try
            {
                foreach (IMultiHeadAttentionInterpLayerTest t in test.Tests)
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
            MultiHeadAttentionInterpLayerTest test = new MultiHeadAttentionInterpLayerTest();

            try
            {
                foreach (IMultiHeadAttentionInterpLayerTest t in test.Tests)
                {
                    t.TestBackward();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IMultiHeadAttentionInterpLayerTest : ITest
    {
        void TestForward();
        void TestBackward();
    }

    class MultiHeadAttentionInterpLayerTest : TestBase
    {
        public MultiHeadAttentionInterpLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("MultiHeadAttentionInterp Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new MultiHeadAttentionInterpLayerTest<double>(strName, nDeviceID, engine);
            else
                return new MultiHeadAttentionInterpLayerTest<float>(strName, nDeviceID, engine);
        }
    }

    class MultiHeadAttentionInterpLayerTest<T> : TestEx<T>, IMultiHeadAttentionInterpLayerTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();

        public MultiHeadAttentionInterpLayerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION_INTERP);
            p.multihead_attention_interp_param.embed_dim = 64;
            p.multihead_attention_interp_param.num_heads = 4;
            Layer<T> layer = null;
            Blob<T> blobQ = null;
            Blob<T> blobK = null;
            Blob<T> blobV = null;
            Blob<T> blobMask = null;
            Blob<T> blobY = null;
            Blob<T> blobAttnOut = null;
            Blob<T> blobAttnScores = null;
            Blob<T> blobExp = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobQ = new Blob<T>(m_cuda, m_log);
                blobK = new Blob<T>(m_cuda, m_log);
                blobV = new Blob<T>(m_cuda, m_log);
                blobMask = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobAttnOut = new Blob<T>(m_cuda, m_log);
                blobAttnScores = new Blob<T>(m_cuda, m_log);
                blobExp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION_INTERP, "The layer type is incorrect.");

                blobQ.LoadFromNumpy(strPath + "test_imha_q.npy");
                blobK.LoadFromNumpy(strPath + "test_imha_k.npy");
                blobV.LoadFromNumpy(strPath + "test_imha_v.npy");
                blobMask.LoadFromNumpy(strPath + "test_imha_mask.npy");
                BottomVec.Clear();
                BottomVec.Add(blobQ);
                BottomVec.Add(blobK);
                BottomVec.Add(blobV);
                BottomVec.Add(blobMask);
                TopVec.Clear();
                TopVec.Add(blobY);
                TopVec.Add(blobAttnOut);
                TopVec.Add(blobAttnScores);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "test_imha.w_q.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "test_imha.w_q.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "test_imha.w_k.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "test_imha.w_k.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "test_imha.w_v.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "test_imha.w_v.bias.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "test_imha.out.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "test_imha.out.bias.npy");

                layer.Forward(BottomVec, TopVec);

                blobExp.LoadFromNumpy(strPath + "test_imha_output.npy");
                m_log.CHECK(TopVec[0].Compare(blobExp, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs do not match.");
                blobExp.LoadFromNumpy(strPath + "test_imha_attention_outputs.npy");
                m_log.CHECK(TopVec[1].Compare(blobExp, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 7e-07), "The blobs do not match.");
                blobExp.LoadFromNumpy(strPath + "test_imha_attention_scores.npy");
                m_log.CHECK(TopVec[2].Compare(blobExp, blobWork), "The blobs do not match.");
            }
            finally
            {
                dispose(ref blobQ);
                dispose(ref blobK);
                dispose(ref blobV);
                dispose(ref blobMask);
                dispose(ref blobWork);
                dispose(ref blobY);
                dispose(ref blobAttnOut);
                dispose(ref blobAttnScores);
                dispose(ref blobExp);

                if (layer != null)
                    layer.Dispose();
            }
        }

        public void TestBackward()
        {
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION_INTERP);
            p.multihead_attention_interp_param.embed_dim = 64;
            p.multihead_attention_interp_param.num_heads = 4;
            Layer<T> layer = null;
            Blob<T> blobQ = null;
            Blob<T> blobK = null;
            Blob<T> blobV = null;
            Blob<T> blobMask = null;
            Blob<T> blobY = null;
            Blob<T> blobAttnOut = null;
            Blob<T> blobAttnScores = null;
            Blob<T> blobExp = null;
            Blob<T> blobWork = null;
            string strPath = getTestDataPath();
            string strPathWts = getTestWtsPath();

            try
            {
                layer = Layer<T>.Create(m_cuda, m_log, p, null);
                blobQ = new Blob<T>(m_cuda, m_log);
                blobK = new Blob<T>(m_cuda, m_log);
                blobV = new Blob<T>(m_cuda, m_log);
                blobMask = new Blob<T>(m_cuda, m_log);
                blobY = new Blob<T>(m_cuda, m_log);
                blobAttnOut = new Blob<T>(m_cuda, m_log);
                blobAttnScores = new Blob<T>(m_cuda, m_log);
                blobExp = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                m_log.CHECK(layer != null, "The layer was not created correctly.");
                m_log.CHECK(layer.type == LayerParameter.LayerType.MULTIHEAD_ATTENTION_INTERP, "The layer type is incorrect.");

                blobQ.LoadFromNumpy(strPath + "test_imha_q.npy");
                blobK.LoadFromNumpy(strPath + "test_imha_k.npy");
                blobV.LoadFromNumpy(strPath + "test_imha_v.npy");
                blobMask.LoadFromNumpy(strPath + "test_imha_mask.npy");
                BottomVec.Clear();
                BottomVec.Add(blobQ);
                BottomVec.Add(blobK);
                BottomVec.Add(blobV);
                BottomVec.Add(blobMask);
                TopVec.Clear();
                TopVec.Add(blobY);
                TopVec.Add(blobAttnOut);
                TopVec.Add(blobAttnScores);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strPath + "test_imha.w_q.weight.npy");
                layer.blobs[1].LoadFromNumpy(strPath + "test_imha.w_q.bias.npy");
                layer.blobs[2].LoadFromNumpy(strPath + "test_imha.w_k.weight.npy");
                layer.blobs[3].LoadFromNumpy(strPath + "test_imha.w_k.bias.npy");
                layer.blobs[4].LoadFromNumpy(strPath + "test_imha.w_v.weight.npy");
                layer.blobs[5].LoadFromNumpy(strPath + "test_imha.w_v.bias.npy");
                layer.blobs[6].LoadFromNumpy(strPath + "test_imha.out.weight.npy");
                layer.blobs[7].LoadFromNumpy(strPath + "test_imha.out.bias.npy");

                layer.Forward(BottomVec, TopVec);

                blobExp.LoadFromNumpy(strPath + "test_imha_output.npy");
                m_log.CHECK(TopVec[0].Compare(blobExp, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs do not match.");
                blobExp.LoadFromNumpy(strPath + "test_imha_attention_outputs.npy");
                m_log.CHECK(TopVec[1].Compare(blobExp, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 7e-07), "The blobs do not match.");
                blobExp.LoadFromNumpy(strPath + "test_imha_attention_scores.npy");
                m_log.CHECK(TopVec[2].Compare(blobExp, blobWork), "The blobs do not match.");

                TopVec[0].LoadFromNumpy(strPath + "test_imha_y.grad.npy", true);

                layer.Backward(TopVec, new List<bool>() { true }, BottomVec);

                blobExp.LoadFromNumpy(strPath + "test_imha_q.grad.npy", true);
                m_log.CHECK(blobExp.Compare(blobQ, blobWork, true, (typeof(T) == typeof(float)) ? 1e-08 : 2e-07), "The blobs do not match.");
                blobExp.LoadFromNumpy(strPath + "test_imha_k.grad.npy", true);
                m_log.CHECK(blobExp.Compare(blobK, blobWork, true, (typeof(T) == typeof(float)) ? 1e-08 : 3e-07), "The blobs do not match.");
                blobExp.LoadFromNumpy(strPath + "test_imha_v.grad.npy", true);
                m_log.CHECK(blobExp.Compare(blobV, blobWork, true, (typeof(T) == typeof(float)) ? 1e-08 : 4e-07), "The blobs do not match.");

                if (typeof(T) == typeof(float))
                {
                    blobExp.LoadFromNumpy(strPath + "test_imha.w_q.weight.grad.npy", true);
                    m_log.CHECK(blobExp.Compare(layer.blobs[0], blobWork, true), "The blobs do not match.");
                    blobExp.LoadFromNumpy(strPath + "test_imha.w_q.bias.grad.npy", true);
                    m_log.CHECK(blobExp.Compare(layer.blobs[1], blobWork, true, 7e-05), "The blobs do not match.");

                    blobExp.LoadFromNumpy(strPath + "test_imha.w_k.weight.grad.npy", true);
                    m_log.CHECK(blobExp.Compare(layer.blobs[2], blobWork, true), "The blobs do not match.");
                    blobExp.LoadFromNumpy(strPath + "test_imha.w_k.bias.grad.npy", true);
                    m_log.CHECK(blobExp.Compare(layer.blobs[3], blobWork, true, 7e-05), "The blobs do not match.");

                    blobExp.LoadFromNumpy(strPath + "test_imha.w_v.weight.grad.npy", true);
                    m_log.CHECK(blobExp.Compare(layer.blobs[4], blobWork, true), "The blobs do not match.");
                    blobExp.LoadFromNumpy(strPath + "test_imha.w_v.bias.grad.npy", true);
                    m_log.CHECK(blobExp.Compare(layer.blobs[5], blobWork, true, 2e-03), "The blobs do not match.");

                    blobExp.LoadFromNumpy(strPath + "test_imha.out.weight.grad.npy", true);
                    m_log.CHECK(blobExp.Compare(layer.blobs[6], blobWork, true), "The blobs do not match.");
                    blobExp.LoadFromNumpy(strPath + "test_imha.out.bias.grad.npy", true);
                    m_log.CHECK(blobExp.Compare(layer.blobs[7], blobWork, true), "The blobs do not match.");
                }
            }
            finally
            {
                dispose(ref blobQ);
                dispose(ref blobK);
                dispose(ref blobV);
                dispose(ref blobMask);
                dispose(ref blobWork);
                dispose(ref blobY);
                dispose(ref blobAttnOut);
                dispose(ref blobAttnScores);
                dispose(ref blobExp);

                if (layer != null)
                    layer.Dispose();
            }
        }
    }
}
