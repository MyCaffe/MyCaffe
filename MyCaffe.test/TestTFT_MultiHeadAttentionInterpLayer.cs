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

        [TestMethod]
        public void TestForwardFocused()
        {
            MultiHeadAttentionInterpLayerTest test = new MultiHeadAttentionInterpLayerTest();

            try
            {
                foreach (IMultiHeadAttentionInterpLayerTest t in test.Tests)
                {
                    t.TestForwardFocused();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardFocused()
        {
            MultiHeadAttentionInterpLayerTest test = new MultiHeadAttentionInterpLayerTest();

            try
            {
                foreach (IMultiHeadAttentionInterpLayerTest t in test.Tests)
                {
                    t.TestBackwardFocused();
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
        void TestForwardFocused();
        void TestBackwardFocused();
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

        private string getTestBaseDataPath(int nIter = 0)
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\iter_" + nIter.ToString() + ".base_set\\";
        }

        private string getTestDataPath(string strSubPath)
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\" + strSubPath + "\\iter_0\\";
        }

        private string getTestWtsPath()
        {
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\multihead_attn\\";
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
            string strPath = getTestDataPath("imha");
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
            string strPath = getTestDataPath("imha");
            string strPathWts = getTestDataPath("imha") + "weights\\";

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

        private string buildModel(bool bAddDataLayer, int nNumSamples, int nNumHeads, float fDropout, int nLstmLayers, int nNumOutputs, int nStateSize, int nNumHistSteps, int nNumFutureSteps,
            int nNumStaticNumeric, int nNumStaticCategorical, List<int> rgStaticCardinalities,
            int nNumHistNumeric, int nNumHistCategorical, List<int> rgHistCardinalities,
            int nNumFutureNumeric, int nNumFutureCategorical, List<int> rgFutureCardinalities,
            bool bEnableLayerNormPassthrough)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";

            LayerParameter data = new LayerParameter(LayerParameter.LayerType.INPUT);
            data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHistSteps + nNumFutureSteps, nStateSize }));
            data.top.Add("enriched_sequence");
            p.layer.Add(data);


            //---------------------------------
            //  Temporal Self-attention
            //---------------------------------
            LayerParameter statenr_split = new LayerParameter(LayerParameter.LayerType.SPLIT, "statenr_split");
            statenr_split.bottom.Add("enriched_sequence");
            statenr_split.top.Add("enr_seq_a");
            statenr_split.top.Add("enr_seq_b");
            p.layer.Add(statenr_split);

            LayerParameter multihead_attn = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION_INTERP, "mh_attn");
            multihead_attn.multihead_attention_interp_param.embed_dim = (uint)nStateSize;
            multihead_attn.multihead_attention_interp_param.num_heads = (uint)nNumHeads;
            multihead_attn.multihead_attention_interp_param.num_historical_steps = (uint)nNumHistSteps;
            multihead_attn.multihead_attention_interp_param.num_future_steps = (uint)nNumFutureSteps;
            multihead_attn.bottom.Add("enr_seq_a");
            multihead_attn.top.Add("post_attention");
            multihead_attn.top.Add("attention_outputs");
            multihead_attn.top.Add("attention_scores");
            p.layer.Add(multihead_attn);

            LayerParameter post_attn_gate = new LayerParameter(LayerParameter.LayerType.GATEADDNORM, "post_attn_gate");
            post_attn_gate.gateaddnorm_param.residual_channel_offset = nNumHistSteps;
            post_attn_gate.dropout_param.dropout_ratio = fDropout;
            post_attn_gate.layer_norm_param.enable_cuda_impl = false;
            post_attn_gate.layer_norm_param.enable_passthrough = bEnableLayerNormPassthrough;
            post_attn_gate.glu_param.input_dim = nStateSize;
            post_attn_gate.glu_param.axis = 2;
            post_attn_gate.bottom.Add("post_attention");
            post_attn_gate.bottom.Add("enr_seq_b");
            post_attn_gate.top.Add("gated_post_attention");
            p.layer.Add(post_attn_gate);

            LayerParameter silence1 = new LayerParameter(LayerParameter.LayerType.SILENCE);
            silence1.bottom.Add("attention_outputs");
            p.layer.Add(silence1);

            return p.ToProto("root").ToString();
        }

        private void load_weights(string strTag, Net<T> net, string strPath, int nNumStaticNumeric, int nNumStaticCategorical, int nNumHistNumeric, int nNumHistCategorical, int nNumFutureNumeric, int nNumFutureCategorical)
        {
            int nIdx = 0;

            //---------------------------------
            //  Temporal Self-attention (idx=336)
            //---------------------------------
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".multihead_attn.w_q.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".multihead_attn.w_q.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".multihead_attn.w_k.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".multihead_attn.w_k.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".multihead_attn.w_v.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".multihead_attn.w_v.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".multihead_attn.out.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".multihead_attn.out.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".post_attention_gating_attn.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".post_attention_gating_attn.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".post_attention_gating_attn.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".post_attention_gating_attn.gate.module.fc2.bias.npy");
            nIdx++;
        }

        private void compare(double dfTol, List<string> rgstrErrors, int nIdx, Blob<T> blobVal, Blob<T> blob, Blob<T> blobWork, string strFile)
        {
            double dfMin;
            double dfMax;

            blobVal.LoadFromNumpy(strFile, true);
            bool bCompare = blobVal.CompareEx(blob, blobWork, out dfMin, out dfMax, true, dfTol);

            if (!bCompare)
            {
                double dfDiff = Math.Max(Math.Abs(dfMin), Math.Abs(dfMax));
                rgstrErrors.Add(nIdx.ToString() + ".)   " + dfDiff.ToString() + "   " + blob.Name);
                m_log.WriteLine("WARNING: The blobs do not match for blob '" + blob.Name + "!");
            }
        }

        private void compare_weights(string strTag, Net<T> net, string strPath, int nNumStaticNumeric, int nNumStaticCategorical, int nNumHistNumeric, int nNumHistCategorical, int nNumFutureNumeric, int nNumFutureCategorical)
        {
            double dfTol = 1e-05;
            Blob<T> blobVal = new Blob<T>(net.Cuda, m_log);
            Blob<T> blobWork = new Blob<T>(net.Cuda, m_log);
            List<string> rgErrors = new List<string>();

            try
            {
                int nIdx = 12;

                //---------------------------------
                //  *Temporal Self-attention (idx=336)
                //---------------------------------
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + ".post_attention_gating_attn.gate.module.fc2.bias.grad.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + ".post_attention_gating_attn.gate.module.fc2.weight.grad.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + ".post_attention_gating_attn.gate.module.fc1.bias.grad.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + ".post_attention_gating_attn.gate.module.fc1.weight.grad.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + ".multihead_attn.out.bias.grad.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + ".multihead_attn.out.weight.grad.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + ".multihead_attn.w_v.bias.grad.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + ".multihead_attn.w_v.weight.grad.npy");
                nIdx--;
                /*BUG->*/
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + ".multihead_attn.w_k.bias.grad.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + ".multihead_attn.w_k.weight.grad.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + ".multihead_attn.w_q.bias.grad.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + ".multihead_attn.w_q.weight.grad.npy");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);
            }
        }

        public void TestForwardFocused()
        {
            string strPathBase = getTestBaseDataPath();
            string strPath = getTestDataPath("imha");
            string strPathWt = strPath + "weights\\";
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nNumHeads = 4;
            float fDropout = 0;
            int nLstmLayers = 2;
            int nNumOutputs = 3;
            int nStateSize = 64;
            int nNumHistSteps = 90;
            int nNumFutureSteps = 30;
            int nNumStaticNumeric = 0;
            int nNumStaticCategorical = 9;
            List<int> rgStaticCardinalities = new List<int>() { 54, 3627, 23, 17, 6, 18, 33, 320, 3 };
            int nNumHistNumeric = 4;
            int nNumHistCategorical = 7;
            List<int> rgHistCardinalities = new List<int>() { 2, 3, 8, 13, 72, 6, 28 };
            int nNumFutureNumeric = 1;
            int nNumFutureCategorical = 7;
            List<int> rgFutureCardinalities = new List<int>() { 2, 3, 8, 13, 72, 6, 28 };
            string strTag = "tft.test";

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                bool bEnableLayerNormPassthrough = true;
                string strModel = buildModel(false, nNumSamples, nNumHeads, fDropout, nLstmLayers, nNumOutputs, nStateSize, nNumHistSteps, nNumFutureSteps, nNumStaticNumeric, nNumStaticCategorical, rgStaticCardinalities, nNumHistNumeric, nNumHistCategorical, rgHistCardinalities, nNumFutureNumeric, nNumFutureCategorical, rgFutureCardinalities, bEnableLayerNormPassthrough);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, null, null, Phase.TRAIN);

                load_weights(strTag, net, strPathWt, nNumStaticNumeric, nNumStaticCategorical, nNumHistNumeric, nNumHistCategorical, nNumFutureNumeric, nNumFutureCategorical);

                // inputs
                net.FindBlob("enriched_sequence").LoadFromNumpy(strPath + "tft.asa.enriched_sequence.npy");

                BlobCollection<T> colRes = net.Forward();

                // Transform all input channels
                blobVal.LoadFromNumpy(strPath + "tft.ada.gated_post_attention.npy");
                blob1 = net.FindBlob("gated_post_attention");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 1e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.ada.attention_scores.npy");
                blob1 = net.FindBlob("attention_scores");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.asa.attention_scores.npy");
                blob1 = net.FindBlob("attention_scores");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.asa.attention_outputs.npy");
                blob1 = net.FindBlob("attention_outputs");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }

        public void TestBackwardFocused()
        {
            string strPathBase = getTestBaseDataPath();
            string strPath = getTestDataPath("imha");
            string strPathWt = strPath + "weights\\";
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nNumHeads = 4;
            float fDropout = 0;
            int nLstmLayers = 2;
            int nNumOutputs = 3;
            int nStateSize = 64;
            int nNumHistSteps = 90;
            int nNumFutureSteps = 30;
            int nNumStaticNumeric = 0;
            int nNumStaticCategorical = 9;
            List<int> rgStaticCardinalities = new List<int>() { 54, 3627, 23, 17, 6, 18, 33, 320, 3 };
            int nNumHistNumeric = 4;
            int nNumHistCategorical = 7;
            List<int> rgHistCardinalities = new List<int>() { 2, 3, 8, 13, 72, 6, 28 };
            int nNumFutureNumeric = 1;
            int nNumFutureCategorical = 7;
            List<int> rgFutureCardinalities = new List<int>() { 2, 3, 8, 13, 72, 6, 28 };
            string strTag = "tft.test";

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                bool bEnableLayerNormPassthrough = true;
                string strModel = buildModel(false, nNumSamples, nNumHeads, fDropout, nLstmLayers, nNumOutputs, nStateSize, nNumHistSteps, nNumFutureSteps, nNumStaticNumeric, nNumStaticCategorical, rgStaticCardinalities, nNumHistNumeric, nNumHistCategorical, rgHistCardinalities, nNumFutureNumeric, nNumFutureCategorical, rgFutureCardinalities, bEnableLayerNormPassthrough);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);
                param.force_backward = true;

                net = new Net<T>(m_cuda, m_log, param, null, null, Phase.TRAIN);

                load_weights(strTag, net, strPathWt, nNumStaticNumeric, nNumStaticCategorical, nNumHistNumeric, nNumHistCategorical, nNumFutureNumeric, nNumFutureCategorical);

                // inputs
                net.FindBlob("enriched_sequence").LoadFromNumpy(strPath + "tft.asa.enriched_sequence.npy");

                BlobCollection<T> colRes = net.Forward();

                // Transform all input channels
                blobVal.LoadFromNumpy(strPath + "tft.asa.gated_post_attention.npy");
                blob1 = net.FindBlob("gated_post_attention");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.ada.attention_scores.npy");
                blob1 = net.FindBlob("attention_scores");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.asa.attention_scores.npy");
                blob1 = net.FindBlob("attention_scores");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.asa.attention_outputs.npy");
                blob1 = net.FindBlob("attention_outputs");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                //*** BACKWARD ***

                blob1 = net.FindBlob("gated_post_attention");
                blob1.LoadFromNumpy(strPath + "tft.ada.gated_post_attention.grad.npy", true);

                net.Backward();

                blobVal.LoadFromNumpy(strPath + "tft.asa.enriched_sequence.grad.npy", true);
                blob1 = net.FindBlob("enriched_sequence");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, 4e-07), "The blobs are different!");

                compare_weights(strTag, net, strPathWt, nNumStaticNumeric, nNumStaticCategorical, nNumHistNumeric, nNumHistCategorical, nNumFutureNumeric, nNumFutureCategorical);
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }
    }
}
