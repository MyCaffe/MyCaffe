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
/// Testing the SelfAttention.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_SelfAttention
    {
        [TestMethod]
        public void TestForward()
        {
            SelfAttentionTest test = new SelfAttentionTest();

            try
            {
                foreach (ISelfAttentionTest t in test.Tests)
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
            SelfAttentionTest test = new SelfAttentionTest();

            try
            {
                foreach (ISelfAttentionTest t in test.Tests)
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

    interface ISelfAttentionTest : ITest
    {
        void TestForward();
        void TestBackward();
    }

    class SelfAttentionTest : TestBase
    {
        public SelfAttentionTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TFT SelfAttention Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SelfAttentionTest<double>(strName, nDeviceID, engine);
            else
                return new SelfAttentionTest<float>(strName, nDeviceID, engine);
        }
    }

    class SelfAttentionTest<T> : TestEx<T>, ISelfAttentionTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();

        public SelfAttentionTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
            return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\static_enrichment_grn\\";
        }

        private string buildModel(int nNumSamples, int nNumHist, int nNumFuture, float fDropout, int nStateSize, int nNumHeads)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";


            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHist + nNumFuture, nStateSize }));  // enriched_sequence
            input.top.Add("enriched_sequence");
            p.layer.Add(input);

            //---------------------------------
            //  Temporal Self-attention
            //---------------------------------
            LayerParameter multihead_attn = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION_INTERP, "mh_attn");
            multihead_attn.multihead_attention_interp_param.embed_dim = nStateSize;
            multihead_attn.multihead_attention_interp_param.num_heads = nNumHeads;
            multihead_attn.multihead_attention_interp_param.num_historical_steps = nNumHist;
            multihead_attn.multihead_attention_interp_param.num_future_steps = nNumFuture;
            multihead_attn.bottom.Add("enriched_sequence");
            multihead_attn.top.Add("post_attention");
            multihead_attn.top.Add("attention_outputs");
            multihead_attn.top.Add("attention_scores");
            multihead_attn.top.Add("enriched_sequence1");
            p.layer.Add(multihead_attn);

            LayerParameter post_attn_gate = new LayerParameter(LayerParameter.LayerType.GATEADDNORM, "post_attn_gate");
            post_attn_gate.gateaddnorm_param.residual_channel_offset = nNumHist;
            post_attn_gate.dropout_param.dropout_ratio = fDropout;
            post_attn_gate.layer_norm_param.enable_cuda_impl = false;
            post_attn_gate.glu_param.input_dim = nStateSize;
            post_attn_gate.glu_param.axis = 1;
            post_attn_gate.bottom.Add("post_attention");
            post_attn_gate.bottom.Add("enriched_sequence1");
            post_attn_gate.top.Add("gated_post_attention");
            p.layer.Add(post_attn_gate);

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Test the forward pass for self attention
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run test_8_interpmultiheadattn_hist_focused.py on fresh 'test\iter_0' data
        /// 
        /// Fresh test\iter_0 data generated by running:
        /// training.py with TemporalFusionTransformer options: debug=True, tag='tft', use_mycaffe=True
        /// </remarks>
        public void TestForward()
        {
            string strPath = getTestDataPath();
            string strPathWt = getTestWtsPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            float fDropout = 0;
            int nStateSize = 64;
            int nNumSamples = 256;
            int nNumHist = 90;
            int nNumFuture = 30;
            int nNumHeads = 4;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(nNumSamples, nNumHist, nNumFuture, fDropout, nStateSize, nNumHeads);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, null, null);

                blob1 = net.FindBlob("enriched_sequence");
                blob1.LoadFromNumpy(strPath + "tft.ada.enriched_sequence.npy");

                net.parameters[0].LoadFromNumpy(strPath + "tft.test.multihead_attn.w_q.weight.npy");
                net.parameters[1].LoadFromNumpy(strPath + "tft.test.multihead_attn.w_q.bias.npy");
                net.parameters[2].LoadFromNumpy(strPath + "tft.test.multihead_attn.w_k.weight.npy");
                net.parameters[3].LoadFromNumpy(strPath + "tft.test.multihead_attn.w_k.bias.npy");
                net.parameters[4].LoadFromNumpy(strPath + "tft.test.multihead_attn.w_v.weight.npy");
                net.parameters[5].LoadFromNumpy(strPath + "tft.test.multihead_attn.w_v.bias.npy");
                net.parameters[6].LoadFromNumpy(strPath + "tft.test.multihead_attn.out.weight.npy");
                net.parameters[7].LoadFromNumpy(strPath + "tft.test.multihead_attn.out.bias.npy");
                net.parameters[8].LoadFromNumpy(strPath + "tft.test.post_attention_gating_attn.gate.module.fc1.weight.npy");
                net.parameters[9].LoadFromNumpy(strPath + "tft.test.post_attention_gating_attn.gate.module.fc1.bias.npy");
                net.parameters[10].LoadFromNumpy(strPath + "tft.test.post_attention_gating_attn.gate.module.fc2.weight.npy");
                net.parameters[11].LoadFromNumpy(strPath + "tft.test.post_attention_gating_attn.gate.module.fc2.bias.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.asa.gated_post_attention.npy");
                blob1 = net.FindBlob("gated_post_attention");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 8e-07 : 1e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.asa.attention_outputs.npy");
                blob1 = net.FindBlob("attention_outputs");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 1e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.asa.attention_scores.npy");
                blob1 = net.FindBlob("attention_scores");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");
            }
            catch (Exception ex)
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }

        /// <summary>
        /// Test the backward pass for self attention
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run test_8_interpmultiheadattn_hist_focused.py on fresh 'test\iter_0' data
        /// 
        /// Fresh test\iter_0 data generated by running:
        /// training.py with TemporalFusionTransformer options: debug=True, tag='tft', use_mycaffe=True
        /// </remarks>
        public void TestBackward()
        {
            string strPath = getTestDataPath();
            string strPathWt = getTestWtsPath();
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            float fDropout = 0;
            int nStateSize = 64;
            int nNumSamples = 256;
            int nNumHist = 90;
            int nNumFuture = 30;
            int nNumHeads = 4;

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(nNumSamples, nNumHist, nNumFuture, fDropout, nStateSize, nNumHeads);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);
                param.force_backward = true;

                net = new Net<T>(m_cuda, m_log, param, null, null);

                blob1 = net.FindBlob("enriched_sequence");
                blob1.LoadFromNumpy(strPath + "tft.ada.enriched_sequence.npy");

                net.parameters[0].LoadFromNumpy(strPath + "tft.test.multihead_attn.w_q.weight.npy");
                net.parameters[1].LoadFromNumpy(strPath + "tft.test.multihead_attn.w_q.bias.npy");
                net.parameters[2].LoadFromNumpy(strPath + "tft.test.multihead_attn.w_k.weight.npy");
                net.parameters[3].LoadFromNumpy(strPath + "tft.test.multihead_attn.w_k.bias.npy");
                net.parameters[4].LoadFromNumpy(strPath + "tft.test.multihead_attn.w_v.weight.npy");
                net.parameters[5].LoadFromNumpy(strPath + "tft.test.multihead_attn.w_v.bias.npy");
                net.parameters[6].LoadFromNumpy(strPath + "tft.test.multihead_attn.out.weight.npy");
                net.parameters[7].LoadFromNumpy(strPath + "tft.test.multihead_attn.out.bias.npy");
                net.parameters[8].LoadFromNumpy(strPath + "tft.test.post_attention_gating_attn.gate.module.fc1.weight.npy");
                net.parameters[9].LoadFromNumpy(strPath + "tft.test.post_attention_gating_attn.gate.module.fc1.bias.npy");
                net.parameters[10].LoadFromNumpy(strPath + "tft.test.post_attention_gating_attn.gate.module.fc2.weight.npy");
                net.parameters[11].LoadFromNumpy(strPath + "tft.test.post_attention_gating_attn.gate.module.fc2.bias.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.asa.gated_post_attention.npy");
                blob1 = net.FindBlob("gated_post_attention");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 8e-07 : 1e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.asa.attention_outputs.npy");
                blob1 = net.FindBlob("attention_outputs");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.asa.attention_scores.npy");
                blob1 = net.FindBlob("attention_scores");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                //*** BACKWARD ***

                blob1 = net.FindBlob("gated_post_attention");
                blob1.LoadFromNumpy(strPath + "tft.ada.gated_post_attention.grad.npy", true);

                net.Backward();

                blobVal.LoadFromNumpy(strPath + "tft.ada.enriched_sequence1.grad.npy", true);
                blob1 = net.FindBlob("enriched_sequence");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The blobs are different!");
            }
            catch (Exception ex)
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }
    }
}
