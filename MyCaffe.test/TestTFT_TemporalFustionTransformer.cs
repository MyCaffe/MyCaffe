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

// WORK IN PROGRESS
/// <summary>
/// Testing the TemporalFusionTransformer network.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_TemporalFusionTransformer
    {
        [TestMethod]
        public void TestTraining()
        {
            TemporalFusionTransformerTest test = new TemporalFusionTransformerTest();

            try
            {
                foreach (ITemporalFusionTransformerTest t in test.Tests)
                {
                    t.TestTraining();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ITemporalFusionTransformerTest : ITest
    {
        void TestTraining();
    }

    class TemporalFusionTransformerTest : TestBase
    {
        public TemporalFusionTransformerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TemporalFusionTransformer Network Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TemporalFusionTransformerTest<double>(strName, nDeviceID, engine);
            else
                return new TemporalFusionTransformerTest<float>(strName, nDeviceID, engine);
        }
    }

    class TemporalFusionTransformerTest<T> : TestEx<T>, ITemporalFusionTransformerTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();

        public TemporalFusionTransformerTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        private string buildModel(int nNumHeads, float fDropout, int nLstmLayers, List<int> rgOutputQuantiles, int nNumOutputs, int nStateSize, 
            int nNumStaticNumeric, int nNumStaticCategorical, List<int> rgStaticCardinalities,
            int nNumHistNumeric, int nNumHistCategorical, List<int> rgHistCardinalities,
            int nNumFutureNumeric, int nNumFutureCategorical, List<int> rgFutureCardinalities)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";

            //---------------------------------
            //  Data Input Layer
            //---------------------------------
            // TODO: Need to Create

            //---------------------------------
            //  Input Transformations
            //---------------------------------
            LayerParameter static_transform = new LayerParameter(LayerParameter.LayerType.CHANNEL_EMBEDDING, "static_trfm");
            static_transform.numeric_trans_param.num_input = (uint)nNumStaticNumeric;
            static_transform.numeric_trans_param.state_size = (uint)nStateSize;
            static_transform.categorical_trans_param.num_input = (uint)nNumStaticCategorical;
            static_transform.categorical_trans_param.cardinalities = rgStaticCardinalities;
            static_transform.categorical_trans_param.state_size = (uint)nStateSize;
            static_transform.bottom.Add("x_numeric_static");
            static_transform.bottom.Add("x_categorical_static");
            static_transform.top.Add("static_rep");
            p.layer.Add(static_transform);

            LayerParameter hist_ts_transform = new LayerParameter(LayerParameter.LayerType.CHANNEL_EMBEDDING, "hist_ts_trfm");
            hist_ts_transform.numeric_trans_param.num_input = (uint)nNumHistNumeric;
            hist_ts_transform.numeric_trans_param.state_size = (uint)nStateSize;
            hist_ts_transform.categorical_trans_param.num_input = (uint)nNumHistCategorical;
            hist_ts_transform.categorical_trans_param.cardinalities = rgHistCardinalities;
            hist_ts_transform.categorical_trans_param.state_size = (uint)nStateSize;
            hist_ts_transform.bottom.Add("x_numeric_hist");
            hist_ts_transform.bottom.Add("x_categorical_hist");
            hist_ts_transform.top.Add("hist_ts_rep");
            p.layer.Add(hist_ts_transform);

            LayerParameter future_ts_transform = new LayerParameter(LayerParameter.LayerType.CHANNEL_EMBEDDING, "future_ts_trfm");
            future_ts_transform.numeric_trans_param.num_input = (uint)nNumFutureNumeric;
            future_ts_transform.numeric_trans_param.state_size = (uint)nStateSize;
            future_ts_transform.categorical_trans_param.num_input = (uint)nNumFutureCategorical;
            future_ts_transform.categorical_trans_param.cardinalities = rgFutureCardinalities;
            future_ts_transform.categorical_trans_param.state_size = (uint)nStateSize;
            future_ts_transform.bottom.Add("x_numeric_hist");
            future_ts_transform.bottom.Add("x_categorical_hist");
            future_ts_transform.top.Add("future_ts_rep");
            p.layer.Add(future_ts_transform);

            //---------------------------------
            //  Variable Selection Networks - Static
            //---------------------------------
            LayerParameter static_vsn = new LayerParameter(LayerParameter.LayerType.VARSELNET, "static_vsn");
            static_vsn.varselnet_param.input_dim = nStateSize;
            static_vsn.varselnet_param.num_inputs = nNumStaticNumeric + nNumStaticCategorical;
            static_vsn.varselnet_param.hidden_dim = nStateSize;
            static_vsn.varselnet_param.dropout = fDropout;
            static_vsn.bottom.Add("static_rep");
            static_vsn.top.Add("selected_static");
            static_vsn.top.Add("static_wts");
            p.layer.Add(static_vsn);

            //---------------------------------
            //  Static covariate encoders
            //---------------------------------
            LayerParameter static_cov_enc = new LayerParameter(LayerParameter.LayerType.GRN, "static_cov_enc");
            static_cov_enc.grn_param.input_dim = nStateSize;
            static_cov_enc.grn_param.hidden_dim = nStateSize;
            static_cov_enc.grn_param.output_dim = nStateSize;
            static_cov_enc.grn_param.dropout = fDropout;

            LayerParameter static_enc_sel = static_cov_enc.Clone(false);
            static_enc_sel.name = "enc_sel";
            static_enc_sel.bottom.Add("selected_static");
            static_enc_sel.top.Add("c_selection");
            p.layer.Add(static_enc_sel);

            LayerParameter static_enc_enrich = static_cov_enc.Clone(false);
            static_enc_enrich.name = "enc_enr";
            static_enc_enrich.bottom.Add("selected_static");
            static_enc_enrich.top.Add("c_enrichment");
            p.layer.Add(static_enc_enrich);

            LayerParameter static_enc_seq_cell_init = static_cov_enc.Clone(false);
            static_enc_seq_cell_init.name = "enc_seq_cell_init";
            static_enc_seq_cell_init.bottom.Add("selected_static");
            static_enc_seq_cell_init.top.Add("c_seq_cell");
            p.layer.Add(static_enc_seq_cell_init);

            LayerParameter static_enc_seq_state_init = static_cov_enc.Clone(false);
            static_enc_seq_state_init.name = "enc_seq_state_init";
            static_enc_seq_state_init.bottom.Add("selected_static");
            static_enc_seq_state_init.top.Add("c_seq_hidden");
            p.layer.Add(static_enc_seq_state_init);

            //---------------------------------
            //  Variable Selection Networks - Temporal
            //---------------------------------
            LayerParameter hist_vsh_reshape_before = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_hist_b");
            hist_vsh_reshape_before.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.BEFORE;
            hist_vsh_reshape_before.bottom.Add("hist_ts_rep");
            hist_vsh_reshape_before.bottom.Add("c_selection");
            hist_vsh_reshape_before.top.Add("hist_ts_rep1");
            hist_vsh_reshape_before.top.Add("c_selection1");

            LayerParameter hist_vsn = new LayerParameter(LayerParameter.LayerType.VARSELNET, "hist_vsn");
            hist_vsn.varselnet_param.input_dim = nStateSize;
            hist_vsn.varselnet_param.num_inputs = nNumHistNumeric + nNumHistCategorical;
            hist_vsn.varselnet_param.hidden_dim = nStateSize;
            hist_vsn.varselnet_param.dropout = fDropout;
            hist_vsn.varselnet_param.context_dim = nStateSize;
            hist_vsn.bottom.Add("hist_ts_rep1");
            hist_vsn.bottom.Add("c_selection1");
            hist_vsn.top.Add("selected_hist1");
            p.layer.Add(hist_vsn);

            LayerParameter hist_vsh_reshape_after = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_hist_a");
            hist_vsh_reshape_after.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.AFTER;
            hist_vsh_reshape_after.bottom.Add("selected_hist1");
            hist_vsh_reshape_after.top.Add("selected_hist");

            LayerParameter future_vsh_reshape_before = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_fut_b");
            future_vsh_reshape_before.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.BEFORE;
            future_vsh_reshape_before.bottom.Add("future_ts_rep");
            future_vsh_reshape_before.top.Add("future_ts_rep1");

            LayerParameter fut_vsn = new LayerParameter(LayerParameter.LayerType.VARSELNET, "future_vsn");
            fut_vsn.varselnet_param.input_dim = nStateSize;
            fut_vsn.varselnet_param.num_inputs = nNumFutureNumeric + nNumFutureCategorical;
            fut_vsn.varselnet_param.hidden_dim = nStateSize;
            fut_vsn.varselnet_param.dropout = fDropout;
            fut_vsn.varselnet_param.context_dim = nStateSize;
            fut_vsn.bottom.Add("future_ts_rep1");
            fut_vsn.bottom.Add("c_selection1");
            fut_vsn.top.Add("selected_fut1");
            p.layer.Add(fut_vsn);

            LayerParameter future_vsh_reshape_after = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_fut_a");
            future_vsh_reshape_after.reshape_temporal_param.mode = param.tft.ReshapeTemporalParameter.MODE.AFTER;
            future_vsh_reshape_after.bottom.Add("selected_fut1");
            future_vsh_reshape_after.top.Add("selected_fut");


            //---------------------------------
            //  Locality Enhancement with Seq2Seq processing
            //---------------------------------
            LayerParameter lstm_input = new LayerParameter(LayerParameter.LayerType.CONCAT, "lstm_input");
            lstm_input.concat_param.axis = 1;
            lstm_input.bottom.Add("selected_hist");
            lstm_input.bottom.Add("selected_fut");
            lstm_input.top.Add("lstm_input");
            p.layer.Add(lstm_input);

            LayerParameter past_lstm = new LayerParameter(LayerParameter.LayerType.LSTM, "past_lstm");
            past_lstm.recurrent_param.num_output = (uint)nStateSize;
            past_lstm.recurrent_param.num_layers = (uint)nLstmLayers;
            past_lstm.recurrent_param.dropout_ratio = fDropout;
            past_lstm.recurrent_param.engine = EngineParameter.Engine.CUDNN;
            past_lstm.bottom.Add("selected_hist");
            past_lstm.bottom.Add("c_seq_hidden");
            past_lstm.bottom.Add("c_seq_cell");
            past_lstm.top.Add("past_lstm_output");
            past_lstm.top.Add("hidden");
            p.layer.Add(past_lstm);

            LayerParameter future_lstm = new LayerParameter(LayerParameter.LayerType.LSTM, "future_lstm");
            future_lstm.recurrent_param.num_output = (uint)nStateSize;
            future_lstm.recurrent_param.num_layers = (uint)nLstmLayers;
            future_lstm.recurrent_param.dropout_ratio = fDropout;
            future_lstm.recurrent_param.engine = EngineParameter.Engine.CUDNN;
            future_lstm.bottom.Add("selected_fut");
            future_lstm.bottom.Add("hidden");
            future_lstm.top.Add("future_lstm_output");
            p.layer.Add(future_lstm);

            LayerParameter lstm_output = new LayerParameter(LayerParameter.LayerType.CONCAT, "lstm_output");
            lstm_output.concat_param.axis = 1;
            lstm_output.bottom.Add("past_lstm_output");
            lstm_output.bottom.Add("future_lstm_output");
            lstm_output.top.Add("lstm_output");
            p.layer.Add(lstm_output);

            LayerParameter post_lstm_gating = new LayerParameter(LayerParameter.LayerType.GATEADDNORM, "post_lstm_gate");
            post_lstm_gating.dropout_param.dropout_ratio = fDropout;
            post_lstm_gating.layer_norm_param.enable_cuda_impl = false;
            post_lstm_gating.glu_param.input_dim = nStateSize;
            post_lstm_gating.glu_param.axis = 1;
            post_lstm_gating.bottom.Add("lstm_output");
            post_lstm_gating.bottom.Add("lstm_input");
            post_lstm_gating.top.Add("gated_lstm_output");
            p.layer.Add(post_lstm_gating);

            //---------------------------------
            //  Static enrichment
            //---------------------------------
            LayerParameter static_enrich_grn = new LayerParameter(LayerParameter.LayerType.GRN, "static_enrich_gru");
            static_enrich_grn.grn_param.input_dim = nStateSize;
            static_enrich_grn.grn_param.hidden_dim = nStateSize;
            static_enrich_grn.grn_param.output_dim = nStateSize;
            static_enrich_grn.grn_param.context_dim = nStateSize;
            static_enrich_grn.grn_param.dropout = fDropout;
            static_enrich_grn.bottom.Add("gated_lstm_output");
            static_enrich_grn.bottom.Add("c_enrichment");
            static_enrich_grn.top.Add("enriched_sequence");
            p.layer.Add(static_enrich_grn);

            //---------------------------------
            //  Temporal Self-attention
            //---------------------------------
            LayerParameter multihead_attn = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION_INTERP, "mh_attn");
            multihead_attn.multihead_attention_interp_param.embed_dim = nStateSize;
            multihead_attn.multihead_attention_interp_param.num_heads = nNumHeads;
            multihead_attn.bottom.Add("enriched_sequence");
            multihead_attn.top.Add("post_attention");
            p.layer.Add(multihead_attn);

            LayerParameter post_attn_gate = new LayerParameter(LayerParameter.LayerType.GATEADDNORM, "post_attn_gate");
            post_attn_gate.dropout_param.dropout_ratio = fDropout;
            post_attn_gate.layer_norm_param.enable_cuda_impl = false;
            post_attn_gate.glu_param.input_dim = nStateSize;
            post_attn_gate.glu_param.axis = 1;
            post_attn_gate.bottom.Add("post_attention");
            post_attn_gate.bottom.Add("enriched_sequence");
            post_attn_gate.top.Add("gated_post_attention");
            post_attn_gate.top.Add("attention_outputs");
            post_attn_gate.top.Add("attention_scores");
            p.layer.Add(post_attn_gate);

            LayerParameter silence1 = new LayerParameter(LayerParameter.LayerType.SILENCE);
            silence1.bottom.Add("attention_outputs");
            p.layer.Add(silence1);

            //---------------------------------
            //  Position-wise feed forward
            //---------------------------------
            LayerParameter pos_wise_ff_grn = new LayerParameter(LayerParameter.LayerType.GRN, "pos_wise_ff_grn");
            pos_wise_ff_grn.grn_param.input_dim = nStateSize;
            pos_wise_ff_grn.grn_param.hidden_dim = nStateSize;
            pos_wise_ff_grn.grn_param.output_dim = nStateSize;
            pos_wise_ff_grn.grn_param.context_dim = nStateSize;
            pos_wise_ff_grn.grn_param.dropout = fDropout;
            pos_wise_ff_grn.bottom.Add("gated_post_attention");
            pos_wise_ff_grn.top.Add("post_poswise_ff_grn");
            p.layer.Add(pos_wise_ff_grn);

            LayerParameter pos_wise_ff_gate = new LayerParameter(LayerParameter.LayerType.GATEADDNORM, "pos_wise_ff_gate");
            pos_wise_ff_gate.dropout_param.dropout_ratio = fDropout;
            pos_wise_ff_gate.layer_norm_param.enable_cuda_impl = false;
            pos_wise_ff_gate.glu_param.input_dim = nStateSize;
            pos_wise_ff_gate.glu_param.axis = 1;
            pos_wise_ff_gate.bottom.Add("post_poswise_ff_grn");
            pos_wise_ff_gate.bottom.Add("gated_lstm_output");
            pos_wise_ff_gate.top.Add("gated_poswise_ff");
            p.layer.Add(pos_wise_ff_gate);

            //---------------------------------
            //  Output layer
            //---------------------------------
            LayerParameter output = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "output");
            output.inner_product_param.num_output = (uint)nNumOutputs;
            output.bottom.Add("gated_poswise_ff");
            output.top.Add("predicted_quantiles");
            p.layer.Add(output);

            return p.ToProto("root").ToString();
        }

        // WORK IN PROGRESS
        public void TestTraining()
        {
            Net<T> net = null;

            try
            {
                //string strModel = buildModel();
            }
            catch (Exception ex)
            {
                if (net != null)
                    net.Dispose();
            }
        }
    }
}
