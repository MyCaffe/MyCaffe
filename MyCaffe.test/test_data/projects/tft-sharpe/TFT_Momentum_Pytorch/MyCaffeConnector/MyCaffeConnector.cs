using MyCaffe;
using MyCaffe.basecode;
using MyCaffe.param.gpt;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.common;
using MyCaffe.solvers;
using System.Runtime.InteropServices;
using System.IO;
using System.Reflection;
using MyCaffe.layers;
using System.Reflection.Emit;
using OptimizerLib;
using MyCaffe.param.tft;
using System.Threading;
using System.Diagnostics;

namespace MyCaffeConnector
{
    public class MyCaffeConnector : IDisposable
    {
        CancelEvent m_evtCancel = new CancelEvent();
        Log m_log = new Log("MyCaffeConnector");
        MyCaffeControl<float> m_mycaffe;
        Dictionary<string, Layer<float>> m_rgLayers = new Dictionary<string, Layer<float>>();
        Net<float> m_net;
        BlobCollection<float> m_colTop = new BlobCollection<float>();
        BlobCollection<float> m_colBtm = new BlobCollection<float>();
        Blob<float> m_blobBtm;
        Blob<float> m_blobBtm1;
        Blob<float> m_blobBtm2;
        Blob<float> m_blobBtm3;
        Blob<float> m_blobTop;
        Blob<float> m_blobTop1;
        Blob<float> m_blobTop2;
        Blob<float> m_blobTmp;
        Blob<float> m_blobVal;
        Blob<float> m_blobWork;
        float m_fLastAccuracy = 0;
        Dictionary<string, string> m_rgTagLookup = new Dictionary<string, string>();
        int m_nIter = 0;

        public MyCaffeConnector()
        {
            SettingsCaffe s = new SettingsCaffe() { GpuIds = "0" };
            string strCudaPath = "C:\\Program Files\\SignalPop\\MyCaffe\\cuda_11.8\\CudaDnnDll.11.8.dll";
            m_mycaffe = new MyCaffeControl<float>(s, m_log, m_evtCancel, null, null, null, null, strCudaPath);

            m_rgTagLookup.Add("YYY.past_lstm", "past_lstm");
            m_rgTagLookup.Add("output", "output");
        }

        public void Dispose()
        {
            CleanUp();
        }

        public void Test(float[] ptr)
        {
        }

        public static string AssemblyDirectory
        {
            get
            {
                string codeBase = Assembly.GetExecutingAssembly().CodeBase;
                UriBuilder uri = new UriBuilder(codeBase);
                string path = Uri.UnescapeDataString(uri.Path);
                return Path.GetDirectoryName(path);
            }
        }

        private string buildSolver(double dfLr)
        {
            SolverParameter p = new SolverParameter();

            p.base_lr = dfLr;
            p.weight_decay = 0;
            p.momentum = 0.9;
            p.momentum2 = 0.999;
            p.delta = 1e-8;
            p.lr_policy = "fixed";
            p.type = SolverParameter.SolverType.ADAM;
            p.test_iter.Clear();
            p.test_interval = -1;

            return p.ToProto("root").ToString();
        }

        private string buildModel(string strSrc, bool bDecoderOnly, bool bAddDataLayer, int nNumSamples, int nNumHeads, float fDropout, int nLstmLayers, int nNumOutputs, int nStateSize, int nNumHistSteps, int nNumFutureSteps,
            int nNumStaticNumeric, int nNumStaticCategorical, List<int> rgStaticCardinalities,
            int nNumHistNumeric, int nNumHistCategorical, List<int> rgHistCardinalities,
            int nNumFutureNumeric, int nNumFutureCategorical, List<int> rgFutureCardinalities)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";

            //---------------------------------
            //  Data Temporal Input
            //---------------------------------
            if (bAddDataLayer)
            {
                LayerParameter data = new LayerParameter(LayerParameter.LayerType.DATA_TEMPORAL, "data");
                data.data_temporal_param.batch_size = (uint)nNumSamples;
                data.data_temporal_param.num_historical_steps = (uint)nNumHistSteps;
                data.data_temporal_param.num_future_steps = (uint)nNumFutureSteps;
                data.data_temporal_param.source = strSrc;
                data.data_temporal_param.source_type = DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE;
                data.data_temporal_param.shuffle_item_data = false;
                data.data_temporal_param.seed = 1704;
                data.include.Add(new NetStateRule(Phase.TRAIN));
                data.top.Add("x_numeric_static");
                data.top.Add("x_categorical_static");
                data.top.Add("x_numeric_hist");
                data.top.Add("x_categorical_hist");
                data.top.Add("x_numeric_future");
                data.top.Add("x_categorical_future");
                data.top.Add("target");
                p.layer.Add(data);

                data = new LayerParameter(LayerParameter.LayerType.DATA_TEMPORAL, "data");
                data.data_temporal_param.batch_size = (uint)nNumSamples;
                data.data_temporal_param.num_historical_steps = (uint)nNumHistSteps;
                data.data_temporal_param.num_future_steps = (uint)nNumFutureSteps;
                data.data_temporal_param.source = strSrc;
                data.data_temporal_param.source_type = DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE;
                data.data_temporal_param.shuffle_item_data = false;
                data.data_temporal_param.seed = 1704;
                data.include.Add(new NetStateRule(Phase.TEST));
                data.top.Add("x_numeric_static");
                data.top.Add("x_categorical_static");
                data.top.Add("x_numeric_hist");
                data.top.Add("x_categorical_hist");
                data.top.Add("x_numeric_future");
                data.top.Add("x_categorical_future");
                data.top.Add("target");
                p.layer.Add(data);

                data = new LayerParameter(LayerParameter.LayerType.DATA_TEMPORAL, "data");
                data.data_temporal_param.batch_size = (uint)nNumSamples;
                data.data_temporal_param.num_historical_steps = (uint)nNumHistSteps;
                data.data_temporal_param.num_future_steps = (uint)nNumFutureSteps;
                data.data_temporal_param.source = strSrc;
                data.data_temporal_param.source_type = DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE;
                data.data_temporal_param.shuffle_item_data = false;
                data.data_temporal_param.seed = 1704;
                data.include.Add(new NetStateRule(Phase.RUN));
                data.top.Add("x_numeric_static");
                data.top.Add("x_categorical_static");
                data.top.Add("x_numeric_hist");
                data.top.Add("x_categorical_hist");
                data.top.Add("x_numeric_future");
                data.top.Add("x_categorical_future");
                data.top.Add("target");
                p.layer.Add(data);
            }
            else
            {
                LayerParameter data = new LayerParameter(LayerParameter.LayerType.INPUT);
                data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumStaticNumeric }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumStaticCategorical }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHistSteps, nNumHistNumeric }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHistSteps, nNumHistCategorical }));
                if (bDecoderOnly == false)
                {
                    data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFutureSteps, nNumFutureNumeric }));
                    data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFutureSteps, nNumFutureCategorical }));
                }
                data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, (bDecoderOnly == false) ? nNumFutureSteps : nNumHistSteps }));
                data.top.Add("x_numeric_static");
                data.top.Add("x_categorical_static");
                data.top.Add("x_numeric_hist");
                data.top.Add("x_categorical_hist");
                if (bDecoderOnly == false)
                {
                    data.top.Add("x_numeric_future");
                    data.top.Add("x_categorical_future");
                }
                data.top.Add("target");
                p.layer.Add(data);
            }

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

            if (bDecoderOnly == false)
            {
                LayerParameter future_ts_transform = new LayerParameter(LayerParameter.LayerType.CHANNEL_EMBEDDING, "future_ts_trfm");
                future_ts_transform.numeric_trans_param.num_input = (uint)nNumFutureNumeric;
                future_ts_transform.numeric_trans_param.state_size = (uint)nStateSize;
                future_ts_transform.categorical_trans_param.num_input = (uint)nNumFutureCategorical;
                future_ts_transform.categorical_trans_param.cardinalities = rgFutureCardinalities;
                future_ts_transform.categorical_trans_param.state_size = (uint)nStateSize;
                future_ts_transform.bottom.Add("x_numeric_future");
                future_ts_transform.bottom.Add("x_categorical_future");
                future_ts_transform.top.Add("future_ts_rep");
                p.layer.Add(future_ts_transform);
            }


            //---------------------------------
            //  Variable Selection Networks - Static
            //---------------------------------
            LayerParameter static_vsn = new LayerParameter(LayerParameter.LayerType.VARSELNET, "static_vsn");
            static_vsn.varselnet_param.input_dim = nStateSize;
            static_vsn.varselnet_param.num_inputs = nNumStaticNumeric + nNumStaticCategorical;
            static_vsn.varselnet_param.hidden_dim = nStateSize;
            static_vsn.varselnet_param.dropout_ratio = fDropout;
            static_vsn.bottom.Add("static_rep");
            static_vsn.top.Add("selected_static");
            p.layer.Add(static_vsn);


            //---------------------------------
            //  Static covariate encoders
            //---------------------------------
            LayerParameter selstat_split = new LayerParameter(LayerParameter.LayerType.SPLIT, "selstat_split");
            selstat_split.bottom.Add("selected_static");
            selstat_split.top.Add("selstat_a");
            selstat_split.top.Add("selstat_b");
            selstat_split.top.Add("selstat_c");
            selstat_split.top.Add("selstat_d");
            p.layer.Add(selstat_split);

            LayerParameter static_cov_enc = new LayerParameter(LayerParameter.LayerType.GRN, "static_cov_enc");
            static_cov_enc.grn_param.input_dim = nStateSize;
            static_cov_enc.grn_param.hidden_dim = nStateSize;
            static_cov_enc.grn_param.output_dim = nStateSize;
            static_cov_enc.grn_param.dropout_ratio = fDropout;

            LayerParameter static_enc_sel = static_cov_enc.Clone(false);
            static_enc_sel.name = "enc_sel";
            static_enc_sel.bottom.Add("selstat_a");
            static_enc_sel.top.Add("c_selection");
            p.layer.Add(static_enc_sel);

            LayerParameter static_enc_enrich = static_cov_enc.Clone(false);
            static_enc_enrich.name = "enc_enr";
            static_enc_enrich.bottom.Add("selstat_b");
            static_enc_enrich.top.Add("c_enrichment");
            p.layer.Add(static_enc_enrich);

            LayerParameter static_enc_seq_cell_init = static_cov_enc.Clone(false);
            static_enc_seq_cell_init.name = "enc_seq_cell_init";
            static_enc_seq_cell_init.bottom.Add("selstat_c");
            static_enc_seq_cell_init.top.Add("c_seq_cell");
            p.layer.Add(static_enc_seq_cell_init);

            LayerParameter static_enc_seq_state_init = static_cov_enc.Clone(false);
            static_enc_seq_state_init.name = "enc_seq_state_init";
            static_enc_seq_state_init.bottom.Add("selstat_d");
            static_enc_seq_state_init.top.Add("c_seq_hidden");
            p.layer.Add(static_enc_seq_state_init);

            string strCSelectionH = "c_selection";
            if (bDecoderOnly == false)
            {
                LayerParameter c_sel_split = new LayerParameter(LayerParameter.LayerType.SPLIT, "c_sel_split");
                c_sel_split.bottom.Add("c_selection");
                c_sel_split.top.Add("c_selection_h");
                c_sel_split.top.Add("c_selection_f");
                p.layer.Add(c_sel_split);
                strCSelectionH = "c_selection_h";
            }

            //---------------------------------
            //  Variable Selection Networks - Temporal
            //---------------------------------
            LayerParameter hist_vsh_reshape_before = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_hist_b");
            hist_vsh_reshape_before.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.BEFORE;
            hist_vsh_reshape_before.bottom.Add("hist_ts_rep");
            hist_vsh_reshape_before.bottom.Add(strCSelectionH);
            hist_vsh_reshape_before.top.Add("hist_ts_rep1");
            hist_vsh_reshape_before.top.Add("c_selection1h");
            p.layer.Add(hist_vsh_reshape_before);

            LayerParameter hist_vsn = new LayerParameter(LayerParameter.LayerType.VARSELNET, "hist_vsn");
            hist_vsn.varselnet_param.input_dim = nStateSize;
            hist_vsn.varselnet_param.num_inputs = nNumHistNumeric + nNumHistCategorical;
            hist_vsn.varselnet_param.hidden_dim = nStateSize;
            hist_vsn.varselnet_param.dropout_ratio = fDropout;
            hist_vsn.varselnet_param.context_dim = nStateSize;
            hist_vsn.bottom.Add("hist_ts_rep1");
            hist_vsn.bottom.Add("c_selection1h");
            hist_vsn.top.Add("selected_hist1");
            p.layer.Add(hist_vsn);

            LayerParameter hist_vsh_reshape_after = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_hist_a");
            hist_vsh_reshape_after.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.AFTER;
            hist_vsh_reshape_after.reshape_temporal_param.enable_clip_output = true;
            hist_vsh_reshape_after.bottom.Add("selected_hist1");
            hist_vsh_reshape_after.top.Add("selected_hist");
            hist_vsh_reshape_after.top.Add("selected_hist_clip");
            p.layer.Add(hist_vsh_reshape_after);

            if (bDecoderOnly == false)
            {
                LayerParameter future_vsh_reshape_before = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_fut_b");
                future_vsh_reshape_before.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.BEFORE;
                future_vsh_reshape_before.bottom.Add("future_ts_rep");
                future_vsh_reshape_before.bottom.Add("c_selection_f");
                future_vsh_reshape_before.top.Add("future_ts_rep1");
                future_vsh_reshape_before.top.Add("c_selection1f");
                p.layer.Add(future_vsh_reshape_before);

                LayerParameter fut_vsn = new LayerParameter(LayerParameter.LayerType.VARSELNET, "future_vsn");
                fut_vsn.varselnet_param.input_dim = nStateSize;
                fut_vsn.varselnet_param.num_inputs = nNumFutureNumeric + nNumFutureCategorical;
                fut_vsn.varselnet_param.hidden_dim = nStateSize;
                fut_vsn.varselnet_param.dropout_ratio = fDropout;
                fut_vsn.varselnet_param.context_dim = nStateSize;
                fut_vsn.bottom.Add("future_ts_rep1");
                fut_vsn.bottom.Add("c_selection1f");
                fut_vsn.top.Add("selected_fut1");
                p.layer.Add(fut_vsn);

                LayerParameter future_vsh_reshape_after = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_fut_a");
                future_vsh_reshape_after.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.AFTER;
                future_vsh_reshape_after.reshape_temporal_param.enable_clip_output = true;
                future_vsh_reshape_after.bottom.Add("selected_fut1");
                future_vsh_reshape_after.top.Add("selected_fut");
                future_vsh_reshape_after.top.Add("selected_fut_clip");
                p.layer.Add(future_vsh_reshape_after);
            }

            //---------------------------------
            //  Locality Enhancement with Seq2Seq processing
            //---------------------------------
            string strLstmInput = "selected_hist";
            if (bDecoderOnly == false)
            {
                LayerParameter selhist_split = new LayerParameter(LayerParameter.LayerType.SPLIT, "selhist_split");
                selhist_split.bottom.Add("selected_hist");
                selhist_split.top.Add("selhist_a");
                selhist_split.top.Add("selhist_b");
                p.layer.Add(selhist_split);

                LayerParameter selfut_split = new LayerParameter(LayerParameter.LayerType.SPLIT, "selfut_split");
                selfut_split.bottom.Add("selected_fut");
                selfut_split.top.Add("selfut_a");
                selfut_split.top.Add("selfut_b");
                p.layer.Add(selfut_split);

                LayerParameter lstm_input = new LayerParameter(LayerParameter.LayerType.CONCAT, "lstm_input");
                lstm_input.concat_param.axis = 1;
                lstm_input.bottom.Add("selhist_a");
                lstm_input.bottom.Add("selfut_a");
                lstm_input.top.Add("lstm_input");
                p.layer.Add(lstm_input);
                strLstmInput = "lstm_input";
            }

            LayerParameter past_lstm = new LayerParameter(LayerParameter.LayerType.LSTM, "past_lstm");
            past_lstm.recurrent_param.num_output = (uint)nStateSize;
            past_lstm.recurrent_param.num_layers = (uint)nLstmLayers;
            past_lstm.recurrent_param.dropout_ratio = fDropout;
            past_lstm.recurrent_param.expose_hidden_input = true;
            past_lstm.recurrent_param.expose_hidden_output = !bDecoderOnly;
            past_lstm.recurrent_param.batch_first = true;
            past_lstm.recurrent_param.auto_repeat_hidden_states_across_layers = true;
            past_lstm.recurrent_param.use_cudnn_rnn8_if_supported = true;
            past_lstm.recurrent_param.engine = EngineParameter.Engine.CUDNN;
            past_lstm.bottom.Add(strLstmInput);
            past_lstm.bottom.Add("selected_hist_clip");
            past_lstm.bottom.Add("c_seq_hidden");
            past_lstm.bottom.Add("c_seq_cell");
            past_lstm.top.Add("past_lstm_output");
            if (bDecoderOnly == false)
            {
                past_lstm.top.Add("hidden1");
                past_lstm.top.Add("cell1");
            }
            p.layer.Add(past_lstm);

            string strLstmOutput = "past_lstm_output";
            if (bDecoderOnly == false)
            {
                LayerParameter future_lstm = new LayerParameter(LayerParameter.LayerType.LSTM, "future_lstm");
                future_lstm.recurrent_param.num_output = (uint)nStateSize;
                future_lstm.recurrent_param.num_layers = (uint)nLstmLayers;
                future_lstm.recurrent_param.dropout_ratio = fDropout;
                future_lstm.recurrent_param.expose_hidden_input = true;
                future_lstm.recurrent_param.batch_first = true;
                future_lstm.recurrent_param.auto_repeat_hidden_states_across_layers = true;
                future_lstm.recurrent_param.use_cudnn_rnn8_if_supported = true;
                future_lstm.recurrent_param.engine = EngineParameter.Engine.CUDNN;
                future_lstm.bottom.Add("selfut_b");
                future_lstm.bottom.Add("selected_fut_clip");
                future_lstm.bottom.Add("hidden1");
                future_lstm.bottom.Add("cell1");
                future_lstm.top.Add("future_lstm_output");
                p.layer.Add(future_lstm);

                LayerParameter lstm_output = new LayerParameter(LayerParameter.LayerType.CONCAT, "lstm_output");
                lstm_output.concat_param.axis = 1;
                lstm_output.bottom.Add("past_lstm_output");
                lstm_output.bottom.Add("future_lstm_output");
                lstm_output.top.Add("lstm_output");
                p.layer.Add(lstm_output);
                strLstmOutput = "lstm_output";
            }

            LayerParameter post_lstm_gating = new LayerParameter(LayerParameter.LayerType.GATEADDNORM, "post_lstm_gate");
            post_lstm_gating.dropout_param.dropout_ratio = fDropout;
            post_lstm_gating.layer_norm_param.enable_cuda_impl = false;
            post_lstm_gating.layer_norm_param.epsilon = 1e-10;
            post_lstm_gating.glu_param.input_dim = nStateSize;
            post_lstm_gating.glu_param.axis = 2;
            post_lstm_gating.bottom.Add(strLstmOutput);
            post_lstm_gating.bottom.Add(strLstmInput);
            post_lstm_gating.top.Add("gated_lstm_output");
            p.layer.Add(post_lstm_gating);


            //---------------------------------
            //  Static enrichment
            //---------------------------------
            LayerParameter glstmout_split = new LayerParameter(LayerParameter.LayerType.SPLIT, "glstmout_split");
            glstmout_split.bottom.Add("gated_lstm_output");
            glstmout_split.top.Add("glstmout_a");
            glstmout_split.top.Add("glstmout_b");
            p.layer.Add(glstmout_split);

            LayerParameter static_enrich_grn_reshape_before = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_statenr_a");
            static_enrich_grn_reshape_before.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.BEFORE;
            static_enrich_grn_reshape_before.bottom.Add("glstmout_a");
            static_enrich_grn_reshape_before.bottom.Add("c_enrichment");
            static_enrich_grn_reshape_before.top.Add("gated_lstm_output1");
            static_enrich_grn_reshape_before.top.Add("c_enrichment1");
            p.layer.Add(static_enrich_grn_reshape_before);

            LayerParameter static_enrich_grn = new LayerParameter(LayerParameter.LayerType.GRN, "static_enrich_gru");
            static_enrich_grn.grn_param.input_dim = nStateSize;
            static_enrich_grn.grn_param.hidden_dim = nStateSize;
            static_enrich_grn.grn_param.output_dim = nStateSize;
            static_enrich_grn.grn_param.context_dim = nStateSize;
            static_enrich_grn.grn_param.dropout_ratio = fDropout;
            static_enrich_grn.bottom.Add("gated_lstm_output1");
            static_enrich_grn.bottom.Add("c_enrichment1");
            static_enrich_grn.top.Add("enriched_sequence1a");
            p.layer.Add(static_enrich_grn);

            LayerParameter static_enrich_grn_reshape_after = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_statenr_b");
            static_enrich_grn_reshape_after.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.AFTER;
            static_enrich_grn_reshape_after.bottom.Add("enriched_sequence1a");
            static_enrich_grn_reshape_after.top.Add("enriched_sequence");
            p.layer.Add(static_enrich_grn_reshape_after);


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
            post_attn_gate.gateaddnorm_param.residual_channel_offset = (bDecoderOnly == false) ? nNumHistSteps : 0;
            post_attn_gate.dropout_param.dropout_ratio = fDropout;
            post_attn_gate.layer_norm_param.enable_cuda_impl = false;
            post_attn_gate.glu_param.input_dim = nStateSize;
            post_attn_gate.glu_param.axis = 2;
            post_attn_gate.bottom.Add("post_attention");
            post_attn_gate.bottom.Add("enr_seq_b");
            post_attn_gate.top.Add("gated_post_attention");
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
            pos_wise_ff_grn.grn_param.axis = 2;
            pos_wise_ff_grn.grn_param.dropout_ratio = fDropout;
            pos_wise_ff_grn.bottom.Add("gated_post_attention");
            pos_wise_ff_grn.top.Add("post_poswise_ff_grn");
            p.layer.Add(pos_wise_ff_grn);

            LayerParameter pos_wise_ff_gate = new LayerParameter(LayerParameter.LayerType.GATEADDNORM, "pos_wise_ff_gate");
            pos_wise_ff_gate.gateaddnorm_param.residual_channel_offset = (bDecoderOnly == false) ? nNumHistSteps : 0;
            pos_wise_ff_gate.dropout_param.dropout_ratio = fDropout;
            pos_wise_ff_gate.layer_norm_param.enable_cuda_impl = false;
            pos_wise_ff_gate.glu_param.input_dim = nStateSize;
            pos_wise_ff_gate.glu_param.axis = 2;
            pos_wise_ff_gate.bottom.Add("post_poswise_ff_grn");
            pos_wise_ff_gate.bottom.Add("glstmout_b");
            pos_wise_ff_gate.top.Add("gated_poswise_ff");
            p.layer.Add(pos_wise_ff_gate);


            //---------------------------------
            //  Output layer
            //---------------------------------
            LayerParameter output = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "output");
            output.inner_product_param.num_output = (uint)nNumOutputs;
            output.inner_product_param.axis = 2;
            output.inner_product_param.bias_term = true;
            output.bottom.Add("gated_poswise_ff");
            output.top.Add("predicted_quantiles");
            p.layer.Add(output);

            LayerParameter tanh = new LayerParameter(LayerParameter.LayerType.TANH, "tanh");
            tanh.bottom.Add("predicted_quantiles");
            tanh.top.Add("predicted_quantiles");
            p.layer.Add(tanh);


            //---------------------------------
            //  Loss
            //---------------------------------
            if (bDecoderOnly == false)
            {
                LayerParameter loss = new LayerParameter(LayerParameter.LayerType.QUANTILE_LOSS, "loss");
                loss.quantile_loss_param.desired_quantiles.Add(0.1f);
                loss.quantile_loss_param.desired_quantiles.Add(0.5f);
                loss.quantile_loss_param.desired_quantiles.Add(0.9f);
                loss.loss_weight.Add(1); // for loss
                loss.loss_weight.Add(0); // for q_risk
                loss.loss_param.normalization = LossParameter.NormalizationMode.NONE;
                loss.bottom.Add("predicted_quantiles");
                loss.bottom.Add("target");
                loss.top.Add("loss");
                loss.top.Add("q_risk");
                p.layer.Add(loss);
            }
            else
            {
                LayerParameter loss = new LayerParameter(LayerParameter.LayerType.SHARPE_LOSS, "loss");
                loss.loss_weight.Add(1); // for loss
                loss.loss_param.normalization = LossParameter.NormalizationMode.NONE;
                loss.bottom.Add("predicted_quantiles");
                loss.bottom.Add("target");
                loss.top.Add("loss");
                p.layer.Add(loss);
            }

            return p.ToProto("root").ToString();
        }

        public void Initialize()
        {
            int nNumSamples = 256;
            int nNumHeads = 4;
            int fDropout = 0;
            int nLstmLayers = 2;
            int nNumOutputs = 1;
            int nStateSize = 10;
            int nNumHistSteps = 63;
            int nNumFutureSteps = 0;
            int nNumStaticNumeric = 0;
            int nNumStaticCategorical = 1;
            int nNumHistoricalNumeric = 8;
            int nNumHistoricalCategorical = 0;
            int nNumFutureNumeric = 0;
            int nNumFutureCategorical = 0;
            List<int> rgStaticCardinalities = new List<int>() { 512 };
            List<int> rgHistCardinalities = new List<int>();
            List<int> rgFutureCardinalities = new List<int>();

            string strSolver = buildSolver(0.01);
            string strModel = buildModel("", true, false, nNumSamples, nNumHeads, fDropout, nLstmLayers, nNumOutputs, nStateSize, nNumHistSteps, nNumFutureSteps,
                                          nNumStaticNumeric, nNumStaticCategorical, rgStaticCardinalities,
                                          nNumHistoricalNumeric, nNumHistoricalCategorical, rgHistCardinalities,
                                          nNumFutureNumeric, nNumFutureCategorical, rgFutureCardinalities);

            try
            {
                m_mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);
                m_net = m_mycaffe.GetInternalNet(Phase.TRAIN);
                string strPath = "C:\\Data\\Data\\SS_Projects\\Intelligence\\GitHub\\MyCaffe\\MyCaffe.test\\test_data\\projects\\tft-sharpe\\TFT_Momentum_Pytorch\\TFT_Momentum_Pytorch\\test\\tft.sharpe\\weights\\";
                load_weights("", m_net, strPath, nNumStaticNumeric, nNumStaticCategorical, nNumHistoricalNumeric, nNumHistoricalCategorical, nNumFutureNumeric, nNumFutureCategorical);

                m_blobBtm = m_mycaffe.CreateBlob("btm");
                m_blobBtm1 = m_mycaffe.CreateBlob("btm1");
                m_blobBtm2 = m_mycaffe.CreateBlob("btm2");
                m_blobBtm3 = m_mycaffe.CreateBlob("btm3");
                m_blobTop = m_mycaffe.CreateBlob("top");
                m_blobTop1 = m_mycaffe.CreateBlob("top1");
                m_blobTop2 = m_mycaffe.CreateBlob("top2");
                m_blobTmp = m_mycaffe.CreateBlob("tmp");
                m_blobTmp.Reshape(1, 1, 1, 1);
                m_colTop.Clear();
                m_colTop.Add(m_blobTop);
                m_colBtm.Clear();
                m_colBtm.Add(m_blobBtm);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
        }

        private void load_weights(string strTag1, Net<float> net, string strPath, int nNumStaticNumeric, int nNumStaticCategorical, int nNumHistNumeric, int nNumHistCategorical, int nNumFutureNumeric, int nNumFutureCategorical)
        {
            string strTag;

            //-------------------------------------------
            // Load input channel embedding weights.
            //-------------------------------------------
            int nIdx = 0;
            strTag = "static_transform\\";
            for (int i = 0; i < nNumStaticCategorical; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "categorical_transform.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                nIdx++;
            }

            strTag = "hist_ts_transform\\";
            for (int i = 0; i < nNumHistNumeric; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "numeric_transform.module.numeric_projection_layers." + i.ToString() + ".weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "numeric_transform.module.numeric_projection_layers." + i.ToString() + ".bias.npy");
                nIdx++;
            }

            strTag = "hist_ts_transform\\";
            for (int i = 0; i < nNumHistCategorical; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "categorical_transform.module.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                nIdx++;
            }

            strTag = "future_ts_transform\\";
            for (int i = 0; i < nNumFutureNumeric; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "numeric_transform.module.numeric_projection_layers." + i.ToString() + ".weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "numeric_transform.module.numeric_projection_layers." + i.ToString() + ".bias.npy");
                nIdx++;
            }

            strTag = "future_ts_transform\\";
            for (int i = 0; i < nNumFutureCategorical; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "categorical_transform.module.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                nIdx++;
            }

            //-------------------------------------------
            // Load varselnet weights - static (idx=33)
            //-------------------------------------------
            strTag = "static_selection\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.skip_layer.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.skip_layer.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.gate.module.fc2.bias.npy");
            nIdx++;

            for (int i = 0; i < nNumStaticNumeric + nNumStaticCategorical; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                nIdx++;
            }

            //---------------------------------
            //  Static covariate encoders (idx=115)
            //---------------------------------
            strTag = "static_encoder_selection\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.bias.npy");
            nIdx++;

            strTag = "static_encoder_enrichment\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.bias.npy");
            nIdx++;

            strTag = "static_encoder_sequential_cell_init\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.bias.npy");
            nIdx++;

            strTag = "static_encoder_sequential_state_init\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.bias.npy");
            nIdx++;

            //-------------------------------------------
            // Load varselnet weights - historical (idx=147)
            //-------------------------------------------
            strTag = "hist_ts_selection\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.skip_layer.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.skip_layer.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.context_projection.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.gate.module.fc2.bias.npy");
            nIdx++;

            for (int i = 0; i < nNumHistNumeric + nNumHistCategorical; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                nIdx++;
            }

            //-------------------------------------------
            // Load varselnet weights - future (idx=246)
            //-------------------------------------------
            if (nNumFutureCategorical > 0 || nNumFutureNumeric > 0)
            {
                strTag = "future_ts_selection\\";
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.skip_layer.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.skip_layer.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.fc1.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.fc1.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.context_projection.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.fc2.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.fc2.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.gate.module.fc1.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.gate.module.fc1.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.gate.module.fc2.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "flattened_grn.gate.module.fc2.bias.npy");
                nIdx++;

                for (int i = 0; i < nNumFutureNumeric + nNumFutureCategorical; i++)
                {
                    net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx++;
                    net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx++;
                }
            }

            //---------------------------------
            //  Locality Enhancement with Seq2Seq processing (idx=321)
            //---------------------------------
            strTag = "past_lstm\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "ZZZ.YYY.past_lstm.lstm.wt0.npy");
            nIdx++;
            if (nNumFutureCategorical > 0 || nNumFutureNumeric > 0)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "ZZZ.YYY.future_lstm.lstm.wt0.npy");
                nIdx++;
            }
            strTag = "post_lstm_gating\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.bias.npy");
            nIdx++;


            //---------------------------------
            //  Temporal Static Enrichment (idx=327)
            //---------------------------------
            strTag = "static_enrichment_grn\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "context_projection.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.bias.npy");
            nIdx++;


            //---------------------------------
            //  Temporal Self-attention (idx=336)
            //---------------------------------
            strTag = "multihead_attn\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "w_q.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "w_q.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "w_k.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "w_k.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "w_v.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "w_v.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "out.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "out.bias.npy");
            nIdx++;
            strTag = "post_attention_gating\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.bias.npy");
            nIdx++;

            //---------------------------------
            //  Pos wise FF (idx=348)
            //---------------------------------
            strTag = "pos_wise_ff_grn\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.bias.npy");
            nIdx++;

            //---------------------------------
            //  Pos wise FF Gate (idx=356)
            //---------------------------------
            strTag = "pos_wise_ff_gating\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "gate.module.fc2.bias.npy");
            nIdx++;

            //---------------------------------
            //  Output (idx=360)
            //---------------------------------
            strTag = "output_layer\\";
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + "bias.npy");
            nIdx++;
        }

        public float[] get_return_values(BlobCollection<float> colBlobs, bool bDiff)
        {
            int nTotal = 0;
            List<int> rgCounts = new List<int>();
            List<float> rgData = new List<float>();
            rgData.Add(colBlobs.Count);

            for (int i = 0; i < colBlobs.Count; i++)
            {
                int nCount = colBlobs[i].count();
                rgCounts.Add(nCount);
                rgData.Add(nCount);
                nTotal += nCount;
            }

            for (int i = 0; i < colBlobs.Count; i++)
            {
                List<int> rgShape = colBlobs[i].shape();
                rgData.Add(rgShape.Count());
                for (int j = 0; j < rgShape.Count; j++)
                {
                    rgData.Add(rgShape[j]);
                }
            }

            for (int i = 0; i < colBlobs.Count; i++)
            {
                int nCount = colBlobs[i].count();
                if (nCount > 0)
                {
                    float[] rg;

                    if (bDiff)
                        rg = colBlobs[i].update_cpu_diff();
                    else
                        rg = colBlobs[i].update_cpu_data();

                    if (rg.Length != rgCounts[i])
                        Trace.WriteLine("bad count.");

                    rgData.AddRange(rg);
                }
            }

            return rgData.ToArray();
        }

        public void load_input_values(float[] rgData, BlobCollection<float> colBlobs, bool bDiff)
        {
            int nIdx = 0;
            int nBlobCount = (int)rgData[nIdx];
            List<int> rgItemCounts = new List<int>();
            List<List<int>> rgrgShapes = new List<List<int>>();
           
            nIdx++;

            for (int i = 0; i < nBlobCount; i++)
            {
                List<int> rgShape = new List<int>();

                int nShapeCount = (int)rgData[nIdx];
                nIdx++;

                for (int j = 0; j < nShapeCount; j++)
                {
                    rgShape.Add((int)rgData[nIdx]);
                    nIdx++;
                }

                rgrgShapes.Add(rgShape);
            }

            for (int i = 0; i < nBlobCount; i++)
            {
                int nItemCount = (int)rgData[nIdx];
                nIdx++;

                rgItemCounts.Add(nItemCount);
            }

            for (int i = 0; i < nBlobCount; i++)
            {
                int nCount = rgItemCounts[i];
                if (nCount == 0)
                    continue;

                float[] rgDataFloat = new float[nCount];

                Array.Copy(rgData, nIdx, rgDataFloat, 0, nCount);
                nIdx += nCount;

                colBlobs[i].Reshape(rgrgShapes[i]);

                if (bDiff)
                    colBlobs[i].mutable_cpu_diff = rgDataFloat;
                else
                    colBlobs[i].mutable_cpu_data = rgDataFloat;
            }
        }

        public float[] model_fwd(float[] rgIn, int nStart=0, int nEnd = -1)
        {
            Net<float> net = m_mycaffe.GetInternalNet(Phase.TRAIN);

            if (nEnd == -1)
                nEnd = net.layers.Count - 1;

            m_nIter++;

            if (rgIn != null)
            {
                BlobCollection<float> colBtm = new BlobCollection<float>();
                //colBtm.Add(net.FindBlob("x_numeric_static"));
                //colBtm.Add(net.FindBlob("x_categorical_static"));
                //colBtm.Add(net.FindBlob("x_numeric_hist"));
                //colBtm.Add(net.FindBlob("x_categorical_hist"));
                //colBtm.Add(net.FindBlob("x_numeric_future"));
                //colBtm.Add(net.FindBlob("x_categorical_future"));

                if (nStart == 0) // at input
                {
                    colBtm.Add(net.FindBlob("x_categorical_static"));
                    colBtm.Add(net.FindBlob("x_numeric_hist"));
                    colBtm.Add(m_blobTmp);
                }
                else if (nStart == 3) // at stat selection
                {
                    colBtm.Add(net.FindBlob("static_rep"));
                    colBtm.Add(net.FindBlob("hist_ts_rep"));
                }
                else if (nStart == 4) // at stat selection
                {
                    colBtm.Add(net.FindBlob("selected_static"));
                    colBtm.Add(net.FindBlob("hist_ts_rep"));
                }
                else if (nStart == 9) // at hist selection
                {
                    colBtm.Add(net.FindBlob("hist_ts_rep"));
                    colBtm.Add(net.FindBlob("c_selection"));
                    colBtm.Add(net.FindBlob("c_seq_hidden"));
                    colBtm.Add(net.FindBlob("c_seq_cell"));
                    colBtm.Add(net.FindBlob("c_enrichment"));
                }
                else if (nStart == 12) // at static enrichment
                {
                    colBtm.Add(net.FindBlob("selected_hist"));
                    colBtm.Add(net.FindBlob("c_seq_hidden"));
                    colBtm.Add(net.FindBlob("c_seq_cell"));
                    colBtm.Add(net.FindBlob("c_enrichment"));
                }
                else if (nStart == 15) // at static enrichment
                {
                    colBtm.Add(net.FindBlob("gated_lstm_output"));
                    colBtm.Add(net.FindBlob("c_enrichment"));
                }
                else if (nStart == 19) // at attention
                {
                    colBtm.Add(net.FindBlob("enriched_sequence"));
                    colBtm.Add(net.FindBlob("glstmout_b"));
                }
                else if (nStart == 23) // at gated post attention
                {
                    colBtm.Add(net.FindBlob("gated_post_attention"));
                    colBtm.Add(net.FindBlob("glstmout_b"));
                }
                else if (nStart == 24) // at post poswise ff
                {
                    colBtm.Add(net.FindBlob("post_poswise_ff_grn"));
                    colBtm.Add(net.FindBlob("glstmout_b"));
                }
                else if (nStart == 25) // at output IP
                {
                    colBtm.Add(net.FindBlob("gated_poswise_ff"));
                }
                else if (nStart == 27) // at loss
                {
                    colBtm.Add(net.FindBlob("predicted_quantiles"));
                    colBtm.Add(net.FindBlob("target"));
                }

                if (colBtm.Count > 0)
                    load_input_values(rgIn, colBtm, false);
            }

            net.ForwardFromTo(nStart, nEnd);

            //if (m_blobVal == null)
            //    m_blobVal = m_mycaffe.CreateBlob("val");

            //if (m_blobWork == null)
            //    m_blobWork = m_mycaffe.CreateBlob("work");

            //Blob<float> blob;
            //Blob<float> blobVal = m_blobVal;
            //Blob<float> blobWork = m_blobWork;
            //string strPath = "C:\\Data\\Data\\SS_Projects\\Intelligence\\GitHub\\MyCaffe\\MyCaffe.test\\test_data\\projects\\tft-sharpe\\TFT_Momentum_Pytorch\\TFT_Momentum_Pytorch\\test\\";
            //string strTag = "tft.all";

            //if (nStart <= 3)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".historical_ts_rep.npy");
            //    blob = net.FindBlob("hist_ts_rep");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".static_rep.npy");
            //    blob = net.FindBlob("static_rep");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));
            //}

            //if (nStart <= 4)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".selected_static.npy");
            //    blob = net.FindBlob("selected_static");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));
            //}

            //if (nStart <= 9)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".c_enrichment.XX.npy");
            //    blob = net.FindBlob("c_enrichment");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".c_selection.XX.npy");
            //    blob = net.FindBlob("c_selection");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".c_seq_hidden.XX.npy");
            //    blob = net.FindBlob("c_seq_hidden");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".c_seq_cell.XX.npy");
            //    blob = net.FindBlob("c_seq_cell");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));
            //}

            //if (nStart <= 12)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".hist.temporal_flattened_embedding.npy");
            //    blob = net.FindBlob("hist_ts_rep1");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".hist.time_distributed_context.npy");
            //    blob = net.FindBlob("c_selection1h");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));

            //    //**bug in VSN

            //    blobVal.LoadFromNumpy(strPath + strTag + ".hist.temporal_selection_output1.ZZ.npy");
            //    blob = net.FindBlob("selected_hist1");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".selected_historical.npy");
            //    blob = net.FindBlob("selected_hist");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));
            //}

            //if (nStart <= 15)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".gated_lstm_output.npy");
            //    blob = net.FindBlob("gated_lstm_output");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));
            //}

            //if (nStart <= 19)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".enriched_sequence.npy");
            //    blob = net.FindBlob("enriched_sequence");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));
            //}

            //if (nStart <= 23)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".gated_post_attention.npy");
            //    blob = net.FindBlob("gated_post_attention");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));
            //}

            //if (nStart <= 24)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".post_poswise_ff_grn.npy");
            //    blob = net.FindBlob("post_poswise_ff_grn");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));
            //}

            //if (nStart <= 25)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".gated_poswise_ff.npy");
            //    blob = net.FindBlob("gated_poswise_ff");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));
            //}

            //if (nStart <= 27)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".predicted_quantiles.npy");
            //    blob = net.FindBlob("predicted_quantiles");
            //    Trace.Assert(blobVal.Compare(blob, blobWork));
            //}

            BlobCollection<float> colTop = new BlobCollection<float>();
            colTop.Add(net.top_vecs[nEnd]);
            if (nEnd == net.top_vecs.Count - 1)
            {
                colTop.Add(net.FindBlob("predicted_quantiles"));
                colTop.Add(net.FindBlob("target"));
            }

            return get_return_values(colTop, false);
        }

        public float[] model_bwd(float[] rgIn, int nStart=0, int nEnd = -1)
        {
            Net<float> net = m_mycaffe.GetInternalNet(Phase.TRAIN);

            if (nEnd == -1)
                nEnd = net.layers.Count - 1;

            if (rgIn != null)
            {
                BlobCollection<float> col = new BlobCollection<float>();
                col.Add(net.top_vecs[nEnd][0]);
                load_input_values(rgIn, col, true);
            }

            //if (m_blobVal == null)
            //    m_blobVal = m_mycaffe.CreateBlob("val");

            //if (m_blobWork == null)
            //    m_blobWork = m_mycaffe.CreateBlob("work");

            //Blob<float> blob;
            //Blob<float> blobVal = m_blobVal;
            //Blob<float> blobWork = m_blobWork;
            //string strPath = "C:\\Data\\Data\\SS_Projects\\Intelligence\\GitHub\\MyCaffe\\MyCaffe.test\\test_data\\projects\\tft-sharpe\\TFT_Momentum_Pytorch\\TFT_Momentum_Pytorch\\test\\";
            //string strTag = "tft.all";

            //if (nStart <= 25)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".predicted_quantiles.grad.npy", true);
            //    blob = net.FindBlob("predicted_quantiles");
            //    blob.CopyFrom(blobVal, true);
            //}

            net.Backward(nEnd, nStart);

            //if (nStart == 27)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".predicted_quantiles.grad.npy", true);
            //    blob = net.FindBlob("predicted_quantiles");
            //    bool bVal = blobVal.Compare(blob, blobWork, true);
            //}

            //if (nStart <= 25)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".gated_poswise_ff.grad.npy", true);
            //    blob = net.FindBlob("gated_poswise_ff");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));
            //}

            //if (nStart <= 24)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".post_poswise_ff_grn.grad.npy", true);
            //    blob = net.FindBlob("post_poswise_ff_grn");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));
            //}

            //if (nStart <= 23)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".gated_post_attention.grad.npy", true);
            //    blob = net.FindBlob("gated_post_attention");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));
            //    //blobVal.LoadFromNumpy(strPath + strTag + ".attention_scores.grad.npy", true);
            //    blob = net.FindBlob("attention_scores");
            //    //Trace.Assert(blobVal.Compare(blob, blobWork, true));
            //}

            //if (nStart <= 19)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".ada.gated_post_attention.grad.npy", true);
            //    blob = net.FindBlob("gated_post_attention");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".ada.post_attention.grad.npy", true);
            //    blob = net.FindBlob("post_attention");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".ada.enriched_sequence_b.grad.npy", true);
            //    blob = net.FindBlob("enr_seq_b");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".ada.q1.grad.npy", true);
            //    blob = net.layers[20].internal_blobs.FindBlob("mh_attn.q");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".ada.k1.grad.npy", true);
            //    blob = net.layers[20].internal_blobs.FindBlob("mh_attn.k");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".ada.v1.grad.npy", true);
            //    blob = net.layers[20].internal_blobs.FindBlob("mh_attn.v");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".ada.enriched_sequence_a.grad.npy", true);
            //    blob = net.FindBlob("enr_seq_a");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".enriched_sequence.grad.npy", true);
            //    blob = net.FindBlob("enriched_sequence");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));
            //}

            //if (nStart <= 15)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".gated_lstm_output.grad.npy", true);
            //    blob = net.FindBlob("gated_lstm_output");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));
            //}

            //if (nStart <= 12)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".hist.temporal_selection_output.YYY.grad.npy", true);
            //    blob = net.FindBlob("selected_hist");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".hist.temporal_selection_output1.ZZ.grad.npy", true);
            //    blob = net.FindBlob("selected_hist1");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".hist.temporal_flattened_embedding.grad.npy", true);
            //    blob = net.FindBlob("hist_ts_rep1");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".hist.time_distributed_context.grad.npy", true);
            //    blob = net.FindBlob("c_selection1h");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));
            //}

            //if (nStart <= 9)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".c_enrichment.XX.grad.npy", true);
            //    blob = net.FindBlob("c_enrichment");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".c_seq_hidden.XX.grad.npy", true);
            //    blob = net.FindBlob("c_seq_hidden");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".c_seq_cell.XX.grad.npy", true);
            //    blob = net.FindBlob("c_seq_cell");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));
            //}

            //if (nStart <= 4)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".selected_static.grad.npy", true);
            //    blob = net.FindBlob("selected_static");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));
            //}

            //if (nStart <= 3)
            //{
            //    blobVal.LoadFromNumpy(strPath + strTag + ".historical_ts_rep.grad.npy", true);
            //    blob = net.FindBlob("hist_ts_rep");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));

            //    blobVal.LoadFromNumpy(strPath + strTag + ".static_rep.grad.npy", true);
            //    blob = net.FindBlob("static_rep");
            //    Trace.Assert(blobVal.Compare(blob, blobWork, true));
            //}

            BlobCollection<float> colBtm = new BlobCollection<float>();
            //colBtm.Add(net.FindBlob("x_numeric_static"));
            //colBtm.Add(net.FindBlob("x_categorical_static"));
            //colBtm.Add(net.FindBlob("x_numeric_hist"));
            //colBtm.Add(net.FindBlob("x_categorical_hist"));
            //colBtm.Add(net.FindBlob("x_numeric_future"));
            //colBtm.Add(net.FindBlob("x_categorical_future"));

            if (nStart == 0) // at input
            {
                colBtm.Add(net.FindBlob("x_categorical_static"));
                colBtm.Add(net.FindBlob("x_numeric_hist"));
                colBtm.Add(m_blobTmp);
            }
            else if (nStart == 3) // at stat selection
            {
                colBtm.Add(net.FindBlob("static_rep"));
                colBtm.Add(net.FindBlob("hist_ts_rep"));
            }
            else if (nStart == 4) // at stat selection
            {
                colBtm.Add(net.FindBlob("selected_static"));
                colBtm.Add(net.FindBlob("hist_ts_rep"));
            }
            else if (nStart == 9) // at hist selection
            {
                colBtm.Add(net.FindBlob("hist_ts_rep"));
                colBtm.Add(net.FindBlob("c_selection"));
                colBtm.Add(net.FindBlob("c_seq_hidden"));
                colBtm.Add(net.FindBlob("c_seq_cell"));
                colBtm.Add(net.FindBlob("c_enrichment"));
            }
            else if (nStart == 12) // at static enrichment
            {
                colBtm.Add(net.FindBlob("selected_hist"));
                colBtm.Add(net.FindBlob("c_seq_hidden"));
                colBtm.Add(net.FindBlob("c_seq_cell"));
                colBtm.Add(net.FindBlob("c_enrichment"));
            }
            else if (nStart == 15) // at static enrichment
            {
                colBtm.Add(net.FindBlob("gated_lstm_output"));
                colBtm.Add(net.FindBlob("c_enrichment"));
            }
            else if (nStart == 19) // at attention
            {
                colBtm.Add(net.FindBlob("enriched_sequence"));
                colBtm.Add(net.FindBlob("glstmout_b"));
            }
            else if (nStart == 23) // at gated post attention
            {
                colBtm.Add(net.FindBlob("gated_post_attention"));
                colBtm.Add(net.FindBlob("glstmout_b"));
            }
            else if (nStart == 24) // at post poswise ff
            {
                colBtm.Add(net.FindBlob("post_poswise_ff_grn"));
                colBtm.Add(net.FindBlob("glstmout_b"));
            }
            else if (nStart == 25) // at output IP
            {
                colBtm.Add(net.FindBlob("gated_poswise_ff"));
            }
            else if (nStart == 27) // at Loss
            {
                colBtm.Add(net.FindBlob("predicted_quantiles"));
            }

            if (colBtm.Count > 0)
                return get_return_values(colBtm, true);

            return null;
        }

        public void model_update(int nIter)
        {
            Solver<float> solver = m_mycaffe.GetInternalSolver();
            solver.ApplyUpdate(nIter);
        }

        public float[] sum(int nN, int nC, int nH, int nW, float[] rgIn)
        {
            BlobCollection<float> colBtm = new BlobCollection<float>();
            CudaDnn<float> cuda = m_mycaffe.Cuda;

            if (m_blobBtm.gpu_data == 0)
                m_blobBtm = m_mycaffe.CreateBlob("btm");

            if (m_blobTop.gpu_data == 0)
                m_blobTop = m_mycaffe.CreateBlob("top");

            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobTop.Reshape(nN, nC, 1, 1);

            colBtm.Add(m_blobBtm);
            load_input_values(rgIn, colBtm, false);

            int nCount = m_blobBtm.count();
            int nInnerNum = m_blobBtm.count(2);

            cuda.channel_sum(nCount, nN, nC, nInnerNum, m_blobBtm.gpu_data, m_blobTop.mutable_gpu_data);

            colBtm.Clear();
            colBtm.Add(m_blobTop);

            return get_return_values(colBtm, false);
        }

        public float[] lstm_wts(string strTag)
        {
            Layer<float> layer = findLayer(LayerParameter.LayerType.LSTM, strTag);
            int nCount = layer.blobs.Count;
            List<int> rgCounts = new List<int>();
            List<float> rgData = new List<float>();

            for (int i = 0; i < nCount; i++)
            {
                int nCount1 = layer.blobs[i].count();
                rgCounts.Add(nCount1);

                float[] rgData1 = layer.blobs[i].mutable_cpu_data;
                rgData.AddRange(rgData1);
            }

            for (int i = rgCounts.Count - 1; i >= 0; i--)
            {
                rgData.Insert(0, rgCounts[i]);
            }

            rgData.Insert(0, nCount);

            return rgData.ToArray();
        }

        public float[] lstm_grad(string strTag)
        {
            Layer<float> layer = findLayer(LayerParameter.LayerType.LSTM, strTag);
            int nCount = layer.blobs.Count;
            List<int> rgCounts = new List<int>();
            List<float> rgData = new List<float>();

            for (int i = 0; i < nCount; i++)
            {
                int nCount1 = layer.blobs[i].count();
                rgCounts.Add(nCount1);

                float[] rgData1 = layer.blobs[i].mutable_cpu_diff;
                rgData.AddRange(rgData1);
            }

            for (int i = rgCounts.Count - 1; i >= 0; i--)
            {
                rgData.Insert(0, rgCounts[i]);
            }

            rgData.Insert(0, nCount);

            return rgData.ToArray();
        }

        public float[] lstm_fwd(string strTag, int nStateSize, int nNumLayers, int nN, int nC, int nH, float[] rg, float[] rgH, float[] rgC)
        {
            m_blobBtm.Reshape(new List<int>() { nN, nC, nH });
            m_blobBtm.mutable_cpu_data = rg;
            m_blobBtm1.Reshape(new List<int>() { nN, nC });
            m_blobBtm1.SetData(1);
            m_blobBtm2.Reshape(new List<int>() { nNumLayers, nN, nH });
            m_blobBtm2.mutable_cpu_data = rgH;
            m_blobBtm3.Reshape(new List<int>() { nNumLayers, nN, nH });
            m_blobBtm3.mutable_cpu_data = rgC;

            m_colBtm.Add(m_blobBtm1);
            m_colBtm.Add(m_blobBtm2);
            m_colBtm.Add(m_blobBtm3);
            m_colTop.Add(m_blobTop1);
            m_colTop.Add(m_blobTop2);

            Layer<float> layer = findLayer(LayerParameter.LayerType.LSTM, strTag);
            layer.Reshape(m_colBtm, m_colTop);
            layer.Forward(m_colBtm, m_colTop);

            while (m_colBtm.Count > 1)
            {
                m_colBtm.RemoveAt(1);
            }

            while (m_colTop.Count > 1)
            {
                m_colTop.RemoveAt(1);
            }

            float[] rgData = m_blobTop.mutable_cpu_data;
            float[] rgHidden = m_blobTop1.mutable_cpu_data;
            float[] rgCell = m_blobTop2.mutable_cpu_data;
            List<float> rgAll = new List<float>();
            rgAll.Add(rgData.Length);
            rgAll.Add(rgHidden.Length);
            rgAll.AddRange(rgData);
            rgAll.AddRange(rgHidden);
            rgAll.AddRange(rgCell);

            return rgAll.ToArray();
        }

        public float[] lstm_bwd(string strTag, int nNumLayers, int nN, int nC, int nH, float[] rgY, float[] rgH, float[] rgC, float[] rgYGrad, float[] rgHygrad, float[] rgCygrad)
        {
            List<int> rgShape = new List<int>() { nN, nC, nH };
            m_blobTop.Reshape(rgShape);
            m_blobBtm.Reshape(rgShape);
            m_blobBtm1.Reshape(new List<int>() { nN, nC });
            m_blobBtm1.SetData(1);
            m_blobBtm2.Reshape(new List<int>() { nNumLayers, nN, nH });
            m_blobBtm2.mutable_cpu_data = rgH;
            m_blobBtm3.Reshape(new List<int>() { nNumLayers, nN, nH });
            m_blobBtm3.mutable_cpu_data = rgC;

            m_colBtm.Add(m_blobBtm1);
            m_colBtm.Add(m_blobBtm2);
            m_colBtm.Add(m_blobBtm3);

            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            if (rgHygrad != null && rgCygrad != null)
            {
                m_blobTop1.mutable_cpu_diff = rgHygrad;
                m_blobTop2.mutable_cpu_diff = rgCygrad;

                if (m_blobTop1.asum_diff() != 0 || m_blobTop2.asum_diff() != 0)
                { 
                    m_colTop.Add(m_blobTop1);
                    m_colTop.Add(m_blobTop2);
                }
            }

            Layer<float> layer = findLayer(LayerParameter.LayerType.LSTM, strTag);
            layer.Backward(m_colTop, new List<bool>() { true, true, true }, m_colBtm);

            while (m_colBtm.Count > 1)
            {
                m_colBtm.RemoveAt(1);
            }

            while (m_colTop.Count > 1)
            {
                m_colTop.RemoveAt(1);
            }

            float[] rgBtm0 = m_blobBtm.mutable_cpu_diff;
            float[] rgBtm2 = m_blobBtm2.mutable_cpu_diff;
            float[] rgBtm3 = m_blobBtm3.mutable_cpu_diff;

            List<float> rgAll = new List<float>();
            rgAll.Add(rgBtm0.Length);
            rgAll.Add(rgBtm2.Length);
            rgAll.AddRange(rgBtm0);
            rgAll.AddRange(rgBtm2);
            rgAll.AddRange(rgBtm3);
            
            return rgAll.ToArray();
        }

        public void update_wts(int i)
        {
            Solver<float> solver = m_mycaffe.GetInternalSolver();
            solver.ApplyUpdate(i + 1);
        }

        public float[] elu_fwd(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.ELU);
                p.elu_param.engine = EngineParameter.Engine.CAFFE;
                p.elu_param.alpha = 1;
                Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                layer1.Setup(m_colBtm, m_colTop);
                m_rgLayers.Add(strTag, layer1);
            }

            Layer<float> layer = m_rgLayers[strTag];
            layer.Forward(m_colBtm, m_colTop);

            return m_blobTop.mutable_cpu_data;
        }

        public float[] elu_bwd(string strTag, int nN, int nC, int nH, int nW, float[] rgY, float[] rgYGrad, float[] rgX)
        {
            m_blobTop.Reshape(nN, nC, nH, nW);
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobBtm.mutable_cpu_data = rgX;
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        public float[] sigmoid_fwd(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SIGMOID);
                p.sigmoid_param.engine = EngineParameter.Engine.DEFAULT;
                Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                layer1.Setup(m_colBtm, m_colTop);
                m_rgLayers.Add(strTag, layer1);
            }

            Layer<float> layer = m_rgLayers[strTag];
            layer.Forward(m_colBtm, m_colTop);

            return m_blobTop.mutable_cpu_data;
        }

        public float[] sigmoid_bwd(string strTag, int nN, int nC, int nH, int nW, float[] rgY, float[] rgYGrad)
        {
            m_blobTop.Reshape(nN, nC, nH, nW);
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        public float[] innerproduct_wts(string strTag)
        {
            Layer<float> layer = findLayer(LayerParameter.LayerType.INNERPRODUCT, strTag);
            if (layer == null)
                layer = m_rgLayers[strTag];

            int nCount = layer.blobs.Count;
            List<int> rgCounts = new List<int>();
            List<float> rgData = new List<float>();

            for (int i = 0; i < nCount; i++)
            {
                int nCount1 = layer.blobs[i].count();
                rgCounts.Add(nCount1);

                float[] rgData1 = layer.blobs[i].mutable_cpu_data;
                rgData.AddRange(rgData1);
            }

            for (int i = rgCounts.Count - 1; i >= 0; i--)
            {
                rgData.Insert(0, rgCounts[i]);
            }

            rgData.Insert(0, nCount);

            return rgData.ToArray();
        }

        public void innerproduct_init(string strTag, bool bBias, int nAxis, int nNumOut, int nN, int nC, int nH, int nW, float[] rgWt, float[] rgB)
        {
            m_blobBtm.Reshape(nN, nC, nH, nW);

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
                p.inner_product_param.num_output = (uint)nNumOut;
                p.inner_product_param.axis = nAxis;
                p.inner_product_param.bias_term = bBias;
                Layer<float> layer = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                layer.Setup(m_colBtm, m_colTop);
                m_rgLayers.Add(strTag, layer);

                layer.blobs[0].mutable_cpu_data = rgWt;
                if (bBias)
                    layer.blobs[1].mutable_cpu_data = rgB;
            }
        }

        public float[] innerproduct_fwd(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobBtm.mutable_cpu_data = rg;

            Layer<float> layer = findLayer(LayerParameter.LayerType.INNERPRODUCT, strTag);
            if (layer == null)
                layer = m_rgLayers[strTag];

            layer.Forward(m_colBtm, m_colTop);

            return m_blobTop.mutable_cpu_data;
        }

        public float[] innerproduct_bwd(string strTag, int nN, int nCy, int nHy, int nWy, int nCx, int nHx, int nWx, float[] rgY, float[] rgYGrad)
        {
            m_blobTop.Reshape(nN, nCy, nHy, nWy);
            m_blobBtm.Reshape(nN, nCx, nHx, nWx);
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = findLayer(LayerParameter.LayerType.INNERPRODUCT, strTag);
            if (layer == null)
                layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        public float[] embedding_wts(string strTag)
        {
            Layer<float> layer = findLayer(LayerParameter.LayerType.EMBED, strTag);
            if (layer == null)
                layer = m_rgLayers[strTag];

            int nCount = layer.blobs.Count;
            List<int> rgCounts = new List<int>();
            List<float> rgData = new List<float>();

            for (int i = 0; i < nCount; i++)
            {
                int nCount1 = layer.blobs[i].count();
                rgCounts.Add(nCount1);

                float[] rgData1 = layer.blobs[i].mutable_cpu_data;
                rgData.AddRange(rgData1);
            }

            for (int i = rgCounts.Count - 1; i >= 0; i--)
            {
                rgData.Insert(0, rgCounts[i]);
            }

            rgData.Insert(0, nCount);

            return rgData.ToArray();
        }

        public float[] embedding_fwd(string strTag, int nNumOut, int nN, int nC, int nH, int nW, float[] rg)
        {
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobBtm.mutable_cpu_data = rg;

            Layer<float> layer = findLayer(LayerParameter.LayerType.EMBED, strTag);
            if (layer == null)
            {
                if (!m_rgLayers.ContainsKey(strTag))
                {
                    LayerParameter p = new LayerParameter(LayerParameter.LayerType.EMBED);
                    p.embed_param.num_output = (uint)nNumOut;
                    layer = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                    layer.Setup(m_colBtm, m_colTop);
                    m_rgLayers.Add(strTag, layer);
                }
                else
                {
                    layer = m_rgLayers[strTag];
                }
            }

            layer.Forward(m_colBtm, m_colTop);

            return m_blobTop.mutable_cpu_data;
        }

        public float[] embedding_bwd(string strTag, int nN, int nCy, int nHy, int nWy, int nCx, int nHx, int nWx, float[] rgY, float[] rgYGrad)
        {
            m_blobTop.Reshape(nN, nCy, nHy, nWy);
            m_blobBtm.Reshape(nN, nCx, nHx, nWx);
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = findLayer(LayerParameter.LayerType.EMBED, strTag);
            if (layer == null)
                layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        public float[] softmax_fwd(string strTag, int nAxis, int nN, int nC, int nH, int nW, float[] rg)
        {
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
                p.softmax_param.engine = EngineParameter.Engine.CUDNN;
                p.softmax_param.axis = nAxis;
                Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                layer1.Setup(m_colBtm, m_colTop);
                m_rgLayers.Add(strTag, layer1);
            }

            Layer<float> layer = m_rgLayers[strTag];
            layer.Forward(m_colBtm, m_colTop);

            return m_blobTop.mutable_cpu_data;
        }

        public float[] softmax_bwd(string strTag, int nN, int nC, int nH, int nW, float[] rgY, float[] rgYGrad)
        {
            m_blobTop.Reshape(nN, nC, nH, nW);
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        public float[] layernorm_fwd(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            List<int> rgShape = new List<int>() { nN, nC };
            if (nH > 1)
            {
                rgShape.Add(nH);
                if (nW > 1)
                    rgShape.Add(nW);
            }

            m_blobBtm.Reshape(rgShape);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
                p.layer_norm_param.enable_cuda_impl = false;
                Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                layer1.Setup(m_colBtm, m_colTop);
                m_rgLayers.Add(strTag, layer1);
            }

            Layer<float> layer = m_rgLayers[strTag];
            layer.Forward(m_colBtm, m_colTop);

            return m_blobTop.mutable_cpu_data;
        }

        public float[] layernorm_bwd(string strTag, int nN, int nC, int nH, int nW, float[] rgY, float[] rgYGrad)
        {
            List<int> rgShape = new List<int>() { nN, nC };
            if (nH > 1)
            {
                rgShape.Add(nH);
                if (nW > 1)
                    rgShape.Add(nW);
            }

            m_blobTop.Reshape(rgShape);
            m_blobBtm.Reshape(rgShape);
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = m_rgLayers[strTag];
            layer.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            return m_blobBtm.mutable_cpu_diff;
        }

        private Layer<float> findLayer(LayerParameter.LayerType type, string strTag)
        {
            if (m_rgTagLookup.ContainsKey(strTag))
                strTag = m_rgTagLookup[strTag];

            Layer<float> layer = m_net.FindLayer(type, strTag);
            return layer;
        }

        public float[] channel_sum_fwd(int nN, int nC, int nH, float[] rgX)
        {
            List<int> rgShapeB = new List<int>() { nN, nC, nH };
            m_blobBtm.Reshape(rgShapeB);
            List<int> rgShapeT = new List<int>() { nN, nC };
            m_blobTop.Reshape(rgShapeT);
            m_blobBtm.mutable_cpu_data = rgX;
            m_blobTop.SetData(0);
            m_mycaffe.Cuda.channel_sum(m_blobBtm.count(), nN, nC, nH, m_blobBtm.gpu_data, m_blobTop.mutable_gpu_data, false, DIR.FWD);
            return m_blobTop.mutable_cpu_data;
        }

        public float[] channel_sum_bwd(int nN, int nC, int nH, float[] rgY)
        {
            List<int> rgShapeB = new List<int>() { nN, nC, nH };
            m_blobBtm.Reshape(rgShapeB);
            List<int> rgShapeT = new List<int>() { nN, nC };
            m_blobTop.Reshape(rgShapeT);
            m_blobTop.mutable_cpu_diff = rgY;
            m_mycaffe.Cuda.channel_sum(m_blobBtm.count(), nN, nC, nH, m_blobBtm.mutable_gpu_diff, m_blobTop.gpu_diff, false, DIR.BWD);
            return m_blobBtm.mutable_cpu_diff;
        }

        public void save(int nIter)
        {
            SaveWeights("C:\\temp\\projects\\TransformerTranslator\\TransformerTranslator\\state\\", nIter);
        }

        public void SaveWeights(string strPath, int nIter)
        {
            string strDir = Directory.GetCurrentDirectory();
            try
            {
                Directory.SetCurrentDirectory(AssemblyDirectory);
                
                byte[] rgb = m_mycaffe.GetWeights();
                File.WriteAllBytes(strPath + nIter.ToString() + "_mycaffe.mycaffemodel", rgb);

                Net<float> net = m_mycaffe.GetInternalNet(Phase.TRAIN);
                RawProto proto = net.net_param.ToProto("root");
                string strModel = proto.ToString();
                File.WriteAllText(strPath + "mycaffe.prototxt", strModel);

                Solver<float> solver = m_mycaffe.GetInternalSolver();
                RawProto proto2 = solver.parameter.ToProto("root");
                string strSolver = proto2.ToString();
                File.WriteAllText(strPath + "mycaffe.solver.prototxt", strSolver);
            }
            finally
            {
                Directory.SetCurrentDirectory(strDir);
            }
        }

        public float CurrentLoss
        {
            get
            {
                Solver<float> solver = m_mycaffe.GetInternalSolver();
                return (float)solver.smoothed_loss;
            }
        }

        public float CurrentAccuracy
        {
            get { return m_fLastAccuracy; }
        }

        private void dispose(ref Blob<float> b)
        {
            if (b != null)
                b.Dispose();
            b = null;
        }

        public void CleanUp()
        {
            dispose(ref m_blobBtm);
            dispose(ref m_blobBtm1);
            dispose(ref m_blobBtm2);
            dispose(ref m_blobBtm3);
            dispose(ref m_blobTop);
            dispose(ref m_blobTop1);
            dispose(ref m_blobTop2);
            dispose(ref m_blobWork);
            dispose(ref m_blobVal);

            if (m_mycaffe != null)
                m_mycaffe.Dispose();            
        }
    }
}
