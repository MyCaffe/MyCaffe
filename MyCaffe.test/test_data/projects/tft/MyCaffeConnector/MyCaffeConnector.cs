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

namespace MyCaffeConnector
{
    public class MyCaffeConnector : IDisposable
    {
        CancelEvent m_evtCancel = new CancelEvent();
        Log m_log = new Log("MyCaffeConnector");
        MyCaffeControl<float> m_mycaffe;
        Dictionary<string, Layer<float>> m_rgLayers = new Dictionary<string, Layer<float>>();
        BlobCollection<float> m_colTop = new BlobCollection<float>();
        BlobCollection<float> m_colBtm = new BlobCollection<float>();
        Blob<float> m_blobEncIn;
        Blob<float> m_blobDecIn;
        Blob<float> m_blobDecOut;
        Blob<float> m_blobEncMask;
        Blob<float> m_blobDecMask;
        Blob<float> m_blobLoss;
        Blob<float> m_blobBtm;
        Blob<float> m_blobBtm1;
        Blob<float> m_blobBtm2;
        Blob<float> m_blobBtm3;
        Blob<float> m_blobTop;
        Blob<float> m_blobTop1;
        Blob<float> m_blobTop2;
        Dictionary<string, Blob<float>> m_rgBlobM = new Dictionary<string, Blob<float>>();
        Dictionary<string, Blob<float>> m_rgBlobV = new Dictionary<string, Blob<float>>();
        float m_fLastAccuracy = 0;
        
        public MyCaffeConnector()
        {
            SettingsCaffe s = new SettingsCaffe() { GpuIds = "0" };
            string strCudaPath = "C:\\Program Files\\SignalPop\\MyCaffe\\cuda_11.8\\CudaDnnDll.11.8.dll";
            m_mycaffe = new MyCaffeControl<float>(s, m_log, m_evtCancel, null, null, null, null, strCudaPath, true);

            m_blobBtm = m_mycaffe.CreateBlob("btm");
            m_blobBtm1 = m_mycaffe.CreateBlob("btm1");
            m_blobBtm2 = m_mycaffe.CreateBlob("btm2");
            m_blobBtm3 = m_mycaffe.CreateBlob("btm3");
            m_blobTop = m_mycaffe.CreateBlob("top");
            m_blobTop1 = m_mycaffe.CreateBlob("top1");
            m_blobTop2 = m_mycaffe.CreateBlob("top2");
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

        private string buildSolver()
        {
            SolverParameter solver = new SolverParameter();
            solver.base_lr = 0.001;
            solver.type = SolverParameter.SolverType.ADAMW;
            solver.lr_policy = "fixed";
            solver.test_initialization = false;
            solver.weight_decay = 0;
            solver.adamw_decay = 0;
            solver.test_interval = 10000;
            solver.momentum = 0.9;
            solver.momentum2 = 0.999;
            solver.test_iter.Clear();

            return solver.ToProto("root").ToString();
        }

        private string buildModelEx(NetParameter net, uint nBatch, uint nBlockSize, uint nEmbed, uint nEncVocabSize, uint nDecVocabSize, double dfDropout, bool bAddInput = false, Phase phase = Phase.TRAIN)
        {
            if (bAddInput)
            {
                LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
                input.name = "input";
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize } });
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize } });
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize } });
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize } });
                input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { (int)nBatch, (int)nBlockSize, (int)nBlockSize } });
                input.top.Add("enc");
                input.top.Add("dec");
                input.top.Add("tgt");
                input.top.Add("emsk");
                input.top.Add("dmsk");
                net.layer.Add(input);
            }

            LayerParameter emb1 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb1.name = "embed1";
            emb1.embed_param.bias_term = false;
            emb1.embed_param.input_dim = nEncVocabSize;
            emb1.embed_param.num_output = nEmbed;
            emb1.bottom.Add("enc");
            emb1.top.Add("emb1");
            net.layer.Add(emb1);

            LayerParameter emb2 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb2.name = "embed2";
            emb2.embed_param.bias_term = false;
            emb2.embed_param.input_dim = nDecVocabSize;
            emb2.embed_param.num_output = nEmbed;
            emb2.bottom.Add("dec");
            emb2.top.Add("emb2");
            net.layer.Add(emb2);

            LayerParameter pos1 = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODER);
            pos1.positional_encoder_param.block_size = nBlockSize;
            pos1.positional_encoder_param.embed = nEmbed;
            pos1.name = "posenc1";
            pos1.bottom.Add("emb1");
            pos1.top.Add("pos1");
            net.layer.Add(pos1);

            LayerParameter pos2 = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODER);
            pos2.positional_encoder_param.block_size = nBlockSize;
            pos2.positional_encoder_param.embed = nEmbed;
            pos2.name = "posenc2";
            pos2.bottom.Add("emb2");
            pos2.top.Add("pos2");
            net.layer.Add(pos2);

            string strEncBtm = "pos1";
            int nLayers = 6;
            for (int i = 0; i < nLayers; i++)
            {
                LayerParameter enc = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                enc.name = "enc" + (i + 1).ToString();
                enc.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.ENCODER;
                enc.transformer_block_param.heads = 8;
                enc.transformer_block_param.embed = nEmbed;
                enc.transformer_block_param.block_size = nBlockSize;
                enc.transformer_block_param.layers = (uint)nLayers;
                enc.transformer_block_param.activation = TransformerBlockParameter.ACTIVATION.RELU;
                enc.transformer_block_param.attn_dropout = dfDropout;
                enc.transformer_block_param.resid_dropout = dfDropout;
                enc.bottom.Add(strEncBtm);
                enc.bottom.Add("emsk");
                enc.top.Add(enc.name);
                net.layer.Add(enc);

                strEncBtm = enc.name;
            }

            LayerParameter ln1 = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            ln1.name = "ln1";
            ln1.layer_norm_param.enable_cuda_impl = false;
            ln1.bottom.Add(strEncBtm);
            ln1.top.Add("ln1");
            net.layer.Add(ln1);

            string strDecBtm = "pos2";
            for (int i = 0; i < nLayers; i++)
            {
                LayerParameter dec = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                dec.name = "dec" + (i + 1).ToString();
                dec.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.DECODER;
                dec.transformer_block_param.heads = 8;
                dec.transformer_block_param.embed = nEmbed;
                dec.transformer_block_param.block_size = nBlockSize;
                dec.transformer_block_param.layers = (uint)nLayers;
                dec.transformer_block_param.activation = TransformerBlockParameter.ACTIVATION.RELU;
                dec.transformer_block_param.attn_dropout = dfDropout;
                dec.transformer_block_param.resid_dropout = dfDropout;
                dec.bottom.Add(strDecBtm);
                dec.bottom.Add("dmsk");
                dec.bottom.Add("ln1");
                dec.bottom.Add("emsk");
                dec.top.Add(dec.name);
                net.layer.Add(dec);

                strDecBtm = dec.name;
            }

            LayerParameter ln2 = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            ln2.name = "ln2";
            ln2.layer_norm_param.enable_cuda_impl = false;
            ln2.bottom.Add(strDecBtm);
            ln2.top.Add("ln2");
            net.layer.Add(ln2);

            LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ip1.name = "ip1";
            ip1.inner_product_param.axis = 2;
            ip1.inner_product_param.num_output = nDecVocabSize;
            ip1.bottom.Add("ln2");
            ip1.top.Add("logits");
            net.layer.Add(ip1);

            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            softmax.name = "softmax";
            softmax.softmax_param.axis = 2;
            softmax.softmax_param.algorithm = SOFTMAX_ALGORITHM.LOG;
            softmax.softmax_param.algorithm_train = SOFTMAX_ALGORITHM.LOG;
            softmax.bottom.Add("logits");
            softmax.top.Add("prob");
            net.layer.Add(softmax);

            if (phase == Phase.TRAIN)
            {
                LayerParameter loss = new LayerParameter(LayerParameter.LayerType.NLL_LOSS);
                loss.name = "loss";
                loss.nll_loss_param.axis = 2;
                loss.loss_param.normalization = LossParameter.NormalizationMode.VALID;
                loss.bottom.Add("prob");
                loss.bottom.Add("tgt");
                loss.top.Add("loss");
                loss.include.Add(new NetStateRule(Phase.TRAIN));
                net.layer.Add(loss);
            }

            if (phase == Phase.TRAIN)
            {
                LayerParameter accuracy = new LayerParameter(LayerParameter.LayerType.ACCURACY);
                accuracy.name = "accuracy";
                accuracy.accuracy_param.axis = 2;
                accuracy.accuracy_param.ignore_labels.Add(0);
                accuracy.accuracy_param.enable_simple_accuracy = true;
                accuracy.bottom.Add("prob");
                accuracy.bottom.Add("tgt");
                accuracy.top.Add("accuracy");
                accuracy.include.Add(new NetStateRule(Phase.TRAIN));
                net.layer.Add(accuracy);
            }

            return net.ToProto("root").ToString();
        }

        private string buildModel(bool bAddDataLayer, int nNumSamples, int nNumHeads, float fDropout, int nLstmLayers, int nNumOutputs, int nStateSize, int nNumHistSteps, int nNumFutureSteps,
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
                data.data_temporal_param.source = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita";
                data.data_temporal_param.source_type = DataTemporalParameter.SOURCE_TYPE.PATH_NPY_FILE;
                data.data_temporal_param.shuffle_data = false;
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
            }
            else
            {
                LayerParameter data = new LayerParameter(LayerParameter.LayerType.INPUT);
                data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumStaticNumeric }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumStaticCategorical }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHistSteps, nNumHistNumeric }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHistSteps, nNumHistCategorical }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFutureSteps, nNumFutureNumeric }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFutureSteps, nNumFutureCategorical }));
                data.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFutureSteps }));
                data.top.Add("x_numeric_static");
                data.top.Add("x_categorical_static");
                data.top.Add("x_numeric_hist");
                data.top.Add("x_categorical_hist");
                data.top.Add("x_numeric_future");
                data.top.Add("x_categorical_future");
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

            LayerParameter c_sel_split = new LayerParameter(LayerParameter.LayerType.SPLIT, "c_sel_split");
            c_sel_split.bottom.Add("c_selection");
            c_sel_split.top.Add("c_selection_h");
            c_sel_split.top.Add("c_selection_f");
            p.layer.Add(c_sel_split);

            //---------------------------------
            //  Variable Selection Networks - Temporal
            //---------------------------------
            LayerParameter hist_vsh_reshape_before = new LayerParameter(LayerParameter.LayerType.RESHAPE_TEMPORAL, "reshtmp_hist_b");
            hist_vsh_reshape_before.reshape_temporal_param.mode = ReshapeTemporalParameter.MODE.BEFORE;
            hist_vsh_reshape_before.bottom.Add("hist_ts_rep");
            hist_vsh_reshape_before.bottom.Add("c_selection_h");
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


            //---------------------------------
            //  Locality Enhancement with Seq2Seq processing
            //---------------------------------
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

            LayerParameter past_lstm = new LayerParameter(LayerParameter.LayerType.LSTM, "past_lstm");
            past_lstm.recurrent_param.num_output = (uint)nStateSize;
            past_lstm.recurrent_param.num_layers = (uint)nLstmLayers;
            past_lstm.recurrent_param.dropout_ratio = fDropout;
            past_lstm.recurrent_param.expose_hidden_input = true;
            past_lstm.recurrent_param.expose_hidden_output = true;
            past_lstm.recurrent_param.batch_first = true;
            past_lstm.recurrent_param.auto_repeat_hidden_states_across_layers = true;
            past_lstm.recurrent_param.use_cudnn_rnn8_if_supported = true;
            past_lstm.recurrent_param.engine = EngineParameter.Engine.CUDNN;
            past_lstm.bottom.Add("selhist_b");
            past_lstm.bottom.Add("selected_hist_clip");
            past_lstm.bottom.Add("c_seq_hidden");
            past_lstm.bottom.Add("c_seq_cell");
            past_lstm.top.Add("past_lstm_output");
            past_lstm.top.Add("hidden1");
            past_lstm.top.Add("cell1");
            p.layer.Add(past_lstm);

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

            LayerParameter post_lstm_gating = new LayerParameter(LayerParameter.LayerType.GATEADDNORM, "post_lstm_gate");
            post_lstm_gating.dropout_param.dropout_ratio = fDropout;
            post_lstm_gating.layer_norm_param.enable_cuda_impl = false;
            post_lstm_gating.layer_norm_param.epsilon = 1e-10;
            post_lstm_gating.glu_param.input_dim = nStateSize;
            post_lstm_gating.glu_param.axis = 2;
            post_lstm_gating.bottom.Add("lstm_output");
            post_lstm_gating.bottom.Add("lstm_input");
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
            post_attn_gate.gateaddnorm_param.residual_channel_offset = nNumHistSteps;
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
            pos_wise_ff_gate.gateaddnorm_param.residual_channel_offset = nNumHistSteps;
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


            //---------------------------------
            //  Quartile Loss
            //---------------------------------
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

            return p.ToProto("root").ToString();
        }

        public void Initialize()
        {
            m_colTop.Clear();
            m_colTop.Add(m_blobTop);
            m_colBtm.Clear();
            m_colBtm.Add(m_blobBtm);
        }

        public void InitializeEx(uint nBatch, uint nBlockSize, uint nEmbed, uint nEncVocabSize, uint nDecVocabSize, double dfDropout)
        {
            NetParameter net_param = new NetParameter();
            string strSolver = buildSolver();
            string strModel = buildModelEx(net_param, nBatch, nBlockSize, nEmbed, nEncVocabSize, nDecVocabSize, dfDropout, true);

            m_mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);
            
            m_blobEncIn = m_mycaffe.CreateBlob("encin");
            m_blobEncIn.Reshape((int)nBatch, (int)nBlockSize, 1, 1);
            m_blobDecIn = m_mycaffe.CreateBlob("decin");
            m_blobDecIn.Reshape((int)nBatch, (int)nBlockSize, 1, 1);
            m_blobDecOut = m_mycaffe.CreateBlob("decout");
            m_blobDecOut.Reshape((int)nBatch, (int)nBlockSize, 1, 1);
            m_blobEncMask = m_mycaffe.CreateBlob("e_mask");
            m_blobEncMask.Reshape((int)nBatch, (int)nBlockSize, 1, 1);
            m_blobDecMask = m_mycaffe.CreateBlob("d_mask");
            m_blobDecMask.Reshape((int)nBatch, (int)nBlockSize, (int)nBlockSize, 1);
            m_blobLoss = m_mycaffe.CreateBlob("loss");
            m_blobLoss.Reshape(1, 1, 1, 1);
        }

        public void InitializeTFT()
        {
            int nNumSamples = 256;
            int nNumHeads = 4;
            int fDropout = 0;
            int nLstmLayers = 2;
            int nNumOutputs = 3;
            int nStateSize = 64;
            int nNumHistSteps = 90;
            int nNumFutureSteps = 30;
            int nNumStaticNumeric = 0;
            int nNumStaticCategorical = 9;
            int nNumHistoricalNumeric = 4;
            int nNumHistoricalCategorical = 7;
            int nNumFutureNumeric = 1;
            int nNumFutureCategorical = 7;
            List<int> rgStaticCardinalities = new List<int>() { 54, 3627, 23, 17, 6, 18, 33, 320, 3 };
            List<int> rgHistCardinalities = new List<int>() { 2, 3, 8, 13, 72, 6, 28 };
            List<int> rgFutureCardinalities = new List<int>() { 2, 3, 8, 13, 72, 6, 28 };

            string strSolver = buildSolver();
            string strModel = buildModel(true, nNumSamples, nNumHeads, fDropout, nLstmLayers, nNumOutputs, nStateSize, nNumHistSteps, nNumFutureSteps,
                                          nNumStaticNumeric, nNumStaticCategorical, rgStaticCardinalities,
                                          nNumHistoricalNumeric, nNumHistoricalCategorical, rgHistCardinalities,
                                          nNumFutureNumeric, nNumFutureCategorical, rgFutureCardinalities);

            try
            {
                m_mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);
                string strPath = "C:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\_weights_from_run\\";
                load_weights("tft.all", m_mycaffe.GetInternalNet(Phase.TRAIN), strPath, nNumStaticNumeric, nNumStaticCategorical, nNumHistoricalNumeric, nNumHistoricalCategorical, nNumFutureNumeric, nNumFutureCategorical);
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
        }

        private void load_weights(string strTag, Net<float> net, string strPath, int nNumStaticNumeric, int nNumStaticCategorical, int nNumHistNumeric, int nNumHistCategorical, int nNumFutureNumeric, int nNumFutureCategorical)
        {
            //-------------------------------------------
            // Load input channel embedding weights.
            //-------------------------------------------
            int nIdx = 0;
            for (int i = 0; i < nNumStaticCategorical; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_transform.categorical_transform.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                nIdx++;
            }

            for (int i = 0; i < nNumHistNumeric; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_transform.numeric_transform.module.numeric_projection_layers." + i.ToString() + ".weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_transform.numeric_transform.module.numeric_projection_layers." + i.ToString() + ".bias.npy");
                nIdx++;
            }

            for (int i = 0; i < nNumHistCategorical; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_transform.categorical_transform.module.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                nIdx++;
            }

            for (int i = 0; i < nNumFutureNumeric; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_transform.numeric_transform.module.numeric_projection_layers." + i.ToString() + ".weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_transform.numeric_transform.module.numeric_projection_layers." + i.ToString() + ".bias.npy");
                nIdx++;
            }

            for (int i = 0; i < nNumFutureCategorical; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_transform.categorical_transform.module.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                nIdx++;
            }

            //-------------------------------------------
            // Load varselnet weights - static (idx=33)
            //-------------------------------------------
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.flattened_grn.skip_layer.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.flattened_grn.skip_layer.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.flattened_grn.fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.flattened_grn.fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.flattened_grn.fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.flattened_grn.fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.flattened_grn.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.flattened_grn.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.flattened_grn.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.flattened_grn.gate.module.fc2.bias.npy");
            nIdx++;

            for (int i = 0; i < nNumStaticNumeric + nNumStaticCategorical; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_selection.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                nIdx++;
            }

            //---------------------------------
            //  Static covariate encoders (idx=115)
            //---------------------------------
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_selection.fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_selection.fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_selection.fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_selection.fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_selection.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_selection.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_selection.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_selection.gate.module.fc2.bias.npy");
            nIdx++;

            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_enrichment.fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_enrichment.fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_enrichment.fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_enrichment.fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_enrichment.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_enrichment.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_enrichment.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_enrichment.gate.module.fc2.bias.npy");
            nIdx++;

            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_cell_init.fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_cell_init.fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_cell_init.fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_cell_init.fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_cell_init.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_cell_init.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_cell_init.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_cell_init.gate.module.fc2.bias.npy");
            nIdx++;

            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_state_init.fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_state_init.fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_state_init.fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_state_init.fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_state_init.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_state_init.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_state_init.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_encoder_sequential_state_init.gate.module.fc2.bias.npy");
            nIdx++;

            //-------------------------------------------
            // Load varselnet weights - historical (idx=147)
            //-------------------------------------------
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.flattened_grn.skip_layer.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.flattened_grn.skip_layer.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.flattened_grn.fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.flattened_grn.fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.flattened_grn.context_projection.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.flattened_grn.fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.flattened_grn.fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.flattened_grn.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.flattened_grn.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.flattened_grn.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.flattened_grn.gate.module.fc2.bias.npy");
            nIdx++;

            for (int i = 0; i < nNumHistNumeric + nNumHistCategorical; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".historical_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                nIdx++;
            }

            //-------------------------------------------
            // Load varselnet weights - future (idx=246)
            //-------------------------------------------
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.flattened_grn.skip_layer.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.flattened_grn.skip_layer.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.flattened_grn.fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.flattened_grn.fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.flattened_grn.context_projection.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.flattened_grn.fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.flattened_grn.fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.flattened_grn.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.flattened_grn.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.flattened_grn.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.flattened_grn.gate.module.fc2.bias.npy");
            nIdx++;

            for (int i = 0; i < nNumFutureNumeric + nNumFutureCategorical; i++)
            {
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                nIdx++;
                net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".future_ts_selection.single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                nIdx++;
            }

            //---------------------------------
            //  Locality Enhancement with Seq2Seq processing (idx=321)
            //---------------------------------
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".ZZZ.YYY.past_lstm.lstm.wt0.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".ZZZ.YYY.future_lstm.lstm.wt0.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".post_lstm_gating.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".post_lstm_gating.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".post_lstm_gating.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".post_lstm_gating.gate.module.fc2.bias.npy");
            nIdx++;


            //---------------------------------
            //  Temporal Static Enrichment (idx=327)
            //---------------------------------
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_enrichment_grn.fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_enrichment_grn.fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_enrichment_grn.context_projection.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_enrichment_grn.fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_enrichment_grn.fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_enrichment_grn.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_enrichment_grn.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_enrichment_grn.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".static_enrichment_grn.gate.module.fc2.bias.npy");
            nIdx++;


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
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".post_attention_gating.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".post_attention_gating.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".post_attention_gating.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".post_attention_gating.gate.module.fc2.bias.npy");
            nIdx++;

            //---------------------------------
            //  Pos wise FF (idx=348)
            //---------------------------------
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".pos_wise_ff_grn.fc1.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".pos_wise_ff_grn.fc1.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".pos_wise_ff_grn.fc2.module.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".pos_wise_ff_grn.fc2.module.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".pos_wise_ff_grn.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".pos_wise_ff_grn.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".pos_wise_ff_grn.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".pos_wise_ff_grn.gate.module.fc2.bias.npy");
            nIdx++;

            //---------------------------------
            //  Pos wise FF Gate (idx=356)
            //---------------------------------
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".pos_wise_ff_gating.gate.module.fc1.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".pos_wise_ff_gating.gate.module.fc1.bias.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".pos_wise_ff_gating.gate.module.fc2.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".pos_wise_ff_gating.gate.module.fc2.bias.npy");
            nIdx++;

            //---------------------------------
            //  Output (idx=360)
            //---------------------------------
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".output_layer.weight.npy");
            nIdx++;
            net.parameters[nIdx].LoadFromNumpy(strPath + strTag + ".output_layer.bias.npy");
            nIdx++;
        }

        public float[] get_return_values(BlobCollection<float> colBlobs, bool bDiff)
        {
            List<float> rgData = new List<float>();
            rgData.Add(colBlobs.Count);

            for (int i = 0; i < colBlobs.Count; i++)
            {
                int nCount = colBlobs[i].count();
                rgData.Add(nCount);
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
                    if (bDiff)
                        rgData.AddRange(colBlobs[i].mutable_cpu_diff);
                    else
                        rgData.AddRange(colBlobs[i].mutable_cpu_data);
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

            if (rgIn != null)
            {
                BlobCollection<float> colBtm = new BlobCollection<float>();
                //colBtm.Add(net.FindBlob("x_numeric_static"));
                //colBtm.Add(net.FindBlob("x_categorical_static"));
                //colBtm.Add(net.FindBlob("x_numeric_hist"));
                //colBtm.Add(net.FindBlob("x_categorical_hist"));
                //colBtm.Add(net.FindBlob("x_numeric_future"));
                //colBtm.Add(net.FindBlob("x_categorical_future"));

                if (nStart == 4) // just after input transform
                {
                    colBtm.Add(net.FindBlob("static_rep"));
                    colBtm.Add(net.FindBlob("hist_ts_rep"));
                    colBtm.Add(net.FindBlob("future_ts_rep"));
                    colBtm.Add(net.FindBlob("target"));
                }
                else if (nStart == 10) // split before hist vsn
                {
                    colBtm.Add(net.FindBlob("hist_ts_rep"));
                    colBtm.Add(net.FindBlob("future_ts_rep"));
                    colBtm.Add(net.FindBlob("c_selection"));
                    colBtm.Add(net.FindBlob("c_seq_hidden"));
                    colBtm.Add(net.FindBlob("c_seq_cell"));
                    colBtm.Add(net.FindBlob("c_enrichment"));
                    colBtm.Add(net.FindBlob("target"));
                }
                else if (nStart == 11) // hist vsn
                {
                    colBtm.Add(net.FindBlob("hist_ts_rep"));
                    colBtm.Add(net.FindBlob("c_selection_h"));
                    colBtm.Add(net.FindBlob("future_ts_rep"));
                    colBtm.Add(net.FindBlob("c_selection_f"));
                    colBtm.Add(net.FindBlob("c_seq_hidden"));
                    colBtm.Add(net.FindBlob("c_seq_cell"));
                    colBtm.Add(net.FindBlob("c_enrichment"));
                    colBtm.Add(net.FindBlob("target"));
                }
                else if (nStart == 14) // future vsn
                {
                    colBtm.Add(net.FindBlob("selected_hist"));
                    colBtm.Add(net.FindBlob("future_ts_rep"));
                    colBtm.Add(net.FindBlob("c_selection_f"));
                    colBtm.Add(net.FindBlob("c_seq_hidden"));
                    colBtm.Add(net.FindBlob("c_seq_cell"));
                    colBtm.Add(net.FindBlob("c_enrichment"));
                    colBtm.Add(net.FindBlob("target"));
                }
                else if (nStart == 17) // just before LSTM
                {
                    colBtm.Add(net.FindBlob("selected_hist"));
                    colBtm.Add(net.FindBlob("selected_fut"));
                    colBtm.Add(net.FindBlob("c_seq_hidden"));
                    colBtm.Add(net.FindBlob("c_seq_cell"));
                    colBtm.Add(net.FindBlob("c_enrichment"));
                    colBtm.Add(net.FindBlob("target"));
                }
                else if (nStart == 23) // at GAN just after LSTM
                {
                    colBtm.Add(net.FindBlob("lstm_output"));
                    colBtm.Add(net.FindBlob("lstm_input"));
                    colBtm.Add(net.FindBlob("c_enrichment"));
                    colBtm.Add(net.FindBlob("target"));
                }
                else if (nStart == 24) // at split before static enrichment
                {
                    colBtm.Add(net.FindBlob("gated_lstm_output"));
                    colBtm.Add(net.FindBlob("c_enrichment"));
                    colBtm.Add(net.FindBlob("target"));
                }
                else if (nStart == 28) // at split before multi-head
                {
                    colBtm.Add(net.FindBlob("enriched_sequence"));
                    colBtm.Add(net.FindBlob("glstmout_b"));
                    colBtm.Add(net.FindBlob("target"));
                }
                else if (nStart == 30) // at gan, post multi-head
                {
                    colBtm.Add(net.FindBlob("post_attention"));
                    colBtm.Add(net.FindBlob("enr_seq_b"));
                    colBtm.Add(net.FindBlob("glstmout_b"));
                    colBtm.Add(net.FindBlob("target"));
                }
                else if (nStart == 32) // post gan and multi-head
                {
                    colBtm.Add(net.FindBlob("gated_post_attention"));
                    colBtm.Add(net.FindBlob("glstmout_b"));
                    colBtm.Add(net.FindBlob("target"));
                }
                else if (nStart == 33) // pos wise ff gating
                {
                    colBtm.Add(net.FindBlob("post_poswise_ff_grn"));
                    colBtm.Add(net.FindBlob("glstmout_b"));
                    colBtm.Add(net.FindBlob("target"));
                }
                else if (nStart == 34) // at output IP
                {
                    colBtm.Add(net.FindBlob("gated_poswise_ff"));
                    colBtm.Add(net.FindBlob("target"));
                }

                if (colBtm.Count > 0)
                    load_input_values(rgIn, colBtm, false);
            }

            net.ForwardFromTo(nStart, nEnd);

            BlobCollection<float> colTop = new BlobCollection<float>();
            colTop.Add(net.top_vecs[nEnd]);
            colTop.Add(net.FindBlob("predicted_quantiles"));
            colTop.Add(net.FindBlob("target"));

            return get_return_values(colTop, false);
        }

        public float[] model_bwd(float[] rgIn, int nStart=0, int nEnd = -1)
        {
            Net<float> net = m_mycaffe.GetInternalNet(Phase.TRAIN);

            if (nEnd == -1)
                nEnd = net.layers.Count - 1;

            if (rgIn != null)
                load_input_values(rgIn, net.top_vecs[nEnd], true);

            net.Backward(nEnd, nStart);

            BlobCollection<float> colBtm = new BlobCollection<float>();
            //colBtm.Add(net.FindBlob("x_numeric_static"));
            //colBtm.Add(net.FindBlob("x_categorical_static"));
            //colBtm.Add(net.FindBlob("x_numeric_hist"));
            //colBtm.Add(net.FindBlob("x_categorical_hist"));
            //colBtm.Add(net.FindBlob("x_numeric_future"));
            //colBtm.Add(net.FindBlob("x_categorical_future"));

            if (nStart == 4)
            {
                colBtm.Add(net.FindBlob("static_rep"));
                colBtm.Add(net.FindBlob("hist_ts_rep"));
                colBtm.Add(net.FindBlob("future_ts_rep"));
                colBtm.Add(net.FindBlob("target"));
            }
            else if (nStart == 10) // split before hist vsn
            {
                colBtm.Add(net.FindBlob("hist_ts_rep"));
                colBtm.Add(net.FindBlob("future_ts_rep"));
                colBtm.Add(net.FindBlob("c_selection"));
                colBtm.Add(net.FindBlob("c_seq_hidden"));
                colBtm.Add(net.FindBlob("c_seq_cell"));
                colBtm.Add(net.FindBlob("c_enrichment"));
                colBtm.Add(net.FindBlob("target"));
            }
            else if (nStart == 11) // hist vsn
            {
                colBtm.Add(net.FindBlob("hist_ts_rep"));
                colBtm.Add(net.FindBlob("c_selection_h"));
                colBtm.Add(net.FindBlob("future_ts_rep"));
                colBtm.Add(net.FindBlob("c_selection_f"));
                colBtm.Add(net.FindBlob("c_seq_hidden"));
                colBtm.Add(net.FindBlob("c_seq_cell"));
                colBtm.Add(net.FindBlob("c_enrichment"));
                colBtm.Add(net.FindBlob("target"));
            }
            else if (nStart == 14) // future vsn
            {
                colBtm.Add(net.FindBlob("selected_hist"));
                colBtm.Add(net.FindBlob("future_ts_rep"));
                colBtm.Add(net.FindBlob("c_selection_f"));
                colBtm.Add(net.FindBlob("c_seq_hidden"));
                colBtm.Add(net.FindBlob("c_seq_cell"));
                colBtm.Add(net.FindBlob("c_enrichment"));
                colBtm.Add(net.FindBlob("target"));
            }
            else if (nStart == 17) // just before LSTM
            {
                colBtm.Add(net.FindBlob("selected_hist"));
                colBtm.Add(net.FindBlob("selected_fut"));
                colBtm.Add(net.FindBlob("c_seq_hidden"));
                colBtm.Add(net.FindBlob("c_seq_cell"));
                colBtm.Add(net.FindBlob("c_enrichment"));
                colBtm.Add(net.FindBlob("target"));
            }
            else if (nStart == 23) // at GAN just after LSTM
            {
                colBtm.Add(net.FindBlob("lstm_output"));
                colBtm.Add(net.FindBlob("lstm_input"));
                colBtm.Add(net.FindBlob("c_enrichment"));
                colBtm.Add(net.FindBlob("target"));
            }
            else if (nStart == 24) // at split before static enrichment
            {
                colBtm.Add(net.FindBlob("gated_lstm_output"));
                colBtm.Add(net.FindBlob("c_enrichment"));
                colBtm.Add(net.FindBlob("target"));
            }
            else if (nStart == 28) // at split before multi-head
            {
                colBtm.Add(net.FindBlob("enriched_sequence"));
                colBtm.Add(net.FindBlob("glstmout_b"));
                colBtm.Add(net.FindBlob("target"));
            }
            else if (nStart == 30) // at gan, post multi-head
            {
                colBtm.Add(net.FindBlob("post_attention"));
                colBtm.Add(net.FindBlob("enr_seq_b"));
                colBtm.Add(net.FindBlob("glstmout_b"));
                colBtm.Add(net.FindBlob("target"));
            }
            else if (nStart == 32) // post gan and multi-head
            {
                colBtm.Add(net.FindBlob("gated_post_attention"));
                colBtm.Add(net.FindBlob("glstmout_b"));
                colBtm.Add(net.FindBlob("target"));
            }
            else if (nStart == 33) // pos wise ff gating
            {
                colBtm.Add(net.FindBlob("post_poswise_ff_grn"));
                colBtm.Add(net.FindBlob("glstmout_b"));
                colBtm.Add(net.FindBlob("target"));
            }
            else if (nStart == 34) // at output IP
            {
                colBtm.Add(net.FindBlob("gated_poswise_ff"));
                colBtm.Add(net.FindBlob("target"));
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
            Layer<float> layer = m_rgLayers[strTag];
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
            Layer<float> layer = m_rgLayers[strTag];
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

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.LSTM, "lstm");
                p.recurrent_param.num_output = (uint)nStateSize;
                p.recurrent_param.num_layers = (uint)nNumLayers;
                p.recurrent_param.dropout_ratio = 0;
                p.recurrent_param.expose_hidden_input = true;
                p.recurrent_param.expose_hidden_output = true;
                p.recurrent_param.batch_first = true;
                p.recurrent_param.auto_repeat_hidden_states_across_layers = false;
                p.recurrent_param.use_cudnn_rnn8_if_supported = true;
                p.recurrent_param.engine = EngineParameter.Engine.CUDNN;
                Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);
                layer1.SetPhase(Phase.TRAIN);

                layer1.Setup(m_colBtm, m_colTop);
                m_rgLayers.Add(strTag, layer1);
            }

            Layer<float> layer = m_rgLayers[strTag];
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

            Layer<float> layer = m_rgLayers[strTag];
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

        public void lstm_update_wts(string strTag, float fLr, float fDecay, float fBeta1, float fBeta2, int nT, float fEps)
        {
            Layer<float> layer = m_rgLayers[strTag];
            CustomOptimizer opt = new CustomOptimizer(0);

            if (!m_rgBlobM.ContainsKey(strTag))
            {
                m_rgBlobM.Add(strTag, new Blob<float>(m_mycaffe.Cuda, m_mycaffe.Log));
                m_rgBlobM[strTag].ReshapeLike(layer.blobs[0]);
            }

            if (!m_rgBlobV.ContainsKey(strTag))
            {
                m_rgBlobV.Add(strTag, new Blob<float>(m_mycaffe.Cuda, m_mycaffe.Log));
                m_rgBlobV[strTag].ReshapeLike(layer.blobs[0]);
            }

            Blob<float> blobM = m_rgBlobM[strTag];
            Blob<float> blobV = m_rgBlobV[strTag];

            opt.update_step(fLr, fDecay, fBeta1, fBeta2, nT, fEps);

            float[] rgM = blobM.mutable_cpu_data;
            float[] rgV = blobV.mutable_cpu_data;
            float[] rgW = layer.blobs[0].mutable_cpu_data;
            float[] rgG = layer.blobs[0].mutable_cpu_diff;

            float[] rgW2 = opt.step(rgW, rgG, rgM, rgV);

            layer.blobs[0].mutable_cpu_data = rgW2;
            blobM.mutable_cpu_data = rgM;
            blobV.mutable_cpu_data = rgV;
            layer.blobs[0].SetDiff(0);
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

        public float[] innerproduct_fwd(string strTag, bool bBias, int nNumOut, int nAxis, int nN, int nC, int nH, int nW, float[] rg)
        {
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
                p.inner_product_param.axis = nAxis;
                p.inner_product_param.num_output = (uint)nNumOut;
                p.inner_product_param.bias_term = bBias;
                Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                layer1.Setup(m_colBtm, m_colTop);
                m_rgLayers.Add(strTag, layer1);
            }

            Layer<float> layer = m_rgLayers[strTag];
            layer.Forward(m_colBtm, m_colTop);

            return m_blobTop.mutable_cpu_data;
        }

        public float[] innerproduct_bwd(string strTag, int nN, int nC, int nH, int nW, float[] rgY, float[] rgYGrad)
        {
            m_blobTop.Reshape(nN, nC, nH, nW);
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobTop.mutable_cpu_data = rgY;
            m_blobTop.mutable_cpu_diff = rgYGrad;

            Layer<float> layer = m_rgLayers[strTag];
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

        public float[] logsoftmax_fwd(string strTag, int nN, int nC, int nH, int nW, float[] rg)
        {
            m_blobBtm.Reshape(nN, nC, nH, nW);
            m_blobBtm.mutable_cpu_data = rg;

            if (!m_rgLayers.ContainsKey(strTag))
            {
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
                p.softmax_param.engine = EngineParameter.Engine.CUDNN;
                p.softmax_param.algorithm = SOFTMAX_ALGORITHM.LOG;
                p.softmax_param.axis = 2;
                Layer<float> layer1 = Layer<float>.Create(m_mycaffe.Cuda, m_mycaffe.Log, p, null);

                layer1.Setup(m_colBtm, m_colTop);
                m_rgLayers.Add(strTag, layer1);
            }

            Layer<float> layer = m_rgLayers[strTag];
            layer.Forward(m_colBtm, m_colTop);

            return m_blobTop.mutable_cpu_data;
        }

        public float[] logsoftmax_bwd(string strTag, int nN, int nC, int nH, int nW, float[] rgY, float[] rgYGrad)
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

        public float[] channel_sum(int nN, int nC, int nH, float[] rgX)
        {
            List<int> rgShapeB = new List<int>() { nN, nC, nH };
            m_blobBtm.Reshape(rgShapeB);
            List<int> rgShapeT = new List<int>() { nN, nC };
            m_blobTop.Reshape(rgShapeT);
            m_blobBtm.mutable_cpu_data = rgX;
            m_mycaffe.Cuda.channel_sum(nN * nC, nN * nC, nH, 1, m_blobBtm.gpu_data, m_blobTop.mutable_gpu_data);
            return m_blobTop.mutable_cpu_data;
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

        public float[] Step(int nIter, float[] rgEncIn, float[] rgDecIn, float[] rgDecOut, float[] rgEncMask, float[] rgDecMask)
        {
            m_blobEncIn.mutable_cpu_data = rgEncIn;
            m_blobDecIn.mutable_cpu_data = rgDecIn;
            m_blobDecOut.mutable_cpu_data = rgDecOut;
            m_blobEncMask.mutable_cpu_data = rgEncMask;
            m_blobDecMask.mutable_cpu_data = rgDecMask;

            Net<float> net = m_mycaffe.GetInternalNet(Phase.TRAIN);

            BlobCollection<float> colInput = new BlobCollection<float>();
            colInput.Add(m_blobEncIn);
            colInput.Add(m_blobDecIn);
            colInput.Add(m_blobDecOut);
            colInput.Add(m_blobEncMask);
            colInput.Add(m_blobDecMask);

            net.ClearParamDiffs();

            double dfLoss;
            net.ForwardBackward(colInput, out dfLoss);
            
            Solver<float> solver = m_mycaffe.GetInternalSolver();
            solver.UpdateSmoothedLoss(dfLoss, nIter);
            solver.ApplyUpdate(nIter);

            //if (nIter % 500 == 0)
            //    save(nIter);

            Blob<float> blobAccuracy = net.FindBlob("accuracy");
            m_fLastAccuracy = blobAccuracy.GetData(0);

            Blob<float> blobOutput = net.FindBlob("prob");
            return blobOutput.mutable_cpu_data;                        
        }

        private void dispose(ref Blob<float> b)
        {
            if (b != null)
                b.Dispose();
            b = null;
        }

        public void CleanUp()
        {
            dispose(ref m_blobEncIn);
            dispose(ref m_blobDecIn);
            dispose(ref m_blobDecOut);
            dispose(ref m_blobEncMask);
            dispose(ref m_blobDecMask);
            dispose(ref m_blobLoss);
            dispose(ref m_blobBtm);
            dispose(ref m_blobBtm1);
            dispose(ref m_blobBtm2);
            dispose(ref m_blobBtm3);
            dispose(ref m_blobTop);
            dispose(ref m_blobTop1);
            dispose(ref m_blobTop2);

            foreach (KeyValuePair<string, Layer<float>> kv in m_rgLayers)
            {
                kv.Value.Dispose();
            }

            if (m_mycaffe != null)
                m_mycaffe.Dispose();            
        }
    }
}
