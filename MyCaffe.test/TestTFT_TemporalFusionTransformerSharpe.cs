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
using MyCaffe.solvers;
using System.Diagnostics;
using System.IO;
using MyCaffe.param.tft;
using System.Xml.XPath;

/// <summary>
/// Testing the TemporalFusionTransformer network.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_TemporalFusionTransformerSharpe
    {
        [TestMethod]
        public void TestForward()
        {
            TemporalFutionTransformerSharpeTest test = new TemporalFutionTransformerSharpeTest();

            try
            {
                foreach (ITemporalFutionTransformerSharpeTest t in test.Tests)
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
            TemporalFutionTransformerSharpeTest test = new TemporalFutionTransformerSharpeTest();

            try
            {
                foreach (ITemporalFutionTransformerSharpeTest t in test.Tests)
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
        public void TestSharpeLoss()
        {
            TemporalFutionTransformerSharpeTest test = new TemporalFutionTransformerSharpeTest();

            try
            {
                foreach (ITemporalFutionTransformerSharpeTest t in test.Tests)
                {
                    t.TestSharpeLoss();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ITemporalFutionTransformerSharpeTest : ITest
    {
        void TestForward();
        void TestBackward();
        void TestSharpeLoss();
    }

    class TemporalFutionTransformerSharpeTest : TestBase
    {
        public TemporalFutionTransformerSharpeTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TemporalFusionTransformerSharpe Network Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TemporalFutionTransformerSharpeTest<double>(strName, nDeviceID, engine);
            else
                return new TemporalFutionTransformerSharpeTest<float>(strName, nDeviceID, engine);
        }
    }

    class TemporalFutionTransformerSharpeTest<T> : TestEx<T>, ITemporalFutionTransformerSharpeTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();
        CalculationArray m_rgLoss = new CalculationArray(50);

        public TemporalFutionTransformerSharpeTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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

        private string getTestBaseDataPath(int nIter=0)
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft-sharpe\\test\\tft.sharpe\\batch256\\batch_0\\";
        }

        private string getTestDataPath(string strTag, int nIter = 0)
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft-sharpe\\test\\" + strTag + "\\";
        }

        private string getTestWtsPath(string strTag, int nIter = 0)
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft-sharpe\\test\\tft.sharpe\\weights\\";
        }

        private void verifyFileDownload(string strPath, string strFile)
        {
            if (!File.Exists(strPath + strFile))
                throw new Exception("ERROR: You need to download the TFT test data by running the MyCaffe Test Application and selecting the 'Download Test Data | TFT' menu.");
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
            output.top.Add("predicted_quantiles1");
            p.layer.Add(output);

            LayerParameter tanh = new LayerParameter(LayerParameter.LayerType.TANH, "tanh");
            tanh.bottom.Add("predicted_quantiles1");
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

        private void load_weights(string strTag1, Net<T> net, string strPath, int nNumStaticNumeric, int nNumStaticCategorical, int nNumHistNumeric, int nNumHistCategorical, int nNumFutureNumeric, int nNumFutureCategorical)
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

        private void compare(double dfTol, List<string> rgstrErrors, int nIdx, Blob<T> blobVal, Blob<T> blob, Blob<T> blobWork, string strFile)
        {
            double dfMin;
            double dfMax;

            blobVal.LoadFromNumpy(strFile);
            bool bCompare = blobVal.CompareEx(blob, blobWork, out dfMin, out dfMax, false, dfTol);

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
                int nIdx = (nNumFutureCategorical == 0 && nNumFutureNumeric == 0) ? 182 : 362;

                //---------------------------------
                //  *Output (idx=360)
                //---------------------------------
                strTag = "output_layer\\";
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "weight.npy");


                //---------------------------------
                //  *Pos wise FF Gate (idx=356)
                //---------------------------------
                strTag = "pos_wise_ff_gating\\";
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.weight.npy");

                //---------------------------------
                //  *Pos wise FF (idx=348)
                //---------------------------------
                strTag = "pos_wise_ff_grn\\";
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc2.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc2.module.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc1.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc1.module.weight.npy");

                //---------------------------------
                //  *Temporal Self-attention (idx=336)
                //---------------------------------
                strTag = "post_attention_gating\\";
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.weight.npy");
                strTag = "multihead_attn\\";
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "out.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "out.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "w_v.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "w_v.weight.npy");
                nIdx--;
/*BUG->*/       compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "w_k.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "w_k.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "w_q.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "w_q.weight.npy");

                //---------------------------------
                //  *Temporal Static Enrichment (idx=327)
                //---------------------------------
                strTag = "static_enrichment_grn\\";
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc2.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc2.module.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "context_projection.module.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc1.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc1.module.weight.npy");

                //---------------------------------
                //  *Locality Enhancement with Seq2Seq processing (idx=321)
                //---------------------------------
                strTag = "post_lstm_gating\\";
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.weight.npy");
                nIdx--;

                strTag = "past_lstm\\";
                if (nNumFutureCategorical > 0 || nNumFutureNumeric > 0)
                {
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "ZZZ.YYY.future_lstm.lstm.wt0.npy");
                    nIdx--;
                }
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "ZZZ.YYY.past_lstm.lstm.wt0.npy");

                //-------------------------------------------
                // *Load varselnet weights - future (idx=246)
                //-------------------------------------------
                if (nNumFutureCategorical > 0 || nNumFutureNumeric > 0)
                {
                    strTag = "future_ts_selection\\";
                    for (int i = (nNumFutureNumeric + nNumFutureCategorical) - 1; i >= 0; i--)
                    {
                        nIdx--;
                        compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                        nIdx--;
                        compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                        nIdx--;
                        compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                        nIdx--;
                        compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                        nIdx--;
                        compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                        nIdx--;
                        compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                        nIdx--;
                        compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                        nIdx--;
                        compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                    }

                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.gate.module.fc2.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.gate.module.fc2.weight.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.gate.module.fc1.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.gate.module.fc1.weight.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.fc2.module.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.fc2.module.weight.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.context_projection.module.weight.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.fc1.module.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.fc1.module.weight.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.skip_layer.module.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.skip_layer.module.weight.npy");
                }

                //-------------------------------------------
                // *Load varselnet weights - historical (idx=147)
                //-------------------------------------------
                strTag = "hist_ts_selection\\";
                for (int i = (nNumHistNumeric + nNumHistCategorical) - 1; i>=0; i--)
                {
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                }

                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.gate.module.fc2.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.gate.module.fc2.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.gate.module.fc1.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.gate.module.fc1.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.fc2.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.fc2.module.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.context_projection.module.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.fc1.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.fc1.module.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.skip_layer.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.skip_layer.module.weight.npy");

                //---------------------------------
                //  *Static covariate encoders (idx=115)
                //---------------------------------
                strTag = "static_encoder_sequential_state_init\\";
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc2.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc2.module.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc1.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc1.module.weight.npy");

                strTag = "static_encoder_sequential_cell_init\\";
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc2.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc2.module.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc1.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc1.module.weight.npy");

                // Sample gradients for each are 0, which appear to be incorrect.
                strTag = "static_encoder_enrichment\\";
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc2.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc2.module.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc1.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc1.module.weight.npy");

                // Sample gradients for each are 0, which appear to be incorrect.
                strTag = "static_encoder_selection\\";
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc2.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "gate.module.fc1.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc2.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc2.module.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc1.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "fc1.module.weight.npy");


                //-------------------------------------------
                // *Load varselnet weights - static (idx=33)
                //-------------------------------------------
                // Sample gradients for each are 0, which appear to be incorrect.
                strTag = "static_selection\\";
                for (int i = (nNumStaticNumeric + nNumStaticCategorical) - 1; i>=0; i--)
                {
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc2.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc2.weight.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc1.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".gate.module.fc1.weight.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".fc2.module.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".fc2.module.weight.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".fc1.module.bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "single_variable_grns." + i.ToString() + ".fc1.module.weight.npy");
                }

                // Sample gradients for each are 0, which appear to be incorrect.
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.gate.module.fc2.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.gate.module.fc2.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.gate.module.fc1.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.gate.module.fc1.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.fc2.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.fc2.module.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.fc1.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.fc1.module.weight.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.skip_layer.module.bias.npy");
                nIdx--;
                compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "flattened_grn.skip_layer.module.weight.npy");


                //-------------------------------------------
                // *Load input channel embedding weights.
                //-------------------------------------------

                strTag = "future_ts_transform\\";
                for (int i = nNumFutureCategorical-1; i>=0; i--)
                {
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "categorical_transform.module.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                }

                strTag = "future_ts_transform\\";
                for (int i = nNumFutureNumeric-1; i>=0; i--)
                {
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "numeric_transform.module.numeric_projection_layers." + i.ToString() + ".bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "numeric_transform.module.numeric_projection_layers." + i.ToString() + ".weight.npy");
                }

                strTag = "hist_ts_transform\\";
                for (int i = nNumHistCategorical-1; i>=0; i--)
                {
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "categorical_transform.module.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                }

                strTag = "hist_ts_transform\\";
                for (int i = nNumHistNumeric-1; i>=0; i--)
                {
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "numeric_transform.module.numeric_projection_layers." + i.ToString() + ".bias.npy");
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "numeric_transform.module.numeric_projection_layers." + i.ToString() + ".weight.npy");
                }

                strTag = "static_transform\\";
                for (int i = nNumStaticCategorical-1; i>=0; i--)
                {
                    nIdx--;
                    compare(dfTol, rgErrors, nIdx, blobVal, net.parameters[nIdx], blobWork, strPath + strTag + "categorical_transform.categorical_embedding_layers." + i.ToString() + ".weight.npy");
                }
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);
            }
        }

        private void load_blobs(string strTag, Net<T> net, string strPath)
        {
            // Transform all input channels
            net.FindBlob("future_ts_rep").LoadFromNumpy(strPath + strTag + ".future_ts_rep.npy");
            net.FindBlob("hist_ts_rep").LoadFromNumpy(strPath + strTag + ".historical_ts_rep.npy");
            net.FindBlob("static_rep").LoadFromNumpy(strPath + strTag + ".static_rep.npy");


            // Select static
            net.FindBlob("selected_static").LoadFromNumpy(strPath + strTag + ".selected_static.npy");


            // Static Covariate Encoding
            net.FindBlob("c_enrichment").LoadFromNumpy(strPath + strTag + ".c_enrichment.npy");
            net.FindBlob("c_selection").LoadFromNumpy(strPath + strTag + ".c_selection.npy");
            net.FindBlob("c_seq_cell").LoadFromNumpy(strPath + strTag + ".c_seq_cell.npy");
            net.FindBlob("c_seq_hidden").LoadFromNumpy(strPath + strTag + ".c_seq_hidden.npy");


            // Historical varaible selection
            net.FindBlob("selected_hist").LoadFromNumpy(strPath + strTag + ".selected_historical.npy");

            // Future variable selection
            net.FindBlob("selected_fut").LoadFromNumpy(strPath + strTag + ".selected_future.npy");

            // Locality enhancement - seq procesing
            net.FindBlob("gated_lstm_output").LoadFromNumpy(strPath + strTag + ".gated_lstm_output.npy");

            // Static enrichment
            net.FindBlob("enriched_sequence").LoadFromNumpy(strPath + strTag + ".enriched_sequence.npy");

            // Self attention
            net.FindBlob("gated_post_attention").LoadFromNumpy(strPath + strTag + ".gated_post_attention.npy");
            net.FindBlob("attention_scores").LoadFromNumpy(strPath + strTag + ".attention_scores.npy");

            // Position-wise feed-forward
            net.FindBlob("post_poswise_ff_grn").LoadFromNumpy(strPath + strTag + ".post_poswise_ff_grn.npy");
            net.FindBlob("gated_poswise_ff").LoadFromNumpy(strPath + strTag + ".gated_poswise_ff.npy");

            // Output
            net.FindBlob("predicted_quantiles").LoadFromNumpy(strPath + strTag + ".predicted_quantiles.npy");
        }

        /// <summary>
        /// Test the forward pass for full model
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run tft-sharpe project
        /// </remarks>
        public void TestForward()
        {
            string strSrc = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\sharpe\\";
            string strPathBase = getTestBaseDataPath();
            string strPath = getTestDataPath("all");
            string strPathWt = getTestWtsPath("full");
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nNumHeads = 4;
            float fDropout = 0;
            int nLstmLayers = 2;
            int nNumOutputs = 1;
            int nStateSize = 10;
            int nNumHistSteps = 63;
            int nNumFutureSteps = 0;
            int nNumStaticNumeric = 0;
            int nNumStaticCategorical = 1;
            List<int> rgStaticCardinalities = new List<int>() { 512 };
            int nNumHistNumeric = 8;
            int nNumHistCategorical = 0;
            List<int> rgHistCardinalities = new List<int>();
            int nNumFutureNumeric = 0;
            int nNumFutureCategorical = 0;
            List<int> rgFutureCardinalities = new List<int>();
            string strTag = "tft.sharpe";

            verifyFileDownload(strPathBase, "historical_ts_numeric.npy");

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(strSrc, true, false, nNumSamples, nNumHeads, fDropout, nLstmLayers, nNumOutputs, nStateSize, nNumHistSteps, nNumFutureSteps, nNumStaticNumeric, nNumStaticCategorical, rgStaticCardinalities, nNumHistNumeric, nNumHistCategorical, rgHistCardinalities, nNumFutureNumeric, nNumFutureCategorical, rgFutureCardinalities);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, null, null, Phase.TRAIN);

                load_weights(strTag, net, strPathWt, nNumStaticNumeric, nNumStaticCategorical, nNumHistNumeric, nNumHistCategorical, nNumFutureNumeric, nNumFutureCategorical);

                // inputs
                //blob1 = net.FindBlob("x_numeric_static");
                //blob1.LoadFromNumpy(strPathBase + "tft.static_feats_numeric.npy");
                blob1 = net.FindBlob("x_categorical_static");
                blob1.LoadFromNumpy(strPathBase + "static_feats_categorical.npy");
                blob1 = net.FindBlob("x_numeric_hist");
                blob1.LoadFromNumpy(strPathBase + "historical_ts_numeric.npy");
                //blob1 = net.FindBlob("x_categorical_hist");
                //blob1.LoadFromNumpy(strPathBase + "0_historical_ts_categorical.npy");
                //blob1 = net.FindBlob("x_numeric_future");
                //blob1.LoadFromNumpy(strPathBase + "0_future_ts_numeric.npy");
                //blob1 = net.FindBlob("x_categorical_future");
                //blob1.LoadFromNumpy(strPathBase + "0_future_ts_categorical.npy");
                blob1 = net.FindBlob("target");
                blob1.LoadFromNumpy(strPathBase + "target.npy");

                BlobCollection<T> colRes = net.Forward();

                // Transform all input channels
                //blobVal.LoadFromNumpy(strPath + strTag + ".future_ts_rep.npy");
                //blob1 = net.FindBlob("future_ts_rep");
                //m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-08), "The blobs are different!");

                strTag = "tft.all";
                blobVal.LoadFromNumpy(strPath + strTag + ".historical_ts_rep.npy");
                blob1 = net.FindBlob("hist_ts_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float) ? 1e-08 : 5e-04)), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".static_rep.npy");
                blob1 = net.FindBlob("static_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");


                // Select static
                blobVal.LoadFromNumpy(strPath + strTag + ".selected_static.npy");
                blob1 = net.FindBlob("selected_static");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 3e-07), "The blobs are different!");


                // Static Covariate Encoding
                blobVal.LoadFromNumpy(strPath + strTag + ".c_enrichment.XX.npy");
                blob1 = net.FindBlob("c_enrichment");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 3e-07), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".c_selection.XX.npy");
                blob1 = net.FindBlob("c_selection");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 4e-07), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".c_seq_cell.XX.npy");
                blob1 = net.FindBlob("c_seq_cell");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 3e-07), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".c_seq_hidden.XX.npy");
                blob1 = net.FindBlob("c_seq_hidden");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 3e-07), "The blobs are different!");


                // Historical varaible selection
                blobVal.LoadFromNumpy(strPath + strTag + ".selected_historical.npy");
                blob1 = net.FindBlob("selected_hist");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 1e-06), "The blobs are different!");

                // Future variable selection
                //blobVal.LoadFromNumpy(strPath + strTag + ".selected_future.npy");
                //blob1 = net.FindBlob("selected_fut");
                //m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 1e-08), "The blobs are different!");

                // Locality enhancement - seq procesing
                blobVal.LoadFromNumpy(strPath + strTag + ".gated_lstm_output.npy");
                blob1 = net.FindBlob("gated_lstm_output");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");

                // Static enrichment
                blobVal.LoadFromNumpy(strPath + strTag + ".enriched_sequence.npy");
                blob1 = net.FindBlob("enriched_sequence");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");

                // Self attention
                blobVal.LoadFromNumpy(strPath + strTag + ".ada.post_attention.npy");
                blob1 = net.FindBlob("post_attention");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 1e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".gated_post_attention.npy");
                blob1 = net.FindBlob("gated_post_attention");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".attention_scores.npy");
                blob1 = net.FindBlob("attention_scores");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 1e-07), "The blobs are different!");

                // Position-wise feed-forward
                blobVal.LoadFromNumpy(strPath + strTag + ".post_poswise_ff_grn.npy");
                blob1 = net.FindBlob("post_poswise_ff_grn");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 1e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".gated_poswise_ff.npy");
                blob1 = net.FindBlob("gated_poswise_ff");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");

                // Output
                blobVal.LoadFromNumpy(strPath + strTag + ".predicted_quantiles1.npy");
                blob1 = net.FindBlob("predicted_quantiles1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".predicted_quantiles.npy");
                blob1 = net.FindBlob("predicted_quantiles");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }

        /// <summary>
        /// Test the backward pass for for full model
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run tft-sharpe project.
        /// </remarks>
        public void TestBackward()
        {
            string strSrc = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\sharpe\\";
            string strPathBase = getTestBaseDataPath();
            string strPath = getTestDataPath("all");
            string strPathWt = getTestWtsPath("full");
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nNumHeads = 4;
            float fDropout = 0;
            int nLstmLayers = 2;
            int nNumOutputs = 1;
            int nStateSize = 10;
            int nNumHistSteps = 63;
            int nNumFutureSteps = 0;
            int nNumStaticNumeric = 0;
            int nNumStaticCategorical = 1;
            List<int> rgStaticCardinalities = new List<int>() { 512 };
            int nNumHistNumeric = 8;
            int nNumHistCategorical = 0;
            List<int> rgHistCardinalities = new List<int>();
            int nNumFutureNumeric = 0;
            int nNumFutureCategorical = 0;
            List<int> rgFutureCardinalities = new List<int>();
            string strTag = "tft.sharpe";

            verifyFileDownload(strPathBase, "historical_ts_numeric.npy");

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(strSrc, true, false, nNumSamples, nNumHeads, fDropout, nLstmLayers, nNumOutputs, nStateSize, nNumHistSteps, nNumFutureSteps, nNumStaticNumeric, nNumStaticCategorical, rgStaticCardinalities, nNumHistNumeric, nNumHistCategorical, rgHistCardinalities, nNumFutureNumeric, nNumFutureCategorical, rgFutureCardinalities);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                m_cuda.debug();

                net = new Net<T>(m_cuda, m_log, param, null, null, Phase.TRAIN);

                load_weights(strTag, net, strPathWt, nNumStaticNumeric, nNumStaticCategorical, nNumHistNumeric, nNumHistCategorical, nNumFutureNumeric, nNumFutureCategorical);

                // inputs
                //blob1 = net.FindBlob("x_numeric_static");
                //blob1.LoadFromNumpy(strPathBase + "tft.static_feats_numeric.npy");
                blob1 = net.FindBlob("x_categorical_static");
                blob1.LoadFromNumpy(strPathBase + "static_feats_categorical.npy");
                blob1 = net.FindBlob("x_numeric_hist");
                blob1.LoadFromNumpy(strPathBase + "historical_ts_numeric.npy");
                //blob1 = net.FindBlob("x_categorical_hist");
                //blob1.LoadFromNumpy(strPathBase + "0_historical_ts_categorical.npy");
                //blob1 = net.FindBlob("x_numeric_future");
                //blob1.LoadFromNumpy(strPathBase + "0_future_ts_numeric.npy");
                //blob1 = net.FindBlob("x_categorical_future");
                //blob1.LoadFromNumpy(strPathBase + "0_future_ts_categorical.npy");
                blob1 = net.FindBlob("target");
                blob1.LoadFromNumpy(strPathBase + "target.npy");

                m_cuda.debug();

                BlobCollection<T> colRes = net.Forward();

                // Transform all input channels
                //blobVal.LoadFromNumpy(strPath + strTag + ".future_ts_rep.npy");
                //blob1 = net.FindBlob("future_ts_rep");
                //m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-08), "The blobs are different!");

                strTag = "tft.all";
                blobVal.LoadFromNumpy(strPath + strTag + ".historical_ts_rep.npy");
                blob1 = net.FindBlob("hist_ts_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float) ? 1e-08 : 5e-04)), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".static_rep.npy");
                blob1 = net.FindBlob("static_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");


                // Select static
                blobVal.LoadFromNumpy(strPath + strTag + ".selected_static.npy");
                blob1 = net.FindBlob("selected_static");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 3e-07), "The blobs are different!");


                // Static Covariate Encoding
                blobVal.LoadFromNumpy(strPath + strTag + ".c_enrichment.XX.npy");
                blob1 = net.FindBlob("c_enrichment");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 3e-07), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".c_selection.XX.npy");
                blob1 = net.FindBlob("c_selection");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 4e-07), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".c_seq_cell.XX.npy");
                blob1 = net.FindBlob("c_seq_cell");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 3e-07), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".c_seq_hidden.XX.npy");
                blob1 = net.FindBlob("c_seq_hidden");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 3e-07), "The blobs are different!");


                // Historical varaible selection
                blobVal.LoadFromNumpy(strPath + strTag + ".selected_historical.npy");
                blob1 = net.FindBlob("selected_hist");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 1e-06), "The blobs are different!");

                // Future variable selection
                //blobVal.LoadFromNumpy(strPath + strTag + ".selected_future.npy");
                //blob1 = net.FindBlob("selected_fut");
                //m_log.CHECK(blobVal.Compare(blob1, blobWork, false, 1e-08), "The blobs are different!");

                // Locality enhancement - seq procesing
                blobVal.LoadFromNumpy(strPath + strTag + ".gated_lstm_output.npy");
                blob1 = net.FindBlob("gated_lstm_output");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");

                // Static enrichment
                blobVal.LoadFromNumpy(strPath + strTag + ".enriched_sequence.npy");
                blob1 = net.FindBlob("enriched_sequence");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");

                // Self attention
                blobVal.LoadFromNumpy(strPath + strTag + ".ada.post_attention.npy");
                blob1 = net.FindBlob("post_attention");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 1e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".gated_post_attention.npy");
                blob1 = net.FindBlob("gated_post_attention");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".attention_scores.npy");
                blob1 = net.FindBlob("attention_scores");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 1e-07), "The blobs are different!");

                // Position-wise feed-forward
                blobVal.LoadFromNumpy(strPath + strTag + ".post_poswise_ff_grn.npy");
                blob1 = net.FindBlob("post_poswise_ff_grn");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 1e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".gated_poswise_ff.npy");
                blob1 = net.FindBlob("gated_poswise_ff");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");

                // Output
                blobVal.LoadFromNumpy(strPath + strTag + ".predicted_quantiles1.npy");
                blob1 = net.FindBlob("predicted_quantiles1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".predicted_quantiles.npy");
                blob1 = net.FindBlob("predicted_quantiles");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float)) ? 1e-08 : 2e-06), "The blobs are different!");

                ///=========================================
                /// Backward Tests
                ///=========================================

                m_cuda.debug();

                net.Backward(27, 27);

                blob1 = net.FindBlob("predicted_quantiles");
                blobVal.LoadFromNumpy(strPath + strTag + ".predicted_quantiles.grad.npy", true);
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, 1e-06), "The gradients are different!");
                blob1.CopyFrom(blobVal, true);

                net.Backward(26);

                m_cuda.debug();

                blob1 = net.FindBlob("predicted_quantiles");
                blobVal.LoadFromNumpy(strPath + strTag + ".predicted_quantiles.grad.npy", true);
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The gradients are different!");

                // Predicted Quantiles
                blobVal.LoadFromNumpy(strPath + strTag + ".predicted_quantiles1.grad.npy", true);
                blob1 = net.FindBlob("predicted_quantiles1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The gradients are different!");

                // Position-wise feed-forward
                blobVal.LoadFromNumpy(strPath + strTag + ".gated_poswise_ff.grad.npy", true);
                blob1 = net.FindBlob("gated_poswise_ff");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The gradients are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".post_poswise_ff_grn.grad.npy", true);
                blob1 = net.FindBlob("post_poswise_ff_grn");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The gradients are different!");

                // Self-attention
                blobVal.LoadFromNumpy(strPath + strTag + ".gated_post_attention.grad.npy", true);
                blob1 = net.FindBlob("gated_post_attention");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The gradients are different!");

                // Static enrichment
                blobVal.LoadFromNumpy(strPath + strTag + ".enriched_sequence.grad.npy", true);
                blob1 = net.FindBlob("enriched_sequence");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The gradients are different!");

                // Locality enhancement - seq processing
                blobVal.LoadFromNumpy(strPath + strTag + ".gated_lstm_output.grad.npy", true);
                blob1 = net.FindBlob("gated_lstm_output");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, (typeof(T) == typeof(float)) ? 1e-08 : 2e-08), "The gradients are different!");

                // Future variable selection
                //blobVal.LoadFromNumpy(strPath + strTag + ".selected_future.grad.npy", true);
                //blob1 = net.FindBlob("selected_fut");
                //m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The gradients are different!");

                // Historical variable selection
                blobVal.LoadFromNumpy(strPath + strTag + ".selected_historical.grad.npy", true);
                blob1 = net.FindBlob("selected_hist");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, (typeof(T) == typeof(float)) ? 1e-08 : 2e-08), "The gradients are different!");

                // Static Covariate Encoding
                blobVal.LoadFromNumpy(strPath + strTag + ".c_enrichment.XX.grad.npy", true);
                blob1 = net.FindBlob("c_enrichment");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true, (typeof(T) == typeof(float)) ? 1e-08 : 2e-08), "The grad are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".historical_varsel_context.grad.npy", true);
                blob1 = net.FindBlob("c_selection1h");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                //blobVal.LoadFromNumpy(strPath + strTag + ".future_varsel_context.grad.npy", true);
                //blob1 = net.FindBlob("c_selection1f");
                //m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                //blobVal.LoadFromNumpy(strPath + strTag + ".future.static_selection_signal.grad.npy", true);
                //blob1 = net.FindBlob("c_selection_f");
                //m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                // Select static
                blobVal.LoadFromNumpy(strPath + strTag + ".c_enrichment.XX.grad.npy", true);
                blob1 = net.FindBlob("c_enrichment");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".c_selection_h.grad.npy", true);
                blob1 = net.FindBlob("c_selection");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".c_seq_hidden.XX.grad.npy", true);
                blob1 = net.FindBlob("c_seq_hidden");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".c_seq_cell.XX.grad.npy", true);
                blob1 = net.FindBlob("c_seq_cell");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".selstat_a.YY.grad.npy", true);
                blob1 = net.FindBlob("selstat_a");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".selstat_b.YY.grad.npy", true);
                blob1 = net.FindBlob("selstat_b");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".selstat_c.YY.grad.npy", true);
                blob1 = net.FindBlob("selstat_c");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".selstat_d.YY.grad.npy", true);
                blob1 = net.FindBlob("selstat_d");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".selected_static.grad.npy", true);
                blob1 = net.FindBlob("selected_static");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                // Transform all input channels
                //blobVal.LoadFromNumpy(strPath + strTag + ".future.temporal_selection_output.grad.npy", true);
                //blob1 = net.FindBlob("selected_fut1");
                //m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The gradients are different!");

                //blobVal.LoadFromNumpy(strPath + strTag + ".future.temporal_flattened_embedding.grad.npy", true);
                //blob1 = net.FindBlob("future_ts_rep1");
                //m_log.CHECK(blobVal.Compare(blob1, blobWork, true, (typeof(T) == typeof(float)) ? 2e-05 : 5e-05), "The grad are different!");

                //blobVal.LoadFromNumpy(strPath + strTag + ".future_ts_rep.grad.npy", true);
                //blob1 = net.FindBlob("future_ts_rep");
                //m_log.CHECK(blobVal.Compare(blob1, blobWork, true, (typeof(T) == typeof(float)) ? 2e-05 : 5e-05), "The grad are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".historical_ts_rep.grad.npy", true);
                blob1 = net.FindBlob("hist_ts_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");

                blobVal.LoadFromNumpy(strPath + strTag + ".static_rep.grad.npy", true);
                blob1 = net.FindBlob("static_rep");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The grad are different!");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();
            }
        }

        /// <summary>
        /// Test the training for for full model
        /// </summary>
        /// <remarks>
        /// To generate test data:
        /// Run tft-sharpe project
        /// WORK IN PROGRESS
        /// </remarks>
        public void TestTraining()
        {
            string strSrc = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft-sharpe\\data\\sharpe\\";
            string strPathBase = getTestBaseDataPath();
            string strPath = getTestDataPath("all");
            string strPathWt = getTestWtsPath("full");
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            Solver<T> solver = null;
            int nNumSamples = 256;
            int nNumHeads = 4;
            float fDropout = 0;
            int nLstmLayers = 2;
            int nNumOutputs = 1;
            int nStateSize = 10;
            int nNumHistSteps = 63;
            int nNumFutureSteps = 0;
            int nNumStaticNumeric = 0;
            int nNumStaticCategorical = 1;
            List<int> rgStaticCardinalities = new List<int>() { 512 };
            int nNumHistNumeric = 8;
            int nNumHistCategorical = 0;
            List<int> rgHistCardinalities = new List<int>();
            int nNumFutureNumeric = 0;
            int nNumFutureCategorical = 0;
            List<int> rgFutureCardinalities = new List<int>();
            string strTag = "tft.sharpe";
            CancelEvent evtCancel = new CancelEvent();

            verifyFileDownload(strPathBase, "historical_ts_numeric.npy");

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(strSrc, true, false, nNumSamples, nNumHeads, fDropout, nLstmLayers, nNumOutputs, nStateSize, nNumHistSteps, nNumFutureSteps, nNumStaticNumeric, nNumStaticCategorical, rgStaticCardinalities, nNumHistNumeric, nNumHistCategorical, rgHistCardinalities, nNumFutureNumeric, nNumFutureCategorical, rgFutureCardinalities);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter netParam = NetParameter.FromProto(rp);

                string strSolver = buildSolver(0.01);
                RawProto rpSolver = RawProto.Parse(strSolver);
                SolverParameter solverParam = SolverParameter.FromProto(rpSolver);

                solverParam.train_net_param = netParam;

                solver = Solver<T>.Create(m_cuda, m_log, solverParam, evtCancel, null, null, null, null);
                net = solver.TrainingNet;

                load_weights(strTag, net, strPathWt, nNumStaticNumeric, nNumStaticCategorical, nNumHistNumeric, nNumHistCategorical, nNumFutureNumeric, nNumFutureCategorical);

                string strBatchBase = getTestDataPath("tft.sharpe.dbg\\batch256\\batch_").TrimEnd('\\');
                string strBatchWtBase = getTestDataPath("tft.sharpe.dbg\\weights\\batch_").TrimEnd('\\');

                for (int i = 0; i < 10; i++)
                {
                    string strBatch = strBatchBase + i.ToString() + "\\";
                    string strBatchWt = strBatchWtBase + i.ToString() + "\\";

                    // inputs
                    blob1 = net.FindBlob("x_categorical_static");
                    blob1.LoadFromNumpy(strBatch + "static_feats_categorical.npy");
                    blob1 = net.FindBlob("x_numeric_hist");
                    blob1.LoadFromNumpy(strBatch + "historical_ts_numeric.npy");
                    blob1 = net.FindBlob("target");
                    blob1.LoadFromNumpy(strBatch + "target.npy");

                    BlobCollection<T> colRes = net.Forward();

                    Blob<T> blobLoss = colRes[1];
                    blobVal.LoadFromNumpy(strBatch + "loss.npy");
                    m_log.CHECK(blobVal.Compare(blobLoss, blobWork, false, 6e-07), "The blobs are different!");

                    net.ClearParamDiffs();
                    net.Backward();

                    blobVal.LoadFromNumpy(strPath + "predicted_pos.grad.npy", true);
                    blob1 = net.FindBlob("predicted_quantiles");
                    m_log.CHECK(blobVal.Compare(blob1, blobWork, true, 3e-07), "The gradients are different!");

                    solver.ApplyUpdate(i);

                    compare_weights("tft.sharpe", net, strBatchWt, nNumStaticNumeric, nNumStaticCategorical, nNumHistNumeric, nNumHistCategorical, nNumFutureNumeric, nNumFutureCategorical);
                }
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);

                if (net != null)
                    net.Dispose();

                if (solver != null)
                    solver.Dispose();
            }
        }

        public void TestSharpeLoss()
        {
            string strSrc = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\sharpe\\";
            string strPathBase = getTestBaseDataPath();
            string strPath = getTestDataPath("all");
            string strPathWt = getTestWtsPath("full");
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blobLoss = null;
            Blob<T> blobPredPos = null;
            Blob<T> blobTarget = null;

            Net<T> net = null;
            int nNumSamples = 256;
            int nNumHeads = 4;
            float fDropout = 0;
            int nLstmLayers = 2;
            int nNumOutputs = 1;
            int nStateSize = 10;
            int nNumHistSteps = 63;
            int nNumFutureSteps = 0;
            int nNumStaticNumeric = 0;
            int nNumStaticCategorical = 1;
            List<int> rgStaticCardinalities = new List<int>() { 512 };
            int nNumHistNumeric = 8;
            int nNumHistCategorical = 0;
            List<int> rgHistCardinalities = new List<int>();
            int nNumFutureNumeric = 0;
            int nNumFutureCategorical = 0;
            List<int> rgFutureCardinalities = new List<int>();
            string strTag = "tft.sharpe";
            CancelEvent evtCancel = new CancelEvent();

            verifyFileDownload(strPathBase, "historical_ts_numeric.npy");

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(strSrc, true, false, nNumSamples, nNumHeads, fDropout, nLstmLayers, nNumOutputs, nStateSize, nNumHistSteps, nNumFutureSteps, nNumStaticNumeric, nNumStaticCategorical, rgStaticCardinalities, nNumHistNumeric, nNumHistCategorical, rgHistCardinalities, nNumFutureNumeric, nNumFutureCategorical, rgFutureCardinalities);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter netParam = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, netParam, null, null, Phase.TRAIN);
                load_weights(strTag, net, strPathWt, nNumStaticNumeric, nNumStaticCategorical, nNumHistNumeric, nNumHistCategorical, nNumFutureNumeric, nNumFutureCategorical);

                blobPredPos = new Blob<T>(m_cuda, m_log);
                blobTarget = new Blob<T>(m_cuda, m_log);
                blobLoss = new Blob<T>(m_cuda, m_log, 1, 1, 1, 1);

                blobPredPos.LoadFromNumpy(strPath + "sharpe.weights.npy");
                blobTarget.LoadFromNumpy(strPath + "sharpe.y_true.npy");

                Layer<T> layer = net.FindLastLayer(LayerParameter.LayerType.SHARPE_LOSS);

                BlobCollection<T> colBtm = new BlobCollection<T>();
                colBtm.Add(blobPredPos);
                colBtm.Add(blobTarget);
                BlobCollection<T> colTop = new BlobCollection<T>();
                colTop.Add(blobLoss);

                layer.Forward(colBtm, colTop);

                blobVal.LoadFromNumpy(strPath + "sharpe.loss.npy");
                m_log.CHECK(blobVal.Compare(blobLoss, blobWork, false, 2e-06), "The blobs are different!");

                blobLoss.SetDiff(1);
                layer.Backward(colTop, new List<bool>() { true , false }, colBtm);

                blobVal.LoadFromNumpy(strPath + "tft.all.predicted_quantiles.grad.npy", true);
                m_log.CHECK(blobVal.Compare(blobPredPos, blobWork, true, 3e-06), "The gradients are different!");
            }
            finally
            {
                dispose(ref blobVal);
                dispose(ref blobWork);
                dispose(ref blobLoss);
                dispose(ref blobPredPos);
                dispose(ref blobTarget);

                if (net != null)
                    net.Dispose();
            }
        }
    }
}
