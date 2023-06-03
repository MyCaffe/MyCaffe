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
using System.IO;

/// <summary>
/// Testing the SequenceProcessing.
/// </remarks> 
namespace MyCaffe.test
{
    [TestClass]
    public class TestTFT_SequenceProcessing
    {
        [TestMethod]
        public void TestForward()
        {
            SequenceProcessingTest test = new SequenceProcessingTest();

            try
            {
                foreach (ISequenceProcessingTest t in test.Tests)
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
            SequenceProcessingTest test = new SequenceProcessingTest();

            try
            {
                foreach (ISequenceProcessingTest t in test.Tests)
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

    interface ISequenceProcessingTest : ITest
    {
        void TestForward();
        void TestBackward();
    }

    class SequenceProcessingTest : TestBase
    {
        public SequenceProcessingTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("TFT SequenceProcessing Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new SequenceProcessingTest<double>(strName, nDeviceID, engine);
            else
                return new SequenceProcessingTest<float>(strName, nDeviceID, engine);
        }
    }

    class SequenceProcessingTest<T> : TestEx<T>, ISequenceProcessingTest
    {
        Blob<T> m_blobBottomLabels;
        BlobCollection<T> m_colData = new BlobCollection<T>();
        BlobCollection<T> m_colLabels = new BlobCollection<T>();

        public SequenceProcessingTest(string strName, int nDeviceID, EngineParameter.Engine engine)
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
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\test\\" + strSubPath + "\\iter_0\\";
            //return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\test\\" + strSubPath + "\\iter_0\\";
        }

        private string getTestWtsPath(string strSubPath)
        {
            return Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\tft\\data\\favorita\\weights\\" + strSubPath + "\\";
            //return "c:\\temp\\projects\\TFT\\tft-torch-sample\\tft-torch-sample\\data\\favorita\\weights\\hist_ts_transform\\";
        }

        private void verifyFileDownload(string strSubPath, string strFile)
        {
            string strPath = getTestDataPath(strSubPath);
            if (!File.Exists(strPath + strFile))
                throw new Exception("ERROR: You need to download the TFT test data by running the MyCaffe Test Application and selecting the 'Download Test Data | TFT' menu.");
        }

        private string buildModel(int nNumSamples, int nNumHist, int nNumFut, float fDropout, int nLstmLayers, int nStateSize)
        {
            NetParameter p = new NetParameter();
            p.name = "tft_net";
            p.force_backward = true;

            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHist, nStateSize }));  // selected_historical
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumHist }));              // selected_historical
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFut, nStateSize }));   // selected_future
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nNumFut }));               // selected_future
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nStateSize }));            // c_seq_hidden
            input.input_param.shape.Add(new BlobShape(new List<int>() { nNumSamples, nStateSize }));            // c_seq_cell
            input.top.Add("selected_hist");
            input.top.Add("selected_hist_clip");
            input.top.Add("selected_fut");
            input.top.Add("selected_fut_clip");
            input.top.Add("c_seq_hidden");
            input.top.Add("c_seq_cell");
            p.layer.Add(input);

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

            return p.ToProto("root").ToString();
        }

        /// <summary>
        /// Test Sequential Processing focused forward pass.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_4_sequential_processing.py
        /// Target Dir: seqproc
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestForward()
        {
            string strPath = getTestDataPath("seqproc");
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            float fDropout = 0;
            int nStateSize = 64;
            int nLstmLayers = 2;
            int nNumSamples = 256;
            int nNumHist = 90;
            int nNumFut = 30;

            verifyFileDownload("seqproc", "tft.selected_historical1.npy");

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(nNumSamples, nNumHist, nNumFut, fDropout, nLstmLayers, nStateSize);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);

                net = new Net<T>(m_cuda, m_log, param, null, null);

                blob1 = net.FindBlob("selected_hist");
                blob1.LoadFromNumpy(strPath + "tft.selected_historical1.npy");
                blob1 = net.FindBlob("selected_hist_clip");
                blob1.SetData(1);
                blob1 = net.FindBlob("selected_fut");
                blob1.LoadFromNumpy(strPath + "tft.selected_future1.npy");
                blob1 = net.FindBlob("selected_fut_clip");
                blob1.SetData(1);
                blob1 = net.FindBlob("c_seq_hidden");
                blob1.LoadFromNumpy(strPath + "tft.c_seq_hidden1.npy");
                blob1 = net.FindBlob("c_seq_cell");
                blob1.LoadFromNumpy(strPath + "tft.c_seq_cell1.npy");

                net.parameters[0].LoadFromNumpy(strPath + "ZZZ.YYY.past_lstm.lstm.wt0.npy");
                net.parameters[1].LoadFromNumpy(strPath + "ZZZ.YYY.future_lstm.lstm.wt0.npy");
                net.parameters[2].LoadFromNumpy(strPath + "tft.post_lstm_gating.gate.module.fc1.weight.npy");
                net.parameters[3].LoadFromNumpy(strPath + "tft.post_lstm_gating.gate.module.fc1.bias.npy");
                net.parameters[4].LoadFromNumpy(strPath + "tft.post_lstm_gating.gate.module.fc2.weight.npy");
                net.parameters[5].LoadFromNumpy(strPath + "tft.post_lstm_gating.gate.module.fc2.bias.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "tft.lstm_input1.npy");
                blob1 = net.FindBlob("lstm_input");
                m_log.CHECK(blobVal.Compare(blob1, blobWork), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.past_lstm_output1.npy");
                blob1 = net.FindBlob("past_lstm_output");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float) ? 1e-08 : 4e-05)), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.hidden0_1.npy");
                blob1 = net.FindBlob("hidden1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float) ? 1e-08 : 8e-05)), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.hidden1_1.npy");
                blob1 = net.FindBlob("cell1");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float) ? 1e-08 : 2e-04)), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.future_lstm_output1.npy");
                blob1 = net.FindBlob("future_lstm_output");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float) ? 1e-08 : 2e-05)), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.lstm_output1.npy");
                blob1 = net.FindBlob("lstm_output");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, false, (typeof(T) == typeof(float) ? 1e-08 : 4e-05)), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "gated_lstm_output.npy");
                m_log.CHECK(blobVal.Compare(colRes[0], blobWork, false, (typeof(T) == typeof(float) ? 8e-07 : 3e-04)), "The blobs are different!");
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
        /// Test Sequential Processing focused backward pass.
        /// </summary>
        /// <remarks>
        /// To generate test data, run the following python code:
        /// 
        /// Code: test_4_sequential_processing.py
        /// Target Dir: seqproc
        /// Base Data Dir: iter_0.base_set
        /// </remarks>
        public void TestBackward()
        {
            string strPath = getTestDataPath("seqproc");
            Blob<T> blobVal = null;
            Blob<T> blobWork = null;
            Blob<T> blob1 = null;

            Net<T> net = null;
            float fDropout = 0;
            int nStateSize = 64;
            int nLstmLayers = 2;
            int nNumSamples = 256;
            int nNumHist = 90;
            int nNumFut = 30;

            verifyFileDownload("seqproc", "tft.selected_historical1.npy");

            try
            {
                blobVal = new Blob<T>(m_cuda, m_log);
                blobWork = new Blob<T>(m_cuda, m_log);

                string strModel = buildModel(nNumSamples, nNumHist, nNumFut, fDropout, nLstmLayers, nStateSize);
                RawProto rp = RawProto.Parse(strModel);
                NetParameter param = NetParameter.FromProto(rp);
                param.force_backward = true;

                net = new Net<T>(m_cuda, m_log, param, null, null, Phase.TRAIN);

                blob1 = net.FindBlob("selected_hist");
                blob1.LoadFromNumpy(strPath + "tft.selected_historical1.npy");
                blob1 = net.FindBlob("selected_hist_clip");
                blob1.SetData(1);
                blob1 = net.FindBlob("selected_fut");
                blob1.LoadFromNumpy(strPath + "tft.selected_future1.npy");
                blob1 = net.FindBlob("selected_fut_clip");
                blob1.SetData(1);
                blob1 = net.FindBlob("c_seq_hidden");
                blob1.LoadFromNumpy(strPath + "tft.c_seq_hidden1.npy");
                blob1 = net.FindBlob("c_seq_cell");
                blob1.LoadFromNumpy(strPath + "tft.c_seq_cell1.npy");

                net.parameters[0].LoadFromNumpy(strPath + "ZZZ.YYY.past_lstm.lstm.wt0.npy");
                net.parameters[1].LoadFromNumpy(strPath + "ZZZ.YYY.future_lstm.lstm.wt0.npy");
                net.parameters[2].LoadFromNumpy(strPath + "tft.post_lstm_gating.gate.module.fc1.weight.npy");
                net.parameters[3].LoadFromNumpy(strPath + "tft.post_lstm_gating.gate.module.fc1.bias.npy");
                net.parameters[4].LoadFromNumpy(strPath + "tft.post_lstm_gating.gate.module.fc2.weight.npy");
                net.parameters[5].LoadFromNumpy(strPath + "tft.post_lstm_gating.gate.module.fc2.bias.npy");

                BlobCollection<T> colRes = net.Forward();

                blobVal.LoadFromNumpy(strPath + "gated_lstm_output.npy");
                m_log.CHECK(blobVal.Compare(colRes[0], blobWork, false, (typeof(T) == typeof(float) ? 8e-07 : 3e-04)), "The blobs are different!");

                //*** BACKWARD ***

                blob1 = net.FindBlob("gated_lstm_output");
                blob1.LoadFromNumpy(strPath + "gated_lstm_output.grad.npy", true);

                net.Backward();

                blobVal.LoadFromNumpy(strPath + "tft.selected_historical1.grad.npy", true);
                blob1 = net.FindBlob("selected_hist");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The blobs are different!");

                blobVal.LoadFromNumpy(strPath + "tft.selected_future1.grad.npy", true);
                blob1 = net.FindBlob("selected_fut");
                m_log.CHECK(blobVal.Compare(blob1, blobWork, true), "The blobs are different!");
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
