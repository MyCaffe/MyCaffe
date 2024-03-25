﻿using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyCaffe.param;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.layers;
using System.Diagnostics;
using System.IO;
using MyCaffe.param.gpt;
using MyCaffe.db.image;
using System.Threading;
using MyCaffe.layers.gpt;
using MyCaffe.solvers;
using MyCaffe.basecode.descriptors;
using System.IO.Compression;
using System.Net;

namespace MyCaffe.test
{
    [TestClass]
    public class TestGPT_TransformerBlockLayer
    {
        [TestMethod]
        public void TestTrain()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestTrain();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainLlama()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestTrainLlama();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTrainLargeLM()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestTrainLargeLM();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestInference()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestInference();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForward()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestForward(null);
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
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestBackward(null);
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
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestGradient();
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        //[TestMethod]
        public void TestGptHuggingFaceImport()
        {
            TransformerBlockLayerTest test = new TransformerBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (ITransformerBlockLayerTest t in test.Tests)
                {
                    t.TestTestHuggingFaceImport();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface ITransformerBlockLayerTest : ITest
    {
        void TestTrain();
        void TestTrainLlama();
        void TestInference();
        void TestForward(string strModel);
        void TestBackward(string strModel);
        void TestGradient();
        void TestTestHuggingFaceImport();
        void TestTrainLargeLM();
    }

    class TransformerBlockLayerTest : TestBase
    {
        public TransformerBlockLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Transformer Block Layer Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new TransformerBlockLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new TransformerBlockLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class TransformerBlockLayerTest2<T> : TestEx<T>, ITransformerBlockLayerTest
    {
        SettingsCaffe m_settings = new SettingsCaffe();
        CancelEvent m_evtCancel = new CancelEvent();
        AutoResetEvent m_evtForceSnapshot = new AutoResetEvent(false);
        AutoResetEvent m_evtForceTest = new AutoResetEvent(false);
        WaitHandle[] m_rgevtCancel;
        List<int> m_rgGpu = new List<int>();
        Net<T> m_netRun = null;
        TokenizedDataLayer<T> m_dataLayer = null;
        Blob<T> m_blobY = null;
        Blob<T> m_blobX = null;
        Blob<T> m_blobPos = null;
        float[] m_rgTestInput;
        MyCaffeControl<T> m_ctrl = null;
        Random m_random = new Random(3407);
        Stopwatch m_swUpdateTimer = new Stopwatch();
        double m_dfLastProgress = 0;
        AutoResetEvent m_evtDownloadDone = new AutoResetEvent(false);

        public TransformerBlockLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;

            List<WaitHandle> rgWait = new List<WaitHandle>();
            rgWait.AddRange(m_evtCancel.Handles);

            m_rgevtCancel = rgWait.ToArray();
            m_settings.DbLoadMethod = DB_LOAD_METHOD.LOAD_ALL;
            m_settings.GpuIds = nDeviceID.ToString();
            m_rgGpu.Add(nDeviceID);
        }

        protected override FillerParameter getFillerParam()
        {
            return base.getFillerParam();
        }

        private void dispose1(ref Blob<T> b)
        {
            if (b != null)
            {
                b.Dispose();
                b = null;
            }
        }

        protected override void dispose()
        {
            base.dispose();
        }

        private int getNextIndex(Blob<T> blob, int nVocabCount, int nTopK, Layer<T> softmax)
        {
            List<Tuple<string, int, double>> rg = m_dataLayer.PostProcessLogitsOutput(-1, blob, softmax, 10);
            return rg[0].Item2;
        }

        private void generate(Net<T> net, Blob<T> blobIdx, Blob<T> blobY, int nMaxNewTokens, int nBlockSize, int nVocabSize, int nTopK)
        {
            Blob<T> blobLogits = net.blob_by_name("logits");
            Layer<T> softmax = net.FindLastLayer(LayerParameter.LayerType.SOFTMAX);
            BlobCollection<T> colBottom = new BlobCollection<T>() { blobIdx };
            double dfLoss;
            List<float> rgfIdx = new List<float>();
            List<float> rgfIdxOut = new List<float>();
            int[] rgShape;

            float[] rgIdx = convertF(blobIdx.mutable_cpu_data);
            rgfIdx.AddRange(rgIdx);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (int i = 0; i < nMaxNewTokens; i++)
            {                
                // Forward pass to get the logits.
                net.Forward(colBottom, out dfLoss, true);                                
                float fIdxVal = getNextIndex(blobLogits, nVocabSize, nTopK, softmax);

                rgfIdx.Add(fIdxVal);
                // Clip to block size.
                if (rgfIdx.Count > nBlockSize)
                    rgfIdx.RemoveAt(0);

                rgfIdxOut.Add(fIdxVal);

                rgShape = new int[] { 1, rgfIdx.Count };
                if (blobIdx.channels < rgfIdx.Count)
                    blobIdx.Reshape(rgShape);
                blobIdx.mutable_cpu_data = convert(rgfIdx.ToArray());
                
                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    sw.Restart();
                    double dfPct = (double)i / nMaxNewTokens;
                    m_log.WriteLine("Generating at " + dfPct.ToString("P") + "...");
                }
            }

            rgShape = new int[] { 1, rgfIdxOut.Count };
            blobY.Reshape(rgShape);
            blobY.mutable_cpu_data = convert(rgfIdxOut.ToArray());
        }

        private void verifyBlob(Log log, Blob<T> blob, Tuple<List<int>, float[]> data, bool bDiff = false, double dfErr = 1e-5)
        {
            if (blob.count() != data.Item2.Length)
                m_log.FAIL(blob.Name + ": The blob count does not match the data count!");

            float[] rgData = (bDiff) ? convertF(blob.mutable_cpu_diff) : convertF(blob.mutable_cpu_data);

            for (int i = 0; i < rgData.Length; i++)
            {
                float fActual = rgData[i];
                float fExpected = data.Item2[i];
                float fDiff = fActual - fExpected;

                if (Math.Abs(fDiff) > dfErr)
                    m_log.FAIL(blob.Name + ": The data at index " + i.ToString() + " does not match!");
            }
        }

        // WORK IN PROGRESS
        //
        // TODO: All that remains is to implement the GPT2 tokenizer and load the GPT2 vocabulary file
        // found at:
        // @see [GPT2 Vocabulary File](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json)
        public void TestTestHuggingFaceImport()
        {
            string strModelPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\huggingface\\gpt2\\";
            string strWtPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\models\\huggingface\\gpt2\\gpt2_weights\\";
            string strSrc = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\input.txt";
            TextInputData input = new TextInputData(strSrc);
            int nVocabSize = 50257; // OpenAi's model vocabulary
            int nBlockSize = 1024;
            int nLayerCount = 12;
            int nHeads = 12;
            int nEmbed = 768;
            int nBatchSize = 1;

            string strModel = createGptModel("GPT2", strSrc, nVocabSize, (uint)nLayerCount, (uint)nHeads, (uint)nEmbed, (uint)nBlockSize, nBatchSize);

            m_log.EnableTrace = true;
            m_log.WriteHeader("GPT-Mini - Test Train");

            try
            {
                m_log.WriteLine("Downloading GPT2 model weights...");
                loadWeightFiles(strModelPath, strModel);

                m_ctrl = new MyCaffeControl<T>(m_settings, m_log, m_evtCancel, m_evtForceSnapshot, m_evtForceTest, null, m_rgGpu, m_cuda.Path);
                m_log.WriteLine("Loading GPT2 model...");
                BlobShape shape = new BlobShape(new List<int>() { 1, nBlockSize });
                m_ctrl.LoadToRun(strModel, null, null, shape, null, null, false, false);

                m_blobY = m_ctrl.CreateBlob("results");
                m_blobX = m_ctrl.CreateBlob("data");
                m_blobPos = m_ctrl.CreateBlob("pos");
                m_blobPos.Reshape(1, 128, 1, 1);
                
                m_netRun = m_ctrl.GetInternalNet(Phase.RUN);
                m_log.WriteLine("Loading GPT2 pre-trained weights into the model...");
                loadWeights(m_netRun, strWtPath);

                if (m_dataLayer == null)
                {
                    LayerParameter param_data = new LayerParameter(LayerParameter.LayerType.TOKENIZED_DATA);
                    param_data.tokenized_data_param.source = strSrc;
                    param_data.tokenized_data_param.block_size = (uint)nBlockSize;
                    param_data.tokenized_data_param.batch_size = (uint)nBatchSize;
                    param_data.tokenized_data_param.input_type = TokenizedDataParameter.INPUT_TYPE.TEXT_FILE;
                    m_dataLayer = new TokenizedDataLayer<T>(m_cuda, m_log, param_data, null, null);

                    BlobCollection<T> colBottom = new BlobCollection<T>();
                    BlobCollection<T> colTop = new BlobCollection<T>() { m_blobX, m_blobPos, m_blobX };

                    m_dataLayer.LayerSetUp(colBottom, colTop);
                }

                string strInput = System.IO.File.ReadAllText(strSrc);
                int nIdx = m_random.Next(strInput.Length - (2 * nBlockSize));

                List<int> rgTokens = m_dataLayer.Tokenize(strInput.Substring(nIdx, nBlockSize), false, false);

                m_rgTestInput = new float[nBlockSize];
                for (int i = 0; i < nBlockSize; i++)
                {
                    if (i < rgTokens.Count)
                        m_rgTestInput[i] = rgTokens[i];
                }

                int[] rgShape = new int[] { 1, m_rgTestInput.Length };
                m_blobX.Reshape(rgShape);
                m_blobX.mutable_cpu_data = convert(m_rgTestInput);

                generate(m_netRun, m_blobX, m_blobY, 500, nBlockSize, nVocabSize, 10);

                string strOut = "";
                float[] rgY = convertF(m_blobY.mutable_cpu_data);
                int nCount = m_blobY.count(1);

                for (int i = 0; i < m_blobY.num; i++)
                {
                    string str1 = m_dataLayer.Detokenize(rgY, i * nCount, nCount);
                    strOut += str1 + Environment.NewLine;
                }

                m_log.WriteLine(strOut);
            }
            finally
            {
                dispose1(ref m_blobY);
                dispose1(ref m_blobX);
                dispose1(ref m_blobPos);

                if (m_dataLayer != null)
                {
                    m_dataLayer.Dispose();
                    m_dataLayer = null;
                }

                m_ctrl.Dispose();
                m_ctrl = null;
            }
        }

        private void setData(Blob<T> blob, string strPath, string strName)
        {
            m_log.WriteLine("Loading blob '" + strName + "' from '" + strPath + "'...");
            string strNpyFile = strPath + strName + ".npy";
            string strShapeFile = strPath + strName + ".txt";

            List<int> rgShape = new List<int>();
            string[] rgstr = System.IO.File.ReadAllLines(strShapeFile);
            foreach (string s in rgstr)
            {
                rgShape.Add((int)float.Parse(s));
            }

            if (!blob.CompareShape(rgShape))
                throw new Exception("The shape of the blob '" + blob.Name + "' does not match the shape of the data!");

            blob.LoadFromNumpy(strNpyFile);
        }
        
        private void loadWeights(Net<T> net, string strPath)
        {
            Stopwatch sw = new Stopwatch();
            string strModelPath = strPath;
            int nTfBlockCount = 12;
            
            setData(net.learnable_parameters[0], strModelPath, "gpt_wte_weight");
            setData(net.learnable_parameters[1], strModelPath, "gpt_wpe_weight");
            int nIdx = 2;

            sw.Start();
            
            for (int i = 0; i < nTfBlockCount; i++)
            {
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_attn_weight");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_attn_bias");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_attn_proj_weight");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_attn_proj_bias");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_fc_weight");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_fc_bias");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_proj_weight");
                nIdx++;
                setData(net.learnable_parameters[nIdx], strModelPath, "tfb_" + (i+1).ToString() + "_proj_bias");
                nIdx++;

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    sw.Restart();
                    double dfPct = (double)i / nTfBlockCount;
                    m_log.WriteLine("Loading pre-trained weights: " + dfPct.ToString("P"));
                }
            }

            setData(net.learnable_parameters[nIdx], strModelPath, "gpt_lm_head_weight");
        }

        private void loadWeightFiles(string strPath, string strModel)
        {
            if (!Directory.Exists(strPath))
                Directory.CreateDirectory(strPath);

            string strModelFile = strPath + "gpt2_model.prototxt";
            if (!System.IO.File.Exists(strModelFile))
                System.IO.File.WriteAllText(strModelFile, strModel);

            string strWtsFile = strPath + "gpt2_weights.zip";
            if (!System.IO.File.Exists(strWtsFile))
            {
                using (WebClient webClient = new WebClient())
                {
                    string strUrl = "https://signalpopcdn.blob.core.windows.net/mycaffesupport/gpt2_weights.zip";
                    string strFile1 = "gpt2_weights.zip";
                    string strFile = strPath + strFile1;

                    m_swUpdateTimer.Start();
                    m_dfLastProgress = 0;
                    
                    webClient.DownloadProgressChanged += WebClient_DownloadProgressChanged;
                    webClient.DownloadFileCompleted += WebClient_DownloadFileCompleted;
                    webClient.DownloadFileAsync(new Uri(strUrl), strFile, strFile1);

                    m_evtDownloadDone.WaitOne();
                }
            }

            string strFirstTarget = strPath + "gpt2_weights\\gpt_lm_head_weight.npy";
            if (!System.IO.File.Exists(strFirstTarget))
            {
                ZipFile.ExtractToDirectory(strWtsFile, strPath + "gpt2_weights");
            }
        }

        private void WebClient_DownloadFileCompleted(object sender, System.ComponentModel.AsyncCompletedEventArgs e)
        {
            bool bTraceEnabled = m_log.EnableTrace;
            m_log.EnableTrace = true;
            m_log.WriteLine("Downloading done.");
            m_log.EnableTrace = bTraceEnabled;

            m_evtDownloadDone.Set();
        }

        private void WebClient_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            if (m_swUpdateTimer.Elapsed.TotalMilliseconds >= 1000)
            {
                if (m_dfLastProgress != e.ProgressPercentage)
                {
                    m_dfLastProgress = e.ProgressPercentage;
                    string strFile = e.UserState.ToString();
                    bool bTraceEnabled = m_log.EnableTrace;
                    m_log.EnableTrace = true;

                    m_log.Progress = e.ProgressPercentage / 100.0;
                    m_log.WriteLine("Downloading '" + strFile + "' at " + m_log.Progress.ToString("P") + "...");
                    m_log.EnableTrace = bTraceEnabled;
                }

                m_swUpdateTimer.Restart();
            }
        }

        private string createGptModel(string strName, string strSrc, int nVocabSize, uint nLayers, uint nHeads, uint nEmbed, uint nBlockSize, int nBatchSize)
        {
            NetParameter p = new NetParameter();
            p.name = strName;

            LayerParameter input = new LayerParameter(LayerParameter.LayerType.INPUT);
            input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { 1, (int)nBlockSize } });
            input.top.Add("tokdata1");
            input.input_param.shape.Add(new BlobShape() { dim = new List<int>() { 1, (int)nBlockSize } });
            input.top.Add("pos");
            p.layer.Add(input);

            LayerParameter embedTok = new LayerParameter(LayerParameter.LayerType.EMBED);
            embedTok.name = "wte";
            embedTok.embed_param.bias_term = false;
            embedTok.embed_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            embedTok.embed_param.input_dim = (uint)nVocabSize;
            embedTok.embed_param.num_output = (uint)nEmbed;
            embedTok.parameters.Add(new ParamSpec(1, 0)); // mult lr, decay
            embedTok.bottom.Add("tokdata1");
            embedTok.top.Add("tok_emb");
            p.layer.Add(embedTok);

            LayerParameter embedPos = new LayerParameter(LayerParameter.LayerType.EMBED);
            embedPos.name = "wpe";
            embedPos.embed_param.bias_term = false;
            embedPos.embed_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            embedPos.embed_param.input_dim = (uint)nBlockSize;
            embedPos.embed_param.num_output = (uint)nEmbed;
            embedPos.parameters.Add(new ParamSpec(1, 0)); // mult lr, decay
            embedPos.bottom.Add("pos");
            embedPos.top.Add("pos_emb");
            p.layer.Add(embedPos);

            LayerParameter add = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            add.name = "eltwise1";
            add.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUM;
            add.eltwise_param.allow_single_batch_input = true;
            add.bottom.Add("tok_emb");
            add.bottom.Add("pos_emb");
            add.top.Add("eltwise1");
            p.layer.Add(add);

            string strTop = "eltwise1";
            LayerParameter drop = new LayerParameter(LayerParameter.LayerType.DROPOUT);
            drop.name = "drop1";
            drop.dropout_param.dropout_ratio = 0.1;
            drop.bottom.Add(strTop);
            drop.top.Add(strTop);
            p.layer.Add(drop);

            for (int i = 0; i < nLayers; i++)
            {
                LayerParameter tfb = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                tfb.name = "tfb" + (i + 1).ToString();
                tfb.transformer_block_param.layers = nLayers;
                tfb.transformer_block_param.heads = nHeads;
                tfb.transformer_block_param.embed = nEmbed;
                tfb.transformer_block_param.block_size = nBlockSize;
                tfb.transformer_block_param.attn_dropout = 0.1;
                tfb.transformer_block_param.resid_dropout = 0.1;
                tfb.parameters.Add(new ParamSpec(1, 1)); // mult lr, decay for Attn c_attn weight/bias
                tfb.parameters.Add(new ParamSpec(1, 0)); // mult lr, decay for Attn c_proj weight/bias
                tfb.parameters.Add(new ParamSpec(1, 0)); // mult lr, decay for FC weight/bias
                tfb.parameters.Add(new ParamSpec(1, 0)); // mult lr, decay for Proj weight/bias
                tfb.bottom.Add(strTop);

                strTop = "tfb" + (i + 1).ToString();
                tfb.top.Add(strTop);
                p.layer.Add(tfb);
            }

            LayerParameter ln = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            ln.name = "ln1";
            ln.bottom.Add(strTop);
            ln.top.Add("ln1");
            p.layer.Add(ln);

            LayerParameter lm_head = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            lm_head.name = "lm_head";
            lm_head.inner_product_param.bias_term = false;
            lm_head.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            lm_head.inner_product_param.num_output = (uint)nVocabSize;
            lm_head.inner_product_param.axis = 2;
            lm_head.parameters.Add(new ParamSpec(1, 1)); // mult lr, decay
            lm_head.bottom.Add("ln1");
            lm_head.top.Add("logits");
            p.layer.Add(lm_head);

            LayerParameter prob = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            prob.name = "prob";
            prob.softmax_param.axis = 2;
            prob.bottom.Add("logits");
            prob.top.Add("prob");
            p.layer.Add(prob);

            RawProto proto = p.ToProto("root");
            return proto.ToString();
        }
        
        private string loadTestData1(string strModel)
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\trfb\\";

            if (!string.IsNullOrEmpty(strModel))
                strPath += strModel + "\\";

            string strFileName = "_transformer_test";

            if (!string.IsNullOrEmpty(strModel))
                strFileName += "_" + strModel;

            strFileName += ".zip";

            string strTestPath = "test\\iter_0";
            string strTestFile = "1_x.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }

        private void load_state(Layer<T> layer, string strPath)
        {
            layer.blobs[0].LoadFromNumpy(strPath + "blk0.attn.c_attn.weight.npy");
            layer.blobs[1].LoadFromNumpy(strPath + "blk0.attn.c_attn.bias.npy");
            layer.blobs[2].LoadFromNumpy(strPath + "blk0.attn.c_proj.weight.npy");
            layer.blobs[3].LoadFromNumpy(strPath + "blk0.attn.c_proj.bias.npy");
            layer.blobs[4].LoadFromNumpy(strPath + "blk0.c_fc.weight.npy");
            layer.blobs[5].LoadFromNumpy(strPath + "blk0.c_fc.bias.npy");
            layer.blobs[6].LoadFromNumpy(strPath + "blk0.c_proj.weight.npy");
            layer.blobs[7].LoadFromNumpy(strPath + "blk0.c_proj.bias.npy");
        }

        /// <summary>
        /// Test the forward pass of the TransformerBlockLayer using the CausalSelfAttention.
        /// </summary>
        /// <remarks>
        /// To regenerate test data, take the following steps:
        /// 1.) constants.py - set mycaffe_layernorm = True, mycaffe_softmax = True, loss_weight = 1, disable_layernorm = False
        /// 2.) main.py - run up to line 104 in trainer.py
        /// 3.) test_transformer.py - run up to line 59.
        /// 4.) MyCaffe CausalSelfAttention configured to use CAFFE version of Softmax
        /// </remarks>
        public void TestForward(string strModel)
        {
            string strPath = loadTestData1(strModel);
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION;

            if (strModel == "pico")
            {
                p.transformer_block_param.heads = 1;
                p.transformer_block_param.embed = 3;
                p.transformer_block_param.block_size = 4;
            }
            else
            {
                p.transformer_block_param.heads = 6;
                p.transformer_block_param.embed = 192;
                p.transformer_block_param.block_size = 128;
            }
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);
            Blob<T> blobVal = new Blob<T>(m_cuda, m_log);

            try
            {
                BlobCollection<T> colBtm = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();

                if (strModel == "pico")
                    blobX.LoadFromNumpy(strPath + "blk0.x.npy");
                else
                    blobX.LoadFromNumpy(strPath + "1_x_emb1.npy");

                colBtm.Add(blobX);
                colTop.Add(blobY);

                layer.Setup(colBtm, colTop);
                load_state(layer, strPath);

                layer.Forward(colBtm, colTop);

                if (strModel == "pico")
                    blobVal.LoadFromNumpy(strPath + "blk0.y.npy");
                else
                    blobVal.LoadFromNumpy(strPath + "12b_out2.npy");

                // TODO: May need to regenerate test data.
                double dfErr = 0.03; // mycaffe_layernorm = True; mycaffe_softmax = True
                m_log.CHECK(blobY.Compare(blobVal, m_blobWork, false, dfErr), "The blobs do not match!");
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);
                dispose(ref blobVal);
                layer.Dispose();
            }
        }

        /// <summary>
        /// Test the backward pass of the TransformerBlockLayer using the CausalSelfAttention.
        /// </summary>
        /// <remarks>
        /// To regenerate test data, take the following steps:
        /// 1.) constants.py - set mycaffe_layernorm = True, mycaffe_softmax = True, loss_weight = 1, disable_layernorm = False
        /// 2.) main.py - run up to line 104 in trainer.py
        /// 3.) test_transformer.py - run up to line 59.
        /// 4.) MyCaffe CausalSelfAttention configured to use CAFFE version of Softmax
        /// </remarks>
        public void TestBackward(string strModel)
        {
            string strPath = loadTestData1(strModel);
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION;
            if (strModel == "pico")
            {
                p.transformer_block_param.heads = 1;
                p.transformer_block_param.embed = 3;
                p.transformer_block_param.block_size = 4;
            }
            else
            {
                p.transformer_block_param.heads = 6;
                p.transformer_block_param.embed = 192;
                p.transformer_block_param.block_size = 128;
            }
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);
            Blob<T> blobVal = new Blob<T>(m_cuda, m_log);

            try
            {
                BlobCollection<T> colBtm = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();

                blobX.LoadFromNumpy(strPath + "1_x_emb1.npy");
                colBtm.Add(blobX);
                colTop.Add(blobY);

                layer.Setup(colBtm, colTop);
                load_state(layer, strPath);

                layer.Forward(colBtm, colTop);

                blobVal.LoadFromNumpy(strPath + "12b_out2.npy");
                //TODO: Regenerate test data.
                double dfErr = 0.03; // mycaffe_layernorm = True; mycaffe_softmax = True
                m_log.CHECK(blobY.Compare(blobVal, m_blobWork, false, dfErr), "The blobs do not match!");

                colTop[0].LoadFromNumpy(strPath + "grad_12b_out2.npy", true);

                layer.Backward(colTop, new List<bool>() { true }, colBtm);

                blobVal.LoadFromNumpy(strPath + "grad_1_x_emb1.npy", true);

                m_log.CHECK(blobVal.Compare(colBtm[0], m_blobWork, true, 2e-04), "The blobs do not match!");
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);
                dispose(ref blobVal);
                layer.Dispose();
            }
        }

        public void TestGradient()
        {
            string strPath = loadTestData1(null);
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION;
            p.transformer_block_param.heads = 6;
            p.transformer_block_param.embed = 192;
            p.transformer_block_param.block_size = 128;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());
            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            Blob<T> blobY = new Blob<T>(m_cuda, m_log);

            try
            {
                BlobCollection<T> colBtm = new BlobCollection<T>();
                BlobCollection<T> colTop = new BlobCollection<T>();

                blobX.LoadFromNumpy(strPath + "1_x_emb1.npy");
                colBtm.Add(blobX);
                colTop.Add(blobY);

                layer.Setup(colBtm, colTop);
                load_state(layer, strPath);

                GradientChecker<T> checker = new GradientChecker<T>(m_cuda, m_log);
                checker.CheckGradient(layer, colBtm, colTop, -1, 200);
            }
            finally
            {
                dispose(ref blobX);
                dispose(ref blobY);
                layer.Dispose();
            }
        }

        private string loadTestData2()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\trfb2\\";
            string strFileName = "_transformer_test2";

            strFileName += ".zip";

            string strTestPath = "test";
            string strTestFile = "iter_0\\1_x.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }

        private string buildModelEx(NetParameter net, uint nBatch, uint nBlockSize, uint nEmbed, uint nEncVocabSize, double dfDropout, Phase phase, int nLayers = 1, bool bEnableLora = false, uint nHiddenDim = 0)
        {
            LayerParameter tok = new LayerParameter(LayerParameter.LayerType.TOKENIZED_DATA);
            tok.tokenized_data_param.input_type = TokenizedDataParameter.INPUT_TYPE.TEXT_FILE;
            tok.tokenized_data_param.vocabulary_type = TokenizedDataParameter.VOCABULARY_TYPE.CHARACTER;
            tok.tokenized_data_param.source = "$ProgramData$\\MyCaffe\\test_data\\data\\text\\input.txt";
            tok.tokenized_data_param.batch_size = nBatch;
            tok.tokenized_data_param.block_size = nBlockSize;
            tok.top.Add("tokdata");
            tok.top.Add("pos");
            if (phase != Phase.RUN)
                tok.top.Add("tgt");
            net.layer.Add(tok);

            LayerParameter emb1 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb1.name = "wte";
            emb1.embed_param.bias_term = false;
            emb1.embed_param.input_dim = nEncVocabSize;
            emb1.embed_param.num_output = nEmbed;
            emb1.embed_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            emb1.parameters.Add(new ParamSpec(1.0, 0.0));
            emb1.bottom.Add("tokdata");
            emb1.top.Add("tok_emb");
            emb1.freeze_learning = bEnableLora;
            net.layer.Add(emb1);

            LayerParameter emb2 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb2.name = "wpe";
            emb2.embed_param.bias_term = false;
            emb2.embed_param.input_dim = nBlockSize;
            emb2.embed_param.num_output = nEmbed;
            emb2.embed_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            emb2.parameters.Add(new ParamSpec(1.0, 0.0));
            emb2.bottom.Add("pos");
            emb2.top.Add("tok_pos");
            emb2.freeze_learning = bEnableLora;
            net.layer.Add(emb2);

            LayerParameter elt = new LayerParameter(LayerParameter.LayerType.ELTWISE);
            elt.name = "eltwise1";
            elt.eltwise_param.operation = EltwiseParameter.EltwiseOp.SUM;
            elt.eltwise_param.allow_single_batch_input = true;
            elt.bottom.Add("tok_emb");
            elt.bottom.Add("tok_pos");
            elt.top.Add("eltwise1");
            elt.freeze_learning = bEnableLora;
            net.layer.Add(elt);

            string strEncBtm = "eltwise1";
            for (int i = 0; i < nLayers; i++)
            {
                LayerParameter enc = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                enc.name = "tfb" + (i + 1).ToString();
                enc.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION;
                enc.transformer_block_param.heads = 1;
                enc.transformer_block_param.embed = nEmbed;
                enc.transformer_block_param.hidden_dim = nHiddenDim;
                enc.transformer_block_param.block_size = nBlockSize;
                enc.transformer_block_param.layers = (uint)nLayers;
                enc.transformer_block_param.activation = TransformerBlockParameter.ACTIVATION.RELU;
                enc.transformer_block_param.attn_dropout = dfDropout;
                enc.transformer_block_param.resid_dropout = dfDropout;
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 0));
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 0));
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 0));
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 0));
                enc.bottom.Add(strEncBtm);
                enc.top.Add(enc.name);
                enc.freeze_learning = bEnableLora;
                net.layer.Add(enc);

                strEncBtm = enc.name;
            }

            LayerParameter ln1 = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            ln1.name = "ln1";
            ln1.layer_norm_param.enable_cuda_impl = false;
            ln1.bottom.Add(strEncBtm);
            ln1.top.Add("ln1");
            ln1.freeze_learning = bEnableLora;
            net.layer.Add(ln1);

            LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ip1.name = "ip1";
            ip1.inner_product_param.axis = 2;
            ip1.inner_product_param.num_output = nEncVocabSize;
            ip1.inner_product_param.bias_term = false;
            ip1.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            ip1.parameters.Add(new ParamSpec(1, 1));
            ip1.bottom.Add("ln1");
            ip1.top.Add("logits");
            ip1.freeze_learning = bEnableLora;
            net.layer.Add(ip1);

            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            softmax.name = "softmax";
            softmax.softmax_param.axis = 2;
            softmax.softmax_param.algorithm = SOFTMAX_ALGORITHM.ACCURATE;
            softmax.softmax_param.algorithm_train = SOFTMAX_ALGORITHM.LOG;
            softmax.bottom.Add("logits");
            softmax.top.Add("prob");
            softmax.freeze_learning = bEnableLora;
            net.layer.Add(softmax);

            if (phase != Phase.RUN)
            {
                LayerParameter loss = new LayerParameter(LayerParameter.LayerType.NLL_LOSS);
                loss.name = "loss";
                loss.nll_loss_param.axis = 2;
                loss.loss_param.normalization = LossParameter.NormalizationMode.VALID;
                loss.bottom.Add("prob");
                loss.bottom.Add("tgt");
                loss.top.Add("loss");
                net.layer.Add(loss);

                LayerParameter accuracy = new LayerParameter(LayerParameter.LayerType.ACCURACY);
                accuracy.name = "accuracy";
                accuracy.accuracy_param.axis = 2;
                accuracy.bottom.Add("prob");
                accuracy.bottom.Add("tgt");
                accuracy.top.Add("accuracy");
                net.layer.Add(accuracy);
            }

            return net.ToProto("root").ToString();
        }

        private string buildSolver()
        {
            SolverParameter solver = new SolverParameter();
            solver.base_lr = 5e-04;
            solver.weight_decay = 0;
            solver.adamw_decay = 0.1;
            solver.momentum = 0.9;
            solver.momentum2 = 0.95;
            solver.delta = 1e-08;
            solver.type = SolverParameter.SolverType.ADAMW;
            solver.lr_policy = "fixed";
            solver.test_initialization = false;

            return solver.ToProto("root").ToString();
        }

        /// <summary>
        /// Test the training cycle of the TransformerBlockLayer using the CausalSelfAttention.
        /// </summary>
        /// <remarks>
        /// To regenerate test data, take the following steps:
        /// 1.) constants.py - set mycaffe_layernorm = True, mycaffe_softmax = True, loss_weight = 1, disable_layernorm = False, model_type = 'gpt_nano1'
        /// 2.) main.py - run up to line 104 in trainer.py
        /// 3.) test_transformer.py - run up to line 59.
        /// 4.) MyCaffe CausalSelfAttention configured to use CAFFE version of Softmax
        /// </remarks>
        public void TestTrain()
        {
            string strPath = loadTestData2();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<float> mycaffe = new MyCaffeControl<float>(s, m_log, m_evtCancel);
            Blob<float> blobVal = null;
            Blob<float> blobWork = null;

            try
            {
                NetParameter net_param = new NetParameter();
                string strSolver = buildSolver();
                string strModel = buildModelEx(net_param, 1, 64, 128, 65, 0.0, Phase.TRAIN);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, null,false, false);
                blobVal = mycaffe.CreateBlob("val");
                blobWork = mycaffe.CreateBlob("work");

                Net<float> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Solver<float> solver = mycaffe.GetInternalSolver();

                net.learnable_parameters[0].LoadFromNumpy(strPath + "iter_0\\1_wte.weight.npy");
                net.learnable_parameters[1].LoadFromNumpy(strPath + "iter_0\\1_wpe.weight.npy");
                net.learnable_parameters[2].LoadFromNumpy(strPath + "iter_0\\blk0.csa.c_attn.weight.npy");
                net.learnable_parameters[3].LoadFromNumpy(strPath + "iter_0\\blk0.csa.c_attn.bias.npy");
                net.learnable_parameters[4].LoadFromNumpy(strPath + "iter_0\\blk0.csa.c_proj.weight.npy");
                net.learnable_parameters[5].LoadFromNumpy(strPath + "iter_0\\blk0.csa.c_proj.bias.npy");
                net.learnable_parameters[6].LoadFromNumpy(strPath + "iter_0\\blk0.c_fc.weight.npy");
                net.learnable_parameters[7].LoadFromNumpy(strPath + "iter_0\\blk0.c_fc.bias.npy");
                net.learnable_parameters[8].LoadFromNumpy(strPath + "iter_0\\blk0.c_proj.weight.npy");
                net.learnable_parameters[9].LoadFromNumpy(strPath + "iter_0\\blk0.c_proj.bias.npy");
                net.learnable_parameters[10].LoadFromNumpy(strPath + "iter_0\\13_lm_head.weight.npy");

                for (int i = 0; i < 1; i++)
                {
                    string strPath1 = strPath + "iter_" + i.ToString() + "\\";

                    blobVal.LoadFromNumpy(strPath1 + "1_wte.weight.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[0], blobWork, false, 1e-08), "1_wte.weight.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "1_wpe.weight.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[1], blobWork, false, 1e-08), "1_wpe.weight.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_attn.weight.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[2], blobWork, false, 1e-08), "blk0.csa.c_attn.weight.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_attn.bias.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[3], blobWork, false, 1e-08), "blk0.csa.c_attn.bias.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_proj.weight.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[4], blobWork, false, 1e-08), "blk0.csa.c_proj.weight.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_proj.bias.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[5], blobWork, false, 1e-08), "blk0.csa.c_proj.bias.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_fc.weight.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[6], blobWork, false, 1e-08), "blk0.c_fc.weight.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_fc.bias.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[7], blobWork, false, 1e-08), "blk0.c_fc.bias.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_proj.weight.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[8], blobWork, false, 1e-08), "blk0.c_proj.weight.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_proj.bias.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[9], blobWork, false, 1e-08), "blk0.c_proj.bias.npy does not match!");

                    net.ClearParamDiffs();

                    double dfLoss = 0;
                    for (int j = 0; j < net.layers.Count; j++)
                    {
                        BlobCollection<float> colTop = net.top_vecs[j];
                        BlobCollection<float> colBottom = net.bottom_vecs[j];

                        if (j == 3)
                        {
                            blobVal.LoadFromNumpy(strPath1 + "1_wpe.weight.npy");
                            m_log.CHECK(blobVal.Compare(net.layers[j].blobs[0], blobWork, false, 1e-08), "1_wpe.weight.npy does not match!");
                        }

                        double dfLayerLoss = net.layers[j].Forward(colBottom, colTop);
                        dfLoss += dfLayerLoss;

                        string strTopData = null;
                        string strBottomData = null;
                        string strLayer = net.layers[j].layer_param.name;
                        Trace.WriteLine("LAYER: " + strLayer);
                        switch (j)
                        {
                            case 0:
                                colTop[0].LoadFromNumpy(strPath1 + "1_x.npy");
                                colTop[2].LoadFromNumpy(strPath1 + "1_targets.npy");
                                break;

                            case 2:
                                strBottomData = "1_x.npy";
                                strTopData = "0_tok_emb.npy";
                                break;

                            case 3:
                                strBottomData = "0_pos.npy";
                                strTopData = "0_pos_emb.npy";
                                break;

                            case 4:
                                strTopData = "0_x.npy";
                                break;

                            case 5:
                                strBottomData = "0_x.npy";
                                strTopData = "0_blk_x.npy";
                                break;

                            case 6:
                                strBottomData = "0_blk_x.npy";
                                strTopData = "12_ln_x.npy";
                                break;

                            case 7:
                                strBottomData = "12_ln_x.npy";
                                strTopData = "13_logits.npy";
                                break;
                        }

                        if (!string.IsNullOrEmpty(strBottomData))
                        {
                            blobVal.LoadFromNumpy(strPath1 + strBottomData);
                            m_log.CHECK(blobVal.Compare(colBottom[0], blobWork, false, 1e-08), strBottomData + " does not match!");
                        }

                        if (!string.IsNullOrEmpty(strTopData))
                        {
                            blobVal.LoadFromNumpy(strPath1 + strTopData);
                            m_log.CHECK(blobVal.Compare(colTop[0], blobWork, false, 1e-08), strTopData + " does not match!");
                        }
                    }

                    net.blobs[net.blobs.Count - 1].SetDiff(1);

                    for (int j = net.layer_need_backward.Count-1; j >= 0; j--)
                    {
                        if (net.layer_need_backward[j])
                        {
                            BlobCollection<float> colTop = net.top_vecs[j];
                            BlobCollection<float> colBottom = net.bottom_vecs[j];
                            List<bool> rgNeedBackward = net.bottom_need_backward[j];

                            net.layers[j].Backward(colTop, rgNeedBackward, colBottom);

                            string strBottomDiff = null;
                            string strTopData = null;
                            string strBottomData = null;
                            switch (j)
                            {
                                case 10:
                                    strBottomDiff = "grad_14_prob.npy"; // NLLLoss -> prob
                                    break;

                                case 8:
                                    strBottomDiff = "grad_13_logits.npy"; // LogSoftmax -> logits
                                    strBottomData = "13_logits.npy";
                                    strTopData = "14_prob.npy";
                                    break;

                                case 7:
                                    strBottomDiff = "grad_12_ln_x.npy"; // Linear -> LayerNorm
                                    strBottomData = "12_ln_x.npy";
                                    strTopData = "13_logits.npy";
                                    break;
                            }

                            if (!string.IsNullOrEmpty(strBottomDiff))
                            {
                                blobVal.LoadFromNumpy(strPath1 + strBottomDiff, true);
                                m_log.CHECK(blobVal.Compare(colBottom[0], blobWork, true, 1e-08), strBottomDiff + " does not match!");

                                if (!string.IsNullOrEmpty(strTopData))
                                {
                                    blobVal.LoadFromNumpy(strPath1 + strTopData);
                                    m_log.CHECK(blobVal.Compare(colTop[0], blobWork, false, 1e-08), strTopData + " does not match!");
                                }

                                if (!string.IsNullOrEmpty(strBottomData))
                                {
                                    blobVal.LoadFromNumpy(strPath1 + strBottomData);
                                    m_log.CHECK(blobVal.Compare(colBottom[0], blobWork, false, 1e-08), strBottomData + " does not match!");
                                }
                            }
                        }
                    }

                    blobVal.LoadFromNumpy(strPath1 + "____.grad_lm_head.wt.npy", true);
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[10], blobWork, true, 1e-08), "___.grad_lm_head.wt.npy does not match!");

                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.c_proj.bias.npy", true);
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[9], blobWork, true, 6e-08), "___.grad_blk0.c_proj.bias.npy does not match!"); //<<-bug
                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.c_proj.wt.npy", true);
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[8], blobWork, true, 1e-08), "___.grad_blk0.c_proj.wt.npy does not match!");

                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.c_fc.bias.npy", true);
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[7], blobWork, true, 1e-08), "___.grad_blk0.c_fc.bias.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.c_fc.wt.npy", true);
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[6], blobWork, true, 1e-08), "___.grad_blk0.c_fc.wt.npy does not match!");

                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.csa.c_proj.bias.npy", true);
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[5], blobWork, true, 3e-07), "___.grad_blk0.csa.c_proj.bias.npy does not match!"); //<<-bug?
                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.csa.c_proj.wt.npy", true);
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[4], blobWork, true, 1e-08), "___.grad_blk0.csa.c_proj.wt.npy does not match!");

                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.csa.c_attn.bias.npy", true);
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[3], blobWork, true, 3e-08), "___.grad_blk0.csa.c_attn.bias.npy does not match!"); //<<-bug?
                    blobVal.LoadFromNumpy(strPath1 + "____.grad_blk0.csa.c_attn.wt.npy", true);
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[2], blobWork, true, 3e-08), "___.grad_blk0.csa.c_attn.wt.npy does not match!"); //<<-bug?

                    blobVal.LoadFromNumpy(strPath1 + "____.grad_tfb.wpe.wt.npy", true);
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[1], blobWork, true, 2e-07), "___.grad_tfb.wpe.wt.npy does not match!"); //<<-bug?
                    blobVal.LoadFromNumpy(strPath1 + "____.grad_tfb.wte.wt.npy", true);
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[0], blobWork, true, 3e-07), "___.grad.tfb.wte.wt.npy does not match!"); //<<-bug?

                    solver.ApplyUpdate(i);

                    blobVal.LoadFromNumpy(strPath1 + "____.param_lm_head.wt.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[10], blobWork, false, 2e-08), "___.param_lm_head.wt.npy does not match!");

                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.c_proj.bias.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[9], blobWork), "___.param_blk0.c_proj.bias.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.c_proj.wt.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[8], blobWork), "___.param_blk0.c_proj.wt.npy does not match!");

                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.c_fc.bias.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[7], blobWork), "___.param_blk0.c_fc.bias.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.c_fc.wt.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[6], blobWork, false, 2e-08), "___.param_blk0.c_fc.wt.npy does not match!");

                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.csa.c_proj.bias.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[5], blobWork), "___.param_blk0.csa.c_proj.bias.npy does not match!"); 
                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.csa.c_proj.wt.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[4], blobWork), "___.param_blk0.csa.c_proj.wt.npy does not match!");

                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.csa.c_attn.bias.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[3], blobWork, false, 2e-04), "___.param_blk0.csa.c_attn.bias.npy does not match!"); //<<-bug?
                    blobVal.LoadFromNumpy(strPath1 + "____.param_blk0.csa.c_attn.wt.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[2], blobWork, false, 2e-08), "___.param_blk0.csa.c_attn.wt.npy does not match!"); 

                    blobVal.LoadFromNumpy(strPath1 + "____.param_tfb.wpe.wt.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[1], blobWork), "___.param_tfb.wpe.wt.npy does not match!");
                    blobVal.LoadFromNumpy(strPath1 + "____.param_tfb.wte.wt.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[0], blobWork), "___.param_tfb.wte.wt.npy does not match!");
                }
            }
            finally
            {
                if (blobVal != null)
                    blobVal.Dispose();

                if (blobWork != null)
                    blobWork.Dispose();

                mycaffe.Dispose();
            }
        }

        /// <summary>
        /// Test loading and training large multi-layer training cycle of the TransformerBlockLayer using the CausalSelfAttention.
        /// </summary>
        public void TestTrainLargeLM()
        {
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<float> mycaffe = new MyCaffeControl<float>(s, m_log, m_evtCancel);
            Blob<float> blobVal = null;
            Blob<float> blobWork = null;

            try
            {
                NetParameter net_param = new NetParameter();
                net_param.enable_lora_only = true;
                net_param.enable_memory_stats = true;

                string strSolver = buildSolver();
                string strModel = buildModelEx(net_param, 1, 512, 2048, 32000, 0.0, Phase.TRAIN, 32, true, 11008);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, null,false, false);
                blobVal = mycaffe.CreateBlob("val");
                blobWork = mycaffe.CreateBlob("work");

                Net<float> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Solver<float> solver = mycaffe.GetInternalSolver();

                for (int i = 0; i < 1; i++)
                {
                    net.ClearParamDiffs();
                    net.Forward();
                    net.Backward();
                    solver.ApplyUpdate(i);
                }
            }
            finally
            {
                if (blobVal != null)
                    blobVal.Dispose();

                if (blobWork != null)
                    blobWork.Dispose();

                mycaffe.Dispose();
            }
        }

        private string loadTestData3()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\trfb3\\";
            string strFileName = "_transformer_test3";

            strFileName += ".zip";

            string strTestPath = "test";
            string strTestFile = "0_blk_x.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }

        /// <summary>
        /// Test the inference cycle of the TransformerBlockLayer using the CausalSelfAttention.
        /// </summary>
        /// <remarks>
        /// To regenerate test data, take the following steps:
        /// 1.) constants.py - set enable_inference_save_testing = True
        /// 1.) constants.py - set mycaffe_layernorm = True, mycaffe_softmax = True, loss_weight = 1, disable_layernorm = False, model_type = 'gpt_nano1'
        /// 2.) main.py - run up to line 104 in trainer.py
        /// 3.) test_transformer.py - run up to line 59.
        /// 4.) MyCaffe CausalSelfAttention configured to use CAFFE version of Softmax
        /// </remarks>
        public void TestInference()
        {
            string strPath = loadTestData3();
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<float> mycaffe = new MyCaffeControl<float>(s, m_log, m_evtCancel);
            Blob<float> blobInput = null;
            Blob<float> blobVal = null;
            Blob<float> blobWork = null;    

            try
            {
                NetParameter net_param = new NetParameter();
                string strModel = buildModelEx(net_param, 1, 64, 128, 65, 0.0, Phase.RUN);

                BlobShape shape = new BlobShape(1, 64, 1, 1);

                mycaffe.LoadToRun(strModel, null, null, shape);
                blobVal = mycaffe.CreateBlob("val");
                blobWork = mycaffe.CreateBlob("work");

                Net<float> net = mycaffe.GetInternalNet(Phase.RUN);

                net.learnable_parameters[0].LoadFromNumpy(strPath + "1_wte.weight.npy");
                net.learnable_parameters[1].LoadFromNumpy(strPath + "1_wpe.weight.npy");
                net.learnable_parameters[2].LoadFromNumpy(strPath + "blk0.csa.c_attn.weight.npy");
                net.learnable_parameters[3].LoadFromNumpy(strPath + "blk0.csa.c_attn.bias.npy");
                net.learnable_parameters[4].LoadFromNumpy(strPath + "blk0.csa.c_proj.weight.npy");
                net.learnable_parameters[5].LoadFromNumpy(strPath + "blk0.csa.c_proj.bias.npy");
                net.learnable_parameters[6].LoadFromNumpy(strPath + "blk0.c_fc.weight.npy");
                net.learnable_parameters[7].LoadFromNumpy(strPath + "blk0.c_fc.bias.npy");
                net.learnable_parameters[8].LoadFromNumpy(strPath + "blk0.c_proj.weight.npy");
                net.learnable_parameters[9].LoadFromNumpy(strPath + "blk0.c_proj.bias.npy");
                net.learnable_parameters[10].LoadFromNumpy(strPath + "13_lm_head.weight.npy");

                // Run forward test pass
                { 
                    string strPath1 = strPath;

                    blobVal.LoadFromNumpy(strPath1 + "1_wte.weight.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[0], blobWork, false, 1e-08), "1_wte.weight.npy does not match");
                    blobVal.LoadFromNumpy(strPath1 + "1_wpe.weight.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[1], blobWork, false, 1e-08), "1_wpe.weight.npy does not match");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_attn.weight.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[2], blobWork, false, 1e-08), "blk0.csa.c_attn.weight.npy does not match");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_attn.bias.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[3], blobWork, false, 1e-08), "blk0.csa.c_attn.bias.npy does not match");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_proj.weight.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[4], blobWork, false, 1e-08), "blk0.csa.c_proj.weight.npy does not match");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.csa.c_proj.bias.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[5], blobWork, false, 1e-08), "blk0.csa.c_proj.bias.npy does not match.");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_fc.weight.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[6], blobWork, false, 1e-08), "blk0.c_fc.weight.npy does not match");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_fc.bias.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[7], blobWork, false, 1e-08), "blk0.c_fc.bias.npy does not match");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_proj.weight.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[8], blobWork, false, 1e-08), "blk0.c_proj.weight.npy does not match");
                    blobVal.LoadFromNumpy(strPath1 + "blk0.c_proj.bias.npy");
                    m_log.CHECK(blobVal.Compare(net.learnable_parameters[9], blobWork, false, 1e-08), "blk0.c_proj.bias.npy does not match");

                    double dfLoss = 0;
                    for (int j = 0; j < net.layers.Count; j++)
                    {
                        BlobCollection<float> colTop = net.top_vecs[j];
                        BlobCollection<float> colBottom = net.bottom_vecs[j];

                        if (j == 0)
                        {
                            PropertySet input = new PropertySet();
                            input.SetProperty("InputData", "O God, O God!");
                            TokenizedDataLayer<float> toklayer = net.layers[j] as TokenizedDataLayer<float>;

                            int nSeqLen;
                            BlobCollection<float> col = toklayer.PreProcessInput(input, out nSeqLen);
                            blobInput = col[0];
                            colBottom[0].CopyFrom(blobInput, false, true);
                        }

                        if (j == 2)
                        {
                            blobVal.LoadFromNumpy(strPath1 + "1_wpe.weight.npy");
                            m_log.CHECK(blobVal.Compare(net.layers[j].blobs[0], blobWork, false, 1e-08), "1_wpe.weight.npy does not match");
                        }

                        double dfLayerLoss = net.layers[j].Forward(colBottom, colTop);
                        dfLoss += dfLayerLoss;

                        string strTopData0 = null;
                        double dfTopErr0 = 1e-08;
                        string strTopData1 = null;
                        double dfTopErr1 = 1e-08;
                        string strBottomData = null;
                        double dfBtmErr = 1e-08;
                        string strLayer = net.layers[j].layer_param.name;
                        Trace.WriteLine("LAYER: " + strLayer);
                        switch (j)
                        {
                            case 0:
                                strTopData0 = "1_x.npy";
                                strTopData1 = "0_pos.npy";
                                break;

                            case 1:
                                strBottomData = "1_x.npy";
                                strTopData0 = "0_tok_emb.npy";
                                break;

                            case 2:
                                strBottomData = "0_pos.npy";
                                strTopData0 = "0_pos_emb.npy";
                                break;

                            case 3:
                                strTopData0 = "0_x.npy";
                                break;

                            case 4:
                                strBottomData = "0_x.npy";
                                strTopData0 = "0_blk_x.npy";
                                dfTopErr0 = 5e-07;
                                break;

                            case 5:
                                strBottomData = "0_blk_x.npy";
                                dfBtmErr = 5e-07;
                                strTopData0 = "12_ln_x.npy";
                                dfTopErr0 = 8e-07;
                                break;

                            case 6:
                                strBottomData = "12_ln_x.npy";
                                dfBtmErr = 8e-07;
                                strTopData0 = "13_logits.npy";
                                dfTopErr0 = 2e-06;
                                break;
                        }

                        if (!string.IsNullOrEmpty(strBottomData))
                        {
                            blobVal.LoadFromNumpy(strPath1 + strBottomData);
                            m_log.CHECK(blobVal.Compare(colBottom[0], blobWork, false, dfBtmErr), strBottomData + " does not match");
                        }

                        if (!string.IsNullOrEmpty(strTopData0))
                        {
                            blobVal.LoadFromNumpy(strPath1 + strTopData0);
                            m_log.CHECK(blobVal.Compare(colTop[0], blobWork, false, dfTopErr0), strTopData0 + " does not match");
                        }

                        if (!string.IsNullOrEmpty(strTopData1))
                        {
                            blobVal.LoadFromNumpy(strPath1 + strTopData1);
                            m_log.CHECK(blobVal.Compare(colTop[1], blobWork, false, dfTopErr1), strTopData1 + " does not match");
                        }
                    }
                }
            }
            finally
            {
                if (blobVal != null)
                    blobVal.Dispose();

                if (blobWork != null)
                    blobWork.Dispose();

                if (blobInput != null)
                    blobInput.Dispose();

                mycaffe.Dispose();
            }
        }

        private string getTestDataLlamaPath(string strSubPath, string strFile)
        {
            if (!string.IsNullOrEmpty(strSubPath))
                strSubPath += "\\";

            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\llama\\test\\" + strSubPath + "iter_0\\";

            if (!File.Exists(strPath + strFile))
                throw new Exception("Could not find the test data file '" + strPath + strFile + "'.  You may need to run the 'Test|Download Test Data | Llama' menu item.");

            return strPath;
        }

        private string getTestDataLlamaPathWt(string strSubPath, string strFile)
        {
            if (!string.IsNullOrEmpty(strSubPath))
                strSubPath += "\\";

            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\llama\\test\\" + strSubPath + "weights\\";

            if (!File.Exists(strPath + strFile))
                throw new Exception("Could not find the test data file '" + strPath + strFile + "'.  You may need to run the 'Test|Download Test Data | Llama' menu item.");

            return strPath;
        }

        private string buildModelLlama(NetParameter net, uint nBatch, uint nBlockSize, uint nEmbed, uint nHeads, uint nEncVocabSize, double dfDropout, Phase phase, int nLayers = 1, bool bEnableLora = false)
        {
            LayerParameter data = new LayerParameter(LayerParameter.LayerType.INPUT);
            data.input_param.shape.Add(new BlobShape(new List<int>() { (int)nBatch, (int)nBlockSize }));
            data.input_param.shape.Add(new BlobShape(new List<int>() { (int)nBatch, (int)nBlockSize }));
            data.top.Add("x");
            data.top.Add("y");
            net.layer.Add(data);

            LayerParameter emb1 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb1.name = "wte";
            emb1.embed_param.bias_term = false;
            emb1.embed_param.input_dim = nEncVocabSize;
            emb1.embed_param.num_output = nEmbed;
            emb1.embed_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            emb1.parameters.Add(new ParamSpec(1.0, 0.0));
            emb1.bottom.Add("x");
            emb1.top.Add("tok_emb");
            emb1.freeze_learning = bEnableLora;
            net.layer.Add(emb1);

            string strEncBtm = "tok_emb";
            for (int i = 0; i < nLayers; i++)
            {
                LayerParameter enc = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                enc.name = "tfb" + (i + 1).ToString();
                enc.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION2;
                enc.transformer_block_param.heads = nHeads;
                enc.transformer_block_param.embed = nEmbed;
                enc.transformer_block_param.hidden_dim = 0;
                enc.transformer_block_param.block_size = nBlockSize;
                enc.transformer_block_param.layers = (uint)nLayers;
                enc.transformer_block_param.activation = TransformerBlockParameter.ACTIVATION.SILU;
                enc.transformer_block_param.attn_dropout = dfDropout;
                enc.transformer_block_param.resid_dropout = dfDropout;
                enc.transformer_block_param.bias_term = false;
                enc.transformer_block_param.enable_rotary_positional_embedding = true;
                enc.transformer_block_param.normalization_type = TransformerBlockParameter.NORMALIZATION.RMS_NORM;
                enc.transformer_block_param.enable_llama_style_head = true;
                enc.transformer_block_param.multiple_of = 2;
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.bottom.Add(strEncBtm);
                enc.top.Add(enc.name);
                enc.freeze_learning = bEnableLora;
                net.layer.Add(enc);

                strEncBtm = enc.name;
            }

            LayerParameter rms1 = new LayerParameter(LayerParameter.LayerType.RMSNORM);
            rms1.name = "rms1";
            rms1.rms_norm_param.axis = 2;
            rms1.bottom.Add(strEncBtm);
            rms1.top.Add("rms1");
            rms1.freeze_learning = bEnableLora;
            net.layer.Add(rms1);

            LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ip1.name = "ip1";
            ip1.inner_product_param.axis = 2;
            ip1.inner_product_param.num_output = nEncVocabSize;
            ip1.inner_product_param.bias_term = false;
            ip1.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            ip1.parameters.Add(new ParamSpec(1, 1));
            ip1.bottom.Add("rms1");
            ip1.top.Add("logits");
            ip1.freeze_learning = bEnableLora;
            net.layer.Add(ip1);

            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            softmax.name = "softmax";
            softmax.softmax_param.axis = 2;
            softmax.softmax_param.algorithm = SOFTMAX_ALGORITHM.ACCURATE;
            softmax.softmax_param.algorithm_train = SOFTMAX_ALGORITHM.LOG;
            softmax.bottom.Add("logits");
            softmax.top.Add("prob");
            softmax.freeze_learning = bEnableLora;
            net.layer.Add(softmax);

            if (phase != Phase.RUN)
            {
                LayerParameter loss = new LayerParameter(LayerParameter.LayerType.NLL_LOSS);
                loss.name = "loss";
                loss.nll_loss_param.axis = 2;
                loss.loss_param.normalization = LossParameter.NormalizationMode.VALID;
                loss.bottom.Add("prob");
                loss.bottom.Add("y");
                loss.top.Add("loss");
                net.layer.Add(loss);

                LayerParameter accuracy = new LayerParameter(LayerParameter.LayerType.ACCURACY);
                accuracy.name = "accuracy";
                accuracy.accuracy_param.axis = 2;
                accuracy.bottom.Add("prob");
                accuracy.bottom.Add("y");
                accuracy.top.Add("accuracy");
                net.layer.Add(accuracy);
            }

            return net.ToProto("root").ToString();
        }

        private string buildSolverLlama()
        {
            SolverParameter solver = new SolverParameter();
            solver.base_lr = 5e-04;
            solver.weight_decay = 0;
            solver.adamw_decay = 0.1;
            solver.momentum = 0.9;
            solver.momentum2 = 0.95;
            solver.delta = 1e-08;
            solver.type = SolverParameter.SolverType.ADAMW;
            solver.lr_policy = "fixed";
            solver.test_initialization = false;

            return solver.ToProto("root").ToString();
        }

        /// <summary>
        /// Test the training cycle of the TransformerBlockLayer using the CausalSelfAttention.
        /// </summary>
        /// <remarks>
        /// To regenerate test data, take the following steps:
        /// 1.) constants.py - set mycaffe_layernorm = True, mycaffe_softmax = True, loss_weight = 1, disable_layernorm = False, model_type = 'gpt_nano1'
        /// 2.) main.py - run up to line 104 in trainer.py
        /// 3.) test_transformer.py - run up to line 59.
        /// 4.) MyCaffe CausalSelfAttention configured to use CAFFE version of Softmax
        /// </remarks>
        public void TestTrainLlama()
        {
            string strPath = getTestDataLlamaPath("llama", "mth0.9.output.npy");
            string strPathWt = getTestDataLlamaPathWt("llama", "tok_embeddings.weight.npy");
            SettingsCaffe s = new SettingsCaffe();
            s.GpuIds = "0";
            MyCaffeControl<float> mycaffe = new MyCaffeControl<float>(s, m_log, m_evtCancel);
            Blob<float> blobVal = null;
            Blob<float> blobWork = null;

            try
            {
                uint nDim = 8;
                int nLayers = 1;
                uint nHeads = 2;
                uint nVocab_size = 16;
                uint nSeqLen = 4;
                float fDropout = 0.0f;
                uint nBatch = 1;

                NetParameter net_param = new NetParameter();
                string strSolver = buildSolverLlama();
                string strModel = buildModelLlama(net_param, nBatch, nSeqLen, nDim, nHeads, nVocab_size, fDropout, Phase.TRAIN, nLayers, false);

                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, null,false, false);
                blobVal = mycaffe.CreateBlob("val");
                blobWork = mycaffe.CreateBlob("work");

                Net<float> net = mycaffe.GetInternalNet(Phase.TRAIN);
                Solver<float> solver = mycaffe.GetInternalSolver();

                net.learnable_parameters[0].LoadFromNumpy(strPathWt + "tok_embeddings.weight.npy");
                net.learnable_parameters[1].LoadFromNumpy(strPathWt + "layers.0.attention_norm.weight.npy");
                net.learnable_parameters[2].LoadFromNumpy(strPathWt + "layers.0.attention.wq.weight.npy");
                net.learnable_parameters[3].LoadFromNumpy(strPathWt + "layers.0.attention.wk.weight.npy");
                net.learnable_parameters[4].LoadFromNumpy(strPathWt + "layers.0.attention.wv.weight.npy");
                net.learnable_parameters[5].LoadFromNumpy(strPathWt + "layers.0.attention.wo.weight.npy");
                net.learnable_parameters[6].LoadFromNumpy(strPathWt + "layers.0.ffn_norm.weight.npy");
                net.learnable_parameters[7].LoadFromNumpy(strPathWt + "layers.0.feed_forward.w1.weight.npy");
                net.learnable_parameters[8].LoadFromNumpy(strPathWt + "layers.0.feed_forward.w3.weight.npy");
                net.learnable_parameters[9].LoadFromNumpy(strPathWt + "layers.0.feed_forward.w2.weight.npy");
                net.learnable_parameters[10].LoadFromNumpy(strPathWt + "norm.weight.npy");
                net.learnable_parameters[11].LoadFromNumpy(strPathWt + "output.weight.npy");

                Blob<float> blobX = net.FindBlob("x");
                Blob<float> blobY = net.FindBlob("y");

                blobX.LoadFromNumpy(strPath + "tfm.tokens.npy");
                blobX.LoadFromNumpy(strPath + "tfm.targets.npy");

                net.Forward();

                Blob<float> blobLogits = net.FindBlob("logits");

                blobVal.LoadFromNumpy(strPath + "tfm.logits.npy");
                m_log.CHECK(blobVal.Compare(blobLogits, blobWork, false, 2e-08), "The logits do not match!");

                //*** Backward Pass ***

                blobLogits.LoadFromNumpy(strPath + "tfm.logits.grad.npy", true);

                // Start at the logit Linear output layer.
                int nLayerIdOutput = 5;
                net.Backward(nLayerIdOutput);

                blobX = net.FindBlob("tok_emb");
                blobVal.LoadFromNumpy(strPath + "tfm.h_emb.grad.npy", true);
                m_log.CHECK(blobVal.Compare(blobX, blobWork, true, 2e-07), "The tok_emb gradient does not match!");
            }
            finally
            {
                if (blobVal != null)
                    blobVal.Dispose();

                if (blobWork != null)
                    blobWork.Dispose();

                mycaffe.Dispose();
            }
        }
    }
}
