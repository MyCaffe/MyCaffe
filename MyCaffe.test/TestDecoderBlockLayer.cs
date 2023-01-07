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
using System.Diagnostics;
using System.IO;
using MyCaffe.param.gpt;
using System.Net;
using System.Threading;
using System.IO.Compression;
using static System.Net.Mime.MediaTypeNames;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.TrackBar;
using System.Drawing.Imaging;
using System.Runtime.CompilerServices;
using System.Windows.Forms;

namespace MyCaffe.test
{
    [TestClass]
    public class TestDecoderBlockLayer
    {
        [TestMethod]
        public void TestForward()
        {
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    t.TestForward(8, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestForwardCuda()
        {
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    t.TestForward(8, true);
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
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    t.TestBackward(8, false);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestBackwardCuda()
        {
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    t.TestBackward(8, true);
                }
            }
            finally
            {
                test.Dispose();
            }
        }

        [TestMethod]
        public void TestTraining()
        {
            DecoderBlockLayerTest test = new DecoderBlockLayerTest(EngineParameter.Engine.CAFFE);

            try
            {
                foreach (IDecoderBlockLayerTest t in test.Tests)
                {
                    if (t.DataType == DataType.FLOAT)
                        t.TestTraining();
                }
            }
            finally
            {
                test.Dispose();
            }
        }
    }

    interface IDecoderBlockLayerTest : ITest
    {
        void TestForward(uint nHeads, bool bEnableCudaImpl);
        void TestBackward(uint nHeads, bool bEnableCudaImpl);
        void TestTraining();
    }

    class DecoderBlockLayerTest : TestBase
    {
        public DecoderBlockLayerTest(EngineParameter.Engine engine = EngineParameter.Engine.DEFAULT)
            : base("Decoder Transformer Block Test", TestBase.DEFAULT_DEVICE_ID, engine)
        {
        }

        protected override ITest create(common.DataType dt, string strName, int nDeviceID, EngineParameter.Engine engine)
        {
            if (dt == common.DataType.DOUBLE)
                return new DecoderBlockLayerTest2<double>(strName, nDeviceID, engine);
            else
                return new DecoderBlockLayerTest2<float>(strName, nDeviceID, engine);
        }
    }

    class DecoderBlockLayerTest2<T> : TestEx<T>, IDecoderBlockLayerTest
    {
        Blob<T> m_blobX;
        Blob<T> m_blobEncOut;
        Blob<T> m_blobMaskEnc;
        Blob<T> m_blobMaskDec;
        Blob<T> m_blobMaskEnc_exp;
        Blob<T> m_blobMaskDec_exp;
        Blob<T> m_blobInput;
        Blob<T> m_blobYexp;

        public DecoderBlockLayerTest2(string strName, int nDeviceID, EngineParameter.Engine engine)
            : base(strName, new List<int>() { 3, 2, 4, 1 }, nDeviceID)
        {
            m_engine = engine;
            m_blobX = new Blob<T>(m_cuda, m_log);
            m_blobEncOut = new Blob<T>(m_cuda, m_log);
            m_blobMaskEnc = new Blob<T>(m_cuda, m_log);
            m_blobMaskDec = new Blob<T>(m_cuda, m_log);
            m_blobMaskEnc_exp = new Blob<T>(m_cuda, m_log);
            m_blobMaskDec_exp = new Blob<T>(m_cuda, m_log);
            m_blobInput = new Blob<T>(m_cuda, m_log);
            m_blobYexp = new Blob<T>(m_cuda, m_log);
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
            dispose(ref m_blobX);
            dispose(ref m_blobEncOut);
            dispose(ref m_blobMaskEnc);
            dispose(ref m_blobMaskDec);
            dispose(ref m_blobMaskEnc_exp);
            dispose(ref m_blobMaskDec_exp);
            dispose(ref m_blobInput);
            dispose(ref m_blobYexp);
            base.dispose();
        }

        private string loadTestData1()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\auto\\dec\\";
            string strFileName = "_decoder_test.zip";
            string strTestPath = "test";
            string strTestFile = "13_out1.npy";
            return loadTestData(strPath, strFileName, strTestPath, strTestFile);
        }

        private void createDecoderMask(Blob<T> blobMask)
        {
            int nBatch = blobMask.num;
            int nDim = blobMask.count(1);

            float[] rgMask = convertF(blobMask.mutable_cpu_data);

            blobMask.Reshape(nBatch, nDim, nDim, 1);
            float[] rgMask1 = new float[nBatch * nDim * nDim];

            for (int n = 0; n < nBatch; n++)
            {
                for (int i = 0; i < nDim; i++)
                {
                    for (int j = 0; j < nDim; j++)
                    {
                        int nIdxSrc = n * nDim + j;
                        float fVal = rgMask[nIdxSrc];

                        if (fVal > 0 && j > i)
                            fVal = 0;

                        int nIdxDst = n * nDim * nDim + i * nDim + j;
                        rgMask1[nIdxDst] = fVal;
                    }
                }
            }

            blobMask.mutable_cpu_data = convert(rgMask1);
        }

        public void TestForward(uint nHeads, bool bEnableCudaImpl)
        {
            string strTestDataPath = loadTestData1();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.DECODER;
            p.transformer_block_param.heads = nHeads;
            p.transformer_block_param.embed = 512;
            p.transformer_block_param.block_size = 200;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            p.transformer_block_param.enable_layernorm_cuda_impl = bEnableCudaImpl;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.TRANSFORMER_BLOCK, "The layer type is incorrect!");

                m_blobX.LoadFromNumpy(strTestDataPath + "dec_in_x0.npy");
                m_blobX.Name = "dec_in_x0";
                m_blobEncOut.LoadFromNumpy(strTestDataPath + "enc_out_x1.npy");
                m_blobEncOut.Name = "enc_out_x1";
                
                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");                
                m_blobMaskEnc.ReshapeLike(m_blobInput);
                m_blobMaskEnc.Name = "e_mask";
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMaskEnc.mutable_gpu_data);
                
                m_blobInput.LoadFromNumpy(strTestDataPath + "trg_input.npy");
                m_blobMaskDec.ReshapeLike(m_blobInput);
                m_blobMaskDec.Name = "d_mask";
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMaskDec.mutable_gpu_data);
                createDecoderMask(m_blobMaskDec);

                m_blobMaskDec_exp.LoadFromNumpy(strTestDataPath + "mh1.1_mask.npy");
                verify(m_blobMaskDec, m_blobMaskDec_exp, false, 1e-09);

                m_blobMaskEnc_exp.LoadFromNumpy(strTestDataPath + "mh2.1_mask.npy");
                verify(m_blobMaskEnc, m_blobMaskEnc_exp, false, 1e-09);

                BottomVec.Clear();
                BottomVec.Add(m_blobX);
                BottomVec.Add(m_blobMaskDec);
                BottomVec.Add(m_blobEncOut);
                BottomVec.Add(m_blobMaskEnc);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strTestDataPath + "mh1.w_q_weight.npy");    // multi-head query weight
                layer.blobs[1].LoadFromNumpy(strTestDataPath + "mh1.w_q_bias.npy");      // multi-head query bias
                layer.blobs[2].LoadFromNumpy(strTestDataPath + "mh1.w_k_weight.npy");    // multi-head key weight
                layer.blobs[3].LoadFromNumpy(strTestDataPath + "mh1.w_k_bias.npy");      // multi-head key bias
                layer.blobs[4].LoadFromNumpy(strTestDataPath + "mh1.w_v_weight.npy");    // multi-head value weight
                layer.blobs[5].LoadFromNumpy(strTestDataPath + "mh1.w_v_bias.npy");      // multi-head value bias
                layer.blobs[6].LoadFromNumpy(strTestDataPath + "mh1.w_o_weight.npy");    // multi-head output weight
                layer.blobs[7].LoadFromNumpy(strTestDataPath + "mh1.w_o_bias.npy");      // multi-head output bias

                layer.blobs[8].LoadFromNumpy(strTestDataPath + "mh2.w_q_weight.npy");    // multi-head query weight
                layer.blobs[9].LoadFromNumpy(strTestDataPath + "mh2.w_q_bias.npy");      // multi-head query bias
                layer.blobs[10].LoadFromNumpy(strTestDataPath + "mh2.w_k_weight.npy");   // multi-head key weight
                layer.blobs[11].LoadFromNumpy(strTestDataPath + "mh2.w_k_bias.npy");     // multi-head key bias
                layer.blobs[12].LoadFromNumpy(strTestDataPath + "mh2.w_v_weight.npy");   // multi-head value weight
                layer.blobs[13].LoadFromNumpy(strTestDataPath + "mh2.w_v_bias.npy");     // multi-head value bias
                layer.blobs[14].LoadFromNumpy(strTestDataPath + "mh2.w_o_weight.npy");   // multi-head output weight
                layer.blobs[15].LoadFromNumpy(strTestDataPath + "mh2.w_o_bias.npy");     // multi-head output bias

                layer.blobs[16].LoadFromNumpy(strTestDataPath + "ff.w_1_weight.npy");    // fc
                layer.blobs[17].LoadFromNumpy(strTestDataPath + "ff.w_1_bias.npy");      // fc
                layer.blobs[18].LoadFromNumpy(strTestDataPath + "ff.w_2_weight.npy");    // proj
                layer.blobs[19].LoadFromNumpy(strTestDataPath + "ff.w_2_bias.npy");      // proj

                layer.Forward(BottomVec, TopVec);

                // Now, check values
                m_blobYexp.LoadFromNumpy(strTestDataPath + "dec.12_output.npy");
                verify(TopVec[0], m_blobYexp, false, 3e-06);

                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (int i = 0; i < 100; i++)
                {
                    layer.Forward(BottomVec, TopVec);
                }

                sw.Stop();
                double dfTime = sw.Elapsed.TotalMilliseconds / 100;
                Trace.WriteLine("Decoder Forward time = " + dfTime.ToString("N6") + " ms.");
            }
            finally
            {
                layer.Dispose();
            }
        }

        public void TestBackward(uint nHeads, bool bEnableCudaImpl)
        {
            string strTestDataPath = loadTestData1();

            LayerParameter p = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
            p.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.DECODER;
            p.transformer_block_param.heads = nHeads;
            p.transformer_block_param.embed = 512;
            p.transformer_block_param.block_size = 200;
            p.transformer_block_param.attn_dropout = 0.0;
            p.transformer_block_param.resid_dropout = 0.0;
            p.transformer_block_param.enable_layernorm_cuda_impl = bEnableCudaImpl;
            Layer<T> layer = Layer<T>.Create(m_cuda, m_log, p, new CancelEvent());

            try
            {
                m_log.CHECK(layer.type == LayerParameter.LayerType.TRANSFORMER_BLOCK, "The layer type is incorrect!");

                m_blobX.LoadFromNumpy(strTestDataPath + "dec_in_x0.npy");
                m_blobX.Name = "dec_in_x0";
                m_blobEncOut.LoadFromNumpy(strTestDataPath + "enc_out_x1.npy");
                m_blobEncOut.Name = "enc_out_x1";

                m_blobInput.LoadFromNumpy(strTestDataPath + "src_input.npy");
                m_blobMaskEnc.ReshapeLike(m_blobInput);
                m_blobMaskEnc.Name = "e_mask";
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMaskEnc.mutable_gpu_data);

                m_blobInput.LoadFromNumpy(strTestDataPath + "trg_input.npy");
                m_blobMaskDec.ReshapeLike(m_blobInput);
                m_blobMaskDec.Name = "d_mask";
                m_cuda.sign(m_blobInput.count(), m_blobInput.gpu_data, m_blobMaskDec.mutable_gpu_data);
                createDecoderMask(m_blobMaskDec);

                m_blobMaskDec_exp.LoadFromNumpy(strTestDataPath + "mh1.1_mask.npy");
                verify(m_blobMaskDec, m_blobMaskDec_exp, false, 1e-09);

                m_blobMaskEnc_exp.LoadFromNumpy(strTestDataPath + "mh2.1_mask.npy");
                verify(m_blobMaskEnc, m_blobMaskEnc_exp, false, 1e-09);

                BottomVec.Clear();
                BottomVec.Add(m_blobX);
                BottomVec.Add(m_blobMaskDec);
                BottomVec.Add(m_blobEncOut);
                BottomVec.Add(m_blobMaskEnc);

                layer.Setup(BottomVec, TopVec);

                layer.blobs[0].LoadFromNumpy(strTestDataPath + "mh1.w_q_weight.npy");    // multi-head query weight
                layer.blobs[1].LoadFromNumpy(strTestDataPath + "mh1.w_q_bias.npy");      // multi-head query bias
                layer.blobs[2].LoadFromNumpy(strTestDataPath + "mh1.w_k_weight.npy");    // multi-head key weight
                layer.blobs[3].LoadFromNumpy(strTestDataPath + "mh1.w_k_bias.npy");      // multi-head key bias
                layer.blobs[4].LoadFromNumpy(strTestDataPath + "mh1.w_v_weight.npy");    // multi-head value weight
                layer.blobs[5].LoadFromNumpy(strTestDataPath + "mh1.w_v_bias.npy");      // multi-head value bias
                layer.blobs[6].LoadFromNumpy(strTestDataPath + "mh1.w_o_weight.npy");    // multi-head output weight
                layer.blobs[7].LoadFromNumpy(strTestDataPath + "mh1.w_o_bias.npy");      // multi-head output bias

                layer.blobs[8].LoadFromNumpy(strTestDataPath + "mh2.w_q_weight.npy");    // multi-head query weight
                layer.blobs[9].LoadFromNumpy(strTestDataPath + "mh2.w_q_bias.npy");      // multi-head query bias
                layer.blobs[10].LoadFromNumpy(strTestDataPath + "mh2.w_k_weight.npy");    // multi-head key weight
                layer.blobs[11].LoadFromNumpy(strTestDataPath + "mh2.w_k_bias.npy");      // multi-head key bias
                layer.blobs[12].LoadFromNumpy(strTestDataPath + "mh2.w_v_weight.npy");    // multi-head value weight
                layer.blobs[13].LoadFromNumpy(strTestDataPath + "mh2.w_v_bias.npy");      // multi-head value bias
                layer.blobs[14].LoadFromNumpy(strTestDataPath + "mh2.w_o_weight.npy");    // multi-head output weight
                layer.blobs[15].LoadFromNumpy(strTestDataPath + "mh2.w_o_bias.npy");      // multi-head output bias

                layer.blobs[16].LoadFromNumpy(strTestDataPath + "ff.w_1_weight.npy");    // fc
                layer.blobs[17].LoadFromNumpy(strTestDataPath + "ff.w_1_bias.npy");      // fc
                layer.blobs[18].LoadFromNumpy(strTestDataPath + "ff.w_2_weight.npy");   // proj
                layer.blobs[19].LoadFromNumpy(strTestDataPath + "ff.w_2_bias.npy");     // proj

                layer.Forward(BottomVec, TopVec);

                // Now, check values from forward
                m_blobYexp.LoadFromNumpy(strTestDataPath + "dec.12_output.npy");
                verify(TopVec[0], m_blobYexp, false, 3e-06);

                // Load the inbound gradients.
                TopVec[0].LoadFromNumpy(strTestDataPath + "grad_dec.12_output.npy", true);

                List<bool> rgProp = new List<bool>() { true };
                layer.Backward(TopVec, rgProp, BottomVec);

                // Now, check values form backward
                m_blobYexp.LoadFromNumpy(strTestDataPath + "grad_dec.1_x.npy", true);
                verify(BottomVec[0], m_blobYexp, true, 1e-08);

                Stopwatch sw = new Stopwatch();
                sw.Start();

                for (int i = 0; i < 100; i++)
                {
                    layer.Backward(TopVec, rgProp, BottomVec);
                }

                sw.Stop();
                double dfTime = sw.Elapsed.TotalMilliseconds / 100;
                Trace.WriteLine("Decoder Backward time = " + dfTime.ToString("N6") + " ms.");
            }
            finally
            {
                layer.Dispose();
            }
        }
        

        private Tuple<string, string, string, string> loadDataFiles1()
        {
            string strPath = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData) + "\\MyCaffe\\test_data\\data\\text\\encdec";
            string strFileName = "en_fr.zip";

            string strTestData = downloadTestData(strPath, strFileName);
            string strTestDataPath = Path.GetDirectoryName(strTestData);
            
            if (!File.Exists(strTestDataPath + "\\en_fr\\src\\train.txt"))
                ZipFile.ExtractToDirectory(strTestData, strPath);

            string strSrcText = strPath + "\\en_fr\\src\\train.txt";
            string strTrgText = strPath + "\\en_fr\\trg\\train.txt";
            string strSrcVocab = strPath + "\\en_fr\\sp\\src_sp.vocab";
            string strTrgVocab = strPath + "\\en_fr\\sp\\trg_sp.vocab";

            return new Tuple<string, string, string, string>(strSrcText, strTrgText, strSrcVocab, strTrgVocab);
        }

        private string buildModel(string strSrcFile, string strSrcVocabFile, string strTrgFile, string strTrgVocabFile, uint nBatch, uint nBlockSize, uint nEmbed, uint nEncVocabSize, uint nDecVocabSize)
        {
            NetParameter net = new NetParameter();
            net.name = "TranslatorNet";

            LayerParameter data = new LayerParameter(LayerParameter.LayerType.TOKENIZED_DATA_PAIRS);
            data.name = "tokdata1";
            data.tokenized_data_pairs_param.target = strTrgFile;
            data.tokenized_data_pairs_param.target_vocab_file = strTrgVocabFile;
            data.tokenized_data_pairs_param.source = strSrcFile;
            data.tokenized_data_pairs_param.source_vocab_file = strSrcVocabFile;
            data.tokenized_data_pairs_param.vocabulary_type = TokenizedDataParameter.VOCABULARY_TYPE.SENTENCEPIECE;
            data.tokenized_data_pairs_param.input_type = TokenizedDataParameter.INPUT_TYPE.TEXT_FILE;
            data.tokenized_data_pairs_param.batch_size = nBatch;
            data.tokenized_data_pairs_param.block_size = nBlockSize;
            data.top.Add("enc");
            data.top.Add("dec");
            data.top.Add("tgt");
            data.top.Add("emsk");
            data.top.Add("dmsk");
            net.layer.Add(data);

            LayerParameter emb1 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb1.name = "embed1";
            emb1.embed_param.input_dim = nEncVocabSize;
            emb1.embed_param.num_output = nEmbed;
            emb1.bottom.Add("enc");
            emb1.top.Add("emb1");
            net.layer.Add(emb1);

            LayerParameter pos1 = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODING);
            pos1.positional_encoder_param.block_size = nBlockSize;
            pos1.positional_encoder_param.embed = nEmbed;
            pos1.name = "posenc1";
            pos1.bottom.Add("emb1");
            pos1.top.Add("emb1");
            net.layer.Add(pos1);

            string strEncBtm = "emb1";
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
                enc.transformer_block_param.attn_dropout = 0.0;
                enc.transformer_block_param.resid_dropout = 0.0;
                enc.bottom.Add(strEncBtm);
                enc.bottom.Add("emsk");
                enc.top.Add(enc.name);
                net.layer.Add(enc);
                
                strEncBtm = enc.name;
            }

            LayerParameter emb2 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb2.name = "embed2";
            emb2.embed_param.input_dim = nDecVocabSize;
            emb2.embed_param.num_output = nEmbed;
            emb2.bottom.Add("dec");
            emb2.top.Add("emb2");
            net.layer.Add(emb2);

            LayerParameter pos2 = new LayerParameter(LayerParameter.LayerType.POSITIONAL_ENCODING);
            pos2.positional_encoder_param.block_size = nBlockSize;
            pos2.positional_encoder_param.embed = nEmbed;
            pos2.name = "posenc2";
            pos2.bottom.Add("emb2");
            pos2.top.Add("emb2");
            net.layer.Add(pos2);

            string strDecBtm = "emb2";
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
                dec.transformer_block_param.attn_dropout = 0.0;
                dec.transformer_block_param.resid_dropout = 0.0;
                dec.bottom.Add(strDecBtm);
                dec.bottom.Add("dmsk");
                dec.bottom.Add(strEncBtm);
                dec.bottom.Add("emsk");
                dec.top.Add(dec.name);
                net.layer.Add(dec);

                strDecBtm = dec.name;
            }

            LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
            ip1.name = "ip1";
            ip1.inner_product_param.axis = 2;
            ip1.inner_product_param.num_output = nDecVocabSize;
            ip1.bottom.Add(strDecBtm);
            ip1.top.Add("logits");
            net.layer.Add(ip1);

            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            softmax.name = "softmax";
            softmax.softmax_param.axis = 2;
            softmax.bottom.Add("logits");
            softmax.top.Add("prob");
            softmax.include.Add(new NetStateRule(Phase.RUN));
            net.layer.Add(softmax);

            LayerParameter loss = new LayerParameter(LayerParameter.LayerType.SOFTMAXWITH_LOSS);
            loss.name = "loss";
            loss.softmax_param.axis = 2;
            loss.bottom.Add("logits");
            loss.bottom.Add("tgt");
            loss.top.Add("loss");
            loss.include.Add(new NetStateRule(Phase.TRAIN));
            net.layer.Add(loss);

            LayerParameter accuracy = new LayerParameter(LayerParameter.LayerType.ACCURACY);
            accuracy.name = "accuracy";
            accuracy.accuracy_param.axis = 2;
            accuracy.bottom.Add("logits");
            accuracy.bottom.Add("tgt");
            accuracy.top.Add("accuracy");
            accuracy.include.Add(new NetStateRule(Phase.TEST));
            net.layer.Add(accuracy);

            return net.ToProto("root").ToString();
        }

        private string buildSolver()
        {
            SolverParameter solver = new SolverParameter();
            solver.base_lr = 1e-4;
            solver.type = SolverParameter.SolverType.ADAM;
            solver.clip_gradients = 1;

            return solver.ToProto("root").ToString();
        }
        
        public void TestTraining()
        {
            Tuple<string, string, string, string> dataFiles = loadDataFiles1();
            string strSrcFile = dataFiles.Item1;
            string strTrgFile = dataFiles.Item2;
            string strSrcVocab = dataFiles.Item3;
            string strTrgVocab = dataFiles.Item4;

            string strModel = buildModel(strSrcFile, strSrcVocab, strTrgFile, strTrgVocab, 20, 200, 512, 14878, 14638);
            string strSolver = buildSolver();

            SettingsCaffe s = new SettingsCaffe
            {
                GpuIds = "1"
            };
            CancelEvent evtCancel = new CancelEvent();
            MyCaffeControl<float> mycaffe = new MyCaffeControl<float>(s, m_log, evtCancel);
            mycaffe.OnTrainingIteration += Mycaffe_OnTrainingIteration;

            Blob<float> blobWork = null;

            try
            {
                mycaffe.LoadLite(Phase.TRAIN, strSolver, strModel, null, false, false);

                Net<float> net = mycaffe.GetInternalNet(Phase.TRAIN);
                net.Forward();

                blobWork = mycaffe.CreateBlob("work");
                for (int i=0; i<net.parameters.Count; i++)
                {
                    Blob<float> blob = net.parameters[i];
                    blobWork.ReshapeLike(blob);

                    Tuple<double, double, double, double> data = blob.minmax_data(blobWork, true);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " min: " + data.Item1.ToString() + Environment.NewLine);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " max: " + data.Item2.ToString() + Environment.NewLine);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " nan: " + data.Item3.ToString() + Environment.NewLine);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " inf: " + data.Item4.ToString() + Environment.NewLine);
                    if (data.Item3 > 0 || data.Item4 > 0)
                        Trace.WriteLine("Nan or Inf detected!");
                    Trace.WriteLine("---------------");                    
                }

                for (int i = 0; i < net.blobs.Count; i++)
                {
                    Blob<float> blob = net.blobs[i];
                    blobWork.ReshapeLike(blob);

                    Tuple<double, double, double, double> data = blob.minmax_data(blobWork, true);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " min: " + data.Item1.ToString() + Environment.NewLine);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " max: " + data.Item2.ToString() + Environment.NewLine);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " nan: " + data.Item3.ToString() + Environment.NewLine);
                    Trace.WriteLine(i.ToString() + ". " + blob.Name + " inf: " + data.Item4.ToString() + Environment.NewLine);
                    if (data.Item3 > 0 || data.Item4 > 0)
                        Trace.WriteLine("Nan or Inf detected!");
                    Trace.WriteLine("---------------");
                }
            }
            finally
            {
                if (blobWork != null)
                    blobWork.Dispose();

                mycaffe.Dispose();
            }
        }

        private void Mycaffe_OnTrainingIteration(object sender, TrainingIterationArgs<float> e)
        {
            Trace.WriteLine(e.Iteration.ToString() + " - " + e.Loss.ToString());
        }
    }
}
