using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.gpt;
using MyCaffe.param.ssd;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.model
{
    /// <summary>
    /// The Config structure defines the configuration of the model, which is also the header in Llama based models.
    /// </summary>
    struct Config
    {
        /// <summary>
        /// Specifies the embed dimension of the model, for Llama7B this is 4096.
        /// </summary>
        public int dim;
        /// <summary>
        /// Specifies the hidden dimension of the model, for Llama7B this is 11008.
        /// </summary>
        public int hidden_dim;
        /// <summary>
        /// Specifies the number of layers in the model, for Llama7B this is 32.
        /// </summary>
        public int n_layers;
        /// <summary>
        /// Specifies the number of heads in the model, for Llama7B this is 32.
        /// </summary>
        public int n_heads;
        /// <summary>
        /// Specifies the number of key-value heads in the model, for Llama7B this is 32.
        /// </summary>
        public int n_kv_heads;
        /// <summary>
        /// Specifies the vocabulary size of the model, for Llama7B this is 32000.
        /// </summary>
        public int vocab_size;
        /// <summary>
        /// Specifies the sequence length of the model, for Llama7B this is 2048.
        /// </summary>
        public int seq_len;
        /// <summary>
        /// Specifies whether or not the classifier is shared, for Llama7B this is 0.
        /// </summary>
        public byte shared_classifier;
    }

    /// <summary>
    /// The LlamaModelBuilder creates a llama based model that supports the Llama models by Meta.
    /// </summary>
    public class LlamaModelBuilder<T> : ModelBuilder<T>
    {
        int m_nGpuID = 0;
        List<int> m_rgGpuID = new List<int>();
        uint m_nBatchSize;
        uint m_nSeqLen = 2048;
        uint m_nVocabSize = 32000;
        double m_dfDropout = 0.1;
        string m_strModel;
        double m_dfBaseLr = 0.01;
        Config m_config;

        /// <summary>
        /// Defines the type of model to create.
        /// </summary>
        public enum MODEL
        {
            /// <summary>
            /// Specifies to create a Llama7B model.
            /// </summary>
            LLAMA_7B
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strBaseDirectory">Specifies the base directory that contains the data and models.</param>
        /// <param name="strModel">Specifies the model to create.</param>
        /// <param name="nBatchSize">specifies the batch size.</param>
        /// <param name="nSeqLen">Specifies the sequence length.</param>
        /// <param name="nVocabSize">Specifies the vocabulary size.</param> 
        /// <param name="dfDropout">Specifies the dropout ratio.</param>
        /// <param name="dfLearningRate">Specifies the base learning rate.</param>
        /// <param name="rgGpuId">Optionally, specifies a set of GPU ID's to use (when null, GPU=0 is used).</param>
        /// <param name="net">Specifies the 'base' net parameter that is to be altered.</param>
        public LlamaModelBuilder(string strBaseDirectory, string strModel, uint nBatchSize = 1, uint nSeqLen = 512, uint nVocabSize = 32000, double dfDropout = 0.0, double dfLearningRate = 0.01, List<int> rgGpuId = null, NetParameter net = null) 
            : base(strBaseDirectory, net)
        {
            if (rgGpuId == null)
                m_rgGpuID.Add(0);
            else
                m_rgGpuID = new List<int>(rgGpuId);

            m_strModel = strModel;
            m_nBatchSize = nBatchSize;
            m_nSeqLen = nSeqLen;
            m_nVocabSize = nVocabSize;
            m_dfDropout = dfDropout;
            m_dfBaseLr = dfLearningRate;
        }

        /// <summary>
        /// Create the base solver to use.
        /// </summary>
        /// <returns>
        /// The solver parameter created is returned.
        /// </returns>
        public override SolverParameter CreateSolver()
        {
            m_solver = new SolverParameter();
            m_solver.type = SolverParameter.SolverType.SGD;
            m_solver.base_lr = m_dfBaseLr;
            m_solver.weight_decay = 0.0005;
            m_solver.LearningRatePolicy = SolverParameter.LearningRatePolicyType.MULTISTEP;
            m_solver.stepvalue = new List<int>() { 80000, 100000, 120000 };
            m_solver.gamma = 0.1;
            m_solver.momentum = 0.9;
            m_solver.iter_size = 1;
            m_solver.max_iter = 120000;
            m_solver.snapshot = 80000;
            m_solver.display = 100;
            m_solver.average_loss = 100;
            m_solver.device_id = m_nGpuID;
            m_solver.debug_info = false;
            m_solver.snapshot_after_train = true;
            m_solver.clip_gradients = 1;

            // Test parameters.
            m_solver.test_iter.Clear();
            m_solver.test_interval = 10000;
            m_solver.test_initialization = false;
            m_solver.eval_type = SolverParameter.EvaluationType.CLASSIFICATION;

            return m_solver;
        }

        /// <summary>
        /// Create the training model.
        /// </summary>
        /// <param name="bDeploy">Optionally, specifies to create a deployment model (default = false).</param>
        public override NetParameter CreateModel(bool bDeploy = false)
        {
            Phase phase = (bDeploy) ? Phase.RUN : Phase.TRAIN;

            m_net = createNet(m_strModel);

            uint nLayers = 0;
            uint nDim = 0;
            uint nHiddenDim = 0;
            uint nHeads = 0;

            switch (m_strModel)
            {
                case "Llama7B":
                    nLayers = 32;
                    nHeads = 32;
                    nDim = 4096;
                    nHiddenDim = 11008;
                    break;
            }

            string strModel = buildModel(m_net, m_nBatchSize, m_nSeqLen, m_nVocabSize, nDim, nHiddenDim, nHeads, nLayers, m_dfDropout, phase);

            return m_net;
        }

        /// <summary>
        /// Create the testing SSD model for the pascal dataset.
        /// </summary>
        public override NetParameter CreateDeployModel()
        {
            return CreateModel(true);
        }

        /// <summary>
        /// Load the model weights from the specified model file, using the specified format.
        /// </summary>
        /// <param name="col">Specifies the learnable weights to load.</param>
        /// <param name="strModelFile">Specifies the model file name.</param>
        /// <param name="strFmt">Specifies the model format.</param>
        /// <returns>A byte array of the weights in MyCaffe format is returned and can be loaded using the MyCaffe.UpdateWeights method.</returns>
        /// <remarks>
        /// The following formats are supported:
        ///     KPTH2 - supports the Karpathy format exported using 'export.py'.
        ///     @see [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy, 2023, GitHub.
        /// </remarks>
        public override byte[] LoadWeights(BlobCollection<T> col, string strModelFile, string strFmt = "KPTH1")
        {
            if (strFmt != "KPTH1")
                throw new Exception("The format '" + strFmt + "' is not supported!");

            FileInfo fi = new FileInfo(strModelFile);
            long lSize = fi.Length;
            long lOffset = 0;
            long hBlobLoader = 0;

            try
            {
                using (var mmf = MemoryMappedFile.CreateFromFile(strModelFile, FileMode.Open, "MyCaffeSharedMemory"))
                {
                    using (var accessor = mmf.CreateViewAccessor(lOffset, lSize, MemoryMappedFileAccess.Read))
                    {
                        byte[] rgHeader = new byte[256];
                        lOffset += accessor.ReadArray(0, rgHeader, 0, 256);

                        using (MemoryStream ms = new MemoryStream(rgHeader))
                        using (BinaryReader br = new BinaryReader(ms))
                        {
                            uint uiMagic = br.ReadUInt32();
                            if (uiMagic != 0x616b3432)
                                throw new Exception("The model file '" + strModelFile + "' does not appear to be in the correct format!");

                            int nVersion = br.ReadInt32();
                            if (nVersion != 1)
                                throw new Exception("The model file '" + strModelFile + "' does not appear to be in the correct format!");

                            m_config.dim = br.ReadInt32();
                            m_config.hidden_dim = br.ReadInt32();
                            m_config.n_layers = br.ReadInt32();
                            m_config.n_heads = br.ReadInt32();
                            m_config.n_kv_heads = br.ReadInt32();
                            m_config.vocab_size = br.ReadInt32();
                            m_config.seq_len = br.ReadInt32();
                            m_config.shared_classifier = br.ReadByte();
                        }
                    }
                }

                long nHeadSize = m_config.dim / m_config.n_heads;
                long lCount;
                int nIdx = 0;

                hBlobLoader = col[0].CreateBlobLoader(strModelFile, lOffset);

                // read in the rms_att_weights                    
                lCount = m_config.n_layers * m_config.dim;
                col[0].AddToBlobLoaderOffset(hBlobLoader, lCount * sizeof(float));
                lOffset += lCount * sizeof(float);
                // read in the rms_ffn_weights
                lCount = m_config.n_layers * m_config.dim;
                col[0].AddToBlobLoaderOffset(hBlobLoader, lCount * sizeof(float));
                lOffset += lCount * sizeof(float);
                // read in the rms_final_weights
                lCount = m_config.dim;
                col[0].AddToBlobLoaderOffset(hBlobLoader, lCount * sizeof(float));
                lOffset += lCount * sizeof(float);

                // read in the token embedding table
                lCount = m_config.vocab_size * m_config.dim;
                col[0].LoadFromBlobLoader(hBlobLoader, lCount, 0);
                long lWeightStartOffset = lOffset + lCount * sizeof(float);
                nIdx++;

                // read in the wq weights
                long lWqCount = m_config.n_layers * m_config.dim * (m_config.n_heads * nHeadSize);
                // read in the wk weights
                long lWkCount = m_config.n_layers * m_config.dim * (m_config.n_kv_heads * nHeadSize);
                // read in the wv weights
                long lWvCount = m_config.n_layers * m_config.dim * (m_config.n_kv_heads * nHeadSize);
                // read in the wo weights
                long lWoCount = m_config.n_layers * (m_config.n_heads * nHeadSize) * m_config.dim;
                // read in the w1 weights
                long lW1Count = m_config.n_layers * m_config.dim * m_config.hidden_dim;
                // read in the w2 weights
                long lW2Count = m_config.n_layers * m_config.hidden_dim * m_config.dim;
                // read in the w3 weights
                long lW3Count = m_config.n_layers * m_config.dim * m_config.hidden_dim;
                // read in the wcls weights
                long lWclsCount = m_config.dim * m_config.vocab_size;
                // Set the base size
                long lBaseDataSize = (typeof(T) == typeof(float)) ? sizeof(float) : sizeof(double);
                float fFirst;

                for (int i = 0; i < m_config.n_layers; i++)
                {
                    col[nIdx].ResetBlobLoaderOffset(hBlobLoader, lWeightStartOffset);

                    long lWqCount1 = (lWqCount / m_config.n_layers);
                    long lWqLocalOffset = i * lWqCount1;
                    col[nIdx].LoadFromBlobLoader(hBlobLoader, lWqCount1, lWqLocalOffset);
                    col[nIdx].AddToBlobLoaderOffset(hBlobLoader, lWqCount);
                    fFirst = col[nIdx].GetDataAsFloat(0);
                    nIdx++;

                    long lWkCount1 = (lWkCount / m_config.n_layers);
                    long lWkLocalOffset = i * lWkCount1;
                    col[nIdx].LoadFromBlobLoader(hBlobLoader, lWkCount1, lWkLocalOffset);
                    col[nIdx].AddToBlobLoaderOffset(hBlobLoader, lWkCount);
                    fFirst = col[nIdx].GetDataAsFloat(0);
                    nIdx++;

                    long lWvCount1 = (lWvCount / m_config.n_layers);
                    long lWvLocalOffset = i * lWvCount1;
                    col[nIdx].LoadFromBlobLoader(hBlobLoader, lWvCount1, lWvLocalOffset);
                    col[nIdx].AddToBlobLoaderOffset(hBlobLoader, lWvCount);
                    fFirst = col[nIdx].GetDataAsFloat(0);
                    nIdx++;

                    long lWoCount1 = (lWoCount / m_config.n_layers);
                    long lWoLocalOffset = i * lWoCount1;
                    col[nIdx].LoadFromBlobLoader(hBlobLoader, lWoCount1, lWoLocalOffset);
                    col[nIdx].AddToBlobLoaderOffset(hBlobLoader, lWoCount);
                    fFirst = col[nIdx].GetDataAsFloat(0);
                    nIdx++;

                    long lW1Count1 = (lW1Count / m_config.n_layers);
                    long lW1LocalOffset = i * lW1Count1;
                    col[nIdx].LoadFromBlobLoader(hBlobLoader, lW1Count1, lW1LocalOffset);
                    col[nIdx].AddToBlobLoaderOffset(hBlobLoader, lW1Count);
                    fFirst = col[nIdx].GetDataAsFloat(0);
                    nIdx++;

                    long lW2Count1 = (lW2Count / m_config.n_layers);
                    long lW2LocalOffset = i * lW2Count1;
                    col[nIdx].LoadFromBlobLoader(hBlobLoader, lW2Count1, lW2LocalOffset);
                    col[nIdx].AddToBlobLoaderOffset(hBlobLoader, lW2Count);
                    fFirst = col[nIdx].GetDataAsFloat(0);
                    nIdx++;

                    long lW3Count1 = (lW3Count / m_config.n_layers);
                    long lW3LocalOffset = i * lW3Count1;
                    col[nIdx].LoadFromBlobLoader(hBlobLoader, lW3Count1, lW3LocalOffset);
                    col[nIdx].AddToBlobLoaderOffset(hBlobLoader, lW3Count);
                    fFirst = col[nIdx].GetDataAsFloat(0);
                    nIdx++;

                    col[0].Log.WriteLine("Loading weights for layer " + i.ToString() + " of " + m_config.n_layers.ToString() + " ...", true);
                    col[0].Log.Progress = ((double)i / (double)m_config.n_layers);
                }

                if (m_config.shared_classifier == 0)
                {
                    col[nIdx].LoadFromBlobLoader(hBlobLoader, lWclsCount, 0);
                    fFirst = col[nIdx].GetDataAsFloat(0);
                }
                else
                {
                    col[nIdx].CopyFrom(col[0]);    
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                if (hBlobLoader != 0)
                    col[0].FreeBlobLoader(hBlobLoader);
            }

            return null;
        }

        private float[] readWeights(MemoryMappedViewAccessor accessor, ref long lOffset, long nCount)
        {
            float[] rgWeights = new float[nCount];

            for (int i = 0; i < nCount; i += 1000000)
            {
                long lCount = Math.Min(nCount - i, 1000000);
                byte[] rgData = new byte[lCount * sizeof(float)];
                lOffset += accessor.ReadArray(lOffset, rgData, 0, rgData.Length);

                Buffer.BlockCopy(rgData, 0, rgWeights, i * sizeof(float), rgData.Length);
            }

            return rgWeights;
        }


        private string buildModel(NetParameter net, uint nBatch, uint nBlockSize, uint nEncVocabSize, uint nEmbed, uint nHiddenDim, uint nHeads, uint nLayers, double dfDropout, Phase phase)
        {
            net.enable_lora = true;
            net.enable_lora_only_load = true;

            LayerParameter tok = new LayerParameter(LayerParameter.LayerType.TOKENIZED_DATA);
            tok.tokenized_data_param.input_type = TokenizedDataParameter.INPUT_TYPE.TEXT_FILE;
            tok.tokenized_data_param.vocabulary_type = TokenizedDataParameter.VOCABULARY_TYPE.CHARACTER;
            tok.tokenized_data_param.source = "$ProgramData$\\MyCaffe\\test_data\\data\\text\\input.txt";
            tok.tokenized_data_param.batch_size = nBatch;
            tok.tokenized_data_param.block_size = nBlockSize;
            tok.freeze_learning = true;
            tok.top.Add("tokdata");
            tok.top.Add("pos");
            if (phase != Phase.RUN)
                tok.top.Add("tgt");
            net.layer.Add(tok);

            LayerParameter silence = new LayerParameter(LayerParameter.LayerType.SILENCE);
            silence.bottom.Add("pos");
            silence.freeze_learning = true;
            net.layer.Add(silence);

            LayerParameter emb1 = new LayerParameter(LayerParameter.LayerType.EMBED);
            emb1.name = "wte";
            emb1.embed_param.bias_term = false;
            emb1.embed_param.input_dim = nEncVocabSize;
            emb1.embed_param.num_output = nEmbed;
            emb1.embed_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
            emb1.parameters.Add(new ParamSpec(1.0, 0.0));
            emb1.bottom.Add("tokdata");
            emb1.top.Add("tok_emb");
            emb1.freeze_learning = true;
            net.layer.Add(emb1);

            string strEncBtm = "tok_emb";
            for (int i = 0; i < nLayers; i++)
            {
                LayerParameter enc = new LayerParameter(LayerParameter.LayerType.TRANSFORMER_BLOCK);
                enc.name = "tfb" + (i + 1).ToString();
                enc.transformer_block_param.block_type = TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION2;
                enc.transformer_block_param.heads = nHeads;
                enc.transformer_block_param.embed = nEmbed;
                enc.transformer_block_param.hidden_dim = nHiddenDim;
                enc.transformer_block_param.block_size = nBlockSize;
                enc.transformer_block_param.layers = (uint)nLayers;
                enc.transformer_block_param.activation = TransformerBlockParameter.ACTIVATION.SILU;
                enc.transformer_block_param.normalization_type = TransformerBlockParameter.NORMALIZATION.LAYER_NORM;
                enc.transformer_block_param.attn_dropout = dfDropout;
                enc.transformer_block_param.resid_dropout = dfDropout;
                enc.transformer_block_param.enable_layernorm_cuda_impl = false;
                enc.transformer_block_param.enable_llama_style_head = true;
                enc.transformer_block_param.enable_rotary_positional_embedding = true;
                enc.transformer_block_param.bias_term = false;
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.parameters.Add(new ParamSpec(1, 1));
                enc.bottom.Add(strEncBtm);
                enc.top.Add(enc.name);
                enc.freeze_learning = true;
                net.layer.Add(enc);

                strEncBtm = enc.name;
            }

            LayerParameter ln1 = new LayerParameter(LayerParameter.LayerType.LAYERNORM);
            ln1.name = "ln1";
            ln1.layer_norm_param.enable_cuda_impl = false;
            ln1.bottom.Add(strEncBtm);
            ln1.top.Add("ln1");
            ln1.freeze_learning = true;
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
            ip1.freeze_learning = true;
            net.layer.Add(ip1);

            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            softmax.name = "softmax";
            softmax.softmax_param.axis = 2;
            softmax.softmax_param.algorithm = SOFTMAX_ALGORITHM.ACCURATE;
            softmax.softmax_param.algorithm_train = SOFTMAX_ALGORITHM.LOG;
            softmax.bottom.Add("logits");
            softmax.top.Add("prob");
            softmax.freeze_learning = true;
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
    }
}
