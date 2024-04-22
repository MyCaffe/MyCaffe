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
        int m_nIterSize = 1;
        Config m_config;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strBaseDirectory">Specifies the base directory that contains the data and models.</param>
        /// <param name="strModel">Specifies the model to create.</param>
        /// <param name="nIterSize">Specifies the number of iterations to accumulate gradients over.</param>
        /// <param name="nBatchSize">specifies the batch size.</param>
        /// <param name="nSeqLen">Specifies the sequence length.</param>
        /// <param name="nVocabSize">Specifies the vocabulary size.</param> 
        /// <param name="dfDropout">Specifies the dropout ratio.</param>
        /// <param name="dfLearningRate">Specifies the base learning rate.</param>
        /// <param name="rgGpuId">Optionally, specifies a set of GPU ID's to use (when null, GPU=0 is used).</param>
        /// <param name="net">Specifies the 'base' net parameter that is to be altered.</param>
        public LlamaModelBuilder(string strBaseDirectory, string strModel, int nIterSize, uint nBatchSize = 1, uint nSeqLen = 512, uint nVocabSize = 32000, double dfDropout = 0.1, double dfLearningRate = 5e-04, List<int> rgGpuId = null, NetParameter net = null) 
            : base(strBaseDirectory, net)
        {
            if (rgGpuId == null)
                m_rgGpuID.Add(0);
            else
                m_rgGpuID = new List<int>(rgGpuId);

            m_strModel = strModel;
            m_nIterSize = nIterSize;
            m_nBatchSize = nBatchSize;
            m_nSeqLen = nSeqLen;
            m_nVocabSize = nVocabSize;
            m_dfDropout = dfDropout;
            m_dfBaseLr = dfLearningRate;
        }

        /// <summary>
        /// Return the model type of the model file.
        /// </summary>
        /// <param name="strModelFile">Specifies the model who's type is to be returned.</param>
        /// <returns>The model type is returned.</returns>
        public override string GetModelType(string strModelFile)
        {
            try
            {
                FileInfo fi = new FileInfo(strModelFile);
                long lSize = 256;
                long lOffset = 0;
                long lHeaderSize = 256;

                using (var mmf = MemoryMappedFile.CreateFromFile(strModelFile, FileMode.Open, "MyCaffeSharedMemory"))
                {
                    using (var accessor = mmf.CreateViewAccessor(lOffset, lSize, MemoryMappedFileAccess.Read))
                    {
                        byte[] rgHeader = new byte[lHeaderSize];
                        accessor.ReadArray(0, rgHeader, 0, (int)lHeaderSize);

                        using (MemoryStream ms = new MemoryStream(rgHeader))
                        using (BinaryReader br = new BinaryReader(ms))
                        {
                            uint uiMagic = br.ReadUInt32();
                            if (uiMagic != 0x616b3432)
                                return "KPTH0";

                            int nVersion = br.ReadInt32();
                            return "KPTH" + nVersion.ToString();
                        }
                    }
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
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
            m_solver.type = SolverParameter.SolverType.ADAMW;
            m_solver.base_lr = m_dfBaseLr;
            m_solver.weight_decay = 0;
            m_solver.LearningRatePolicy = SolverParameter.LearningRatePolicyType.MULTISTEP;
            m_solver.stepvalue = new List<int>() { 80000, 100000, 120000 };
            m_solver.gamma = 0.1;
            m_solver.momentum = 0.9;
            m_solver.iter_size = m_nIterSize;
            m_solver.max_iter = 120000;
            m_solver.snapshot = 80000;
            m_solver.display = 100;
            m_solver.average_loss = 100;
            m_solver.device_id = m_nGpuID;
            m_solver.debug_info = false;
            m_solver.snapshot_after_train = true;
            m_solver.clip_gradients = -1;
            m_solver.use_test_net_for_running = true;

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
        /// <param name="prop">Specifies optional properties.</param>
        /// <param name="phase">Optionally, specifies the phase to use when creating the model (default = TRAIN).</param>
        /// <param name="bEnableLoRA">Optionally, specifies whether or not to enable LoRA (default = false).</param>
        /// <param name="nLayerCountOverride">Optionally, specifies the number of transformer blocks (default = -1, value less than or equal to 0 ignores setting).</param>
        public override NetParameter CreateModel(PropertySet prop, Phase phase = Phase.TRAIN, bool bEnableLoRA = false, int nLayerCountOverride = -1)
        {
            m_net = createNet(m_strModel);

            uint nLayers = 0;
            uint nDim = 0;
            uint nHiddenDim = 0;
            uint nHeads = 0;
            bool bPreTokenizer = false;
            bool bEnableKeyValueCache = true;

            switch (m_strModel)
            {
                case "Llama7B":
                    nLayers = 32;
                    nHeads = 32;
                    nDim = 4096;
                    nHiddenDim = 11008;
                    break;

                case "Stories15M":
                    nLayers = 6;
                    nHeads = 6;
                    nDim = 288;
                    nHiddenDim = 768;
                    break;

                case "Stories15M_Instruct":
                    nLayers = 6;
                    nHeads = 6;
                    nDim = 288;
                    nHiddenDim = 768;
                    bPreTokenizer = true;
                    bEnableKeyValueCache = false;
                    break;
            }

            if (nLayerCountOverride > 0)
                nLayers = (uint)nLayerCountOverride;

            TokenizedDataParameter.VOCABULARY_TYPE vocabType = TokenizedDataParameter.VOCABULARY_TYPE.CHARACTER;

            if (prop != null)
            {
                int nVocabType = prop.GetPropertyAsInt("VocabularyType", -1);
                if (nVocabType != -1)
                    vocabType = (TokenizedDataParameter.VOCABULARY_TYPE)nVocabType;
            }

            string strModel = buildModel(vocabType, m_net, m_nBatchSize, m_nSeqLen, m_nVocabSize, nDim, nHiddenDim, nHeads, nLayers, m_dfDropout, phase, bEnableLoRA, bPreTokenizer, bEnableKeyValueCache);

            return m_net;
        }

        /// <summary>
        /// Create the testing SSD model for the pascal dataset.
        /// </summary>
        public override NetParameter CreateDeployModel()
        {
            return CreateModel(null, Phase.RUN);
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
        ///     KPTH0 - supports the Karpathy original format exported using 'export.py'.
        ///     KPTH1 - supports the Karpathy version 1 format exported using 'export.py'.
        ///     @see [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy, 2023, GitHub.
        /// </remarks>
        public override byte[] LoadWeights(BlobCollection<T> col, string strModelFile, string strFmt = "KPTH1")
        {
            switch (strFmt)
            {
                case "KPTH0":
                    return loadWeights_kpth0(col, strModelFile);

                case "KPTH1":
                    return loadWeights_kpth1(col, strModelFile);

                default:
                    throw new Exception("The format '" + strFmt + "' is not supported!");
            }
        }

        private byte[] loadWeights_kpth0(BlobCollection<T> col, string strModelFile)
        {
            FileInfo fi = new FileInfo(strModelFile);
            long lSize = fi.Length;
            long lOffset = 0;
            long hBlobLoader = 0;
            long lHeaderSize = 28;

            try
            {
                using (var mmf = MemoryMappedFile.CreateFromFile(strModelFile, FileMode.Open, "MyCaffeSharedMemory"))
                {
                    using (var accessor = mmf.CreateViewAccessor(lOffset, lSize, MemoryMappedFileAccess.Read))
                    {
                        byte[] rgHeader = new byte[lHeaderSize];
                        accessor.ReadArray(0, rgHeader, 0, (int)lHeaderSize);

                        using (MemoryStream ms = new MemoryStream(rgHeader))
                        using (BinaryReader br = new BinaryReader(ms))
                        {
                            m_config.dim = br.ReadInt32();
                            m_config.hidden_dim = br.ReadInt32();
                            m_config.n_layers = br.ReadInt32();
                            m_config.n_heads = br.ReadInt32();
                            m_config.n_kv_heads = br.ReadInt32();
                            m_config.vocab_size = br.ReadInt32();
                            m_config.seq_len = br.ReadInt32();
                            m_config.shared_classifier = m_config.vocab_size > 0 ? (byte)1 : (byte)0;
                        }
                    }
                }

                long nHeadSize = m_config.dim / m_config.n_heads;
                int nIdx = 0;
                float fFirst;

                hBlobLoader = col[0].CreateBlobLoader(strModelFile, lOffset);
                col[0].ResetBlobLoaderOffset(hBlobLoader, lHeaderSize);

                // read in the token embedding table
                long lTokenEmbeddingCount = m_config.vocab_size * m_config.dim;
                lOffset += lTokenEmbeddingCount;

                col[0].LoadFromBlobLoader(hBlobLoader, lTokenEmbeddingCount, 0);
                col[0].AddToBlobLoaderOffset(hBlobLoader, lTokenEmbeddingCount);
                fFirst = col[0].GetDataAsFloat(0);

                // read in the rms_att_weights                    
                long lRmsAttCount = m_config.n_layers * m_config.dim;
                // read in the wq weights
                long lWqCount = m_config.n_layers * m_config.dim * (m_config.n_heads * nHeadSize);
                // read in the wk weights
                long lWkCount = m_config.n_layers * m_config.dim * (m_config.n_kv_heads * nHeadSize);
                // read in the wv weights
                long lWvCount = m_config.n_layers * m_config.dim * (m_config.n_kv_heads * nHeadSize);
                // read in the wo weights
                long lWoCount = m_config.n_layers * (m_config.n_heads * nHeadSize) * m_config.dim;
                // read in the rms_ffn_weights
                long lRmsFfnCount = m_config.n_layers * m_config.dim;
                // read in the w1 weights
                long lW1Count = m_config.n_layers * m_config.dim * m_config.hidden_dim;
                // read in the w2 weights
                long lW2Count = m_config.n_layers * m_config.hidden_dim * m_config.dim;
                // read in the w3 weights
                long lW3Count = m_config.n_layers * m_config.dim * m_config.hidden_dim;

                // Set the weight offset to Header + token_embedding
                long lWeightStartOffset = lHeaderSize + lOffset * sizeof(float);

                for (int i = 0; i < m_config.n_layers; i++)
                {
                    int nIdxRms1 = 1 + i * 9;
                    int nIdxRms2 = 1 + i * 9 + 5;
                    int nIdxWq = 2 + i * 9;
                    int nIdxWk = nIdxWq + 1;
                    int nIdxWv = nIdxWq + 2;
                    int nIdxWo = nIdxWq + 3;
                    int nIdxW1 = nIdxWq + 5;
                    int nIdxW2 = nIdxWq + 6;
                    int nIdxW3 = nIdxWq + 7;

                    col[nIdxRms1].ResetBlobLoaderOffset(hBlobLoader, lWeightStartOffset);

                    long lRmsAttCount1 = (lRmsAttCount / m_config.n_layers);
                    long lRmsAttLocalOffset = i * lRmsAttCount1;
                    col[nIdxRms1].LoadFromBlobLoader(hBlobLoader, lRmsAttCount1, lRmsAttLocalOffset);
                    col[nIdxRms1].AddToBlobLoaderOffset(hBlobLoader, lRmsAttCount);
                    fFirst = col[nIdxRms1].GetDataAsFloat(0);

                    long lWqCount1 = (lWqCount / m_config.n_layers);
                    long lWqLocalOffset = i * lWqCount1;
                    col[nIdxWq].LoadFromBlobLoader(hBlobLoader, lWqCount1, lWqLocalOffset);
                    col[nIdxWq].AddToBlobLoaderOffset(hBlobLoader, lWqCount);
                    fFirst = col[nIdxWq].GetDataAsFloat(0);
                    nIdx++;

                    long lWkCount1 = (lWkCount / m_config.n_layers);
                    long lWkLocalOffset = i * lWkCount1;
                    col[nIdxWk].LoadFromBlobLoader(hBlobLoader, lWkCount1, lWkLocalOffset);
                    col[nIdxWk].AddToBlobLoaderOffset(hBlobLoader, lWkCount);
                    fFirst = col[nIdxWk].GetDataAsFloat(0);
                    nIdx++;

                    long lWvCount1 = (lWvCount / m_config.n_layers);
                    long lWvLocalOffset = i * lWvCount1;
                    col[nIdxWv].LoadFromBlobLoader(hBlobLoader, lWvCount1, lWvLocalOffset);
                    col[nIdxWv].AddToBlobLoaderOffset(hBlobLoader, lWvCount);
                    fFirst = col[nIdxWv].GetDataAsFloat(0);
                    nIdx++;

                    long lWoCount1 = (lWoCount / m_config.n_layers);
                    long lWoLocalOffset = i * lWoCount1;
                    col[nIdxWo].LoadFromBlobLoader(hBlobLoader, lWoCount1, lWoLocalOffset);
                    col[nIdxWo].AddToBlobLoaderOffset(hBlobLoader, lWoCount);
                    fFirst = col[nIdxWo].GetDataAsFloat(0);
                    nIdx++;

                    long lRmsFfnCount1 = (lRmsFfnCount / m_config.n_layers);
                    long lRmsFfnLocalOffset = i * lRmsFfnCount1;
                    col[nIdxRms2].LoadFromBlobLoader(hBlobLoader, lRmsFfnCount1, lRmsFfnLocalOffset);
                    col[nIdxRms2].AddToBlobLoaderOffset(hBlobLoader, lRmsFfnCount);
                    fFirst = col[nIdxRms2].GetDataAsFloat(0);

                    long lW1Count1 = (lW1Count / m_config.n_layers);
                    long lW1LocalOffset = i * lW1Count1;
                    col[nIdxW1].LoadFromBlobLoader(hBlobLoader, lW1Count1, lW1LocalOffset);
                    col[nIdxW1].AddToBlobLoaderOffset(hBlobLoader, lW1Count);
                    fFirst = col[nIdxW1].GetDataAsFloat(0);
                    nIdx++;

                    long lW3Count1 = (lW3Count / m_config.n_layers);
                    long lW3LocalOffset = i * lW3Count1;
                    col[nIdxW3].LoadFromBlobLoader(hBlobLoader, lW3Count1, lW3LocalOffset);
                    col[nIdxW3].AddToBlobLoaderOffset(hBlobLoader, lW3Count);
                    fFirst = col[nIdxW3].GetDataAsFloat(0);
                    nIdx++;

                    long lW2Count1 = (lW2Count / m_config.n_layers);
                    long lW2LocalOffset = i * lW2Count1;
                    col[nIdxW2].LoadFromBlobLoader(hBlobLoader, lW2Count1, lW2LocalOffset);
                    col[nIdxW2].AddToBlobLoaderOffset(hBlobLoader, lW2Count);
                    fFirst = col[nIdxW2].GetDataAsFloat(0);
                    nIdx++;

                    col[0].Log.WriteLine("Loading weights for layer " + i.ToString() + " of " + m_config.n_layers.ToString() + " ...", true);
                    col[0].Log.Progress = ((double)i / (double)m_config.n_layers);
                }

                // read in the rms_final_weights
                long lRmsFinalCount = m_config.dim;
                lOffset += lRmsFinalCount;

                int nIdxRmsFinal = 1 + m_config.n_layers * 9;
                col[nIdxRmsFinal].LoadFromBlobLoader(hBlobLoader, m_config.dim, 0);
                col[nIdxRmsFinal].AddToBlobLoaderOffset(hBlobLoader, m_config.dim);
                fFirst = col[nIdxRmsFinal].GetDataAsFloat(0);

                // read in the wcls weights
                long lWclsCount = m_config.dim * m_config.vocab_size;
                // Skip the freq_cis weights
                col[nIdxRmsFinal].AddToBlobLoaderOffset(hBlobLoader, m_config.seq_len * nHeadSize / 2);
                col[nIdxRmsFinal].AddToBlobLoaderOffset(hBlobLoader, m_config.seq_len * nHeadSize / 2);

                int nIdxWcls = nIdxRmsFinal + 1;
                if (m_config.shared_classifier == 0)
                {
                    col[nIdxWcls].LoadFromBlobLoader(hBlobLoader, lWclsCount, 0);
                    fFirst = col[nIdxWcls].GetDataAsFloat(0);
                }
                else
                {
                    col[nIdxWcls].CopyFrom(col[0]);
                    fFirst = col[nIdxWcls].GetDataAsFloat(0);
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

        private byte[] loadWeights_kpth1(BlobCollection<T> col, string strModelFile)
        {
            FileInfo fi = new FileInfo(strModelFile);
            long lSize = fi.Length;
            long lOffset = 0;
            long hBlobLoader = 0;
            long lHeaderSize = 256;

            try
            {
                using (var mmf = MemoryMappedFile.CreateFromFile(strModelFile, FileMode.Open, "MyCaffeSharedMemory"))
                {
                    using (var accessor = mmf.CreateViewAccessor(lOffset, lSize, MemoryMappedFileAccess.Read))
                    {
                        byte[] rgHeader = new byte[lHeaderSize];
                        accessor.ReadArray(0, rgHeader, 0, (int)lHeaderSize);

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
                int nIdx = 0;
                float fFirst;

                hBlobLoader = col[0].CreateBlobLoader(strModelFile, lOffset);
                col[0].ResetBlobLoaderOffset(hBlobLoader, lHeaderSize);

                // read in the rms_att_weights                    
                long lRmsAttCount = m_config.n_layers * m_config.dim;
                lOffset += lRmsAttCount;
                // read in the rms_ffn_weights
                long lRmsFfnCount = m_config.n_layers * m_config.dim;
                lOffset += lRmsFfnCount;

                for (int i = 0; i < m_config.n_layers; i++)
                {
                    int nIdxRms1 = 1 + i * 9;
                    int nIdxRms2 = 1 + i * 9 + 5;

                    col[nIdxRms1].ResetBlobLoaderOffset(hBlobLoader, lHeaderSize);

                    long lRmsAttCount1 = (lRmsAttCount / m_config.n_layers);
                    long lRmsAttLocalOffset = i * lRmsAttCount1;
                    col[nIdxRms1].LoadFromBlobLoader(hBlobLoader, lRmsAttCount1, lRmsAttLocalOffset);
                    col[nIdxRms1].AddToBlobLoaderOffset(hBlobLoader, lRmsAttCount);
                    fFirst = col[nIdxRms1].GetDataAsFloat(0);

                    long lRmsFfnCount1 = (lRmsFfnCount / m_config.n_layers);
                    long lRmsFfnLocalOffset = i * lRmsFfnCount1;
                    col[nIdxRms2].LoadFromBlobLoader(hBlobLoader, lRmsFfnCount1, lRmsFfnLocalOffset);
                    col[nIdxRms2].AddToBlobLoaderOffset(hBlobLoader, lRmsFfnCount);
                    fFirst = col[nIdxRms2].GetDataAsFloat(0);
                }

                // read in the rms_final_weights
                long lRmsFinalCount = m_config.dim;
                lOffset += lRmsFinalCount;

                int nIdxRmsFinal = 1 + m_config.n_layers * 9;
                col[nIdxRmsFinal].LoadFromBlobLoader(hBlobLoader, m_config.dim, 0);
                col[nIdxRmsFinal].AddToBlobLoaderOffset(hBlobLoader, m_config.dim);
                fFirst = col[nIdxRmsFinal].GetDataAsFloat(0);

                // read in the token embedding table

                long lTokenEmbeddingCount = m_config.vocab_size * m_config.dim;
                lOffset += lTokenEmbeddingCount;

                col[0].LoadFromBlobLoader(hBlobLoader, lTokenEmbeddingCount, 0);
                col[0].AddToBlobLoaderOffset(hBlobLoader, lTokenEmbeddingCount);
                fFirst = col[0].GetDataAsFloat(0);

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
                // Set the weight offset to Header + rms_att + rms_ffn + rms_final + token_embedding
                long lWeightStartOffset = lHeaderSize + lOffset * sizeof(float);

                for (int i = 0; i < m_config.n_layers; i++)
                {
                    int nIdxWq = 2 + i * 9;
                    int nIdxWk = nIdxWq + 1;
                    int nIdxWv = nIdxWq + 2;
                    int nIdxWo = nIdxWq + 3;
                    int nIdxW1 = nIdxWq + 5;
                    int nIdxW2 = nIdxWq + 6;
                    int nIdxW3 = nIdxWq + 7;

                    col[nIdxWq].ResetBlobLoaderOffset(hBlobLoader, lWeightStartOffset);

                    long lWqCount1 = (lWqCount / m_config.n_layers);
                    long lWqLocalOffset = i * lWqCount1;
                    col[nIdxWq].LoadFromBlobLoader(hBlobLoader, lWqCount1, lWqLocalOffset);
                    col[nIdxWq].AddToBlobLoaderOffset(hBlobLoader, lWqCount);
                    fFirst = col[nIdxWq].GetDataAsFloat(0);
                    nIdx++;

                    long lWkCount1 = (lWkCount / m_config.n_layers);
                    long lWkLocalOffset = i * lWkCount1;
                    col[nIdxWk].LoadFromBlobLoader(hBlobLoader, lWkCount1, lWkLocalOffset);
                    col[nIdxWk].AddToBlobLoaderOffset(hBlobLoader, lWkCount);
                    fFirst = col[nIdxWk].GetDataAsFloat(0);
                    nIdx++;

                    long lWvCount1 = (lWvCount / m_config.n_layers);
                    long lWvLocalOffset = i * lWvCount1;
                    col[nIdxWv].LoadFromBlobLoader(hBlobLoader, lWvCount1, lWvLocalOffset);
                    col[nIdxWv].AddToBlobLoaderOffset(hBlobLoader, lWvCount);
                    fFirst = col[nIdxWv].GetDataAsFloat(0);
                    nIdx++;

                    long lWoCount1 = (lWoCount / m_config.n_layers);
                    long lWoLocalOffset = i * lWoCount1;
                    col[nIdxWo].LoadFromBlobLoader(hBlobLoader, lWoCount1, lWoLocalOffset);
                    col[nIdxWo].AddToBlobLoaderOffset(hBlobLoader, lWoCount);
                    fFirst = col[nIdxWo].GetDataAsFloat(0);
                    nIdx++;

                    long lW1Count1 = (lW1Count / m_config.n_layers);
                    long lW1LocalOffset = i * lW1Count1;
                    col[nIdxW1].LoadFromBlobLoader(hBlobLoader, lW1Count1, lW1LocalOffset);
                    col[nIdxW1].AddToBlobLoaderOffset(hBlobLoader, lW1Count);
                    fFirst = col[nIdxW1].GetDataAsFloat(0);
                    nIdx++;

                    long lW3Count1 = (lW3Count / m_config.n_layers);
                    long lW3LocalOffset = i * lW3Count1;
                    col[nIdxW3].LoadFromBlobLoader(hBlobLoader, lW3Count1, lW3LocalOffset);
                    col[nIdxW3].AddToBlobLoaderOffset(hBlobLoader, lW3Count);
                    fFirst = col[nIdxW3].GetDataAsFloat(0);
                    nIdx++;

                    long lW2Count1 = (lW2Count / m_config.n_layers);
                    long lW2LocalOffset = i * lW2Count1;
                    col[nIdxW2].LoadFromBlobLoader(hBlobLoader, lW2Count1, lW2LocalOffset);
                    col[nIdxW2].AddToBlobLoaderOffset(hBlobLoader, lW2Count);
                    fFirst = col[nIdxW2].GetDataAsFloat(0);
                    nIdx++;

                    col[0].Log.WriteLine("Loading weights for layer " + i.ToString() + " of " + m_config.n_layers.ToString() + " ...", true);
                    col[0].Log.Progress = ((double)i / (double)m_config.n_layers);
                }

                int nIdxWcls = nIdxRmsFinal + 1;
                if (m_config.shared_classifier == 0)
                {
                    col[nIdxWcls].LoadFromBlobLoader(hBlobLoader, lWclsCount, 0);
                    fFirst = col[nIdxWcls].GetDataAsFloat(0);
                }
                else
                {
                    col[nIdxWcls].CopyFrom(col[0]);
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

        private string buildModel(TokenizedDataParameter.VOCABULARY_TYPE vocabType, NetParameter net, uint nBatch, uint nBlockSize, uint nEncVocabSize, uint nEmbed, uint nHiddenDim, uint nHeads, uint nLayers, double dfDropout, Phase phase, bool bEnableLoRA, bool bPreTokenizer, bool bEnableKeyValueCache)
        {
            net.enable_lora = true;
            net.enable_lora_only = true;
            net.model_type = NetParameter.MODEL_TYPE.LLAMA;

            if (bPreTokenizer)
            {
                LayerParameter tok = new LayerParameter(LayerParameter.LayerType.PRETOKENIZED_DATA, "data");
                tok.pretokenized_data_param.sample_method = PreTokenizedDataParameter.SAMPLE_METHOD.PROBABILITY;
                tok.pretokenized_data_param.source = "$ProgramData$\\MyCaffe\\test_data\\llama\\test\\stories\\instruct_dataset\\";
                tok.pretokenized_data_param.batch_size = nBatch;
                tok.pretokenized_data_param.block_size = nBlockSize;
                tok.pretokenized_data_param.shuffle = true;
                tok.pretokenized_data_param.pad_token = -100;
                tok.pretokenized_data_param.vocabulary_type = PreTokenizedDataParameter.VOCABULARY_TYPE.LLAMA2;
                tok.freeze_learning = true;
                tok.top.Add("tokdata");
                if (phase != Phase.RUN)
                    tok.top.Add("tgt");
                net.layer.Add(tok);
            }
            else
            {
                LayerParameter tok = new LayerParameter(LayerParameter.LayerType.TOKENIZED_DATA, "data");
                tok.tokenized_data_param.input_type = TokenizedDataParameter.INPUT_TYPE.TEXT_FILE;
                tok.tokenized_data_param.sample_method = TokenizedDataParameter.SAMPLE_METHOD.PROBABILITY;
                tok.tokenized_data_param.vocabulary_type = vocabType;
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
            }

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
                enc.transformer_block_param.normalization_type = TransformerBlockParameter.NORMALIZATION.RMS_NORM;
                enc.transformer_block_param.attn_dropout = dfDropout;
                enc.transformer_block_param.resid_dropout = dfDropout;
                enc.transformer_block_param.enable_layernorm_cuda_impl = false;
                enc.transformer_block_param.enable_llama_style_head = true;
                enc.transformer_block_param.enable_rotary_positional_embedding = true;
                enc.transformer_block_param.enable_key_value_cache = bEnableKeyValueCache;
                enc.transformer_block_param.bias_term = false;

                if (bEnableLoRA)
                {
                    enc.transformer_block_param.weight_adapter_q.enabled = true;
                    enc.transformer_block_param.weight_adapter_k.enabled = true;
                    enc.transformer_block_param.weight_adapter_v.enabled = true;
                    enc.transformer_block_param.weight_adapter_out.enabled = true;
                }

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

            LayerParameter ln1 = new LayerParameter(LayerParameter.LayerType.RMSNORM);
            ln1.name = "ln1";
            ln1.rms_norm_param.axis = 2;
            ln1.rms_norm_param.enable_weights = true;
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
                loss.loss_param.ignore_label = -1;
                loss.loss_param.normalization = LossParameter.NormalizationMode.VALID;
                loss.loss_param.loss_scale = 1.0 / m_nIterSize;
                loss.bottom.Add("prob");
                loss.bottom.Add("tgt");
                loss.top.Add("loss");
                net.layer.Add(loss);

                LayerParameter accuracy = new LayerParameter(LayerParameter.LayerType.ACCURACY);
                accuracy.name = "accuracy";
                accuracy.accuracy_param.enable_simple_accuracy = true;
                accuracy.accuracy_param.axis = 2;
                accuracy.accuracy_param.ignore_labels.Add(-1);
                accuracy.bottom.Add("prob");
                accuracy.bottom.Add("tgt");
                accuracy.top.Add("accuracy");
                net.layer.Add(accuracy);
            }

            return net.ToProto("root").ToString();
        }
    }
}
