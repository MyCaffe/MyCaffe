using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using System.IO;
using MyCaffe.db.image;
using MyCaffe.param.gpt;
using System.Net;

namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The TokenizedDataLayer loads and tokenizes data for a transformer model where data is loaded in the form: data, pos, target(optional)
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TokenizedDataLayer<T> : Layer<T>
    {
        CancelEvent m_evtCancel;
        InputData m_data;
        Blob<T> m_blobX = null;
        Blob<T> m_blobY = null;
        Random m_random = new Random();

        /// <summary>
        /// The TokenizedDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Provides TokenizedDataParameter model_data_param with options:
        ///  - source.  The data source(s) where the source is the data input table who's RawImageResults table contains the data for training.
        ///  
        ///  - batch_size.  The batch size.
        ///  
        ///  - time_steps.  The maximum number of time steps.
        ///  
        ///  - input_dim.  The input dimension of the encoder input.
        ///  
        ///  - sample_size.  The number of samples to load for training.
        ///  
        ///  - shuffle.  Whether or not to shuffle the data.
        /// </param>
        /// <param name="db">Specifies the external database to use.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel any pre-fetching operations.</param>
        public TokenizedDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXImageDatabaseBase db, CancelEvent evtCancel)
            : base(cuda, log, p)
        {
            m_evtCancel = evtCancel;
            m_type = LayerParameter.LayerType.TOKENIZED_DATA;
        }

        /// <summary>
        /// Release all internal blobs.
        /// </summary>
        protected override void dispose()
        {
            dispose(ref m_blobY);
            dispose(ref m_blobX);

            base.dispose();
        }

        /// <summary>
        /// No bottom blobs for the data layer, except when running.
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return (m_phase == Phase.RUN) ? 1 : 0; }
        }

        /// <summary>
        /// Returns the minimum number of bottom blobs.
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 0; }
        }

        /// <summary>
        /// Returns the maximum number of required top (output) Blobs: data, pos, target
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 3; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nBlockSize = (int)m_param.tokenized_data_param.block_size;

            switch (m_param.tokenized_data_param.input_type)
            {
                case TokenizedDataParameter.INPUT_TYPE.TEXT_FILE:
                    m_data = new TextInputData(m_param.tokenized_data_param.source, m_param.tokenized_data_param.vocabulary_type, m_param.tokenized_data_param.seed, m_param.tokenized_data_param.debug_index_file, m_param.phase);
                    break;

                default:
                    throw new Exception("Unknown input type '" + m_param.tokenized_data_param.input_type.ToString() + "'");
            }

            Reshape(colBottom, colTop);

            Blob<T> blobPos = colTop[1];
            List<int> rgShape = Utility.Clone<int>(colTop[1].shape());
            rgShape[1] = nBlockSize;
            blobPos.Reshape(rgShape);

            // Set the position data = 0, 1, 2, 3, ... block_size-1
            float[] rgPos = new float[nBlockSize];
            for (int i = 0; i < nBlockSize; i++)
            {
                rgPos[i] = i;
            }

            blobPos.mutable_cpu_data = convert(rgPos);
        }

        /// <summary>
        /// Reshape the top based on the parameter batch and block size.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs - Used only during RUN phase.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_phase == Phase.RUN)
            {
                int nBatchSize = colBottom[0].num;
                int nBlockSize = (int)m_param.tokenized_data_param.block_size;
                int nTokenSize = (int)m_data.TokenSize;

                Blob<T> blobData = colTop[0];
                Blob<T> blobPos = colTop[1];
                Blob<T> blobTarget = colTop[2];

                int nCount = 3;
                if (nTokenSize == 1)
                    nCount = 2;
                int[] rgShape = new int[nCount];

                blobData.SetParameter("vocab_size", m_data.VocabularySize);

                int nC = colBottom[0].channels;
                if (nC > nBlockSize)
                    throw new Exception("The bottom input channel count cannot exceed the block_size=" + nBlockSize.ToString());

                // reshape for single characters (each character is an index into the vocab vector)
                rgShape[0] = nBatchSize;
                rgShape[1] = nC;
                if (rgShape.Length > 2)
                    rgShape[2] = nTokenSize;

                blobData.Reshape(rgShape);
                blobTarget.Reshape(rgShape);

                if (blobPos.count() < nBlockSize)
                {
                    rgShape[0] = 1;
                    if (rgShape.Length > 2)
                        rgShape[2] = 1;

                    blobPos.Reshape(rgShape);
                }

                rgShape[0] = nBatchSize;
                rgShape[1] = nC;
                blobPos.Reshape(rgShape);
            }
            else
            {
                m_log.CHECK_EQ(colBottom.Count, 0, "Data Layer takes no input blobs.");
                m_log.CHECK_EQ(colTop.Count, 3, "The TokenizedDataLayer requires 3 top blobs.");

                int nBatchSize = (int)m_param.tokenized_data_param.batch_size;
                int nBlockSize = (int)m_param.tokenized_data_param.block_size;
                int nTokenSize = (int)m_data.TokenSize;

                Blob<T> blobData = colTop[0];
                Blob<T> blobPos = colTop[1];
                Blob<T> blobTarget = colTop[2];

                int nCount = 3;
                if (nTokenSize == 1)
                    nCount = 2;
                int[] rgShape = new int[nCount];

                blobData.SetParameter("vocab_size", m_data.VocabularySize);
                // reshape for single characters (each character is an index into the vocab vector)
                rgShape[0] = nBatchSize;
                rgShape[1] = nBlockSize;
                if (rgShape.Length > 2)
                    rgShape[2] = nTokenSize;

                blobData.Reshape(rgShape);
                blobTarget.Reshape(rgShape);

                rgShape[0] = 1;
                if (rgShape.Length > 2)
                    rgShape[2] = 1;
                blobPos.Reshape(rgShape);
            }
        }

        /// <summary>
        /// Run the Forward computation, which fills the data into the top (output) Blobs.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">top output blob vector (length 2-3)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the data outputs.  
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the position outputs.
        ///  -# @f$ (N \times C \times H \times W) @f$ (only on training and testing)
        ///     the target outputs
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_phase == Phase.RUN)
            {
                m_log.CHECK_EQ(colBottom.Count, 1, "There must be one input blob when running.");
                colTop[0].CopyFrom(colBottom[0]);
                // Top[1] should already have pos data in it.
                // There is no Top[2] target data when running.
            }
            else
            {
                int[] rgnIdx;
                Tuple<float[], float[]> data = m_data.GetData((int)m_param.tokenized_data_param.batch_size, (int)m_param.tokenized_data_param.block_size, out rgnIdx);

                colTop[0].mutable_cpu_data = convert(data.Item1);
                if (colTop.Count > 2)
                    colTop[2].mutable_cpu_data = convert(data.Item2);
            }
        }

        /// @brief Not implemented - data Layers do not perform backward..
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }

        /// <summary>
        /// Specifies that this layer supports preprocessing.
        /// </summary>
        public override bool SupportsPreProcessing
        {
            get { return true; }
        }

        /// <summary>
        /// Specifies that this layer supports post processing the logits.
        /// </summary>
        public override bool SupportsPostProcessingLogits
        {
            get { return true; }
        }

        /// <summary>
        /// Tokenize an input string using the internal vocabulary.
        /// </summary>
        /// <param name="str">Specifies the string to tokenize.</param>
        /// <returns>A list of tokens corresponding to the input is returned.</returns>
        public List<int> Tokenize(string str)
        {
            return m_data.Tokenize(str);
        }

        /// <summary>
        /// Detokenize a set of tokens from the data specified.
        /// </summary>
        /// <param name="rg">Specifies an array of tokens.</param>
        /// <param name="nStartIdx">Specifies the start index.</param>
        /// <param name="nCount">Specifies the number of tokens to detokenize.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS token.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS token.</param>
        /// <returns>The detokenized string is returned.</returns>
        public string Detokenize(float[] rg, int nStartIdx, int nCount, bool bIgnoreBos = true, bool bIgnoreEos = true)
        {
            return m_data.Detokenize(rg, nStartIdx, nCount, bIgnoreBos, bIgnoreEos);
        }

        /// <summary>
        /// Preproces the input and return as a set of bottom blobs.
        /// </summary>
        /// <param name="customInput">Specifies the custom text input.</param>
        /// <param name="colBottom">The output is placed in the bottom blobs as: tokidx, pos</param>
        /// <returns>The bottom blob collection is returned.</returns>
        public override BlobCollection<T> PreProcessInput(PropertySet customInput, BlobCollection<T> colBottom = null)
        {
            Blob<T> blobIdx = new Blob<T>(m_cuda, m_log, false);

            string strInput = customInput.GetProperty("InputData");
            if (string.IsNullOrEmpty(strInput))
                throw new Exception("Could not find 'InputData' property!");

            int[] rgShape = new int[2];
            rgShape[0] = 1;
            rgShape[1] = strInput.Length;

            blobIdx.Reshape(rgShape);

            List<int> rgTokens = m_data.Tokenize(strInput);
            float[] rgInput = new float[rgTokens.Count];

            for (int i = 0; i < strInput.Length; i++)
            {
                rgInput[i] = rgTokens[i];
            }

            blobIdx.mutable_cpu_data = convert(rgInput);

            return new BlobCollection<T>() { blobIdx };
        }

        /// <summary>
        /// Preproces the input and return as a set of bottom blobs.
        /// </summary>
        /// <param name="str">Specifies the string input, can be null.</param>
        /// <param name="nTokIdx">Specifies the token input.</param>
        /// <param name="colBottom">The output is placed in the bottom blobs as: tokidx, pos</param>
        /// <returns>The bottom blob collection is returned.</returns>
        public override void PreProcessInput(string str, int? nTokIdx, BlobCollection<T> colBottom = null)
        {
            List<float> rgTok = convertF(colBottom[0].mutable_cpu_data).ToList();

            rgTok.Add(nTokIdx.Value);
            if (rgTok.Count > m_param.tokenized_data_param.block_size)
                rgTok.RemoveAt(0);

            List<int> rgShape = Utility.Clone<int>(colBottom[0].shape());
            rgShape[1] = rgTok.Count;
            colBottom[0].Reshape(rgShape);
            
            colBottom[0].mutable_cpu_data = convert(rgTok.ToArray());
        }

        /// <summary>
        /// Allows post processing the logits output data by converting the logits to and selecting 
        /// from the probability distribution produced and detokenizing the results to the string character.
        /// </summary>
        /// <param name="blobLogits">Specifies the output of the last inner product layer.</param>
        /// <param name="softmax">Specifies the softmax layer.</param>
        /// <param name="nK">Specifies the TopK max items of the logits to use, or 0 to ignore.</param>
        /// <returns>
        /// The detokenized data is returned.
        /// </returns>
        public override List<Tuple<string, int, double>> PostProcessLogitsOutput(Blob<T> blobLogits, Layer<T> softmax, int nK = 1)
        {
            float[] rgData = convertF(blobLogits.mutable_cpu_data);
            int nVocabCount = blobLogits.count(softmax.layer_param.softmax_param.axis);
            float[] rgLogits = new float[nVocabCount];
            int nIdxStart = blobLogits.count() - nVocabCount;
            Dictionary<int, float> rgTopK = new Dictionary<int, float>();

            for (int i = nIdxStart; i < blobLogits.count(); i++)
            {
                float fVal = rgData[i];
                rgTopK.Add(i - nIdxStart, fVal);

                if (rgTopK.Count > nK)
                {
                    float fMin = float.MaxValue;
                    int nMinIdx = -1;

                    foreach (KeyValuePair<int, float> kv in rgTopK)
                    {
                        if (kv.Value < fMin)
                        {
                            fMin = kv.Value;
                            nMinIdx = kv.Key;
                        }
                    }

                    rgTopK.Remove(nMinIdx);
                }
            }

            for (int i = 0; i < rgLogits.Count(); i++)
            {
                if (rgTopK.ContainsKey(i))
                    rgLogits[i] = rgTopK[i];
                else
                    rgLogits[i] = -float.MaxValue;
            }

            if (m_blobX == null)
                m_blobX = new Blob<T>(m_cuda, m_log, false);
            if (m_blobY == null)
                m_blobY = new Blob<T>(m_cuda, m_log, false);

            m_blobX.Reshape(1, 1, nVocabCount, 1);
            m_blobX.mutable_cpu_data = convert(rgLogits);

            BlobCollection<T> colBottom = new BlobCollection<T>() { m_blobX };
            BlobCollection<T> colTop = new BlobCollection<T>() { m_blobY };
            softmax.Forward(colBottom, colTop);

            float[] rgProb = convertF(m_blobY.mutable_cpu_data);
            float fRand = (float)m_random.NextDouble();
            float fTotal = 0;
            int nCharIdx = rgProb.Length - 1;

            for (int i = 0; i < rgProb.Length; i++)
            {
                fTotal += rgProb[i];

                if (fTotal >= fRand)
                {
                    nCharIdx = i;
                    break;
                }
            }

            string str = "";
            str += m_data.Detokenize(nCharIdx, true, true);

            return new List<Tuple<string, int, double>>() { new Tuple<string, int, double>(str, nCharIdx, 0) };
        }
    }

    /// <summary>
    /// The TextInputData manages character data read in from a text file.  Data is tokenized into indexes that reference each character
    /// within the vocabulary.
    /// </summary>
    /// <remarks>
    /// For example if the data source contains the text "a red fox ran.",
    /// the vocabulary would be:
    /// 
    /// Vocabulary: ' ', '.', 'a', 'd', 'e', 'f', 'o', 'n', 'r'
    /// Index Vals:  0,   1,   2,   3,   4,   5,   6,   7,   8
    /// 
    /// Tokenizing is the process of converting each input character to its respective 'token' or in this case, index value.
    /// So, for example, 'a' is tokenized as index 2; 'd' is tokenized as index 3, etc.
    /// </remarks>
    public class TextInputData : InputData
    {
        string m_strData;
        IVocabulary m_vocab;
        string m_strDebugIndexFile;
        List<int> m_rgDebugIdx = null;
        int m_nDebugIdx = 0;
        float[] m_rgData = null;
        float[] m_rgTgt = null;
        Phase m_phase;


        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strSrc">Specifies the data source as the filename of the text data file.</param>
        /// <param name="vocabType">Specifies the vocabulary type to use.</param>
        /// <param name="nRandomSeed">Optionally, specifies a random seed for testing.</param>
        /// <param name="strDebugIndexFile">Optionally, specifies the debug index file containing index values in the form 'idx = #', one per line.</param>
        /// <param name="phase">Specifies the currently running phase.</param>
        public TextInputData(string strSrc, TokenizedDataParameter.VOCABULARY_TYPE vocabType = TokenizedDataParameter.VOCABULARY_TYPE.CHARACTER, int? nRandomSeed = null, string strDebugIndexFile = null, Phase phase = Phase.NONE) : base(nRandomSeed)
        {
            string strProgData = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
            strSrc = Utility.ReplaceMacro(strSrc, "$ProgramData$", strProgData);

            m_phase = phase;
            m_strData = File.ReadAllText(strSrc);

            if (File.Exists(strDebugIndexFile))
            {
                m_strDebugIndexFile = Utility.ReplaceMacro(strDebugIndexFile, "$ProgramData$", strProgData);
                m_rgDebugIdx = new List<int>();
                string[] rgLines = File.ReadAllLines(strDebugIndexFile);
                foreach (string strLine in rgLines)
                {
                    if (strLine.StartsWith("idx = "))
                    {
                        string strIdx = strLine.Substring(6).Trim(' ', '\t', '\n', '\r');
                        m_rgDebugIdx.Add(int.Parse(strIdx));
                    }
                }
            }

            if (vocabType == TokenizedDataParameter.VOCABULARY_TYPE.WORD)
                m_vocab = new VocabularyWord(m_random, false, false);
            else
               m_vocab = new VocabularyCharacter(m_random, false, false);
            
            m_vocab.BuildFromString(m_strData);
        }

        /// <summary>
        /// Return the raw data.
        /// </summary>
        public override List<string> RawData
        {
            get
            {
                return new List<string>() { m_strData };
            }
        }

        /// <summary>
        /// The text data token size is a single character.
        /// </summary>
        public override uint TokenSize
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the number of unique characters in the data.
        /// </summary>
        public override uint VocabularySize
        {
            get { return (uint)m_vocab.Count; }
        }

        /// <summary>
        /// Retrieve random blocks from the source data where the data and target are the same
        /// but offset by one element where the target is offset +1 from the data.
        /// </summary>
        /// <param name="nBatchSize">Specifies the batch size.</param>
        /// <param name="nBlockSize">Specifies teh block size.</param>
        /// <param name="rgnIdx">Returns an array of indexes of the data returned.</param>
        /// <returns>A tuple containing the data and target is returned.</returns>
        public override Tuple<float[], float[]> GetData(int nBatchSize, int nBlockSize, out int[] rgnIdx)
        {
            int nSize = nBatchSize * nBlockSize;

            rgnIdx = new int[nBatchSize];

            if (m_rgData == null || m_rgData.Length != nSize)
                m_rgData = new float[nSize];

            if (m_rgTgt == null || m_rgTgt.Length != nSize)
                m_rgTgt = new float[nSize];

            for (int i = 0; i < nBatchSize; i++)
            {
                int nMax = m_strData.Count() - (nBlockSize + 1);
                int nDataIdx = m_random.Next(nMax);
                int nDstIdx = i * nBlockSize;

                rgnIdx[i] = nDataIdx;

                if (m_rgDebugIdx != null)
                {
                    nDataIdx = m_rgDebugIdx[m_nDebugIdx];
                    m_nDebugIdx++;

                    if (m_nDebugIdx >= m_rgDebugIdx.Count)
                        m_nDebugIdx = 0;
                }

                List<int> rgTokens = new List<int>();
                List<int> rgLastTokens;
                int nIdx = 0;
                
                while (rgTokens.Count < nBlockSize + 1)
                {
                    rgLastTokens = m_vocab.Tokenize(m_strData[nDataIdx + nIdx].ToString());
                    if (rgLastTokens.Count > 0)
                        rgTokens.AddRange(rgLastTokens);

                    nIdx++;
                }

                Array.Copy(rgTokens.ToArray(), 0, m_rgData, nDstIdx, nBlockSize);
                rgTokens.RemoveAt(0);
                Array.Copy(rgTokens.ToArray(), 0, m_rgTgt, nDstIdx, nBlockSize);
            }

            return new Tuple<float[], float[]>(m_rgData, m_rgTgt);
        }

        /// <summary>
        /// Specifies the GetDataAt method - Not used.
        /// </summary>
        /// <param name="nBatchSize">Specifies the number of blocks in the batch.</param>
        /// <param name="nBlockSize">Specifies the size of each block.</param>
        /// <param name="rgnIdx">Not used.</param>
        /// <exception cref="NotImplementedException"></exception>
        public override Tuple<float[], float[]> GetDataAt(int nBatchSize, int nBlockSize, int[] rgnIdx)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Tokenize an input string using the internal vocabulary.
        /// </summary>
        /// <param name="str">Specifies the string to tokenize.</param>
        /// <returns>A list of tokens corresponding to the input is returned.</returns>
        public override List<int> Tokenize(string str)
        {
            return m_vocab.Tokenize(str);
        }
        
        /// <summary>
        /// Detokenize a single token.
        /// </summary>
        /// <param name="nTokIdx">Specifies an index to the token to be detokenized.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS token.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS token.</param>
        /// <returns>The detokenized character is returned.</returns>
        public override string Detokenize(int nTokIdx, bool bIgnoreBos, bool bIgnoreEos)
        {
            return m_vocab.Detokenize(nTokIdx, bIgnoreBos, bIgnoreEos);
        }

        /// <summary>
        /// Detokenize an array into a string.
        /// </summary>
        /// <param name="rgfTokIdx">Specifies the array of tokens to detokenize.</param>
        /// <param name="nStartIdx">Specifies the starting index where detokenizing begins.</param>
        /// <param name="nCount">Specifies the number of tokens to detokenize.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS token.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS token.</param>
        /// <returns>The detokenized string is returned.</returns>
        public override string Detokenize(float[] rgfTokIdx, int nStartIdx, int nCount, bool bIgnoreBos, bool bIgnoreEos)
        {
            string str = "";
            for (int i = nStartIdx; i < nStartIdx + nCount; i++)
            {
                string strItem = m_vocab.Detokenize((int)rgfTokIdx[i], bIgnoreBos, bIgnoreEos);
                if (string.IsNullOrEmpty(strItem))
                    break;

                str += strItem;
            }

            return str;
        }

        /// <summary>
        /// Return the special begin of sequence character.
        /// </summary>
        public override char BOS
        { 
            get { return m_vocab.BOS; }
        }

        /// <summary>
        /// Return the special end of sequence character.
        /// </summary>
        public override char EOS
        {
            get { return m_vocab.EOS; }
        }
    }
}
