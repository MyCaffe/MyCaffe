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
        Blob<T> m_blobIdx = null;
        Blob<T> m_blobPos = null;
        Blob<T> m_blobX = null;
        Blob<T> m_blobY = null;
        Blob<T> m_blobProb = null;
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
        ///  - batch_size.  The batch size (currently only 1 supported).
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
            dispose(ref m_blobIdx);
            dispose(ref m_blobPos);
            dispose(ref m_blobY);
            dispose(ref m_blobX);
            dispose(ref m_blobProb);

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
                    m_data = new TextInputData(m_param.tokenized_data_param.source, m_param.tokenized_data_param.seed, m_param.tokenized_data_param.debug_index_file, m_param.phase);
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
        /// Data layers have no bottoms, so reshaping is trivial.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
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

                if (m_param.tokenized_data_param.tokenize_run_input)
                {
                    float[] rgInput = convertF(colBottom[0].mutable_cpu_data);
                    rgInput = m_data.Tokenize(rgInput);
                    colTop[0].mutable_cpu_data = convert(rgInput);
                }
                else
                {
                    colTop[0].CopyFrom(colBottom[0]);
                }
                // Top[1] should already have pos data in it.
                // There is no Top[2] target data when running.
            }
            else
            {
                Tuple<float[], float[]> data = m_data.GetData((int)m_param.tokenized_data_param.batch_size, (int)m_param.tokenized_data_param.block_size);

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
        /// Tokenize the source data by converting it from its native form to index values that reference into the vocabulary.
        /// </summary>
        /// <param name="blobSrc">Specifies the native source data.</param>
        /// <param name="blobDst">Specifies the tokenized destination data.</param>
        public void Tokenize(Blob<T> blobSrc, Blob<T> blobDst)
        {
            float[] rgSrc = convertF(blobSrc.mutable_cpu_data);
            blobDst.mutable_cpu_data = convert(m_data.Tokenize(rgSrc));
        }

        /// <summary>
        /// Detokenize the source data by converting it to its native form.
        /// </summary>
        /// <param name="blobSrc">Specifies the tokenized source data.</param>
        /// <param name="blobDst">Specifies the detokenized destination data.</param>
        public void Detokenize(Blob<T> blobSrc, Blob<T> blobDst)
        {
            float[] rgSrc = convertF(blobSrc.mutable_cpu_data);
            blobDst.mutable_cpu_data = convert(m_data.Detokenize(rgSrc));
        }

        /// <summary>
        /// Preproces the input and return as a set of bottom blobs.
        /// </summary>
        /// <param name="customInput">Specifies the custom text input.</param>
        /// <param name="colBottom">The output is placed in the bottom blobs as: tokidx, pos</param>
        /// <returns>The bottom blob collection is returned.</returns>
        public override BlobCollection<T> PreProcessInput(PropertySet customInput, BlobCollection<T> colBottom = null)
        {
            if (m_blobIdx == null)
                m_blobIdx = new Blob<T>(m_cuda, m_log, false);

            string strInput = customInput.GetProperty("InputData");
            if (string.IsNullOrEmpty(strInput))
                throw new Exception("Could not find 'InputData' property!");

            int[] rgShape = new int[2];
            rgShape[0] = 1;
            rgShape[1] = strInput.Length;

            m_blobIdx.Reshape(rgShape);

            float[] rgInput = new float[strInput.Length];

            for (int i = 0; i < strInput.Length; i++)
            {
                rgInput[i] = (int)strInput[i];
            }

            m_blobIdx.mutable_cpu_data = convert(m_data.Tokenize(rgInput));

            return new BlobCollection<T>() { m_blobIdx };
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
            str += m_data.Detokenize(nCharIdx);

            return new List<Tuple<string, int, double>>() { new Tuple<string, int, double>(str, nCharIdx, 0) };
        }
    }

    /// <summary>
    /// The InputData is an abstract class used to get training data and tokenize input data.
    /// </summary>
    public abstract class InputData
    {
        /// <summary>
        /// Specifies the random object made available to the derived classes.
        /// </summary>
        protected Random m_random;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nRandomSeed">Optionally, specifies the seed to use for testing.</param>
        public InputData(int? nRandomSeed = null)
        {
            if (nRandomSeed.HasValue)
                m_random = new Random(nRandomSeed.Value);
            else
                m_random = new Random();
        }

        /// <summary>
        /// Returns the size of a single token (e.g. 1 for character data)
        /// </summary>
        public abstract uint TokenSize { get; }
        /// <summary>
        /// Returns the size of the vocabulary.
        /// </summary>
        public abstract uint VocabularySize { get; }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="nBatchSize">Specifies the number of blocks in the batch.</param>
        /// <param name="nBlockSize">Specifies the size of each block.</param>
        /// <returns>A tuple containing the data and target is returned.</returns>
        public abstract Tuple<float[], float[]> GetData(int nBatchSize, int nBlockSize);
        /// <summary>
        /// Tokenize the input data.
        /// </summary>
        /// <param name="rgInput">Specifies the untokenized input data.</param>
        /// <returns>The tokenized data is returned.</returns>
        public abstract float[] Tokenize(float[] rgInput);
        /// <summary>
        /// Detokenize the input data, but converting tokenized data to its native form.
        /// </summary>
        /// <param name="rgInput">Specifies the tokenized data.</param>
        /// <returns>The de-tokenized data is returned.</returns>
        public abstract float[] Detokenize(float[] rgInput);
        /// <summary>
        /// Detokenize a single token.
        /// </summary>
        /// <param name="nTokIdx">Specifies an index to the token to be detokenized.</param>
        /// <returns>The detokenized character is returned.</returns>
        public abstract char Detokenize(int nTokIdx);
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
        Dictionary<char, int> m_rgVocabKeyToIdx = new Dictionary<char, int>();
        Dictionary<int, char> m_rgVocabIdxToKey = new Dictionary<int, char>();
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
        /// <param name="nRandomSeed">Optionally, specifies a random seed for testing.</param>
        /// <param name="strDebugIndexFile">Optionally, specifies the debug index file containing index values in the form 'idx = #', one per line.</param>
        /// <param name="phase">Specifies the currently running phase.</param>
        public TextInputData(string strSrc, int? nRandomSeed = null, string strDebugIndexFile = null, Phase phase = Phase.NONE) : base(nRandomSeed)
        {
            m_phase = phase;
            m_strData = File.ReadAllText(strSrc);

            if (File.Exists(strDebugIndexFile))
            {
                m_strDebugIndexFile = strDebugIndexFile;
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

            m_rgVocabKeyToIdx.Clear();
            foreach (char ch in m_strData)
            {
                if (!m_rgVocabKeyToIdx.ContainsKey(ch))
                    m_rgVocabKeyToIdx.Add(ch, 1);
            }

            List<char> rgKeys = m_rgVocabKeyToIdx.Keys.ToList();
            rgKeys.Sort();

            m_rgVocabKeyToIdx.Clear();

            for (int i = 0; i < rgKeys.Count; i++)
            {
                m_rgVocabKeyToIdx.Add(rgKeys[i], i);
                m_rgVocabIdxToKey.Add(i, rgKeys[i]);
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
            get { return (uint)m_rgVocabKeyToIdx.Count; }
        }

        /// <summary>
        /// Retrieve random blocks from the source data where the data and target are the same
        /// but offset by one element where the target is offset +1 from the data.
        /// </summary>
        /// <param name="nBatchSize">Specifies the batch size.</param>
        /// <param name="nBlockSize">Specifies teh block size.</param>
        /// <returns>A tuple containing the data and target is returned.</returns>
        public override Tuple<float[], float[]> GetData(int nBatchSize, int nBlockSize)
        {
            int nSize = nBatchSize * nBlockSize;

            if (m_rgData == null || m_rgData.Length != nSize)
                m_rgData = new float[nSize];

            if (m_rgTgt == null || m_rgTgt.Length != nSize)
                m_rgTgt = new float[nSize];

            for (int i = 0; i < nBatchSize; i++)
            {
                int nMax = m_strData.Count() - (nBlockSize + 1);
                int nDataIdx = m_random.Next(nMax);
                int nDstIdx = i * nBlockSize;

                if (m_rgDebugIdx != null)
                {
                    nDataIdx = m_rgDebugIdx[m_nDebugIdx];
                    m_nDebugIdx++;

                    if (m_nDebugIdx >= m_rgDebugIdx.Count)
                        m_nDebugIdx = 0;
                }

                int? nCharIdxLast = null;
                for (int j = 0; j < nBlockSize + 1; j++)
                {
                    if (nCharIdxLast.HasValue)
                        m_rgData[nDstIdx + j - 1] = nCharIdxLast.Value;

                    char ch = m_strData[nDataIdx + j];
                    int nCharIdx = m_rgVocabKeyToIdx[ch];

                    if (nCharIdx < 0 || nCharIdx > 65)
                        throw new Exception("Token out of range!");

                    if (j > 0)
                        m_rgTgt[nDstIdx + j - 1] = nCharIdx;

                    nCharIdxLast = nCharIdx;
                }
            }

            return new Tuple<float[], float[]>(m_rgData, m_rgTgt);
        }

        /// <summary>
        /// Convert text input (input as a set of ASCII character values) into their respective
        /// char indexes in the vocabulary.
        /// </summary>
        /// <param name="rgInput">Specifies input data where each element is an ASCII character numeric value.</param>
        /// <returns>The tokenized input is returned.</returns>
        public override float[] Tokenize(float[] rgInput)
        {
            for (int i = 0; i < rgInput.Count(); i++)
            {
                char ch = (char)rgInput[i];
                int nCharIdx;

                if (m_rgVocabKeyToIdx.ContainsKey(ch))
                    nCharIdx = m_rgVocabKeyToIdx[ch];
                else
                    nCharIdx = m_random.Next(m_rgVocabIdxToKey.Count);

                rgInput[i] = nCharIdx;
            }

            return rgInput;
        }

        /// <summary>
        /// Convert tokenized data back to its native character form.
        /// </summary>
        /// <param name="rgInput">Specifies the tokenized data.</param>
        /// <returns>The characters in numeric form are returned.</returns>
        public override float[] Detokenize(float[] rgInput)
        {
            for (int i = 0; i < rgInput.Count(); i++)
            {
                int nCharIdx = (int)rgInput[i];
                char ch = m_rgVocabIdxToKey[nCharIdx];
                rgInput[i] = ch;
            }

            return rgInput;
        }


        /// <summary>
        /// Detokenize a single token.
        /// </summary>
        /// <param name="nTokIdx">Specifies an index to the token to be detokenized.</param>
        /// <returns>The detokenized character is returned.</returns>
        public override char Detokenize(int nTokIdx)
        {
            return m_rgVocabIdxToKey[nTokIdx];
        }
    }
}
