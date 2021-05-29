using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using System.IO;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The TextDataLayer loads data from text data files for an encoder/decoder type model.
    /// This layer is initialized with the MyCaffe.param.TextDataParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TextDataLayer<T> : Layer<T>
    {
        DataItem m_currentData = null;
        Data m_data = null;
        Vocabulary m_vocab = null;
        ulong m_lOffset = 0;
        float[] m_rgEncInput1;
        float[] m_rgEncInput2;
        float[] m_rgEncClip;
        float[] m_rgDecInput;
        float[] m_rgDecClip;
        float[] m_rgDecTarget;

        /// <summary>
        /// The TextDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Provides DummyDataParameter hdf5_data_param with options:
        ///  - data_filler. A list of Fillers to use.
        ///  
        ///  - shape.  A list of shapes to use.
        /// </param>
        public TextDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.TEXT_DATA;
        }

        /// <summary>
        /// Release all internal blobs.
        /// </summary>
        protected override void dispose()
        {
            base.dispose();
        }

        /// <summary>
        /// Returns 0 for data layers have no bottom (input) Blobs.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 0; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: dec, dclip, label, enc, eclip, vocabcount
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 6; }
        }

        /// <summary>
        /// Returns the maximum number of required top (output) Blobs: dec, dclip, label, enc, encr, eclip, vocabcount
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 7; }
        }

        /// <summary>
        /// Returns the vocabulary of the data sources.
        /// </summary>
        public Vocabulary Vocabulary
        {
            get { return m_vocab; }
        }

        private string clean(string str)
        {
            string strOut = "";

            foreach (char ch in str)
            {
                if (ch == 'á')
                    strOut += 'a';
                else if (ch == 'é')
                    strOut += 'e';
                else if (ch == 'í')
                    strOut += 'i';
                else if (ch == 'ó')
                    strOut += 'o';
                else if (ch == 'ú')
                    strOut += 'u';
                else if (ch == 'Á')
                    strOut += 'A';
                else if (ch == 'É')
                    strOut += 'E';
                else if (ch == 'Í')
                    strOut += 'I';
                else if (ch == 'Ó')
                    strOut += 'O';
                else if (ch == 'Ú')
                    strOut += 'U';
                else
                    strOut += ch;
            }

            return strOut;
        }

        private List<string> preprocess(string str, int nMaxLen = 0)
        {
            string strInput = clean(str);
            List<string> rgstr = strInput.ToLower().Trim().Split(' ').ToList();

            if (nMaxLen > 0)
            {
                rgstr = rgstr.Take(nMaxLen).ToList();
                if (rgstr.Count < nMaxLen)
                    return null;
            }

            return rgstr;
        }

        /// <summary>
        /// Load the input and target files and convert each into a list of lines each containing a list of words per line.
        /// </summary>
        public void PreProcessInputFiles(TextDataParameter p)
        {
            List<List<string>> rgrgstrInput = new List<List<string>>();
            List<List<string>> rgrgstrTarget = new List<List<string>>();

            string[] rgstrInput = File.ReadAllLines(p.encoder_source);
            string[] rgstrTarget = File.ReadAllLines(p.decoder_source);

            if (rgstrInput.Length != rgstrTarget.Length)
                throw new Exception("Both the input and target files must contains the same number of lines!");

            for (int i = 0; i < p.sample_size; i++)
            {
                int nMaxLenInput = 0;
                int nMaxLenTarget = 0;

                List<string> rgstrInput1 = preprocess(rgstrInput[i], nMaxLenInput);
                List<string> rgstrTarget1 = preprocess(rgstrTarget[i], nMaxLenTarget);

                if (rgstrInput1 != null && rgstrTarget1 != null)
                {
                    rgrgstrInput.Add(rgstrInput1);
                    rgrgstrTarget.Add(rgstrTarget1);
                }
            }

            m_vocab = new Vocabulary();
            m_vocab.Load(rgrgstrInput, rgrgstrTarget);
            m_data = new Data(rgrgstrInput, rgrgstrTarget, m_vocab);
        }


        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Refuse transformation parameters since HDF5 is totally generic.
            if (m_param.transform_param != null)
                m_log.WriteLine("WARNING: " + m_type.ToString() + " does not transform data.");

            m_log.CHECK_EQ(m_param.text_data_param.batch_size, 1, "Currently, only batch_size = 1 supported.");

            if (m_param.text_data_param.enable_normal_encoder_output && m_param.text_data_param.enable_reverse_encoder_output)
                m_log.CHECK_EQ(colTop.Count, 7, "When normal and reverse encoder output used, there must be 5 tops: dec, dclip, enc, encr, eclip, vocabcount");
            else if (m_param.text_data_param.enable_normal_encoder_output || m_param.text_data_param.enable_reverse_encoder_output)
                m_log.CHECK_EQ(colTop.Count, 6, "When normal or reverse encoder output used, there must be 4 tops: dec, dclip, enc | encr, eclip, vocabcount");
            else
                m_log.FAIL("You must specify to enable either normal, reverse or both encoder inputs.");

            // Load the encoder and decoder input files into the Data and Vocabulary.
            PreProcessInputFiles(m_param.text_data_param);

            m_rgDecInput = new float[m_param.text_data_param.batch_size];
            m_rgDecClip = new float[m_param.text_data_param.batch_size];
            m_rgDecTarget = new float[m_param.text_data_param.batch_size];
            m_rgEncInput1 = new float[m_param.text_data_param.batch_size * m_param.text_data_param.time_steps];
            m_rgEncInput2 = new float[m_param.text_data_param.batch_size * m_param.text_data_param.time_steps];
            m_rgEncClip = new float[m_param.text_data_param.batch_size * m_param.text_data_param.time_steps];

            int nTopIdx = 0;

            nTopIdx++;
            nTopIdx++;
            nTopIdx++;

            if (m_param.text_data_param.enable_normal_encoder_output || m_param.text_data_param.enable_reverse_encoder_output)
                nTopIdx++;

            if (m_param.text_data_param.enable_normal_encoder_output && m_param.text_data_param.enable_reverse_encoder_output)
                nTopIdx++;

            nTopIdx++;

            // Reshape and set the vocabulary count variable.
            colTop[nTopIdx].Reshape(1, 1, 1, 1);
            colTop[nTopIdx].SetData(m_vocab.VocabularCount, 0);
        }

        /// <summary>
        /// Skip to the next data input.
        /// </summary>
        /// <returns>Returns true if a skip should occur, false otherwise.</returns>
        protected bool Skip()
        {
            ulong nSize = (ulong)m_param.solver_count;
            ulong nRank = (ulong)m_param.solver_rank;
            // In test mode, only rank 0 runs, so avoid skipping.
            bool bKeep = (m_lOffset % nSize) == nRank || m_param.phase == Phase.TEST;

            return !bKeep;
        }

        /// <summary>
        /// Proceeds to the next data item.  When shuffling, the next item is randomly selected.
        /// </summary>
        protected void Next()
        {
            m_currentData = m_data.GetNextData(m_param.text_data_param.shuffle);
        }

        /// <summary>
        /// Data layers have no bottoms, so reshaping is trivial.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nBatchSize = (int)m_param.text_data_param.batch_size;
            int nT = (int)m_param.text_data_param.time_steps;
            int nTopSize = m_param.top.Count;
            List<int> rgTopShape = new List<int>() { nT, nBatchSize, 1, 1 };
            int nTopIdx = 0;
            
            // Reshape the decoder input.
            colTop[nTopIdx].Reshape(1, nBatchSize, 1, 1);
            nTopIdx++;

            // Reshape the decoder clip.
            colTop[nTopIdx].Reshape(1, nBatchSize, 1, 1);
            nTopIdx++;

            // Reshape the decoder target.
            colTop[nTopIdx].Reshape(1, nBatchSize, 1, 1);
            nTopIdx++;

            // Reshape the encoder data | data reverse.
            if (m_param.text_data_param.enable_normal_encoder_output || m_param.text_data_param.enable_reverse_encoder_output)
            {
                colTop[nTopIdx].Reshape(rgTopShape);
                nTopIdx++;
            }

            // Reshape the encoder data reverse.
            if (m_param.text_data_param.enable_normal_encoder_output && m_param.text_data_param.enable_reverse_encoder_output)
            {
                colTop[nTopIdx].Reshape(rgTopShape);
                nTopIdx++;
            }

            // Reshape the encoder clip for attention.
            colTop[nTopIdx].Reshape(rgTopShape);
            nTopIdx++;
        }

        /// <summary>
        /// Run the Forward computation, which fills the data into the top (output) Blobs.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the data outputs.  
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nBatch = (int)m_param.text_data_param.batch_size;
            int nT = (int)m_param.text_data_param.time_steps;

            Array.Clear(m_rgDecInput, 0, m_rgDecInput.Length);
            Array.Clear(m_rgDecTarget, 0, m_rgDecTarget.Length);
            Array.Clear(m_rgDecClip, 0, m_rgDecClip.Length);
            Array.Clear(m_rgEncInput1, 0, m_rgEncInput1.Length);
            Array.Clear(m_rgEncInput2, 0, m_rgEncInput2.Length);
            Array.Clear(m_rgEncClip, 0, m_rgEncClip.Length);

            for (int i = 0; i < nBatch; i++)
            {
                while (Skip())
                    Next();

                Next();

                int nIdx = i * nT;

                for (int j = 0; j < m_currentData.EncoderInput.Count; j++)
                {
                    m_rgEncInput1[nIdx + j] = m_currentData.EncoderInput[j];
                    m_rgEncInput2[nIdx + j] = m_currentData.EncoderInputReverse[j];
                    m_rgEncClip[nIdx + j] = (j == 0) ? 0 : 1;
                }

                m_rgDecClip[i] = m_currentData.DecoderClip;
                m_rgDecInput[i] = m_currentData.DecoderInput;
                m_rgDecTarget[i] = m_currentData.DecoderTarget;
            }

            int nTopIdx = 0;

            colTop[nTopIdx].mutable_cpu_data = convert(m_rgDecInput);
            nTopIdx++;

            colTop[nTopIdx].mutable_cpu_data = convert(m_rgDecClip);
            nTopIdx++;

            colTop[nTopIdx].mutable_cpu_data = convert(m_rgDecTarget);
            nTopIdx++;

            if (m_param.text_data_param.enable_normal_encoder_output)
            {
                colTop[nTopIdx].mutable_cpu_data = convert(m_rgEncInput1);
                nTopIdx++;
            }

            if (m_param.text_data_param.enable_normal_encoder_output)
            {
                colTop[nTopIdx].mutable_cpu_data = convert(m_rgEncInput2);
                nTopIdx++;
            }

            colTop[nTopIdx].mutable_cpu_data = convert(m_rgEncClip);
        }

        /// @brief Not implemented - data Layers do not perform backward..
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }
    }


#pragma warning disable 1591

    class Data /** @private */
    {
        Random m_random = new Random((int)DateTime.Now.Ticks);
        List<List<string>> m_rgInput;
        List<List<string>> m_rgOutput;
        int m_nCurrentSequence = -1;
        int m_nCurrentOutputIdx = 0;
        int m_nIxInput = 1;
        int m_nIterations = 0;
        int m_nOutputCount = 0;
        Vocabulary m_vocab;

        public Data(List<List<string>> rgInput, List<List<string>> rgOutput, Vocabulary vocab)
        {
            m_vocab = vocab;
            m_rgInput = rgInput;
            m_rgOutput = rgOutput;
        }

        public Vocabulary Vocabulary
        {
            get { return m_vocab; }
        }

        public int VocabularyCount
        {
            get { return m_vocab.VocabularCount; }
        }

        public Tuple<List<int>, int> GetInputData()
        {
            List<int> rgInput = new List<int>();
            foreach (string str in m_rgInput[0])
            {
                rgInput.Add(m_vocab.WordToIndex(str));
            }

            return new Tuple<List<int>, int>(rgInput, 1);
        }

        public DataItem GetNextData(bool bShuffle)
        {
            int nDecClip = 1;

            bool bNewSequence = false;
            bool bNewEpoch = false;

            if (m_nCurrentSequence == -1)
            {
                m_nIterations++;
                bNewSequence = true;

                if (bShuffle)
                {
                    m_nCurrentSequence = m_random.Next(m_rgInput.Count);
                }
                else
                {
                    m_nCurrentSequence++;
                    if (m_nCurrentSequence == m_rgOutput.Count)
                        m_nCurrentSequence = 0;
                }

                m_nOutputCount = m_rgOutput[m_nCurrentSequence].Count;
                nDecClip = 0;

                if (m_nIterations == m_rgOutput.Count)
                {
                    bNewEpoch = true;
                    m_nIterations = 0;
                }
            }

            List<string> rgstrInput = m_rgInput[m_nCurrentSequence];
            List<int> rgInput = new List<int>();
            foreach (string str in rgstrInput)
            {
                rgInput.Add(m_vocab.WordToIndex(str));
            }

            int nIxTarget = 0;

            if (m_nCurrentOutputIdx < m_rgOutput[m_nCurrentSequence].Count)
            {
                string strTarget = m_rgOutput[m_nCurrentSequence][m_nCurrentOutputIdx];
                nIxTarget = m_vocab.WordToIndex(strTarget);
            }

            DataItem data = new DataItem(rgInput, m_nIxInput, nIxTarget, nDecClip, bNewEpoch, bNewSequence, m_nOutputCount);
            m_nIxInput = nIxTarget;

            m_nCurrentOutputIdx++;

            if (m_nCurrentOutputIdx == m_rgOutput[m_nCurrentSequence].Count)
            {
                m_nCurrentSequence = -1;
                m_nCurrentOutputIdx = 0;
                m_nIxInput = 1;
            }

            return data;
        }
    }

    class DataItem /** @private */
    {
        bool m_bNewEpoch;
        bool m_bNewSequence;
        int m_nOutputCount;
        List<int> m_rgInput;
        List<int> m_rgInputReverse;
        int m_nIxInput;
        int m_nIxTarget;
        int m_nDecClip;

        public DataItem(List<int> rgInput, int nIxInput, int nIxTarget, int nDecClip, bool bNewEpoch, bool bNewSequence, int nOutputCount)
        {
            m_rgInput = rgInput;
            m_nIxInput = nIxInput;
            m_nIxTarget = nIxTarget;
            m_nDecClip = nDecClip;
            m_bNewEpoch = bNewEpoch;
            m_bNewSequence = bNewSequence;
            m_nOutputCount = nOutputCount;
            m_rgInputReverse = new List<int>();

            for (int i = rgInput.Count - 1; i >= 0; i--)
            {
                m_rgInputReverse.Add(rgInput[i]);
            }
        }

        public List<int> EncoderInput
        {
            get { return m_rgInput; }
        }

        public List<int> EncoderInputReverse
        {
            get { return m_rgInputReverse; }
        }

        public int DecoderInput
        {
            get { return m_nIxInput; }
        }

        public int DecoderTarget
        {
            get { return m_nIxTarget; }
        }

        public int DecoderClip
        {
            get { return m_nDecClip; }
        }

        public bool NewEpoch
        {
            get { return m_bNewEpoch; }
        }

        public bool NewSequence
        {
            get { return m_bNewSequence; }
        }

        public int OutputCount
        {
            get { return m_nOutputCount; }
        }
    }

#pragma warning restore 1591 

    /// <summary>
    /// The Vocabulary object manages the overall word dictionary and word to index and index to word mappings.
    /// </summary>
    public class Vocabulary 
    {
        Dictionary<string, int> m_rgDictionary = new Dictionary<string, int>();
        Dictionary<string, int> m_rgWordToIndex = new Dictionary<string, int>();
        Dictionary<int, string> m_rgIndexToWord = new Dictionary<int, string>();
        List<string> m_rgstrVocabulary = new List<string>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public Vocabulary()
        {
        }

        /// <summary>
        /// The WordToIndex method maps a word to its corresponding index value.
        /// </summary>
        /// <param name="strWord">Specifies the word to map.</param>
        /// <returns>The word index is returned.</returns>
        public int WordToIndex(string strWord)
        {
            if (!m_rgWordToIndex.ContainsKey(strWord))
                throw new Exception("I do not know the word '" + strWord + "'!");

            return m_rgWordToIndex[strWord];
        }

        /// <summary>
        /// The IndexToWord method maps an index value to its corresponding word.
        /// </summary>
        /// <param name="nIdx">Specifies the index value.</param>
        /// <returns>The word corresponding to the index is returned.</returns>
        public string IndexToWord(int nIdx)
        {
            if (!m_rgIndexToWord.ContainsKey(nIdx))
                return "";

            return m_rgIndexToWord[nIdx];
        }

        /// <summary>
        /// Returns the number of words in the vocabulary.
        /// </summary>
        public int VocabularCount
        {
            get { return m_rgstrVocabulary.Count; }
        }

        /// <summary>
        /// Loads the word to index mappings.
        /// </summary>
        /// <param name="rgrgstrInput">Specifies the input sentences where each inner array is one sentence of words.</param>
        /// <param name="rgrgstrTarget">Specifies the target sentences where each inner array is one sentence of words.</param>
        public void Load(List<List<string>> rgrgstrInput, List<List<string>> rgrgstrTarget)
        {
            m_rgDictionary = new Dictionary<string, int>();

            // Count up all words.
            for (int i = 0; i < rgrgstrInput.Count; i++)
            {
                for (int j = 0; j < rgrgstrInput[i].Count; j++)
                {
                    string strWord = rgrgstrInput[i][j];

                    if (!m_rgDictionary.ContainsKey(strWord))
                        m_rgDictionary.Add(strWord, 1);
                    else
                        m_rgDictionary[strWord]++;
                }

                for (int j = 0; j < rgrgstrTarget[i].Count; j++)
                {
                    string strWord = rgrgstrTarget[i][j];

                    if (!m_rgDictionary.ContainsKey(strWord))
                        m_rgDictionary.Add(strWord, 1);
                    else
                        m_rgDictionary[strWord]++;
                }
            }

            // NOTE: Start at one to save room for START and END tokens where
            // START = 0 in the model word vectors and 
            // END = 0 in the next word softmax.
            int nIdx = 2;
            foreach (KeyValuePair<string, int> kv in m_rgDictionary)
            {
                if (kv.Value > 0)
                {
                    // Add word to vocabulary.
                    m_rgWordToIndex[kv.Key] = nIdx;
                    m_rgIndexToWord[nIdx] = kv.Key;
                    m_rgstrVocabulary.Add(kv.Key);
                    nIdx++;
                }
            }
        }
    }
}
