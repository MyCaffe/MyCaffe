using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using System.IO;
using MyCaffe.layers.beta.TextData;

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
        /// The OnGetTrainingData is called during each forward pass after getting the training data for the pass.
        /// </summary>
        public event EventHandler<OnGetDataArgs> OnGetData;

        /// <summary>
        /// The TextDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Provides TextDataParameter text_data_param with options:
        ///  - encoder_source.  The encoder data source.
        ///  
        ///  - decoder_source.  The decoder data source.
        ///  
        ///  - batch_size.  The batch size (currently only 1 supported).
        ///  
        ///  - time_steps.  The maximum number of time steps.
        ///  
        ///  - sample_size.  The number of samples to load for training.
        ///  
        ///  - shuffle.  Whether or not to shuffle the data.
        ///  
        ///  - enable_normal_encoder_output.  Whether or not to enable the encoder input with normal ordering.
        ///  
        ///  - enable_reverse_encoder_output.  Whether or not to enable the encoder input with reverse ordering.
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
        /// When running in TRAIN or TEST phase, returns 0 for data layers have no bottom (input) Blobs.
        /// When running in RUN phase, returns 3 Blobs: dec_input, enc_input | enc_inputr, enc_clip.
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return (m_phase == Phase.RUN) ? 3 : 0; }
        }

        /// <summary>
        /// When running in TRAIN or TEST phase, returns 0 for data layers have no bottom (input) Blobs.
        /// When running in RUN phase, returns 4 Blobs: dec_input, enc_input, enc_inputr, enc_clip.
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return (m_phase == Phase.RUN) ? 4 : 0; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: dec, dclip, enc, eclip, vocabcount, label (only valid on TRAIN or TEST)
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 6; }
        }

        /// <summary>
        /// Returns the maximum number of required top (output) Blobs: dec, dclip, enc, encr, eclip, vocabcount, label (only valid on TRAIN or TEST)
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

        /// <summary>
        /// Returns information on the current iteration.
        /// </summary>
        public IterationInfo IterationInfo
        {
            get { return (m_currentData == null) ? new IterationInfo(true, true, 0) : m_currentData.IterationInfo; }
        }

        private static string clean(string str)
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

        /// <summary>
        /// Should return true when pre processing methods are overriden.
        /// </summary>
        public override bool SupportsPreProcessing
        {
            get { return true; }
        }

        /// <summary>
        /// Should return true when pre postprocessing methods are overriden.
        /// </summary>
        public override bool SupportsPostProcessing
        {
            get { return true; }
        }

        private static List<string> preprocess(string str, int nMaxLen = 0)
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
                List<string> rgstrInput1 = preprocess(rgstrInput[i]);
                List<string> rgstrTarget1 = preprocess(rgstrTarget[i]);

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
        /// The PreprocessInput allows derivative data layers to convert a property set of input
        /// data into the bottom blob collection used as intput.
        /// </summary>
        /// <param name="customInput">Specifies the custom input data.</param>
        /// <param name="colBottom">Optionally, specifies the bottom data to fill.</param>
        /// <returns>The bottom data is returned.</returns>
        /// <remarks>The blobs returned should match the blob descriptions returned in the LayerParameter's
        /// overrides for 'PrepareRunModelInputs' and 'PrepareRunModel'.</remarks>
        public override BlobCollection<T> PreProcessInput(PropertySet customInput, BlobCollection<T> colBottom = null)
        {
            if (colBottom == null)
            {
                string strInput = m_param.PrepareRunModelInputs();
                RawProto proto = RawProto.Parse(strInput);
                Dictionary<string, BlobShape> rgInput = NetParameter.InputFromProto(proto);
                colBottom = new BlobCollection<T>();

                foreach (KeyValuePair<string, BlobShape> kv in rgInput)
                {
                    Blob<T> blob = new Blob<T>(m_cuda, m_log);
                    blob.Name = kv.Key;
                    blob.Reshape(kv.Value);
                    colBottom.Add(blob);
                }
            }

            string strEncInput = customInput.GetProperty("InputData");
            if (strEncInput == null)
                throw new Exception("Could not find the expected input property 'InputData'!");

            PreProcessInput(strEncInput, null, colBottom);

            return colBottom;
        }

        /// <summary>
        /// Preprocess the input data for the RUN phase.
        /// </summary>
        /// <param name="strEncInput">Specifies the encoder input.</param>
        /// <param name="nDecInput">Specifies the decoder input.</param>
        /// <param name="colBottom">Specifies the bottom blob where the preprocessed data is placed where
        /// colBottom[0] contains the preprocessed decoder input.</param>
        /// colBottom[1] contains the preprocessed encoder input (depending on param settings),
        /// colBottom[2] contains the preprocessed encoder input reversed (depending on param settings), 
        /// <remarks>
        /// NOTE: the LayerSetup must be called before preprocessing input, for during LayerSetup the vocabulary is loaded.
        /// </remarks>
        public override void PreProcessInput(string strEncInput, int? nDecInput, BlobCollection<T> colBottom)
        {
            List<string> rgstrInput = null;
            if (strEncInput != null)
                rgstrInput = preprocess(strEncInput);

            DataItem data = Data.GetInputData(m_vocab, rgstrInput, nDecInput);

            if (m_param.text_data_param.enable_normal_encoder_output && m_param.text_data_param.enable_reverse_encoder_output)
                m_log.CHECK_EQ(colBottom.Count, 4, "The bottom collection must have 3 items: dec_input, enc_input, enc_inputr, enc_clip");
            else
                m_log.CHECK_EQ(colBottom.Count, 3, "The bottom collection must have 3 items: dec_input, enc_input | enc_inputr, enc_clip");

            int nT = (int)m_param.text_data_param.time_steps;
            int nBtmIdx = 0;

            colBottom[nBtmIdx].Reshape(new List<int>() { 1, 1, 1 });
            nBtmIdx++;

            if (m_param.text_data_param.enable_normal_encoder_output)
            {
                colBottom[nBtmIdx].Reshape(new List<int>() { nT, 1, 1 });
                nBtmIdx++;
            }

            if (m_param.text_data_param.enable_reverse_encoder_output)
            {
                colBottom[nBtmIdx].Reshape(new List<int>() { nT, 1, 1 });
                nBtmIdx++;
            }

            colBottom[nBtmIdx].Reshape(new List<int>() { nT, 1 });

            float[] rgEncInput = null;
            float[] rgEncInputR = null;
            float[] rgEncClip = null;
            float[] rgDecInput = new float[1];

            if (data.EncoderInput != null)
            {
                rgEncInput = new float[nT];
                rgEncInputR = new float[nT];
                rgEncClip = new float[nT];

                for (int i = 0; i < nT && i < data.EncoderInput.Count; i++)
                {
                    rgEncInput[i] = data.EncoderInput[i];
                    rgEncInputR[i] = data.EncoderInputReverse[i];
                    rgEncClip[i] = (i == 0) ? 0 : 1;
                }
            }

            rgDecInput[0] = data.DecoderInput;

            nBtmIdx = 0;
            colBottom[nBtmIdx].mutable_cpu_data = convert(rgDecInput);
            nBtmIdx++;

            if (m_param.text_data_param.enable_normal_encoder_output)
            {
                if (rgEncInput != null)
                    colBottom[nBtmIdx].mutable_cpu_data = convert(rgEncInput);
                nBtmIdx++;
            }

            if (m_param.text_data_param.enable_reverse_encoder_output)
            {
                if (rgEncInputR != null)
                    colBottom[nBtmIdx].mutable_cpu_data = convert(rgEncInputR);
                nBtmIdx++;
            }

            if (rgEncClip != null)
                colBottom[nBtmIdx].mutable_cpu_data = convert(rgEncClip);
        }

        /// <summary>
        /// Convert the maximum index within the softmax into the word index, then convert
        /// the word index back into the word and return the word string.
        /// </summary>
        /// <param name="blobSoftmax">Specifies the softmax output.</param>
        /// <param name="nK">Optionally, specifies the K top items to return (default = 1).</param>
        /// <returns>The array of word string, index and probabilities corresponding to the softmax output is returned.</returns>
        public override List<Tuple<string, int, double>> PostProcessOutput(Blob<T> blobSoftmax, int nK = 1)
        {
            m_log.CHECK_EQ(blobSoftmax.channels, 1, "Currently, only batch size = 1 supported.");

            List<Tuple<string, int, double>> rgRes = new List<Tuple<string, int, double>>();

            long lPos;
            double dfProb = blobSoftmax.GetMaxData(out lPos);

            rgRes.Add(new Tuple<string, int, double>(m_vocab.IndexToWord((int)lPos), (int)lPos, dfProb));

            if (nK > 1)
            {
                m_cuda.copy(blobSoftmax.count(), blobSoftmax.gpu_data, blobSoftmax.mutable_gpu_diff);

                for (int i = 1; i < nK; i++)
                {
                    blobSoftmax.SetData(-1000000000, (int)lPos);
                    dfProb = blobSoftmax.GetMaxData(out lPos);

                    string strWord = m_vocab.IndexToWord((int)lPos);
                    if (strWord.Length > 0)
                        rgRes.Add(new Tuple<string, int, double>(strWord, (int)lPos, dfProb));
                }

                m_cuda.copy(blobSoftmax.count(), blobSoftmax.gpu_diff, blobSoftmax.mutable_gpu_data);
                blobSoftmax.SetDiff(0);
            }

            return rgRes;
        }

        /// <summary>
        /// Convert the index to the word.
        /// </summary>
        /// <param name="nIdx">Specifies the index to convert.</param>
        /// <returns>The corresponding word is returned.</returns>
        public override string PostProcessOutput(int nIdx)
        {
            return m_vocab.IndexToWord(nIdx);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Refuse transformation parameters since TextData is totally generic.
            if (m_param.transform_param != null)
                m_log.WriteLine("WARNING: " + m_type.ToString() + " does not transform data.");

            m_log.CHECK_EQ(m_param.text_data_param.batch_size, 1, "Currently, only batch_size = 1 supported.");

            if (m_param.text_data_param.enable_normal_encoder_output && m_param.text_data_param.enable_reverse_encoder_output)
                m_log.CHECK_EQ(colTop.Count, 7, "When normal and reverse encoder output used, there must be 7 tops: dec, dclip, enc, encr, eclip, vocabcount, dectgt (only valid on TEST | TRAIN)");
            else if (m_param.text_data_param.enable_normal_encoder_output || m_param.text_data_param.enable_reverse_encoder_output)
                m_log.CHECK_EQ(colTop.Count, 6, "When normal or reverse encoder output used, there must be 6 tops: dec, dclip, enc | encr, eclip, vocabcount, dectgt (only valid on TEST | TRAIN)");
            else
                m_log.FAIL("You must specify to enable either normal, reverse or both encoder inputs.");

            // Load the encoder and decoder input files into the Data and Vocabulary.
            PreProcessInputFiles(m_param.text_data_param);

            m_rgDecInput = new float[m_param.text_data_param.batch_size];
            m_rgDecClip = new float[m_param.text_data_param.batch_size];
            m_rgEncInput1 = new float[m_param.text_data_param.batch_size * m_param.text_data_param.time_steps];
            m_rgEncInput2 = new float[m_param.text_data_param.batch_size * m_param.text_data_param.time_steps];
            m_rgEncClip = new float[m_param.text_data_param.batch_size * m_param.text_data_param.time_steps];

            if (m_phase != Phase.RUN)
                m_rgDecTarget = new float[m_param.text_data_param.batch_size];

            reshape(colTop, true);
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
            reshape(colTop, false);
        }

        private void reshape(BlobCollection<T> colTop, bool bSetup)
        {
            int nBatchSize = (int)m_param.text_data_param.batch_size;
            int nT = (int)m_param.text_data_param.time_steps;
            List<int> rgTopShape = new List<int>() { nT, nBatchSize, 1 };
            int nTopIdx = 0;

            // Reshape the decoder input.
            if (!bSetup)
                colTop[nTopIdx].Reshape(new List<int>() { 1, nBatchSize, 1 });
            nTopIdx++;

            // Reshape the decoder clip.
            if (!bSetup)
                colTop[nTopIdx].Reshape(new List<int>() { 1, nBatchSize });
            nTopIdx++;

            // Reshape the encoder data | data reverse.
            if (m_param.text_data_param.enable_normal_encoder_output || m_param.text_data_param.enable_reverse_encoder_output)
            {
                if (!bSetup)
                    colTop[nTopIdx].Reshape(rgTopShape);
                nTopIdx++;
            }

            // Reshape the encoder data reverse.
            if (m_param.text_data_param.enable_normal_encoder_output && m_param.text_data_param.enable_reverse_encoder_output)
            {
                if (!bSetup)
                    colTop[nTopIdx].Reshape(rgTopShape);
                nTopIdx++;
            }

            // Reshape the encoder clip for attention.
            if (!bSetup)
                colTop[nTopIdx].Reshape(new List<int>() { nT, nBatchSize });
            nTopIdx++;

            // Reshape the vocab count.
            colTop[nTopIdx].Reshape(new List<int>() { 1 });
            if (bSetup)
                colTop[nTopIdx].SetData(m_vocab.VocabularCount + 2, 0);
            nTopIdx++;

            // Reshape the decoder target.
            if (!bSetup)
                colTop[nTopIdx].Reshape(new List<int>() { 1, nBatchSize, 1 });
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
            if (m_phase != Phase.RUN)
                Array.Clear(m_rgDecTarget, 0, m_rgDecTarget.Length);
            Array.Clear(m_rgDecClip, 0, m_rgDecClip.Length);
            Array.Clear(m_rgEncInput1, 0, m_rgEncInput1.Length);
            Array.Clear(m_rgEncInput2, 0, m_rgEncInput2.Length);
            Array.Clear(m_rgEncClip, 0, m_rgEncClip.Length);

            int nTopIdx = 0;

            if (m_phase != Phase.RUN)
            {
                for (int i = 0; i < nBatch; i++)
                {
                    while (Skip())
                        Next();

                    Next();

                    if (OnGetData != null)
                        OnGetData(this, new OnGetDataArgs(Vocabulary, IterationInfo));

                    int nIdx = i * nT;

                    for (int j = 0; j < nT && j < m_currentData.EncoderInput.Count; j++)
                    {
                        m_rgEncInput1[nIdx + j] = m_currentData.EncoderInput[j];
                        m_rgEncInput2[nIdx + j] = m_currentData.EncoderInputReverse[j];
                        m_rgEncClip[nIdx + j] = (j == 0) ? 0 : 1;
                    }

                    m_rgDecClip[i] = m_currentData.DecoderClip;
                    m_rgDecInput[i] = m_currentData.DecoderInput;
                    m_rgDecTarget[i] = m_currentData.DecoderTarget;
                }

                colTop[nTopIdx].mutable_cpu_data = convert(m_rgDecInput);
                nTopIdx++;

                colTop[nTopIdx].mutable_cpu_data = convert(m_rgDecClip);
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
                nTopIdx++;

                nTopIdx++; // vocab count.

                colTop[nTopIdx].mutable_cpu_data = convert(m_rgDecTarget);
                nTopIdx++;
            }
            else
            {
                int nBtmIdx = 0;
                float fDecInput = convertF(colBottom[nBtmIdx].GetData(0));
                if (fDecInput < 0)
                    fDecInput = 1;

                nBtmIdx++;

                // Decoder input.
                colTop[nTopIdx].SetData(fDecInput, 0);
                nTopIdx++;

                // Decoder clip.
                colTop[nTopIdx].SetData((fDecInput == 1) ? 0 : 1, 0);
                nTopIdx++;

                if (m_param.text_data_param.enable_normal_encoder_output)
                {
                    colTop[nTopIdx].CopyFrom(colBottom[nBtmIdx]);
                    nTopIdx++;
                    nBtmIdx++;
                }

                if (m_param.text_data_param.enable_reverse_encoder_output)
                {
                    colTop[nTopIdx].CopyFrom(colBottom[nBtmIdx]);
                    nTopIdx++;
                    nBtmIdx++;
                }

                // Encoder clip.
                colTop[nTopIdx].CopyFrom(colBottom[nBtmIdx]);
            }
        }

        /// @brief Not implemented - data Layers do not perform backward..
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }
    }


    namespace TextData
    {
#pragma warning disable 1591

        class Data /** @private */
        {
            Random m_random = new Random((int)DateTime.Now.Ticks);
            List<List<string>> m_rgInput;
            List<List<string>> m_rgOutput;
            int m_nCurrentSequence = -1;
            int m_nCurrentOutputIdx = 0;
            int m_nSequenceIdx = 0;
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

            public static DataItem GetInputData(Vocabulary vocab, List<string> rgstrInput, int? nDecInput = null)
            {
                List<int> rgInput = null;

                if (rgstrInput != null)
                {
                    rgInput = new List<int>();
                    foreach (string str in rgstrInput)
                    {
                        rgInput.Add(vocab.WordToIndex(str));
                    }
                }

                int nClip = 1;

                if (!nDecInput.HasValue)
                {
                    nClip = 0;
                    nDecInput = 1;
                }

                return new DataItem(rgInput, nDecInput.Value, -1, nClip, false, true, 0);
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
                        m_nCurrentSequence = m_nSequenceIdx;
                        m_nSequenceIdx++;
                        if (m_nSequenceIdx == m_rgOutput.Count)
                            m_nSequenceIdx = 0;
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
            IterationInfo m_iter;
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
                m_iter = new IterationInfo(bNewEpoch, bNewSequence, nOutputCount);
                m_rgInputReverse = new List<int>();

                if (rgInput != null)
                {
                    for (int i = rgInput.Count - 1; i >= 0; i--)
                    {
                        m_rgInputReverse.Add(rgInput[i]);
                    }
                }
                else
                {
                    m_rgInputReverse = null;
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

            public IterationInfo IterationInfo
            {
                get { return m_iter; }
            }
        }

#pragma warning restore 1591

        /// <summary>
        /// The IterationInfo class contains information about each iteration.
        /// </summary>
        public class IterationInfo
        {
            bool m_bNewEpoch;
            bool m_bNewSequence;
            int m_nOutputCount;

            /// <summary>
            /// The constructor.
            /// </summary>
            /// <param name="bNewEpoch">Specifies whether or not the current iteration is in a new epoch.</param>
            /// <param name="bNewSequence">Specifies whether or not the current iteration is in a new sequence.</param>
            /// <param name="nOutputCount">Specifies the output count of the current sequence.</param>
            public IterationInfo(bool bNewEpoch, bool bNewSequence, int nOutputCount)
            {
                m_bNewEpoch = bNewEpoch;
                m_bNewSequence = bNewSequence;
                m_nOutputCount = nOutputCount;
            }

            /// <summary>
            /// Returns whether or not the current iteration is in a new epoch.
            /// </summary>
            public bool NewEpoch
            {
                get { return m_bNewEpoch; }
            }

            /// <summary>
            /// Returns whether or not the current iteration is in a new sequence.
            /// </summary>
            public bool NewSequence
            {
                get { return m_bNewSequence; }
            }

            /// <summary>
            /// Returns the output count of the current sequence.
            /// </summary>
            public int OutputCount
            {
                get { return m_nOutputCount; }
            }
        }

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

        /// <summary>
        /// Defines the arguments passed to the OnGetData event.
        /// </summary>
        public class OnGetDataArgs : EventArgs
        {
            Vocabulary m_vocab;
            IterationInfo m_iter;

            /// <summary>
            /// The constructor.
            /// </summary>
            /// <param name="vocab">Specifies the vocabulary.</param>
            /// <param name="iter">Specifies the iteration info.</param>
            public OnGetDataArgs(Vocabulary vocab, IterationInfo iter)
            {
                m_vocab = vocab;
                m_iter = iter;
            }

            /// <summary>
            /// Returns the vocabulary.
            /// </summary>
            public Vocabulary Vocabulary
            {
                get { return m_vocab; }
            }

            /// <summary>
            /// Returns the iteration information.
            /// </summary>
            public IterationInfo IterationInfo
            {
                get { return m_iter; }
            }
        }
    }
}
