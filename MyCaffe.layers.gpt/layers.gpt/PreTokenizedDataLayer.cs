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
using MyCaffe.layers.gpt.layers.gpt;
using System.IO.MemoryMappedFiles;

namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The PreTokenizedDataLayer loads pre-tokenized data where the x and y data are offset by one token.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class PreTokenizedDataLayer<T> : Layer<T>
    {
        CancelEvent m_evtCancel;
        Random m_random;
        Layer<T> m_softmax = null;
        IVocabulary m_ivocab = new VocabularyBytePairEncoding();
        Dictionary<int, Tuple<string, string>> m_rgFileShards = new Dictionary<int, Tuple<string, string>>();
        List<int> m_rgShardIdx;
        List<int> m_rgBatchIdx = new List<int>();
        short[] m_rgDataRaw = null;
        short[] m_rgLabelRaw = null;
        float[] m_rgData = null;
        float[] m_rgLabel = null;
        Blob<T> m_blobX = null;
        Blob<T> m_blobY = null;

        /// <summary>
        /// The PreTokenizedDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Provides PreTokenizedDataParameter model_data_param with options:
        ///  - source.  The path to the *.ibin pre-tokenized data and label files reside.
        ///  - batch_size.  The batch size.
        ///  - time_steps.  The maximum number of time steps.
        ///  - shuffle.  Whether or not to shuffle the data.
        /// </param>
        /// <param name="db">Specifies the external database to use.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel any pre-fetching operations.</param>
        public PreTokenizedDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXDatabaseBase db, CancelEvent evtCancel)
            : base(cuda, log, p)
        {
            m_evtCancel = evtCancel;
            m_type = LayerParameter.LayerType.PRETOKENIZED_DATA;

            if (p.pretokenized_data_param.seed.HasValue)
                m_random = new Random(p.pretokenized_data_param.seed.Value);
            else
                m_random = new Random();

            m_blobX = new Blob<T>(cuda, log, false);
            m_blobY = new Blob<T>(cuda, log, false);
        }

        /// <summary>
        /// Release all internal blobs.
        /// </summary>
        protected override void dispose()
        {
            dispose(ref m_blobX);
            dispose(ref m_blobY);
            dispose(ref m_softmax);

            base.dispose();
        }

        /// <summary>
        /// No bottom blobs for the data layer, except when running.
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 0; }
        }

        /// <summary>
        /// Returns the minimum number of bottom blobs.
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 0; }
        }

        /// <summary>
        /// Returns the maximum number of required top (output) Blobs: data, target
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 2; }
        }

        private short[] readFile(string strFile)
        {
            FileInfo fi = new FileInfo(strFile);
            long lCount = fi.Length / sizeof(float);
            short[] rg = new short[lCount];

            using (var mmf = MemoryMappedFile.CreateFromFile(strFile, FileMode.Open))
            {
                using (var accessor = mmf.CreateViewAccessor(0, fi.Length, MemoryMappedFileAccess.Read))
                {
                    accessor.ReadArray<short>(0, rg, 0, (int)lCount);
                }
            }

            return rg;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            string[] rgstrFileShards = Directory.GetFiles(m_param.pretokenized_data_param.source, "*.ibin");
            List<string> rgData = new List<string>();
            List<string> rgLabels = new List<string>();

            rgData = rgstrFileShards.Where(p => p.Contains("\\data")).ToList();
            rgLabels = rgstrFileShards.Where(p => p.Contains("\\label")).ToList();

            if (rgData.Count != rgLabels.Count)
                throw new Exception("The number of data and label files must match.");

            for (int i = 0; i < rgData.Count; i++)
            {
                string strData = Path.GetFileNameWithoutExtension(rgData[i]);
                string strLabel = Path.GetFileNameWithoutExtension(rgLabels[i]);

                if (strData.StartsWith("data") && strLabel.StartsWith("labels"))
                {
                    strData = strData.Substring(4);
                    strLabel = strLabel.Substring(6);
                    int nDataId = int.Parse(strData);
                    int nLabelId = int.Parse(strLabel);

                    if (nDataId == nLabelId)
                        m_rgFileShards.Add(nDataId, new Tuple<string, string>(rgData[i], rgLabels[i]));
                }
            }

            if (m_rgFileShards.Count == 0)
                throw new Exception("No data and label files found.  Make sure file names are of the form 'data##.ibin' and 'label##.ibin'.");

            m_rgShardIdx = m_rgFileShards.Keys.ToList();

            Reshape(colBottom, colTop);
       }

        /// <summary>
        /// Reshape the top based on the parameter batch and block size.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs - Used only during RUN phase.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nBatchSize = (int)m_param.pretokenized_data_param.batch_size;
            int nBlockSize = (int)m_param.pretokenized_data_param.block_size;

            colTop[0].Reshape(nBatchSize, nBlockSize, 1, 1);
            colTop[1].Reshape(nBatchSize, nBlockSize, 1, 1);
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
            int nBatchSize = (int)m_param.pretokenized_data_param.batch_size;

            for (int i = 0; i < nBatchSize; i++)
            {
                int nIdx = 0;
                int nItemIdx = 0;

                if (m_rgBatchIdx.Count == 0)
                {
                    nIdx = 0;
                    if (m_param.pretokenized_data_param.shuffle)
                    {
                        nItemIdx = m_random.Next(0, m_rgShardIdx.Count);
                        nIdx = m_rgShardIdx[nItemIdx];
                        m_rgShardIdx.RemoveAt(nItemIdx);

                        if (m_rgShardIdx.Count == 0)
                            m_rgShardIdx = m_rgFileShards.Keys.ToList();
                    }

                    Tuple<string, string> shard = m_rgFileShards[nIdx];
                    FileInfo fi = new FileInfo(shard.Item1);

                    m_rgDataRaw = readFile(shard.Item1);
                    m_rgLabelRaw = readFile(shard.Item2);

                    int nNumBatches = (m_rgDataRaw.Length / (int)m_param.pretokenized_data_param.block_size) - 1;
                    Utility.LoadSequence(m_rgBatchIdx, 0, nNumBatches - 1);
                }

                nItemIdx = 0;
                if (m_param.pretokenized_data_param.shuffle)
                    nItemIdx = m_random.Next(0, m_rgBatchIdx.Count);

                nIdx = m_rgBatchIdx[nItemIdx];
                m_rgBatchIdx.RemoveAt(nItemIdx);

                int nStart = nIdx * (int)m_param.pretokenized_data_param.block_size;
                int nEnd = nStart + (int)m_param.pretokenized_data_param.block_size + 1;
                int nCount = (nEnd - nStart) - 1;

                if (m_rgData == null)
                    m_rgData = new float[nCount * nBatchSize];
                if (m_rgLabel == null)
                    m_rgLabel = new float[nCount * nBatchSize];

                Array.Copy(m_rgDataRaw, nStart, m_rgData, i * nCount, nCount);
                Array.Copy(m_rgLabelRaw, nStart+1, m_rgLabel, i * nCount, nCount);
            }

            colTop[0].mutable_cpu_data = convert(m_rgData);
            colTop[1].mutable_cpu_data = convert(m_rgLabel);
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
            get { return false; }
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
        /// <param name="bAddBos">Add the begin of sequence token.</param>
        /// <param name="bAddEos">Add the end of sequence token.</param>
        /// <returns>A list of tokens corresponding to the input is returned.</returns>
        public List<int> Tokenize(string str, bool bAddBos, bool bAddEos)
        {
            return null;
        }

        /// <summary>
        /// Detokenize a set of tokens from the data specified.
        /// </summary>
        /// <param name="rg">Specifies an array of tokens.</param>
        /// <param name="nStartIdx">Specifies the start index.</param>
        /// <param name="nCount">Specifies the number of tokens to detokenize.</param>
        /// <param name="bIgnoreBos">Specifies to ignore the BOS token.</param>
        /// <param name="bIgnoreEos">Specifies to ignore the EOS token.</param>
        /// <param name="nPadToken">Optionally, specifies a pad token to ignore (default = null).</param>
        /// <returns>The detokenized string is returned.</returns>
        public string Detokenize(float[] rg, int nStartIdx, int nCount, bool bIgnoreBos = true, bool bIgnoreEos = true, int? nPadToken = null)
        {
            return m_ivocab.Detokenize(rg, bIgnoreBos, bIgnoreEos, nStartIdx, nCount, nPadToken);
        }

        /// <summary>
        /// Preproces the input and return as a set of bottom blobs.
        /// </summary>
        /// <param name="customInput">Specifies the custom text input.</param>
        /// <param name="nSeqLen">Specifies the sequence length.</param>
        /// <param name="colBottom">The output is placed in the bottom blobs as: tokidx, pos</param>
        /// <returns>The bottom blob collection is returned.</returns>
        public override BlobCollection<T> PreProcessInput(PropertySet customInput, out int nSeqLen, BlobCollection<T> colBottom = null)
        {
            nSeqLen = 0;
            return null;
        }

        /// <summary>
        /// Preproces the input and return as a set of bottom blobs.
        /// </summary>
        /// <param name="str">Specifies the string input, can be null.</param>
        /// <param name="nTokIdx">Specifies the token input.</param>
        /// <param name="colBottom">The output is placed in the bottom blobs as: tokidx, pos</param>
        /// <returns>The bottom blob collection is returned.</returns>
        public override bool PreProcessInput(string str, int? nTokIdx, BlobCollection<T> colBottom = null)
        {
            return false;
        }

        /// <summary>
        /// Allows post processing the logits output data by converting the logits to and selecting 
        /// from the probability distribution produced and detokenizing the results to the string character.
        /// </summary>
        /// <param name="nCurIdx">Specifies the current index being processed, or -1 for the last index.</param>
        /// <param name="blobLogits">Specifies the output of the last inner product layer.</param>
        /// <param name="softmax">Specifies the softmax layer.</param>
        /// <param name="nAxis">Specifies the axis of the softmax layer.</param>
        /// <param name="nK">Specifies the TopK max items of the logits to use, or 0 to ignore.</param>
        /// <param name="bSkipDetokenize">Specifies to skip detokenizing.</param>
        /// <returns>
        /// The detokenized data is returned.
        /// </returns>
        public override List<Tuple<string, int, double>> PostProcessLogitsOutput(int nCurIdx, Blob<T> blobLogits, Layer<T> softmax, int nAxis, int nK = 1, bool bSkipDetokenize = false)
        {
            float[] rgData = convertF(blobLogits.mutable_cpu_data);
            int nVocabCount = blobLogits.count(nAxis);
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
            if (softmax == null)
            {
                if (m_softmax == null)
                {
                    LayerParameter softmax_param = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
                    softmax_param.softmax_param.axis = nAxis;
                    m_softmax = Layer<T>.Create(m_cuda, m_log, softmax_param, null);
                    m_softmax.Setup(colBottom, colTop);
                }

                softmax = m_softmax;
            }
            softmax.Forward(colBottom, colTop);

            float[] rgProb = convertF(m_blobY.mutable_cpu_data);
            int nTokenId = (m_param.tokenized_data_param.sample_method == TokenizedDataParameter.SAMPLE_METHOD.PROBABILITY) ? sample(rgProb) : argmax(rgProb);

            string str = "";

            if (!bSkipDetokenize)
                str += m_ivocab.Detokenize(nTokenId, true, true);

            return new List<Tuple<string, int, double>>() { new Tuple<string, int, double>(str, nTokenId, 0) };
        }

        /// <summary>
        /// Detokenize a set of tokens.
        /// </summary>
        /// <param name="rgTokenIds">Specifies the tokens to detokenize into a string.</param>
        /// <returns>The detokenized string is returned.</returns>
        public string Detokenize(List<int> rgTokenIds)
        {
            float[] rgf = rgTokenIds.Select(p => (float)p).ToArray();
            return m_ivocab.Detokenize(rgf, true, true);
        }

        private int argmax(float[] rgData)
        {
            int nMaxIdx = 0;
            float fMax = rgData[0];

            for (int i = 1; i < rgData.Length; i++)
            {
                if (rgData[i] > fMax)
                {
                    fMax = rgData[i];
                    nMaxIdx = i;
                }
            }

            return nMaxIdx;
        }

        private int sample(float[] rgData)
        {
            float fTotal = 0;
            float fRand = (float)m_random.NextDouble();

            for (int i = 0; i < rgData.Length; i++)
            {
                fTotal += rgData[i];

                if (fTotal >= fRand)
                    return i;
            }

            return rgData.Length - 1;
        }

        /// <summary>
        /// Returns true if the token is an end of sequence token (EOS).
        /// </summary>
        /// <param name="nToken">Specifies the token to test of EOS.</param>
        /// <returns>If the token is an EOS token, true is returned.</returns>
        public bool IsEOS(int nToken)
        {
            return m_ivocab.EOS == nToken;
        }
    }
}
