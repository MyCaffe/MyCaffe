using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.layers.python.layers.python
{
    /// <summary>
    /// The PythonLayer provides a wrapper for the TokenizedDataPairsLayer layer implemented in Python.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TokenizedDataPairsLayerPy<T> : Layer<T>
    {
        Random m_random;
        long m_lNum;
        long m_lChannels;
        List<Tuple<float[], float[], float[]>> m_rgRawData = new List<Tuple<float[], float[], float[]>>();        
        float[] m_rgDataEnc = null;
        float[] m_rgDataDec = null;
        float[] m_rgDataTrg = null;
        Blob<T> m_blobTriangle;
        List<int> m_rgIdx = new List<int>();
        int m_nEpoch = 0;

        /// <summary>
        /// The TokenizedDataPairsLayerPy constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Provides TokenizedDataPairsParameter model_data_param with options:
        ///  - source.  The encoder input data source.
        /// 
        ///  - target.  The decoder input/output data source.
        /// 
        ///  - batch_size.  The batch size
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
        public TokenizedDataPairsLayerPy(CudaDnn<T> cuda, Log log, LayerParameter p, IXImageDatabaseBase db, CancelEvent evtCancel)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.TOKENIZED_DATA_PAIRS_PY;

            if (m_param.tokenized_data_pairs_param.seed.HasValue)
                m_random = new Random(m_param.tokenized_data_pairs_param.seed.Value);
            else
                m_random = new Random();

            m_blobTriangle = new Blob<T>(m_cuda, m_log);
        }

        /// <summary>
        /// Release all internal blobs.
        /// </summary>
        protected override void dispose()
        {
            dispose(ref m_blobTriangle);
            base.dispose();
        }

        /// <summary>
        /// Specifies the exact number of bottom blobs (TRAIN|TEST: 0, RUN:2 encin, decin)
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return (m_phase == Phase.RUN) ? 2 : 0; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: enc_in, dec_in, dec_out, e_mask, d_mask
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 5; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            loadCacheData();

            m_nEpoch = 0;
            loadIdx();
        }

        private int getIndex(string strFileName, out string strType)
        {
            string strFile = Path.GetFileName(strFileName);

            int nPos = strFile.IndexOf('_');
            if (nPos < 0)
                m_log.FAIL("Invalid file name '" + strFile + "' - expected 'num_type.npy' format!");

            string strNum = strFile.Substring(0, nPos);
            int nIdx;
            
            if (!int.TryParse(strNum, out nIdx))
                m_log.FAIL("Invalid file name '" + strFile + "' - expected 'num_type.npy' format!");

            strType = strFile.Substring(nPos + 1);

            return nIdx;
        }

        private void loadIdx()
        {
            m_rgIdx.Clear();

            for (int i = 0; i < m_rgRawData.Count; i++)
            {
                m_rgIdx.Add(i);
            }
        }

        private void loadCacheData()
        {
            Stopwatch sw = new Stopwatch();
            string strPath = m_param.tokenized_data_pairs_param.source;
            string[] rgstrFiles = Directory.GetFiles(strPath);

            sw.Start();
            Blob<T> blob = new Blob<T>(m_cuda, m_log);
            int nBatch = 0;
            int nChannels = 0;

            float[] rgEnc = null;
            float[] rgDec = null;
            float[] rgTrg = null;

            int nEncIdx = 0;
            int nDecIdx = 0;
            int nTrgIdx = 0;

            for (int i = 0; i < rgstrFiles.Length; i++)
            {
                string strFile = rgstrFiles[i];
                Tuple<float[], int[]> data = blob.LoadFromNumpy(strFile, false, true);
                int nCount = Utility.Count(data.Item2);

                if (nBatch == 0)
                    nBatch = data.Item2[0];
                else if (nBatch != data.Item2[0])
                    continue;

                if (nChannels == 0)
                    nChannels = data.Item2[1];
                else if (nChannels != data.Item2[1])
                    m_log.FAIL("Data size incorrect at index " + i.ToString() + ", file '" + strFile + "'!");

                string strType;
                int nIdx = getIndex(strFile, out strType);

                if (strType == "enc.npy")
                {
                    if (rgEnc != null)
                        m_log.FAIL("Mismatched files - duplicate enc file found at index " + nIdx.ToString() + ", file '" + strFile + "'!");                    
                    rgEnc = data.Item1;
                    nEncIdx = nIdx;
                }
                else if (strType == "dec.npy")
                {
                    if (rgDec != null)
                        m_log.FAIL("Mismatched files - duplicate dec file found at index " + nIdx.ToString() + ", file '" + strFile + "'!");
                    rgDec = data.Item1;
                    nDecIdx = nIdx;
                }
                else if (strType == "trg.npy")
                {
                    if (rgTrg != null)
                        m_log.FAIL("Mismatched files - duplicate trg file found at index " + nIdx.ToString() + ", file '" + strFile + "'!");
                    rgTrg = data.Item1;
                    nTrgIdx = nIdx;
                }

                if (rgEnc != null && rgDec != null && rgTrg != null)
                {
                    if (nEncIdx != nDecIdx || nEncIdx != nTrgIdx || nDecIdx != nTrgIdx)
                        m_log.FAIL("Mismatched files - indexes dont match at index " + nIdx.ToString() + "'!");

                    for (int j = 0; j < nBatch; j++)
                    {
                        float[] rgEncData = new float[nChannels];
                        float[] rgDecData = new float[nChannels];
                        float[] rgTrgData = new float[nChannels];

                        Array.Copy(rgEnc, j * nChannels, rgEncData, 0, nChannels);
                        Array.Copy(rgDec, j * nChannels, rgDecData, 0, nChannels);
                        Array.Copy(rgTrg, j * nChannels, rgTrgData, 0, nChannels);

                        m_rgRawData.Add(new Tuple<float[], float[], float[]>(rgEncData, rgDecData, rgTrgData));
                    }

                    rgEnc = null;
                    rgDec = null;
                    rgTrg = null;
                }

                if (sw.Elapsed.TotalMilliseconds > 1000)
                {
                    double dfPct = (double)i / (double)rgstrFiles.Length;
                    m_log.WriteLine("Loading raw data at " + dfPct.ToString("P3") + "...", true);
                    sw.Restart();
                }
            }

            m_lNum = nBatch;
            m_lChannels = nChannels;

            blob.Dispose();
        }

        /// <summary>
        /// Reshape the top based on the parameter batch and block size.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs - Used only during RUN phase.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(m_lChannels, m_param.tokenized_data_pairs_param.block_size, "The block size must match the data channels!");

            int nBatch = (int)m_param.tokenized_data_pairs_param.batch_size;
            int nBlk = (int)m_param.tokenized_data_pairs_param.block_size;
            int nCount = nBatch * nBlk;
            if (m_rgDataEnc == null || m_rgDataEnc.Length != nCount)
                m_rgDataEnc = new float[nCount];
            if (m_rgDataDec == null || m_rgDataDec.Length != nCount)
                m_rgDataDec = new float[nCount];
            if (m_rgDataTrg == null || m_rgDataTrg.Length != nCount)
                m_rgDataTrg = new float[nCount];

            List<int> rgShape = new List<int>()
            {
                (int)m_param.tokenized_data_pairs_param.batch_size,
                (int)m_param.tokenized_data_pairs_param.block_size
            };
            
            colTop[0].Reshape(rgShape); // B,L
            colTop[1].Reshape(rgShape); // B,L
            colTop[2].Reshape(rgShape); // B,L

            rgShape.Add(1);
            rgShape.Add(1);
            
            colTop[3].Reshape(rgShape); // B,L,1            

            rgShape[2] = rgShape[1];
            
            colTop[4].Reshape(rgShape); // B,L,L

            int nBlockSize = (int)m_param.tokenized_data_pairs_param.block_size;

            if (!m_blobTriangle.CompareShape(colTop[4].shape()))
            {
                m_blobTriangle.ReshapeLike(colTop[4]);

                T[] rgMask = new T[m_blobTriangle.count()];
                for (int n = 0; n < m_blobTriangle.num; n++)
                {
                    for (int c = 0; c < m_blobTriangle.channels; c++)
                    {
                        for (int h = 0; h < m_blobTriangle.height; h++)
                        {
                            int nIdx = n * nBlockSize * nBlockSize + c * nBlockSize + h;
                            rgMask[nIdx] = (h > c) ? m_tZero : m_tOne;
                        }
                    }
                }

                m_blobTriangle.mutable_cpu_data = rgMask;
            }
        }

        /// <summary>
        /// Run the Forward computation, which fills the data into the top (output) Blobs.
        /// </summary>
        /// <param name="colBottom">bottom input blob(s).</param>
        /// <param name="colTop">top output blob(s)</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nBlk = (int)m_param.tokenized_data_pairs_param.block_size;
            int nDim = nBlk * nBlk;

            for (int i = 0; i < m_param.tokenized_data_pairs_param.batch_size; i++)
            {
                int nIdx = m_random.Next((int)m_rgIdx.Count);
                int nDataIdx = m_rgIdx[nIdx];
                
                m_rgIdx.Remove(nIdx);
                if (m_rgIdx.Count == 0)
                {
                    m_nEpoch++;
                    m_log.WriteLine("EPOCH " + m_nEpoch.ToString(), true);
                    loadIdx();
                }

                Array.Copy(m_rgRawData[nDataIdx].Item1, 0, m_rgDataEnc, i * nBlk, nBlk);
                Array.Copy(m_rgRawData[nDataIdx].Item2, 0, m_rgDataDec, i * nBlk, nBlk);
                Array.Copy(m_rgRawData[nDataIdx].Item3, 0, m_rgDataTrg, i * nBlk, nBlk);
            }

            colTop[0].mutable_cpu_data = convert(m_rgDataEnc);
            colTop[1].mutable_cpu_data = convert(m_rgDataDec);
            colTop[2].mutable_cpu_data = convert(m_rgDataTrg);

            // Fill encoder mask based on encoder input.
            m_cuda.sign(colTop[0].count(), colTop[0].gpu_data, colTop[3].mutable_gpu_data);
            // Fill decoder mask based on decoder input.
            m_cuda.channel_duplicate(colTop[4].count(), colTop[1].num, colTop[1].channels, colTop[4].count(2), colTop[1].gpu_data, colTop[4].mutable_gpu_data);
            m_cuda.sign(colTop[4].count(), colTop[4].gpu_data, colTop[4].mutable_gpu_data);
            // Overlay triangular matrix on decoder mask.
            m_cuda.mul(colTop[4].count(), colTop[4].gpu_data, m_blobTriangle.gpu_data, colTop[4].mutable_gpu_data);
        }

        /// <summary>
        /// Run the Backward computation, which calculates the gradients.
        /// </summary>
        /// <param name="colBottom">bottom input blob(s).</param>
        /// <param name="rgbPropagateDown">Specifies whether or not to propagate backwards.</param>
        /// <param name="colTop">top output blob(s)</param>
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
        /// <param name="strVocab">Specifies the vocabulary to use, ENCODER or DECODER.</param>
        /// <returns>A list of tokens corresponding to the input is returned.</returns>
        public List<int> Tokenize(string str, string strVocab)
        {
            if (strVocab.ToLower() == "encoder")
                return null;
            else
                return null;
        }

        /// <summary>
        /// Detokenize a set of tokens from the data specified.
        /// </summary>
        /// <param name="rg">Specifies an array of tokens.</param>
        /// <param name="nStartIdx">Specifies the start index.</param>
        /// <param name="nCount">Specifies the number of tokens to detokenize.</param>
        /// <param name="strVocab">Specifies the vocabulary to use: ENCODER or DECODER.</param>
        /// <returns>The detokenized string is returned.</returns>
        public string Detokenize(float[] rg, int nStartIdx, int nCount, string strVocab)
        {
            return null;
        }

        /// <summary>
        /// Get the vocabulary size for the specified vocabulary source.
        /// </summary>
        /// <param name="strVocab">Specifies the vocabulary source (ENCODER or DECODER).</param>
        /// <returns>The vocabulary size is returned.</returns>
        public uint GetVocabuarySize(string strVocab)
        {
            return 1;
        }

        /// <summary>
        /// Preproces the input and return as a set of bottom blobs.
        /// </summary>
        /// <param name="customInput">Specifies the custom text input.</param>
        /// <param name="colBottom">The output is placed in the bottom blobs as: tokidx, pos</param>
        /// <returns>The bottom blob collection is returned.</returns>
        public override BlobCollection<T> PreProcessInput(PropertySet customInput, BlobCollection<T> colBottom = null)
        {
            return null;
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
        }

        /// <summary>
        /// Allows post processing the logits output data by converting the logits to and selecting 
        /// from the probability distribution produced and detokenizing the results to the string character.
        /// </summary>
        /// <param name="blobLogits">Specifies the output of the last inner product layer.</param>
        /// <param name="softmax">Specifies the softmax layer.</param>
        /// <param name="nAxis">Specifies the axis of the softmax layer.</param>
        /// <param name="nK">Specifies the TopK max items of the logits to use, or 0 to ignore.</param>
        /// <returns>
        /// The detokenized data is returned.
        /// </returns>
        public override List<Tuple<string, int, double>> PostProcessLogitsOutput(Blob<T> blobLogits, Layer<T> softmax, int nAxis, int nK = 1)
        {
            return null;
        }

        /// <summary>
        /// The PostProcessFullOutput allows derivative data layers to post-process the results, usually be detokenizing the data in the blobSoftmax.
        /// </summary>
        /// <param name="blobSoftmax">Specifies the data to be post processed.</param>
        /// <returns>A string of the post processed data is returned.</returns>
        public override string PostProcessFullOutput(Blob<T> blobSoftmax)
        {
            return null;
        }
    }
}
