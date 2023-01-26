using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.gpt;
using MyCaffe.python;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
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
        PythonInterop m_py;
        Random m_random;
        long m_lNum;
        long m_lChannels;
        float[] m_rgEnc;
        float[] m_rgDec;
        float[] m_rgTrg;
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
            string strPythonPath = getPythonPath(m_param.tokenized_data_pairs_param.python_param.python_path);

            m_type = LayerParameter.LayerType.TOKENIZED_DATA_PAIRS_PY;
            // Load the python runtime interop.
            m_py = new PythonInterop(strPythonPath);

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
            m_py.Dispose();
            base.dispose();
        }

        private string getUserName()
        {
            string strUserName = System.Security.Principal.WindowsIdentity.GetCurrent().Name;
            int nPos = strUserName.LastIndexOf('\\');
            if (nPos >= 0)
                strUserName = strUserName.Substring(nPos + 1);

            return strUserName;
        }

        private string getPythonPath(string strPath)
        {
            if (string.IsNullOrEmpty(strPath) || strPath == "$Default$")
            {
                string strUserName = getUserName();
                strPath = "C:\\Users\\" + strUserName + "\\AppData\\Local\\Programs\\Python\\Python39\\python39.dll";
            }

            if (!File.Exists(strPath))
                m_log.FAIL("Could not find Python 3.9 at '" + strPath + "'!");

            return strPath;
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
            string strPyCode = Properties.Resources.tokenizeddatapairs;

            string strSrc = m_param.tokenized_data_pairs_param.source;
            var uri = new System.Uri(strSrc);
            var strConverted = uri.AbsoluteUri;

            strSrc = strConverted.Substring(8);
            strSrc += "/";

            string strCache = strSrc + "cache/";

            string strEncFile = strCache + "train_enc.npy";
            string strDecFile = strCache + "train_dec.npy";
            string strTrgFile = strCache + "train_trg.npy";

            if (!File.Exists(strEncFile) || !File.Exists(strDecFile) || !File.Exists(strTrgFile))
            {
                m_log.WriteLine("WARNING: Generating the encoder, decoder and target files - this will take several minuts.", true);

                KeyValuePair<string, object>[] rgArg = new KeyValuePair<string, object>[]
                {
                    new KeyValuePair<string, object>("strDataDir", strSrc),
                    new KeyValuePair<string, object>("strCacheDir", strCache),
                    new KeyValuePair<string, object>("nLoadLimit", int.MaxValue),
                    new KeyValuePair<string, object>("bCacheOnly", true)
                };

                object obj = m_py.RunPythonCodeAndReturn(strPyCode, "res", rgArg);
                Dictionary<string, object> rgRes = m_py.ConvertToDictionary(obj);

                m_lNum = (long)rgRes["num"];
                m_lChannels = (long)rgRes["channels"];
                int nCount = (int)(m_lNum * m_lChannels);

                strEncFile = (string)rgRes["encfile"];
                strDecFile = (string)rgRes["decfile"];
                strTrgFile = (string)rgRes["trgfile"];
            }
            
            Blob<T> blob = new Blob<T>(m_cuda, m_log);

            m_log.WriteLine("Loading the cached encoder input file '" + strEncFile + "'", true);
            Tuple < float[], int[]> enc = blob.LoadFromNumpy(strEncFile, false, true);
            m_rgEnc = enc.Item1;
            m_log.WriteLine("Loading the cached decoder input file '" + strDecFile + "'", true);
            Tuple<float[], int[]> dec = blob.LoadFromNumpy(strDecFile, false, true);
            m_rgDec = dec.Item1;
            m_log.WriteLine("Loading the cached decoder target file '" + strTrgFile + "'", true);
            Tuple<float[], int[]> trg = blob.LoadFromNumpy(strTrgFile, false, true);
            m_rgTrg = trg.Item1;

            m_log.WriteLine("All cached files loaded.", true);

            m_lNum = enc.Item2[0];
            m_lChannels = enc.Item2[1];
            
            m_nEpoch = 0;
            loadIdx((int)m_lNum);
        }

        private void loadIdx(int nCount)
        {
            m_rgIdx.Clear();

            for (int i = 0; i < nCount; i++)
            {
                m_rgIdx.Add(i);
            }
        }

        /// <summary>
        /// Reshape the top based on the parameter batch and block size.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs - Used only during RUN phase.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(m_lChannels, m_param.tokenized_data_pairs_param.block_size, "The block size must match the data channels!");

            int nCount = (int)(m_param.tokenized_data_pairs_param.batch_size * m_param.tokenized_data_pairs_param.block_size);
            if (m_rgDataEnc == null || m_rgDataEnc.Length != nCount)
                m_rgDataEnc = new float[nCount];
            if (m_rgDataDec == null || m_rgDataDec.Length != nCount)
                m_rgDataDec = new float[nCount];
            if (m_rgDataTrg == null || m_rgDataTrg.Length != nCount)
                m_rgDataTrg = new float[nCount];

            colTop[0].Reshape((int)m_param.tokenized_data_pairs_param.batch_size, (int)m_param.tokenized_data_pairs_param.block_size, 1, 1);
            colTop[1].Reshape((int)m_param.tokenized_data_pairs_param.batch_size, (int)m_param.tokenized_data_pairs_param.block_size, 1, 1);
            colTop[2].Reshape((int)m_param.tokenized_data_pairs_param.batch_size, (int)m_param.tokenized_data_pairs_param.block_size, 1, 1);
            colTop[3].Reshape((int)m_param.tokenized_data_pairs_param.batch_size, (int)m_param.tokenized_data_pairs_param.block_size, 1, 1);
            colTop[4].Reshape((int)m_param.tokenized_data_pairs_param.batch_size, (int)m_param.tokenized_data_pairs_param.block_size, (int)m_param.tokenized_data_pairs_param.block_size, 1);

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

            for (int i = 0; i < m_param.tokenized_data_pairs_param.batch_size; i++)
            {
                int nIdx = m_random.Next((int)m_rgIdx.Count);
                int nDataIdx = m_rgIdx[nIdx];
                
                m_rgIdx.Remove(nIdx);
                if (m_rgIdx.Count == 0)
                {
                    m_nEpoch++;
                    m_log.WriteLine("EPOCH " + m_nEpoch.ToString(), true);
                    loadIdx((int)m_lNum);
                }

                Array.Copy(m_rgEnc, nDataIdx * nBlk, m_rgDataEnc, i * nBlk, nBlk);
                Array.Copy(m_rgDec, nDataIdx * nBlk, m_rgDataDec, i * nBlk, nBlk);
                Array.Copy(m_rgTrg, nDataIdx * nBlk, m_rgDataTrg, i * nBlk, nBlk);
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
