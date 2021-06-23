using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using System.IO;
using MyCaffe.layers.beta.ModelData;
using MyCaffe.db.image;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The ModelDataLayer loads data from RawImageResults table for an encoder/decoder type model.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ModelDataLayer<T> : Layer<T>
    {
        IXImageDatabase2 m_db;
        CancelEvent m_evtCancel;
        DataItem m_currentData = null;
        Data m_data = null;
        ulong m_lOffset = 0;
        float[] m_rgEncInput1;
        float[] m_rgEncClip;
        float[] m_rgDecInput;
        float[] m_rgDecClip;
        float[] m_rgDecTarget;

        /// <summary>
        /// The OnGetTrainingData is called during each forward pass after getting the training data for the pass.
        /// </summary>
        public event EventHandler<OnGetDataArgs> OnGetData;

        /// <summary>
        /// The ModelDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Provides ModelDataParameter model_data_param with options:
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
        public ModelDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXImageDatabaseBase db, CancelEvent evtCancel)
            : base(cuda, log, p)
        {
            m_db = db as IXImageDatabase2;
            if (m_db == null)
                throw new Exception("The ModelDataLayer requires the ImageDatabase V2 or higher.");

            m_evtCancel = evtCancel;
            m_type = LayerParameter.LayerType.MODEL_DATA;
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
        /// When running in RUN phase, returns 3 Blobs: dec_input, enc_input, enc_clip.
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return (m_phase == Phase.RUN) ? 3 : 0; }
        }

        /// <summary>
        /// When running in TRAIN or TEST phase, returns 0 for data layers have no bottom (input) Blobs.
        /// When running in RUN phase, returns 3 Blobs: dec_input, enc_input, enc_clip.
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return (m_phase == Phase.RUN) ? 3 : 0; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: dec, dclip, enc, eclip, vocabcount, label (only valid on TRAIN or TEST)
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 6; }
        }

        /// <summary>
        /// Returns the maximum number of required top (output) Blobs: dec, dclip, enc, eclip, vocabcount, label (only valid on TRAIN or TEST)
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 6; }
        }

        /// <summary>
        /// Returns information on the current iteration.
        /// </summary>
        public IterationInfo IterationInfo
        {
            get { return (m_currentData == null) ? new IterationInfo(true, true, 0) : m_currentData.IterationInfo; }
        }

        /// <summary>
        /// Returns the decoder vocabulary count.
        /// </summary>
        public int DecoderVocabularyCount
        {
            get { return (m_data == null) ? 0 : m_data.DecoderVocabCount; }
        }

        /// <summary>
        /// Should return true when pre processing methods are overriden.
        /// </summary>
        public override bool SupportsPreProcessing
        {
            get { return false; }
        }

        /// <summary>
        /// Should return true when pre postprocessing methods are overriden.
        /// </summary>
        public override bool SupportsPostProcessing
        {
            get { return false; }
        }

        /// <summary>
        /// Load all model data from the data sources.
        /// </summary>
        /// <param name="p">Specifies the ModelData parameter.</param>
        private void PreProcessData(ModelDataParameter p)
        {
            List<SimpleResult> rgFullList = new List<SimpleResult>();

            foreach (string strSrc in p.source)
            {
                List<SimpleResult> rgRes = m_db.GetAllResults(strSrc);
                rgFullList.AddRange(rgRes);
            }

            rgFullList = rgFullList.OrderBy(p1 => p1.TimeStamp).ThenBy(p1 => p1.Index).ToList();
            m_data = new Data(rgFullList);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Refuse transformation parameters since ModelData is totally generic.
            if (m_param.transform_param != null)
                m_log.WriteLine("WARNING: " + m_type.ToString() + " does not transform data.");

            m_log.CHECK_EQ(m_param.model_data_param.batch_size, 1, "Currently, only batch_size = 1 supported.");
            m_log.CHECK_EQ(colTop.Count, 6, "When normal or reverse encoder output used, there must be 6 tops: dec, dclip, enc | encr, eclip, vocabcount, dectgt (only valid on TEST | TRAIN)");

            // Load the encoder and decoder input data into the Data.
            PreProcessData(m_param.model_data_param);

            m_rgDecInput = new float[m_param.model_data_param.batch_size];
            m_rgDecClip = new float[m_param.model_data_param.batch_size];
            m_rgEncInput1 = new float[m_param.model_data_param.batch_size * m_param.model_data_param.time_steps * m_param.model_data_param.input_dim];
            m_rgEncClip = new float[m_param.model_data_param.batch_size * m_param.model_data_param.time_steps];

            if (m_phase != Phase.RUN)
                m_rgDecTarget = new float[m_param.model_data_param.batch_size];

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
            m_currentData = m_data.GetNextData(m_param.model_data_param.shuffle);
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
            int nBatchSize = (int)m_param.model_data_param.batch_size;
            int nT = (int)m_param.model_data_param.time_steps;
            int nI = (int)m_param.model_data_param.input_dim;
            List<int> rgTopShape = new List<int>() { nT, nBatchSize, nI };
            int nTopIdx = 0;

            // Reshape the decoder input.
            if (!bSetup)
                colTop[nTopIdx].Reshape(new List<int>() { 1, nBatchSize, 1 });
            nTopIdx++;

            // Reshape the decoder clip.
            if (!bSetup)
                colTop[nTopIdx].Reshape(new List<int>() { 1, nBatchSize });
            nTopIdx++;

            // Reshape the encoder data
            if (!bSetup)
                colTop[nTopIdx].Reshape(rgTopShape);
            nTopIdx++;

            // Reshape the encoder clip for attention.
            if (!bSetup)
                colTop[nTopIdx].Reshape(new List<int>() { nT, nBatchSize });
            nTopIdx++;

            // Reshape the vocab count.
            colTop[nTopIdx].Reshape(new List<int>() { 1 });
            if (bSetup)
                colTop[nTopIdx].SetData(m_data.DecoderVocabCount + 2, 0);
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
            int nBatch = (int)m_param.model_data_param.batch_size;
            int nT = (int)m_param.model_data_param.time_steps;

            Array.Clear(m_rgDecInput, 0, m_rgDecInput.Length);
            if (m_phase != Phase.RUN)
                Array.Clear(m_rgDecTarget, 0, m_rgDecTarget.Length);
            Array.Clear(m_rgDecClip, 0, m_rgDecClip.Length);
            Array.Clear(m_rgEncInput1, 0, m_rgEncInput1.Length);
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
                        OnGetData(this, new OnGetDataArgs(IterationInfo));

                    int nIdx = i * nT;

                    Array.Copy(m_currentData.EncoderInput, 0, m_rgEncInput1, i * m_currentData.EncoderInput.Length, m_currentData.EncoderInput.Length);

                    int nEncInputCount = m_currentData.EncoderInput.Length / (int)m_param.model_data_param.input_dim;
                    for (int j = 0; j < nT && j < nEncInputCount; j++)
                    {
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

                colTop[nTopIdx].mutable_cpu_data = convert(m_rgEncInput1);
                nTopIdx++;

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

                colTop[nTopIdx].CopyFrom(colBottom[nBtmIdx]);
                nTopIdx++;
                nBtmIdx++;

                // Encoder clip.
                colTop[nTopIdx].CopyFrom(colBottom[nBtmIdx]);
            }
        }

        /// @brief Not implemented - data Layers do not perform backward..
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }
    }


    namespace ModelData
    {
#pragma warning disable 1591

        class Data /** @private */
        {
            Random m_random = new Random((int)DateTime.Now.Ticks);
            List<SimpleResult> m_rgData;
            int m_nDecoderVocabCount = 1;
            int m_nCurrentSequence = -1;
            int m_nCurrentOutputIdx = 0;
            int m_nSequenceIdx = 0;
            int m_nIxInput = 1;
            int m_nIterations = 0;
            int m_nOutputCount = 0;

            public Data(List<SimpleResult> rgData)
            {
                m_rgData = rgData;
                m_nDecoderVocabCount = rgData[0].Target.Length;
            }

            public int DecoderVocabCount
            {
                get { return m_nDecoderVocabCount; }
            }

            public static DataItem GetInputData(float[] rgfInput, int? nDecInput = null)
            {
                int nClip = 1;

                if (!nDecInput.HasValue)
                {
                    nClip = 0;
                    nDecInput = 1;
                }

                return new DataItem(rgfInput, nDecInput.Value, -1, nClip, false, true, 0);
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
                        m_nCurrentSequence = m_random.Next(m_rgData.Count);
                    }
                    else
                    {
                        m_nCurrentSequence = m_nSequenceIdx;
                        m_nSequenceIdx++;
                        if (m_nSequenceIdx == m_rgData.Count)
                            m_nSequenceIdx = 0;
                    }

                    m_nOutputCount = m_rgData[m_nCurrentSequence].Target.Length;
                    nDecClip = 0;

                    if (m_nIterations == m_rgData.Count)
                    {
                        bNewEpoch = true;
                        m_nIterations = 0;
                    }
                }

                int nIxTarget = 0;

                if (m_nCurrentOutputIdx < m_rgData[m_nCurrentSequence].Target.Length)
                    nIxTarget = m_rgData[m_nCurrentSequence].Target[m_nCurrentOutputIdx];

                DataItem data = new DataItem(m_rgData[m_nCurrentSequence].Result, m_nIxInput, nIxTarget, nDecClip, bNewEpoch, bNewSequence, m_nOutputCount);
                m_nIxInput = nIxTarget;

                m_nCurrentOutputIdx++;

                if (m_nCurrentOutputIdx == m_rgData[m_nCurrentSequence].Target.Length)
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
            float[] m_rgInput;
            int m_nIxInput;
            int m_nIxTarget;
            int m_nDecClip;

            public DataItem(float[] rgInput, int nIxInput, int nIxTarget, int nDecClip, bool bNewEpoch, bool bNewSequence, int nOutputCount)
            {
                m_rgInput = rgInput;
                m_nIxInput = nIxInput;
                m_nIxTarget = nIxTarget;
                m_nDecClip = nDecClip;
                m_iter = new IterationInfo(bNewEpoch, bNewSequence, nOutputCount);
            }

            public float[] EncoderInput
            {
                get { return m_rgInput; }
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
        /// Defines the arguments passed to the OnGetData event.
        /// </summary>
        public class OnGetDataArgs : EventArgs
        {
            IterationInfo m_iter;

            /// <summary>
            /// The constructor.
            /// </summary>
            /// <param name="iter">Specifies the iteration info.</param>
            public OnGetDataArgs(IterationInfo iter)
            {
                m_iter = iter;
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
