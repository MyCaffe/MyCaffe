using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using MyCaffe.layers.beta;

namespace MyCaffe.layers
{
    /// <summary>
    /// The LSTMAttentionLayer adds attention to the long-short term memory layer and
    /// is used in encoder/decoder models.  To use attention, just set 'enable_attention'=true.
    /// When disabled, this layer operates like a standard LSTM layer where inputs are in 
    /// the shape T,B,I with T=timesteps, B=batch and I=input.
    /// 
    /// This layer is initialized with the MyCaffe.param.LSTMAttentionParameter.
    /// </summary>
    /// <remarks>
    /// This layer is an alteration of the original LSTM implementation at: https://github.com/junhyukoh/caffe-lstm
    /// @see [A Clockwork RNN](https://arxiv.org/abs/1402.3511) by Jan Koutnik, Klaus Greff, Faustino Gomez, and Jürgen Schmidhuber, 2014.
    /// @see [Long short-term memory](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.56.7752) by Sepp Hochreiter and Jürgen Schmidhuber, 1997.
    /// @see [Learning to execute](https://arxiv.org/abs/1410.4615) by Wojciech Zaremba and Ilya Sutskever, 2014.
    /// @see [Generating sequences with recurrent neural networks](https://arxiv.org/abs/1308.0850) by Alex Graves, 2013.
    /// @see [Predictive Business Process Monitoring with LSTM Neural Networks](https://arxiv.org/abs/1612.02130) by Niek Tax, Ilya Verenich, Marcello La Rosa, and Marlon Dumas, 2016. 
    /// @see [Using LSTM recurrent neural networks for detecting anomalous behavior of LHC superconducting magnets](https://arxiv.org/abs/1611.06241) by Maciej Wielgosz, Andrzej Skoczeń, and Matej Mertik, 2016.
    /// @see [Spatial, Structural and Temporal Feature Learning for Human Interaction Prediction](https://arxiv.org/abs/1608.05267v2) by Qiuhong Ke, Mohammed Bennamoun, Senjian An, Farid Bossaid, and Ferdous Sohel, 2016.
    /// 
    /// For attention:
    /// @see [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin, 2017, arXiv:1706:03762
    /// @see [Attention is All You Need](https://tzuruey.medium.com/attention-is-all-you-need-98d26aeb3517) by Jenny Ching, 2019, Medium
    /// @see [Attention is All You Need in Speech Separation] by Cem Subakan, Mirco Ravanelli, Samuele Cornell, Mirko Bronzi, Jianyuan Zhong, 2020, arXiv:2010.13154
    /// @see [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) by Ilya Sutskever, Oriol Vinyals, and Quoc V. Le, 2014, arXiv:1409.3215
    /// 
    /// The AttentionLayer implementation was inspired by the C# Seq2SeqLearn implementation by mashmawy for language translation,
    /// @see [mashmawy/Seq2SeqLearn](https://github.com/mashmawy/Seq2SeqLearn) distributed under MIT license.
    /// 
    /// And also inspired by the C# ChatBot implementation by HectorPulido which uses Seq2SeqLearn
    /// @see [HectorPulido/Chatbot-seq2seq-C-](https://github.com/HectorPulido/Chatbot-seq2seq-C-) distributed under [MIT license](https://github.com/HectorPulido/Chatbot-seq2seq-C-/blob/master/LICENSE).
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class LSTMAttentionLayer<T> : Layer<T>
    {
        int m_nI;   // input dimension.
        int m_nH;   // number of hidden units.
        int m_nT;   // length of sequence.
        int m_nN;   // batch size.

        double m_dfClippingThreshold; // threshold for clipped gradient.
        Blob<T> m_blobBiasMultiplier;

        Blob<T> m_blobCell;     // Memory cell.
        Blob<T> m_blobPreGate;  // gate values before nonlinearity.
        Blob<T> m_blobGate;     // gate values after nonlinearity.

        Blob<T> m_blob_C_0;      // previous cell state value.
        Blob<T> m_blob_H_0;      // previous hidden activation value.
        Blob<T> m_blob_C_T;      // next cell state value.
        Blob<T> m_blob_H_T;      // next hidden activation value.

        // Intermediate values.
        Blob<T> m_blob_H_to_Gate;
        Blob<T> m_blob_H_to_H;
        Blob<T> m_blob_C_to_Gate = null;
        Blob<T> m_blobEOutputWhd = null;
        int m_nWeightItoHidx;
        int m_nWeightHtoHidx;
        int m_nWeightBiasidx;
        int m_nWeightCtoHidx;
        int m_nWeightWhdidx;
        int m_nWeightWhdbidx;

        // MaxT 
        Blob<T> m_blobMaxT = null;
        int? m_nMaxT = null;

        // Attention values
        Layer<T> m_attention = null;
        Blob<T> m_blobContext = null;
        Blob<T> m_blobPrevCt = null;
        Blob<T> m_blobContextFull = null;
        BlobCollection<T> m_colInternalBottom = new BlobCollection<T>();
        BlobCollection<T> m_colInternalTop = new BlobCollection<T>();

        /// <summary>
        /// The AttentionDecodeLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type LSTM with parameter lstm_simple_param,
        /// with options:
        ///   - num_output.  The dimension of the output -- must be explicitly set to non-zero.
        ///   
        ///   - clipping_threshold (/b optional, default = 0).  The gradient clipping threshold (0 = no clipping).
        ///   
        ///   - weight_filler (/b optional, default = "gaussian"). The weight filler used to initialize the weights.
        ///   
        ///   - bias_filler (/b optional, default = "constant, 1.0"). The bias filler used to initialize the bias values.
        ///   
        ///   - batch_size (/b optional, default = 1).  The batch size.
        ///   
        ///   - enable_clockwork_forget_bias (/b optional, default = false).  Whether or not to set the forget gate bias to 5.0 as recommended by [1] Koutnik, J., et al.
        ///   
        ///   - enable_attention (/b optional, default = false). Whether or not to add attention to each state as eacth timestep is processed. [i] Vaswani, a., et al.
        /// </param>
        public LSTMAttentionLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.LSTM_ATTENTION;

            m_blobBiasMultiplier = new Blob<T>(m_cuda, m_log);
            m_blobBiasMultiplier.Name = m_param.name + " biasmult";
            m_blobCell = new Blob<T>(m_cuda, m_log);
            m_blobCell.Name = m_param.name + " cell";
            m_blobPreGate = new Blob<T>(m_cuda, m_log);
            m_blobPreGate.Name = m_param.name + " pregate";
            m_blobGate = new Blob<T>(m_cuda, m_log);
            m_blobGate.Name = m_param.name + " gate";
            m_blob_C_0 = new Blob<T>(m_cuda, m_log);
            m_blob_C_0.Name = m_param.name + " c_0";
            m_blob_H_0 = new Blob<T>(m_cuda, m_log);
            m_blob_H_0.Name = m_param.name + " h_0";
            m_blob_C_T = new Blob<T>(m_cuda, m_log);
            m_blob_C_T.Name = m_param.name + " c_t";
            m_blob_H_T = new Blob<T>(m_cuda, m_log);
            m_blob_H_T.Name = m_param.name + " h_t";
            m_blob_H_to_Gate = new Blob<T>(m_cuda, m_log);
            m_blob_H_to_Gate.Name = m_param.name + "h_to_gate";
            m_blob_H_to_H = new Blob<T>(m_cuda, m_log);
            m_blob_H_to_H.Name = m_param.name + " h_to_h";
            m_blobMaxT = new Blob<T>(m_cuda, m_log);
            m_blobMaxT.Name = m_param.name + " maxT";

            if (m_param.lstm_attention_param.num_output_ip > 0)
            {
                m_blobEOutputWhd = new Blob<T>(m_cuda, m_log);
                m_blobEOutputWhd.Name = m_param.name + " ip";
            }
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            dispose(ref m_attention);
            dispose(ref m_blobContext);
            dispose(ref m_blobPrevCt);
            dispose(ref m_blobContextFull);

            dispose(ref m_blobBiasMultiplier);
            dispose(ref m_blobCell);
            dispose(ref m_blobPreGate);
            dispose(ref m_blobGate);
            dispose(ref m_blob_C_0);
            dispose(ref m_blob_C_T);
            dispose(ref m_blob_H_0);
            dispose(ref m_blob_H_T);
            dispose(ref m_blob_H_to_Gate);
            dispose(ref m_blob_H_to_H);
            dispose(ref m_blob_C_to_Gate);

            dispose(ref m_blobMaxT);
            dispose(ref m_blobEOutputWhd);
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobBiasMultiplier);
            col.Add(m_blobCell);
            col.Add(m_blobPreGate);
            col.Add(m_blobGate);
            col.Add(m_blob_C_0);
            col.Add(m_blob_H_0);
            col.Add(m_blob_C_T);
            col.Add(m_blob_H_T);
            col.Add(m_blob_H_to_Gate);
            col.Add(m_blob_H_to_H);
            col.Add(m_blobMaxT);

            if (m_blobEOutputWhd != null)
                col.Add(m_blobEOutputWhd);

            if (m_attention != null)
            {
                col.Add(m_blob_C_to_Gate);
                col.Add(m_blobPrevCt);

                foreach (Blob<T> b in m_attention.internal_blobs)
                {
                    col.Add(b);
                }
            }
        }

        /// <summary>
        /// Returns the minimum number of required bottom (input) Blobs: input
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of required bottom (input) Blobs: 
        ///     input, 
        ///     inputClip, 
        ///     encoding,     (used when enable_attention = true).
        ///     encodingClip, (used when enable_attention = true).
        ///     vocabcount (optional)
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 5; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: output (ht).
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        private void addInternal(Blob<T> bottom, Blob<T> top)
        {
            m_colInternalBottom.Clear();
            m_colInternalBottom.Add(bottom);

            m_colInternalTop.Clear();
            m_colInternalTop.Add(top);
        }

        private void addInternal(List<Blob<T>> rgBottom, Blob<T> top)
        {
            m_colInternalBottom.Clear();

            for (int i = 0; i < rgBottom.Count; i++)
            {
                m_colInternalBottom.Add(rgBottom[i]);
            }

            m_colInternalTop.Clear();
            m_colInternalTop.Add(top);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            LSTMAttentionParameter p = m_param.lstm_attention_param;

            if (m_param.lstm_attention_param.enable_attention)
            {
                m_log.CHECK_GE(colBottom.Count, 4, "When using attention, four bottoms are required: x, xClip, encoding, encodingClip.");
                m_log.CHECK_LE(colBottom.Count, 5, "When using attention, four bottoms are required: x, xClip, encoding, encodingClip, vocabcount (optional).");

                if (colBottom.Count == 5)
                {
                    if (p.num_output_ip != 0)
                        p.num_output_ip = (uint)convertF(colBottom[4].GetData(0));
                }
            }
            else
            {
                m_log.CHECK_GE(colBottom.Count, 1, "When not using attention, at least one bottom is required: x.");
                m_log.CHECK_LE(colBottom.Count, 2, "When not using attention, no more than two bottoms is required: x, clip.");
            }

            m_dfClippingThreshold = p.clipping_threshold;
            m_nN = colBottom[0].channels;
            m_nH = (int)p.num_output;      // number of hidden units.
            m_nI = colBottom[0].count(2);  // input dimension.

            // Check if we need to set up the weights.
            if (m_colBlobs.Count > 0)
            {
                m_log.WriteLine("Skipping parameter initialization.");
            }
            else
            {
                m_colBlobs = new BlobCollection<T>();

                Filler<T> weight_filler = Filler<T>.Create(m_cuda, m_log, p.weight_filler);
                Filler<T> bias_filler = Filler<T>.Create(m_cuda, m_log, p.bias_filler);

                // input-to-hidden weights
                // Initialize the weight.
                List<int> rgShape1 = new List<int>() { 4 * m_nH, m_nI };
                Blob<T> blobWeights_I_H = new Blob<T>(m_cuda, m_log);
                blobWeights_I_H.Name = m_param.name + " weights I to H";
                blobWeights_I_H.type = BLOB_TYPE.WEIGHT;

                if (!shareParameter(blobWeights_I_H, rgShape1))
                {
                    blobWeights_I_H.Reshape(rgShape1);
                    weight_filler.Fill(blobWeights_I_H);
                }
                m_nWeightItoHidx = m_colBlobs.Count;
                m_colBlobs.Add(blobWeights_I_H);

                // hidden-to-hidden weights
                // Initialize the weight.
                List<int> rgShape2 = new List<int>() { 4 * m_nH, m_nH };
                Blob<T> blobWeights_H_H = new Blob<T>(m_cuda, m_log);
                blobWeights_H_H.Name = m_param.name + " weights H to H";
                blobWeights_H_H.type = BLOB_TYPE.WEIGHT;

                if (!shareParameter(blobWeights_H_H, rgShape2))
                {
                    blobWeights_H_H.Reshape(rgShape2);
                    weight_filler.Fill(blobWeights_H_H);
                }
                m_nWeightHtoHidx = m_colBlobs.Count;
                m_colBlobs.Add(blobWeights_H_H);

                // If necessary, initialize and fill the bias term.
                List<int> rgShape3 = new List<int>() { 4 * m_nH };
                Blob<T> blobBias = new Blob<T>(m_cuda, m_log);
                blobBias.Name = m_param.name + " bias weights";
                blobBias.type = BLOB_TYPE.WEIGHT;

                if (!shareParameter(blobBias, rgShape3))
                {
                    blobBias.Reshape(rgShape3);
                    bias_filler.Fill(blobBias);
                }
                m_nWeightBiasidx = m_colBlobs.Count;
                m_colBlobs.Add(blobBias);

                // Initialize the bias for the forget gate to 5.0 as described in the
                // Clockwork RNN paper: 
                // [1] Koutnik, J., Greff, K., Gomez, F., Schmidhuber, J., 'A Clockwork RNN', 2014"
                if (p.enable_clockwork_forgetgate_bias)
                {
                    double[] rgBias = convertD(blobBias.mutable_cpu_data);

                    for (int i=m_nH; i<2*m_nH; i++)
                    {
                        rgBias[i] = 5.0;
                    }

                    blobBias.mutable_cpu_data = convert(rgBias);
                }

                if (m_param.lstm_attention_param.num_output_ip > 0)
                {
                    Blob<T> blobWeightWhd = new Blob<T>(m_cuda, m_log);
                    blobWeightWhd.Name = m_param.name + " weights Whd";
                    blobWeightWhd.type = BLOB_TYPE.WEIGHT;

                    List<int> rgShapeWhd = new List<int>() { m_nH, (int)m_param.lstm_attention_param.num_output_ip };
                    if (!shareParameter(blobWeightWhd, rgShapeWhd))
                    {
                        blobWeightWhd.Reshape(rgShapeWhd); 
                        weight_filler.Fill(blobWeightWhd);
                    }
                    m_nWeightWhdidx = m_colBlobs.Count;
                    m_colBlobs.Add(blobWeightWhd);

                    Blob<T> blobWeightWhdb = new Blob<T>(m_cuda, m_log);
                    blobWeightWhdb.Name = m_param.name + " weights Whdb";
                    blobWeightWhdb.type = BLOB_TYPE.WEIGHT;

                    List<int> rgShapeWhdb = new List<int>() { 1, (int)m_param.lstm_attention_param.num_output_ip };
                    if (!shareParameter(blobWeightWhdb, rgShape1))
                    {
                        blobWeightWhdb.Reshape(rgShapeWhdb); 
                        bias_filler.Fill(blobWeightWhdb);
                    }
                    m_nWeightWhdbidx = m_colBlobs.Count;
                    m_colBlobs.Add(blobWeightWhdb);
                }

                if (m_param.lstm_attention_param.enable_attention)
                {
                    // context-to-hidden weights
                    // Initialize the weight.
                    Blob<T> blobWeights_C_H = new Blob<T>(m_cuda, m_log);
                    blobWeights_C_H.Name = m_param.name + " weights C to H";
                    blobWeights_C_H.type = BLOB_TYPE.WEIGHT;

                    if (!shareParameter(blobWeights_C_H, rgShape1))
                    {
                        blobWeights_C_H.Reshape(rgShape1); // same shape as I to H
                        weight_filler.Fill(blobWeights_C_H);
                    }
                    m_nWeightCtoHidx = m_colBlobs.Count;
                    m_colBlobs.Add(blobWeights_C_H);
                }
            }

            m_rgbParamPropagateDown = new DictionaryMap<bool>(m_colBlobs.Count, true);

            List<int> rgCellShape = new List<int>() { m_nN, m_nH };
            m_blob_C_0.Reshape(rgCellShape);
            m_blob_H_0.Reshape(rgCellShape);
            m_blob_C_T.Reshape(rgCellShape);
            m_blob_H_T.Reshape(rgCellShape);
            m_blob_H_to_H.Reshape(rgCellShape);

            List<int> rgGateShape = new List<int>() { m_nN, 4, m_nH };
            m_blob_H_to_Gate.Reshape(rgGateShape);

            // Attention settings
            if (m_param.lstm_attention_param.enable_attention)
            {
                m_blob_C_to_Gate = new Blob<T>(m_cuda, m_log, false);
                m_blob_C_to_Gate.Name = m_param.name + "c_to_gate";
                m_blob_C_to_Gate.Reshape(rgGateShape);

                m_blobContext = new Blob<T>(m_cuda, m_log);
                m_blobContext.Name = "context_out";

                m_blobContextFull = new Blob<T>(m_cuda, m_log);
                m_blobContextFull.Name = "context_full";

                m_blobPrevCt = new Blob<T>(m_cuda, m_log);
                m_blobPrevCt.Name = "prev_ct";

                LayerParameter attentionParam = new LayerParameter(LayerParameter.LayerType.ATTENTION);
                attentionParam.attention_param.axis = 2;
                attentionParam.attention_param.dim = m_param.lstm_attention_param.num_output;
                attentionParam.attention_param.weight_filler = m_param.lstm_attention_param.weight_filler;
                attentionParam.attention_param.bias_filler = m_param.lstm_attention_param.bias_filler;

                if (m_param is LayerParameterEx<T>)
                {
                    LayerParameterEx<T> pEx = m_param as LayerParameterEx<T>;
                    attentionParam = new LayerParameterEx<T>(attentionParam, pEx.SharedBlobs, pEx.SharedLayerBlobs, pEx.SharedLayer);
                }

                m_attention = new AttentionLayer<T>(m_cuda, m_log, attentionParam);

                Blob<T> blobEncoding = colBottom[2];
                Blob<T> blobEncodingClip = colBottom[3];
                addInternal(new List<Blob<T>>() { blobEncoding, m_blob_C_T, blobEncodingClip }, m_blobContext);
                m_attention.Setup(m_colInternalBottom, m_colInternalTop);

                foreach (Blob<T> b in m_attention.blobs)
                {
                    m_colBlobs.Add(b);
                }
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_bNetReshapeRequest)
            {
                m_nMaxT = null;
                m_nN = colBottom[0].channels;
                m_bNetReshapeRequest = false;
            }
            else
            {
                if (!reshapeNeeded(colBottom, colTop))
                    return;
            }

            // Figure out the dimensions.
            m_nT = colBottom[0].num;  // length of sequence.
            m_log.CHECK_EQ(colBottom[0].count() / m_nT / m_nN, m_nI, "The input size is incompatible with inner product parameters.");

            // Gate initialization.
            List<int> rgGateShape = new List<int>() { m_nT, m_nN, 4, m_nH };
            m_blobPreGate.Reshape(rgGateShape);
            m_blobGate.Reshape(rgGateShape);
            m_blob_H_to_Gate.Reshape(rgGateShape);

            List<int> rgTopShape = new List<int>() { m_nT, m_nN, m_nH };
            m_blobCell.Reshape(rgTopShape);
            colTop[0].Reshape(rgTopShape);

            // Setup the bias multipler.
            List<int> rgMultiplierShape = new List<int>() { m_nT, m_nN };
            m_blobBiasMultiplier.Reshape(rgMultiplierShape);
            m_blobBiasMultiplier.SetData(1.0);

            List<int> rgCellShape = new List<int>() { m_nN, m_nH };
            m_blob_C_0.Reshape(rgCellShape);
            m_blob_H_0.Reshape(rgCellShape);
            m_blob_C_T.Reshape(rgCellShape);
            m_blob_H_T.Reshape(rgCellShape);
            m_blob_H_to_H.Reshape(rgCellShape);

            if (colBottom.Count > 1)
                m_blobMaxT.Reshape(new List<int>() { 1, colBottom[1].channels });

            if (m_param.lstm_attention_param.num_output_ip > 0)
            {
                List<int> rgIpShape = new List<int>() { m_nT, m_nN, (int)m_param.lstm_attention_param.num_output_ip };
                m_blobEOutputWhd.Reshape(rgIpShape);
                colTop[0].Reshape(rgIpShape);
            }

            // Attention reshape
            if (m_param.lstm_attention_param.enable_attention)
            {
                m_blob_C_to_Gate.Reshape(rgGateShape);

                Blob<T> blobEncoding = colBottom[2];
                Blob<T> blobEncodingClip = colBottom[3];
                addInternal(new List<Blob<T>>() { blobEncoding, m_blob_C_T, blobEncodingClip }, m_blobContext);
                m_attention.Reshape(m_colInternalBottom, m_colInternalTop);

                List<int> rgShape = Utility.Clone<int>(m_blobContext.shape());
                rgShape[0] = m_nT;
                m_blobContextFull.Reshape(rgShape);

                m_blobPrevCt.ReshapeLike(m_blobCell);
            }
        }

        // Find the longest clip length.
        private int calculate_maxT(Blob<T> blob, out int nInitialClip)
        {
            int nMax = 1;

            if (blob.count() > 1)
            {
                m_blobMaxT.SetData(0);

                for (int t = 0; t < blob.num; t++)
                {
                    int nSrcIdx = t * blob.channels;
                    m_cuda.add(m_blobMaxT.count(), blob.gpu_data, m_blobMaxT.gpu_data, m_blobMaxT.mutable_gpu_data, 1.0, 1.0, nSrcIdx, 0, 0);
                }

                long lPos;
                nMax = (int)m_cuda.max(m_blobMaxT.count(), m_blobMaxT.gpu_data, out lPos);
                if (convertF(blob.GetData(0)) == 0)
                    nMax++;
            }

            nInitialClip = (int)convertF(blob.GetData(0));

            return nMax;
        }

        /// <summary>
        /// Forward computation.
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$ input
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$ cont
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hTopData = colTop[0].mutable_gpu_data;
            long hBottomData = colBottom[0].gpu_data;
            long hClipData = 0;
            int nMaxT = m_nT;
            int nInitialClip = 0;
            double dfOriginalClip = 0;

            if (colBottom.Count > 1)
            {
                hClipData = colBottom[1].gpu_data;
                m_log.CHECK_EQ(colBottom[0].count(0, 2), colBottom[1].count(), "The bottom[1].count() should equal the bottom[0].count(0,2).");

                m_nMaxT = calculate_maxT(colBottom[1], out nInitialClip);
                nMaxT = m_nMaxT.Value;
            }

            long hWeight_i = m_colBlobs[m_nWeightItoHidx].gpu_data;
            long hWeight_h = m_colBlobs[m_nWeightHtoHidx].gpu_data;
            long hBias = m_colBlobs[m_nWeightBiasidx].gpu_data;
            long hWeight_c = (m_param.lstm_attention_param.enable_attention) ? m_colBlobs[m_nWeightCtoHidx].gpu_data : 0;
            long hPreGateData = m_blobPreGate.mutable_gpu_data;
            long hGateData = m_blobGate.mutable_gpu_data;
            long hCellData = m_blobCell.mutable_gpu_data;
            long hHtoGateData = m_blob_H_to_Gate.mutable_gpu_data;
            long hCtoGateData = 0;

            // Initialize previous state.
            if (hClipData != 0 && nInitialClip != 0)
            {
                m_cuda.copy(m_blob_C_0.count(), m_blob_C_T.gpu_data, m_blob_C_0.mutable_gpu_data);
                m_cuda.copy(m_blob_H_0.count(), m_blob_H_T.gpu_data, m_blob_H_0.mutable_gpu_data);                               
            }
            else
            {
                m_blob_C_0.SetData(0.0);
                m_blob_H_0.SetData(0.0);
            }

            m_cuda.gemm(false, true, m_nT * m_nN, 4 * m_nH, m_nI, m_tOne, hBottomData, hWeight_i, m_tZero, hPreGateData);
            m_cuda.gemm(false, false, m_nT * m_nN, 4 * m_nH, 1, m_tOne, m_blobBiasMultiplier.gpu_data, hBias, m_tOne, hPreGateData);

            if (m_param.lstm_attention_param.enable_attention)
            {
                m_blobContextFull.SetData(0);
                if (nInitialClip == 0)
                    m_blobPrevCt.SetData(0);

                // Reset the clip for we want to use the initial context.
                dfOriginalClip = convertD(colBottom[1].GetData(0));
                colBottom[1].SetData(1.0, 0);
            }

            // Compute recurrent forward propagation                
            for (int t = 0; t < nMaxT; t++)
            {
                int nTopOffset = colTop[0].offset(t);
                int nCellOffset = m_blobCell.offset(t);
                int nPreGateOffset = m_blobPreGate.offset(t);
                int nGateOffset = m_blobGate.offset(t);
                int nClipOffset = (hClipData != 0) ? colBottom[1].offset(t) : 0;
                int nHT1Offset;
                long hHT1Data;
                int nCT1Offset;
                long hCT1Data;
                long hContext = 0;

                if (t == 0)
                {
                    hHT1Data = m_blob_H_0.gpu_data;
                    nHT1Offset = 0;
                    hCT1Data = m_blob_C_0.gpu_data;
                    nCT1Offset = 0;
                }
                else
                {
                    hHT1Data = m_blob_H_T.gpu_data;
                    nHT1Offset = -colTop[0].offset(1);
                    hCT1Data = m_blob_C_T.gpu_data;
                    nCT1Offset = -m_blobCell.offset(1);
                }

                if (m_param.lstm_attention_param.enable_attention)
                {
                    Blob<T> blobEncoding = colBottom[2];
                    Blob<T> blobEncodingClip = colBottom[3];

                    addInternal(new List<Blob<T>>() { blobEncoding, m_blobPrevCt, blobEncodingClip }, m_blobContext);
                    m_attention.Forward(m_colInternalBottom, m_colInternalTop);
                    hContext = m_blobContext.gpu_data;
                    hCtoGateData = m_blob_C_to_Gate.mutable_gpu_data;

                    int nCount = m_blobContext.count();
                    m_cuda.copy(nCount, hContext, m_blobContextFull.mutable_gpu_data, 0, t * nCount);
                }

                m_cuda.lstm_fwd(t,
                                m_nN,
                                m_nH,
                                m_nI,
                                hWeight_h,
                                hWeight_i,
                                hClipData,
                                nClipOffset,
                                hTopData,       // h_t data
                                nTopOffset,     // h_t offset
                                hCellData,      // c_t data
                                nCellOffset,    // c_t offset
                                hPreGateData,
                                nPreGateOffset,
                                hGateData,
                                nGateOffset,
                                hHT1Data,
                                nHT1Offset,
                                hCT1Data,
                                nCT1Offset,
                                hHtoGateData,
                                hContext,
                                hWeight_c,
                                hCtoGateData);

                if (m_param.lstm_attention_param.enable_attention)
                    m_blobPrevCt.CopyFrom(m_blobCell);
            }

            // Preserve cell state and output value for truncated BPTT
            m_cuda.copy(m_nN * m_nH, hCellData, m_blob_C_T.mutable_gpu_data, m_blobCell.offset(nMaxT - 1));
            m_cuda.copy(m_nN * m_nH, hTopData, m_blob_H_T.mutable_gpu_data, colTop[0].offset(nMaxT - 1));

            if (m_param.lstm_attention_param.num_output_ip > 0)
            {
                int nM = m_nT * m_nN;
                int nN = (int)m_param.lstm_attention_param.num_output_ip;
                int nK = m_nH;
                m_cuda.gemm(false, false, nM, nN, nK, Blob<T>.One, hTopData, m_colBlobs[m_nWeightWhdidx].gpu_data, Blob<T>.Zero, m_blobEOutputWhd.mutable_gpu_data);
                m_cuda.add(colTop[0].count(), m_blobEOutputWhd.gpu_data, m_colBlobs[m_nWeightWhdbidx].gpu_data, m_blobEOutputWhd.mutable_gpu_data);
                colTop[0].CopyFrom(m_blobEOutputWhd);
            }

            if (m_param.lstm_attention_param.enable_attention)
            {
                // Reset the clip to original value.
                colBottom[1].SetData(dfOriginalClip, 0);
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$</param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///  </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopData = colTop[0].gpu_data;
            long hBottomData = colBottom[0].gpu_data;
            long hClipData = 0;
            int nMaxT = m_nT;

            List<bool> rgbPropagate = new List<bool>() { true, true };

            if (colBottom.Count > 1)
            {
                hClipData = colBottom[1].gpu_data;
                m_cuda.sign(colBottom[1].count(), hClipData, hClipData); // Set to 1 or 0.
                m_log.CHECK_EQ(colBottom[0].count(0, 2), colBottom[1].count(), "The bottom[1].count() should equal the bottom[0].count(0,2).");
                nMaxT = m_nMaxT.Value;
            }

            long hWeight_i = m_colBlobs[m_nWeightItoHidx].gpu_data;
            long hWeight_h = m_colBlobs[m_nWeightHtoHidx].gpu_data;
            long hGateData = m_blobGate.gpu_data;
            long hCellData = m_blobCell.gpu_data;

            long hTopDiff = colTop[0].mutable_gpu_diff;
            long hPreGateDiff = m_blobPreGate.mutable_gpu_diff;
            long hGateDiff = m_blobGate.mutable_gpu_diff;
            long hCellDiff = m_blobCell.mutable_gpu_diff;
            long hHtoHDiff = m_blob_H_to_H.mutable_gpu_diff;

            m_blobCell.SetDiff(0);
            m_blobGate.SetDiff(0);
            m_blobPreGate.SetDiff(0);
            m_blob_H_to_H.SetDiff(0);
            m_blob_H_to_Gate.SetDiff(0);

            long hWeight_c = 0;
            long hContextData = 0;
            long hContextDiff = 0;

            if (m_param.lstm_attention_param.num_output_ip > 0)
            {
                int nM = m_nT * m_nN;
                int nN = (int)m_param.lstm_attention_param.num_output_ip;
                int nK = m_nH;

                m_cuda.copy(colTop[0].count(), colTop[0].gpu_diff, m_blobEOutputWhd.mutable_gpu_diff);
                m_cuda.add(colTop[0].count(), colTop[0].gpu_diff, m_colBlobs[m_nWeightWhdbidx].gpu_diff, m_colBlobs[m_nWeightWhdbidx].mutable_gpu_diff);
                m_cuda.gemm(false, true, nM, nK, nN, Blob<T>.One, m_blobEOutputWhd.gpu_diff, m_colBlobs[m_nWeightWhdidx].gpu_data, Blob<T>.Zero, m_blob_H_T.mutable_gpu_diff);
                m_cuda.gemm(true, false, nK, nN, nM, Blob<T>.One, m_blob_H_T.gpu_data, m_blobEOutputWhd.gpu_diff, Blob<T>.One, m_colBlobs[m_nWeightWhdidx].mutable_gpu_diff);
                hTopDiff = m_blob_H_T.gpu_diff;
                hTopData = m_blob_H_T.gpu_data;
            }

            if (m_param.lstm_attention_param.enable_attention)
            {
                m_blobContext.SetDiff(0);
                m_blob_C_to_Gate.SetDiff(0);
                hWeight_c = m_colBlobs[m_nWeightCtoHidx].gpu_data;
                hContextData = m_blobContext.gpu_data;
                hContextDiff = m_blobContext.mutable_gpu_diff;
                m_cuda.sign(colBottom[3].count(), colBottom[3].gpu_data, colBottom[3].mutable_gpu_data); // Set to 1 or 0.
            }

            m_blob_C_T.SetDiff(0);
            m_cuda.copy(m_nN * m_nH, m_blob_C_T.gpu_diff, hCellDiff, 0, m_blobCell.offset(nMaxT - 1));

            for (int t = nMaxT - 1; t >= 0; t--)
            {
                int nTopOffset = colTop[0].offset(t);
                int nCellOffset = m_blobCell.offset(t);
                int nGateOffset = m_blobGate.offset(t);
                int nPreGateOffset = m_blobPreGate.offset(t);
                int nClipOffset = (hClipData == 0) ? 0 : colBottom[1].offset(t);
                int nCT1Offset;
                long hCT1Data;
                int nDHT1Offset;
                long hDHT1Diff;
                int nDCT1Offset;
                long hDCT1Diff;

                if (t == 0)
                {
                    nCT1Offset = 0;
                    hCT1Data = m_blob_C_0.gpu_data;
                    nDHT1Offset = 0;
                    hDHT1Diff = m_blob_H_0.mutable_gpu_diff;
                    nDCT1Offset = 0;
                    hDCT1Diff = m_blob_C_0.mutable_gpu_diff;
                }
                else
                {
                    nCT1Offset = m_blobCell.offset(t - 1);
                    hCT1Data = hCellData;
                    nDHT1Offset = colTop[0].offset(t - 1);
                    hDHT1Diff = hTopDiff;
                    nDCT1Offset = m_blobCell.offset(t - 1);
                    hDCT1Diff = hCellDiff;
                }

                m_cuda.lstm_bwd(t, 
                                m_nN, 
                                m_nH, 
                                m_nI,
                                m_dfClippingThreshold, 
                                hWeight_h, 
                                hClipData, 
                                nClipOffset, 
                                hTopDiff, 
                                nTopOffset, 
                                hCellData, 
                                hCellDiff, 
                                nCellOffset, 
                                hPreGateDiff, 
                                nPreGateOffset, 
                                hGateData, 
                                hGateDiff, 
                                nGateOffset, 
                                hCT1Data, 
                                nCT1Offset, 
                                hDHT1Diff, 
                                nDHT1Offset, 
                                hDCT1Diff, 
                                nDCT1Offset, 
                                hHtoHDiff,
                                hContextDiff,
                                hWeight_c);

                if (m_param.lstm_attention_param.enable_attention)
                {
                    Blob<T> blobEncoding = colBottom[2];
                    Blob<T> blobEncodingClip = colBottom[3];
                    addInternal(new List<Blob<T>>() { blobEncoding, m_blob_C_T, blobEncodingClip }, m_blobContext);
                    m_attention.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                    int nCount = m_blobContext.count();
                    m_cuda.copy(nCount, hContextDiff, m_blobContextFull.mutable_gpu_diff, 0, t * nCount);
                }
            }

            if (m_rgbParamPropagateDown[0])
            {
                // Gradient w.r.t input-to-hidden weight
                m_cuda.gemm(true, false, 4 * m_nH, m_nI, m_nT * m_nN, m_tOne, hPreGateDiff, hBottomData, m_tOne, m_colBlobs[m_nWeightItoHidx].mutable_gpu_diff);
            }

            if (m_rgbParamPropagateDown[1])
            {
                // Gradient w.r.t. hidden-to-hidden weight
                m_cuda.gemm(true, false, 4 * m_nH, m_nH, (m_nT - 1) * m_nN, m_tOne, hPreGateDiff, hTopData, m_tOne, m_colBlobs[m_nWeightHtoHidx].mutable_gpu_diff, m_blobPreGate.offset(1));

                // Add gradient from previous time-step.
                m_cuda.gemm(true, false, 4 * m_nH, m_nH, 1, m_tOne, hPreGateDiff, m_blob_H_0.gpu_data, m_tOne, m_colBlobs[m_nWeightHtoHidx].mutable_gpu_diff);
            }

            if (m_rgbParamPropagateDown[2])
            {
                // Gradient w.r.t. bias.
                m_cuda.gemv(true, m_nT * m_nN, 4 * m_nH, m_tOne, hPreGateDiff, m_blobBiasMultiplier.gpu_data, m_tOne, m_colBlobs[m_nWeightBiasidx].mutable_gpu_diff);
            }

            if (m_rgbParamPropagateDown[3] && m_param.lstm_attention_param.enable_attention)
            {
                // Gradient w.r.t. context data.
                m_cuda.gemm(true, false, 4 * m_nH, m_nI, m_nT * m_nN, m_tOne, hPreGateDiff, m_blobContextFull.gpu_data, m_tOne, m_colBlobs[m_nWeightCtoHidx].mutable_gpu_diff);
            }

            if (rgbPropagateDown[0])
            {
                // Gradient w.r.t. bottom data.
                m_cuda.gemm(false, false, m_nT * m_nN, m_nI, 4 * m_nH, m_tOne, hPreGateDiff, hWeight_i, m_tZero, colBottom[0].mutable_gpu_diff);
            }
        }
    }
}
