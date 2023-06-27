using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the LSTMAttentionLayer that provides an attention based LSTM layer used for decoding in 
    /// an encoder/decoder based model.
    /// </summary>
    /// <remarks>
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
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class LSTMAttentionParameter : LayerParameterBase
    {
        uint m_nNumOutput;
        uint m_nNumIpOutput = 0;
        double m_dfClippingThreshold = 0;
        FillerParameter m_fillerWeights = new FillerParameter("xavier");
        FillerParameter m_fillerBias = new FillerParameter("constant", 0.1);
        bool m_bEnableClockworkForgetGateBias = false;
        bool m_bEnableAttention = false;

        /** @copydoc LayerParameterBase */
        public LSTMAttentionParameter()
        {
        }

        /// <summary>
        /// Specifies the number of outputs for the layer.
        /// </summary>
        [Description("Specifies the number of outputs for the layer.")]
        public uint num_output
        {
            get { return m_nNumOutput; }
            set { m_nNumOutput = value; }
        }

        /// <summary>
        /// Specifies the number of IP outputs for the layer.  Note, when 0, no inner product is performed.
        /// </summary>
        [Description("Specifies the number of outputs for the layer.  Note, whenb 0, no inner product is performed.")]
        public uint num_output_ip
        {
            get { return m_nNumIpOutput; }
            set { m_nNumIpOutput = value; }
        }

        /// <summary>
        /// Specifies the gradient clipping threshold, default = 0.0 (i.e. no clipping).
        /// </summary>
        [Description("Specifies the gradient clipping threshold, default = 0.0 (i.e. no clipping).")]
        public double clipping_threshold
        {
            get { return m_dfClippingThreshold; }
            set { m_dfClippingThreshold = value; }
        }

        /// <summary>
        /// Specifies the filler parameters for the weight filler.
        /// </summary>
        [Description("Specifies the filler parameters for the weight filler.")]
        [TypeConverter(typeof(ExpandableObjectConverter))]
        public FillerParameter weight_filler
        {
            get { return m_fillerWeights; }
            set { m_fillerWeights = value; }
        }

        /// <summary>
        /// Specifies the filler parameters for the bias filler.
        /// </summary>
        [Description("Specifies the filler parameters for the bias filler.")]
        [TypeConverter(typeof(ExpandableObjectConverter))]
        public FillerParameter bias_filler
        {
            get { return m_fillerBias; }
            set { m_fillerBias = value; }
        }

        /// <summary>
        /// When enabled, the forget gate bias is set to 5.0.
        /// </summary>
        /// <remarks>
        /// @see [A Clockwork RNN](https://arxiv.org/abs/1402.3511) by Koutnik, et al., 2014
        /// </remarks>
        [Description("When true, the forget gate bias is set to 5.0 as recommended by [1] Koutnik, J., Greff, K., Gomez, F., Schmidhuber, J., 'A Clockwork RNN', 2014")]
        public bool enable_clockwork_forgetgate_bias
        {
            get { return m_bEnableClockworkForgetGateBias; }
            set { m_bEnableClockworkForgetGateBias = value; }
        }

        /// <summary>
        /// (default=false) When enabled, attention is applied to the input state on each cycle through the LSTM.  Attention is used with encoder/decoder models.  When disabled,
        /// this layer operates like a standard LSTM layer with input in the shape T,B,I, where T=timesteps, b=batch and i=input.
        /// </summary>
        [Description("When enabled, attention is applied to the input state on each cycle through the LSTM.  Attention is used with encoder/decoder models.")]
        public bool enable_attention
        {
            get { return m_bEnableAttention; }
            set { m_bEnableAttention = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            LSTMAttentionParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            LSTMAttentionParameter p = (LSTMAttentionParameter)src;

            m_nNumOutput = p.m_nNumOutput;
            m_nNumIpOutput = p.m_nNumIpOutput;
            m_dfClippingThreshold = p.m_dfClippingThreshold;
            m_fillerWeights = p.m_fillerWeights.Clone();
            m_fillerBias = p.m_fillerBias.Clone();
            m_bEnableClockworkForgetGateBias = p.m_bEnableClockworkForgetGateBias;
            m_bEnableAttention = p.m_bEnableAttention;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            LSTMAttentionParameter p = new LSTMAttentionParameter();
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Convert the parameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name to associate with the RawProto.</param>
        /// <returns>The new RawProto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("num_output", m_nNumOutput.ToString());
            rgChildren.Add("num_output_ip", m_nNumIpOutput.ToString());

            if (m_dfClippingThreshold != 0)
                rgChildren.Add("clipping_threshold", m_dfClippingThreshold.ToString());

            rgChildren.Add(m_fillerWeights.ToProto("weight_filler"));
            rgChildren.Add(m_fillerBias.ToProto("bias_filler"));

            if (m_bEnableClockworkForgetGateBias != false)
                rgChildren.Add("enable_clockwork_forgetgate_bias", m_bEnableClockworkForgetGateBias.ToString());

            rgChildren.Add("enable_attention", m_bEnableAttention.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static LSTMAttentionParameter FromProto(RawProto rp)
        {
            string strVal;
            LSTMAttentionParameter p = new LSTMAttentionParameter();

            if ((strVal = rp.FindValue("num_output")) != null)
                p.num_output = uint.Parse(strVal);

            if ((strVal = rp.FindValue("num_output_ip")) != null)
                p.num_output_ip = uint.Parse(strVal);

            if ((strVal = rp.FindValue("clipping_threshold")) != null)
                p.clipping_threshold = ParseDouble(strVal);

            RawProto rpWeightFiller = rp.FindChild("weight_filler");
            if (rpWeightFiller != null)
                p.weight_filler = FillerParameter.FromProto(rpWeightFiller);

            RawProto rpBiasFiller = rp.FindChild("bias_filler");
            if (rpBiasFiller != null)
                p.bias_filler = FillerParameter.FromProto(rpBiasFiller);

            if ((strVal = rp.FindValue("enable_clockwork_forgetgate_bias")) != null)
                p.enable_clockwork_forgetgate_bias = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_attention")) != null)
                p.enable_attention = bool.Parse(strVal);

            return p;
        }
    }
}
