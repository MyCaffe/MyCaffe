using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.tft
{
    /// <summary>
    /// Specifies the parameters for the MultiHeadAttentionInterpLayer (Interpretable Multi-Head Attention Layer).  
    /// </summary>
    /// <remarks>
    /// The Multi-Headed Attention layer learns long-term relationships across different time-steps.  This version of 
    /// the layer is modified to enhance explainability.  On this modification, the 'values' signal is shared across
    /// all heads - the additive aggregation is employed across all heads.  According to the paper by Lim et al., each
    /// head can learn different temporal patterns, while attending to a common set of input features which can be
    /// interpreted as a simple ensemble over attention weights into a combined matrix, which compared to the  original
    /// multi-head attention matrix, yields an increased representation capacity in an efficient way.
    /// 
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L443) by Playtika Research, 2021.
    /// </remarks>
    public class MultiHeadAttentionInterpParameter : LayerParameterBase
    {
        FillerParameter m_fillerParam_weights = new FillerParameter("xavier");
        FillerParameter m_fillerParam_bias = new FillerParameter("constant", 0.1);
        bool m_bEnableNoise = false;
        double m_dfSigmaInit = 0.017;
        uint m_nEmbedDim;
        uint m_nNumHeads;
        uint m_nNumHistoricalSteps = 0;  
        uint m_nNumFutureSteps = 0;
        bool m_bEnableSelfAttention = true;

        /** @copydoc LayerParameterBase */
        public MultiHeadAttentionInterpParameter()
        {
        }

        /// <summary>
        /// Specifies to enable self attention (one input, default = true).
        /// </summary>
        [Description("Specifies to enable self attention (one input, default = true).")]
        public bool enable_self_attention
        {
            get { return m_bEnableSelfAttention; }
            set { m_bEnableSelfAttention = value; }
        }

        /// <summary>
        /// Specifies the number of historical steps
        /// </summary>
        [Description("Specifies the number of historical steps.")]
        public uint num_historical_steps
        {
            get { return m_nNumHistoricalSteps; }
            set { m_nNumHistoricalSteps = value; }
        }

        /// <summary>
        /// Specifies the number of future steps
        /// </summary>
        [Description("Specifies the number of future steps.")]
        public uint num_future_steps
        {
            get { return m_nNumFutureSteps; }
            set { m_nNumFutureSteps = value; }
        }

        /// <summary>
        /// Specifies the state size corresponding to both the input and output sizes.
        /// </summary>
        [Description("Specifies the state size corresponding to both the input and output sizes.")]
        public uint embed_dim
        {
            get { return m_nEmbedDim; }
            set { m_nEmbedDim = value; }
        }

        /// <summary>
        /// Specifies number of attention heads used in the multi-attention.
        /// </summary>
        [Description("Specifies number of attention heads used in the multi-attention.")]
        public uint num_heads
        {
            get { return m_nNumHeads; }
            set { m_nNumHeads = value; }
        }

        /// <summary>
        /// Enable/disable noise in the inner-product layer (default = false).
        /// </summary>
        /// <remarks>
        /// When enabled, noise is only used during the training phase.
        /// </remarks>
        [Description("Enable/disable noise in the inner-product layer (default = false).")]
        public bool enable_noise
        {
            get { return m_bEnableNoise; }
            set { m_bEnableNoise = value; }
        }

        /// <summary>
        /// Specifies the initialization value for the sigma weight and sigma bias used when 'enable_noise' = <i>true</i>.
        /// </summary>
        [Description("Specifies the initialization value for the sigma weight and sigma bias used when 'enable_noise' = true.")]
        public double sigma_init
        {
            get { return m_dfSigmaInit; }
            set { m_dfSigmaInit = value; }
        }

        /// <summary>
        /// The filler for the weights.
        /// </summary>
        [Category("Fillers")]
        [Description("The filler for the weights.")]
        public FillerParameter weight_filler
        {
            get { return m_fillerParam_weights; }
            set { m_fillerParam_weights = value; }
        }

        /// <summary>
        /// The filler for the bias.
        /// </summary>
        [Category("Fillers")]
        [Description("The filler for the bias.")]
        public FillerParameter bias_filler
        {
            get { return m_fillerParam_bias; }
            set { m_fillerParam_bias = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            MultiHeadAttentionInterpParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            MultiHeadAttentionInterpParameter p = (MultiHeadAttentionInterpParameter)src;

            m_bEnableSelfAttention = p.enable_self_attention;
            m_nNumHistoricalSteps = p.num_historical_steps;
            m_nNumFutureSteps = p.num_future_steps;

            m_nEmbedDim = p.embed_dim;
            m_nNumHeads = p.num_heads;

            if (p.m_fillerParam_bias != null)
                m_fillerParam_bias = p.m_fillerParam_bias.Clone();

            if (p.m_fillerParam_weights != null)
                m_fillerParam_weights = p.m_fillerParam_weights.Clone();

            m_bEnableNoise = p.m_bEnableNoise;
            m_dfSigmaInit = p.m_dfSigmaInit;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            MultiHeadAttentionInterpParameter p = new MultiHeadAttentionInterpParameter();
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

            rgChildren.Add("enable_self_attention", enable_self_attention.ToString());
            rgChildren.Add("num_historical_steps", num_historical_steps.ToString());
            rgChildren.Add("num_future_steps", num_future_steps.ToString());

            rgChildren.Add("embed_dim", embed_dim.ToString());
            rgChildren.Add("num_heads", num_heads.ToString());

            if (weight_filler != null)
                rgChildren.Add(weight_filler.ToProto("weight_filler"));

            if (bias_filler != null)
                rgChildren.Add(bias_filler.ToProto("bias_filler"));

            if (m_bEnableNoise)
            {
                rgChildren.Add("enable_noise", m_bEnableNoise.ToString());
                rgChildren.Add("sigma_init", m_dfSigmaInit.ToString());
            }

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static MultiHeadAttentionInterpParameter FromProto(RawProto rp)
        {
            string strVal;
            MultiHeadAttentionInterpParameter p = new MultiHeadAttentionInterpParameter();

            if ((strVal = rp.FindValue("enable_self_attention")) != null)
                p.enable_self_attention = bool.Parse(strVal);

            if ((strVal = rp.FindValue("embed_dim")) != null)
                p.embed_dim = uint.Parse(strVal);

            if ((strVal = rp.FindValue("num_heads")) != null)
                p.num_heads = uint.Parse(strVal);

            RawProto rpWeightFiller = rp.FindChild("weight_filler");
            if (rpWeightFiller != null)
                p.weight_filler = FillerParameter.FromProto(rpWeightFiller);

            RawProto rpBiasFiller = rp.FindChild("bias_filler");
            if (rpBiasFiller != null)
                p.bias_filler = FillerParameter.FromProto(rpBiasFiller);

            if ((strVal = rp.FindValue("enable_noise")) != null)
                p.enable_noise = bool.Parse(strVal);

            if ((strVal = rp.FindValue("sigma_init")) != null)
                p.sigma_init = ParseDouble(strVal);

            if ((strVal = rp.FindValue("num_historical_steps")) != null)
                p.num_historical_steps = uint.Parse(strVal);

            if ((strVal = rp.FindValue("num_future_steps")) != null)
                p.num_future_steps = uint.Parse(strVal);

            return p;
        }
    }
}
