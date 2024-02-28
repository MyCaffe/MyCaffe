using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.gpt
{
    /// <summary>
    /// Specifies the parameters for the TransformerBlockLayer.
    /// </summary>
    /// <remarks>
    /// When using with Llama models, the following configuation is used:
    ///     block_type = CAUSAL_SELF_ATTENTION2
    ///     normalization_type = RMS_NORM
    ///     activation = SILU
    ///     enable_rotary_positional_embedding = true
    ///     enable_llama_style_head = true
    ///     bias_term = false
    ///     multiple_of = 256
    ///     hidden_dim = 0
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class TransformerBlockParameter : LayerParameterBase
    {
        uint m_nHeads = 6;
        uint m_nEmbed = 192;
        uint m_nHiddenDim = 0; // When 0, embed * 4 is used.
        double m_dfAttnDropout = 0.1;
        double m_dfResidDropout = 0.1;
        uint m_nBlockSize = 128;
        uint m_nLayers = 6;
        uint m_nMultipleOf = 64;
        ACTIVATION m_activation = ACTIVATION.RELU;
        BLOCK_TYPE m_type = BLOCK_TYPE.CAUSAL_SELF_ATTENTION;
        NORMALIZATION m_normalization = NORMALIZATION.LAYER_NORM;
        bool m_bEnableLayerNormCudaImplementation = false;
        bool m_bEnableCudaScaledDotProductAttention = false;
        bool m_bEnableRotaryPositionalEmbedding = false;
        bool m_bEnableLlamaStyleHead = false;
        bool m_bBiasTerm = true;
        OutputAdapterParameter m_output_adapter_q = new OutputAdapterParameter("q");
        OutputAdapterParameter m_output_adapter_k = new OutputAdapterParameter("k");
        OutputAdapterParameter m_output_adapter_v = new OutputAdapterParameter("v");
        OutputAdapterParameter m_output_adapter_out = new OutputAdapterParameter("out");

        /// <summary>
        /// Defines the type of transformer block
        /// </summary>
        public enum BLOCK_TYPE
        {
            /// <summary>
            /// Specifies to configure a causal self attention block.
            /// </summary>
            CAUSAL_SELF_ATTENTION,
            /// <summary>
            /// Same as CAUSAL_SELF_ATTENTION, but internally uses MultiheadAttentionLayer which supports LoRA.
            /// </summary>
            CAUSAL_SELF_ATTENTION2,
            /// <summary>
            /// Specifies to configure an encoder transformer block.
            /// </summary>
            ENCODER,
            /// <summary>
            /// Specifies to configure a decoder transformer block
            /// </summary>
            DECODER
        }

        /// <summary>
        /// Defines the various activations supported by the TransformerBlock.
        /// </summary>
        public enum ACTIVATION
        {
            /// <summary>
            /// Specifies to use the RELU activation (default)
            /// </summary>
            RELU = 0,
            /// <summary>
            /// Specifies to use the GELU activation.
            /// </summary>
            GELU = 1,
            /// <summary>
            /// Specifies to use the special GELU activation used in BERT models.
            /// </summary>
            GELU_BERT = 2,
            /// <summary>
            /// Specifies to use the SiLU activation (used with Llama models).
            /// </summary>
            SILU = 3
        }

        /// <summary>
        /// Defines the various normalization types supported by the TransformerBlock.
        /// </summary>        
        public enum NORMALIZATION
        {
            /// <summary>
            /// Specifies to use the LayerNorm normalization (default)
            /// </summary>
            LAYER_NORM = 0,
            /// <summary>
            /// Specifies to use the RMSNorm normalization.
            /// </summary>
            RMS_NORM = 1
        }

        /** @copydoc LayerParameterBase */
        public TransformerBlockParameter()
        {            
        }

        /// <summary>
        /// Specifies to use the low-level full cuda implementation of LayerNorm (default = false).
        /// </summary>
        /// <remarks>
        /// The cuda implementation runs around 30% faster when using float base types.
        /// </remarks>
        public bool enable_layernorm_cuda_impl
        {
            get { return m_bEnableLayerNormCudaImplementation; }
            set { m_bEnableLayerNormCudaImplementation = value; }
        }

        /// <summary>
        /// Specifies the activation type to use (default = RELU)
        /// </summary>
        public ACTIVATION activation
        {
            get { return m_activation; }
            set { m_activation = value; }
        }

        /// <summary>
        /// Specifies the type of transformer block to configure.
        /// </summary>
        public BLOCK_TYPE block_type
        {
            get { return m_type; }
            set { m_type = value; }
        }

        /// <summary>
        /// Specifies the normalization type to use.
        /// </summary>
        [Description("Specifies the normalization type to use.")]
        public NORMALIZATION normalization_type
        {
            get { return m_normalization; }
            set { m_normalization = value; }
        }

        /// <summary>
        /// The number of layers (transformer blocks) used.
        /// </summary>
        [Description("Specifies number of layers (transformer blocks) used.")]
        public uint layers
        {
            get { return m_nLayers; }
            set { m_nLayers = value; }
        }

        /// <summary>
        /// The number of heads used.
        /// </summary>
        [Description("Specifies number of heads used.")]
        public uint heads
        {
            get { return m_nHeads; }
            set { m_nHeads = value; }
        }

        /// <summary>
        /// Specifies size of the embed.
        /// </summary>
        public uint embed
        {
            get { return m_nEmbed; }
            set { m_nEmbed = value; }
        }

        /// <summary>
        /// Specifies size of the hidden_dim.  When 0, embed * 4 is used.
        /// </summary>
        public uint hidden_dim
        {
            get { return m_nHiddenDim; }
            set { m_nHiddenDim = value; }
        }

        /// <summary>
        /// Specifies size of the block.
        /// </summary>
        public uint block_size
        {
            get { return m_nBlockSize; }
            set { m_nBlockSize = value; }
        }
        
        /// <summary>
        /// Specifies dropout probability used on the attention weights.
        /// </summary>
        public double attn_dropout
        {
            get { return m_dfAttnDropout; }
            set { m_dfAttnDropout = value; }
        }

        /// <summary>
        /// Specifies dropout probability used on the residual weights.
        /// </summary>
        public double resid_dropout
        {
            get { return m_dfResidDropout; }
            set { m_dfResidDropout = value; }
        }

        /// <summary>
        /// Specifies whether or not to enable the CudaScaledDotProductAttention.  When enabled, the scaled dot product attention is computed at the CUDA level.
        /// </summary>
        [Description("Specifies whether or not to enable the CudaScaledDotProductAttention.  When enabled, the scaled dot product attention is computed at the CUDA level.")]
        public bool enable_cuda_scaled_dot_product_attention
        {
            get { return m_bEnableCudaScaledDotProductAttention; }
            set { m_bEnableCudaScaledDotProductAttention = value; }
        }

        /// <summary>
        /// Specifies whether or not to enable the rotary positional embedding.
        /// </summary>
        [Description("Specifies whether or not to enable the rotary positional embedding.")]
        public bool enable_rotary_positional_embedding
        {
            get { return m_bEnableRotaryPositionalEmbedding; }
            set { m_bEnableRotaryPositionalEmbedding = value; }
        }

        /// <summary>
        /// Specifies whether or not to enable the llama style head.  When using the Llama style head, the normalized output of the attention 
        /// is output as:
        ///     x = w2(w1(x) * w3(act(x))), 
        /// whereas when using the standard head, the normalized output of the attention is output as:
        ///     x = w2(act(w1(x)))
        /// </summary>
        public bool enable_llama_style_head
        {
            get { return m_bEnableLlamaStyleHead; }
            set { m_bEnableLlamaStyleHead = value; }
        }

        /// <summary>
        /// Specifies to use a bias term in the multihead attention layer Linear layers (default = true).
        /// </summary>
        [Description("Specifies to use a bias term in the multihead attention layer Linear layers (default = true).")]
        public bool bias_term
        {
            get { return m_bBiasTerm; }
            set { m_bBiasTerm = value; }
        }

        /// <summary>
        /// Specifies the multiple of the dim used to calculate internal MLP hidden dimensions.
        /// </summary>
        /// <remarks>
        /// When using Llama models, the multiple_of is set to 256 and used to calculate the hidden dimensions of the internal MLP layers
        /// using the following formula:
        ///     hidden_dim = (int)(2 * (dim * 4) / 3)
        ///     hidden_dim = multiple_of * (int)((hidden_dim + multiple_of-1) / multiple_of)
        /// </remarks>
        [Description("Specifies the multiple of the dim used to calculate internal MLP hidden dimensions.")]
        public uint multiple_of
        {
            get { return m_nMultipleOf; }
            set { m_nMultipleOf = value; }
        }

        /// <summary>
        /// Specifies the output adapter for the 'q' Linear layer.
        /// </summary>
        [Description("Specifies the output adapter for the 'q' Linear layer.")]
        public OutputAdapterParameter output_adapter_q
        {
            get { return m_output_adapter_q; }
            set { m_output_adapter_q = value; }
        }

        /// <summary>
        /// Specifies the output adapter for the 'q' Linear layer.
        /// </summary>
        [Description("Specifies the output adapter for the 'k' Linear layer.")]
        public OutputAdapterParameter output_adapter_k
        {
            get { return m_output_adapter_k; }
            set { m_output_adapter_k = value; }
        }

        /// <summary>
        /// Specifies the output adapter for the 'v' Linear layer.
        /// </summary>
        [Description("Specifies the output adapter for the 'v' Linear layer.")]
        public OutputAdapterParameter output_adapter_v
        {
            get { return m_output_adapter_v; }
            set { m_output_adapter_v = value; }
        }

        /// <summary>
        /// Specifies the output adapter for the 'out' Linear layer.
        /// </summary>
        [Description("Specifies the output adapter for the 'out' Linear layer.")]
        public OutputAdapterParameter output_adapter_out
        {
            get { return m_output_adapter_out; }
            set { m_output_adapter_out = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            TransformerBlockParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            TransformerBlockParameter p = (TransformerBlockParameter)src;

            m_nLayers = p.layers;
            m_nHeads = p.heads;
            m_nEmbed = p.embed;
            m_nHiddenDim = p.hidden_dim;
            m_nBlockSize = p.block_size;
            m_dfAttnDropout = p.attn_dropout;
            m_dfResidDropout = p.resid_dropout;
            m_activation = p.activation;
            m_type = p.block_type;
            m_bEnableLayerNormCudaImplementation = p.enable_layernorm_cuda_impl;
            m_bEnableCudaScaledDotProductAttention = p.enable_cuda_scaled_dot_product_attention;
            m_bEnableRotaryPositionalEmbedding = p.enable_rotary_positional_embedding;
            m_bEnableLlamaStyleHead = p.enable_llama_style_head;
            m_output_adapter_q = p.output_adapter_q.Clone();
            m_output_adapter_k = p.output_adapter_k.Clone();
            m_output_adapter_v = p.output_adapter_v.Clone();
            m_output_adapter_out = p.output_adapter_out.Clone();
            m_bBiasTerm = p.bias_term;
            m_normalization = p.normalization_type;
            m_nMultipleOf = p.multiple_of;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            TransformerBlockParameter p = new TransformerBlockParameter();
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

            rgChildren.Add("layers", layers.ToString());
            rgChildren.Add("heads", heads.ToString());
            rgChildren.Add("embed", embed.ToString());
            rgChildren.Add("hidden_dim", hidden_dim.ToString());
            rgChildren.Add("block_size", block_size.ToString());
            rgChildren.Add("attn_dropout", attn_dropout.ToString());
            rgChildren.Add("resid_dropout", resid_dropout.ToString());
            rgChildren.Add("activation", activation.ToString());
            rgChildren.Add("block_type", block_type.ToString());
            rgChildren.Add("normaliation", normalization_type.ToString());
            rgChildren.Add("enable_ln_cuda_impl", enable_layernorm_cuda_impl.ToString());
            rgChildren.Add("enable_cuda_scaled_dot_product_attention", enable_cuda_scaled_dot_product_attention.ToString());
            rgChildren.Add("enable_rotary_positional_embedding", enable_rotary_positional_embedding.ToString());
            rgChildren.Add("enable_llama_style_head", enable_llama_style_head.ToString());
            rgChildren.Add("bias_term", bias_term.ToString());
            rgChildren.Add("multiple_of", multiple_of.ToString());
            rgChildren.Add(output_adapter_q.ToProto("output_adapter_q"));
            rgChildren.Add(output_adapter_k.ToProto("output_adapter_k"));
            rgChildren.Add(output_adapter_v.ToProto("output_adapter_v"));
            rgChildren.Add(output_adapter_out.ToProto("output_adapter_out"));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static TransformerBlockParameter FromProto(RawProto rp)
        {
            string strVal;
            TransformerBlockParameter p = new TransformerBlockParameter();

            if ((strVal = rp.FindValue("layers")) != null)
                p.layers = uint.Parse(strVal);

            if ((strVal = rp.FindValue("heads")) != null)
                p.heads = uint.Parse(strVal);
            
            if ((strVal = rp.FindValue("embed")) != null)
                p.embed = uint.Parse(strVal);

            if ((strVal = rp.FindValue("hidden_dim")) != null)
                p.hidden_dim = uint.Parse(strVal);

            if ((strVal = rp.FindValue("block_size")) != null)
                p.block_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("attn_dropout")) != null)
                p.attn_dropout = double.Parse(strVal);

            if ((strVal = rp.FindValue("resid_dropout")) != null)
                p.resid_dropout = double.Parse(strVal);

            if ((strVal = rp.FindValue("activation")) != null)
            {
                if (strVal == ACTIVATION.GELU.ToString())
                    p.activation = ACTIVATION.GELU;
                else if (strVal == ACTIVATION.GELU_BERT.ToString())
                    p.activation = ACTIVATION.GELU_BERT;
                else if (strVal == ACTIVATION.SILU.ToString())
                    p.activation = ACTIVATION.SILU;
                else
                    p.activation = ACTIVATION.RELU;
            }

            if ((strVal = rp.FindValue("block_type")) != null)
            {
                if (strVal == BLOCK_TYPE.CAUSAL_SELF_ATTENTION.ToString())
                    p.block_type = BLOCK_TYPE.CAUSAL_SELF_ATTENTION;
                else if (strVal == BLOCK_TYPE.CAUSAL_SELF_ATTENTION2.ToString())
                    p.block_type = BLOCK_TYPE.CAUSAL_SELF_ATTENTION2;
                else if (strVal == BLOCK_TYPE.ENCODER.ToString())
                    p.block_type = BLOCK_TYPE.ENCODER;
                else if (strVal == BLOCK_TYPE.DECODER.ToString())
                    p.block_type = BLOCK_TYPE.DECODER;
                else
                    throw new Exception("Unknown block type '" + strVal + "' found.");
            }

            if ((strVal = rp.FindValue("normaliation")) != null)
            {
                if (strVal == NORMALIZATION.RMS_NORM.ToString())
                    p.normalization_type = NORMALIZATION.RMS_NORM;
                else
                    p.normalization_type = NORMALIZATION.LAYER_NORM;
            }

            if ((strVal = rp.FindValue("enable_ln_cuda_impl")) != null)
                p.enable_layernorm_cuda_impl = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_cuda_scaled_dot_product_attention")) != null)
                p.enable_cuda_scaled_dot_product_attention = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_rotary_positional_embedding")) != null)
                p.enable_rotary_positional_embedding = bool.Parse(strVal);

            if ((strVal = rp.FindValue("bias_term")) != null)
                p.bias_term = bool.Parse(strVal);

            if ((strVal = rp.FindValue("multiple_of")) != null)
                p.multiple_of = uint.Parse(strVal);

            if ((strVal = rp.FindValue("enable_llama_style_head")) != null)
                p.enable_llama_style_head = bool.Parse(strVal);

            RawProto rp1 = rp.FindChild("output_adapter_q");
            if (rp1 != null)
                p.output_adapter_q = OutputAdapterParameter.FromProto(rp1);

            rp1 = rp.FindChild("output_adapter_k");
            if (rp1 != null)
                p.output_adapter_k = OutputAdapterParameter.FromProto(rp1);

            rp1 = rp.FindChild("output_adapter_v");
            if (rp1 != null)
                p.output_adapter_v = OutputAdapterParameter.FromProto(rp1);

            rp1 = rp.FindChild("output_adapter_out");
            if (rp1 != null)
                p.output_adapter_out = OutputAdapterParameter.FromProto(rp1);

            return p;
        }
    }
}
