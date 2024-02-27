using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.gpt
{
    /// <summary>
    /// Specifies the parameters for the MultiheadAttentionLayer.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class MultiheadAttentionParameter : LayerParameterBase
    {
        uint m_nHeads = 6;
        uint m_nEmbed = 192;     // d_model
        double m_dfAttnDropout;
        double m_dfResidDropout;
        uint m_nBlockSize = 128;
        uint m_nLayers = 6;
        WEIGHT_INIT m_weightInit = WEIGHT_INIT.ENCODER_DECODER;
        OutputAdapterParameter m_output_adapter_q = new OutputAdapterParameter("q");
        OutputAdapterParameter m_output_adapter_k = new OutputAdapterParameter("k");
        OutputAdapterParameter m_output_adapter_v = new OutputAdapterParameter("v");
        OutputAdapterParameter m_output_adapter_out = new OutputAdapterParameter("out");
        bool m_bEnableFlashScaledDotProductAttention = false;
        bool m_bEnableRotaryPositionalEmbedding = false;
        int m_nRopeSharedIndex = 1;
        bool m_bBiasTerm = true;

        /// <summary>
        /// Defines the weight initialization strategy.
        /// </summary>
        public enum WEIGHT_INIT
        {
            /// <summary>
            /// Specifies to use the GPT style weight strategy.
            /// </summary>
            GPT,
            /// <summary>
            /// Specifies to use the XAVIER initialization on both weight and bias.
            /// </summary>
            ENCODER_DECODER
        }

        /** @copydoc LayerParameterBase */
        public MultiheadAttentionParameter()
        {
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
        /// Specifies whether or not to enable the FlashScaledDotProductAttention.  When enabled, the scaled dot product attention is computed at the CUDA level.
        /// </summary>
        [Description("Specifies whether or not to enable the FlashScaledDotProductAttention.  When enabled, the scaled dot product attention is computed at the CUDA level.")]
        public bool enable_flash_scaled_dot_product_attention
        {
            get { return m_bEnableFlashScaledDotProductAttention; }
            set { m_bEnableFlashScaledDotProductAttention = value; }
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
        /// Specifies the rope shared index so that only one rope is used for all layers. To use a unique rope for each layer, set this value to -1.
        /// </summary>
        [Description("Specifies the rope shared index so that only one rope is used for all layers. To use a unique rope for each layer, set this value to -1.")]
        public int rope_shared_index
        {
            get { return m_nRopeSharedIndex; }
            set { m_nRopeSharedIndex = value; }
        }

        /// <summary>
        /// Specifies whether or not to use a bias term on wq, wk, wv, and wo.
        /// </summary>
        [Description("Specifies whether or not to use a bias term on wq, wk, wv, and wo.")]
        public bool bias_term
        {
            get { return m_bBiasTerm; }
            set { m_bBiasTerm = value; }
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
        /// Specifies the weight initialization strategy (default = ENCODER_DECODER).
        /// </summary>
        public WEIGHT_INIT weight_init
        {
            get { return m_weightInit; }
            set { m_weightInit = value; }
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
            MultiheadAttentionParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            MultiheadAttentionParameter p = (MultiheadAttentionParameter)src;

            m_nLayers = p.layers;
            m_nHeads = p.heads;
            m_nEmbed = p.embed;
            m_nBlockSize = p.block_size;
            m_dfAttnDropout = p.attn_dropout;
            m_dfResidDropout = p.resid_dropout;
            m_weightInit = p.weight_init;
            m_output_adapter_q = p.output_adapter_q.Clone();
            m_output_adapter_k = p.output_adapter_k.Clone();
            m_output_adapter_v = p.output_adapter_v.Clone();
            m_output_adapter_out = p.output_adapter_out.Clone();
            m_bEnableFlashScaledDotProductAttention = p.enable_flash_scaled_dot_product_attention;
            m_bEnableRotaryPositionalEmbedding = p.enable_rotary_positional_embedding;
            m_nRopeSharedIndex = p.rope_shared_index;
            m_bBiasTerm = p.bias_term;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            MultiheadAttentionParameter p = new MultiheadAttentionParameter();
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
            rgChildren.Add("block_size", block_size.ToString());
            rgChildren.Add("attn_dropout", attn_dropout.ToString());
            rgChildren.Add("resid_dropout", resid_dropout.ToString());
            rgChildren.Add("weight_init", weight_init.ToString());
            rgChildren.Add("enable_flash_scaled_dot_product_attention", enable_flash_scaled_dot_product_attention.ToString());
            rgChildren.Add("enable_rotary_positional_embedding", enable_rotary_positional_embedding.ToString());
            rgChildren.Add("rope_shared_index", rope_shared_index.ToString());
            rgChildren.Add("bias_term", bias_term.ToString());
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
        public static MultiheadAttentionParameter FromProto(RawProto rp)
        {
            string strVal;
            MultiheadAttentionParameter p = new MultiheadAttentionParameter();

            if ((strVal = rp.FindValue("layers")) != null)
                p.layers = uint.Parse(strVal);
            
            if ((strVal = rp.FindValue("heads")) != null)
                p.heads = uint.Parse(strVal);
            
            if ((strVal = rp.FindValue("embed")) != null)
                p.embed = uint.Parse(strVal);

            if ((strVal = rp.FindValue("block_size")) != null)
                p.block_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("attn_dropout")) != null)
                p.attn_dropout = double.Parse(strVal);

            if ((strVal = rp.FindValue("resid_dropout")) != null)
                p.resid_dropout = double.Parse(strVal);

            if ((strVal = rp.FindValue("weight_init")) != null)
            {
                if (strVal == WEIGHT_INIT.GPT.ToString())
                    p.weight_init = WEIGHT_INIT.GPT;
                else if (strVal == WEIGHT_INIT.ENCODER_DECODER.ToString())
                    p.weight_init = WEIGHT_INIT.ENCODER_DECODER;
                else
                    throw new Exception("Unknown weight init strategy '" + strVal + "'!");
            }
            
            if ((strVal = rp.FindValue("enable_flash_scaled_dot_product_attention")) != null)
                p.enable_flash_scaled_dot_product_attention = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_rotary_positional_embedding")) != null)
                p.enable_rotary_positional_embedding = bool.Parse(strVal);

            if ((strVal = rp.FindValue("rope_shared_index")) != null)
                p.rope_shared_index = int.Parse(strVal);

            if ((strVal = rp.FindValue("bias_term")) != null)
                p.bias_term = bool.Parse(strVal);

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
