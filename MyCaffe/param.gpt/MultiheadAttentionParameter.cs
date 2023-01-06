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
    /// <remarks>
    /// </remarks>
    public class MultiheadAttentionParameter : LayerParameterBase
    {
        uint m_nHeads = 6;
        uint m_nEmbed = 192;     // d_model
        double m_dfAttnDropout;
        double m_dfResidDropout;
        uint m_nBlockSize = 128;
        uint m_nLayers = 6;
        WEIGHT_INIT m_weightInit = WEIGHT_INIT.ENCODER_DECODER;

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

            return p;
        }
    }
}
