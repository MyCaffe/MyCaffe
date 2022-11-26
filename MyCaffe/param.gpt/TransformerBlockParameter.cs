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
    /// </remarks>
    public class TransformerBlockParameter : LayerParameterBase
    {
        int m_nHeads = 6;
        int m_nEmbed = 192;
        double m_dfAttnDropout = 0.1;
        double m_dfResidDropout = 0.1;
        int m_nBlockSize = 128;
        int m_nLayers = 6;
        ACTIVATION m_activation = ACTIVATION.RELU;

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
            GELU_BERT = 2
        }

        /** @copydoc LayerParameterBase */
        public TransformerBlockParameter()
        {
            
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
        /// The number of layers (transformer blocks) used.
        /// </summary>
        [Description("Specifies number of layers (transformer blocks) used.")]
        public int layers
        {
            get { return m_nLayers; }
            set { m_nLayers = value; }
        }

        /// <summary>
        /// The number of heads used.
        /// </summary>
        [Description("Specifies number of heads used.")]
        public int heads
        {
            get { return m_nHeads; }
            set { m_nHeads = value; }
        }

        /// <summary>
        /// Specifies size of the embed.
        /// </summary>
        public int embed
        {
            get { return m_nEmbed; }
            set { m_nEmbed = value; }
        }

        /// <summary>
        /// Specifies size of the block.
        /// </summary>
        public int block_size
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
            m_nBlockSize = p.block_size;
            m_dfAttnDropout = p.attn_dropout;
            m_dfResidDropout = p.resid_dropout;
            m_activation = p.activation;
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
            rgChildren.Add("block_size", block_size.ToString());
            rgChildren.Add("attn_dropout", attn_dropout.ToString());
            rgChildren.Add("resid_dropout", resid_dropout.ToString());
            rgChildren.Add("activation", activation.ToString());

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
                p.layers = int.Parse(strVal);

            if ((strVal = rp.FindValue("heads")) != null)
                p.heads = int.Parse(strVal);
            
            if ((strVal = rp.FindValue("embed")) != null)
                p.embed = int.Parse(strVal);

            if ((strVal = rp.FindValue("block_size")) != null)
                p.block_size = int.Parse(strVal);

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
                else
                    p.activation = ACTIVATION.RELU;
            }

            return p;
        }
    }
}
