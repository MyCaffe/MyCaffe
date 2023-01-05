using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.gpt
{
    /// <summary>
    /// Specifies the parameters for the PositionalEncoderLayer.
    /// </summary>
    /// <remarks>
    /// </remarks>
    public class PositionalEncodingParameter : LayerParameterBase
    {
        int m_nEmbed = 192;     // d_model
        int m_nBlockSize = 128;

        /** @copydoc LayerParameterBase */
        public PositionalEncodingParameter()
        {
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

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            PositionalEncodingParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            PositionalEncodingParameter p = (PositionalEncodingParameter)src;

            m_nEmbed = p.embed;
            m_nBlockSize = p.block_size;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            PositionalEncodingParameter p = new PositionalEncodingParameter();
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

            rgChildren.Add("embed", embed.ToString());
            rgChildren.Add("block_size", block_size.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static PositionalEncodingParameter FromProto(RawProto rp)
        {
            string strVal;
            PositionalEncodingParameter p = new PositionalEncodingParameter();

            if ((strVal = rp.FindValue("embed")) != null)
                p.embed = int.Parse(strVal);

            if ((strVal = rp.FindValue("block_size")) != null)
                p.block_size = int.Parse(strVal);

            return p;
        }
    }
}
