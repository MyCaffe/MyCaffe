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
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class PositionalEncoderParameter : LayerParameterBase
    {
        uint m_nEmbed = 192;     // d_model
        uint m_nBlockSize = 128;

        /** @copydoc LayerParameterBase */
        public PositionalEncoderParameter()
        {
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

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            PositionalEncoderParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            PositionalEncoderParameter p = (PositionalEncoderParameter)src;

            m_nEmbed = p.embed;
            m_nBlockSize = p.block_size;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            PositionalEncoderParameter p = new PositionalEncoderParameter();
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
        public static PositionalEncoderParameter FromProto(RawProto rp)
        {
            string strVal;
            PositionalEncoderParameter p = new PositionalEncoderParameter();

            if ((strVal = rp.FindValue("embed")) != null)
                p.embed = uint.Parse(strVal);

            if ((strVal = rp.FindValue("block_size")) != null)
                p.block_size = uint.Parse(strVal);

            return p;
        }
    }
}
