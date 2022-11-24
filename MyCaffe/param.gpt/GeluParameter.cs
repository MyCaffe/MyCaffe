using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.gpt
{
    /// <summary>
    /// Specifies the parameters for the GeluLayer.
    /// </summary>
    /// <remarks>
    /// </remarks>
    public class GeluParameter : LayerParameterBase
    {
        bool m_bEnableBertVersion = false;

        /** @copydoc LayerParameterBase */
        public GeluParameter()
        {
        }

        /// <summary>
        /// Specifies to use the special BERT version used in GPT models.
        /// </summary>
        [Description("Specifies to use the special BERT version used in GPT models.")]
        public bool enable_bert_version
        {
            get { return m_bEnableBertVersion; }
            set { m_bEnableBertVersion = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            GeluParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            GeluParameter p = (GeluParameter)src;

            m_bEnableBertVersion = p.enable_bert_version;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            GeluParameter p = new GeluParameter();
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

            rgChildren.Add("enable_bert_version", enable_bert_version.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static GeluParameter FromProto(RawProto rp)
        {
            string strVal;
            GeluParameter p = new GeluParameter();

            if ((strVal = rp.FindValue("enable_bert_version")) != null)
                p.enable_bert_version = bool.Parse(strVal);

            return p;
        }
    }
}
