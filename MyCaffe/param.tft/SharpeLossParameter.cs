using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.tft
{
    /// <summary>
    /// Specifies the parameters for the SharpeLossLayer used in TFT models
    /// </summary>
    /// <remarks>
    /// @see [Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture](https://arxiv.org/abs/2112.08534) by Kieran Wood, Sven Giegerich, Stephen Roberts, and Stefan Zohren, 2022, arXiv:2112.08534
    /// @see [Github - kieranjwood/trading-momentum-transformer](https://github.com/kieranjwood/trading-momentum-transformerh) by Kieran Wood, 2022.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class SharpeLossParameter : LayerParameterBase
    {
        bool m_bAnnualize = true;

        /** @copydoc LayerParameterBase */
        public SharpeLossParameter()
        {
        }

        /// <summary>
        /// Specifies to annualize the loss calculations.
        /// </summary>
        [Description("Specifies to annualize the loss calculations (default = true).")]
        public bool annualize
        {
            get { return m_bAnnualize; }
            set { m_bAnnualize = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            SharpeLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            SharpeLossParameter p = (SharpeLossParameter)src;
            m_bAnnualize = p.m_bAnnualize;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            SharpeLossParameter p = new SharpeLossParameter();
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

            rgChildren.Add("annualize", annualize.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static SharpeLossParameter FromProto(RawProto rp)
        {
            string strVal;
            SharpeLossParameter p = new SharpeLossParameter();

            if ((strVal = rp.FindValue("annualize")) != null)
                p.annualize = bool.Parse(strVal);

            return p;
        }
    }
}
