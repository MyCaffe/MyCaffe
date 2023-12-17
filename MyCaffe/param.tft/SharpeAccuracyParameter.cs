using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.tft
{
    /// <summary>
    /// Specifies the parameters for the SharpeAccuracyLayer used in TFT models
    /// </summary>
    /// <remarks>
    /// @see [Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture](https://arxiv.org/abs/2112.08534) by Kieran Wood, Sven Giegerich, Stephen Roberts, and Stefan Zohren, 2022, arXiv:2112.08534
    /// @see [Github - kieranjwood/trading-momentum-transformer](https://github.com/kieranjwood/trading-momentum-transformerh) by Kieran Wood, 2022.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class SharpeAccuracyParameter : LayerParameterBase
    {
        ACCURACY_TYPE m_type = ACCURACY_TYPE.SHARPE;

        /// <summary>
        /// Defines the accuracy type to return.
        /// </summary>
        public enum ACCURACY_TYPE
        {
            /// <summary>
            /// Specifies to return the sharpe ratio.
            /// </summary>
            SHARPE = 0,
            /// <summary>
            /// Specifies to return the actual returns.
            /// </summary>
            RETURNS = 1
        }

        /** @copydoc LayerParameterBase */
        public SharpeAccuracyParameter()
        {
        }

        /// <summary>
        /// Specifies the accuracy type to return.
        /// </summary>
        [Description("Specifies the accuracy type to return.")]
        public ACCURACY_TYPE accuracy_type
        {
            get { return m_type; }
            set { m_type = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            SharpeAccuracyParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            SharpeAccuracyParameter p = (SharpeAccuracyParameter)src;
            m_type = p.m_type;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            SharpeAccuracyParameter p = new SharpeAccuracyParameter();
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

            rgChildren.Add("accuracy_type", accuracy_type.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static SharpeAccuracyParameter FromProto(RawProto rp)
        {
            string strVal;
            SharpeAccuracyParameter p = new SharpeAccuracyParameter();

            if ((strVal = rp.FindValue("accuracy_type")) != null)
            {
                if (!Enum.TryParse<ACCURACY_TYPE>(strVal, out p.m_type))
                    p.m_type = ACCURACY_TYPE.SHARPE;
            }

            return p;
        }
    }
}
