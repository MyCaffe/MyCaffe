using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.lnn
{
    /// <summary>
    /// Specifies the parameters used by the CfcLayer.  Note, you must also fill out the CfcUnitParameter.
    /// </summary>
    /// <remarks>
    /// @see [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898) by Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus, 2021, arXiv:2106.13898
    /// @see [Closed-form continuous-time neural networks](https://www.nature.com/articles/s42256-022-00556-7) by Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus, 2021, nature machine intelligence
    /// @see [GitHub:raminmh/CfC](https://github.com/raminmh/CfC) by Raminmn, 2021, GitHub (distributed under Apache 2.0 license)
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class CfcParameter : LayerParameterBase
    {
        int m_nInputFeatures = 0;
        int m_nHiddenSize = 0;
        int m_nOutputFeatures = 0;
        bool m_bReturnSequence = false;

        /** @copydoc LayerParameterBase */
        public CfcParameter()
        {
        }

        /// <summary>
        /// Specifies the number of input features.
        /// </summary>
        [Description("Specifies the number of input features.")]
        public int input_features
        {
            get { return m_nInputFeatures; }
            set { m_nInputFeatures = value; }
        }

        /// <summary>
        /// Specifies the hidden size used to size the backbone units and other internal layers.
        /// </summary>
        [Description("Specifies the hidden size used to size the backbone units and other internal layers.")]
        public int hidden_size
        {
            get { return m_nHiddenSize; }
            set { m_nHiddenSize = value; }
        }

        /// <summary>
        /// Specifies the number of output features
        /// </summary>
        [Description("Specifies the number of output features.")]
        public int output_features
        {
            get { return m_nOutputFeatures; }
            set { m_nOutputFeatures = value; }
        }

        /// <summary>
        /// Specifies whether or not to return the sequence.
        /// </summary>
        [Description("Specifies whether or not to return the sequence.")]
        public bool return_sequences
        {
            get { return m_bReturnSequence; }
            set { m_bReturnSequence = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            CfcParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            CfcParameter p = (CfcParameter)src;

            m_nInputFeatures = p.input_features;
            m_nHiddenSize = p.hidden_size;
            m_nOutputFeatures = p.output_features;
            m_bReturnSequence = p.m_bReturnSequence;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            CfcParameter p = new CfcParameter();
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

            rgChildren.Add("input_features", input_features.ToString());
            rgChildren.Add("hidden_size", hidden_size.ToString());
            rgChildren.Add("output_features", output_features.ToString());
            rgChildren.Add("return_sequences", return_sequences.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static CfcParameter FromProto(RawProto rp)
        {
            string strVal;
            CfcParameter p = new CfcParameter();

            if ((strVal = rp.FindValue("input_features")) != null)
                p.input_features = int.Parse(strVal);

            if ((strVal = rp.FindValue("hidden_size")) != null)
                p.hidden_size = int.Parse(strVal);

            if ((strVal = rp.FindValue("output_features")) != null)
                p.output_features = int.Parse(strVal);

            if ((strVal = rp.FindValue("return_sequences")) != null)
                p.return_sequences = bool.Parse(strVal);

            return p;
        }
    }
}
