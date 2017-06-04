using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the InfogainLossLayer.
    /// </summary>
    /// <remarks>
    /// @see [DeepGaze II: Reading fixations from deep features trained on object recognition](https://arxiv.org/abs/1610.01563) by Matthias Kümmerer, Thomas S. A. Wallis, and Matthias Bethge, 2016.
    /// </remarks>
    public class InfogainLossParameter : LayerParameterBase
    {
        string m_strSource;
        int m_nAxis = 1;

        /** @copydoc LayerParameterBase */
        public InfogainLossParameter()
        {
        }

        /// <summary>
        /// Specifies the infogain matrix source.
        /// </summary>
        [Description("Specifies the infogain matrix source.")]
        public string source
        {
            get { return m_strSource; }
            set { m_strSource = value; }
        }

        /// <summary>
        /// [\b optional, default = 1] Specifies the axis of the probability.
        /// </summary>
        [Description("Specifies the axis of the probability, default = 1")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            InfogainLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            InfogainLossParameter p = (InfogainLossParameter)src;
            m_strSource = p.m_strSource;
            m_nAxis = p.m_nAxis;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            InfogainLossParameter p = new InfogainLossParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("source", "\"" + source + "\"");

            if (axis != 1)
                rgChildren.Add("axis", axis.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static InfogainLossParameter FromProto(RawProto rp)
        {
            string strVal;
            InfogainLossParameter p = new InfogainLossParameter();

            if ((strVal = rp.FindValue("source")) != null)
                p.source = strVal;

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            return p;
        }
    }
}
