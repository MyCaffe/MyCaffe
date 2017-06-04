using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the EluLayer.
    /// </summary>
    /// <remarks>
    /// @see [Deep Residual Networks with Exponential Linear Unit](https://arxiv.org/abs/1604.04112) by Anish Shah, Eashan Kadam, Hena Shah, Sameer Shinde, and Sandip Shingade, 2016.
    /// @see [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289) by Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter, 2015.
    /// </remarks>
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class EluParameter : LayerParameterBase
    {
        double m_dfAlpha = 1.0;

        /** @copydoc LayerParameterBase */
        public EluParameter()
        {
        }

        /// <summary>
        /// Described in [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289) by Clevert, et al., 2015
        /// </summary>
        /// <remarks>
        /// Also see [Deep Residual Networks with Exponential Linear Unit](https://arxiv.org/abs/1604.04112) by Shah, et al., 2016
        /// </remarks>
        [Description("Described in 'Clevert, D. -A, Unterthiner, T., & Hochreiter, S. (2015).  Fast and Accurate Deep Network Learning from Exponential Linear Units (ELUs). arXiv")]
        public double alpha
        {
            get { return m_dfAlpha; }
            set { m_dfAlpha = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            EluParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            EluParameter p = (EluParameter)src;
            m_dfAlpha = p.m_dfAlpha;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            EluParameter p = new EluParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (alpha != 1.0)
                rgChildren.Add("alpha", alpha.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static EluParameter FromProto(RawProto rp)
        {
            string strVal;
            EluParameter p = new EluParameter();

            if ((strVal = rp.FindValue("alpha")) != null)
                p.alpha = double.Parse(strVal);

            return p;
        }
    }
}
