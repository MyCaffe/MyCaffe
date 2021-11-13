using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the ConvolutionOctaveLayer.
    /// </summary>
    /// <remarks>
    /// </remarks>
    public class ConvolutionOctaveParameter : ConvolutionParameter
    {
        double m_dfAlphaIn = 0.5;
        double m_dfAlphaOut = 0.5;

        /** @copydoc LayerParameterBase */
        public ConvolutionOctaveParameter()
        {
        }

        /// <summary>
        /// Specifies alpha applied to the input channels.
        /// </summary>
        [Description("Specifies the alpha applied to the input channels.")]
        public double alpha_in
        {
            get { return m_dfAlphaIn; }
            set { m_dfAlphaIn = value; }
        }

        /// <summary>
        /// Specifies alpha applied to the output channels.
        /// </summary>
        [Description("Specifies the alpha applied to the output channels.")]
        public double alpha_out
        {
            get { return m_dfAlphaOut; }
            set { m_dfAlphaOut = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ConvolutionOctaveParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is ConvolutionOctaveParameter)
            {
                ConvolutionOctaveParameter p = (ConvolutionOctaveParameter)src;
                m_dfAlphaIn = p.alpha_in;
                m_dfAlphaOut = p.alpha_out;
            }
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ConvolutionOctaveParameter p = new ConvolutionOctaveParameter();
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
            RawProto rpBase = base.ToProto("convolution");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);
            rgChildren.Add("alpha_in", alpha_in.ToString());
            rgChildren.Add("alpha_out", alpha_out.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new ConvolutionOctaveParameter FromProto(RawProto rp)
        {
            string strVal;
            ConvolutionOctaveParameter p = new ConvolutionOctaveParameter();

            ((ConvolutionParameter)p).Copy(ConvolutionParameter.FromProto(rp));

            if ((strVal = rp.FindValue("alpha_in")) != null)
                p.alpha_in = double.Parse(strVal);

            if ((strVal = rp.FindValue("alpha_out")) != null)
                p.alpha_out = double.Parse(strVal);

            return p;
        }
    }
}
