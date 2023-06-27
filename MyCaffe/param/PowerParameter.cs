using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the PowerLayer.
    /// </summary>
    /// <remarks>
    /// The PowerLayer computes outputs:
    /// @f$ y = (shift + scale * x)^power @f$
    /// 
    /// @see [Optimizing a Shallow Multi-Scale Network for Tiny-Imagenet Classification](http://cs231n.stanford.edu/reports/2015/pdfs/dashb_CS231n_Paper.pdf) by Dash Bodington, 2015.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class PowerParameter : LayerParameterBase
    {
        double m_dfPower = 1.0;
        double m_dfScale = 1.0;
        double m_dfShift = 0.0;

        /** @copydoc LayerParameterBase */
        public PowerParameter()
        {
        }

        /// <summary>
        /// Specifies power value in the formula @f$ x = (shift + scale * x)^power @f$.
        /// </summary>
        [Description("Specifies power value in the formula 'x = (shift + scale * x)^power'.")]
        public double power
        {
            get { return m_dfPower; }
            set { m_dfPower = value; }
        }

        /// <summary>
        /// Specifies scale value in the formula @f$ x = (shift + scale * x)^power @f$.
        /// </summary>
        [Description("Specifies scale value in the formula 'x = (shift + scale * x)^power'.")]
        public double scale
        {
            get { return m_dfScale; }
            set { m_dfScale = value; }
        }

        /// <summary>
        /// Specifies shift value in the formula @f$ x = (shift + scale * x)^power @f$.
        /// </summary>
        [Description("Specifies shift value in the formula 'x = (shift + scale * x)^power'.")]
        public double shift
        {
            get { return m_dfShift; }
            set { m_dfShift = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            PowerParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            PowerParameter p = (PowerParameter)src;

            m_dfPower = p.m_dfPower;
            m_dfScale = p.m_dfScale;
            m_dfShift = p.m_dfShift;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            PowerParameter p = new PowerParameter();
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

            rgChildren.Add("power", power.ToString());
            rgChildren.Add("scale", scale.ToString());
            rgChildren.Add("shift", shift.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static PowerParameter FromProto(RawProto rp)
        {
            string strVal;
            PowerParameter p = new PowerParameter();

            if ((strVal = rp.FindValue("power")) != null)
                p.power = ParseDouble(strVal);

            if ((strVal = rp.FindValue("scale")) != null)
                p.scale = ParseDouble(strVal);

            if ((strVal = rp.FindValue("shift")) != null)
                p.shift = ParseDouble(strVal);

            return p;
        }
    }
}
