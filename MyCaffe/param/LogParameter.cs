using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the LogLayer.
    /// </summary>
    /// <remarks>
    /// The LogLayer computes outputs:
    /// 
    /// @f$ y = log_base(shift + scale * x) @f$, for base > 0.
    /// 
    /// Or if base is set to the default (-1), base is set to e.,
    /// so:
    /// 
    /// @f$ y = ln(shift + scale * x) @f$
    /// </remarks>
    public class LogParameter : LayerParameterBase
    {
        double m_dfBase = -1.0;
        double m_dfScale = 1.0;
        double m_dfShift = 0.0;

        /** @copydoc LayerParameterBase */
        public LogParameter()
        {
        }

        /// <summary>
        /// Specifies the base to use for the log, where @f$ y = log_base(shift + scale * x) @f$, for base > 0.
        /// </summary>
        [Description("Specifies the base to use for the log, where y=log_base(shift + scale *x), for base > 0.")]
        public double base_val
        {
            get { return m_dfBase; }
            set { m_dfBase = value; }
        }

        /// <summary>
        /// Specifies the scale to use for the log, where @f$ y = log_base(shift + scale * x) @f$, for base > 0.
        /// </summary>
        [Description("Specifies the scale to use for the log, where 'y = log_babse(shift + scale * x), for base > 0.")]
        public double scale
        {
            get { return m_dfScale; }
            set { m_dfScale = value; }
        }

        /// <summary>
        /// Specifies the shift to use for the log, where @f$ y = log_base(shift + scale * x) @f$, for base > 0.
        /// </summary>
        [Description("Specifies the shift to use for the log, where 'y = log_babse(shift + scale * x), for base > 0.")]
        public double shift
        {
            get { return m_dfShift; }
            set { m_dfShift = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            LogParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            LogParameter p = (LogParameter)src;
            m_dfBase = p.m_dfBase;
            m_dfScale = p.m_dfScale;
            m_dfShift = p.m_dfShift;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            LogParameter p = new LogParameter();
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

            rgChildren.Add("base", base_val.ToString());

            if (scale != 1.0)
                rgChildren.Add("scale", scale.ToString());

            if (shift != 0)
                rgChildren.Add("shift", shift.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static LogParameter FromProto(RawProto rp)
        {
            string strVal;
            LogParameter p = new LogParameter();

            if ((strVal = rp.FindValue("base")) != null)
                p.base_val = ParseDouble(strVal);

            if ((strVal = rp.FindValue("scale")) != null)
                p.scale = ParseDouble(strVal);

            if ((strVal = rp.FindValue("shift")) != null)
                p.shift = ParseDouble(strVal);

            return p;
        }
    }
}
