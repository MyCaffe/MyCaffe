using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the MathLayer.
    /// </summary>
    /// <remarks>
    /// The MathLayer computes outputs:
    /// 
    /// @f$ y = function @f$, where function is a mathematical function.
    /// </remarks> 
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class MathParameter : LayerParameterBase
    {
        MyCaffe.common.MATH_FUNCTION m_function = MyCaffe.common.MATH_FUNCTION.NOP;

        /** @copydoc LayerParameterBase */
        public MathParameter()
        {
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            MathParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Get/set the function to run.
        /// </summary>
        public MyCaffe.common.MATH_FUNCTION function
        {
            get { return m_function; }
            set { m_function = value; }
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            MathParameter p = (MathParameter)src;
            m_function = p.m_function;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            MathParameter p = new MathParameter();
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

            rgChildren.Add("fuction", function.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static MathParameter FromProto(RawProto rp)
        {
            string strVal;
            MathParameter p = new MathParameter();

            if ((strVal = rp.FindValue("function")) != null)
            {
                if (strVal == MyCaffe.common.MATH_FUNCTION.COS.ToString())
                    p.function = MyCaffe.common.MATH_FUNCTION.COS;
                else if (strVal == MyCaffe.common.MATH_FUNCTION.SIN.ToString())
                    p.function = MyCaffe.common.MATH_FUNCTION.SIN;
                else if (strVal == MyCaffe.common.MATH_FUNCTION.TAN.ToString())
                    p.function = MyCaffe.common.MATH_FUNCTION.TAN;
                else
                    p.function = MyCaffe.common.MATH_FUNCTION.NOP;
            }

            return p;
        }
    }
}
