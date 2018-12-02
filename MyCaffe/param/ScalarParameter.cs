using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the ScalarLayer
    /// </summary>
    public class ScalarParameter : LayerParameterBase 
    {
        double m_dfVal;
        ScalarOp m_op;
        bool m_bPassthroughGradient = false;

        public enum ScalarOp
        {
            MUL,
            ADD
        }

        /** @copydoc LayerParameterBase */
        public ScalarParameter()
        {
        }

        /// <summary>
        /// Specifies the scalar value to apply.
        /// </summary>
        [Description("Specifies the scalar value to apply.")]
        public double value
        {
            get { return m_dfVal; }
            set { m_dfVal = value; }
        }

        /// <summary>
        /// Specifies the scalar operation to apply (mul, add, etc).
        /// </summary>
        [Description("Specifies the scalar operatioon to apply (mul, add, etc).")]
        public ScalarOp operation
        {
            get { return m_op; }
            set { m_op = value; }
        }

        /// <summary>
        /// Specifies whether or not to pass-through the gradient without performing the back-prop calculation (default = <i>false</i>).
        /// </summary>
        public bool passthrough_gradient
        {
            get { return m_bPassthroughGradient; }
            set { m_bPassthroughGradient = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ScalarParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            ScalarParameter p = (ScalarParameter)src;
            m_dfVal = p.m_dfVal;
            m_op = p.m_op;
            m_bPassthroughGradient = p.m_bPassthroughGradient;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ScalarParameter p = new ScalarParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("value", m_dfVal.ToString());
            rgChildren.Add("operation", m_op.ToString());
            rgChildren.Add("passthrough_gradient", passthrough_gradient.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ScalarParameter FromProto(RawProto rp)
        {
            string strVal;
            ScalarParameter p = new ScalarParameter();

            if ((strVal = rp.FindValue("value")) != null)
                p.value = double.Parse(strVal);

            if ((strVal = rp.FindValue("operation")) != null)
            {
                if (strVal == ScalarOp.MUL.ToString())
                    p.m_op = ScalarOp.MUL;
                else if (strVal == ScalarOp.ADD.ToString())
                    p.m_op = ScalarOp.ADD;
                else
                    throw new Exception("Unknown scalar operation '" + strVal + "'");
            }

            if ((strVal = rp.FindValue("passthrough_gradient")) != null)
                p.passthrough_gradient = bool.Parse(strVal);

            return p;
        }
    }
}
