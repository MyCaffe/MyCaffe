using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param.nt
{
    /// <summary>
    /// Specifies the parameters for the ScalarLayer
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ScalarParameter : LayerParameterBase 
    {
        double m_dfVal;
        ScalarOp m_op;
        bool m_bPassthroughGradient = false;

        /// <summary>
        /// Defines the scalar operations that may be performed.
        /// </summary>
        public enum ScalarOp
        {
            /// <summary>
            /// Specifies to run a mul_scalar on the layer.
            /// </summary>
            MUL,
            /// <summary>
            /// Specifies to run an add_scalar on the layer.
            /// </summary>
            ADD
        }

        /// <summary>
        /// The constructor.
        /// </summary>
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

        /// <summary>
        /// Load the parameter from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bNewInstance">When <i>true</i> a new instance is created (the default), otherwise the existing instance is loaded from the binary reader.</param>
        /// <returns>Returns an instance of the parameter.</returns>
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ScalarParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Copy on parameter to another.
        /// </summary>
        /// <param name="src">Specifies the parameter to copy.</param>
        public override void Copy(LayerParameterBase src)
        {
            ScalarParameter p = (ScalarParameter)src;
            m_dfVal = p.m_dfVal;
            m_op = p.m_op;
            m_bPassthroughGradient = p.m_bPassthroughGradient;
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public override LayerParameterBase Clone()
        {
            ScalarParameter p = new ScalarParameter();
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
                p.value = ParseDouble(strVal);

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
