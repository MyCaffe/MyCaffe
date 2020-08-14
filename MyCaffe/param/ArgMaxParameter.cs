using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the ArgMaxLayer
    /// </summary>
    /// <remarks>
    /// @see [Detecting Unexpected Obstacles for Self-Driving Cars: Fusing Deep Learning and Geometric Modeling](https://arxiv.org/abs/1612.06573v1) by Sebastian Ramos, Stefan Gehrig, Peter Pinggera, Uwe Franke, and Carsten Rother, 2016. 
    /// </remarks>
    public class ArgMaxParameter : LayerParameterBase 
    {
        bool m_bOutMaxVal = false;
        uint m_nTopK = 1;
        int? m_nAxis = null;

        /** @copydoc LayerParameterBase */
        public ArgMaxParameter()
        {
        }

        /// <summary>
        /// If true produce pairs (argmax, maxval)
        /// </summary>
        [Description("If true, produce pairs (argmax, maxval).")]
        public bool out_max_val
        {
            get { return m_bOutMaxVal; }
            set { m_bOutMaxVal = value; }
        }

        /// <summary>
        /// When computing accuracy, count as correct by comparing the true label to
        /// the top_k scoring classes.  By default, only compare the top scoring
        /// class (i.e. argmax).
        /// </summary>
        [Description("When computing accuracy, count as correct by comparing the true label to the 'top_k' scoring classes.  By default, only compare the top scoring classes (i.e. argmax).")]
        public uint top_k
        {
            get { return m_nTopK; }
            set { m_nTopK = value; }
        }

        /// <summary>
        /// The axis along which to maximize -- may be negative to index from the
        /// end (e.g., -1 for the last axis).
        /// By default ArgMaxLayer maximizes over the flattened trailing dimensions
        /// for each index of the first / num dimension.
        /// </summary>
        [Description("Specifies the axis along which to maximize -- may be negative to index from end (e.g., -1 for the last axis). By default the ArgMaxLayer maximizes over the flattened trailing dimensions for each index of the first / num dimension.")]
        public int? axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ArgMaxParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            ArgMaxParameter p = (ArgMaxParameter)src;
            m_bOutMaxVal = p.m_bOutMaxVal;
            m_nTopK = p.m_nTopK;
            m_nAxis = p.m_nAxis;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ArgMaxParameter p = new ArgMaxParameter();
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

            if (out_max_val != false)
                rgChildren.Add("out_max_val", out_max_val.ToString());

            if (top_k != 1)
                rgChildren.Add("top_k", top_k.ToString());

            if (axis.HasValue)
                rgChildren.Add("axis", axis.Value.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ArgMaxParameter FromProto(RawProto rp)
        {
            string strVal;
            ArgMaxParameter p = new ArgMaxParameter();

            if ((strVal = rp.FindValue("out_max_val")) != null)
                p.out_max_val = bool.Parse(strVal);

            if ((strVal = rp.FindValue("top_k")) != null)
                p.top_k = uint.Parse(strVal);

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            return p;
        }
    }
}
