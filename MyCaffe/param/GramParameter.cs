using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the GramLayer
    /// </summary>
    /// <remarks>
    /// @see [ftokarev/caffe-neural-style Github](https://github.com/ftokarev/caffe-neural-style) by ftokarev, 2017. 
    /// @see [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, 2015 
    /// </remarks>
    public class GramParameter : LayerParameterBase 
    {
        int m_nAxis = 2;
        double m_dfAlpha = 1.0;
        double m_dfBeta = 1.0;

        /** @copydoc LayerParameterBase */
        public GramParameter()
        {
        }

        /// <summary>
        /// The first axis to be lumped into a single Gram matrix computation;
        /// all preceding axes are retained in the output.
        /// May be negtive to index from the end (e.g. -1 for the last axis)
        /// For exapmle, if axis == 2 and the input is (N x C x H x W), the output
        /// will be (N x C x C)
        /// </summary>
        [Description("The first axis to be lumped into a single Gram matrix computation -- may be negative to index from end (e.g., -1 for the last axis). For example, if axis == 2 and the input is (N x C x H x W), the output will be (N x C x C).")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Specifies the scaling factor applied before the gram operation.
        /// </summary>
        public double alpha
        {
            get { return m_dfAlpha; }
            set { m_dfAlpha = value; }
        }

        /// <summary>
        /// Specifies the scaling factor applied after the gram operation.
        /// </summary>
        public double beta
        {
            get { return m_dfBeta; }
            set { m_dfBeta = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            GramParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            GramParameter p = (GramParameter)src;
            m_nAxis = p.m_nAxis;
            m_dfAlpha = p.m_dfAlpha;
            m_dfBeta = p.m_dfBeta;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            GramParameter p = new GramParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("axis", axis.ToString());
            rgChildren.Add("alpha", alpha.ToString());
            rgChildren.Add("beta", beta.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static GramParameter FromProto(RawProto rp)
        {
            string strVal;
            GramParameter p = new GramParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("alpha")) != null)
                p.alpha = double.Parse(strVal);

            if ((strVal = rp.FindValue("beta")) != null)
                p.beta = double.Parse(strVal);

            return p;
        }
    }
}
