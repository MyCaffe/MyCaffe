using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the TVLossLayer
    /// </summary>
    /// <remarks>
    /// @see [ftokarev/caffe-neural-style Github](https://github.com/ftokarev/caffe-neural-style) by ftokarev, 2017. 
    /// @see [Understanding Deep Image Representations by Inverting Them](https://arxiv.org/abs/1412.0035) by A. Mahendran and A. Vedaldi, CVPR, 2015.
    /// @see [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, 2015 
    /// </remarks>
    public class TVLossParameter : LayerParameterBase 
    {
        float m_fBeta = 2;

        /** @copydoc LayerParameterBase */
        public TVLossParameter()
        {
        }

        /// <summary>
        /// The beta value.
        /// </summary>
        [Description("The beta value.")]
        public float beta
        {
            get { return m_fBeta; }
            set { m_fBeta = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            TVLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            TVLossParameter p = (TVLossParameter)src;
            m_fBeta = p.m_fBeta;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            TVLossParameter p = new TVLossParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("beta", beta.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static TVLossParameter FromProto(RawProto rp)
        {
            string strVal;
            TVLossParameter p = new TVLossParameter();

            if ((strVal = rp.FindValue("beta")) != null)
                p.beta = int.Parse(strVal);

            return p;
        }
    }
}
