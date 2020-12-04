using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using MyCaffe.basecode;

namespace MyCaffe.param.beta
{
    /// <summary>
    /// Specifies the parameters for the SqueezeLayer.
    /// </summary>
    public class SqueezeParameter : LayerParameterBase
    {
        BlobShape m_shape = new BlobShape();

        /** @copydoc LayerParameterBase */
        public SqueezeParameter()
        {
            m_shape.dim = new List<int>();
        }

        /// <summary>
        /// Specifies the axes to remove if dim=1 on squeeze, or add dim=1 on unsqueeze.
        /// </summary>
        /// <remarks>
        /// For example, an unsqueeze axes { 0, 4 } changes a size of { 3, 4, 5 } to { 1, 3, 4, 5, 1 }
        /// </remarks>
        [Description("Specifies the axes to remove if dim=1 on squeeze, or add dim=1 on unsqueeze.")]
        public List<int> axes
        {
            get { return m_shape.dim; }
            set { m_shape.dim = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            SqueezeParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            SqueezeParameter p = (SqueezeParameter)src;
            m_shape = p.m_shape.Clone();
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            SqueezeParameter p = new SqueezeParameter();
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
            return m_shape.ToProto(strName);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static SqueezeParameter FromProto(RawProto rp)
        {
            SqueezeParameter p = new SqueezeParameter();
            p.m_shape = BlobShape.FromProto(rp);

            return p;
        }
    }
}
