using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using MyCaffe.basecode;

namespace MyCaffe.param.beta
{
    /// <summary>
    /// Specifies the parameters for the TransposeLayer.
    /// </summary>
    public class TransposeParameter : LayerParameterBase
    {
        BlobShape m_shape = new BlobShape();

        /** @copydoc LayerParameterBase */
        public TransposeParameter()
        {
        }

        /// <summary>
        /// Specifies the dimensions to transpose.
        /// </summary>
        /// <remarks>
        /// For example, if you want to transpose NxCxHxW into WxNxHxC,
        /// the parameter should be the following:
        /// transpose_param { dim: 3 dim: 0 dim: 2 dim: 1 }
        /// ie, if the i-th dim has value n, then the i-th axis of top is equal to the n-th axis of bottom.
        /// </remarks>
        [Description("Specifies the first axis to flatten: all preceding axes are retained in the output.")]
        public List<int> dim
        {
            get { return m_shape.dim; }
            set { m_shape.dim = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            TransposeParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            TransposeParameter p = (TransposeParameter)src;
            m_shape = p.m_shape.Clone();
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            TransposeParameter p = new TransposeParameter();
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
        public static TransposeParameter FromProto(RawProto rp)
        {
            TransposeParameter p = new TransposeParameter();
            p.m_shape = BlobShape.FromProto(rp);

            return p;
        }
    }
}
