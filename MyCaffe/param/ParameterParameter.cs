using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the ParameterLayer
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ParameterParameter : LayerParameterBase 
    {
        BlobShape m_shape = new BlobShape();

        /** @copydoc LayerParameterBase */
        public ParameterParameter()
        {
        }

        /// <summary>
        /// Specifies the parameter shape.
        /// </summary>
        [Description("Specifies the parameter shape.")]
        public BlobShape shape
        {
            get { return m_shape; }
            set { m_shape = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ParameterParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            ParameterParameter p = (ParameterParameter)src;
            m_shape = p.m_shape.Clone();
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ParameterParameter p = new ParameterParameter();
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

            rgChildren.Add(m_shape.ToProto("shape"));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ParameterParameter FromProto(RawProto rp)
        {
            ParameterParameter p = new ParameterParameter();

            RawProto rp1 = rp.FindChild("shape");
            if (rp1 != null)
                p.shape = BlobShape.FromProto(rp1);

            return p;
        }
    }
}
