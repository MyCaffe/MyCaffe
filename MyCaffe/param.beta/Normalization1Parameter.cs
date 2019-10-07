using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.beta
{
    /// <summary>
    /// Specifies the parameters for the Normalization1Layer.
    /// </summary>
    /// <remarks>
    /// @see [Layer Normalization](https://arxiv.org/abs/1607.06450) by Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton, 2016.
    /// </remarks>
    public class Normalization1Parameter : LayerParameterBase 
    {
        Norm m_norm = Norm.L2;

        /// <summary>
        /// Defines the normalization type.
        /// </summary>
        public enum Norm
        {
            /// <summary>
            /// L1 normalization.
            /// </summary>
            L1 = 1,
            /// <summary>
            /// L2 normalization.
            /// </summary>
            L2 = 2
        }

        /** @copydoc LayerParameterBase */
        public Normalization1Parameter()
        {
        }

        /// <summary>
        /// Specifies the normalization method to use.
        /// </summary>
        [Description("Specifies the normalization method to use.")]
        public Norm norm
        {
            get { return m_norm; }
            set { m_norm = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            Normalization1Parameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            Normalization1Parameter p = (Normalization1Parameter)src;
            m_norm = p.m_norm;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            Normalization1Parameter p = new Normalization1Parameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("norm", m_norm.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static Normalization1Parameter FromProto(RawProto rp)
        {
            string strVal;
            Normalization1Parameter p = new Normalization1Parameter();

            if ((strVal = rp.FindValue("norm")) != null)
            {
                if (strVal.ToLower() == Norm.L1.ToString().ToLower())
                    p.m_norm = Norm.L1;
                else
                    p.m_norm = Norm.L2;
            }

            return p;
        }
    }
}
