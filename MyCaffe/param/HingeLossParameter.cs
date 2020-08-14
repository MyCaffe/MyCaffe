using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the HingLossLayer.
    /// </summary>
    /// <remarks>
    /// @see [CNN-based Patch Matching for Optical Flow with Thresholded Hinge Loss](https://arxiv.org/abs/1607.08064) by Christian Bailer, Kiran Varanasi, and Didier Stricker, 2016.
    /// @see [Hinge-Loss Markov Random Fields and Probabilistic Soft Logic](https://arxiv.org/abs/1505.04406) by Stephen H. Bach, Matthias Broecheler, Bert Huang, and Lise Getoor, 2015.
    /// </remarks>
    public class HingeLossParameter : LayerParameterBase
    {
        /// <summary>
        /// Defines the type of normalization.
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

        Norm m_norm = Norm.L1;

        /** @copydoc LayerParameterBase */
        public HingeLossParameter()
        {
        }

        /// <summary>
        /// Specify the Norm to use L1 or L2
        /// </summary>
        [Description("Specify the Norm to use: L1 or L2")]
        public Norm norm
        {
            get { return m_norm; }
            set { m_norm = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            HingeLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            HingeLossParameter p = (HingeLossParameter)src;
            m_norm = p.m_norm;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            HingeLossParameter p = new HingeLossParameter();
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

            rgChildren.Add("norm", norm.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static HingeLossParameter FromProto(RawProto rp)
        {
            string strVal;
            HingeLossParameter p = new HingeLossParameter();

            if ((strVal = rp.FindValue("norm")) != null)
            {
                switch (strVal)
                {
                    case "L1":
                        p.norm = Norm.L1;
                        break;

                    case "L2":
                        p.norm = Norm.L2;
                        break;

                    default:
                        throw new Exception("Unknown 'norm' value: " + strVal);
                }
            }

            return p;
        }
    }
}
