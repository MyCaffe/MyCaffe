using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.beta
{
    /// <summary>
    /// Specifies the parameters for the BCEWithLogitsLossLayer.
    /// </summary>
    /// <remarks>
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class BCEWithLogitsLossParameter : LayerParameterBase
    {
        List<float> m_rgWeights = null;
        REDUCTION m_reduction = REDUCTION.MEAN;

        /// <summary>
        /// Defines the type of reduction to perform on the loss.
        /// </summary>
        public enum REDUCTION
        {
            /// <summary>
            /// Specifies to take the mean of the loss vector and return a single element vector.
            /// </summary>
            MEAN,
            /// <summary>
            /// Specifies to take the sum of the loss vector and return a single element vector.
            /// </summary>
            SUM
        }

        /** @copydoc LayerParameterBase */
        public BCEWithLogitsLossParameter()
        {
        }

        /// <summary>
        /// Specifies the fixed weights.
        /// </summary>
        [Description("Specifies the fixed weights.")]
        public List<float> weights
        {
            get { return m_rgWeights; }
            set { m_rgWeights = value; }
        }

        /// <summary>
        /// Specifies the loss reduction to use.
        /// </summary>
        [Description("Specifies the loss reduction to use.")]
        public REDUCTION reduction
        {
            get { return m_reduction; }
            set { m_reduction = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            BCEWithLogitsLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            BCEWithLogitsLossParameter p = (BCEWithLogitsLossParameter)src;
            m_reduction = p.m_reduction;
            m_rgWeights = Utility.Clone<float>(p.m_rgWeights);
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            BCEWithLogitsLossParameter p = new BCEWithLogitsLossParameter();
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

            rgChildren.Add("reduction", reduction.ToString());
            rgChildren.Add<float>("weight", m_rgWeights);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static BCEWithLogitsLossParameter FromProto(RawProto rp)
        {
            string strVal;
            BCEWithLogitsLossParameter p = new BCEWithLogitsLossParameter();

            p.m_rgWeights = rp.FindArray<float>("weight");

            if ((strVal = rp.FindValue("reduction")) != null)
            {
                if (strVal == "MEAN")
                    p.reduction = REDUCTION.MEAN;
                else if (strVal == "SUM")
                    p.reduction = REDUCTION.SUM;
                else
                    throw new Exception("Unknown reduction '" + strVal + "'!");
            }

            return p;
        }
    }
}
