using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the MAELossLayer.
    /// </summary>
    /// <remarks>
    /// @see [Mean Absolute Error](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-absolute-error) by Peltarion.
    /// </remarks>
    public class MAELossParameter : LayerParameterBase
    {
        int m_nAxis = 1; // Axis used to calculate the loss normalization.

        /** @copydoc LayerParameterBase */
        public MAELossParameter()
        {
        }

        /// <summary>
        /// [\b optional, default = 1] Specifies the axis of the probability.
        /// </summary>
        [Description("Specifies the axis of the probability, default = 1")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            MAELossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            MAELossParameter p = (MAELossParameter)src;
            m_nAxis = p.m_nAxis;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            MAELossParameter p = new MAELossParameter();
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

            rgChildren.Add("axis", axis.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static MAELossParameter FromProto(RawProto rp)
        {
            string strVal;
            MAELossParameter p = new MAELossParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            return p;
        }
    }
}
