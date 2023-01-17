using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.gpt
{
    /// <summary>
    /// Specifies the parameters for the NLLLossLayer.
    /// </summary>
    /// <remarks>
    /// @see [NLLOSS](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) by PyTorch
    /// @see [NLLLoss implementation](https://forums.fast.ai/t/nllloss-implementation/20028) by bny6613 Nick, 2018
    /// </remarks>
    public class NLLLossParameter : LayerParameterBase
    {
        int m_nAxis = 1;

        /** @copydoc LayerParameterBase */
        public NLLLossParameter()
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
            NLLLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            NLLLossParameter p = (NLLLossParameter)src;
            m_nAxis = p.m_nAxis;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            NLLLossParameter p = new NLLLossParameter();
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
        public static NLLLossParameter FromProto(RawProto rp)
        {
            string strVal;
            NLLLossParameter p = new NLLLossParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            return p;
        }
    }
}
