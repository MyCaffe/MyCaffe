using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the PReLULayer.
    /// </summary>
    /// <remarks>
    /// @see [Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853) by Bing Xu, Naiyan Wang, Tianqi Chen, and Mu Li, 2015.
    /// @see [Revise Saturated Activation Functions](https://arxiv.org/abs/1602.05980?context=cs) by Bing Xu, Ruitong Huang, and Mu Li, 2016.
    /// @see [Understanding Deep Neural Networks with Rectified Linear Units](https://arxiv.org/abs/1611.01491) by Raman Arora, Amitabh Basu, Poorya Mianjy, and Anirbit Mukherjee, 2016.
    /// @see [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852v1) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, 2015.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class PReLUParameter : LayerParameterBase
    {
        FillerParameter m_filler = new FillerParameter("constant", 0.25);
        bool m_bChannelShared = false;

        /** @copydoc LayerParameterBase */
        public PReLUParameter()
        {
        }

        /// <summary>
        /// Specifies initial value of @f$ a_i @f$.  Default is @f$a_i = 0.25 @f$ for all i.
        /// </summary>
        [Description("Specifies filler used for initial value of 'a_i'.  Default is 'a_i' = 0.25 for all 'i'.")]
        public FillerParameter filler
        {
            get { return m_filler; }
            set { m_filler = value; }
        }

        /// <summary>
        /// Specifies whether or not slope parameters are shared across channels.
        /// </summary>
        [Description("Specifies whether or not slope parameters are shared across channels.")]
        public bool channel_shared
        {
            get { return m_bChannelShared; }
            set { m_bChannelShared = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            PReLUParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            PReLUParameter p = (PReLUParameter)src;

            if (p.m_filler != null)
                m_filler = p.m_filler.Clone();

            m_bChannelShared = p.m_bChannelShared;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            PReLUParameter p = new PReLUParameter();
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

            if (m_filler != null)
                rgChildren.Add(m_filler.ToProto("filler"));

            if (channel_shared != false)
                rgChildren.Add("channel_shared", channel_shared.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static PReLUParameter FromProto(RawProto rp)
        {
            string strVal;
            PReLUParameter p = new PReLUParameter();

            RawProto rpFiller = rp.FindChild("filler");
            if (rpFiller != null)
                p.m_filler = FillerParameter.FromProto(rpFiller);

            if ((strVal = rp.FindValue("channel_shared")) != null)
                p.channel_shared = bool.Parse(strVal);

            return p;
        }
    }
}
