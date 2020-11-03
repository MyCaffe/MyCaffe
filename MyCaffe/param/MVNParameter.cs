using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the MVNLayer.
    /// </summary>
    /// <remarks>
    /// @see [Layer Normalization](https://arxiv.org/abs/1607.06450) by Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton, 2016.
    /// @see [Learning weakly supervised multimodal phoneme embeddings](https://arxiv.org/abs/1704.06913v1) by Rahma Chaabouni, Ewan Dunbar, Neil Zeghidour, and Emmanuel Dupoux, 2017. 
    /// @see [Estimating Phoneme Class Conditional Probabilities from Raw Speech Signal using Convolutional Neural Networks](https://arxiv.org/abs/1304.1018) by Dimitri Palaz, Ronan Collobert, and Mathew Magimai-Doss, 2013.
    /// </remarks>
    public class MVNParameter : LayerParameterBase
    {
        bool m_bNormalizeVariance = true;
        bool m_bAcrossChannels = false;
        double m_dfEps = 1e-9;

        /** @copydoc LayerParameterBase */
        public MVNParameter()
        {
        }

        /// <summary>
        /// Specifies whether or not to normalize the variance.
        /// </summary>
        [Description("Specifies whether or not to normalize the variance.")]
        public bool normalize_variance
        {
            get { return m_bNormalizeVariance; }
            set { m_bNormalizeVariance = value; }
        }

        /// <summary>
        /// Specifies whether or not to normalize accross channels.
        /// </summary>
        [Description("Specifies whether or not to normalize accross channels.")]
        public bool across_channels
        {
            get { return m_bAcrossChannels; }
            set { m_bAcrossChannels = value; }
        }

        /// <summary>
        /// Specifies a small value to avoid divide by zero.
        /// </summary>
        [Description("Specifies a small value to avoid divide by zero.")]
        [Browsable(false)]
        public double eps
        {
            get { return m_dfEps; }
            set { m_dfEps = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            MVNParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            MVNParameter p = (MVNParameter)src;
            m_bNormalizeVariance = p.m_bNormalizeVariance;
            m_bAcrossChannels = p.m_bAcrossChannels;
            m_dfEps = p.m_dfEps;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            MVNParameter p = new MVNParameter();
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

            if (normalize_variance != true)
                rgChildren.Add("normalize_variance", normalize_variance.ToString());

            if (across_channels != false)
                rgChildren.Add("across_channels", across_channels.ToString());

            if (eps != 1e-9)
                rgChildren.Add("eps", eps.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static MVNParameter FromProto(RawProto rp)
        {
            string strVal;
            MVNParameter p = new MVNParameter();

            if ((strVal = rp.FindValue("normalize_variance")) != null)
                p.normalize_variance = bool.Parse(strVal);

            if ((strVal = rp.FindValue("across_channels")) != null)
                p.across_channels = bool.Parse(strVal);

            if ((strVal = rp.FindValue("eps")) != null)
                p.eps = parseDouble(strVal);

            return p;
        }
    }
}
