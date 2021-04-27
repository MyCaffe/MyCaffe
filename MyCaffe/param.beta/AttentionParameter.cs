using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the AttentionLayer.
    /// </summary>
    /// <remarks>
    /// </remarks>
    public class AttentionParameter : LayerParameterBase
    {
        int m_nAxis = 2;
        uint m_nDim = 1;
        FillerParameter m_fillerParam_weights = new FillerParameter("xavier");
        FillerParameter m_fillerParam_bias = new FillerParameter("constant", 0.1);

        /** @copydoc LayerParameterBase */
        public AttentionParameter()
        {
        }

        /// <summary>
        /// The axis along which to perform the softmax -- may be negative to index
        /// from the end (e.g., -1 for the last axis).
        /// Any other axes will be evaluated as independent softmaxes.
        /// </summary>
        [Description("Specifies the axis along which to perform the softmax - may be negative to index from the end (e.g., -1 for the last axis).")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Specifies the dim of the attention unit which should match the LSTM output size.
        /// </summary>
        public uint dim
        {
            get { return m_nDim; }
            set { m_nDim = value; }
        }

        /// <summary>
        /// The filler for the weights.
        /// </summary>
        [Category("Fillers")]
        [Description("The filler for the weights.")]
        public FillerParameter weight_filler
        {
            get { return m_fillerParam_weights; }
            set { m_fillerParam_weights = value; }
        }

        /// <summary>
        /// The filler for the bias.
        /// </summary>
        [Category("Fillers")]
        [Description("The filler for the bias.")]
        public FillerParameter bias_filler
        {
            get { return m_fillerParam_bias; }
            set { m_fillerParam_bias = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            AttentionParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            AttentionParameter p = (AttentionParameter)src;

            m_nDim = p.dim;
            m_nAxis = p.m_nAxis;

            if (p.m_fillerParam_bias != null)
                m_fillerParam_bias = p.m_fillerParam_bias.Clone();

            if (p.m_fillerParam_weights != null)
                m_fillerParam_weights = p.m_fillerParam_weights.Clone();
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            AttentionParameter p = new AttentionParameter();
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
            rgChildren.Add("dim", dim.ToString());

            if (weight_filler != null)
                rgChildren.Add(weight_filler.ToProto("weight_filler"));

            if (bias_filler != null)
                rgChildren.Add(bias_filler.ToProto("bias_filler"));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static AttentionParameter FromProto(RawProto rp)
        {
            string strVal;
            AttentionParameter p = new AttentionParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("dim")) != null)
                p.dim = uint.Parse(strVal);

            RawProto rpWeightFiller = rp.FindChild("weight_filler");
            if (rpWeightFiller != null)
                p.weight_filler = FillerParameter.FromProto(rpWeightFiller);

            RawProto rpBiasFiller = rp.FindChild("bias_filler");
            if (rpBiasFiller != null)
                p.bias_filler = FillerParameter.FromProto(rpBiasFiller);

            return p;
        }
    }
}
