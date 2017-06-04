using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters used by the EmbedLayer.
    /// </summary>
    public class EmbedParameter : LayerParameterBase 
    {
        uint m_nNumOutput;
        uint m_nInputDim;
        bool m_bBiasTerm = true;
        FillerParameter m_fillerParam_weights = new FillerParameter("gaussian");
        FillerParameter m_fillerParam_bias = new FillerParameter("constant", 1.0);

        /** @copydoc LayerParameterBase */
        public EmbedParameter()
        {
        }

        /// <summary>
        ///  Specifies the number of outputs for the layer.
        /// </summary>
        [Description("Specifies the number of outputs for the layer.")]
        public uint num_output
        {
            get { return m_nNumOutput; }
            set { m_nNumOutput = value; }
        }

        /// <summary>
        /// Specifies the input given as integers to be interpreted as one-hot
        /// vector indices with dimension num_input.  Hence num_input should be
        /// 1 greater than the maximum possible input value.
        /// </summary>
        [Description("Specifies the input given as integers to be interpreted as one-hot vector indices with dimension 'num_init'. Hence 'num_input' should be 1 greater than the maximum possible input value.")]
        public uint input_dim
        {
            get { return m_nInputDim; }
            set { m_nInputDim = value; }
        }

        /// <summary>
        /// Specifies whether to use a bias term or not.
        /// </summary>
        [Description("Specifies wheter ot use a bias term or not.")]
        public bool bias_term
        {
            get { return m_bBiasTerm; }
            set { m_bBiasTerm = value; }
        }

        /// <summary>
        /// Specifies the filler for the weights.
        /// </summary>
        [Description("Specifies the filler for the weights.")]
        public FillerParameter weight_filler
        {
            get { return m_fillerParam_weights; }
            set { m_fillerParam_weights = value; }
        }

        /// <summary>
        /// Specifies the filler for the bias.
        /// </summary>
        [Description("Specifies the filler for the bias.")]
        public FillerParameter bias_filler
        {
            get { return m_fillerParam_bias; }
            set { m_fillerParam_bias = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            EmbedParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            EmbedParameter p = (EmbedParameter)src;

            m_nNumOutput = p.m_nNumOutput;
            m_nInputDim = p.m_nInputDim;
            m_bBiasTerm = p.m_bBiasTerm;

            if (p.m_fillerParam_bias != null)
                m_fillerParam_bias = p.m_fillerParam_bias.Clone();

            if (p.m_fillerParam_weights != null)
                m_fillerParam_weights = p.m_fillerParam_weights.Clone();
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            EmbedParameter p = new EmbedParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("num_output", num_output.ToString());
            rgChildren.Add("input_dim", input_dim.ToString());

            if (bias_term != true)
                rgChildren.Add("bias_term", bias_term.ToString());

            if (weight_filler != null)
                rgChildren.Add(weight_filler.ToProto("weight_filler"));

            if (bias_term && bias_filler != null)
                rgChildren.Add(bias_filler.ToProto("bias_filler"));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static EmbedParameter FromProto(RawProto rp)
        {
            string strVal;
            EmbedParameter p = new EmbedParameter();

            if ((strVal = rp.FindValue("num_output")) != null)
                p.num_output = uint.Parse(strVal);

            if ((strVal = rp.FindValue("input_dim")) != null)
                p.input_dim = uint.Parse(strVal);

            if ((strVal = rp.FindValue("bias_term")) != null)
                p.bias_term = bool.Parse(strVal);

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
