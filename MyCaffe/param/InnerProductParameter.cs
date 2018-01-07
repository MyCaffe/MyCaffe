using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the InnerProductLayer.
    /// </summary>
    /// <remarks>
    /// @see [Product-based Neural Networks for User Response Prediction](https://arxiv.org/abs/1611.00144) by Yanru Qu, Kan Cai, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang, 2016. 
    /// </remarks>
    public class InnerProductParameter : LayerParameterBase
    {
        uint m_nNumOutput = 0;
        bool m_bBiasTerm = true;
        FillerParameter m_fillerParam_weights = new FillerParameter("xavier");
        FillerParameter m_fillerParam_bias = new FillerParameter("constant", 0.1);
        int m_nAxis = 1;
        bool m_bTranspose = false;

        /** @copydoc LayerParameterBase */
        public InnerProductParameter()
        {
        }

        /// <summary>
        /// The number of outputs for the layer.
        /// </summary>
        [Description("The number of outputs for the layer.")]
        public uint num_output
        {
            get { return m_nNumOutput; }
            set { m_nNumOutput = value; }
        }

        /// <summary>
        /// Whether to have bias terms or not.
        /// </summary>
        [Description("Whether to have bias terms or not.")]
        public bool bias_term
        {
            get { return m_bBiasTerm; }
            set { m_bBiasTerm = value; }
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

        /// <summary>
        /// Specifies the first axis to be lumped into a single inner product computation;
        /// all preceding axes are retained in the output.
        /// May be negative to index from the end (e.g., -1 for the last axis)
        /// </summary>
        [Description("Specifies the first axis to be lumped into a single inner product computation; all preceding axes are retained in the output.")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Specifies whether to transpose the weight matrix or not.
        /// If transpose == true, any operations will be performed on the transpose
        /// of the weight matrix.  The weight matrix itself is not going to be transposed
        /// but rather the transfer flag of operations will be toggled accordingly.
        /// </summary>
        [Description("Specifies whether to transpose the weight matrix or not.")]
        public bool transpose
        {
            get { return m_bTranspose; }
            set { m_bTranspose = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            InnerProductParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            InnerProductParameter p = (InnerProductParameter)src;

            m_nNumOutput = p.m_nNumOutput;
            m_bBiasTerm = p.m_bBiasTerm;

            if (p.m_fillerParam_bias != null)
                m_fillerParam_bias = p.m_fillerParam_bias.Clone();

            if (p.m_fillerParam_weights != null)
                m_fillerParam_weights = p.m_fillerParam_weights.Clone();

            m_nAxis = p.m_nAxis;
            m_bTranspose = p.m_bTranspose;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            InnerProductParameter p = new InnerProductParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("num_output", num_output.ToString());
            rgChildren.Add("bias_term", bias_term.ToString());

            if (weight_filler != null)
                rgChildren.Add(weight_filler.ToProto("weight_filler"));

            if (bias_filler != null)
                rgChildren.Add(bias_filler.ToProto("bias_filler"));

            rgChildren.Add("axis", axis.ToString());

            if (transpose != false)
                rgChildren.Add("tranpose", transpose.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static InnerProductParameter FromProto(RawProto rp)
        {
            string strVal;
            InnerProductParameter p = new InnerProductParameter();

            if ((strVal = rp.FindValue("num_output")) != null)
                p.num_output = uint.Parse(strVal);

            if ((strVal = rp.FindValue("bias_term")) != null)
                p.bias_term = bool.Parse(strVal);

            RawProto rpWeightFiller = rp.FindChild("weight_filler");
            if (rpWeightFiller != null)
                p.weight_filler = FillerParameter.FromProto(rpWeightFiller);

            RawProto rpBiasFiller = rp.FindChild("bias_filler");
            if (rpBiasFiller != null)
                p.bias_filler = FillerParameter.FromProto(rpBiasFiller);

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("transpose")) != null)
                p.transpose = bool.Parse(strVal);

            return p;
        }
    }
}
