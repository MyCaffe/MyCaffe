using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.gpt
{
    /// <summary>
    /// Specifies the parameters for the RMSNormalizationLayer.
    /// </summary>
    /// <remarks>
    /// @see [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) by Zhang et al., 2019, arXiv:1910.07467
    /// @see [RMSNorm](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) PyTorch
    /// @see [GitHub:karpathy/llama2.c](https://github.com/karpathy/llama2.c) by Karpathy (MIT Liceense).
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class RMSNormParameter : LayerParameterBase 
    {
        int m_nAxis = 2;
        double m_dfEpsilon = 1e-5;
        bool m_bEnableWeights = true;

        /** @copydoc LayerParameterBase */
        public RMSNormParameter()
        {
        }

        /// <summary>
        /// Specifies whether to enable weights (default = <i>false</i>).
        /// </summary>
        [Description("Specifies whether to enable weights (default = false).")]
        public bool enable_weights
        {
            get { return m_bEnableWeights; }
            set { m_bEnableWeights = value; }
        }

        /// <summary>
        /// Specifies the epsilon value used to avoid invalid values (default = 1e-5).
        /// </summary>
        [Description("Specifies the epsilon value used to avoid invalid values (default = 1e-5).")]
        public double epsilon
        {
            get { return m_dfEpsilon; }
            set { m_dfEpsilon = value; }
        }

        /// <summary>
        /// Specifies the first axis to be lumped into a single RmsNorm computation;
        /// all preceding axes are retained in the output.
        /// May be negative to index from the end (e.g., -1 for the last axis)
        /// </summary>
        [Description("Specifies the first axis to be lumped into a single RmsNorm computation; all preceding axes are retained in the output.")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Load the parameter from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bNewInstance">When <i>true</i> a new instance is created (the default), otherwise the existing instance is loaded from the binary reader.</param>
        /// <returns>Returns an instance of the parameter.</returns>
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            RMSNormParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Copy on parameter to another.
        /// </summary>
        /// <param name="src">Specifies the parameter to copy.</param>
        public override void Copy(LayerParameterBase src)
        {
            RMSNormParameter p = (RMSNormParameter)src;
            m_dfEpsilon = p.epsilon;
            m_nAxis = p.m_nAxis;
            m_bEnableWeights = p.enable_weights;
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public override LayerParameterBase Clone()
        {
            RMSNormParameter p = new RMSNormParameter();
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

            rgChildren.Add("epsilon", m_dfEpsilon.ToString());
            rgChildren.Add("axis", axis.ToString());
            rgChildren.Add("enable_weights", m_bEnableWeights.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static RMSNormParameter FromProto(RawProto rp)
        {
            string strVal;
            RMSNormParameter p = new RMSNormParameter();

            if ((strVal = rp.FindValue("epsilon")) != null)
                p.m_dfEpsilon = double.Parse(strVal);

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("enable_weights")) != null)
                p.enable_weights = bool.Parse(strVal);

            return p;
        }
    }
}
