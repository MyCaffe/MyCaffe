using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.tft
{
    /// <summary>
    /// Specifies the parameters for the GluLayer (Gated Linear Unit).  
    /// </summary>
    /// <remarks>
    /// The output of the layer is a linear projection (X * W + b) modulated by the gates **sigmoid** (X * V + c).  These
    /// gates multiply each element of the matrix X * W + b and control the information passed in.  The simplified gating
    /// mechanism in this layer is for non-deterministic gates that reduce the vanishing gradient problem, by having linear
    /// units couypled to the gates.  This retains the non-linear capabilities of the layer while allowing the gradient
    /// to propagate through the linear unit without scaling.
    /// 
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L11) by Playtika Research, 2021.
    /// @see ["Language modeling with gated convolution networks](https://arxiv.org/abs/1612.08083) by Dauphin, Yann N., et al., International conference on machine learning, PMLR, 2017
    /// </remarks>
    public class GluParameter : LayerParameterBase
    {
        FillerParameter m_fillerParam_weights = new FillerParameter("xavier");
        FillerParameter m_fillerParam_bias = new FillerParameter("constant", 0.1);
        int m_nAxis = 1;
        int m_nInputDim;
        bool m_bBiasTerm = true;
        bool m_bEnableNoise = false;
        double m_dfSigmaInit = 0.017;
        MODULATION m_modulation = MODULATION.SIGMOID;

        /// <summary>
        /// Defines the modulation type.
        /// </summary>
        public enum MODULATION
        {
            /// <summary>
            /// Specifies to use Sigmoid modulation.
            /// </summary>
            SIGMOID
        }

        /** @copydoc LayerParameterBase */
        public GluParameter()
        {
        }

        /// <summary>
        /// Specifies the input dimension.
        /// </summary>
        [Description("Specifies the input dimension.")]
        public int input_dim
        {
            get { return m_nInputDim; }
            set { m_nInputDim = value; }
        }

        /// <summary>
        /// Specifies the gate modulation type.
        /// </summary>
        [Description("Specifies the gate modulation type.")]
        public MODULATION modulation
        {
            get { return m_modulation; }
            set { m_modulation = value; }
        }

        /// <summary>
        /// Enable/disable noise in the inner-product layer (default = false).
        /// </summary>
        /// <remarks>
        /// When enabled, noise is only used during the training phase.
        /// </remarks>
        [Description("Enable/disable noise in the inner-product layer (default = false).")]
        public bool enable_noise
        {
            get { return m_bEnableNoise; }
            set { m_bEnableNoise = value; }
        }

        /// <summary>
        /// Specifies the initialization value for the sigma weight and sigma bias used when 'enable_noise' = <i>true</i>.
        /// </summary>
        [Description("Specifies the initialization value for the sigma weight and sigma bias used when 'enable_noise' = true.")]
        public double sigma_init
        {
            get { return m_dfSigmaInit; }
            set { m_dfSigmaInit = value; }
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


        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            GluParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            GluParameter p = (GluParameter)src;

            m_modulation = p.modulation;
            m_nInputDim = p.input_dim;
            m_bBiasTerm = p.m_bBiasTerm;

            if (p.m_fillerParam_bias != null)
                m_fillerParam_bias = p.m_fillerParam_bias.Clone();

            if (p.m_fillerParam_weights != null)
                m_fillerParam_weights = p.m_fillerParam_weights.Clone();

            m_nAxis = p.m_nAxis;
            m_bEnableNoise = p.m_bEnableNoise;
            m_dfSigmaInit = p.m_dfSigmaInit;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            GluParameter p = new GluParameter();
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

            rgChildren.Add("modulation", modulation.ToString());
            rgChildren.Add("input_dim", input_dim.ToString());
            rgChildren.Add("bias_term", bias_term.ToString());

            if (weight_filler != null)
                rgChildren.Add(weight_filler.ToProto("weight_filler"));

            if (bias_filler != null)
                rgChildren.Add(bias_filler.ToProto("bias_filler"));

            rgChildren.Add("axis", axis.ToString());

            if (m_bEnableNoise)
            {
                rgChildren.Add("enable_noise", m_bEnableNoise.ToString());
                rgChildren.Add("sigma_init", m_dfSigmaInit.ToString());
            }

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static GluParameter FromProto(RawProto rp)
        {
            string strVal;
            GluParameter p = new GluParameter();

            if ((strVal = rp.FindValue("input_dim")) != null)
                p.input_dim = int.Parse(strVal);

            if ((strVal = rp.FindValue("modulation")) != null)
            {
                if (strVal == MODULATION.SIGMOID.ToString())
                    p.modulation = MODULATION.SIGMOID;
            }

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

            if ((strVal = rp.FindValue("enable_noise")) != null)
                p.enable_noise = bool.Parse(strVal);

            if ((strVal = rp.FindValue("sigma_init")) != null)
                p.sigma_init = ParseDouble(strVal);

            return p;
        }
    }
}
