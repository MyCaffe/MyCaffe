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
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class InnerProductParameter : LayerParameterBase
    {
        uint m_nNumOutput = 0;
        bool m_bBiasTerm = true;
        FillerParameter m_fillerParam_weights = new FillerParameter("xavier");
        FillerParameter m_fillerParam_bias = new FillerParameter("constant", 0.1);
        int m_nAxis = 1;
        int m_nMinTopAxes = -1;
        bool m_bTranspose = false;
        bool m_bEnableNoise = false;
        double m_dfSigmaInit = 0.017;
        double m_dfBiasGradScale = 1.0;
        bool m_bOutputContainsPredictions = false;

        /** @copydoc LayerParameterBase */
        public InnerProductParameter()
        {
        }

        /// <summary>
        /// Specifies that the output contains predictions and that the output blob is marked as BLOB_TYPE.PREDICTION.
        /// </summary>
        [Description("Specifies that the output contains predictions and that the output blob is marked as BLOB_TYPE.PREDICTION.")]
        public bool output_contains_predictions
        {
            get { return m_bOutputContainsPredictions; }
            set { m_bOutputContainsPredictions = value; }
        }

        /// <summary>
        /// Specifies a scaling value applied to the bias mutliplier and then unapplied after calculating the bias - used to help improve float accuracy (default = 1.0).  A value of 1.0 is ignored.
        /// </summary>
        [Description("Specifies a scaling value applied to the bias mutliplier and then unapplied after calculating the bias - used to help improve float accuracy (default = 1.0).  A value of 1.0 is ignored.")]
        public double bias_grad_scale
        {
            get { return m_dfBiasGradScale; }
            set { m_dfBiasGradScale = value;}
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
        /// The number of outputs for the layer.
        /// </summary>
        [Description("The number of outputs for the layer.")]
        public uint num_output
        {
            get { return m_nNumOutput; }
            set { m_nNumOutput = value; }
        }

        /// <summary>
        /// Optionally, specifies the minimum top axes (default = -1, which ignores this setting).
        /// </summary>
        /// <remarks>
        /// NOTE: The Deconvolution Layer requires 'min_top_axes' = 4.
        /// </remarks>
        [Description("Optionally, specifies the minimum top axes (default = -1, which ignores this setting).  NOTE: Deconvolution requies 'min_top_axes' = 4.")]
        public int min_top_axes
        {
            get { return m_nMinTopAxes; }
            set { m_nMinTopAxes = value; }
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
            m_nMinTopAxes = p.m_nMinTopAxes;
            m_bEnableNoise = p.m_bEnableNoise;
            m_dfSigmaInit = p.m_dfSigmaInit;
            m_dfBiasGradScale = p.m_dfBiasGradScale;
            m_bOutputContainsPredictions = p.m_bOutputContainsPredictions;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            InnerProductParameter p = new InnerProductParameter();
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

            if (output_contains_predictions)
                rgChildren.Add("output_contains_predictions", output_contains_predictions.ToString());

            rgChildren.Add("num_output", num_output.ToString());
            rgChildren.Add("bias_term", bias_term.ToString());

            if (weight_filler != null)
                rgChildren.Add(weight_filler.ToProto("weight_filler"));

            if (bias_filler != null)
                rgChildren.Add(bias_filler.ToProto("bias_filler"));

            rgChildren.Add("axis", axis.ToString());

            if (transpose != false)
                rgChildren.Add("transpose", transpose.ToString());

            if (min_top_axes != -1)
                rgChildren.Add("min_top_axes", min_top_axes.ToString());

            if (m_bEnableNoise)
            {
                rgChildren.Add("enable_noise", m_bEnableNoise.ToString());
                rgChildren.Add("sigma_init", m_dfSigmaInit.ToString());
            }

            if (bias_grad_scale != 1.0)
                rgChildren.Add("bias_grad_scale", m_dfBiasGradScale.ToString());

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

            if ((strVal = rp.FindValue("output_contains_predictions")) != null)
                p.output_contains_predictions = bool.Parse(strVal);

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

            if ((strVal = rp.FindValue("min_top_axes")) != null)
                p.min_top_axes = int.Parse(strVal);

            if ((strVal = rp.FindValue("enable_noise")) != null)
                p.enable_noise = bool.Parse(strVal);

            if ((strVal = rp.FindValue("sigma_init")) != null)
                p.sigma_init = ParseDouble(strVal);

            if ((strVal = rp.FindValue("bias_grad_scale")) != null)
                p.bias_grad_scale = ParseDouble(strVal);

            return p;
        }
    }
}
