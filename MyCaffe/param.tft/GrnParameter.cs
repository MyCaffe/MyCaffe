using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using static MyCaffe.param.tft.GluParameter;

namespace MyCaffe.param.tft
{
    /// <summary>
    /// Specifies the parameters for the GrnLayer (Gated Response Network).  
    /// </summary>
    /// <remarks>
    /// This layer takes as input a primary input 'x' and optional context vector 'c'.  A GLU (Gated Linear Unit) is used
    /// for controlling the extent to which the module will contribute to the original input 'x', potentially skipping
    /// over the layer entirely as the GLU outputs could all be close to zero by the GLU supressing.  In cases where
    /// no context vector is used, the GRN treats the context input as zero.  During training dropout is applied before
    /// gating.
    /// 
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L44) by Playtika Research, 2021.
    /// </remarks>
    public class GrnParameter : LayerParameterBase
    {
        int m_nInputDim;
        int m_nHiddenDim;
        int m_nOutputDim;
        float m_fDropoutRatio = 0.05f;
        int? m_nContextDim = null;
        bool m_bBatchFirst = true;
        FillerParameter m_fillerParam_weights = new FillerParameter("xavier");
        FillerParameter m_fillerParam_bias = new FillerParameter("constant", 0.1);
        int m_nAxis = 1;
        ACTIVATION m_activaiton = ACTIVATION.ELU;

        /// <summary>
        /// Defines the activation type.
        /// </summary>
        public enum ACTIVATION
        {
            /// <summary>
            /// Specifies to use an ELU activation (default).
            /// </summary>
            ELU,

            /// <summary>
            /// Specifies to use a RELU activation.
            /// </summary>
            RELU
        }

        /** @copydoc LayerParameterBase */
        public GrnParameter()
        {
        }

        /// <summary>
        /// Specifies the activation type to use (default=ELU)
        /// </summary>
        [Description("Specifies the activation type to use (default=ELU)")]
        public ACTIVATION activation
        {
            get { return m_activaiton; }
            set { m_activaiton = value; }
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
        /// Specifies the input dimension.
        /// </summary>
        [Description("Specifies the hidden dimension.")]
        public int hidden_dim
        {
            get { return m_nHiddenDim; }
            set { m_nHiddenDim = value; }
        }

        /// <summary>
        /// Specifies the output dimension.
        /// </summary>
        [Description("Specifies the output dimension.")]
        public int output_dim
        {
            get { return m_nOutputDim; }
            set { m_nOutputDim = value; }
        }

        /// <summary>
        /// Specifies the context dimension, or null to ignore.
        /// </summary>
        [Description("Specifies the context dimension, or null to ignore.")]
        public int? context_dim
        {
            get { return m_nContextDim; }
            set { m_nContextDim = value; }
        }

        /// <summary>
        /// Specifies the dropout ratio (default = 0.05 or 5%).
        /// </summary>
        [Description("Specifies the dropout ratio (default = 0.05 or 5%).")]
        public float dropout_ratio
        {
            get { return m_fDropoutRatio; }
            set { m_fDropoutRatio = value; }
        }

        /// <summary>
        /// Specifies the batch_first setting.
        /// </summary>
        [Description("Specifies the batch_first setting.")]
        public bool batch_first
        {
            get { return m_bBatchFirst; }
            set { m_bBatchFirst = value; }
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
            GrnParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            GrnParameter p = (GrnParameter)src;

            m_nInputDim = p.input_dim;
            m_nHiddenDim = p.hidden_dim;
            m_nOutputDim = p.output_dim;
            m_nContextDim = p.context_dim;
            m_fDropoutRatio = p.dropout_ratio;
            m_bBatchFirst = p.batch_first;
            m_nAxis = p.m_nAxis;
            m_activaiton = p.activation;

            if (p.m_fillerParam_bias != null)
                m_fillerParam_bias = p.m_fillerParam_bias.Clone();

            if (p.m_fillerParam_weights != null)
                m_fillerParam_weights = p.m_fillerParam_weights.Clone();
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            GrnParameter p = new GrnParameter();
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

            rgChildren.Add("input_dim", input_dim.ToString());
            rgChildren.Add("hidden_dim", hidden_dim.ToString());
            rgChildren.Add("output_dim", output_dim.ToString());
            if (context_dim.HasValue)
                rgChildren.Add("context_dim", context_dim.Value.ToString());
            rgChildren.Add("dropout_ratio", dropout_ratio.ToString());
            rgChildren.Add("batch_first", batch_first.ToString());

            if (weight_filler != null)
                rgChildren.Add(weight_filler.ToProto("weight_filler"));

            if (bias_filler != null)
                rgChildren.Add(bias_filler.ToProto("bias_filler"));

            rgChildren.Add("axis", axis.ToString());
            rgChildren.Add("activation", activation.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static GrnParameter FromProto(RawProto rp)
        {
            string strVal;
            GrnParameter p = new GrnParameter();

            if ((strVal = rp.FindValue("input_dim")) != null)
                p.input_dim = int.Parse(strVal);

            if ((strVal = rp.FindValue("hidden_dim")) != null)
                p.hidden_dim = int.Parse(strVal);

            if ((strVal = rp.FindValue("output_dim")) != null)
                p.output_dim = int.Parse(strVal);

            if ((strVal = rp.FindValue("context_dim")) != null)
                p.context_dim = int.Parse(strVal);

            if ((strVal = rp.FindValue("dropout_ratio")) != null)
                p.dropout_ratio = float.Parse(strVal);

            if ((strVal = rp.FindValue("batch_first")) != null)
                p.batch_first = bool.Parse(strVal);

            RawProto rpWeightFiller = rp.FindChild("weight_filler");
            if (rpWeightFiller != null)
                p.weight_filler = FillerParameter.FromProto(rpWeightFiller);

            RawProto rpBiasFiller = rp.FindChild("bias_filler");
            if (rpBiasFiller != null)
                p.bias_filler = FillerParameter.FromProto(rpBiasFiller);

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("activation")) != null)
            {
                if (strVal == ACTIVATION.RELU.ToString())
                    p.activation = ACTIVATION.RELU;
                else
                    p.activation = ACTIVATION.ELU;
            }

            return p;
        }
    }
}
