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
    /// Specifies the parameters for the VarSelNetLayer (Variable Selection Network).  
    /// </summary>
    /// <remarks>
    /// The VSN enables instance-wise variable selection and is applied to both the static covariates and time-dependent covariates as the
    /// specific contribution of each input to the output is typically unknown.  The VSN provides insights into which variables contribute
    /// the most for the prediction problem and allows the model to remove unnecessarily noisy inputs which could negatively impact the
    /// performance.
    /// 
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L149) by Playtika Research, 2021.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class VarSelNetParameter : LayerParameterBase
    {
        int m_nNumInputs;
        int m_nInputDim;
        int m_nHiddenDim;
        float m_fDropout = 0.05f;
        int? m_nContextDim = null;
        bool m_bBatchFirst = true;
        FillerParameter m_fillerParam_weights = new FillerParameter("xavier");
        FillerParameter m_fillerParam_bias = new FillerParameter("constant", 0.1);
        int m_nAxis = 1;

        /** @copydoc LayerParameterBase */
        public VarSelNetParameter()
        {
        }

        /// <summary>
        /// Specifies the quantity of input variables, including both numeric and categorical for the relevant channel.
        /// </summary>
        [Description("Specifies the quantity of input variables, including both numeric and categorical for the relevant channel.")]
        public int num_inputs
        {
            get { return m_nNumInputs; }
            set { m_nNumInputs = value; }
        }

        /// <summary>
        /// Specifies the attribute/embedding dimension of the input, associated witht he 'state_size' of the model.
        /// </summary>
        [Description("Specifies the attribute/embedding dimension of the input, associated witht he 'state_size' of the model.")]
        public int input_dim
        {
            get { return m_nInputDim; }
            set { m_nInputDim = value; }
        }

        /// <summary>
        /// Specifies the embedding width of the output.
        /// </summary>
        [Description("Specifies the embedding width of the output.")]
        public int hidden_dim
        {
            get { return m_nHiddenDim; }
            set { m_nHiddenDim = value; }
        }

        /// <summary>
        /// Specifies the embedding width of the context signal expected to be fed as an auxiliary input (optional, can be null).
        /// </summary>
        [Description("Specifies the embedding width of the context signal expected to be fed as an auxiliary input (optional, can be null).")]
        public int? context_dim
        {
            get { return m_nContextDim; }
            set { m_nContextDim = value; }
        }

        /// <summary>
        /// Specifies the dropout ratio used with the GRNs (default = 0.05 or 5%).
        /// </summary>
        [Description("Specifies the dropout ratio used with the GRNs (default = 0.05 or 5%).")]
        public float dropout_ratio
        {
            get { return m_fDropout; }
            set { m_fDropout = value; }
        }

        /// <summary>
        /// Specifies a boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
        /// </summary>
        [Description("Specifies a boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.")]
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
        /// Specifies the first axis to be lumped into a single inner product computations;
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
            VarSelNetParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            VarSelNetParameter p = (VarSelNetParameter)src;

            m_nNumInputs = p.num_inputs;
            m_nInputDim = p.input_dim;
            m_nHiddenDim = p.hidden_dim;
            m_nContextDim = p.context_dim;
            m_fDropout = p.dropout_ratio;
            m_bBatchFirst = p.batch_first;
            m_nAxis = p.m_nAxis;

            if (p.m_fillerParam_bias != null)
                m_fillerParam_bias = p.m_fillerParam_bias.Clone();

            if (p.m_fillerParam_weights != null)
                m_fillerParam_weights = p.m_fillerParam_weights.Clone();
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            VarSelNetParameter p = new VarSelNetParameter();
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
            rgChildren.Add("num_inputs", num_inputs.ToString());
            if (context_dim.HasValue)
                rgChildren.Add("context_dim", context_dim.Value.ToString());
            rgChildren.Add("dropout_ratio", dropout_ratio.ToString());
            rgChildren.Add("batch_first", batch_first.ToString());

            if (weight_filler != null)
                rgChildren.Add(weight_filler.ToProto("weight_filler"));

            if (bias_filler != null)
                rgChildren.Add(bias_filler.ToProto("bias_filler"));

            rgChildren.Add("axis", axis.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static VarSelNetParameter FromProto(RawProto rp)
        {
            string strVal;
            VarSelNetParameter p = new VarSelNetParameter();

            if ((strVal = rp.FindValue("input_dim")) != null)
                p.input_dim = int.Parse(strVal);

            if ((strVal = rp.FindValue("hidden_dim")) != null)
                p.hidden_dim = int.Parse(strVal);

            if ((strVal = rp.FindValue("num_inputs")) != null)
                p.num_inputs = int.Parse(strVal);

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

            return p;
        }
    }
}
