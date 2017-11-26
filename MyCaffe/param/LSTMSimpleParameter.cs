using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the LSTMSimpleLayer.
    /// </summary>
    /// <remarks>
    /// @see [A Clockwork RNN](https://arxiv.org/abs/1402.3511) by Jan Koutnik, Klaus Greff, Faustino Gomez, and Jürgen Schmidhuber, 2014.
    /// @see [Long short-term memory](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.56.7752) by Sepp Hochreiter and Jürgen Schmidhuber, 1997.
    /// @see [Learning to execute](https://arxiv.org/abs/1410.4615) by Wojciech Zaremba and Ilya Sutskever, 2014.
    /// @see [Generating sequences with recurrent neural networks](https://arxiv.org/abs/1308.0850) by Alex Graves, 2013.
    /// @see [Predictive Business Process Monitoring with LSTM Neural Networks](https://arxiv.org/abs/1612.02130) by Niek Tax, Ilya Verenich, Marcello La Rosa, and Marlon Dumas, 2016. 
    /// @see [Using LSTM recurrent neural networks for detecting anomalous behavior of LHC superconducting magnets](https://arxiv.org/abs/1611.06241) by Maciej Wielgosz, Andrzej Skoczeń, and Matej Mertik, 2016.
    /// @see [Spatial, Structural and Temporal Feature Learning for Human Interaction Prediction](https://arxiv.org/abs/1608.05267v2) by Qiuhong Ke, Mohammed Bennamoun, Senjian An, Farid Bossaid, and Ferdous Sohel, 2016.
    /// </remarks>
    public class LSTMSimpleParameter : LayerParameterBase
    {
        uint m_nNumOutput;
        double m_dfClippingThreshold = 0;
        FillerParameter m_fillerWeights = new FillerParameter("xavier");
        FillerParameter m_fillerBias = new FillerParameter("constant", 0.1);
        uint m_nBatchSize = 1;
        bool m_bEnableClockworkForgetGateBias = false;

        /** @copydoc LayerParameterBase */
        public LSTMSimpleParameter()
        {
        }

        /// <summary>
        /// Specifies the number of outputs for the layer.
        /// </summary>
        [Description("Specifies the number of outputs for the layer.")]
        public uint num_output
        {
            get { return m_nNumOutput; }
            set { m_nNumOutput = value; }
        }

        /// <summary>
        /// Specifies the gradient clipping threshold, default = 0.0 (i.e. no clipping).
        /// </summary>
        [Description("Specifies the gradient clipping threshold, default = 0.0 (i.e. no clipping).")]
        public double clipping_threshold
        {
            get { return m_dfClippingThreshold; }
            set { m_dfClippingThreshold = value; }
        }

        /// <summary>
        /// Specifies the filler parameters for the weight filler.
        /// </summary>
        [Description("Specifies the filler parameters for the weight filler.")]
        [TypeConverter(typeof(ExpandableObjectConverter))]
        public FillerParameter weight_filler
        {
            get { return m_fillerWeights; }
            set { m_fillerWeights = value; }
        }

        /// <summary>
        /// Specifies the filler parameters for the bias filler.
        /// </summary>
        [Description("Specifies the filler parameters for the bias filler.")]
        [TypeConverter(typeof(ExpandableObjectConverter))]
        public FillerParameter bias_filler
        {
            get { return m_fillerBias; }
            set { m_fillerBias = value; }
        }

        /// <summary>
        /// Specifies the batch size, default = 1.
        /// </summary>
        [Description("Specifies the batch size, default = 1.")]
        public uint batch_size
        {
            get { return m_nBatchSize; }
            set { m_nBatchSize = value; }
        }

        /// <summary>
        /// When enabled, the forget gate bias is set to 5.0.
        /// </summary>
        /// <remarks>
        /// @see [A Clockwork RNN](https://arxiv.org/abs/1402.3511) by Koutnik, et al., 2014
        /// </remarks>
        [Description("When true, the forget gate bias is set to 5.0 as recommended by [1] Koutnik, J., Greff, K., Gomez, F., Schmidhuber, J., 'A Clockwork RNN', 2014")]
        public bool enable_clockwork_forgetgate_bias
        {
            get { return m_bEnableClockworkForgetGateBias; }
            set { m_bEnableClockworkForgetGateBias = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            LSTMSimpleParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            LSTMSimpleParameter p = (LSTMSimpleParameter)src;

            m_nNumOutput = p.m_nNumOutput;
            m_dfClippingThreshold = p.m_dfClippingThreshold;
            m_fillerWeights = p.m_fillerWeights.Clone();
            m_fillerBias = p.m_fillerBias.Clone();
            m_nBatchSize = p.m_nBatchSize;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            LSTMSimpleParameter p = new LSTMSimpleParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("num_output", m_nNumOutput.ToString());

            if (m_dfClippingThreshold != 0)
                rgChildren.Add("clipping_threshold", m_dfClippingThreshold.ToString());

            rgChildren.Add(m_fillerWeights.ToProto("weight_filler"));
            rgChildren.Add(m_fillerBias.ToProto("bias_filler"));

            if (m_nBatchSize != 1)
                rgChildren.Add("batch_size", m_nBatchSize.ToString());

            if (m_bEnableClockworkForgetGateBias != false)
                rgChildren.Add("enable_clockwork_forgetgate_bias", m_bEnableClockworkForgetGateBias.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static LSTMSimpleParameter FromProto(RawProto rp)
        {
            string strVal;
            LSTMSimpleParameter p = new LSTMSimpleParameter();

            if ((strVal = rp.FindValue("num_output")) != null)
                p.num_output = uint.Parse(strVal);

            if ((strVal = rp.FindValue("clipping_threshold")) != null)
                p.clipping_threshold = double.Parse(strVal);

            RawProto rpWeightFiller = rp.FindChild("weight_filler");
            if (rpWeightFiller != null)
                p.weight_filler = FillerParameter.FromProto(rpWeightFiller);

            RawProto rpBiasFiller = rp.FindChild("bias_filler");
            if (rpBiasFiller != null)
                p.bias_filler = FillerParameter.FromProto(rpBiasFiller);

            if ((strVal = rp.FindValue("batch_size")) != null)
                p.batch_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("enable_clockwork_forgetgate_bias")) != null)
                p.enable_clockwork_forgetgate_bias = bool.Parse(strVal);

            return p;
        }
    }
}
