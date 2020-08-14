using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameter for the data normalizer layer.
    /// </summary>
    /// <remarks>
    /// The data normalizer layer provides a detailed normalization that is applied to the data (and label if desired).
    /// </remarks>
    public class DataNormalizerParameter : LayerParameterBase
    {
        bool m_bNormalizeAcrossDataAndLabel = false;
        List<int> m_rgIgnoreChannels = new List<int>();
        List<NORMALIZATION_STEP> m_rgNormalizationSteps = new List<NORMALIZATION_STEP>();
        double m_dfOutputMin = 0;
        double m_dfOutputMax = 1;
        double m_dfInputMin = 0;
        double m_dfInputMax = 0;
        double? m_dfInputStdDev = null;
        double? m_dfInputMean = null;
        int? m_nLabelDataChannel = null;

        /// <summary>
        /// Specifies the normalization step to run.
        /// </summary>
        public enum NORMALIZATION_STEP
        {
            /// <summary>
            /// Center the data by subtracting the mean.
            /// </summary>
            CENTER,
            /// <summary>
            /// Normalize the data by dividing by the standard deviation.
            /// </summary>
            STDEV,
            /// <summary>
            /// Normalize the data by fitting the data into the 'output_min'/'output_max' range.
            /// </summary>
            RANGE,
            /// <summary>
            /// Add each data value to the previous data value.
            /// </summary>
            ADDITIVE,
            /// <summary>
            /// Create the percentage change of the current data from the previous.
            /// </summary>
            RETURNS,
            /// <summary>
            /// Normalize the data by taking the LOG on each item.
            /// </summary>
            LOG
        }

        /** @copydoc LayerParameterBase */
        public DataNormalizerParameter()
        {
        }

        /// <summary>
        /// Specifies to data channel used for the label (if any).
        /// </summary>
        /// <remarks>
        /// Some models, such as LSTM, use input data as part of the label.  The label_data_channel specifies which channel within
        /// the data is used as the label.  When not specified, the label must have the same number of channels as the data.
        /// </remarks>
        [Description("Specifies to data channel used for the label (if any).")]
        public int? label_data_channel
        {
            get { return m_nLabelDataChannel; }
            set { m_nLabelDataChannel = value; }
        }

        /// <summary>
        /// Specifies to normalize across both the data and label data together.
        /// </summary>
        /// <remarks>
        /// When <i>true</i> centering and other normalization takes place across the entire data range within both the data and label together.
        /// When <i>false</i> centering and other normalization takes place separately for data and separately for the label (if supplied).
        /// </remarks>
        [Description("Specifies to normalize across both the data and label data together.")]
        public bool across_data_and_label
        {
            get { return m_bNormalizeAcrossDataAndLabel; }
            set { m_bNormalizeAcrossDataAndLabel = value; }
        }

        /// <summary>
        /// Specifies the normalization steps which are performed in the order for which they are listed.
        /// </summary>
        [Description("Specifies the normalization steps which are performed in the order for which they are listed.")]
        public List<NORMALIZATION_STEP> steps
        {
            get { return m_rgNormalizationSteps; }
            set { m_rgNormalizationSteps = value; }
        }

        /// <summary>
        /// Specifies the channels to ignore and just pass through in their original form.
        /// </summary>
        [Description("Specifies the channels to ignore and just pass through in their original form.")]
        public List<int> ignore_channels
        {
            get { return m_rgIgnoreChannels; }
            set { m_rgIgnoreChannels = value; }
        }

        /// <summary>
        /// Specifies the input standard deviation, if known.  When not specified input_stdev is determined dynamically from the data input itself.
        /// </summary>
        [Description("Specifies the minimum data range of the intput, if known.  When not specified the input_stdev is determined dynamically from the data input itself.")]
        public double? input_stdev
        {
            get { return m_dfInputStdDev; }
            set { m_dfInputStdDev = value; }
        }

        /// <summary>
        /// Specifies the input mean, if known.  When not specified the input_mean is determined dynamically from the data input itself.
        /// </summary>
        [Description("Specifies the minimum data range of the intput, if known.  When not specified the input_mean is determined dynamically from the data input itself.")]
        public double? input_mean
        {
            get { return m_dfInputMean; }
            set { m_dfInputMean = value; }
        }

        /// <summary>
        /// Specifies the minimum data range of the intput, if known.  If both input_min and input_max are 0 the input_min/input_max are determined dynamically from the data input itself.
        /// </summary>
        [Description("Specifies the minimum data range of the intput, if known.  If both input_min and input_max are 0 the input_min/input_max are determined dynamically from the data input itself.")]
        public double input_min
        {
            get { return m_dfInputMin; }
            set { m_dfInputMin = value; }
        }

        /// <summary>
        /// Specifies the maximum data range of the intput, if known.  If both input_min and input_max are 0 the input_min/input_max are determined dynamically from the data input itself.
        /// </summary>
        [Description("Specifies the maximum data range of the intput, if known.  If both input_min and input_max are 0 the input_min/input_max are determined dynamically from the data input itself.")]
        public double input_max
        {
            get { return m_dfInputMax; }
            set { m_dfInputMax = value; }
        }

        /// <summary>
        /// Specifies the minimum data range of the output.
        /// </summary>
        [Description("Specifies the minimum data range of the output.")]
        public double output_min
        {
            get { return m_dfOutputMin; }
            set { m_dfOutputMin = value; }
        }

        /// <summary>
        /// Specifies the maximum data range of the output.
        /// </summary>
        [Description("Specifies the maximum data range of the output.")]
        public double output_max
        {
            get { return m_dfOutputMax; }
            set { m_dfOutputMax = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            DataNormalizerParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            DataNormalizerParameter p = (DataNormalizerParameter)src;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            DataParameter p = new DataParameter();
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
            List<string> rgstrStep = new List<string>();

            for (int i = 0; i < m_rgNormalizationSteps.Count; i++)
            {
                rgstrStep.Add(steps[i].ToString().ToLower());
            }

            rgChildren.Add<string>("step", rgstrStep);           
            rgChildren.Add("across_data_and_label", "\"" + across_data_and_label.ToString() + "\"");
            rgChildren.Add<int>("ignore_ch", ignore_channels);
            rgChildren.Add("input_min", input_min.ToString());
            rgChildren.Add("input_max", input_max.ToString());
            rgChildren.Add("output_min", output_min.ToString());
            rgChildren.Add("output_max", output_max.ToString());

            if (input_mean.HasValue)
                rgChildren.Add("input_mean", input_mean.Value.ToString());

            if (input_stdev.HasValue)
                rgChildren.Add("input_stdev", input_stdev.Value.ToString());

            if (label_data_channel.HasValue)
                rgChildren.Add("label_data_channel", label_data_channel.Value.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <param name="p">Optionally, specifies an instance to load.  If <i>null</i>, a new instance is created and loaded.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static DataNormalizerParameter FromProto(RawProto rp, DataNormalizerParameter p = null)
        {
            string strVal;

            if (p == null)
                p = new DataNormalizerParameter();

            List<string> rgstrStep = rp.FindArray<string>("step");
            p.steps = new List<NORMALIZATION_STEP>();
            
            foreach (string strStep in rgstrStep)
            {
                p.steps.Add(convertStep(strStep));
            }

            if ((strVal = rp.FindValue("across_data_and_label")) != null)
                p.across_data_and_label = bool.Parse(strVal);

            p.ignore_channels = rp.FindArray<int>("ignore_ch");

            if ((strVal = rp.FindValue("input_min")) != null)
                p.input_min = double.Parse(strVal);

            if ((strVal = rp.FindValue("input_max")) != null)
                p.input_max = double.Parse(strVal);

            if ((strVal = rp.FindValue("output_min")) != null)
                p.output_min = double.Parse(strVal);

            if ((strVal = rp.FindValue("output_max")) != null)
                p.output_max = double.Parse(strVal);

            if ((strVal = rp.FindValue("input_mean")) != null)
                p.input_mean = double.Parse(strVal);
            else
                p.input_mean = null;

            if ((strVal = rp.FindValue("input_stdev")) != null)
                p.input_stdev = double.Parse(strVal);
            else
                p.input_stdev = null;

            if ((strVal = rp.FindValue("label_data_channel")) != null)
                p.label_data_channel = int.Parse(strVal);
            else
                p.label_data_channel = null;

            return p;
        }

        private static NORMALIZATION_STEP convertStep(string str)
        {
            str = str.ToUpper();

            if (str == NORMALIZATION_STEP.ADDITIVE.ToString())
                return NORMALIZATION_STEP.ADDITIVE;

            if (str == NORMALIZATION_STEP.CENTER.ToString())
                return NORMALIZATION_STEP.CENTER;

            if (str == NORMALIZATION_STEP.LOG.ToString())
                return NORMALIZATION_STEP.LOG;

            if (str == NORMALIZATION_STEP.RANGE.ToString())
                return NORMALIZATION_STEP.RANGE;

            if (str == NORMALIZATION_STEP.RETURNS.ToString())
                return NORMALIZATION_STEP.RETURNS;

            if (str == NORMALIZATION_STEP.STDEV.ToString())
                return NORMALIZATION_STEP.STDEV;

            throw new Exception("The step '" + str + "' is unknown!");
        }
    }
}
