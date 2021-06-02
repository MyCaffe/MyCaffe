using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameter for the Text data layer.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class TextDataParameter : LayerParameterBase
    {
        string m_strEncoderSource = null;
        string m_strDecoderSource = null;
        uint m_nBatchSize = 1;
        uint m_nTimeSteps = 80;
        uint m_nSampleSize = 1000;
        bool m_bShuffle = true;
        bool m_bEnableNormalEncoderOutput = true;
        bool m_bEnableReverseEncoderOutput = true;

        /// <summary>
        /// This event is, optionally, called to verify the batch size of the TextDataParameter.
        /// </summary>
        public event EventHandler<VerifyBatchSizeArgs> OnVerifyBatchSize;

        /** @copydoc LayerParameterBase */
        public TextDataParameter()
        {
        }

        /// <summary>
        /// This method gives derivative classes a chance specify model inputs required
        /// by the run model.
        /// </summary>
        /// <returns>The model inputs required by the layer (if any) or null.</returns>
        public override string PrepareRunModelInputs()
        {
            string strInput = "";
            int nBatch = (int)m_nBatchSize;

            strInput += "input: \"idec\"" + Environment.NewLine;
            strInput += "input_shape { dim: 1 dim: " + nBatch.ToString() + " dim: 1 } " + Environment.NewLine;

            strInput += "input: \"ienc\"" + Environment.NewLine;
            strInput += "input_shape { dim: " + m_nTimeSteps.ToString() + " dim: " + nBatch.ToString() + " dim: 1 } " + Environment.NewLine;

            strInput += "input: \"iencr\"" + Environment.NewLine;
            strInput += "input_shape { dim: " + m_nTimeSteps.ToString() + " dim: " + nBatch.ToString() + " dim: 1 } " + Environment.NewLine;

            strInput += "input: \"iencc\"" + Environment.NewLine;
            strInput += "input_shape { dim: " + m_nTimeSteps.ToString() + " dim: " + nBatch.ToString() + " } " + Environment.NewLine;

            return strInput;
        }

        /// <summary>
        /// This method gives derivative classes a chance modify the layer parameter for a run model.
        /// </summary>
        public override void PrepareRunModel(LayerParameter p)
        {
            p.bottom.Add("idec");
            p.bottom.Add("ienc");
            p.bottom.Add("iencr");
            p.bottom.Add("iencc");
        }

        /// <summary>
        /// Specifies the encoder data source.
        /// </summary>
        [Description("Specifies the encoder data source.")]
        public string encoder_source
        {
            get { return m_strEncoderSource; }
            set { m_strEncoderSource = value; }
        }

        /// <summary>
        /// Specifies the decoder data source.
        /// </summary>
        [Description("Specifies the decoder data source.")]
        public string decoder_source
        {
            get { return m_strDecoderSource; }
            set { m_strDecoderSource = value; }
        }

        /// <summary>
        /// Specifies the batch size.
        /// </summary>
        [Description("Specifies the batch size of images to collect and train on each iteration of the network.  NOTE: Setting the training netorks batch size >= to the testing net batch size will conserve memory by allowing the training net to share its gpu memory with the testing net.")]
        public virtual uint batch_size
        {
            get { return m_nBatchSize; }
            set
            {
                if (OnVerifyBatchSize != null)
                {
                    VerifyBatchSizeArgs args = new VerifyBatchSizeArgs(value);
                    OnVerifyBatchSize(this, args);
                    if (args.Error != null)
                        throw args.Error;
                }

                m_nBatchSize = value;
            }
        }

        /// <summary>
        /// Specifies the maximum length for each encoder input.
        /// </summary>
        [Description("Specifies the maximum length for the encoder inputs.")]
        public uint time_steps
        {
            get { return m_nTimeSteps; }
            set { m_nTimeSteps = value; }
        }

        /// <summary>
        /// Specifies the sample size to select from the data sources.
        /// </summary>
        [Description("Specifies the sample size to select from the data sources.")]
        public uint sample_size
        {
            get { return m_nSampleSize; }
            set { m_nSampleSize = value; }
        }

        /// <summary>
        /// Specifies the whether to shuffle the data or now.
        /// </summary>
        [Description("Specifies whether to shuffle the data or now.")]
        public bool shuffle
        {
            get { return m_bShuffle; }
            set { m_bShuffle = value; }
        }

        /// <summary>
        /// When enabled, the normal ordered encoder data is output (default = true).
        /// </summary>
        [Description("When enabled, the normal ordered encoder data is output (default = true).")]
        public bool enable_normal_encoder_output
        {
            get { return m_bEnableNormalEncoderOutput; }
            set { m_bEnableNormalEncoderOutput = value; }
        }

        /// <summary>
        /// When enabled, the reverse ordered encoder data is output (default = true).
        /// </summary>
        [Description("When enabled, the reverse ordered encoder data is output (default = true).")]
        public bool enable_reverse_encoder_output
        {
            get { return m_bEnableReverseEncoderOutput; }
            set { m_bEnableReverseEncoderOutput = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            TextDataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            TextDataParameter p = (TextDataParameter)src;
            m_strEncoderSource = p.m_strEncoderSource;
            m_strDecoderSource = p.m_strDecoderSource;
            m_nBatchSize = p.m_nBatchSize;
            m_nTimeSteps = p.m_nTimeSteps;
            m_nSampleSize = p.m_nSampleSize;
            m_bShuffle = p.m_bShuffle;
            m_bEnableNormalEncoderOutput = p.m_bEnableNormalEncoderOutput;
            m_bEnableReverseEncoderOutput = p.m_bEnableReverseEncoderOutput;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            TextDataParameter p = new TextDataParameter();
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

            rgChildren.Add("encoder_source", "\"" + encoder_source + "\"");
            rgChildren.Add("decoder_source", "\"" + decoder_source + "\"");
            rgChildren.Add("batch_size", batch_size.ToString());
            rgChildren.Add("time_steps", time_steps.ToString());
            rgChildren.Add("sample_size", sample_size.ToString());
            rgChildren.Add("shuffle", shuffle.ToString());
            rgChildren.Add("enable_normal_encoder_output", enable_normal_encoder_output.ToString());
            rgChildren.Add("enable_reverse_encoder_output", enable_reverse_encoder_output.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <param name="p">Optionally, specifies an instance to load.  If <i>null</i>, a new instance is created and loaded.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static TextDataParameter FromProto(RawProto rp, TextDataParameter p = null)
        {
            string strVal;

            if (p == null)
                p = new TextDataParameter();

            if ((strVal = rp.FindValue("encoder_source")) != null)
                p.encoder_source = strVal.Trim('\"');

            if ((strVal = rp.FindValue("decoder_source")) != null)
                p.decoder_source = strVal.Trim('\"');

            if ((strVal = rp.FindValue("batch_size")) != null)
                p.batch_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("time_steps")) != null)
                p.time_steps = uint.Parse(strVal);

            if ((strVal = rp.FindValue("shuffle")) != null)
                p.shuffle = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_normal_encoder_output")) != null)
                p.enable_normal_encoder_output = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_reverse_encoder_output")) != null)
                p.enable_reverse_encoder_output = bool.Parse(strVal);

            return p;
        }
    }
}
