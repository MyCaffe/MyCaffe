using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameter for the model data layer.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ModelDataParameter : LayerParameterBase
    {
        List<string> m_rgstrSource = new List<string>();
        uint m_nBatchSize = 1;
        uint m_nTimeSteps = 80;
        uint m_nInputDim = 5;
        uint m_nSampleSize = 1000;
        bool m_bShuffle = true;

        /// <summary>
        /// This event is, optionally, called to verify the batch size of the TextDataParameter.
        /// </summary>
        public event EventHandler<VerifyBatchSizeArgs> OnVerifyBatchSize;

        /** @copydoc LayerParameterBase */
        public ModelDataParameter()
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
            int nInput = (int)m_nInputDim;

            strInput += "input: \"idec\"" + Environment.NewLine;
            strInput += "input_shape { dim: 1 dim: " + nBatch.ToString() + " dim: 1 } " + Environment.NewLine;

            strInput += "input: \"ienc\"" + Environment.NewLine;
            strInput += "input_shape { dim: " + m_nTimeSteps.ToString() + " dim: " + nBatch.ToString() + " dim: " + nInput.ToString() + " } " + Environment.NewLine;

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
            p.bottom.Add("iencc");
        }

        /// <summary>
        /// Specifies the data 'sources' within the database. Each source must already have pre-calculated RawImageResult data within the RawImageResults table including both Results for encoder input and ExtraData for decoder input.
        /// </summary>
        [Description("Specifies the data 'sources' within the database. Each source must already have pre-calculated RawImageResult data within the RawImageResults table including both Results for encoder input and ExtraData for decoder input.")]
        public List<string> source
        {
            get { return m_rgstrSource; }
            set { m_rgstrSource = value; }
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
        /// Specifies the input dimension for each encoder input.
        /// </summary>
        [Description("Specifies the input dimension the encoder inputs.")]
        public uint input_dim
        {
            get { return m_nInputDim; }
            set { m_nInputDim = value; }
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

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ModelDataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            ModelDataParameter p = (ModelDataParameter)src;
            m_rgstrSource = Utility.Clone<string>(p.source);
            m_nBatchSize = p.m_nBatchSize;
            m_nTimeSteps = p.m_nTimeSteps;
            m_nInputDim = p.m_nInputDim;
            m_nSampleSize = p.m_nSampleSize;
            m_bShuffle = p.m_bShuffle;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ModelDataParameter p = new ModelDataParameter();
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

            rgChildren.Add<string>("source", m_rgstrSource);
            rgChildren.Add("batch_size", batch_size.ToString());
            rgChildren.Add("time_steps", time_steps.ToString());
            rgChildren.Add("input_dim", input_dim.ToString());
            rgChildren.Add("sample_size", sample_size.ToString());
            rgChildren.Add("shuffle", shuffle.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <param name="p">Optionally, specifies an instance to load.  If <i>null</i>, a new instance is created and loaded.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ModelDataParameter FromProto(RawProto rp, ModelDataParameter p = null)
        {
            string strVal;

            if (p == null)
                p = new ModelDataParameter();

            p.source = rp.FindArray<string>("source");

            if ((strVal = rp.FindValue("batch_size")) != null)
                p.batch_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("time_steps")) != null)
                p.time_steps = uint.Parse(strVal);

            if ((strVal = rp.FindValue("input_dim")) != null)
                p.input_dim = uint.Parse(strVal);

            if ((strVal = rp.FindValue("shuffle")) != null)
                p.shuffle = bool.Parse(strVal);

            return p;
        }
    }
}
