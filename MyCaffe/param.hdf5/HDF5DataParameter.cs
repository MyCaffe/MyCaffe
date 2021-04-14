using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameter for the HDF5 data layer.
    /// </summary>
    /// <remarks>
    /// Note: given the new use of the Transformation Parameter, the
    /// depreciated elements of the HDF5DataParameter have been removed.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class HDF5DataParameter : LayerParameterBase
    {
        string m_strSource = null;
        uint m_nBatchSize;
        bool m_bShuffle;

        /// <summary>
        /// This event is, optionally, called to verify the batch size of the HDF5DataParameter.
        /// </summary>
        public event EventHandler<VerifyBatchSizeArgs> OnVerifyBatchSize;

        /** @copydoc LayerParameterBase */
        public HDF5DataParameter()
        {
        }

        /// <summary>
        /// Specifies the data source.
        /// </summary>
        [Description("Specifies the data source.")]
        public string source
        {
            get { return m_strSource; }
            set { m_strSource = value; }
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
            HDF5DataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            HDF5DataParameter p = (HDF5DataParameter)src;
            m_strSource = p.m_strSource;
            m_nBatchSize = p.m_nBatchSize;
            m_bShuffle = p.m_bShuffle;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            HDF5DataParameter p = new HDF5DataParameter();
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

            rgChildren.Add("source", "\"" + source + "\"");
            rgChildren.Add("batch_size", batch_size.ToString());
            rgChildren.Add("backend", shuffle.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <param name="p">Optionally, specifies an instance to load.  If <i>null</i>, a new instance is created and loaded.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static HDF5DataParameter FromProto(RawProto rp, HDF5DataParameter p = null)
        {
            string strVal;

            if (p == null)
                p = new HDF5DataParameter();

            if ((strVal = rp.FindValue("source")) != null)
                p.source = strVal.Trim('\"');

            if ((strVal = rp.FindValue("batch_size")) != null)
                p.batch_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("shuffle")) != null)
                p.shuffle = bool.Parse(strVal);

            return p;
        }
    }
}
