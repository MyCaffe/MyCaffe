using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters used by the MemoryDataLayer.
    /// </summary>
    public class MemoryDataParameter : LayerParameterBase 
    {
        uint m_nBatchSize;
        uint m_nChannels;
        uint m_nHeight;
        uint m_nWidth;
        LABEL_TYPE m_labelType = LABEL_TYPE.SINGLE;
        bool m_bPrimaryData = true;

        /** @copydoc LayerParameterBase */
        public MemoryDataParameter()
        {
        }

        /// <summary>
        /// (\b optional, default = true) Specifies whether or not the data is the primary datset as opposed to a secondary, target dataset.
        /// </summary>
        [Category("Data Selection"), Description("Specifies whether or not this data is the primary dataset as opposed to the target dataset.  By default, this is set to 'true'.")]
        public bool primary_data
        {
            get { return m_bPrimaryData; }
            set { m_bPrimaryData = value; }
        }

        /// <summary>
        /// (\b optional, default = SINGLE) Specifies the label type: SINGLE - the default which uses the 'Label' field, MULTIPLE - which uses the 'DataCriteria' field, or ONEHOTVECTOR - which uses the data itself as the label. Multiple labels are used in tasks such as segmentation learning.  One-Hot-Vectors are used in AutoEncoder learning.  
        /// </summary>
        [Category("Labels"), Description("Specifies the label type: SINGLE - the default which uses the 'Label' field, MULTIPLE - which uses the 'DataCriteria' field, or ONEHOTVECTOR - which uses the data itself as the label. Multiple labels are used in tasks such as segmentation learning.  One-Hot-Vectors are used in AutoEncoder learning.")]
        public LABEL_TYPE label_type
        {
            get { return m_labelType; }
            set { m_labelType = value; }
        }

        /// <summary>
        /// Batch size.
        /// </summary>
        [Description("Batch size.")]
        public uint batch_size
        {
            get { return m_nBatchSize; }
            set { m_nBatchSize = value; }
        }

        /// <summary>
        /// The number of channels in the data.
        /// </summary>
        [Description("The number of channels in the data.")]
        public uint channels
        {
            get { return m_nChannels; }
            set { m_nChannels = value; }
        }

        /// <summary>
        /// The height of the data.
        /// </summary>
        [Description("Specifies the height of the data.")]
        public uint height
        {
            get { return m_nHeight; }
            set { m_nHeight = value; }
        }

        /// <summary>
        /// The width of the data.
        /// </summary>
        [Description("Specifies the width of the data.")]
        public uint width
        {
            get { return m_nWidth; }
            set { m_nWidth = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            MemoryDataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            MemoryDataParameter p = (MemoryDataParameter)src;
            m_nBatchSize = p.m_nBatchSize;
            m_nChannels = p.m_nChannels;
            m_nHeight = p.m_nHeight;
            m_nWidth = p.m_nWidth;
            m_labelType = p.m_labelType;
            m_bPrimaryData = p.m_bPrimaryData;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            MemoryDataParameter p = new MemoryDataParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("batch_size", batch_size.ToString());
            rgChildren.Add("channels", channels.ToString());
            rgChildren.Add("height", height.ToString());
            rgChildren.Add("width", width.ToString());

            if (label_type != LABEL_TYPE.SINGLE)
                rgChildren.Add("label_type", label_type.ToString());

            if (primary_data == false)
                rgChildren.Add("primary_data", primary_data.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static MemoryDataParameter FromProto(RawProto rp)
        {
            string strVal;
            MemoryDataParameter p = new MemoryDataParameter();

            if ((strVal = rp.FindValue("batch_size")) != null)
                p.m_nBatchSize = uint.Parse(strVal);

            if ((strVal = rp.FindValue("channels")) != null)
                p.channels = uint.Parse(strVal);

            if ((strVal = rp.FindValue("height")) != null)
                p.height = uint.Parse(strVal);

            if ((strVal = rp.FindValue("width")) != null)
                p.width = uint.Parse(strVal);

            if ((strVal = rp.FindValue("label_type")) != null)
            {
                switch (strVal)
                {
                    case "SINGLE":
                        p.label_type = LABEL_TYPE.SINGLE;
                        break;

                    case "MULTIPLE":
                        p.label_type = LABEL_TYPE.MULTIPLE;
                        break;

                    default:
                        throw new Exception("Unknown 'label_type' value " + strVal);
                }
            }

            if ((strVal = rp.FindValue("primary_data")) != null)
                p.primary_data = bool.Parse(strVal);

            return p;
        }
    }
}
