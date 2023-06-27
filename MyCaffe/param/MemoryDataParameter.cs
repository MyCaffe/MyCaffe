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
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class MemoryDataParameter : LayerParameterBase 
    {
        uint m_nBatchSize;
        uint m_nDataChannels;
        uint m_nDataHeight;
        uint m_nDataWidth;
        uint m_nLabelChannels = 1;
        uint m_nLabelHeight = 1;
        uint m_nLabelWidth = 1;
        uint m_nClipLength1 = 0;
        uint m_nClipLength2 = 0;
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
        /// (\b optional, default = SINGLE) Specifies the label type: SINGLE - the default which uses the 'Label' field, or MULTIPLE - which uses the 'DataCriteria' field.  
        /// </summary>
        [Category("Labels"), Description("Specifies the label type: SINGLE - the default which uses the 'Label' field, or MULTIPLE - which uses the 'DataCriteria' field.")]
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
            get { return m_nDataChannels; }
            set { m_nDataChannels = value; }
        }

        /// <summary>
        /// The height of the data.
        /// </summary>
        [Description("Specifies the height of the data.")]
        public uint height
        {
            get { return m_nDataHeight; }
            set { m_nDataHeight = value; }
        }

        /// <summary>
        /// The width of the data.
        /// </summary>
        [Description("Specifies the width of the data.")]
        public uint width
        {
            get { return m_nDataWidth; }
            set { m_nDataWidth = value; }
        }

        /// <summary>
        /// The number of channels in the label.
        /// </summary>
        [Description("The number of channels in the label (default = 1).")]
        public uint label_channels
        {
            get { return m_nLabelChannels; }
            set { m_nLabelChannels = value; }
        }

        /// <summary>
        /// The height of the label.
        /// </summary>
        [Description("Specifies the height of the label (default = 1).")]
        public uint label_height
        {
            get { return m_nLabelHeight; }
            set { m_nLabelHeight = value; }
        }

        /// <summary>
        /// The width of the label.
        /// </summary>
        [Description("Specifies the width of the label (default = 1).")]
        public uint label_width
        {
            get { return m_nLabelWidth; }
            set { m_nLabelWidth = value; }
        }

        /// <summary>
        /// Specifies the clip length 1 (default = 0, which means unused).
        /// </summary>
        /// <remarks>
        /// The clip length 1 is only used when a top named 'clip' exists which is used when feeding data
        /// into an LSTM layer which requires a 'clip'input.
        /// </remarks>
        [Description("Specifies the first clip length used with LSTM layers (default = 0, which means unused)")]
        public uint clip_length1
        {
            get { return m_nClipLength1; }
            set { m_nClipLength1 = value; }
        }

        /// <summary>
        /// Specifies the clip length 2 (default = 0, which means unused).
        /// </summary>
        /// <remarks>
        /// The clip length 2 is only used when a top named 'clip' exists which is used when feeding data
        /// into an LSTM layer which requires a 'clip'input.
        /// </remarks>
        [Description("Specifies the second clip length used with LSTM layers (default = 0, which means unused)")]
        public uint clip_length2
        {
            get { return m_nClipLength2; }
            set { m_nClipLength2 = value; }
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
            m_nDataChannels = p.m_nDataChannels;
            m_nDataHeight = p.m_nDataHeight;
            m_nDataWidth = p.m_nDataWidth;
            m_nLabelChannels = p.m_nLabelChannels;
            m_nLabelHeight = p.m_nLabelHeight;
            m_nLabelWidth = p.m_nLabelWidth;
            m_labelType = p.m_labelType;
            m_bPrimaryData = p.m_bPrimaryData;
            m_nClipLength1 = p.m_nClipLength1;
            m_nClipLength2 = p.m_nClipLength2;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            MemoryDataParameter p = new MemoryDataParameter();
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

            rgChildren.Add("batch_size", batch_size.ToString());
            rgChildren.Add("channels", channels.ToString());
            rgChildren.Add("height", height.ToString());
            rgChildren.Add("width", width.ToString());
            rgChildren.Add("label_channels", label_channels.ToString());
            rgChildren.Add("label_height", label_height.ToString());
            rgChildren.Add("label_width", label_width.ToString());

            if (label_type != LABEL_TYPE.SINGLE)
                rgChildren.Add("label_type", label_type.ToString());

            if (primary_data == false)
                rgChildren.Add("primary_data", primary_data.ToString());

            if (m_nClipLength1> 0)
                rgChildren.Add("clip_length1", m_nClipLength1.ToString());

            if (m_nClipLength2 > 0)
                rgChildren.Add("clip_length2", m_nClipLength2.ToString());

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

            if ((strVal = rp.FindValue("label_channels")) != null)
                p.label_channels = uint.Parse(strVal);

            if ((strVal = rp.FindValue("label_height")) != null)
                p.label_height = uint.Parse(strVal);

            if ((strVal = rp.FindValue("label_width")) != null)
                p.label_width = uint.Parse(strVal);

            if ((strVal = rp.FindValue("label_type")) != null)
            {
                switch (strVal)
                {
                    case "NONE":
                        p.label_type = LABEL_TYPE.NONE;
                        break;

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

            if ((strVal = rp.FindValue("clip_length1")) != null)
            {
                int nVal = int.Parse(strVal);
                if (nVal < 0)
                    nVal = 0;

                p.clip_length1 = (uint)nVal;
            }

            if ((strVal = rp.FindValue("clip_length2")) != null)
            {
                int nVal = int.Parse(strVal);
                if (nVal < 0)
                    nVal = 0;

                p.clip_length2 = (uint)nVal;
            }

            return p;
        }
    }
}
