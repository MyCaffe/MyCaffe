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

        /** @copydoc LayerParameterBase */
        public MemoryDataParameter()
        {
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

            return p;
        }
    }
}
