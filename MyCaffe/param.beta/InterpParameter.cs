using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the InterpLayer.
    /// </summary>
    /// <remarks>
    /// </remarks>
    public class InterpParameter : LayerParameterBase
    {
        int? m_nHeight = null;
        int? m_nWidth = null;
        int? m_nZoomFactor = null;
        int? m_nShrinkFactor = null;
        int m_nPadBeg = 0;
        int m_nPadEnd = 0;

        /** @copydoc LayerParameterBase */
        public InterpParameter()
        {
        }

        /// <summary>
        /// Specifies the height of the output.
        /// </summary>
        [Description("Specifies the height of the output.")]
        public int? height
        {
            get { return m_nHeight; }
            set { m_nHeight = value; }
        }

        /// <summary>
        /// Specifies the width of the output.
        /// </summary>
        [Description("Specifies the width of the output.")]
        public int? width
        {
            get { return m_nWidth; }
            set { m_nWidth = value; }
        }

        /// <summary>
        /// Specifies the height of the output.
        /// </summary>
        [Description("Specifies the zoom factor of the output.")]
        public int? zoom_factor
        {
            get { return m_nZoomFactor; }
            set { m_nZoomFactor = value; }
        }

        /// <summary>
        /// Specifies the shrink factor of the output.
        /// </summary>
        [Description("Specifies the shrink factor of the output.")]
        public int? shrink_factor
        {
            get { return m_nShrinkFactor; }
            set { m_nShrinkFactor = value; }
        }

        /// <summary>
        /// Specifies the padding at the begin of the output.
        /// </summary>
        [Description("Specifies the padding at the begin of the output.")]
        public int pad_beg
        {
            get { return m_nPadBeg; }
            set { m_nPadBeg = value; }
        }

        /// <summary>
        /// Specifies the padding at the end of the output.
        /// </summary>
        [Description("Specifies the padding at the end of the output.")]
        public int pad_end
        {
            get { return m_nPadEnd; }
            set { m_nPadEnd = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            InterpParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            InterpParameter p = (InterpParameter)src;

            m_nHeight = p.height;
            m_nWidth = p.width;
            m_nZoomFactor = p.zoom_factor;
            m_nShrinkFactor = p.shrink_factor;
            m_nPadBeg = p.pad_beg;
            m_nPadEnd = p.pad_end;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            InterpParameter p = new InterpParameter();
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

            if (height.HasValue)
                rgChildren.Add("height", height.ToString());
            if (width.HasValue)
                rgChildren.Add("width", width.ToString());
            if (zoom_factor.HasValue)
                rgChildren.Add("zoom_factor", zoom_factor.ToString());
            if (shrink_factor.HasValue)
                rgChildren.Add("shrink_factor", shrink_factor.ToString());
            rgChildren.Add("pad_beg", pad_beg.ToString());
            rgChildren.Add("pad_end", pad_end.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static InterpParameter FromProto(RawProto rp)
        {
            string strVal;
            InterpParameter p = new InterpParameter();

            if ((strVal = rp.FindValue("height")) != null)
                p.height = int.Parse(strVal);

            if ((strVal = rp.FindValue("width")) != null)
                p.width = int.Parse(strVal);

            if ((strVal = rp.FindValue("zoom_factor")) != null)
                p.zoom_factor = int.Parse(strVal);

            if ((strVal = rp.FindValue("shrink_factor")) != null)
                p.shrink_factor = int.Parse(strVal);

            if ((strVal = rp.FindValue("pad_beg")) != null)
                p.pad_beg = int.Parse(strVal);

            if ((strVal = rp.FindValue("pad_end")) != null)
                p.pad_end = int.Parse(strVal);

            return p;
        }
    }
}
