using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.ssd
{
    /// <summary>
    /// Specifies the parameters for the Normalization2Layer used in SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class Normalization2Parameter : LayerParameterBase 
    {
        bool m_bAcrossSpatial = true;        
        FillerParameter m_scaleFiller = new FillerParameter("constant", 1.0);
        bool m_bChannelShared = true;
        float m_fEps = 1e-10f;

        /// <summary>
        /// The constructor.
        /// </summary>
        public Normalization2Parameter()
        {
        }

        /// <summary>
        /// Specifies to normalize across the spatial dimensions.
        /// </summary>
        [Description("Specifies to normalize across the spatial dimensions.")]
        public bool across_spatial
        {
            get { return m_bAcrossSpatial; }
            set { m_bAcrossSpatial = value; }
        }

        /// <summary>
        /// Specifies the filler for the initial value of scale, default is 1.0 for all.
        /// </summary>
        [Description("Specifies the filler for the initial value of scale, default is 1.0 for all.")]
        public FillerParameter scale_filler
        {
            get { return m_scaleFiller; }
            set { m_scaleFiller = value; }
        }

        /// <summary>
        /// Specifies whether or not the scale parameters are shared across channels.
        /// </summary>
        [Description("Specifies whether or not the scale parameters are shared across channels.")]
        public bool channel_shared
        {
            get { return m_bChannelShared; }
            set { m_bChannelShared = value; }
        }

        /// <summary>
        /// Specifies the epsilon for not dividing by zero while normalizing variance.
        /// </summary>
        [Description("Specifies the epsilon for not dividing by zero while normalizing variance.")]
        public float eps
        {
            get { return m_fEps; }
            set { m_fEps = value; }
        }

        /// <summary>
        /// Load the parameter from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bNewInstance">When <i>true</i> a new instance is created (the default), otherwise the existing instance is loaded from the binary reader.</param>
        /// <returns>Returns an instance of the parameter.</returns>
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            Normalization2Parameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Copy on parameter to another.
        /// </summary>
        /// <param name="src">Specifies the parameter to copy.</param>
        public override void Copy(LayerParameterBase src)
        {
            Normalization2Parameter p = (Normalization2Parameter)src;
            m_bAcrossSpatial = p.m_bAcrossSpatial;
            m_bChannelShared = p.m_bChannelShared;
            m_fEps = p.m_fEps;
            m_scaleFiller = p.m_scaleFiller.Clone();
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public override LayerParameterBase Clone()
        {
            Normalization2Parameter p = new Normalization2Parameter();
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

            rgChildren.Add("across_spatial", m_bAcrossSpatial.ToString());
            rgChildren.Add("channel_shared", m_bChannelShared.ToString());
            rgChildren.Add("esp", m_fEps.ToString());
            rgChildren.Add(scale_filler.ToProto("scale_filler"));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static Normalization2Parameter FromProto(RawProto rp)
        {
            string strVal;
            Normalization2Parameter p = new Normalization2Parameter();

            if ((strVal = rp.FindValue("across_spatial")) != null)
                p.across_spatial = bool.Parse(strVal);

            if ((strVal = rp.FindValue("channel_shared")) != null)
                p.channel_shared = bool.Parse(strVal);

            if ((strVal = rp.FindValue("eps")) != null)
                p.eps = ParseFloat(strVal);

            RawProto rgScaleFiller = rp.FindChild("scale_filler");
            if (rgScaleFiller != null)
                p.scale_filler = FillerParameter.FromProto(rgScaleFiller);

            return p;
        }
    }
}
