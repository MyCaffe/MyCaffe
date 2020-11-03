using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param.ssd
{
    /// <summary>
    /// Specifies the sample of a bbox in the normalized space [0,1] with provided constraints used in SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class Sampler : BaseParameter, ICloneable, IComparable, IBinaryPersist
    {
        float m_fMinScale = 1.0f;
        float m_fMaxScale = 1.0f;
        float m_fMinAspectRatio = 1.0f;
        float m_fMaxAspectRatio = 1.0f;

        /// <summary>
        /// The Sample constructor.
        /// </summary>
        public Sampler()
        {
        }

        /// <summary>
        /// Save the Sample to a binary writer.
        /// </summary>
        /// <param name="bw">The binary writer to use.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write(m_fMinScale);
            bw.Write(m_fMaxScale);
            bw.Write(m_fMinAspectRatio);
            bw.Write(m_fMaxAspectRatio);
        }

        /// <summary>
        /// Load the Sample from a binary reader.
        /// </summary>
        /// <param name="br">The binary reader to use.</param>
        /// <param name="bNewInstance">When <i>true</i>, a the Sample is read into a new instance, otherwise it is read into the current instance.</param>
        /// <returns>The Sample instance is returned.</returns>
        public object Load(BinaryReader br, bool bNewInstance)
        {
            Sampler b = this;
            
            if (bNewInstance)
                b = new Sampler();

            b.m_fMinScale = br.ReadSingle();
            b.m_fMaxScale = br.ReadSingle();
            b.m_fMinAspectRatio = br.ReadSingle();
            b.m_fMaxAspectRatio = br.ReadSingle();

            return b;
        }

        /// <summary>
        /// Load the Sample from a binary reader.
        /// </summary>
        /// <param name="br">The binary reader to use.</param>
        /// <returns>A new Sample instance is returned.</returns>
        public static Sampler Load(BinaryReader br)
        {
            Sampler b = new Sampler();
            return (Sampler)b.Load(br, true);
        }

        /// <summary>
        /// Get/set the minimum scale of the sampled bbox.
        /// </summary>
        public float min_scale
        {
            get { return m_fMinScale; }
            set { m_fMinScale = value; }
        }

        /// <summary>
        /// Get/set the maximum scale of the sampled bbox.
        /// </summary>
        public float max_scale
        {
            get { return m_fMaxScale; }
            set { m_fMaxScale = value; }
        }

        /// <summary>
        /// Get/set the minimum aspect ratio of the sampled bbox.
        /// </summary>
        public float min_aspect_ratio
        {
            get { return m_fMinAspectRatio; }
            set { m_fMinAspectRatio = value; }
        }

        /// <summary>
        /// Get/set the maximum aspect ratio of the sampled bbox.
        /// </summary>
        public float max_aspect_ratio
        {
            get { return m_fMaxAspectRatio; }
            set { m_fMaxAspectRatio = value; }
        }

        /// <summary>
        /// Creates a copy of the Sample.
        /// </summary>
        /// <returns>A new instance of the Sample is returned.</returns>
        public Sampler Clone()
        {
            Sampler bs = new Sampler();

            bs.m_fMinScale = m_fMinScale;
            bs.m_fMaxScale = m_fMaxScale;
            bs.m_fMinAspectRatio = m_fMinAspectRatio;
            bs.m_fMaxAspectRatio = m_fMaxAspectRatio;

            return bs;
        }

        /// <summary>
        /// Creates a copy of the Sample.
        /// </summary>
        /// <returns>A new instance of the Sample is returned.</returns>
        object ICloneable.Clone()
        {
            return Clone();
        }

        /// <summary>
        /// Compares this Sample to another.
        /// </summary>
        /// <param name="bs">Specifies the other Sample to compare this one to.</param>
        /// <returns>If the two Sample's are the same <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Compare(Sampler bs)
        {
            if (bs.m_fMinScale != m_fMinScale)
                return false;
            if (bs.m_fMaxScale != m_fMaxScale)
                return false;

            if (bs.m_fMinAspectRatio != m_fMinAspectRatio)
                return false;
            if (bs.m_fMaxAspectRatio != m_fMaxAspectRatio)
                return false;

            return true;
        }

        /// <summary>
        /// Compares this Sample to another.
        /// </summary>
        /// <param name="obj">Specifies the other Sample to compare this one to.</param>
        /// <returns>If the two Sample's are the same <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public int CompareTo(object obj)
        {
            Sampler bs = obj as Sampler;

            if (bs == null)
                return 1;

            if (!Compare(bs))
                return 1;

            return 0;
        }

        /// <summary>
        /// Converts the Sample to a RawProto.
        /// </summary>
        /// <param name="strName">Specifies a name for the RawProto.</param>
        /// <returns>A new RawProto representing the Sample is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(new RawProto("min_scale", m_fMinScale.ToString()));
            rgChildren.Add(new RawProto("max_scale", m_fMaxScale.ToString()));
            rgChildren.Add(new RawProto("min_aspect_ratio", m_fMinAspectRatio.ToString()));
            rgChildren.Add(new RawProto("max_aspect_ratio", m_fMaxAspectRatio.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parse a new Sample from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto containing a representation of the Sample.</param>
        /// <returns>A new instance of the Sample is returned.</returns>
        public static Sampler FromProto(RawProto rp)
        {
            string strVal;
            Sampler p = new Sampler();

            if ((strVal = rp.FindValue("min_scale")) != null)
                p.min_scale = BaseParameter.parseFloat(strVal);
            if ((strVal = rp.FindValue("max_scale")) != null)
                p.max_scale = BaseParameter.parseFloat(strVal);

            if ((strVal = rp.FindValue("min_aspect_ratio")) != null)
                p.min_aspect_ratio = BaseParameter.parseFloat(strVal);
            if ((strVal = rp.FindValue("max_aspect_ratio")) != null)
                p.max_aspect_ratio = BaseParameter.parseFloat(strVal);

            return p;
        }

        /// <summary>
        /// Return the string representation of the shape.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            string strOut = "";

            strOut += "min_scale = " + min_scale.ToString() + Environment.NewLine;
            strOut += "max_scale = " + max_scale.ToString() + Environment.NewLine;
            strOut += "min_aspect_ratio = " + min_aspect_ratio.ToString() + Environment.NewLine;
            strOut += "max_aspect_ratio = " + max_aspect_ratio.ToString() + Environment.NewLine;

            return strOut;
        }
    }
}
