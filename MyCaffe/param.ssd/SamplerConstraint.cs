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
    /// Specifies the constratins for selecting sampled bbox used in SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class SamplerConstraint : BaseParameter, ICloneable, IComparable, IBinaryPersist
    {
        float? m_fMinJaccardOverlap = null;
        float? m_fMaxJaccardOverlap = null;
        float? m_fMinSampleCoverage = null;
        float? m_fMaxSampleCoverage = null;
        float? m_fMinObjectCoverage = null;
        float? m_fMaxObjectCoverage = null;

        /// <summary>
        /// The SampleConstraint constructor.
        /// </summary>
        public SamplerConstraint()
        {
        }

        private void save(BinaryWriter bw, float? f)
        {
            bw.Write(f.HasValue);
            if (f.HasValue)
                bw.Write(f.Value);
        }

        private float? load(BinaryReader br)
        {
            if (br.ReadBoolean())
                return br.ReadSingle();
            else
                return null;
        }

        /// <summary>
        /// Save the SampleConstraint to a binary writer.
        /// </summary>
        /// <param name="bw">The binary writer to use.</param>
        public void Save(BinaryWriter bw)
        {
            save(bw, m_fMinJaccardOverlap);
            save(bw, m_fMaxJaccardOverlap);
            save(bw, m_fMinSampleCoverage);
            save(bw, m_fMaxSampleCoverage);
            save(bw, m_fMinObjectCoverage);
            save(bw, m_fMaxSampleCoverage);
        }

        /// <summary>
        /// Load the SampleConstraint from a binary reader.
        /// </summary>
        /// <param name="br">The binary reader to use.</param>
        /// <param name="bNewInstance">When <i>true</i>, a the SampleConstraint is read into a new instance, otherwise it is read into the current instance.</param>
        /// <returns>The SampleConstraint instance is returned.</returns>
        public object Load(BinaryReader br, bool bNewInstance)
        {
            SamplerConstraint b = this;
            
            if (bNewInstance)
                b = new SamplerConstraint();

            b.m_fMinJaccardOverlap = load(br);
            b.m_fMaxJaccardOverlap = load(br);
            b.m_fMinSampleCoverage = load(br);
            b.m_fMaxSampleCoverage = load(br);
            b.m_fMinObjectCoverage = load(br);
            b.m_fMaxObjectCoverage = load(br);

            return b;
        }

        /// <summary>
        /// Load the SampleConstraint from a binary reader.
        /// </summary>
        /// <param name="br">The binary reader to use.</param>
        /// <returns>A new SampleConstraint instance is returned.</returns>
        public static SamplerConstraint Load(BinaryReader br)
        {
            SamplerConstraint b = new SamplerConstraint();
            return (SamplerConstraint)b.Load(br, true);
        }

        /// <summary>
        /// Get/set the minimum Jaccard overlap between sampled bbox and all boxes in AnnotationGroup.
        /// </summary>
        public float? min_jaccard_overlap
        {
            get { return m_fMinJaccardOverlap; }
            set { m_fMinJaccardOverlap = value; }
        }

        /// <summary>
        /// Get/set the maximum Jaccard overlap between sampled bbox and all boxes in AnnotationGroup.
        /// </summary>
        public float? max_jaccard_overlap
        {
            get { return m_fMaxJaccardOverlap; }
            set { m_fMaxJaccardOverlap = value; }
        }

        /// <summary>
        /// Get/set the minimum Sample coverage between sampled bbox and all boxes in AnnotationGroup.
        /// </summary>
        public float? min_sample_coverage
        {
            get { return m_fMinSampleCoverage; }
            set { m_fMinSampleCoverage = value; }
        }

        /// <summary>
        /// Get/set the maximum Sample coverage between sampled bbox and all boxes in AnnotationGroup.
        /// </summary>
        public float? max_sample_coverage
        {
            get { return m_fMaxSampleCoverage; }
            set { m_fMaxSampleCoverage = value; }
        }

        /// <summary>
        /// Get/set the minimum Object coverage between sampled bbox and all boxes in AnnotationGroup.
        /// </summary>
        public float? min_object_coverage
        {
            get { return m_fMinObjectCoverage; }
            set { m_fMinObjectCoverage = value; }
        }

        /// <summary>
        /// Get/set the maximum Object coverage between sampled bbox and all boxes in AnnotationGroup.
        /// </summary>
        public float? max_object_coverage
        {
            get { return m_fMaxObjectCoverage; }
            set { m_fMaxObjectCoverage = value; }
        }

        /// <summary>
        /// Creates a copy of the SampleConstraint.
        /// </summary>
        /// <returns>A new instance of the SampleConstraint is returned.</returns>
        public SamplerConstraint Clone()
        {
            SamplerConstraint bs = new SamplerConstraint();

            bs.m_fMinJaccardOverlap = m_fMinJaccardOverlap;
            bs.m_fMaxJaccardOverlap = m_fMaxJaccardOverlap;
            bs.m_fMinSampleCoverage = m_fMinSampleCoverage;
            bs.m_fMaxSampleCoverage = m_fMaxSampleCoverage;
            bs.m_fMinObjectCoverage = m_fMinObjectCoverage;
            bs.m_fMaxObjectCoverage = m_fMaxObjectCoverage;

            return bs;
        }

        /// <summary>
        /// Creates a copy of the SampleConstraint.
        /// </summary>
        /// <returns>A new instance of the SampleConstraint is returned.</returns>
        object ICloneable.Clone()
        {
            return Clone();
        }

        /// <summary>
        /// Compares this SampleConstraint to another.
        /// </summary>
        /// <param name="bs">Specifies the other SampleConstraint to compare this one to.</param>
        /// <returns>If the two SampleConstraint's are the same <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Compare(SamplerConstraint bs)
        {
            if (bs.m_fMinJaccardOverlap != m_fMinJaccardOverlap)
                return false;
            if (bs.m_fMaxJaccardOverlap != m_fMaxJaccardOverlap)
                return false;

            if (bs.m_fMinSampleCoverage != m_fMinSampleCoverage)
                return false;
            if (bs.m_fMaxSampleCoverage != m_fMaxSampleCoverage)
                return false;

            if (bs.m_fMinObjectCoverage != m_fMinObjectCoverage)
                return false;
            if (bs.m_fMaxObjectCoverage != m_fMaxObjectCoverage)
                return false;

            return true;
        }

        /// <summary>
        /// Compares this SampleConstraint to another.
        /// </summary>
        /// <param name="obj">Specifies the other SampleConstraint to compare this one to.</param>
        /// <returns>If the two SampleConstraint's are the same <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public int CompareTo(object obj)
        {
            SamplerConstraint bs = obj as SamplerConstraint;

            if (bs == null)
                return 1;

            if (!Compare(bs))
                return 1;

            return 0;
        }

        /// <summary>
        /// Converts the SampleConstraint to a RawProto.
        /// </summary>
        /// <param name="strName">Specifies a name for the RawProto.</param>
        /// <returns>A new RawProto representing the SampleConstraint is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (m_fMinJaccardOverlap.HasValue)
                rgChildren.Add(new RawProto("min_jaccard_overlap", m_fMinJaccardOverlap.Value.ToString()));
            if (m_fMaxJaccardOverlap.HasValue)
                rgChildren.Add(new RawProto("max_jaccard_overlap", m_fMaxJaccardOverlap.Value.ToString()));

            if (m_fMinSampleCoverage.HasValue)
                rgChildren.Add(new RawProto("min_sample_coverage", m_fMinSampleCoverage.Value.ToString()));
            if (m_fMaxSampleCoverage.HasValue)
                rgChildren.Add(new RawProto("max_sample_coverage", m_fMaxSampleCoverage.Value.ToString()));

            if (m_fMinObjectCoverage.HasValue)
                rgChildren.Add(new RawProto("min_object_coverage", m_fMinObjectCoverage.Value.ToString()));
            if (m_fMaxObjectCoverage.HasValue)
                rgChildren.Add(new RawProto("max_object_coverage", m_fMaxObjectCoverage.Value.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parse a new SampleConstraint from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto containing a representation of the SampleConstraint.</param>
        /// <returns>A new instance of the SampleConstraint is returned.</returns>
        public static SamplerConstraint FromProto(RawProto rp)
        {
            string strVal;
            SamplerConstraint p = new SamplerConstraint();

            if ((strVal = rp.FindValue("min_jaccard_overlap")) != null)
                p.min_jaccard_overlap = BaseParameter.ParseFloat(strVal);
            if ((strVal = rp.FindValue("max_jaccard_overlap")) != null)
                p.max_jaccard_overlap = BaseParameter.ParseFloat(strVal);

            if ((strVal = rp.FindValue("min_sample_coverage")) != null)
                p.min_sample_coverage = BaseParameter.ParseFloat(strVal);
            if ((strVal = rp.FindValue("max_sample_coverage")) != null)
                p.max_sample_coverage = BaseParameter.ParseFloat(strVal);

            if ((strVal = rp.FindValue("min_object_coverage")) != null)
                p.min_object_coverage = BaseParameter.ParseFloat(strVal);
            if ((strVal = rp.FindValue("max_object_coverage")) != null)
                p.max_object_coverage = BaseParameter.ParseFloat(strVal);

            return p;
        }

        /// <summary>
        /// Return the string representation of the shape.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            string strOut = "";

            if (min_jaccard_overlap.HasValue)
                strOut += "min_jaccard_overlap = " + min_jaccard_overlap.Value.ToString() + Environment.NewLine;
            if (max_jaccard_overlap.HasValue)
                strOut += "max_jaccard_overlap = " + max_jaccard_overlap.Value.ToString() + Environment.NewLine;

            if (min_sample_coverage.HasValue)
                strOut += "min_sample_coverage = " + min_sample_coverage.Value.ToString() + Environment.NewLine;
            if (max_sample_coverage.HasValue)
                strOut += "max_sample_coverage = " + max_sample_coverage.Value.ToString() + Environment.NewLine;

            if (min_object_coverage.HasValue)
                strOut += "min_object_coverage = " + min_object_coverage.Value.ToString() + Environment.NewLine;
            if (max_object_coverage.HasValue)
                strOut += "max_object_coverage = " + max_object_coverage.Value.ToString() + Environment.NewLine;

            return strOut;
        }
    }
}
