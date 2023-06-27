using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using MyCaffe.basecode;
using MyCaffe.common;
using System.ComponentModel;

namespace MyCaffe.param.ssd
{
    /// <summary>
    /// Specifies a sample of batch of bboxes with provided constraints in SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class BatchSampler : BaseParameter, ICloneable, IComparable, IBinaryPersist
    {
        bool m_bUseOriginalImage = true;
        Sampler m_sampler = new Sampler();
        SamplerConstraint m_constraint = new SamplerConstraint();
        uint m_nMaxSample = 0;
        uint m_nMaxTrials = 100;

        /// <summary>
        /// The BatchSampler constructor.
        /// </summary>
        public BatchSampler()
        {
        }

        /// <summary>
        /// Save the BatchSampler to a binary writer.
        /// </summary>
        /// <param name="bw">The binary writer to use.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write(m_bUseOriginalImage);
            bw.Write(m_nMaxSample);
            bw.Write(m_nMaxTrials);
            m_sampler.Save(bw);
            m_constraint.Save(bw);
        }

        /// <summary>
        /// Load the BatchSampler from a binary reader.
        /// </summary>
        /// <param name="br">The binary reader to use.</param>
        /// <param name="bNewInstance">When <i>true</i>, a the BatchSampler is read into a new instance, otherwise it is read into the current instance.</param>
        /// <returns>The BatchSampler instance is returned.</returns>
        public object Load(BinaryReader br, bool bNewInstance)
        {
            BatchSampler b = this;
            
            if (bNewInstance)
                b = new BatchSampler();

            m_bUseOriginalImage = br.ReadBoolean();
            m_nMaxSample = br.ReadUInt32();
            m_nMaxTrials = br.ReadUInt32();
            m_sampler = Sampler.Load(br);
            m_constraint = SamplerConstraint.Load(br);

            return b;
        }

        /// <summary>
        /// Load the BatchSampler from a binary reader.
        /// </summary>
        /// <param name="br">The binary reader to use.</param>
        /// <returns>A new BatchSampler instance is returned.</returns>
        public static BatchSampler Load(BinaryReader br)
        {
            BatchSampler b = new BatchSampler();
            return (BatchSampler)b.Load(br, true);
        }

        /// <summary>
        /// Use the original image as the source for sampling.
        /// </summary>
        public bool use_original_image
        {
            get { return m_bUseOriginalImage; }
            set { m_bUseOriginalImage = value; }
        }

        /// <summary>
        /// If provided (greater than zero), break when found certain number of samples satisfying the sample constraint.
        /// </summary>
        public uint max_sample
        {
            get { return m_nMaxSample; }
            set { m_nMaxSample = value; }
        }

        /// <summary>
        /// Maximum number of trials for sampling to avoid an infinite loop.
        /// </summary>
        public uint max_trials
        {
            get { return m_nMaxTrials; }
            set { m_nMaxTrials = value; }
        }

        /// <summary>
        /// Specifies the constraints for sampling the bbox
        /// </summary>
        public Sampler sampler
        {
            get { return m_sampler; }
            set { m_sampler = value; }
        }

        /// <summary>
        /// Get/set the sample constraint.
        /// </summary>
        public SamplerConstraint sample_constraint
        {
            get { return m_constraint; }
            set { m_constraint = value; }
        }

        /// <summary>
        /// Creates a copy of the BatchSampler.
        /// </summary>
        /// <returns>A new instance of the BatchSampler is returned.</returns>
        public BatchSampler Clone()
        {
            BatchSampler bs = new BatchSampler();

            bs.m_bUseOriginalImage = m_bUseOriginalImage;
            bs.m_nMaxSample = m_nMaxSample;
            bs.m_nMaxTrials = m_nMaxTrials;
            bs.m_sampler = m_sampler.Clone();
            bs.m_constraint = m_constraint.Clone();

            return bs;
        }

        /// <summary>
        /// Creates a copy of the BatchSampler.
        /// </summary>
        /// <returns>A new instance of the BatchSampler is returned.</returns>
        object ICloneable.Clone()
        {
            return Clone();
        }

        /// <summary>
        /// Compares this BatchSampler to another.
        /// </summary>
        /// <param name="bs">Specifies the other BatchSampler to compare this one to.</param>
        /// <returns>If the two BatchSampler's are the same <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Compare(BatchSampler bs)
        {
            if (bs.m_bUseOriginalImage != m_bUseOriginalImage)
                return false;

            if (bs.m_nMaxSample != m_nMaxSample)
                return false;

            if (bs.m_nMaxTrials != m_nMaxTrials)
                return false;

            if (!bs.m_sampler.Compare(m_sampler))
                return false;

            if (!bs.m_constraint.Compare(m_constraint))
                return true;

            return true;
        }

        /// <summary>
        /// Compares this BatchSampler to another.
        /// </summary>
        /// <param name="obj">Specifies the other BatchSampler to compare this one to.</param>
        /// <returns>If the two BatchSampler's are the same <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public int CompareTo(object obj)
        {
            BatchSampler bs = obj as BatchSampler;

            if (bs == null)
                return 1;

            if (!Compare(bs))
                return 1;

            return 0;
        }

        /// <summary>
        /// Converts the BatchSampler to a RawProto.
        /// </summary>
        /// <param name="strName">Specifies a name for the RawProto.</param>
        /// <returns>A new RawProto representing the BatchSampler is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(new RawProto("use_original_image", m_bUseOriginalImage.ToString()));
            rgChildren.Add(new RawProto("max_sample", m_nMaxSample.ToString()));
            rgChildren.Add(new RawProto("max_trials", m_nMaxTrials.ToString()));

            rgChildren.Add(m_sampler.ToProto("sampler"));
            rgChildren.Add(m_constraint.ToProto("sample_constraint"));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parse a new BatchSampler from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto containing a representation of the BatchSampler.</param>
        /// <returns>A new instance of the BatchSampler is returned.</returns>
        public static BatchSampler FromProto(RawProto rp)
        {
            string strVal;
            BatchSampler p = new BatchSampler();

            if ((strVal = rp.FindValue("use_original_image")) != null)
                p.use_original_image = bool.Parse(strVal);

            if ((strVal = rp.FindValue("max_sample")) != null)
                p.max_sample = uint.Parse(strVal);

            if ((strVal = rp.FindValue("max_trials")) != null)
                p.max_trials = uint.Parse(strVal);

            RawProto protoSampler = rp.FindChild("sampler");
            if (protoSampler != null)
                p.sampler = Sampler.FromProto(protoSampler);

            RawProto protoConstraint = rp.FindChild("sample_constraint");
            if (protoConstraint != null)
                p.sample_constraint = SamplerConstraint.FromProto(protoConstraint);

            return p;
        }

        /// <summary>
        /// Return the string representation of the shape.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            string strOut = "";

            strOut += "use_original_image = " + use_original_image.ToString() + Environment.NewLine;
            strOut += "max_sample = " + max_sample.ToString() + Environment.NewLine;
            strOut += "max_trials = " + max_trials.ToString() + Environment.NewLine;
            strOut += m_sampler.ToString();
            strOut += m_constraint.ToString();

            return strOut;
        }
    }
}
