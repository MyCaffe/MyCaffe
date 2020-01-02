using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the ContrastiveLossLayer.
    /// </summary>
    /// <remarks>
    /// @see [Object cosegmentation using deep Siamese network](https://arxiv.org/pdf/1803.02555.pdf) by Prerana Mukherjee, Brejesh Lall and Snehith Lattupally, 2018.
    /// @see [Learning Deep Representations of Medical Images using Siamese CNNs with Application to Content-Based Image Retrieval](https://arxiv.org/abs/1711.08490) by Yu-An Chung and Wei-Hung Weng, 2017.
    /// @see [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549) by Luca Bertinetto, Jack Valmadre, João F. Henriques, Andrea Vedaldi, and Philip H. S. Torr, 2016.
    /// @see [Learning visual similarity for product design with convolutional neural networks](https://www.cs.cornell.edu/~kb/publications/SIG15ProductNet.pdf) by Sean Bell and Kavita Bala, Cornell University, 2015. 
    /// @see [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) by Raia Hadsel, Sumit Chopra, and Yann LeCun, 2006.
    /// @see [Similarity Learning with (or without) Convolutional Neural Network](http://slazebni.cs.illinois.edu/spring17/lec09_similarity.pdf) by Moitreya Chatterjee and Yunan Luo, 2017. 
    /// Centroids:
    /// @see [A New Loss Function for CNN Classifier Based on Pre-defined Evenly-Distributed Class Centroids](https://arxiv.org/abs/1904.06008) by Qiuyu Zhu, Pengju Zhang, and Xin Ye, arXiv:1904.06008, 2019.
    /// </remarks>
    public class ContrastiveLossParameter : LayerParameterBase
    {
        double m_dfMargin = 1.0;
        bool m_bLegacyVersion = false;
        bool m_bOutputMatches = false;
        bool m_bEnableCentroidLearning = false;

        /** @copydoc LayerParameterBase */
        public ContrastiveLossParameter()
        {
        }

        /// <summary>
        /// Margin for dissimilar pair.
        /// </summary>
        [Description("Specifies the margin for dissimilar pair.")]
        public double margin
        {
            get { return m_dfMargin; }
            set { m_dfMargin = value; }
        }

        /// <summary>
        /// The first implementation of this cost did not exactly match the cost of
        /// Hadsell et al 2006 -- using (margin - d^2) instead of (margin - d)^2.
        /// 
        /// legacy_version = false (the default) uses (margin - d)^2 as proposed in the
        /// Hadsell paper.  New models should probably use this version.
        /// 
        /// legacy_version = true uses (margin - d^2).  This is kept to support /
        /// repoduce existing models and results.
        /// </summary>
        [Description("Specifies to use the legacy version or not.  When true the legacy version '(margin - d^2)' is used.  Otherwise the default is to use the version '(margin - d)^2' proposed in the Hadsell paper.")]
        //[ReadOnly(true)] // currently legacy version causes bug in auto test on backwards
        public bool legacy_version
        {
            get { return m_bLegacyVersion; }
            set { m_bLegacyVersion = value; }
        }

        /// <summary>
        /// Optionally, specifies to output match information (default = false).
        /// </summary>
        [Description("Optionally, specifies to output match information (default = false).")]
        public bool output_matches
        {
            get { return m_bOutputMatches; }
            set { m_bOutputMatches = value; }
        }

        /// <summary>
        /// Optionally, specifies to use centroid learning as soon as the centroids (from the DecodeLayer) are ready - e.g. they have an asum > 0 (default = false, meaning no centroid learning occurs).
        /// </summary>
        [Description("Optionally, specifies to use centroid learning as soon as the centroids (from the DecodeLayer) are ready - e.g. they have an asum > 0 (default = false, meaning no centroid learning occurs).")]
        public bool enable_centroid_learning
        {
            get { return m_bEnableCentroidLearning; }
            set { m_bEnableCentroidLearning = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ContrastiveLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            ContrastiveLossParameter p = (ContrastiveLossParameter)src;
            m_dfMargin = p.m_dfMargin;
            m_bLegacyVersion = p.m_bLegacyVersion;
            m_bOutputMatches = p.m_bOutputMatches;
            m_bEnableCentroidLearning = p.m_bEnableCentroidLearning;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ContrastiveLossParameter p = new ContrastiveLossParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (margin != 1.0)
                rgChildren.Add("margin", margin.ToString());

            if (legacy_version != false)
                rgChildren.Add("legacy_version", legacy_version.ToString());

            if (output_matches)
                rgChildren.Add("output_matches", output_matches.ToString());

            if (enable_centroid_learning)
                rgChildren.Add("enable_centroid_learning", enable_centroid_learning.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ContrastiveLossParameter FromProto(RawProto rp)
        {
            string strVal;
            ContrastiveLossParameter p = new ContrastiveLossParameter();

            if ((strVal = rp.FindValue("margin")) != null)
                p.margin = double.Parse(strVal);

            if ((strVal = rp.FindValue("legacy_version")) != null)
                p.legacy_version = bool.Parse(strVal);

            if ((strVal = rp.FindValue("output_matches")) != null)
                p.output_matches = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_centroid_learning")) != null)
                p.enable_centroid_learning = bool.Parse(strVal);

            return p;
        }
    }
}
