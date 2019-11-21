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
    /// @see [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549) by Luca Bertinetto, Jack Valmadre, João F. Henriques, Andrea Vedaldi, and Philip H. S. Torr, 2016.
    /// @see [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) by Raia Hadsel, Sumit Chopra, and Yann LeCun, 2006.
    /// </remarks>
    public class ContrastiveLossParameter : LayerParameterBase
    {
        double m_dfMargin = 1.0;
        bool m_bLegacyVersion = false;
        bool m_bOutputMatches = false;

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

            return p;
        }
    }
}
