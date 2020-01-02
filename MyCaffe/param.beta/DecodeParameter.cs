using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

/// <summary>
/// The MyCaffe.param.beta parameters are used by the MyCaffe.layer.beta layers.
/// </summary>
/// <remarks>
/// Using parameters within the MyCaffe.layer.beta namespace are used by layers that require the MyCaffe.layers.beta.dll.
/// Centroids:
/// @see [A New Loss Function for CNN Classifier Based on Pre-defined Evenly-Distributed Class Centroids](https://arxiv.org/abs/1904.06008) by Qiuyu Zhu, Pengju Zhang, and Xin Ye, arXiv:1904.06008, 2019.
/// </remarks>
namespace MyCaffe.param.beta
{
    /// <summary>
    /// Specifies the parameters for the DecodeLayer and the AccuracyEncodingLayer.
    /// </summary>
    public class DecodeParameter : LayerParameterBase 
    {
        int m_nCentroidThresholdStart = 300;
        int m_nCentroidThresholdEnd = 500;
        bool m_bOutputCentroids = false;
        int m_nActiveLabelCount = 0;

        /** @copydoc LayerParameterBase */
        public DecodeParameter()
        {
        }

        /// <summary>
        /// Specifies the starting iteration where observed items are used to calculate the centroid for each label, before this value, the centroids should not be used for their calculation is not complete (default = 300).
        /// </summary>
        [Description("Specifies the starting iteration where observed items are used to calculate the centroid for each label, before this value, the centroids are set to 0 and should not be used (default = 300).")]
        public int centroid_threshold_start
        {
            get { return m_nCentroidThresholdStart; }
            set { m_nCentroidThresholdStart = value; }
        }

        /// <summary>
        /// Specifies the ending iteration where observed items are used to calculate the centroid for each label, after this value, the previously calculated centroids returned (default = 500).
        /// </summary>
        [Description("Specifies the ending iteration where observed items are used to calculate the centroid for each label, after this value, the previously calculated centroids returned (default = 500).")]
        public int centroid_threshold_end
        {
            get { return m_nCentroidThresholdEnd; }
            set { m_nCentroidThresholdEnd = value; }
        }

        /// <summary>
        /// Optionally, specifies to output the centroids in top[1] (default = false).
        /// </summary>
        [Description("Optionally, specifies to output the centroids in top[1] (default = false).")]
        public bool output_centroids
        {
            get { return m_bOutputCentroids; }
            set { m_bOutputCentroids = value; }
        }

        /// <summary>
        /// Optionally, specifies a number of active labels that are less than the actual label count - this is used when only a subset of the labels within the label range are actually used (default = 0, which then expects all labels).
        /// </summary>
        [Description("Optionally, specifies a number of active labels that are less than the actual label count - this is used when only a subset of the labels within the label range are actually used (default = 0, which then expects all labels).")]
        public int active_label_count
        {
            get { return m_nActiveLabelCount; }
            set { m_nActiveLabelCount = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            DecodeParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            DecodeParameter p = (DecodeParameter)src;
            m_nCentroidThresholdStart = p.m_nCentroidThresholdStart;
            m_nCentroidThresholdEnd = p.m_nCentroidThresholdEnd;
            m_bOutputCentroids = p.m_bOutputCentroids;
            m_nActiveLabelCount = p.m_nActiveLabelCount;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            DecodeParameter p = new DecodeParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("centroid_threshold_start", centroid_threshold_start.ToString());
            rgChildren.Add("centroid_threshold_end", centroid_threshold_end.ToString());
            rgChildren.Add("output_centroids", output_centroids.ToString());

            if (active_label_count > 0)
                rgChildren.Add("active_label_count", active_label_count.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static DecodeParameter FromProto(RawProto rp)
        {
            string strVal;
            DecodeParameter p = new DecodeParameter();

            if ((strVal = rp.FindValue("centroid_threshold_start")) != null)
                p.centroid_threshold_start = int.Parse(strVal);

            if ((strVal = rp.FindValue("centroid_threshold_end")) != null)
                p.centroid_threshold_end = int.Parse(strVal);

            if ((strVal = rp.FindValue("output_centroids")) != null)
                p.output_centroids = bool.Parse(strVal);

            if ((strVal = rp.FindValue("active_label_count")) != null)
                p.active_label_count = int.Parse(strVal);

            return p;
        }
    }
}
