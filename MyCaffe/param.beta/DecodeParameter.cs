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
/// 
/// Centroids:
/// @see [A New Loss Function for CNN Classifier Based on Pre-defined Evenly-Distributed Class Centroids](https://arxiv.org/abs/1904.06008) by Qiuyu Zhu, Pengju Zhang, and Xin Ye, arXiv:1904.06008, 2019.
/// 
/// KNN:
/// @see [Constellation Loss: Improving the efficiency of deep metric learning loss functions for optimal embedding](https://arxiv.org/abs/1905.10675) by Alfonso Medela and Artzai Picon, arXiv:1905.10675, 2019
/// </remarks>
namespace MyCaffe.param.beta
{
    /// <summary>
    /// Specifies the parameters for the DecodeLayer and the AccuracyEncodingLayer.
    /// </summary>
    public class DecodeParameter : LayerParameterBase 
    {
        int m_nCentroidOutputIteration = 300;
        bool m_bOutputCentroids = false;
        List<int> m_rgIgnoreLabels = new List<int>();
        TARGET m_target = TARGET.CENTROID;
        int m_nCacheSize = 100;
        int m_nK = 5;

        /// <summary>
        /// Defines the target type.
        /// </summary>
        public enum TARGET
        {
            /// <summary>
            /// Specifies to use the centroid as the target.
            /// </summary>
            CENTROID,
            /// <summary>
            /// Specifies to use the k-nearest neighbor as the target.
            /// </summary>
            KNN
        }

        /** @copydoc LayerParameterBase */
        public DecodeParameter()
        {
        }

        /// <summary>
        /// Specifies the iteration where calculated centroids are output for each label, before this value, the centroids should not be used for their calculation is not complete (default = 300).
        /// </summary>
        [Description("Specifies the iteration where calculated centroids are output for each label, before this value, the centroids should not be used for their calculation is not complete (default = 300).")]
        public int centroid_output_iteration
        {
            get { return m_nCentroidOutputIteration; }
            set { m_nCentroidOutputIteration = value; }
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
        /// Optionally, specifies one or more labels to ignore. (default = none, which then expects all labels).
        /// </summary>
        [Description("Optionally, specifies one or more labels to ignore. (default = none, which then expects all labels).")]
        public List<int> ignore_labels
        {
            get { return m_rgIgnoreLabels; }
            set { m_rgIgnoreLabels = value; }
        }

        /// <summary>
        /// Optionally, specifies the target type to use (default = CENTROID).
        /// </summary>
        [Description("Optionally, specifies the target type to use (default = CENTROID).")]
        public TARGET target
        {
            get { return m_target; }
            set { m_target = value; }
        }

        /// <summary>
        /// Specifies the size of the cache (in number of batches) used when calculating the CENTROID and KNN values (default = 300).
        /// </summary>
        [Description("Specifies the size of the cache (in number of batches) used when calculating the CENTROID and KNN values (default = 300).")]
        public int cache_size
        {
            get { return m_nCacheSize; }
            set { m_nCacheSize = value; }
        }

        /// <summary>
        /// Optionally, specifies the K value to use with the KNN target (default = 5).
        /// </summary>
        [Description("Optionally, specifies the K value to use with the KNN target (default = 5).")]
        public int k
        {
            get { return m_nK; }
            set { m_nK = value; }
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
            m_nCentroidOutputIteration = p.m_nCentroidOutputIteration;
            m_nCacheSize = p.m_nCacheSize;
            m_bOutputCentroids = p.m_bOutputCentroids;

            if (p.m_rgIgnoreLabels != null)
                m_rgIgnoreLabels = Utility.Clone<int>(p.m_rgIgnoreLabels);

            m_target = p.m_target;
            m_nK = p.m_nK;
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

            rgChildren.Add("centroid_output_iteration", centroid_output_iteration.ToString());
            rgChildren.Add("cache_size", cache_size.ToString());
            rgChildren.Add("output_centroids", output_centroids.ToString());
            rgChildren.Add("target", target.ToString());

            foreach (int nLabel in ignore_labels)
            {
                rgChildren.Add("ignore_label", nLabel.ToString());
            }

            if (target == TARGET.KNN)
                rgChildren.Add("k", k.ToString());

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

            if ((strVal = rp.FindValue("centroid_output_iteration")) != null)
                p.centroid_output_iteration = int.Parse(strVal);

            if ((strVal = rp.FindValue("cache_size")) != null)
                p.cache_size = int.Parse(strVal);

            if ((strVal = rp.FindValue("output_centroids")) != null)
                p.output_centroids = bool.Parse(strVal);

            p.ignore_labels = new List<int>();
            RawProtoCollection rpIgnore = rp.FindChildren("ignore_label");
            foreach (RawProto rplabel in rpIgnore)
            {
                int nLabel = int.Parse(rplabel.Value);
                if (!p.ignore_labels.Contains(nLabel))
                    p.ignore_labels.Add(nLabel);
            }

            if ((strVal = rp.FindValue("target")) != null)
            {
                if (strVal == TARGET.KNN.ToString())
                    p.target = TARGET.KNN;
            }

            if ((strVal = rp.FindValue("k")) != null)
                p.k = int.Parse(strVal);

            return p;
        }
    }
}
