using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;

namespace MyCaffe.param.ssd
{
    /// <summary>
    /// Specifies the parameters for the MultiBoxLossParameter.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// @see [Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/abs/1604.03540) by Abhinav Shrivastava, Abhinav Gupta, Ross Girshick, 2016.
    /// </remarks>
    public class MultiBoxLossParameter : LayerParameterBase
    {
        LocLossType m_locLossType = LocLossType.SMOOTH_L1;
        ConfLossType m_confLossType = ConfLossType.SOFTMAX;
        float m_fLocWeight = 1.0f;
        uint m_nNumClasses;
        bool m_bShareLocation = true;
        MatchType m_matchType = MatchType.PER_PREDICTION;
        float m_fOverlapThreshold = 0.5f;
        uint m_nBackgroundLabelId = 0;
        bool m_bUseDifficultGt = true;
        bool? m_bDoNegMining = null;
        float m_fNegPosRatio = 3.0f;
        float m_fNegOverlap = 0.5f;
        PriorBoxParameter.CodeType m_codeType = PriorBoxParameter.CodeType.CORNER;
        bool m_bEncodeVarianceInTarget = false;
        bool m_bMapObjectToAgnostic = false;
        bool m_bIgnoreCrossBoundaryBbox = false;
        bool m_bBpInside = false;
        MiningType m_miningType = MiningType.MAX_NEGATIVE;
        NonMaximumSuppressionParameter m_nmsParam = new NonMaximumSuppressionParameter(true);
        int m_nSampleSize = 64;
        bool m_bUsePriorForNms = false;
        bool m_bUsePriorForMatching = true;
        bool m_bUseGpu = false;

        /// <summary>
        /// Defines the localization loss types.
        /// </summary>
        public enum LocLossType
        {
            /// <summary>
            /// Specifies to use L2 loss.
            /// </summary>
            L2,
            /// <summary>
            /// Specifies to use smooth L1 loss.
            /// </summary>
            SMOOTH_L1
        }

        /// <summary>
        /// Convert a string into a LocLossType.
        /// </summary>
        /// <param name="str">Specifies the string.</param>
        /// <returns>The associated LocLossType is returned.</returns>
        public static LocLossType LocLossTypeFromString(string str)
        {
            switch (str)
            {
                case "L2":
                    return LocLossType.L2;

                case "SMOOTH_L1":
                    return LocLossType.SMOOTH_L1;

                default:
                    throw new Exception("Unknown LocLossType '" + str + "'!");
            }
        }

        /// <summary>
        /// Defines the confidence loss types.
        /// </summary>
        public enum ConfLossType
        {
            /// <summary>
            /// Specifies to use softmax.
            /// </summary>
            SOFTMAX,
            /// <summary>
            /// Specifies to use logistic.
            /// </summary>
            LOGISTIC
        }

        /// <summary>
        /// Convert a string into a ConfLossType.
        /// </summary>
        /// <param name="str">Specifies the string.</param>
        /// <returns>The associated ConfLossType is returned.</returns>
        public static ConfLossType ConfLossTypeFromString(string str)
        {
            switch (str)
            {
                case "SOFTMAX":
                    return ConfLossType.SOFTMAX;

                case "LOGISTIC":
                    return ConfLossType.LOGISTIC;

                default:
                    throw new Exception("Unknown ConfLossType '" + str + "'!");
            }
        }

        /// <summary>
        /// Defines the matching method used during training.
        /// </summary>
        public enum MatchType
        {
            /// <summary>
            /// Specifies to use Bi-Partite.
            /// </summary>
            BIPARTITE,
            /// <summary>
            /// Specifies to use per-prediction matching.
            /// </summary>
            PER_PREDICTION
        }

        /// <summary>
        /// Convert a string into a MatchType.
        /// </summary>
        /// <param name="str">Specifies the string.</param>
        /// <returns>The associated MatchType is returned.</returns>
        public static MatchType MatchTypeFromString(string str)
        {
            switch (str)
            {
                case "BIPARTITE":
                    return MatchType.BIPARTITE;

                case "PER_PREDICTION":
                    return MatchType.PER_PREDICTION;

                default:
                    throw new Exception("Unknown MatchType '" + str + "'!");
            }
        }


        /// <summary>
        /// Defines the mining type used during training.
        /// </summary>
        public enum MiningType
        {
            /// <summary>
            /// Use all negatives.
            /// </summary>
            NONE,
            /// <summary>
            /// Select negatives based on the score.
            /// </summary>
            MAX_NEGATIVE,
            /// <summary>
            /// Select hard examples based on Shrivastava et. al. method.
            /// </summary>
            /// <remarks>
            /// @see [Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/abs/1604.03540) by Abhinav Shrivastava, Abhinav Gupta, Ross Girshick, 2016.
            /// </remarks>
            HARD_EXAMPLE
        }

        /// <summary>
        /// Convert a string into a MiningType.
        /// </summary>
        /// <param name="str">Specifies the string.</param>
        /// <returns>The associated MiningType is returned.</returns>
        public static MiningType MiningTypeFromString(string str)
        {
            switch (str)
            {
                case "NONE":
                    return MiningType.NONE;

                case "MAX_NEGATIVE":
                    return MiningType.MAX_NEGATIVE;

                case "HARD_EXAMPLE":
                    return MiningType.HARD_EXAMPLE;

                default:
                    throw new Exception("Unknown MiningType '" + str + "'!");
            }
        }


        /// <summary>
        /// The constructor.
        /// </summary>
        public MultiBoxLossParameter()
        {
        }

        /// <summary>
        /// Get/set the localization loss type (default = SMOOTH_L1).
        /// </summary>
        [Description("Get/set the localization loss type (default = SMOOTH_L1).")]
        public LocLossType loc_loss_type
        {
            get { return m_locLossType; }
            set { m_locLossType = value; }
        }

        /// <summary>
        /// Get/set the confidence loss type (default = SOFTMAX).
        /// </summary>
        [Description("Get/set the confidence loss type (default = SOFTMAX).")]
        public ConfLossType conf_loss_type
        {
            get { return m_confLossType; }
            set { m_confLossType = value; }
        }

        /// <summary>
        /// Get/set the weight for the localization loss (default = 1.0).
        /// </summary>
        [Description("Get/set the weight for the localization loss (default = 1.0).")]
        public float loc_weight
        {
            get { return m_fLocWeight; }
            set { m_fLocWeight = value; }
        }

        /// <summary>
        /// Get/set the number of classes to be predicted - required!
        /// </summary>
        [Description("Get/set the number of classes to be predicted - required")]
        public uint num_classes
        {
            get { return m_nNumClasses; }
            set { m_nNumClasses = value; }
        }

        /// <summary>
        /// Get/sets whether or not the bounding box is shared among different classes (default = true).
        /// </summary>
        [Description("Get/sets whether or not the bounding box is shared among different classes (default = true).")]
        public bool share_location
        {
            get { return m_bShareLocation; }
            set { m_bShareLocation = value; }
        }

        /// <summary>
        /// Get/set the matching method used during training (default = PER_PREDICTION).
        /// </summary>
        [Description("Get/set the matching method used during training (default = PER_PREDICTION).")]
        public MatchType match_type
        {
            get { return m_matchType; }
            set { m_matchType = value; }
        }

        /// <summary>
        /// Get/set the overlap threshold (default = 0.5).
        /// </summary>
        [Description("Get/set the overlap threshold (default = 0.5).")]
        public float overlap_threshold
        {
            get { return m_fOverlapThreshold; }
            set { m_fOverlapThreshold = value; }
        }

        /// <summary>
        /// Get/set the background label id.
        /// </summary>
        [Description("Get/set the background label id.")]
        public uint background_label_id
        {
            get { return m_nBackgroundLabelId; }
            set { m_nBackgroundLabelId = value; }
        }

        /// <summary>
        /// Get/set whether or not to consider the difficult ground truth (defalt = true).
        /// </summary>
        [Description("Get/set whether or not to consider the difficult ground truth (defalt = true).")]
        public bool use_difficult_gt
        {
            get { return m_bUseDifficultGt; }
            set { m_bUseDifficultGt = value; }
        }

        /// <summary>
        /// DEPRECIATED: Get/set whether or not to perform negative mining (default = false).
        /// </summary>
        /// <remarks>
        /// DEPRECIATED: using 'mining_type' instead.
        /// </remarks>
        [Description("DEPRECIATED: Get/set whether or not to perform negative mining (default = false).")]
        public bool? do_neg_mining
        {
            get { return m_bDoNegMining; }
            set { m_bDoNegMining = value; }
        }

        /// <summary>
        /// Get/set the negative/positive ratio (default = 3.0).
        /// </summary>
        [Description("Get/set the negative/positive ratio (default = 3.0).")]
        public float neg_pos_ratio
        {
            get { return m_fNegPosRatio; }
            set { m_fNegPosRatio = value; }
        }

        /// <summary>
        /// Get/set the negative overlap upperbound for the unmatched predictions (default = 0.5).
        /// </summary>
        [Description("Get/set the negative overlap upperbound for the unmatched predictions (default = 0.5).")]
        public float neg_overlap
        {
            get { return m_fNegOverlap; }
            set { m_fNegOverlap = value; }
        }

        /// <summary>
        /// Get/set the coding method for the bounding box.
        /// </summary>
        [Description("Get/set the coding method for the bounding box.")]
        public PriorBoxParameter.CodeType code_type
        {
            get { return m_codeType; }
            set { m_codeType = value; }
        }

        /// <summary>
        /// Get/set whether or not to encode the variance of the prior box in the loc loss target instead of in the bbox (default = false).
        /// </summary>
        [Description("Get/set whether or not to encode the variance of the prior box in the loc loss target instead of in the bbox (default = false).")]
        public bool encode_variance_in_target
        {
            get { return m_bEncodeVarianceInTarget; }
            set { m_bEncodeVarianceInTarget = value; }
        }

        /// <summary>
        /// Get/set whether or not to map all object classes to an agnostic class (default = false).  This is useful when learning objectness detector.
        /// </summary>
        [Description("Get/set whether or not to map all object classes to an agnostic class (default = false).  This is useful when learning objectness detector.")]
        public bool map_object_to_agnostic
        {
            get { return m_bMapObjectToAgnostic; }
            set { m_bMapObjectToAgnostic = value; }
        }

        /// <summary>
        /// Get/set whether or not to ignore cross boundary bbox during matching (default = false).  The cross boundary bbox is a bbox who is outside
        /// of the image region.
        /// </summary>
        [Description("Get/set whether or not to ignore cross boundary bbox during matching (default = false).  The cross boundary bbox is a bbox who is outside of the image region.")]
        public bool ignore_cross_boundary_bbox
        {
            get { return m_bIgnoreCrossBoundaryBbox; }
            set { m_bIgnoreCrossBoundaryBbox = value; }
        }

        /// <summary>
        /// Get/set whether or not to only backpropagate on corners which are inside of the image region when encode type is CORNER or CORNER_SIZE (default = false).
        /// </summary>
        [Description("Get/set whether or not to only backpropagate on corners which are inside of the image region when encode type is CORNER or CORNER_SIZE (default = false).")]
        public bool bp_inside
        {
            get { return m_bBpInside; }
            set { m_bBpInside = value; }
        }

        /// <summary>
        /// Get/set the mining type used during training (default = MAX_NEGATIVE).
        /// </summary>
        [Description("Get/set the mining type used during training (default = MAX_NEGATIVE).")]
        public MiningType mining_type
        {
            get { return m_miningType; }
            set { m_miningType = value; }
        }

        /// <summary>
        /// Get/set the parameters used for the non maximum suppression during hard example training.
        /// </summary>
        [Description("Get/set the parameters used for the non maximum suppression during hard example training.")]
        public NonMaximumSuppressionParameter nms_param
        {
            get { return m_nmsParam; }
            set { m_nmsParam = value; }
        }

        /// <summary>
        /// Get/set the number of samples (default = 64).
        /// </summary>
        [Description("Get/set the number of samples (default = 64).")]
        public int sample_size
        {
            get { return m_nSampleSize; }
            set { m_nSampleSize = value; }
        }

        /// <summary>
        /// Get/set whether or not to use the prior bbox for nms.
        /// </summary>
        [Description("Get/set whether or not to use the prior bbox for nms.")]
        public bool use_prior_for_nms
        {
            get { return m_bUsePriorForNms; }
            set { m_bUsePriorForNms = value; }
        }

        /// <summary>
        /// Get/set whether or not to use prior for matching.
        /// </summary>
        [Description("Get/set whether or not to use prior for matching.")]
        public bool use_prior_for_matching
        {
            get { return m_bUsePriorForMatching; }
            set { m_bUsePriorForMatching = value; }
        }

        /// <summary>
        /// Use the GPU version of the algorithm.
        /// </summary>
        [Description("Use the GPU version of the algorithm.")]
        public bool use_gpu
        {
            get { return m_bUseGpu; }
            set { m_bUseGpu = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            MultiBoxLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            MultiBoxLossParameter p = src as MultiBoxLossParameter;

            m_locLossType = p.loc_loss_type;
            m_confLossType = p.conf_loss_type;
            m_fLocWeight = p.loc_weight;
            m_nNumClasses = p.num_classes;
            m_bShareLocation = p.share_location;
            m_matchType = p.match_type;
            m_fOverlapThreshold = p.overlap_threshold;
            m_nBackgroundLabelId = p.background_label_id;
            m_bUseDifficultGt = p.use_difficult_gt;
            m_bDoNegMining = p.do_neg_mining;
            m_fNegPosRatio = p.neg_pos_ratio;
            m_fNegOverlap = p.neg_overlap;
            m_codeType = p.code_type;
            m_bEncodeVarianceInTarget = p.encode_variance_in_target;
            m_bMapObjectToAgnostic = p.map_object_to_agnostic;
            m_bIgnoreCrossBoundaryBbox = p.ignore_cross_boundary_bbox;
            m_bBpInside = p.bp_inside;
            m_miningType = p.mining_type;
            m_nmsParam = p.nms_param.Clone();
            m_nSampleSize = p.sample_size;
            m_bUsePriorForNms = p.use_prior_for_nms;
            m_bUsePriorForMatching = p.use_prior_for_matching;
            m_bUseGpu = p.use_gpu;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            MultiBoxLossParameter p = new param.ssd.MultiBoxLossParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(new RawProto("loc_loss_type", loc_loss_type.ToString()));
            rgChildren.Add(new RawProto("conf_loss_type", conf_loss_type.ToString()));
            rgChildren.Add(new RawProto("loc_weight", loc_weight.ToString()));
            rgChildren.Add(new RawProto("num_classes", num_classes.ToString()));
            rgChildren.Add(new RawProto("share_location", share_location.ToString()));
            rgChildren.Add(new RawProto("match_type", match_type.ToString()));
            rgChildren.Add(new RawProto("overlap_threshold", overlap_threshold.ToString()));
            rgChildren.Add(new RawProto("background_label_id", background_label_id.ToString()));
            rgChildren.Add(new RawProto("use_difficult_gt", use_difficult_gt.ToString()));
            rgChildren.Add(new RawProto("do_neg_mining", do_neg_mining.ToString()));
            rgChildren.Add(new RawProto("neg_pos_ratio", neg_pos_ratio.ToString()));
            rgChildren.Add(new RawProto("neg_overlap", neg_overlap.ToString()));
            rgChildren.Add(new RawProto("code_type", code_type.ToString()));
            rgChildren.Add(new RawProto("encode_variance_in_target", encode_variance_in_target.ToString()));
            rgChildren.Add(new RawProto("map_object_to_agnostic", map_object_to_agnostic.ToString()));
            rgChildren.Add(new RawProto("ignore_cross_boundary_bbox", ignore_cross_boundary_bbox.ToString()));
            rgChildren.Add(new RawProto("bp_inside", bp_inside.ToString()));
            rgChildren.Add(new RawProto("mining_type", mining_type.ToString()));
            rgChildren.Add(nms_param.ToProto("nms_param"));
            rgChildren.Add(new RawProto("sample_size", sample_size.ToString()));
            rgChildren.Add(new RawProto("use_prior_for_nms", use_prior_for_nms.ToString()));
            rgChildren.Add(new RawProto("use_prior_for_matching", use_prior_for_matching.ToString()));
            rgChildren.Add(new RawProto("use_gpu", use_gpu.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static MultiBoxLossParameter FromProto(RawProto rp)
        {
            MultiBoxLossParameter p = new MultiBoxLossParameter();
            string strVal;

            if ((strVal = rp.FindValue("loc_loss_type")) != null)
                p.loc_loss_type = LocLossTypeFromString(strVal);

            if ((strVal = rp.FindValue("conf_loss_type")) != null)
                p.conf_loss_type = ConfLossTypeFromString(strVal);

            if ((strVal = rp.FindValue("loc_weight")) != null)
                p.loc_weight = float.Parse(strVal);

            if ((strVal = rp.FindValue("num_classes")) != null)
                p.num_classes = uint.Parse(strVal);

            if ((strVal = rp.FindValue("share_location")) != null)
                p.share_location = bool.Parse(strVal);

            if ((strVal = rp.FindValue("match_type")) != null)
                p.match_type = MatchTypeFromString(strVal);

            if ((strVal = rp.FindValue("overlap_threshold")) != null)
                p.overlap_threshold = float.Parse(strVal);

            if ((strVal = rp.FindValue("background_label_id")) != null)
                p.background_label_id = uint.Parse(strVal);

            if ((strVal = rp.FindValue("use_difficult_gt")) != null)
                p.use_difficult_gt = bool.Parse(strVal);

            if ((strVal = rp.FindValue("do_neg_mining")) != null)
                p.do_neg_mining = bool.Parse(strVal);

            if ((strVal = rp.FindValue("neg_pos_ratio")) != null)
                p.neg_pos_ratio = float.Parse(strVal);

            if ((strVal = rp.FindValue("neg_overlap")) != null)
                p.neg_overlap = float.Parse(strVal);

            if ((strVal = rp.FindValue("code_type")) != null)
                p.code_type = PriorBoxParameter.CodeTypeFromString(strVal);

            if ((strVal = rp.FindValue("encode_variance_in_target")) != null)
                p.encode_variance_in_target = bool.Parse(strVal);

            if ((strVal = rp.FindValue("map_object_to_agnostic")) != null)
                p.map_object_to_agnostic = bool.Parse(strVal);

            if ((strVal = rp.FindValue("ignore_corss_boundary_bbox")) != null)
                p.ignore_cross_boundary_bbox = bool.Parse(strVal);

            if ((strVal = rp.FindValue("bp_inside")) != null)
                p.bp_inside = bool.Parse(strVal);

            if ((strVal = rp.FindValue("mining_type")) != null)
                p.mining_type = MiningTypeFromString(strVal);

            RawProto rpNms = rp.FindChild("nms_param");
            if (rpNms != null)
                p.nms_param = NonMaximumSuppressionParameter.FromProto(rpNms);

            if ((strVal = rp.FindValue("sample_size")) != null)
                p.sample_size = int.Parse(strVal);

            if ((strVal = rp.FindValue("use_prior_for_nms")) != null)
                p.use_prior_for_nms = bool.Parse(strVal);

            if ((strVal = rp.FindValue("use_prior_for_matching")) != null)
                p.use_prior_for_matching = bool.Parse(strVal);

            if ((strVal = rp.FindValue("use_gpu")) != null)
                p.use_gpu = bool.Parse(strVal);

            return p;
        }
    }
}
