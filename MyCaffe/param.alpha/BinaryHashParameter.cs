using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

/// <summary>
/// The MyCaffe.param.alpha parameters are used by the MyCaffe.layer.alpha layers.
/// </summary>
/// <remarks>
/// Using parameters within the MyCaffe.layer.alpha namespace will require the use of the MyCaffe.layers.alpha.dll.
/// </remarks>
namespace MyCaffe.param.alpha
{
    /// <summary>
    /// Specifies the parameters for the BinaryHashLayer
    /// </summary>
    /// <remarks>
    /// @see [Deep Learning of Binary Hash Codes for Fast Image Retrieval](http://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W03/html/Lin_Deep_Learning_of_2015_CVPR_paper.html) by Kevin Lin, Heui-Fang Yang, Jen-Hao Hsiao and Chu-Song Chen, 2015. 
    /// </remarks>
    public class BinaryHashParameter : LayerParameterBase 
    {
        int m_nCacheDepth = 40;
        int m_nPoolSize = 20;
        double m_dfBinaryThresholdForPooling = 0.5;
        uint m_nTopK = 5;
        bool m_bEnableDebug = false;
        SELECTION_METHOD m_selMethod = SELECTION_METHOD.MINIMUM_DISTANCE;
        int m_nIterationEnable = 0;
        bool m_bEnableDuringTesting = true;
        bool m_bEnableTripletLoss = false;
        DISTANCE_TYPE m_distPass1 = DISTANCE_TYPE.HAMMING;
        DISTANCE_TYPE m_distPass2 = DISTANCE_TYPE.EUCLIDEAN;
        double m_dfAlpha = 0.1;

        /// <summary>
        /// Defines the type of distance calculation to use.
        /// </summary>
        public enum DISTANCE_TYPE
        {
            /// <summary>
            /// Calculate the hamming distance (uses the binary threshold value).
            /// </summary>
            HAMMING,
            /// <summary>
            /// Calculate the euclidean distance.
            /// </summary>
            EUCLIDEAN
        }

        /// <summary>
        /// Defines the method of selecting the final item after the second fine search completes.
        /// </summary>
        public enum SELECTION_METHOD
        {
            /// <summary>
            /// Select the item with the minimum distance and use its class.
            /// </summary>
            MINIMUM_DISTANCE,
            /// <summary>
            /// Select the class with the most votes (the class with the highest count).
            /// </summary>
            HIGHEST_VOTE,
        }

        /** @copydoc LayerParameterBase */
        public BinaryHashParameter()
        {
        }

        /// <summary>
        /// Specifies whether or not to enable the triplet loss error adjustment during back-propagation.
        /// </summary>
        [Description("Specifies whether or not to enable the triplet loss error adjustment during back-propagation.")]
        [Category("Features")]
        public bool enable_triplet_loss
        {
            get { return m_bEnableTripletLoss; }
            set { m_bEnableTripletLoss = value; }
        }

        /// <summary>
        /// Specifies to enable during the Phase::TEST.
        /// </summary>
        /// <remarks>
        /// When enabled, the testing phase will slow down.
        /// </remarks>
        [Description("Specifies to enable during the testing Phase (this will slow down the testing phase).")]
        [Category("Features")]
        public bool enable_test
        {
            get { return m_bEnableDuringTesting; }
            set { m_bEnableDuringTesting = value; }
        }

        /// <summary>
        /// Specifies whether or not to calculate the Euclidean distance  from 0 to each item in each cache
        /// and return the values in the debug Blob.
        /// </summary>
        [Description("Specifies whether or not to calculate the Euclidean distance  from 0 to each item in each cache and return the values in the debug Blob.")]
        [Category("Features")]
        public bool enable_debug
        {
            get { return m_bEnableDebug; }
            set { m_bEnableDebug = value; }
        }

        /// <summary>
        /// Specifies the triplet loss margin 'alpha' used when 'enable_triplet_loss' is set to <i>true</i>.
        /// </summary>
        [Description("Specifies the triplet loss margin 'alpha' used when 'enable_triplet_loss' is set to true.")]
        public double alpha
        {
            get { return m_dfAlpha; }
            set { m_dfAlpha = value; }
        }

        /// <summary>
        /// Specifies the distance calculation to use on the first (rough) pass.
        /// </summary>
        [Description("Specifies the distance calculation to use on the first (rough) pass.")]
        public DISTANCE_TYPE dist_calc_pass1
        {
            get { return m_distPass1; }
            set { m_distPass1 = value; }
        }

        /// <summary>
        /// Specifies the distance calculation to use on the second (fine tuning pass).
        /// </summary>
        [Description("Specifies the distance calculation to use on the second (fine tuning pass).")]
        public DISTANCE_TYPE dist_calc_pass2
        {
            get { return m_distPass2; }
            set { m_distPass2 = value; }
        }

        /// <summary>
        /// Specifies when to enable the binary hash caching - caching only begins after
        /// the iteration exceeds the iteration_enable value.  A value of 0 always
        /// enables the caching.
        /// </summary>
        [Description("Specifies when to enable the binary hash caching - caching only begins after the iteration exceeds the iteration_enable value.  A value of 0 always enables the caching.")]
        public int iteration_enable
        {
            get { return m_nIterationEnable; }
            set { m_nIterationEnable = value; }
        }

        /// <summary>
        /// Specifies the selection method used to make the final class determination.
        /// </summary>
        [Description("Specifies the selection method used to make the final class determination.")]
        public SELECTION_METHOD selection_method
        {
            get { return m_selMethod; }
            set { m_selMethod = value; }
        }

        /// <summary>
        /// Specifies the threshold used to determine whether to set the value to 1 or 0 when binarizing the outputs from
        /// the first layer.  For example if a given output is > than the threshold then the associated value is set to 1
        /// otherwise it is set to 0.  So outputs of 0.2, 2.1, 0.5 with a threshold of 0.5 would produce 0, 1, 1.
        /// </summary>
        [Description("Specifies the threshold used to determine whether to set the value to 1 or 0 when binarizing the outputs from the first layer.  For example if a given output is > than the threshold then the associated value is set to 1 otherwise it is set to 0.  So outputs of 0.2, 2.1, 0.5 with a threshold of 0.5 would produce 0, 1, 1.")]
        public double binary_threshold
        {
            get { return m_dfBinaryThresholdForPooling; }
            set { m_dfBinaryThresholdForPooling = value; }
        }

        /// <summary>
        /// Specifies the cache depth (per class) of the caches used to cache outputs
        /// of the layers feeding into the BinaryHashLayer.
        /// </summary>
        [Description("Specifies the cache depth (per class) of the first and second caches used to cache outputs of the layers feeding into the BinaryHashLayer.")]
        public int cache_depth
        {
            get { return m_nCacheDepth; }
            set { m_nCacheDepth = value; }
        }

        /// <summary>
        /// Specifies the size of the pool to fill from the first rough-pass.  The items from the rough-pass with the 
        /// lowest hamming distance to the target item fill this pool.
        /// </summary>
        [Description("Specifies the size of the pool to fill from the first rough-pass.  The items from the rough-pass with the lowest hamming distance to the target item fill this pool.")]
        public int pool_size
        {
            get { return m_nPoolSize; }
            set { m_nPoolSize = value; }
        }

        /// <summary>
        /// Specifies the number of top results to select from the fine-grained selection pass.  The final
        /// class is selected from these items either by highest rank, or by vote.
        /// </summary>
        [Description("Specifies the number of top results to select from the fine-grained selection pass.  The final class is selected from these items either by highest rank, or by vote.")]
        public uint top_k
        {
            get { return m_nTopK; }
            set { m_nTopK = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            BinaryHashParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            BinaryHashParameter p = (BinaryHashParameter)src;
            m_nCacheDepth = p.m_nCacheDepth;
            m_dfBinaryThresholdForPooling = p.m_dfBinaryThresholdForPooling;
            m_nTopK = p.m_nTopK;
            m_nPoolSize = p.m_nPoolSize;
            m_bEnableDebug = p.m_bEnableDebug;
            m_selMethod = p.m_selMethod;
            m_nIterationEnable = p.m_nIterationEnable;
            m_bEnableDuringTesting = p.m_bEnableDuringTesting;
            m_distPass1 = p.m_distPass1;
            m_distPass2 = p.m_distPass2;
            m_bEnableTripletLoss = p.m_bEnableTripletLoss;
            m_dfAlpha = p.m_dfAlpha;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            BinaryHashParameter p = new BinaryHashParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("cache_depth", cache_depth.ToString());
            rgChildren.Add("pool_size", pool_size.ToString());
            rgChildren.Add("top_k", top_k.ToString());
            rgChildren.Add("binary_threshold", binary_threshold.ToString());
            rgChildren.Add("enable_debug", enable_debug.ToString());
            rgChildren.Add("selection_method", selection_method.ToString());
            rgChildren.Add("iteration_enable", m_nIterationEnable.ToString());
            rgChildren.Add("enable_test", enable_test.ToString());
            rgChildren.Add("dist_calc_pass1", dist_calc_pass1.ToString());
            rgChildren.Add("dist_calc_pass2", dist_calc_pass2.ToString());
            rgChildren.Add("enable_triplet_loss", enable_triplet_loss.ToString());
            rgChildren.Add("alpha", alpha.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static BinaryHashParameter FromProto(RawProto rp)
        {
            string strVal;
            BinaryHashParameter p = new BinaryHashParameter();

            if ((strVal = rp.FindValue("cache_depth")) != null)
                p.cache_depth = int.Parse(strVal);

            if ((strVal = rp.FindValue("pool_size")) != null)
                p.pool_size = int.Parse(strVal);

            if ((strVal = rp.FindValue("top_k")) != null)
                p.top_k = uint.Parse(strVal);

            if ((strVal = rp.FindValue("binary_threshold")) != null)
                p.binary_threshold = double.Parse(strVal);

            if ((strVal = rp.FindValue("enable_debug")) != null)
                p.enable_debug = bool.Parse(strVal);

            if ((strVal = rp.FindValue("selection_method")) != null)
            {
                if (strVal == SELECTION_METHOD.HIGHEST_VOTE.ToString())
                    p.selection_method = SELECTION_METHOD.HIGHEST_VOTE;
                else
                    p.selection_method = SELECTION_METHOD.MINIMUM_DISTANCE;
            }

            if ((strVal = rp.FindValue("iteration_enable")) != null)
                p.iteration_enable = int.Parse(strVal);

            if ((strVal = rp.FindValue("enable_test")) != null)
                p.enable_test = bool.Parse(strVal);

            if ((strVal = rp.FindValue("dist_calc_pass1")) != null)
            {
                if (strVal == DISTANCE_TYPE.EUCLIDEAN.ToString())
                    p.dist_calc_pass1 = DISTANCE_TYPE.EUCLIDEAN;
                else
                    p.dist_calc_pass1 = DISTANCE_TYPE.HAMMING;
            }

            if ((strVal = rp.FindValue("dist_calc_pass2")) != null)
            {
                if (strVal == DISTANCE_TYPE.HAMMING.ToString())
                    p.dist_calc_pass2 = DISTANCE_TYPE.HAMMING;
                else
                    p.dist_calc_pass2 = DISTANCE_TYPE.EUCLIDEAN;
            }

            if ((strVal = rp.FindValue("enable_triplet_loss")) != null)
                p.enable_triplet_loss = bool.Parse(strVal);

            if ((strVal = rp.FindValue("alpha")) != null)
                p.alpha = double.Parse(strVal);

            return p;
        }
    }
}
