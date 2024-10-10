﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameter for the data layer.
    /// </summary>
    /// <remarks>
    /// Note: given the new use of the Transformation Parameter, the
    /// depreciated elements of the DataParameter have been removed.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class DataParameter : LayerParameterBase
    {
        /// <summary>
        /// Defines the database type to use.
        /// </summary>
        public enum DB
        {
            /// <summary>
            /// Specifies that no backend database is used as is the case with the ImageDataLayer.
            /// </summary>
            NONE = 0,
            /// <summary>
            /// Specifies to use the MyCaffeImageDatabase.  Currently this is the only option.
            /// </summary>
            IMAGEDB = 1
        }

        string m_strSource = null;
        uint m_nBatchSize;
        DB m_backend = DB.IMAGEDB;
        uint m_nPrefetch = 4;
        bool? m_bEnableRandomSelection = null;
        bool? m_bEnablePairSelection = null;
        bool m_bDisplayTiming = false;
        LABEL_TYPE m_labelType = LABEL_TYPE.SINGLE;
        bool m_bPrimaryData = true;
        string m_strSynchronizeWith = null;
        bool m_bSyncTarget = false;
        int m_nImagesPerBlob = 1;
        bool m_bOutputAllLabels = false;
        bool m_bBalanceMatches = true;
        bool m_bOutputImageInfo = false;
        int m_nForcedPrimaryLabel = -1;
        bool m_bEnableNoiseForNonMatch = false;
        DataNoiseParameter m_dataNoiseParam = new DataNoiseParameter();
        bool m_bEnableDebugOutput = false;
        DataDebugParameter m_dataDebugParam = new DataDebugParameter();
        int m_nOneHotLabelEncodingSize = 0; // Note when using OneHotLabelEncoding, m_labelType must = LABEL_TYPE.MULTIPLE
        bool m_bUseScoreAsLabel = false;
        bool m_bEnableScoreAsLabelNormalization = false;

        /// <summary>
        /// This event is, optionally, called to verify the batch size of the DataParameter.
        /// </summary>
        public event EventHandler<VerifyBatchSizeArgs> OnVerifyBatchSize;

        /** @copydoc LayerParameterBase */
        public DataParameter()
        {
        }

        /// <summary>
        /// When used with the DATA parameter, specifies the data 'source' within the database.  Some sources are used for training whereas others are used for testing.  When used with the IMAGE_DATA parameter, the 'source' specifies the data 'source' file containing the list of image file names.  Each dataset has both a training and testing data source.
        /// </summary>
        [Description("When used with the DATA parameter, specifies the data 'source' within the database.  Some sources are used for training whereas others are used for testing.  When used with the IMAGE_DATA parameter, the 'source' specifies the data 'source' file containing the list of image file names.  Each dataset has both a training and testing data source.")]
        public string source
        {
            get { return m_strSource; }
            set { m_strSource = value; }
        }

        /// <summary>
        /// Specifies the batch size.
        /// </summary>
        [Description("Specifies the batch size of images to collect and train on each iteration of the network.  NOTE: Setting the training netorks batch size >= to the testing net batch size will conserve memory by allowing the training net to share its gpu memory with the testing net.")]
        public virtual uint batch_size
        {
            get { return m_nBatchSize; }
            set
            {
                if (OnVerifyBatchSize != null)
                {
                    VerifyBatchSizeArgs args = new VerifyBatchSizeArgs(value);
                    OnVerifyBatchSize(this, args);
                    if (args.Error != null)
                        throw args.Error;
                }

                m_nBatchSize = value;
            }
        }

        /// <summary>
        /// Specifies the backend database.
        /// </summary>
        /// <remarks>
        /// NOTE: Currently only the IMAGEDB is supported, which is a separate
        /// component used to load and manage all images within a given dataset.
        /// </remarks>
        [Description("Specifies the backend database type.  Currently only the IMAGEDB database type is supported.  However protofiles specifying the 'LMDB' backend are converted into the 'IMAGEDB' type.")]
        public DB backend
        {
            get { return m_backend; }
            set { m_backend = value; }
        }

        /// <summary>
        /// Prefetch queue (Number of batches to prefetch to host memory, increase if
        /// data access bandwidth varies).
        /// </summary>
        [Description("Specifies the number of batches to prefetch to host memory.  Increase this value if data access bandwidth varies.")]
        public uint prefetch
        {
            get { return m_nPrefetch; }
            set { m_nPrefetch = value; }
        }

        /// <summary>
        /// (\b optional, default = null) Specifies whether or not to randomly query images from the data source.  When enabled, images are queried in sequence which can often have poorer training results.
        /// </summary>
        [Category("Data Selection"), Description("Specifies whether or not to randomly query images from the data source.  When false, images are queried in sequence which can often have poorer training results.")]
        public bool? enable_random_selection
        {
            get { return m_bEnableRandomSelection; }
            set { m_bEnableRandomSelection = value; }
        }

        /// <summary>
        /// (\b optional, default = null) Specifies whether or not to select images in a pair sequence.  When enabled, the first image queried is queried using the 'random' selection property, and then the second image queried is the image just after the first image queried (even if queried randomly).
        /// </summary>
        [Category("Data Selection"), Description("Specifies whether or not to select images in a pair sequence.  When enabled, the first image queried is queried using the 'random' selection property, and then the second image queried is the image just after the first image queried (even if queried randomly).")]
        public bool? enable_pair_selection
        {
            get { return m_bEnablePairSelection; }
            set { m_bEnablePairSelection = value; }
        }

        /// <summary>
        /// (\b optional, default = false) Specifies whether or not to display the timing of each image read.
        /// </summary>
        [Category("Debugging"), Description("Specifies whether or not to display the timing of each image read.")]
        public bool display_timing
        {
            get { return m_bDisplayTiming; }
            set { m_bDisplayTiming = value; }
        }

        /// <summary>
        /// (\b optional, default = SINGLE) Specifies the label type: SINGLE - the default which uses the 'Label' field, or MULTIPLE - which uses the 'DataCriteria' field.  
        /// </summary>
        [Category("Labels"), Description("Specifies the label type: SINGLE - the default which uses the 'Label' field, or MULTIPLE - which uses the 'DataCriteria' field.")]
        public LABEL_TYPE label_type
        {
            get { return m_labelType; }
            set { m_labelType = value; }
        }

        /// <summary>
        /// (\b optional, default = true) Specifies whether or not the data is the primary datset as opposed to a secondary, target dataset.
        /// </summary>
        [Category("Data Selection"), Description("Specifies whether or not this data is the primary dataset as opposed to the target dataset.  By default, this is set to 'true'.")]
        public bool primary_data
        {
            get { return m_bPrimaryData; }
            set { m_bPrimaryData = value; }
        }

        /// <summary>
        /// (\b optional, default = false) Specifies whether or not this is a to be synchronized with another data layer as the target.
        /// </summary>
        [Category("Synchronization"), Description("Specifies whether or not this is to be synchronized with another data layer as the target.")]
        public bool synchronize_target
        {
            get { return m_bSyncTarget; }
            set { m_bSyncTarget = value; }
        }

        /// <summary>
        /// (\b optional, default = null) Specifies a secondary (target) dataset to syncrhonize with.
        /// </summary>
        /// <remarks>
        /// When synchronizing with another dataset the ordering of labels is guaranteed to be the same from both data sets even though
        /// the images selected are selected at random.
        /// </remarks>
        [Category("Synchronization"), Description("Specifies a secondary (target) dataset to synchronize with.")]
        public string synchronize_with
        {
            get { return m_strSynchronizeWith; }
            set { m_strSynchronizeWith = value; }
        }

        /// <summary>
        /// (\b optional, default = 1) Specifies the number of images to load into each blob channel.  For example when set to 2 two 3 channel images are loaded and stacked on the channel dimension,
        /// thus loading a 6 channel blob (2 images x 3 channels each). 
        /// </summary>
        /// <remarks>
        /// Loading images in pairs (images_per_blob = 2) is used with the siamese network, where the channel of each blob contains the first image followed by the second image.  The total individual
        /// image channel count equals the blob channel count divided by 2.
        /// </remarks>
        [Category("Multi-Image"), Description("Optionally, specifies the number of images to load into each blob channel, where each image is stacked on the channel dimension so a 3 channel item then becomes a 6 channel item when two images are stacked (default = 1).")]
        public int images_per_blob
        {
            get { return m_nImagesPerBlob; }
            set { m_nImagesPerBlob = value; }
        }

        /// <summary>
        /// (\b optional, default = false) When using images_per_blob > 1, 'output_all_labels' specifies to output all labels for the stacked images instead of just the comparison.
        /// </summary>
        [Category("Multi-Image"), Description("Optionally, specifies to output all labels for the stacked images instead of just the comparison.  This setting only applies when 'images_per_blob' > 1 (default = false).")]
        public bool output_all_labels
        {
            get { return m_bOutputAllLabels; }
            set { m_bOutputAllLabels = value; }
        }

        /// <summary>
        /// (\b optional, default = true) When using images_per_blob > 1, 'balance_matches' specifies to query images by alternating similar matches followed by dissimilar matches in the next query.
        /// </summary>
        [Category("Multi-Image"), Description("Optionally, specifies to balance the matches by alternating between matching classes and non matching classes.  This setting only applies when 'images_per_blob' > 1 (default = true).")]
        public bool balance_matches
        {
            get { return m_bBalanceMatches; }
            set { m_bBalanceMatches = value; }
        }

        /// <summary>
        /// (\b optional, default = false) When <i>true</i> image information such as index and label are output. IMPORTANT: enabling this setting can dramatically slow down training and is only used for debugging.
        /// </summary>
        [Category("Debugging"), Description("Optionally, specifies to output image information such as index and label which is only intended for debugging as it dramatically slows down processing. (default = false).")]
        public bool output_image_information
        {
            get { return m_bOutputImageInfo; }
            set { m_bOutputImageInfo = value; }
        }

        /// <summary>
        /// (\b optional, default = -1) When >= 0, this label is used as the primary image label when 'images_per_blob' > 1.
        /// </summary>
        [Category("Multi-Image"), Description("Optionally, specifies to force the label of the primary image to this value.  This setting only applies when 'images_per_blob' > 1 (default = -1 which causes this setting to be ignored).")]
        public int forced_primary_label
        {
            get { return m_nForcedPrimaryLabel; }
            set { m_nForcedPrimaryLabel = value; }
        }

        /// <summary>
        /// (\b optional, default = false) When <i>true</i> an image consisting of noise initialized with noise filler.
        /// </summary>
        [Category("Multi-Image"), Description("Optionally, specifies to use a noise generated image for the non matching image instead of an image from a different class.  This setting only applies when 'images_per_blob' > 1 (default = false).")]
        public bool enable_noise_for_nonmatch
        {
            get { return m_bEnableNoiseForNonMatch; }
            set { m_bEnableNoiseForNonMatch = value; }
        }

        /// <summary>
        /// Specifies the DataNoiseParameter used when 'enable_noise_for_nonmatch' = True.
        /// </summary>
        [Category("Multi-Image"), Description("Optionally, specifies the DataNoiseParameter that defines the noise used when 'enable_noise_for_nonmatch' = True.  This setting only applies when 'images_per_blob' > 1.")]
        public DataNoiseParameter data_noise_param
        {
            get { return m_dataNoiseParam; }
            set { m_dataNoiseParam = value; }
        }

        /// <summary>
        /// (\b optional, default = false) When <i>true</i> the data sent out through the top are saved as images into the debug directory specified by the data_debug_param.
        /// </summary>
        [Category("Debugging"), Description("Optionally, specifies to output debug information about the data sent out through the top. (default = false).")]
        public bool enable_debug_output
        {
            get { return m_bEnableDebugOutput; }
            set { m_bEnableDebugOutput = value; }
        }

        /// <summary>
        /// Specifies the DataDebugParameter used when 'enable_debug_output' = True.
        /// </summary>
        [Category("Debugging"), Description("Optionally, specifies the parameters used when 'enable_debug_output' = True.")]
        public DataDebugParameter data_debug_param
        {
            get { return m_dataDebugParam; }
            set { m_dataDebugParam = value; }
        }

        /// <summary>
        /// When greater than 0 (default = 0), labels are one-hot encoded to a vector of the one-hot label size (e.g., when size = 4: 3 -> 0 1 1 0; 4 -> 1 0 0 0) 
        /// </summary>
        [Category("Labels"), Description("When greater than 0 (default = 0), labels are one-hot encoded to a vector of the one-hot label size (e.g., when size = 4: 3 -> 0 1 1 0; 4 -> 1 0 0 0)")]
        public int one_hot_label_size
        {
            get { return m_nOneHotLabelEncodingSize; }
            set { m_nOneHotLabelEncodingSize = value; }
        }

        /// <summary>
        /// When enabled the score is used as the label, which is useful in regression models (default = false).
        /// </summary>
        [Category("Labels"), Description("When enabled the score is used as the label, which is useful in regression models (default = false).")]
        public bool use_score_as_label
        {
            get { return m_bUseScoreAsLabel; }
            set { m_bUseScoreAsLabel = value; }
        }

        /// <summary>
        /// When enabled, score as label normalization is attempted (default = false).
        /// </summary>
        /// <remarks>
        /// Score as label normalization requires that a Mean image exist in the dataset with the following
        /// image parameters set in the database.
        ///     'Mean' - specifies the mean score.
        ///     'StdDev' - specifies the standard deviation of the score.
        /// During normalization, these values are used to perform Z-score normalization where the mean score is
        /// subtracted from each score then divided by the score standard deviation.
        /// 
        /// If these parameters or the mean image do not exist, a warning is produced and no normalization
        /// takes place.
        /// </remarks>
        [Category("Labels"), Description("When enabled, score as label normalization is attempted using z-score normalization (default = false).")]
        public bool enable_score_as_label_normalization
        {
            get { return m_bEnableScoreAsLabelNormalization; }
            set { m_bEnableScoreAsLabelNormalization = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            DataParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            DataParameter p = (DataParameter)src;
            m_strSource = p.m_strSource;
            m_nBatchSize = p.m_nBatchSize;
            m_backend = p.m_backend;
            m_nPrefetch = p.m_nPrefetch;
            m_bEnableRandomSelection = p.m_bEnableRandomSelection;
            m_bEnablePairSelection = p.m_bEnablePairSelection;
            m_bDisplayTiming = p.m_bDisplayTiming;
            m_labelType = p.m_labelType;
            m_bPrimaryData = p.m_bPrimaryData;
            m_strSynchronizeWith = p.m_strSynchronizeWith;
            m_bSyncTarget = p.m_bSyncTarget;
            m_nImagesPerBlob = p.m_nImagesPerBlob;
            m_bOutputAllLabels = p.m_bOutputAllLabels;
            m_bBalanceMatches = p.m_bBalanceMatches;
            m_bOutputImageInfo = p.m_bOutputImageInfo;
            m_bEnableNoiseForNonMatch = p.m_bEnableNoiseForNonMatch;
            m_dataNoiseParam.Copy(p.m_dataNoiseParam);
            m_bEnableDebugOutput = p.m_bEnableDebugOutput;
            m_dataDebugParam.Copy(p.m_dataDebugParam);
            m_nForcedPrimaryLabel = p.m_nForcedPrimaryLabel;
            m_nOneHotLabelEncodingSize = p.m_nOneHotLabelEncodingSize;
            m_bUseScoreAsLabel = p.m_bUseScoreAsLabel;
            m_bEnableScoreAsLabelNormalization = p.m_bEnableScoreAsLabelNormalization;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            DataParameter p = new DataParameter();
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

            rgChildren.Add("source", "\"" + source + "\"");
            rgChildren.Add("batch_size", batch_size.ToString());
            rgChildren.Add("backend", backend.ToString());

            if (prefetch != 4)
                rgChildren.Add("prefetch", prefetch.ToString());

            rgChildren.Add("enable_random_selection", enable_random_selection.GetValueOrDefault(true).ToString());

            if (enable_pair_selection.GetValueOrDefault(false) == true)
                rgChildren.Add("enable_pair_selection", enable_pair_selection.Value.ToString());

            if (display_timing == true)
                rgChildren.Add("display_timing", display_timing.ToString());

            if (label_type != LABEL_TYPE.SINGLE)
                rgChildren.Add("label_type", label_type.ToString());

            if (primary_data == false)
                rgChildren.Add("primary_data", primary_data.ToString());

            if (synchronize_with != null)
                rgChildren.Add("synchronize_with", m_strSynchronizeWith);

            if (synchronize_target)
                rgChildren.Add("synchronize_target", m_bSyncTarget.ToString());

            if (m_nImagesPerBlob > 1)
            {
                rgChildren.Add("images_per_blob", m_nImagesPerBlob.ToString());
                rgChildren.Add("output_all_labels", m_bOutputAllLabels.ToString());
                rgChildren.Add("balance_matches", m_bBalanceMatches.ToString());
            }

            if (output_image_information)
                rgChildren.Add("output_image_information", m_bOutputImageInfo.ToString());

            if (enable_noise_for_nonmatch)
            {
                rgChildren.Add("enable_noise_for_nonmatch", m_bEnableNoiseForNonMatch.ToString());
                rgChildren.Add(m_dataNoiseParam.ToProto("data_noise_param"));
            }

            if (enable_debug_output)
            {
                rgChildren.Add("enable_debug_output", m_bEnableDebugOutput.ToString());
                rgChildren.Add(m_dataDebugParam.ToProto("data_debug_param"));
            }

            if (m_nForcedPrimaryLabel >= 0)
                rgChildren.Add("forced_primary_label", m_nForcedPrimaryLabel.ToString());

            if (one_hot_label_size > 0)
                rgChildren.Add("one_hot_label_size", one_hot_label_size.ToString());

            if (use_score_as_label)
            {
                rgChildren.Add("use_score_as_label", use_score_as_label.ToString());

                if (enable_score_as_label_normalization)
                    rgChildren.Add("enable_score_as_label_normalization", enable_score_as_label_normalization.ToString());
            }

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <param name="p">Optionally, specifies an instance to load.  If <i>null</i>, a new instance is created and loaded.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static DataParameter FromProto(RawProto rp, DataParameter p = null)
        {
            string strVal;

            if (p == null)
                p = new DataParameter();

            if ((strVal = rp.FindValue("source")) != null)
                p.source = strVal.Trim('\"');

            if ((strVal = rp.FindValue("batch_size")) != null)
                p.batch_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("backend")) != null)
            {
                switch (strVal)
                {
                    case "IMAGEDB":
                        p.backend = DB.IMAGEDB;
                        break;

                    case "LMDB":
                        p.backend = DB.IMAGEDB;
                        break;

                    case "NONE":
                        p.backend = DB.NONE;
                        break;

                    default:
                        throw new Exception("Unknown 'backend' value " + strVal);
                }
            }

            if ((strVal = rp.FindValue("prefetch")) != null)
                p.prefetch = uint.Parse(strVal);

            if ((strVal = rp.FindValue("enable_random_selection")) != null)
                p.enable_random_selection = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_pair_selection")) != null)
                p.enable_pair_selection = bool.Parse(strVal);

            if ((strVal = rp.FindValue("display_timing")) != null)
                p.display_timing = bool.Parse(strVal);

            if ((strVal = rp.FindValue("label_type")) != null)
            {
                switch (strVal)
                {
                    case "SINGLE":
                        p.label_type = LABEL_TYPE.SINGLE;
                        break;

                    case "MULTIPLE":
                        p.label_type = LABEL_TYPE.MULTIPLE;
                        break;

                    default:
                        throw new Exception("Unknown 'label_type' value " + strVal);
                }
            }

            if ((strVal = rp.FindValue("primary_data")) != null)
                p.primary_data = bool.Parse(strVal);

            p.synchronize_with = rp.FindValue("synchronize_with");

            if ((strVal = rp.FindValue("synchronize_target")) != null)
                p.synchronize_target = bool.Parse(strVal);

            if ((strVal = rp.FindValue("images_per_blob")) != null)
                p.images_per_blob = int.Parse(strVal);

            if ((strVal = rp.FindValue("output_all_labels")) != null)
                p.output_all_labels = bool.Parse(strVal);

            if ((strVal = rp.FindValue("balance_matches")) != null)
                p.balance_matches = bool.Parse(strVal);

            if ((strVal = rp.FindValue("output_image_information")) != null)
                p.output_image_information = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_noise_for_nonmatch")) != null)
                p.enable_noise_for_nonmatch = bool.Parse(strVal);

            RawProto rpDataNoise = rp.FindChild("data_noise_param");
            if (rpDataNoise != null)
                p.data_noise_param = DataNoiseParameter.FromProto(rpDataNoise);

            if ((strVal = rp.FindValue("enable_debug_output")) != null)
                p.enable_debug_output = bool.Parse(strVal);

            RawProto rpDataDebug = rp.FindChild("data_debug_param");
            if (rpDataDebug != null)
                p.data_debug_param = DataDebugParameter.FromProto(rpDataDebug);

            if ((strVal = rp.FindValue("forced_primary_label")) != null)
                p.forced_primary_label = int.Parse(strVal);

            if ((strVal = rp.FindValue("one_hot_label_size")) != null)
                p.one_hot_label_size = int.Parse(strVal);

            if ((strVal = rp.FindValue("use_score_as_label")) != null)
                p.use_score_as_label = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_score_as_label_normalization")) != null)
                p.enable_score_as_label_normalization = bool.Parse(strVal);

            return p;
        }
    }

    /// <summary>
    /// The VerifyBatchSizeArgs class defines the arguments of the OnVerifyBatchSize event.
    /// </summary>
    public class VerifyBatchSizeArgs : EventArgs
    {
        uint m_uiBatchSize;
        Exception m_err = null;

        /// <summary>
        /// VerifyBatchSizeArgs constructor.
        /// </summary>
        /// <param name="uiBatchSize"></param>
        public VerifyBatchSizeArgs(uint uiBatchSize)
        {
            m_uiBatchSize = uiBatchSize;
        }

        /// <summary>
        /// Get/set the error value.  For example if the receiver of the event determines that the batch size is in error, 
        /// then the receiver should set the error appropriately.
        /// </summary>
        public Exception Error
        {
            get { return m_err; }
            set { m_err = value; }
        }

        /// <summary>
        /// Specifies the proposed batch size that the DataLayer would like to use.
        /// </summary>
        public uint ProposedBatchSize
        {
            get { return m_uiBatchSize; }
        }
    }
}
