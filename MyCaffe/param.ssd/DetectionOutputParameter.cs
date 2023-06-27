using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.ssd
{
    /// <summary>
    /// Specifies the parameters for the DetectionOutputLayer.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class DetectionOutputParameter : LayerParameterBase 
    {
        uint m_nNumClasses;
        bool m_bShareLocation = true;
        int m_nBackgroundLabelId = 0;
        NonMaximumSuppressionParameter m_nmsParam = new NonMaximumSuppressionParameter(true);
        SaveOutputParameter m_saveOutputParam = new SaveOutputParameter(true);
        PriorBoxParameter.CodeType m_codeType = PriorBoxParameter.CodeType.CORNER;
        bool m_bVarianceEncodedInTarget = false;
        int m_nKeepTopK = -1;
        float? m_fConfidenceThreshold;
        bool m_bVisualize = false;
        float? m_fVisualizeThreshold = null;
        string m_strSaveFile = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        public DetectionOutputParameter()
        {
        }

        /// <summary>
        /// Specifies the number of classes that are actually predicted - required!
        /// </summary>
        [Description("Specifies the number of classes that are actually predicted - required!")]
        public uint num_classes
        {
            get { return m_nNumClasses; }
            set { m_nNumClasses = value; }
        }

        /// <summary>
        /// Specifies whether or not to sare the bounding box is shared among different classes (default = true).
        /// </summary>
        [Description("If true, bounding boxe is shared among different classes.")]
        public bool share_location
        {
            get { return m_bShareLocation; }
            set { m_bShareLocation = value; }
        }

        /// <summary>
        /// Specifies the background class.
        /// </summary>
        /// <remarks>
        /// If there is no background label this should be set to -1.
        /// </remarks>
        [Description("Specifies the background class, set to -1 when there is no background class.")]
        public int background_label_id
        {
            get { return m_nBackgroundLabelId; }
            set { m_nBackgroundLabelId = value; }
        }

        /// <summary>
        /// Specifies the parameter used for non maximum suppression.
        /// </summary>
        [Description("Parameter used for non maximum suppression.")]
        public NonMaximumSuppressionParameter nms_param
        {
            get { return m_nmsParam; }
            set { m_nmsParam = value; }
        }

        /// <summary>
        /// Specifies the parameter used for saving the detection results.
        /// </summary>
        [Description("Specifies the parameter used for saving detection results.")]
        public SaveOutputParameter save_output_param
        {
            get { return m_saveOutputParam; }
            set { m_saveOutputParam = value; }
        }

        /// <summary>
        /// Specifies the coding method for the bbox.
        /// </summary>
        [Description("Specifies the coding method for the bbox.")]
        public PriorBoxParameter.CodeType code_type
        {
            get { return m_codeType; }
            set { m_codeType = value; }
        }

        /// <summary>
        /// Specifies whether or not the variance is encoded in the target; otherwise we need to adjust the predicted offset accordingly.
        /// </summary>
        [Description("Specifies whether or not the variance is encoded in the target; otherwise we need to adjust the predicted offset accordingly.")]
        public bool variance_encoded_in_target
        {
            get { return m_bVarianceEncodedInTarget; }
            set { m_bVarianceEncodedInTarget = value; }
        }

        /// <summary>
        /// Specifies the number of total bboxes to be kept per image after nms step, -1 means keeping all bboxes after nms step.
        /// </summary>
        [Description("Specifies the number of total bboxes to be kept per image after nms step, -1 means keeping all bboxes after nms step.")]
        public int keep_top_k
        {
            get { return m_nKeepTopK; }
            set { m_nKeepTopK = value; }
        }

        /// <summary>
        /// Specifies the threshold for deciding which detections to consider - only those which are larger than this threshold.
        /// </summary>
        [Description("Specifies the threshold for deciding which detections to consider - only those which are larger than this threshold.")]
        public float? confidence_threshold
        {
            get { return m_fConfidenceThreshold; }
            set { m_fConfidenceThreshold = value; }
        }

        /// <summary>
        /// Specifies whether or not to visualize the detection results.
        /// </summary>
        [Description("Specifies whether or not to visualize the detection results.")]
        public bool visualize
        {
            get { return m_bVisualize; }
            set { m_bVisualize = value; }
        }

        /// <summary>
        /// Specifies the theshold used to visualize detection results.
        /// </summary>
        [Description("Specifies the theshold used to visualize detection results.")]
        public float? visualize_threshold
        {
            get { return m_fVisualizeThreshold; }
            set { m_fVisualizeThreshold = value; }
        }

        /// <summary>
        /// When provided, specifies the outputs to the video file.
        /// </summary>
        [Description("When provided, specifies the outputs to the video file.")]
        public string save_file
        {
            get { return m_strSaveFile; }
            set { m_strSaveFile = value; }
        }

        /// <summary>
        /// Load the parameter from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bNewInstance">When <i>true</i> a new instance is created (the default), otherwise the existing instance is loaded from the binary reader.</param>
        /// <returns>Returns an instance of the parameter.</returns>
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            DetectionOutputParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Copy on parameter to another.
        /// </summary>
        /// <param name="src">Specifies the parameter to copy.</param>
        public override void Copy(LayerParameterBase src)
        {
            DetectionOutputParameter p = (DetectionOutputParameter)src;

            m_nNumClasses = p.m_nNumClasses;
            m_bShareLocation = p.m_bShareLocation;
            m_nBackgroundLabelId = p.m_nBackgroundLabelId;
            m_nmsParam = p.m_nmsParam.Clone();
            m_saveOutputParam = p.save_output_param.Clone() as SaveOutputParameter;
            m_codeType = p.m_codeType;
            m_bVarianceEncodedInTarget = p.m_bVarianceEncodedInTarget;
            m_nKeepTopK = p.m_nKeepTopK;
            m_fConfidenceThreshold = p.m_fConfidenceThreshold;
            m_bVisualize = p.m_bVisualize;
            m_fVisualizeThreshold = p.m_fVisualizeThreshold;
            m_strSaveFile = p.m_strSaveFile;
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public override LayerParameterBase Clone()
        {
            DetectionOutputParameter p = new DetectionOutputParameter();
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

            rgChildren.Add(new RawProto("num_classes", num_classes.ToString()));
            rgChildren.Add(new RawProto("share_location", share_location.ToString()));
            rgChildren.Add(new RawProto("background_label_id", background_label_id.ToString()));

            if (nms_param != null)
                rgChildren.Add(nms_param.ToProto("nms_param"));

            if (save_output_param != null)
                rgChildren.Add(save_output_param.ToProto("save_output_param"));

            rgChildren.Add(new RawProto("code_type", code_type.ToString()));
            rgChildren.Add(new RawProto("variance_encoded_in_target", variance_encoded_in_target.ToString()));
            rgChildren.Add(new RawProto("keep_top_k", keep_top_k.ToString()));

            if (confidence_threshold.HasValue)
                rgChildren.Add(new RawProto("confidence_threshold", confidence_threshold.Value.ToString()));

            rgChildren.Add(new RawProto("visualize", visualize.ToString()));

            if (visualize_threshold.HasValue)
                rgChildren.Add(new RawProto("visualize_threshold", visualize_threshold.Value.ToString()));

            rgChildren.Add(new RawProto("save_file", save_file));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static DetectionOutputParameter FromProto(RawProto rp)
        {
            DetectionOutputParameter p = new DetectionOutputParameter();
            string strVal;

            if ((strVal = rp.FindValue("num_classes")) != null)
                p.num_classes = uint.Parse(strVal);

            if ((strVal = rp.FindValue("share_location")) != null)
                p.share_location = bool.Parse(strVal);

            if ((strVal = rp.FindValue("background_label_id")) != null)
                p.background_label_id = int.Parse(strVal);

            RawProto rpNms = rp.FindChild("nms_param");
            if (rpNms != null)
                p.nms_param = NonMaximumSuppressionParameter.FromProto(rpNms);

            RawProto rpSave = rp.FindChild("save_output_param");
            if (rpSave != null)
                p.save_output_param = SaveOutputParameter.FromProto(rpSave);

            if ((strVal = rp.FindValue("code_type")) != null)
            {
                if (strVal == PriorBoxParameter.CodeType.CENTER_SIZE.ToString())
                    p.code_type = PriorBoxParameter.CodeType.CENTER_SIZE;
                else if (strVal == PriorBoxParameter.CodeType.CORNER.ToString())
                    p.code_type = PriorBoxParameter.CodeType.CORNER;
                else if (strVal == PriorBoxParameter.CodeType.CORNER_SIZE.ToString())
                    p.code_type = PriorBoxParameter.CodeType.CORNER_SIZE;
                else
                    throw new Exception("Unknown PriorBoxParameter.CodeType '" + strVal + "'!");
            }

            if ((strVal = rp.FindValue("variance_encoded_in_target")) != null)
                p.variance_encoded_in_target = bool.Parse(strVal);

            if ((strVal = rp.FindValue("keep_top_k")) != null)
                p.keep_top_k = int.Parse(strVal);

            if ((strVal = rp.FindValue("confidence_threshold")) != null)
                p.confidence_threshold = ParseFloat(strVal);

            if ((strVal = rp.FindValue("visualize")) != null)
                p.visualize = bool.Parse(strVal);

            if ((strVal = rp.FindValue("visualize_threshold")) != null)
                p.visualize_threshold = ParseFloat(strVal);

            if ((strVal = rp.FindValue("save_file")) != null)
                p.save_file = strVal;

            return p;
        }
    }
}
