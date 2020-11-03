using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.ssd
{
    /// <summary>
    /// Specifies the parameters for the DetectionEvaluateLayer.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class DetectionEvaluateParameter : LayerParameterBase 
    {
        uint m_nNumClasses;
        uint m_nBackgroundLabelId = 0;
        float m_fOverlapThreshold = 0.5f;
        bool m_bEvaluateDifficultGt = true;
        string m_strNameSizeFile;
        ResizeParameter m_resizeParam = new ResizeParameter(false);

        /// <summary>
        /// The constructor.
        /// </summary>
        public DetectionEvaluateParameter()
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
        /// Specifies the background class.
        /// </summary>
        /// <remarks>
        /// Needed for sanity check so that background class is neither in the ground truth nor the detections.
        /// </remarks>
        [Description("Specifies the background class, needed for sanity check so that the background class is neither in the ground truth nor the detections.")]
        public uint background_label_id
        {
            get { return m_nBackgroundLabelId; }
            set { m_nBackgroundLabelId = value; }
        }

        /// <summary>
        /// Specifies the threshold for deciding true/false positive.
        /// </summary>
        [Description("Specifies the threshold for deciding true/false positive.")]
        public float overlap_threshold
        {
            get { return m_fOverlapThreshold; }
            set { m_fOverlapThreshold = value; }
        }

        /// <summary>
        /// Specifies whether or not to consider the ground truth for evaluation.
        /// </summary>
        [Description("If true, also consider the difficult ground truth for evaluation.")]
        public bool evaulte_difficult_gt
        {
            get { return m_bEvaluateDifficultGt; }
            set { m_bEvaluateDifficultGt = value; }
        }

        /// <summary>
        /// Specifies the file which contains a list of names and sizes in the same order of the input database.  If provided, we scale the prediction and ground truth NormalizedBBox for evaluation.
        /// </summary>
        /// <remarks>
        /// This file is in the following format:
        ///    name height width
        ///    ...
        /// </remarks>
        [Description("Specifies the file which contains a list of names and sizes in the same order of the input database.  If provided, we scale the prediction and ground truth NormalizedBBox for evaluation.")]
        public string name_size_file
        {
            get { return m_strNameSizeFile; }
            set { m_strNameSizeFile = value; }
        }

        /// <summary>
        /// Specifies the resize parameter used in converting the NormalizedBBox to the original size.
        /// </summary>
        [Description("Specifies the resize parameter used in converting the NormalizedBBox to the original size.")]
        public ResizeParameter resize_param
        {
            get { return m_resizeParam; }
            set { m_resizeParam = value; }
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
            DetectionEvaluateParameter p = FromProto(proto);

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
            DetectionEvaluateParameter p = (DetectionEvaluateParameter)src;

            m_nNumClasses = p.m_nNumClasses;
            m_nBackgroundLabelId = p.m_nBackgroundLabelId;
            m_fOverlapThreshold = p.m_fOverlapThreshold;
            m_bEvaluateDifficultGt = p.m_bEvaluateDifficultGt;
            m_strNameSizeFile = p.m_strNameSizeFile;

            if (p.m_resizeParam == null)
                m_resizeParam = null;
            else
                m_resizeParam = p.resize_param.Clone();
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public override LayerParameterBase Clone()
        {
            DetectionEvaluateParameter p = new DetectionEvaluateParameter();
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

            rgChildren.Add(new RawProto("num_classes", m_nNumClasses.ToString()));
            rgChildren.Add(new RawProto("background_label_id", m_nBackgroundLabelId.ToString()));
            rgChildren.Add(new RawProto("overlap_threshold", m_fOverlapThreshold.ToString()));
            rgChildren.Add(new RawProto("evaluate_difficult_gt", m_bEvaluateDifficultGt.ToString()));
            rgChildren.Add(new RawProto("name_size_file", m_strNameSizeFile));

            if (m_resizeParam != null)
                rgChildren.Add(m_resizeParam.ToProto("resize_param"));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static DetectionEvaluateParameter FromProto(RawProto rp)
        {
            DetectionEvaluateParameter p = new DetectionEvaluateParameter();
            string strVal;

            if ((strVal = rp.FindValue("num_classes")) != null)
                p.num_classes = uint.Parse(strVal);

            if ((strVal = rp.FindValue("background_label_id")) != null)
                p.background_label_id = uint.Parse(strVal);

            if ((strVal = rp.FindValue("overlap_threshold")) != null)
                p.overlap_threshold = BaseParameter.parseFloat(strVal);

            if ((strVal = rp.FindValue("evaluate_difficult_gt")) != null)
                p.evaulte_difficult_gt = bool.Parse(strVal);

            p.name_size_file = rp.FindValue("name_size_file");

            RawProto rpResize = rp.FindChild("resize_param");
            if (rpResize != null)
                p.resize_param = ResizeParameter.FromProto(rpResize);

            return p;
        }
    }
}
