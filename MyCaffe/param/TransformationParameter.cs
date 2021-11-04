using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param.ssd;

namespace MyCaffe.param
{
    /// <summary>
    /// Stores parameters used to apply transformation 
    /// to the data layer's data.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class TransformationParameter : LayerParameterBase 
    {
        double m_dfScale = 1;
        SCALE_OPERATOR? m_scaleOperator = null;
        bool m_bMirror = false;
        uint m_nCropSize = 0;
        bool m_bUseImageDbMean = false;
        List<double> m_rgMeanValue = new List<double>();
        bool m_bForceColor = false;
        bool m_bForceGray = false;
        double m_dfForcedPositiveRangeMax = 0.0;
        int? m_nRandomSeed = null;
        string m_strMeanFile = null;
        COLOR_ORDER m_colorOrder = COLOR_ORDER.RGB;
        ResizeParameter m_resize = new ResizeParameter(false);
        NoiseParameter m_noise = new NoiseParameter(false);
        DistortionParameter m_distortion = new DistortionParameter(false);
        ExpansionParameter m_expansion = new ExpansionParameter(false);
        EmitConstraint m_emitConstraint = new EmitConstraint(false);
        MaskParameter m_mask = new MaskParameter(false);
        DataLabelMappingParameter m_labelMapping = new DataLabelMappingParameter(false);

        /// <summary>
        /// Defines the type of scale operator to use (if any).
        /// </summary>
        public enum SCALE_OPERATOR
        {
            /// <summary>
            /// Specifies to not scale the data.
            /// </summary>
            NONE,
            /// <summary>
            /// Specifies to use the multiplication operator where the scale is multiplied by the result of the data-mean.
            /// </summary>
            MUL,
            /// <summary>
            /// Specifies to use the power operator where the data-mean is raised to the power of the scale value.
            /// </summary>
            POW
        }

        /// <summary>
        /// Defines the color ordering used to tranform the input data.
        /// </summary>
        public enum COLOR_ORDER
        {
            /// <summary>
            /// Orders the channels by 'R'ed, 'G'reen, then 'B'lue.
            /// </summary>
            RGB = 0,
            /// <summary>
            /// Orders the channels by 'B'lue, 'G'reen, then 'R'ed.  This ordering is typically used by Native C++ Caffe.
            /// </summary>
            BGR = 1
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        public TransformationParameter()
        {
        }

        /// <summary>
        /// Specifies whether or not to fit the data into a forced range 
        /// of [0, forced_positive_range_max].  
        /// </summary>
        [Category("Data Pre-Processing"), Description("When specified (value > 0), the data values are fit into the positive range [0, forced_positive_range_max].  When set to 0 no range fitting is performed.")]
        public double forced_positive_range_max
        {
            get { return m_dfForcedPositiveRangeMax; }
            set { m_dfForcedPositiveRangeMax = value; }
        }

        /// <summary>
        /// For data pre-processing, we can do simple scaling and subtracting the
        /// data mean, if provided.  Note that the mean subtraction is always carried
        /// out before scaling.
        /// </summary>
        [Category("Data Pre-Processing"), Description("This value is used for simple scaling and subtracting the data mean if provided.  Note that the mean subtraction is always carried out before scaling.")]
        public double scale
        {
            get { return m_dfScale; }
            set { m_dfScale = value; }
        }

        /// <summary>
        /// Get/set the scale operator used to apply the scale value to the data-mean or data result.
        /// </summary>
        public SCALE_OPERATOR? scale_operator
        {
            get { return m_scaleOperator; }
            set { m_scaleOperator = value; }
        }

        /// <summary>
        /// Specify if we want to randomly mirror the data.
        /// </summary>
        [Category("Data Pre-Processing"), Description("Specify if we want to randomly mirror the data.")]
        public bool mirror
        {
            get { return m_bMirror; }
            set { m_bMirror = value; }
        }

        /// <summary>
        /// Specify if we would like to randomly crop an image.
        /// </summary>
        [Category("Data Pre-Processing"), Description("Specify if we want to randomly crop the image.  A value of 0 disables the croping.")]
        public uint crop_size
        {
            get { return m_nCropSize; }
            set { m_nCropSize = value; }
        }

        /// <summary>
        /// Specifies whether to subtract the mean image from the image database,
        /// subtract the mean values, or neither and do no mean subtraction.
        /// </summary>
        [Category("Data Mean"), Description("Specifies whether or not to use the image mean for the data source from the image database.  When true, the mean image is subtracted from the current image.")]
        public bool use_imagedb_mean
        {
            get { return m_bUseImageDbMean; }
            set { m_bUseImageDbMean = value; }
        }

        /// <summary>
        /// If specified can be repeated once (would subtract it from all the channels
        /// or can be repeated the same number of times as channels
        /// (would subtract them from the corresponding channel).
        /// </summary>
        /// <remarks>
        /// So for example if there are 3 channels, mean_value could have 3 values,
        /// one for each channel -- or just one value which would be applied to
        /// all channels.
        /// </remarks>
        [Category("Data Mean"), Description("If specified can be repeated once (will subtract the value from all of teh channels, or can be repeated the same number of times as channels which will then subtract each corresponding value from each associated channel)  So for example if there are 3 channels, mean values could have 3 values (one for each channel) or just one value that is then applied to all channels.")]
        public List<double> mean_value
        {
            get { return m_rgMeanValue; }
            set { m_rgMeanValue = value; }
        }

        /// <summary>
        /// Force the decoded image to have 3 color channels.
        /// </summary>
        [Category("Data Color"), Description("When true, force the decoded image to have 3 color channels.")]
        public bool force_color
        {
            get { return m_bForceColor; }
            set { m_bForceColor = value; }
        }

        /// <summary>
        /// Force the decoded image to have 1 color channel.
        /// </summary>
        [Category("Data Color"), Description("When true, force the decoded image to have 1 color channel.")]
        public bool force_gray
        {
            get { return m_bForceGray; }
            set { m_bForceGray = value; }
        }

        /// <summary>
        /// Only used during testing.
        /// </summary>
        [Category("Testing"), Description("Only used during testing.")]
        public int? random_seed
        {
            get { return m_nRandomSeed; }
            set { m_nRandomSeed = value; }
        }

        /// <summary>
        /// Specifies the path to file containing the image mean in the proto buffer format of a BlobProto.
        /// </summary>
        /// <remarks>
        /// The mean file is used when specified and the 'use_imagedb_mean' = <i>true</i>.  If the 'use_imagedb_mean' = <i>true</i> and
        /// the mean file is not set, the Caffe Image Database is queried for the calculated mean image.
        /// </remarks>
        [Category("Data Mean"), Description("The mean file is used when specified and 'use_imagedb_mean' = true.  If the 'use_imagedb_mean' is true and the 'mean_file' is not set, then the image database is queried for the mean image to use.")]
        public string mean_file
        {
            get { return m_strMeanFile; }
            set { m_strMeanFile = value; }
        }

        /// <summary>
        /// Specifies the color ordering to use.  Native Caffe models often uses COLOR_ORDER.BGR, whereas MyCaffe datasets often
        /// uses the COLOR_ORDER.RGB ordering.
        /// </summary>
        [Category("Data Color"), Description("Specifies the color ordering to use.  Native Caffe models expect the BGR color ordering.")]
        public COLOR_ORDER color_order
        {
            get { return m_colorOrder; }
            set { m_colorOrder = value; }
        }

        /// <summary>
        /// Optionally, specifies the resize policy, otherwise this is <i>null</i>.
        /// </summary>
        /// <remarks>
        /// Currently, this parameter is only used by the AnnotatedDataLayer.
        /// </remarks>
        [Category("Image"), Description("When active, used as the resize policy for altering image data.")]
        public ResizeParameter resize_param
        {
            get { return m_resize; }
            set { m_resize = value; }
        }

        /// <summary>
        /// Optionally, specifies the noise policy, otherwise this is <i>null</i>.
        /// </summary>
        /// <remarks>
        /// Currently, this parameter is only used by the DataLayer.
        /// </remarks>
        [Category("Image"), Description("When active, used as the noise policy for altering image data.")]
        public NoiseParameter noise_param
        {
            get { return m_noise; }
            set { m_noise = value; }
        }

        /// <summary>
        /// Optionally, specifies the distortion policy, otherwise this is <i>null</i>.
        /// </summary>
        /// <remarks>
        /// Currently, this parameter is only used by the AnnotatedDataLayer.
        /// </remarks>
        [Category("Image"), Description("When active, used as the distortion policy for altering image data.")]
        public DistortionParameter distortion_param
        {
            get { return m_distortion; }
            set { m_distortion = value; }
        }

        /// <summary>
        /// Optionally, specifies the expansion policy, otherwise this is <i>null</i>.
        /// </summary>
        /// <remarks>
        /// Currently, this parameter is only used by the AnnotatedDataLayer.
        /// </remarks>
        [Category("Image"), Description("When active, used as the expansion policy for altering image data.")]
        public ExpansionParameter expansion_param
        {
            get { return m_expansion; }
            set { m_expansion = value; }
        }

        /// <summary>
        /// Optionally, specifies the emit constraint on emitting annotation after transformation, otherwise this is <i>null</i>.
        /// </summary>
        /// <remarks>
        /// Currently, this parameter is only used by the AnnotatedDataLayer.
        /// </remarks>
        [Category("Image"), Description("When active, used as the emit constratin for emitting annotation after transformation.")]
        public EmitConstraint emit_constraint
        {
            get { return m_emitConstraint; }
            set { m_emitConstraint = value; }
        }

        /// <summary>
        /// Optionally, specifies the image mask which defines the boundary area that is set to black on the image thus masking that area out.
        /// </summary>
        /// <remarks>
        /// The mask is applied last, after all other alterations are made.
        /// 
        /// Currently, this parameter is only used by the DataLayer.
        /// </remarks>
        [Category("Image"), Description("When active, used to mask portions of the image (set to Black) as defined by the boundary of the mask.  The mask is applied after all other alterations.")]
        public MaskParameter mask_param
        {
            get { return m_mask; }
            set { m_mask = value; }
        }

        /// <summary>
        /// Optionally, specifies the label mapping which defines how to map lables when calling the DataTransformer.TransformLabel method.
        /// </summary>
        /// <remarks>
        /// Currently, this parameter is only used by the DataLayer.
        /// </remarks>
        public DataLabelMappingParameter label_mapping
        {
            get { return m_labelMapping; }
            set { m_labelMapping = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            TransformationParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            TransformationParameter p = (TransformationParameter)src;
            
            m_bUseImageDbMean = p.m_bUseImageDbMean;
            m_bForceColor = p.m_bForceColor;
            m_bForceGray = p.m_bForceGray;
            m_bMirror = p.m_bMirror;
            m_dfScale = p.m_dfScale;
            m_scaleOperator = p.m_scaleOperator;
            m_nCropSize = p.m_nCropSize;
            m_rgMeanValue = Utility.Clone<double>(p.m_rgMeanValue);
            m_dfForcedPositiveRangeMax = p.m_dfForcedPositiveRangeMax;
            m_nRandomSeed = p.m_nRandomSeed;
            m_strMeanFile = p.m_strMeanFile;
            m_colorOrder = p.m_colorOrder;

            m_resize = (p.resize_param == null) ? null : p.resize_param.Clone();
            m_noise = (p.noise_param == null) ? null : p.noise_param.Clone();
            m_distortion = (p.distortion_param == null) ? null : p.distortion_param.Clone();
            m_expansion = (p.expansion_param == null) ? null : p.expansion_param.Clone();
            m_emitConstraint = (p.emit_constraint == null) ? null : p.emit_constraint.Clone();

            if (p.mask_param != null)
                m_mask = p.mask_param.Clone();

            if (p.label_mapping != null)
                m_labelMapping = p.label_mapping.Clone();
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            TransformationParameter p = new TransformationParameter();
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

            if (scale != 1.0)
                rgChildren.Add("scale", scale.ToString());

            if (scale_operator.HasValue)
                rgChildren.Add("scale_operator", scale_operator.Value.ToString());

            if (mirror != false)
                rgChildren.Add("mirror", mirror.ToString());

            if (crop_size != 0)
                rgChildren.Add("crop_size", crop_size.ToString());

            if (use_imagedb_mean != false)
                rgChildren.Add("use_imagedb_mean", use_imagedb_mean.ToString());

            rgChildren.Add<double>("mean_value", mean_value);

            if (force_color != false)
                rgChildren.Add("force_color", force_color.ToString());

            if (force_gray != false)
                rgChildren.Add("force_gray", force_gray.ToString());

            if (forced_positive_range_max != 0)
                rgChildren.Add("force_positive_range_max", forced_positive_range_max.ToString());

            if (mean_file != null && mean_file.Length > 0)
                rgChildren.Add("mean_file", mean_file);

            rgChildren.Add("color_order", m_colorOrder.ToString());

            if (m_resize != null)
                rgChildren.Add(m_resize.ToProto("resize_param"));

            if (m_noise != null)
                rgChildren.Add(m_noise.ToProto("noise_param"));

            if (m_distortion != null)
                rgChildren.Add(m_distortion.ToProto("distortion_param"));

            if (m_expansion != null)
                rgChildren.Add(m_expansion.ToProto("expansion_param"));

            if (m_emitConstraint != null)
                rgChildren.Add(m_emitConstraint.ToProto("emit_constraint"));

            if (m_mask != null)
                rgChildren.Add(m_mask.ToProto("mask_param"));

            if (m_labelMapping != null)
                rgChildren.Add(m_labelMapping.ToProto("label_mapping"));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static TransformationParameter FromProto(RawProto rp)
        {
            string strVal;
            TransformationParameter p = new TransformationParameter();

            if ((strVal = rp.FindValue("scale")) != null)
                p.scale = ParseDouble(strVal);

            if ((strVal = rp.FindValue("scale_operator")) != null)
            {
                if (strVal == SCALE_OPERATOR.MUL.ToString())
                    p.scale_operator = SCALE_OPERATOR.MUL;
                else if (strVal == SCALE_OPERATOR.POW.ToString())
                    p.scale_operator = SCALE_OPERATOR.POW;
                else
                    p.scale_operator = SCALE_OPERATOR.NONE;
            }
            else
            {
                p.scale_operator = null;
            }

            if ((strVal = rp.FindValue("mirror")) != null)
                p.mirror = bool.Parse(strVal);

            if ((strVal = rp.FindValue("crop_size")) != null)
                p.crop_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("use_image_mean")) != null ||
                (strVal = rp.FindValue("use_imagedb_mean")) != null)
                p.use_imagedb_mean = bool.Parse(strVal);

            if ((strVal = rp.FindValue("mean_file")) != null)
                p.use_imagedb_mean = true;

            p.mean_value = rp.FindArray<double>("mean_value");

            if ((strVal = rp.FindValue("force_color")) != null)
                p.force_color = bool.Parse(strVal);

            if ((strVal = rp.FindValue("force_gray")) != null)
                p.force_gray = bool.Parse(strVal);

            if ((strVal = rp.FindValue("force_positive_range_max")) != null)
                p.forced_positive_range_max = ParseDouble(strVal);

            if ((strVal = rp.FindValue("mean_file")) != null)
                p.mean_file = strVal;

            if ((strVal = rp.FindValue("color_order")) != null)
            {
                if (strVal == COLOR_ORDER.BGR.ToString())
                    p.color_order = COLOR_ORDER.BGR;
                else
                    p.color_order = COLOR_ORDER.RGB;
            }

            RawProto rpResize = rp.FindChild("resize_param");
            if (rpResize != null)
                p.resize_param = ResizeParameter.FromProto(rpResize);
            else
                p.resize_param = null;

            RawProto rpNoise = rp.FindChild("noise_param");
            if (rpNoise != null)
                p.noise_param = NoiseParameter.FromProto(rpNoise);
            else
                p.noise_param = null;

            RawProto rpDistort = rp.FindChild("distortion_param");
            if (rpDistort != null)
                p.distortion_param = DistortionParameter.FromProto(rpDistort);

            RawProto rpExpand = rp.FindChild("expansion_param");
            if (rpExpand != null)
                p.expansion_param = ExpansionParameter.FromProto(rpExpand);
            else
                p.expansion_param = null;

            RawProto rpEmitCon = rp.FindChild("emit_constraint");
            if (rpEmitCon != null)
                p.emit_constraint = EmitConstraint.FromProto(rpEmitCon);
            else
                p.emit_constraint = null;

            RawProto rpMask = rp.FindChild("mask_param");
            if (rpMask != null)
                p.mask_param = MaskParameter.FromProto(rpMask);

            RawProto rpLabelMapping = rp.FindChild("label_mapping");
            if (rpLabelMapping != null)
                p.label_mapping = DataLabelMappingParameter.FromProto(rpLabelMapping);

            return p;
        }
    }
}
