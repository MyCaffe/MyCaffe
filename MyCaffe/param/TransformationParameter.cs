using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param
{
    /// <summary>
    /// Stores parameters used to apply transformation 
    /// to the data layer's data.
    /// </summary>
    public class TransformationParameter : LayerParameterBase 
    {
        double m_dfScale = 1;
        bool m_bMirror = false;
        uint m_nCropSize = 0;
        bool m_bUseImageMean = false;
        List<double> m_rgMeanValue = new List<double>();
        bool m_bForceColor = false;
        bool m_bForceGray = false;
        double m_dfForcedPositiveRangeMax = 0.0;
        int? m_nRandomSeed = null;
        string m_strMeanFile = null;
        COLOR_ORDER m_colorOrder = COLOR_ORDER.RGB;

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
        public bool use_image_mean
        {
            get { return m_bUseImageMean; }
            set { m_bUseImageMean = value; }
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
        public int? random_seed
        {
            get { return m_nRandomSeed; }
            set { m_nRandomSeed = value; }
        }

        /// <summary>
        /// Specifies the path to file containing the image mean in the proto buffer format of a BlobProto.
        /// </summary>
        /// <remarks>
        /// The mean file is used when specified and the 'use_image_mean' = <i>true</i>.  If the 'use_image_mean' = <i>true</i> and
        /// the mean file is not set, the Caffe Image Database is queried for the calculated mean image.
        /// </remarks>
        public string mean_file
        {
            get { return m_strMeanFile; }
            set { m_strMeanFile = value; }
        }

        /// <summary>
        /// Specifies the color ordering to use.  Native Caffe models often uses COLOR_ORDER.BGR, whereas MyCaffe datasets often
        /// uses the COLOR_ORDER.RGB ordering.
        /// </summary>
        public COLOR_ORDER color_order
        {
            get { return m_colorOrder; }
            set { m_colorOrder = value; }
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
            
            m_bUseImageMean = p.m_bUseImageMean;
            m_bForceColor = p.m_bForceColor;
            m_bForceGray = p.m_bForceGray;
            m_bMirror = p.m_bMirror;
            m_dfScale = p.m_dfScale;
            m_nCropSize = p.m_nCropSize;
            m_rgMeanValue = Utility.Clone<double>(p.m_rgMeanValue);
            m_dfForcedPositiveRangeMax = p.m_dfForcedPositiveRangeMax;
            m_nRandomSeed = p.m_nRandomSeed;
            m_strMeanFile = p.m_strMeanFile;
            m_colorOrder = p.m_colorOrder;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            TransformationParameter p = new TransformationParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (scale != 1.0)
                rgChildren.Add("scale", scale.ToString());

            if (mirror != false)
                rgChildren.Add("mirror", mirror.ToString());

            if (crop_size != 0)
                rgChildren.Add("crop_size", crop_size.ToString());

            if (use_image_mean != false)
                rgChildren.Add("use_image_mean", use_image_mean.ToString());

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
                p.scale = double.Parse(strVal);

            if ((strVal = rp.FindValue("mirror")) != null)
                p.mirror = bool.Parse(strVal);

            if ((strVal = rp.FindValue("crop_size")) != null)
                p.crop_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("use_image_mean")) != null)
                p.use_image_mean = bool.Parse(strVal);

            if ((strVal = rp.FindValue("mean_file")) != null)
                p.use_image_mean = true;

            p.mean_value = rp.FindArray<double>("mean_value");

            if ((strVal = rp.FindValue("force_color")) != null)
                p.force_color = bool.Parse(strVal);

            if ((strVal = rp.FindValue("force_gray")) != null)
                p.force_gray = bool.Parse(strVal);

            if ((strVal = rp.FindValue("force_positive_range_max")) != null)
                p.forced_positive_range_max = double.Parse(strVal);

            if ((strVal = rp.FindValue("mean_file")) != null)
                p.mean_file = strVal;

            if ((strVal = rp.FindValue("color_order")) != null)
            {
                if (strVal == COLOR_ORDER.BGR.ToString())
                    p.color_order = COLOR_ORDER.BGR;
                else
                    p.color_order = COLOR_ORDER.RGB;
            }

            return p;
        }
    }
}
