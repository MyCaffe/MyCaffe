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
    /// Specifies the parameters for the DistortionParameter used with SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class DistortionParameter : OptionalParameter
    {
        float m_fBrightnessProb = 0;
        float m_fBrightnessDelta = 0.0f;

        float m_fContrastProb = 0;
        float m_fContrastLower = 0.5f;
        float m_fContrastUpper = 1.5f;

        float m_fSaturationProb = 0;
        float m_fSaturationLower = 0.5f;
        float m_fSaturationUpper = 1.5f;

        float m_fRandomOrderProb = 0;

        long m_lRandomSeed = 0;
        bool m_bUseGpu = true;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bActive">Specifies whether or not the parameter is active or not.</param>
        public DistortionParameter(bool bActive) : base(bActive)
        {
        }

        /// <summary>
        /// Get/set the random seed (default = 0, only used when testing).
        /// </summary>
        /// <remarks>The random seed is only used when using the GPU version of distort image.</remarks>
        public long random_seed
        {
            get { return m_lRandomSeed; }
            set { m_lRandomSeed = value; }
        }

        /// <summary>
        /// Get/set whether or not to use the GPU for the distortion operations (default = true).
        /// </summary>
        public bool use_gpu
        {
            get { return m_bUseGpu; }
            set { m_bUseGpu = value; }
        }

        /// <summary>
        /// Get/set probability of adjusting the brightness (default = 0).
        /// </summary>
        [Description("Get/set probability of adjusting the brightness.")]
        public float brightness_prob
        {
            get { return m_fBrightnessProb; }
            set { m_fBrightnessProb = value; }
        }

        /// <summary>
        /// Get/set amount to add to the pixel values within [-delta,delta] (default = 0)
        /// </summary>
        [Description("Get/set amount to add to the pixel values within [-delta,delta]")]
        public float brightness_delta
        {
            get { return m_fBrightnessDelta; }
            set { m_fBrightnessDelta = value; }
        }

        /// <summary>
        /// Get/set probability of adjusting the contrast (default = 0).
        /// </summary>
        [Description("Get/set probability of adjusting the contrast.")]
        public float contrast_prob
        {
            get { return m_fContrastProb; }
            set { m_fContrastProb = value; }
        }

        /// <summary>
        /// Get/set lower bound for random contrast factor (default = 0.5).
        /// </summary>
        [Description("Get/set lower bound for random contrast factor.")]
        public float contrast_lower
        {
            get { return m_fContrastLower; }
            set { m_fContrastLower = value; }
        }

        /// <summary>
        /// Get/set upper bound for random contrast factor (default = 1.5).
        /// </summary>
        [Description("Get/set upper bound for random contrast factor.")]
        public float contrast_upper
        {
            get { return m_fContrastUpper; }
            set { m_fContrastUpper = value; }
        }

        /// <summary>
        /// Get/set probability of adjusting the saturation (default = 0).
        /// </summary>
        [Description("Get/set probability of adjusting the saturation.")]
        public float saturation_prob
        {
            get { return m_fSaturationProb; }
            set { m_fSaturationProb = value; }
        }

        /// <summary>
        /// Get/set lower bound for random saturation factor (default = 0.5).
        /// </summary>
        [Description("Get/set lower bound for random saturation factor.")]
        public float saturation_lower
        {
            get { return m_fSaturationLower; }
            set { m_fSaturationLower = value; }
        }

        /// <summary>
        /// Get/set upper bound for random saturation factor (default = 1.5).
        /// </summary>
        [Description("Get/set upper bound for random saturation factor.")]
        public float saturation_upper
        {
            get { return m_fSaturationUpper; }
            set { m_fSaturationUpper = value; }
        }

        /// <summary>
        /// Get/set the probability of randomly ordering the image channels (default = 0).
        /// </summary>
        [Description("Get/set the probability of randomly ordering the image channels.")]
        public float random_order_prob
        {
            get { return m_fRandomOrderProb; }
            set { m_fRandomOrderProb = value; }
        }

        /// <summary>
        /// Copy the object.
        /// </summary>
        /// <param name="src">The copy is placed in this parameter.</param>
        public override void Copy(OptionalParameter src)
        {
            base.Copy(src);

            if (src is DistortionParameter)
            {
                DistortionParameter p = (DistortionParameter)src;
                m_fBrightnessProb = p.brightness_prob;
                m_fBrightnessDelta = p.brightness_delta;

                m_fContrastProb = p.contrast_prob;
                m_fContrastLower = p.contrast_lower;
                m_fContrastUpper = p.contrast_upper;

                m_fSaturationProb = p.saturation_prob;
                m_fSaturationLower = p.saturation_lower;
                m_fSaturationUpper = p.saturation_upper;

                m_fRandomOrderProb = p.random_order_prob;

                m_lRandomSeed = p.m_lRandomSeed;
                m_bUseGpu = p.use_gpu;
            }
        }

        /// <summary>
        /// Return a clone of the object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public DistortionParameter Clone()
        {
            DistortionParameter p = new param.ssd.DistortionParameter(Active);
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Convert this object to a raw proto.
        /// </summary>
        /// <param name="strName">Specifies the name of the proto.</param>
        /// <returns>The new proto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("option");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase);
            rgChildren.Add(new RawProto("active", Active.ToString()));
            rgChildren.Add(new RawProto("brightness_prob", brightness_prob.ToString()));
            rgChildren.Add(new RawProto("brightness_delta", brightness_delta.ToString()));
            rgChildren.Add(new RawProto("contrast_prob", contrast_prob.ToString()));
            rgChildren.Add(new RawProto("contrast_lower", contrast_lower.ToString()));
            rgChildren.Add(new RawProto("contrast_upper", contrast_upper.ToString()));
            rgChildren.Add(new RawProto("saturation_prob", saturation_prob.ToString()));
            rgChildren.Add(new RawProto("saturation_lower", saturation_lower.ToString()));
            rgChildren.Add(new RawProto("saturation_upper", saturation_upper.ToString()));
            rgChildren.Add(new RawProto("random_order_prob", random_order_prob.ToString()));
            rgChildren.Add(new RawProto("use_gpu", use_gpu.ToString()));
            rgChildren.Add(new RawProto("random_seed", random_seed.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new DistortionParameter FromProto(RawProto rp)
        {
            DistortionParameter p = new DistortionParameter(false);
            string strVal;

            RawProto rpOption = rp.FindChild("option");
            if (rpOption != null)
                ((OptionalParameter)p).Copy(OptionalParameter.FromProto(rpOption));

            if ((strVal = rp.FindValue("active")) != null)
                p.Active = bool.Parse(strVal);

            if ((strVal = rp.FindValue("brightness_prob")) != null)
                p.brightness_prob = float.Parse(strVal);

            if ((strVal = rp.FindValue("brightness_delta")) != null)
                p.brightness_delta = float.Parse(strVal);

            if ((strVal = rp.FindValue("contrast_prob")) != null)
                p.contrast_prob = float.Parse(strVal);

            if ((strVal = rp.FindValue("contrast_lower")) != null)
                p.contrast_lower = float.Parse(strVal);

            if ((strVal = rp.FindValue("contrast_upper")) != null)
                p.contrast_upper = float.Parse(strVal);

            if ((strVal = rp.FindValue("saturation_prob")) != null)
                p.saturation_prob = float.Parse(strVal);

            if ((strVal = rp.FindValue("saturation_lower")) != null)
                p.saturation_lower = float.Parse(strVal);

            if ((strVal = rp.FindValue("saturation_upper")) != null)
                p.saturation_upper = float.Parse(strVal);

            if ((strVal = rp.FindValue("random_order_prob")) != null)
                p.random_order_prob = float.Parse(strVal);

            if ((strVal = rp.FindValue("use_gpu")) != null)
                p.use_gpu = bool.Parse(strVal);

            if ((strVal = rp.FindValue("random_seed")) != null)
                p.random_seed = long.Parse(strVal);

            return p;
        }
    }
}
