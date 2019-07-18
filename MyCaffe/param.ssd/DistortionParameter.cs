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
    /// Specifies the parameters for the DistortionParameter.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class DistortionParameter
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

        /// <summary>
        /// The constructor.
        /// </summary>
        public DistortionParameter()
        {
        }

        /// <summary>
        /// Get/set probability of adjusting the brightness.
        /// </summary>
        public float brightness_prob
        {
            get { return m_fBrightnessProb; }
            set { m_fBrightnessProb = value; }
        }

        /// <summary>
        /// Get/set amount to add to the pixel values within [-delta,delta]
        /// </summary>
        public float brightness_delta
        {
            get { return m_fBrightnessDelta; }
            set { m_fBrightnessDelta = value; }
        }

        /// <summary>
        /// Get/set probability of adjusting the contrast.
        /// </summary>
        public float contrast_prob
        {
            get { return m_fContrastProb; }
            set { m_fContrastProb = value; }
        }

        /// <summary>
        /// Get/set lower bound for random contrast factor.
        /// </summary>
        public float contrast_lower
        {
            get { return m_fContrastLower; }
            set { m_fContrastLower = value; }
        }

        /// <summary>
        /// Get/set upper bound for random contrast factor.
        /// </summary>
        public float contrast_upper
        {
            get { return m_fContrastUpper; }
            set { m_fContrastUpper = value; }
        }

        /// <summary>
        /// Get/set probability of adjusting the saturation.
        /// </summary>
        public float saturation_prob
        {
            get { return m_fSaturationProb; }
            set { m_fSaturationProb = value; }
        }

        /// <summary>
        /// Get/set lower bound for random saturation factor.
        /// </summary>
        public float saturation_lower
        {
            get { return m_fSaturationLower; }
            set { m_fSaturationLower = value; }
        }

        /// <summary>
        /// Get/set upper bound for random saturation factor.
        /// </summary>
        public float saturation_upper
        {
            get { return m_fSaturationUpper; }
            set { m_fSaturationUpper = value; }
        }

        /// <summary>
        /// Get/set the probability of randomly ordering the image channels.
        /// </summary>
        public float random_order_prob
        {
            get { return m_fRandomOrderProb; }
            set { m_fRandomOrderProb = value; }
        }

        /// <summary>
        /// Copy the object.
        /// </summary>
        /// <param name="p">The copy is placed in this parameter.</param>
        public void Copy(DistortionParameter src)
        {
            m_fBrightnessProb = src.brightness_prob;
            m_fBrightnessDelta = src.brightness_delta;

            m_fContrastProb = src.contrast_prob;
            m_fContrastLower = src.contrast_lower;
            m_fContrastUpper = src.contrast_upper;

            m_fSaturationProb = src.saturation_prob;
            m_fSaturationLower = src.saturation_lower;
            m_fSaturationUpper = src.saturation_upper;

            m_fRandomOrderProb = src.random_order_prob;
        }

        /// <summary>
        /// Return a clone of the object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public DistortionParameter Clone()
        {
            DistortionParameter p = new param.ssd.DistortionParameter();
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Convert this object to a raw proto.
        /// </summary>
        /// <param name="strName">Specifies the name of the proto.</param>
        /// <returns>The new proto is returned.</returns>
        public RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(new RawProto("brightness_prob", brightness_prob.ToString()));
            rgChildren.Add(new RawProto("brightness_delta", brightness_delta.ToString()));
            rgChildren.Add(new RawProto("contrast_prob", contrast_prob.ToString()));
            rgChildren.Add(new RawProto("contrast_lower", contrast_lower.ToString()));
            rgChildren.Add(new RawProto("contrast_upper", contrast_upper.ToString()));
            rgChildren.Add(new RawProto("saturation_prob", saturation_prob.ToString()));
            rgChildren.Add(new RawProto("saturation_lower", saturation_lower.ToString()));
            rgChildren.Add(new RawProto("saturation_upper", saturation_upper.ToString()));
            rgChildren.Add(new RawProto("random_order_prob", random_order_prob.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static DistortionParameter FromProto(RawProto rp)
        {
            DistortionParameter p = new DistortionParameter();
            string strVal;

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

            return p;
        }
    }
}
