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
    /// Specifies the parameters for the NoiseParameter used with SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class NoiseParameter : OptionalParameter
    {
        float m_fProb = 0;
        bool m_bHistEq = false;
        bool m_bInverse = false;
        bool m_bDecolorize = false;
        bool m_bGaussBlur = false;
        float m_fJpeg = -1;
        bool m_bPosterize = false;
        bool m_bErode = false;
        bool m_bSaltPepper = false;
        SaltPepperParameter m_saltPepper = new SaltPepperParameter(true);
        bool m_bClahe = false;
        bool m_bConvertToHsv = false;
        bool m_bConvertToLab = false;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bActive">Specifies whether or not the parameter is active or not.</param>
        public NoiseParameter(bool bActive) : base(bActive)
        {
        }

        /// <summary>
        /// Get/set probability of using this resize policy.
        /// </summary>
        public float prob
        {
            get { return m_fProb; }
            set { m_fProb = value; }
        }

        /// <summary>
        /// Get/set histogram equalized.
        /// </summary>
        public bool hist_eq
        {
            get { return m_bHistEq; }
            set { m_bHistEq = value; }
        }

        /// <summary>
        /// Get/set color inversion.
        /// </summary>
        public bool inverse
        {
            get { return m_bInverse; }
            set { m_bInverse = value; }
        }

        /// <summary>
        /// Get/set grayscale.
        /// </summary>
        public bool decolorize
        {
            get { return m_bDecolorize; }
            set { m_bDecolorize = value; }
        }

        /// <summary>
        /// Get/set gaussian blur.
        /// </summary>
        public bool gauss_blur
        {
            get { return m_bGaussBlur; }
            set { m_bGaussBlur = value; }
        }

        /// <summary>
        /// Get/set jpeg quality.
        /// </summary>
        public float jpeg
        {
            get { return m_fJpeg; }
            set { m_fJpeg = value; }
        }

        /// <summary>
        /// Get/set posterization.
        /// </summary>
        public bool posterize
        {
            get { return m_bPosterize; }
            set { m_bPosterize = value; }
        }

        /// <summary>
        /// Get/set erosion.
        /// </summary>
        public bool erode
        {
            get { return m_bErode; }
            set { m_bErode = value; }
        }

        /// <summary>
        /// Get/set salt-n-pepper noise.
        /// </summary>
        public bool saltpepper
        {
            get { return m_bSaltPepper; }
            set { m_bSaltPepper = value; }
        }

        /// <summary>
        /// Get/set the salt-n-pepper parameter.
        /// </summary>
        public SaltPepperParameter saltpepper_param
        {
            get { return m_saltPepper; }
            set { m_saltPepper = value; }
        }

        /// <summary>
        /// Get/set the local histogram equalization.
        /// </summary>
        public bool clahe
        {
            get { return m_bClahe; }
            set { m_bClahe = value; }
        }

        /// <summary>
        /// Get/set color space conversion to hsv.
        /// </summary>
        public bool convert_to_hsv
        {
            get { return m_bConvertToHsv; }
            set { m_bConvertToHsv = value; }
        }

        /// <summary>
        /// Get/set color space convertion to lab.
        /// </summary>
        public bool convert_to_lab
        {
            get { return m_bConvertToLab; }
            set { m_bConvertToLab = value; }
        }

        /// <summary>
        /// Copy the object.
        /// </summary>
        /// <param name="src">The copy is placed in this parameter.</param>
        public override void Copy(OptionalParameter src)
        {
            base.Copy(src);

            if (src is NoiseParameter)
            {
                NoiseParameter p = (NoiseParameter)src;
                m_fProb = p.prob;
            }
        }

        /// <summary>
        /// Return a clone of the object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public NoiseParameter Clone()
        {
            NoiseParameter p = new param.ssd.NoiseParameter(Active);
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
            rgChildren.Add(new RawProto("prob", prob.ToString()));
            rgChildren.Add(new RawProto("hist_eq", hist_eq.ToString()));
            rgChildren.Add(new RawProto("inverse", inverse.ToString()));
            rgChildren.Add(new RawProto("decolorize", decolorize.ToString()));
            rgChildren.Add(new RawProto("gauss_blur", gauss_blur.ToString()));
            rgChildren.Add(new RawProto("jpeg", jpeg.ToString()));
            rgChildren.Add(new RawProto("posterize", posterize.ToString()));
            rgChildren.Add(new RawProto("erode", erode.ToString()));
            rgChildren.Add(new RawProto("saltpepper", saltpepper.ToString()));
            rgChildren.Add(m_saltPepper.ToProto("saltpepper_param"));
            rgChildren.Add(new RawProto("clahe", clahe.ToString()));
            rgChildren.Add(new RawProto("convert_to_hsv", convert_to_hsv.ToString()));
            rgChildren.Add(new RawProto("convert_to_lab", convert_to_lab.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new NoiseParameter FromProto(RawProto rp)
        {
            NoiseParameter p = new NoiseParameter(false);
            string strVal;

            RawProto rpOption = rp.FindChild("option");
            if (rpOption != null)
                ((OptionalParameter)p).Copy(OptionalParameter.FromProto(rpOption));

            if ((strVal = rp.FindValue("prob")) != null)
                p.prob = float.Parse(strVal);

            if ((strVal = rp.FindValue("hist_eq")) != null)
                p.hist_eq = bool.Parse(strVal);

            if ((strVal = rp.FindValue("inverse")) != null)
                p.inverse = bool.Parse(strVal);

            if ((strVal = rp.FindValue("decolorize")) != null)
                p.decolorize = bool.Parse(strVal);

            if ((strVal = rp.FindValue("gauss_blur")) != null)
                p.gauss_blur = bool.Parse(strVal);

            if ((strVal = rp.FindValue("jpeg")) != null)
                p.jpeg = float.Parse(strVal);

            if ((strVal = rp.FindValue("posterize")) != null)
                p.posterize = bool.Parse(strVal);

            if ((strVal = rp.FindValue("erode")) != null)
                p.erode = bool.Parse(strVal);

            if ((strVal = rp.FindValue("saltpepper")) != null)
                p.saltpepper = bool.Parse(strVal);

            RawProto rp1 = rp.FindChild("saltpepper_param");
            if (rp1 != null)
                p.saltpepper_param = SaltPepperParameter.FromProto(rp1);

            if ((strVal = rp.FindValue("clahe")) != null)
                p.clahe = bool.Parse(strVal);

            if ((strVal = rp.FindValue("convert_to_hsv")) != null)
                p.convert_to_hsv = bool.Parse(strVal);

            if ((strVal = rp.FindValue("convert_to_lab")) != null)
                p.convert_to_lab = bool.Parse(strVal);

            return p;
        }
    }
}
