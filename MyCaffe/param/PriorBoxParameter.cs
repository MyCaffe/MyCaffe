using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the PriorBoxParameter.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class PriorBoxParameter
    {
        List<float> m_rgMinSize = new List<float>();
        List<float> m_rgMaxSize = new List<float>();
        List<float> m_rgAspectRatio = new List<float>();
        bool m_bFlip = true;
        bool m_bClip = false;
        List<float> m_rgVariance = new List<float>();
        uint? m_nImgSize = null;
        uint? m_nImgH = null;
        uint? m_nImgW = null;
        float? m_fStep = null;
        float? m_fStepH = null;
        float? m_fStepW = null;
        float m_fOffset = 0.5f;

        /// <summary>
        /// Defines the encode/decode type.
        /// </summary>
        public enum CodeType
        {
            /// <summary>
            /// Encode the corner.
            /// </summary>
            CORNER = 1,
            /// <summary>
            /// Encode the center size.
            /// </summary>
            CENTER_SIZE = 2,
            /// <summary>
            /// Encode the corner size.
            /// </summary>
            CORNER_SIZE = 3
        }

        /// <summary>
        /// Convert a string into a CodeType.
        /// </summary>
        /// <param name="str">Specifies the string.</param>
        /// <returns>The associated CodeType is returned.</returns>
        public static CodeType CodeTypeFromString(string str)
        {
            switch (str)
            {
                case "CORNER":
                    return CodeType.CORNER;

                case "CENTER_SIZE":
                    return CodeType.CENTER_SIZE;

                case "CORNER_SIZE":
                    return CodeType.CORNER_SIZE;

                default:
                    throw new Exception("Unknown CodeType '" + str + "'!");
            }
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        public PriorBoxParameter()
        {
        }

        /// <summary>
        /// Specifies the minimum box size (in pixels) and is required!
        /// </summary>
        public List<float> min_size
        {
            get { return m_rgMinSize; }
            set { m_rgMinSize = value; }
        }

        /// <summary>
        /// Specifies the maximum box size (in pixels) and is required!
        /// </summary>
        public List<float> max_size
        {
            get { return m_rgMaxSize; }
            set { m_rgMaxSize = value; }
        }

        /// <summary>
        /// Specifies various aspect ratios.  Duplicate ratios are ignored.
        /// If none are provided, a default ratio of 1 is used.
        /// </summary>
        public List<float> aspect_ratio
        {
            get { return m_rgAspectRatio; }
            set { m_rgAspectRatio = value; }
        }

        /// <summary>
        /// Specifies whether or not to flip each aspect ratio.
        /// For example, if there is an aspect ratio 'r'
        /// we will generate aspect ratio '1.0/r' as well.
        /// </summary>
        public bool flip
        {
            get { return m_bFlip; }
            set { m_bFlip = value; }
        }

        /// <summary>
        /// Specifies whether or not to clip the prior so that it is within [0,1].
        /// </summary>
        public bool clip
        {
            get { return m_bClip; }
            set { m_bClip = value; }
        }

        /// <summary>
        /// Specifies the variance for adjusting the prior boxes.
        /// </summary>
        public List<float> variance
        {
            get { return m_rgVariance; }
            set { m_rgVariance = value; }
        }

        /// <summary>
        /// Specifies the image size.  By default we calculate
        /// the img_height, img_width, step_x and step_y based
        /// on bottom[0] (feat) and bottom[1] (img).  Unless these
        /// values are explicitly provided here.
        /// 
        /// Either the img_h and img_w are used or the img_size,
        /// but not both.
        /// </summary>
        public uint? img_size
        {
            get { return m_nImgSize; }
            set { m_nImgSize = value; }
        }

        /// <summary>
        /// Specifies the image height.  By default we calculate
        /// the img_height, img_width, step_x and step_y based
        /// on bottom[0] (feat) and bottom[1] (img).  Unless these
        /// values are explicitly provided here.
        /// 
        /// Either the img_h and img_w are used or the img_size,
        /// but not both.
        /// </summary>
        public uint? img_h
        {
            get { return m_nImgH; }
            set { m_nImgH = value; }
        }

        /// <summary>
        /// Specifies the image width.  By default we calculate
        /// the img_height, img_width, step_x and step_y based
        /// on bottom[0] (feat) and bottom[1] (img).  Unless these
        /// values are explicitly provided here.
        /// 
        /// Either the img_h and img_w are used or the img_size,
        /// but not both.
        /// </summary>
        public uint? img_w
        {
            get { return m_nImgW; }
            set { m_nImgW = value; }
        }

        /// <summary>
        /// Specifies the excplicit step size to use.
        /// </summary>
        public float? step
        {
            get { return m_fStep; }
            set { m_fStep = value; }
        }

        /// <summary>
        /// Specifies the explicit step size to use along height.
        /// 
        /// Either the step_h and step_w are used or the step,
        /// but not both.
        /// </summary>
        public float? step_h
        {
            get { return m_fStepH; }
            set { m_fStepH = value; }
        }

        /// <summary>
        /// Specifies the explicit step size to use along width.
        /// 
        /// Either the step_h and step_w are used or the step,
        /// but not both.
        /// </summary>
        public float? step_w
        {
            get { return m_fStepW; }
            set { m_fStepW = value; }
        }

        /// <summary>
        /// Specifies the offset to the top left corner of each cell.
        /// </summary>
        public float offset
        {
            get { return m_fOffset; }
            set { m_fOffset = value; }
        }

        /// <summary>
        /// Copy the object.
        /// </summary>
        /// <param name="p">The copy is placed in this parameter.</param>
        public void Copy(PriorBoxParameter src)
        {
            m_rgMinSize = Utility.Clone<float>(src.min_size);
            m_rgMaxSize = Utility.Clone<float>(src.max_size);
            m_rgAspectRatio = Utility.Clone<float>(src.aspect_ratio);
            m_bFlip = src.flip;
            m_bClip = src.clip;
            m_rgVariance = Utility.Clone<float>(src.variance);
            m_nImgSize = src.img_size;
            m_nImgH = src.img_h;
            m_nImgW = src.img_w;
            m_fStep = src.step;
            m_fStepH = src.step_h;
            m_fStepW = src.step_w;
            m_fOffset = src.offset;
        }

        /// <summary>
        /// Return a clone of the object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public PriorBoxParameter Clone()
        {
            PriorBoxParameter p = new param.PriorBoxParameter();
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

            rgChildren.Add<float>("min_size", min_size);
            rgChildren.Add<float>("max_size", max_size);
            rgChildren.Add<float>("aspect_ratio", aspect_ratio);
            rgChildren.Add("flip", flip.ToString());
            rgChildren.Add("clip", clip.ToString());
            rgChildren.Add<float>("variance", variance);

            if (img_size.HasValue)
            {
                rgChildren.Add("img_size", img_size.Value.ToString());
            }
            else
            {
                rgChildren.Add("img_h", img_h.Value.ToString());
                rgChildren.Add("img_w", img_w.Value.ToString());
            }

            if (step.HasValue)
            {
                rgChildren.Add("step", step.Value.ToString());
            }
            else
            {
                rgChildren.Add("step_h", step_h.Value.ToString());
                rgChildren.Add("step_w", step_w.Value.ToString());
            }

            rgChildren.Add("offset", offset.ToString());

            return new RawProto(strName, "", rgChildren);
        }



        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static PriorBoxParameter FromProto(RawProto rp)
        {
            PriorBoxParameter p = new PriorBoxParameter();
            string strVal;

            p.min_size = rp.FindArray<float>("min_size");
            p.max_size = rp.FindArray<float>("max_size");
            p.aspect_ratio = rp.FindArray<float>("aspect_ratio");

            if ((strVal = rp.FindValue("flip")) != null)
                p.flip = bool.Parse(strVal);

            if ((strVal = rp.FindValue("clip")) != null)
                p.clip = bool.Parse(strVal);

            p.variance = rp.FindArray<float>("variance");

            if ((strVal = rp.FindValue("img_size")) != null)
            {
                p.img_size = uint.Parse(strVal);
            }
            else
            {
                if ((strVal = rp.FindValue("img_h")) != null)
                    p.img_h = uint.Parse(strVal);

                if ((strVal = rp.FindValue("img_w")) != null)
                    p.img_w = uint.Parse(strVal);
            }

            if ((strVal = rp.FindValue("step")) != null)
            {
                p.step = float.Parse(strVal);
            }
            else
            {
                if ((strVal = rp.FindValue("step_h")) != null)
                    p.step_h = float.Parse(strVal);

                if ((strVal = rp.FindValue("step_w")) != null)
                    p.step_w = float.Parse(strVal);
            }

            if ((strVal = rp.FindValue("offset")) != null)
                p.offset = float.Parse(strVal);

            return p;
        }
    }
}
