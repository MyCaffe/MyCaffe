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
    /// Specifies the parameters for the ResizeParameter for use with SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class ResizeParameter
    {
        float m_fProb = 0;
        ResizeMode m_mode = ResizeMode.WARP;
        PadMode m_pad = PadMode.CONSTANT;
        List<InterpMode> m_rgInterp = new List<InterpMode>();
        uint m_nHeight = 0;
        uint m_nWidth = 0;
        uint m_nHeightScale = 0;
        uint m_nWidthScale = 0;
        List<float> m_rgfPadValue = new List<float>();

        /// <summary>
        /// Defines the resizing mode.
        /// </summary>
        public enum ResizeMode
        {
            /// <summary>
            /// Specifies to warp the sizing.
            /// </summary>
            WARP = 1,
            /// <summary>
            /// Specifies to fit into a small size.
            /// </summary>
            FIT_SMALL_SIZE = 2,
            /// <summary>
            /// Specifies to fit into a large padded size.
            /// </summary>
            FIT_LARGE_SIZE_AND_PAD = 3
        }

        /// <summary>
        /// Defines the padding mode.
        /// </summary>
        public enum PadMode
        {
            /// <summary>
            /// Use constant padding.
            /// </summary>
            CONSTANT = 1,
            /// <summary>
            /// Use mirrored padding.
            /// </summary>
            MIRRORED = 2,
            /// <summary>
            /// Repeat the nearest padding.
            /// </summary>
            REPEAT_NEAREST = 3
        }

        /// <summary>
        /// Defines the interpolation mode.
        /// </summary>
        public enum InterpMode
        {
            /// <summary>
            /// Use linear interpolation.
            /// </summary>
            LINEAR = 1,
            /// <summary>
            /// Use area interpolation.
            /// </summary>
            AREA = 2,
            /// <summary>
            /// Use nearest neighbor interpolation.
            /// </summary>
            NEAREST = 3,
            /// <summary>
            /// Use cubic interpolation.
            /// </summary>
            CUBIC = 4,
            /// <summary>
            /// Use LanCZos4 interpolation.
            /// </summary>
            LANCZOS4 = 5
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        public ResizeParameter()
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
        /// Get/set the resizing mode.
        /// </summary>
        public ResizeMode resize_mode
        {
            get { return m_mode; }
            set { m_mode = value; }
        }

        /// <summary>
        /// Get/set the resizing height.
        /// </summary>
        public uint height
        {
            get { return m_nHeight; }
            set { m_nHeight = value; }
        }

        /// <summary>
        /// Get/set the resizing width.
        /// </summary>
        public uint width
        {
            get { return m_nWidth; }
            set { m_nWidth = value; }
        }

        /// <summary>
        /// Get/set the resizing height scale used with FIT_SMALL_SIZE mode.
        /// </summary>
        public uint height_scale
        {
            get { return m_nHeightScale; }
            set { m_nHeightScale = value; }
        }

        /// <summary>
        /// Get/set the resizing width scale used with FIT_SMALL_SIZE_mode.
        /// </summary>
        public uint width_scale
        {
            get { return m_nWidthScale; }
            set { m_nWidthScale = value; }
        }

        /// <summary>
        /// Get/set the pad mode for FIT_LARGE_SIZE_AND_PAD mode.
        /// </summary>
        public PadMode pad_mode
        {
            get { return m_pad; }
            set { m_pad = value; }
        }

        /// <summary>
        /// Get/set the pad value which is repeated once for all channels, or provided one pad value per channel.
        /// </summary>
        public List<float> pad_value
        {
            get { return m_rgfPadValue; }
            set { m_rgfPadValue = value; }
        }

        /// <summary>
        /// Get/set the interp mode which is repeated once for all channels, or provided once per channel.
        /// </summary>
        public List<InterpMode> interp_mode
        {
            get { return m_rgInterp; }
            set { m_rgInterp = value; }
        }


        /// <summary>
        /// Load the and return a new ResizeParameter. 
        /// </summary>
        /// <param name="br"></param>
        /// <param name="bNewInstance"></param>
        /// <returns>The new object is returned.</returns>
        public ResizeParameter Load(BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ResizeParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Copy the source object.
        /// </summary>
        /// <param name="src">Specifies the source data.</param>
        public void Copy(ResizeParameter src)
        {
            m_fProb = src.prob;
            m_mode = src.resize_mode;
            m_pad = src.pad_mode;

            m_rgInterp = new List<InterpMode>();
            foreach (InterpMode interp in src.m_rgInterp)
            {
                m_rgInterp.Add(interp);
            }

            m_nHeight = src.height;
            m_nWidth = src.width;
            m_nHeightScale = src.height_scale;
            m_nWidthScale = src.width_scale;

            m_rgfPadValue = new List<float>();
            foreach (float fPad in src.pad_value)
            {
                m_rgfPadValue.Add(fPad);
            }
        }

        /// <summary>
        /// Return a copy of this object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public ResizeParameter Clone()
        {
            ResizeParameter p = new param.ssd.ResizeParameter();
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

            rgChildren.Add(new RawProto("prob", prob.ToString()));
            rgChildren.Add(new RawProto("resize_mode", m_mode.ToString()));
            rgChildren.Add(new RawProto("pad_mode", m_pad.ToString()));
            rgChildren.Add(new RawProto("height", m_nHeight.ToString()));
            rgChildren.Add(new RawProto("width", m_nWidth.ToString()));
            rgChildren.Add(new RawProto("height_scale", m_nHeightScale.ToString()));
            rgChildren.Add(new RawProto("width_scale", m_nWidthScale.ToString()));

            foreach (InterpMode interp in m_rgInterp)
            {
                rgChildren.Add(new RawProto("interp_mode", interp.ToString()));
            }

            foreach (float fPad in m_rgfPadValue)
            {
                rgChildren.Add(new RawProto("pad_value", fPad.ToString()));
            }

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ResizeParameter FromProto(RawProto rp)
        {
            ResizeParameter p = new ResizeParameter();
            string strVal;

            if ((strVal = rp.FindValue("prob")) != null)
                p.prob = float.Parse(strVal);

            if ((strVal = rp.FindValue("resize_mode")) != null)
            {
                switch (strVal)
                {
                    case "WARP":
                        p.resize_mode = ResizeMode.WARP;
                        break;

                    case "FIT_SMALL_SIZE":
                        p.resize_mode = ResizeMode.FIT_SMALL_SIZE;
                        break;

                    case "FIT_LARGE_SIZE_AND_PAD":
                        p.resize_mode = ResizeMode.FIT_LARGE_SIZE_AND_PAD;
                        break;

                    default:
                        throw new Exception("Unknown resize_mode '" + strVal + "'!");
                }
            }

            if ((strVal = rp.FindValue("height")) != null)
                p.height = uint.Parse(strVal);

            if ((strVal = rp.FindValue("width")) != null)
                p.width = uint.Parse(strVal);

            if ((strVal = rp.FindValue("height_scale")) != null)
                p.height_scale = uint.Parse(strVal);

            if ((strVal = rp.FindValue("width_scale")) != null)
                p.width_scale = uint.Parse(strVal);

            if ((strVal = rp.FindValue("pad_mode")) != null)
            {
                switch (strVal)
                {
                    case "CONSTANT":
                        p.pad_mode = PadMode.CONSTANT;
                        break;

                    case "MIRRORED":
                        p.pad_mode = PadMode.MIRRORED;
                        break;

                    case "REPEAT_NEAREST":
                        p.pad_mode = PadMode.REPEAT_NEAREST;
                        break;

                    default:
                        throw new Exception("Unknown pad_mode '" + strVal + "'!");
                }
            }

            p.pad_value = new List<float>();
            RawProtoCollection col = rp.FindChildren("pad_value");
            foreach (RawProto rp1 in col)
            {
                if ((strVal = rp.FindValue("pad_value")) != null)
                    p.pad_value.Add(float.Parse(strVal));
            }

            p.interp_mode = new List<InterpMode>();
            RawProtoCollection col1 = rp.FindChildren("interp_mode");
            foreach (RawProto pr1 in col1)
            {
                strVal = pr1.Value;

                switch (strVal)
                {
                    case "LINEAR":
                        p.interp_mode.Add(InterpMode.LINEAR);
                        break;
                
                    case "AREA":
                        p.interp_mode.Add(InterpMode.AREA);
                        break;
                
                    case "NEAREST":
                        p.interp_mode.Add(InterpMode.NEAREST);
                        break;
                
                    case "CUBIC":
                        p.interp_mode.Add(InterpMode.CUBIC);
                        break;
                
                    case "LANCZOS4":
                        p.interp_mode.Add(InterpMode.LANCZOS4);
                        break;
                
                    default:
                        throw new Exception("Unknown interp_mode '" + strVal + "'!");
                }
            }

            return p;
        }
    }
}
