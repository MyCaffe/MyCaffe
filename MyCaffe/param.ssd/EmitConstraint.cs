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
    /// Specifies the parameters for the EmitConstraint.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class EmitConstraint : OptionalParameter
    {
        EmitType m_emitType = EmitType.CENTER;
        float m_fEmitOverlap = 0;

        /// <summary>
        /// Specifies the emit type.
        /// </summary>
        public enum EmitType
        {
            /// <summary>
            /// Specifies to center the data.
            /// </summary>
            CENTER = 0,
            /// <summary>
            /// Specifies to overlap the data.
            /// </summary>
            MIN_OVERLAP = 1
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bActive">Specifies whether or not the parameter is active or not.</param>
        public EmitConstraint(bool bActive) : base(bActive)
        {
        }

        /// <summary>
        /// Get/set the emit type.
        /// </summary>
        public EmitType emit_type
        {
            get { return m_emitType; }
            set { m_emitType = value; }
        }

        /// <summary>
        /// Get/set the emit overlap used with MIN_OVERLAP.
        /// </summary>
        public float emit_overlap
        {
            get { return m_fEmitOverlap; }
            set { m_fEmitOverlap = value; }
        }

        /// <summary>
        /// Copy the source object.
        /// </summary>
        /// <param name="src">Specifies the source data.</param>
        public void Copy(EmitConstraint src)
        {
            m_emitType = src.emit_type;
            m_fEmitOverlap = src.emit_overlap;
        }

        /// <summary>
        /// Return a copy of this object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public EmitConstraint Clone()
        {
            EmitConstraint p = new param.ssd.EmitConstraint(Active);
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

            rgChildren.Add(new RawProto("active", Active.ToString()));
            rgChildren.Add(new RawProto("emit_type", m_emitType.ToString()));
            rgChildren.Add(new RawProto("emit_overlap", m_fEmitOverlap.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static EmitConstraint FromProto(RawProto rp)
        {
            EmitConstraint p = new EmitConstraint(true);
            string strVal;

            if ((strVal = rp.FindValue("active")) != null)
                p.Active = bool.Parse(strVal);

            if ((strVal = rp.FindValue("emit_type")) != null)
            {
                switch (strVal)
                {
                    case "CENTER":
                        p.emit_type = EmitType.CENTER;
                        break;

                    case "MIN_OVERLAP":
                        p.emit_type = EmitType.MIN_OVERLAP;
                        break;

                    default:
                        throw new Exception("Unknown emit_type '" + strVal + "'!");
                }
            }

            if ((strVal = rp.FindValue("emit_overlap")) != null)
                p.emit_overlap = float.Parse(strVal);

            return p;
        }
    }
}
