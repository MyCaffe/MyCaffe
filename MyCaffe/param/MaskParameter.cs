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
    /// Specifies the parameters for the MaskParameter used to mask portions of the transformed data when enabled.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class MaskParameter : OptionalParameter
    {        
        int m_nMaskLeft = 0;
        int m_nMaskRight = 0;
        int m_nMaskTop = 0;
        int m_nMaskBottom = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bActive">Specifies whether or not the parameter is active or not.</param>
        public MaskParameter(bool bActive) : base(bActive)
        {
        }

        /// <summary>
        /// Get/set the mask boundary left.
        /// </summary>
        public int boundary_left
        {
            get { return m_nMaskLeft; }
            set { m_nMaskLeft = value; }
        }

        /// <summary>
        /// Get/set the mask boundary left.
        /// </summary>
        public int boundary_right
        {
            get { return m_nMaskRight; }
            set { m_nMaskRight = value; }
        }

        /// <summary>
        /// Get/set the mask boundary top.
        /// </summary>
        public int boundary_top
        {
            get { return m_nMaskTop; }
            set { m_nMaskTop = value; }
        }

        /// <summary>
        /// Get/set the mask boundary bottom.
        /// </summary>
        public int boundary_bottom
        {
            get { return m_nMaskBottom; }
            set { m_nMaskBottom = value; }
        }


        /// <summary>
        /// Load the and return a new MaskParameter. 
        /// </summary>
        /// <param name="br"></param>
        /// <param name="bNewInstance"></param>
        /// <returns>The new object is returned.</returns>
        public MaskParameter Load(BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            MaskParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Copy the source object.
        /// </summary>
        /// <param name="src">Specifies the source data.</param>
        public override void Copy(OptionalParameter src)
        {
            base.Copy(src);

            if (src is MaskParameter)
            {
                MaskParameter p = (MaskParameter)src;
                m_nMaskLeft = p.m_nMaskLeft;
                m_nMaskRight = p.m_nMaskRight;
                m_nMaskTop = p.m_nMaskTop;
                m_nMaskBottom = p.m_nMaskBottom;
            }
        }

        /// <summary>
        /// Return a copy of this object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public MaskParameter Clone()
        {
            MaskParameter p = new MaskParameter(Active);
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
            rgChildren.Add(new RawProto("boundary_left", boundary_left.ToString()));
            rgChildren.Add(new RawProto("boundary_right", boundary_right.ToString()));
            rgChildren.Add(new RawProto("boundary_top", boundary_top.ToString()));
            rgChildren.Add(new RawProto("boundary_bottom", boundary_bottom.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new MaskParameter FromProto(RawProto rp)
        {
            MaskParameter p = new MaskParameter(true);
            string strVal;

            RawProto rpOption = rp.FindChild("option");
            if (rpOption != null)
                ((OptionalParameter)p).Copy(OptionalParameter.FromProto(rpOption));

            if ((strVal = rp.FindValue("boundary_left")) != null)
                p.boundary_left = int.Parse(strVal);

            if ((strVal = rp.FindValue("boundary_right")) != null)
                p.boundary_right = int.Parse(strVal);

            if ((strVal = rp.FindValue("boundary_top")) != null)
                p.boundary_top = int.Parse(strVal);

            if ((strVal = rp.FindValue("boundary_bottom")) != null)
                p.boundary_bottom = int.Parse(strVal);

            return p;
        }
    }
}
