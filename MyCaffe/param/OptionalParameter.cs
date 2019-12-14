using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.param
{
    /// <summary>
    /// The OptionalParameter is the base class for parameters that are optional such as the MaskParameter, DistorationParameter, ExpansionParameter, NoiseParameter, and ResizeParameter.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class OptionalParameter
    {
        bool m_bActive = false;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bActive">Specifies whether or not the parameter is active (default = false).</param>
        public OptionalParameter(bool bActive = false)
        {
            m_bActive = bActive;
        }

        /// <summary>
        /// When active, the parameter is used, otherwise it is ignored.
        /// </summary>
        public bool Active
        {
            get { return m_bActive; }
            set { m_bActive = value; }
        }

        /// <summary>
        /// Copy the source object.
        /// </summary>
        /// <param name="src">Specifies the source data.</param>
        public virtual void Copy(OptionalParameter src)
        {
            m_bActive = src.m_bActive;
        }

        /// <summary>
        /// Convert this object to a raw proto.
        /// </summary>
        /// <param name="strName">Specifies the name of the proto.</param>
        /// <returns>The new proto is returned.</returns>
        public virtual RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("active", m_bActive.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static OptionalParameter FromProto(RawProto rp)
        {
            string strVal;
            OptionalParameter p = new OptionalParameter();

            if ((strVal = rp.FindValue("active")) != null)
                p.Active = bool.Parse(strVal);

            return p;
        }
    }
}
