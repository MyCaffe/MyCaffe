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
    /// Specifies the parameters for the SaltPepperParameter used with SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class SaltPepperParameter : OptionalParameter
    {
        float m_fFraction = 0;
        List<float> m_rgValue = new List<float>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bActive">Specifies whether or not the parameter is active or not.</param>
        public SaltPepperParameter(bool bActive) : base(bActive)
        {
        }

        /// <summary>
        /// Get/set the percentage of pixels.
        /// </summary>
        [Description("Get/set the percentage of pixels.")]
        public float fraction
        {
            get { return m_fFraction; }
            set { m_fFraction = value; }
        }

        /// <summary>
        /// Get/set the values.
        /// </summary>
        [Description("Get/set the values.")]
        public List<float> value
        {
            get { return m_rgValue; }
            set { m_rgValue = value; }
        }

        /// <summary>
        /// Copy the object.
        /// </summary>
        /// <param name="src">The copy is placed in this parameter.</param>
        public override void Copy(OptionalParameter src)
        {
            base.Copy(src);

            if (src is SaltPepperParameter)
            {
                SaltPepperParameter p = (SaltPepperParameter)src;
                m_fFraction = p.fraction;
                m_rgValue = new List<float>();

                foreach (float fVal in p.value)
                {
                    m_rgValue.Add(fVal);
                }
            }
        }

        /// <summary>
        /// Return a clone of the object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public SaltPepperParameter Clone()
        {
            SaltPepperParameter p = new param.ssd.SaltPepperParameter(Active);
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
            rgChildren.Add(new RawProto("fraction", m_fFraction.ToString()));

            foreach (float fVal in m_rgValue)
            {
                rgChildren.Add(new RawProto("value", fVal.ToString()));
            }

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new SaltPepperParameter FromProto(RawProto rp)
        {
            SaltPepperParameter p = new SaltPepperParameter(false);
            string strVal;

            RawProto rpOption = rp.FindChild("option");
            if (rpOption != null)
                ((OptionalParameter)p).Copy(OptionalParameter.FromProto(rpOption));

            if ((strVal = rp.FindValue("fraction")) != null)
                p.fraction = BaseParameter.parseFloat(strVal);

            p.value = new List<float>();
            RawProtoCollection col = rp.FindChildren("value");
            foreach (RawProto rp1 in col)
            {
                if ((strVal = rp.FindValue("value")) != null)
                    p.value.Add(BaseParameter.parseFloat(strVal));
            }

            return p;
        }
    }
}
