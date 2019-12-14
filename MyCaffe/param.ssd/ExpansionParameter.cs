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
    /// Specifies the parameters for the ExpansionParameter used with SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ExpansionParameter : OptionalParameter
    {
        float m_fProb = 0;
        float m_fMaxExpandRatio = 1.0f;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bActive">Specifies whether or not the parameter is active or not.</param>
        public ExpansionParameter(bool bActive) : base(bActive)
        {
        }

        /// <summary>
        /// Get/set probability of using this expansion policy.
        /// </summary>
        public float prob
        {
            get { return m_fProb; }
            set { m_fProb = value; }
        }

        /// <summary>
        /// Get/set the ratio to expand the image.
        /// </summary>
        public float max_expand_ratio
        {
            get { return m_fMaxExpandRatio; }
            set { m_fMaxExpandRatio = value; }
        }

        /// <summary>
        /// Copy the object.
        /// </summary>
        /// <param name="src">The copy is placed in this parameter.</param>
        public override void Copy(OptionalParameter src)
        {
            base.Copy(src);

            if (src is ExpansionParameter)
            {
                ExpansionParameter p = (ExpansionParameter)src;
                m_fProb = p.prob;
                m_fMaxExpandRatio = p.max_expand_ratio;
            }
        }

        /// <summary>
        /// Return a clone of the object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public ExpansionParameter Clone()
        {
            ExpansionParameter p = new param.ssd.ExpansionParameter(Active);
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
            rgChildren.Add(new RawProto("prob", prob.ToString()));
            rgChildren.Add(new RawProto("max_expand_ratio", max_expand_ratio.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new ExpansionParameter FromProto(RawProto rp)
        {
            ExpansionParameter p = new ExpansionParameter(true);
            string strVal;

            RawProto rpOption = rp.FindChild("option");
            if (rpOption != null)
                ((OptionalParameter)p).Copy(OptionalParameter.FromProto(rpOption));

            if ((strVal = rp.FindValue("prob")) != null)
                p.prob = float.Parse(strVal);

            if ((strVal = rp.FindValue("max_expand_ratio")) != null)
                p.max_expand_ratio = float.Parse(strVal);

            return p;
        }
    }
}
