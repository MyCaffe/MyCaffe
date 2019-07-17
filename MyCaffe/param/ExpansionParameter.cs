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
    /// Specifies the parameters for the ExpansionParameter.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class ExpansionParameter
    {
        float m_fProb = 0;
        float m_fMaxExpandRatio = 1.0f;

        /// <summary>
        /// The constructor.
        /// </summary>
        public ExpansionParameter()
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
        /// <param name="p">The copy is placed in this parameter.</param>
        public void Copy(ExpansionParameter src)
        {
            m_fProb = src.prob;
            m_fMaxExpandRatio = src.max_expand_ratio;
        }

        /// <summary>
        /// Return a clone of the object.
        /// </summary>
        /// <returns>A new copy of the object is returned.</returns>
        public ExpansionParameter Clone()
        {
            ExpansionParameter p = new param.ExpansionParameter();
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
            rgChildren.Add(new RawProto("max_expand_ratio", max_expand_ratio.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ExpansionParameter FromProto(RawProto rp)
        {
            ExpansionParameter p = new ExpansionParameter();
            string strVal;

            if ((strVal = rp.FindValue("prob")) != null)
                p.prob = float.Parse(strVal);

            if ((strVal = rp.FindValue("max_expand_ratio")) != null)
                p.max_expand_ratio = float.Parse(strVal);

            return p;
        }
    }
}
