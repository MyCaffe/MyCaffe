using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.beta
{
    /// <summary>
    /// Stores the parameters used by the SerfLayer
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class SerfParameter : EngineParameter
    {
        double m_dfThreshold = 20;

        /** @copydoc LayerParameterBase */
        public SerfParameter()
            : base()
        {
        }

        /// <summary>
        /// Returns the reason that Caffe version was used instead of [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns></returns>
        public string useCaffeReason()
        {
            return "Currenly only CAFFE supported.";
        }

        /// <summary>
        /// Queries whether or not to use [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns>Returns <i>true</i> when cuDnn is to be used, <i>false</i> otherwise.</returns>
        /// <remarks>Currently, only CAFFE supported.</remarks>
        public bool useCudnn()
        {
            return false;
        }

        /// <summary>
        /// Specifies the max threshold value used in exp functions.
        /// </summary>
        /// <remarks>
        /// @see [thomasbrandon/mish-cuda](https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h) by thomasbrandon, 2019 MIT License.
        /// </remarks>
        [Description("Specifies the max threshold value used in exp functions.")]
        public double threshold
        {
            get { return m_dfThreshold; }
            set { m_dfThreshold = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            SerfParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc EngineParameter::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is SerfParameter)
            {
                SerfParameter p = (SerfParameter)src;
                m_dfThreshold = p.m_dfThreshold;
            }
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            SerfParameter p = new SerfParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc EngineParameter::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("engine");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);

            if (threshold != 0)
                rgChildren.Add("threshold", threshold.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new SerfParameter FromProto(RawProto rp)
        {
            string strVal;
            SerfParameter p = new SerfParameter();

            ((EngineParameter)p).Copy(EngineParameter.FromProto(rp));

            if ((strVal = rp.FindValue("threshold")) != null)
                p.threshold = ParseDouble(strVal);

            return p;
        }
    }
}
