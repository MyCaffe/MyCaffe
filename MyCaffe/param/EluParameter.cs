using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.ComponentModel;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the EluLayer.
    /// </summary>
    /// <remarks>
    /// @see [Deep Residual Networks with Exponential Linear Unit](https://arxiv.org/abs/1604.04112) by Anish Shah, Eashan Kadam, Hena Shah, Sameer Shinde, and Sandip Shingade, 2016.
    /// @see [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289) by Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter, 2015.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class EluParameter : EngineParameter
    {
        double m_dfAlpha = 1.0;

        /** @copydoc LayerParameterBase */
        public EluParameter()
            : base()
        {
        }

        /// <summary>
        /// Returns the reason that Caffe version was used instead of [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns></returns>
        public string useCaffeReason()
        {
            if (engine == Engine.CAFFE)
                return "The engine setting is set on CAFFE.";

            if (m_dfAlpha != 1.0)
                return "cuDnn only supports Alpha = 1.0";

            return "";
        }

        /// <summary>
        /// Queries whether or not to use [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns>Returns <i>true</i> when cuDnn is to be used, <i>false</i> otherwise.</returns>
        public bool useCudnn()
        {
            if (engine == EngineParameter.Engine.CAFFE || m_dfAlpha != 1.0)
                return false;

            return true;
        }

        /// <summary>
        /// Described in [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289) by Clevert, et al., 2015
        /// </summary>
        /// <remarks>
        /// Also see [Deep Residual Networks with Exponential Linear Unit](https://arxiv.org/abs/1604.04112) by Shah, et al., 2016
        /// </remarks>
        [Description("Described in 'Clevert, D. -A, Unterthiner, T., & Hochreiter, S. (2015).  Fast and Accurate Deep Network Learning from Exponential Linear Units (ELUs). arXiv")]
        public double alpha
        {
            get { return m_dfAlpha; }
            set { m_dfAlpha = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            EluParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is EluParameter)
            {
                EluParameter p = (EluParameter)src;
                m_dfAlpha = p.m_dfAlpha;
            }
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            EluParameter p = new EluParameter();
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Convert the parameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name to associate with the RawProto.</param>
        /// <returns>The new RawProto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("engine");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);

            if (alpha != 1.0)
                rgChildren.Add("alpha", alpha.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new EluParameter FromProto(RawProto rp)
        {
            string strVal;
            EluParameter p = new EluParameter();

            ((EngineParameter)p).Copy(EngineParameter.FromProto(rp));

            if ((strVal = rp.FindValue("alpha")) != null)
                p.alpha = ParseDouble(strVal);

            return p;
        }
    }
}
