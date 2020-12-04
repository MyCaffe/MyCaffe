using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Stores the parameters used by the ClipLayer
    /// </summary>
    public class ClipParameter : LayerParameterBase
    {
        double m_dfMin;
        double m_dfMax;

        /** @copydoc LayerParameterBase */
        public ClipParameter()
            : base()
        {
        }

        /// <summary>
        /// Returns the reason that Caffe version was used instead of [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns>The reason for using Caffe is returned.</returns>
        public string useCaffeReason()
        {
            return "This layer is curently only supported in Caffe.";
        }

        /// <summary>
        /// Queries whether or not to use [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns>Returns <i>true</i> when cuDnn is to be used, <i>false</i> otherwise.</returns>
        public bool useCudnn()
        {
            return false;
        }

        /// <summary>
        /// Specifies the min value for the Clip activation function.
        /// </summary>
        [Description("Specifies the min parameter for the Clip function.")]
        public double min
        {
            get { return m_dfMin; }
            set { m_dfMin = value; }
        }

        /// <summary>
        /// Specifies the max value for the Clip activation function.
        /// </summary>
        [Description("Specifies the max parameter for the Clip function.")]
        public double max
        {
            get { return m_dfMax; }
            set { m_dfMax = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ClipParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            if (src is ClipParameter)
            {
                ClipParameter p = (ClipParameter)src;
                m_dfMin = p.m_dfMin;
                m_dfMax = p.m_dfMax;
            }
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            ClipParameter p = new ClipParameter();
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
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("min", min.ToString());
            rgChildren.Add("max", max.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ClipParameter FromProto(RawProto rp)
        {
            string strVal;
            ClipParameter p = new ClipParameter();

            p.Copy(EngineParameter.FromProto(rp));

            if ((strVal = rp.FindValue("min")) != null)
                p.min = ParseDouble(strVal);

            if ((strVal = rp.FindValue("max")) != null)
                p.max = ParseDouble(strVal);

            return p;
        }
    }
}
