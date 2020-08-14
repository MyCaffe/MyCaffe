using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Stores the parameters used by the SwishLayer
    /// </summary>
    /// <remarks>
    /// @see [Activation Functions](https://arxiv.org/abs/1710.05941v2) by Prajit Ramachandran, Barret Zoph, Quoc V. Le., 2017.
    /// </remarks>
    public class SwishParameter : EngineParameter
    {
        double m_dfBeta = 1;

        /** @copydoc LayerParameterBase */
        public SwishParameter()
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

            return "";
        }

        /// <summary>
        /// Queries whether or not to use [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns>Returns <i>true</i> when cuDnn is to be used, <i>false</i> otherwise.</returns>
        public bool useCudnn()
        {
            if (engine == EngineParameter.Engine.CAFFE)
                return false;

            return true;
        }

        /// <summary>
        /// Specifies the beta value for the Swish activation function.
        /// </summary>
        [Description("Specifies the Beta parameter for the Swish activation function.")]
        public double beta
        {
            get { return m_dfBeta; }
            set { m_dfBeta = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            SwishParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is SwishParameter)
            {
                SwishParameter p = (SwishParameter)src;
                m_dfBeta = p.m_dfBeta;
            }
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            SwishParameter p = new SwishParameter();
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

            if (beta != 0)
                rgChildren.Add("beta", beta.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new SwishParameter FromProto(RawProto rp)
        {
            string strVal;
            SwishParameter p = new SwishParameter();

            p.Copy(EngineParameter.FromProto(rp));

            if ((strVal = rp.FindValue("beta")) != null)
                p.beta = double.Parse(strVal);

            return p;
        }
    }
}
