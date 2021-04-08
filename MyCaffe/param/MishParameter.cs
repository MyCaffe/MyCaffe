using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Stores the parameters used by the MishLayer
    /// </summary>
    /// <remarks>
    /// @see [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681v1) by Diganta Misra, 2019.
    /// @see [Meet Mish — New State of the Art AI Activation Function. The successor to ReLU?](https://lessw.medium.com/meet-mish-new-state-of-the-art-ai-activation-function-the-successor-to-relu-846a6d93471f) by Less Wright, 2019
    /// @see [Swish Vs Mish: Latest Activation Functions](https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/) by Krutika Bapat, 2020
    /// </remarks>
    public class MishParameter : EngineParameter
    {
        /** @copydoc LayerParameterBase */
        public MishParameter()
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

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            MishParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            MishParameter p = new MishParameter();
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new MishParameter FromProto(RawProto rp)
        {
            MishParameter p = new MishParameter();

            ((EngineParameter)p).Copy(EngineParameter.FromProto(rp));

            return p;
        }
    }
}
