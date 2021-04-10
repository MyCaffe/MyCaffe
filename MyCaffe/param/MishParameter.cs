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
        double m_dfThreshold = 20;
        int m_nMethod = 0;

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

        /// <summary>
        /// Specifies the method used for backward computation, where method = 0 uses the default scalled method, and method = 1 uses the straight derivative method.
        /// </summary>
        public int method
        {
            get { return m_nMethod; }
            set { m_nMethod = value; }
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

        /** @copydoc EngineParameter::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is MishParameter)
            {
                MishParameter p = (MishParameter)src;
                m_dfThreshold = p.m_dfThreshold;
                m_nMethod = p.m_nMethod;
            }
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            MishParameter p = new MishParameter();
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

            if (method != 0)
                rgChildren.Add("method", threshold.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new MishParameter FromProto(RawProto rp)
        {
            string strVal;
            MishParameter p = new MishParameter();

            ((EngineParameter)p).Copy(EngineParameter.FromProto(rp));

            if ((strVal = rp.FindValue("threshold")) != null)
                p.threshold = ParseDouble(strVal);

            if ((strVal = rp.FindValue("method")) != null)
                p.method = int.Parse(strVal);

            return p;
        }
    }
}
