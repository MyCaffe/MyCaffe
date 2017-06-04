using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the PoolingLayer.
    /// </summary>
    /// <remarks>
    /// @see [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) by Vincent Dumoulin and Francesco Visin, 2016.
    /// @see [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150) by Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba, 2015.
    /// @see [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) by Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner, 1998.
    /// </remarks>
    public class PoolingParameter : KernelParameter
    {
        PoolingMethod m_pool = PoolingMethod.MAX;
        bool m_bGlobalPooling = false;

        /// <summary>
        /// Defines the pooling method.
        /// </summary>
        public enum PoolingMethod
        {
            /// <summary>
            /// Select the maximum value from the pooling kernel.
            /// </summary>
            MAX = 0,
            /// <summary>
            /// Select the average value from the pooling kernel.
            /// </summary>
            AVE = 1,
            /// <summary>
            /// Select the stochastic value from the pooling kernel.
            /// </summary>
            STOCHASTIC = 2
        }

        /** @copydoc KernelParameter */
        public PoolingParameter()
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

            if (pool == PoolingMethod.STOCHASTIC)
                return "The STOCHASTIC pooing method is currently not supported with cuDnn.";

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

            if (pool == PoolingParameter.PoolingMethod.STOCHASTIC)
                return false;

            return true;
        }

        /// <summary>
        /// Specifies the pooling method.
        /// </summary>
        [Description("Specifies pooling method to use.")] 
        public PoolingMethod pool
        {
            get { return m_pool; }
            set { m_pool = value; }
        }

        /// <summary>
        /// Specifies whether or not to enable global pooling.
        /// </summary>
        [Description("Specifies whether or not to enable global pooling.")]
        public bool global_pooling
        {
            get { return m_bGlobalPooling; }
            set { m_bGlobalPooling = value; }
        }

        /** @copydoc KernelParameter::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            PoolingParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc KernelParameter::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is PoolingParameter)
            {
                PoolingParameter p = (PoolingParameter)src;
                m_pool = p.m_pool;
                m_bGlobalPooling = p.m_bGlobalPooling;
            }
        }

        /** @copydoc KernelParameter::Clone */
        public override LayerParameterBase Clone()
        {
            PoolingParameter p = new PoolingParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc KernelParameter::ToProto */
        public override RawProto ToProto(string strName)
        {
            dilation.Clear();
            RawProto rpBase = base.ToProto("kernel");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);
            rgChildren.Add("pool", pool.ToString());

            if (global_pooling != false)
                rgChildren.Add("global_pooling", global_pooling.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new PoolingParameter FromProto(RawProto rp)
        {
            string strVal;
            PoolingParameter p = new PoolingParameter();

            ((KernelParameter)p).Copy(KernelParameter.FromProto(rp));

            if ((strVal = rp.FindValue("pool")) != null)
            {
                switch (strVal)
                {
                    case "MAX":
                        p.pool = PoolingMethod.MAX;
                        break;

                    case "AVE":
                        p.pool = PoolingMethod.AVE;
                        break;

                    case "STOCHASTIC":
                        p.pool = PoolingMethod.STOCHASTIC;
                        break;

                    default:
                        throw new Exception("Unknown pooling 'method' value: " + strVal);
                }
            }

            if ((strVal = rp.FindValue("global_pooling")) != null)
                p.global_pooling = bool.Parse(strVal);

            return p;
        }
    }
}
