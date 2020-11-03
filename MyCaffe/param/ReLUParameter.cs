using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the ReLULayer
    /// </summary>
    /// <remarks>
    /// @see [Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853) by Bing Xu, Naiyan Wang, Tianqi Chen, and Mu Li, 2015.
    /// @see [Revise Saturated Activation Functions](https://arxiv.org/abs/1602.05980?context=cs) by Bing Xu, Ruitong Huang, and Mu Li, 2016.
    /// @see [Rectifier Nonlinearities Improve Neural Network Acoustic Models](http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf) by Andrew L. Maas, Awni Y. Hannun, and Andrew Y. Ng, 2013.
    /// </remarks>
    public class ReLUParameter : EngineParameter 
    {
        double m_dfNegativeSlope = 0;

        /** @copydoc EngineParameter */
        public ReLUParameter()
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

            if (negative_slope != 0)
                return "Leaky ReLU (negative slope != 0) currently not supported with cuDnn.";

            return "";
        }

        /// <summary>
        /// Queries whether or not to use [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns>Returns <i>true</i> when cuDnn is to be used, <i>false</i> otherwise.</returns>
        public bool useCudnn()
        {
            if (engine == EngineParameter.Engine.CAFFE ||
                negative_slope != 0)
                return false;

            return true;
        }

        /// <summary>
        /// Specifies the negative slope.  Allow non-zero slope for negative inputs to speed up optimization.
        /// </summary>
        [Description("Specifies the negative slope.  Allow non-zero slope for negative inputs to speed up optimization.")]
        public double negative_slope
        {
            get { return m_dfNegativeSlope; }
            set { m_dfNegativeSlope = value; }
        }

        /** @copydoc EngineParameter::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ReLUParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc EngineParameter::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is ReLUParameter)
            {
                ReLUParameter p = (ReLUParameter)src;
                m_dfNegativeSlope = p.m_dfNegativeSlope;
            }
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            ReLUParameter p = new ReLUParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc EngineParameter::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("engine");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);

            if (negative_slope != 0)
                rgChildren.Add("negative_slope", negative_slope.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /** @copydoc EngineParameter::FromProto */
        public static new ReLUParameter FromProto(RawProto rp)
        {
            string strVal;
            ReLUParameter p = new ReLUParameter();

            p.Copy(EngineParameter.FromProto(rp));

            if ((strVal = rp.FindValue("negative_slope")) != null)
                p.negative_slope = parseDouble(strVal);

            return p;
        }
    }
}
