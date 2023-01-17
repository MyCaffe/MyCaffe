using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the SoftmaxLayer
    /// </summary>
    /// <remarks>
    /// @see [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580v1) by Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R. Salakhutdinov, 2012.
    /// @see [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144v2) by Wu, et al., 2016.
    /// @see [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538v1) by Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean, 2017.
    /// @see [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410v2) by Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu, 2016.
    /// </remarks>
    public class SoftmaxParameter : EngineParameter 
    {
        int m_nAxis = 1;

        /** @copydoc EngineParameter */
        public SoftmaxParameter()
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
        /// The axis along which to perform the softmax -- may be negative to index
        /// from the end (e.g., -1 for the last axis).
        /// Any other axes will be evaluated as independent softmaxes.
        /// </summary>
        [Description("Specifies the axis along which to perform the softmax - may be negative to index from the end (e.g., -1 for the last axis).")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /** @copydoc EngineParameter::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            SoftmaxParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc EngineParameter::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is SoftmaxParameter)
            {
                SoftmaxParameter p = (SoftmaxParameter)src;
                m_nAxis = p.m_nAxis;
            }
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            SoftmaxParameter p = new SoftmaxParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc EngineParameter::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("engine");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);
            rgChildren.Add("axis", axis.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new SoftmaxParameter FromProto(RawProto rp)
        {
            string strVal;
            SoftmaxParameter p = new SoftmaxParameter();

            p.Copy(EngineParameter.FromProto(rp));

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            return p;
        }
    }
}
