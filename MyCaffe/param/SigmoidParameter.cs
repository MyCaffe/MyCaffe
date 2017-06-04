using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the SigmoidLayer.
    /// </summary>
    /// <remarks>
    /// @see [eXpose: A Character-Level Convolutional Neural Network with Embeddings For Detecting Malicious URLs, File Paths and Registry Keys](https://arxiv.org/abs/1702.08568v1) by Joshua Saxe and Konstantin Berlin, 2017. 
    /// @see [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904v1) by Fei Wang, Mengquing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, and Xiaoou Tang, 2017.
    /// @see [Attention and Localization based on a Deep Convolutional Recurrent Model for Weakly Supervised Audio Tagging](https://arxiv.org/abs/1703.06052v1) by Yong Xu, Qiuqiang Kong, Qiang Huang, Wenwu Wang, and Mark D. Plumbley, 2017.
    /// </remarks>
    public class SigmoidParameter : EngineParameter 
    {
        /** @copydoc EngineParameter */
        public SigmoidParameter()
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


        /** @copydoc EngineParameter::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            SigmoidParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            SigmoidParameter p = new SigmoidParameter();
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new SigmoidParameter FromProto(RawProto rp)
        {
            SigmoidParameter p = new SigmoidParameter();

            p.Copy(EngineParameter.FromProto(rp));

            return p;
        }
    }
}
