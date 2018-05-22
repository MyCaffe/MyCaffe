using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameter for the LRNLayer.
    /// </summary>
    /// <remarks>
    /// @see [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) by Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R. Salakhutdinov, 2012.
    /// @see [Layer Normalization](https://arxiv.org/abs/1607.06450) by Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton, 2016.
    /// </remarks>
    public class LRNParameter : EngineParameter 
    {
        uint m_nLocalSize = 5;
        double m_dfAlpha = 1e-4;     // caffe default = 1.0, cudnn default = 1e-4
        double m_dfBeta = 0.75;      // caffe default = 0.75, cudnn default = 0.75
        NormRegion m_normRegion = NormRegion.ACROSS_CHANNELS;
        double m_dfK = 2.0;          // caffe default = 1.0, cudnn default = 2.0

        /// <summary>
        /// Defines the normalization region.
        /// </summary>
        public enum NormRegion
        {
            /// <summary>
            /// Normalize across channels.
            /// </summary>
            ACROSS_CHANNELS = 0,
            /// <summary>
            /// Normalize within channels.
            /// </summary>
            WITHIN_CHANNEL = 1
        }

        /** @copydoc EngineParameter */
        public LRNParameter()
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

            if (norm_region == NormRegion.WITHIN_CHANNEL)
                return "Currently using the normalization region 'WITHIN_CHANNEL' returns inconsistent errors from Caffe.";

            return "";
        }

        /// <summary>
        /// Queries whether or not to use [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns>Returns <i>true</i> when cuDnn is to be used, <i>false</i> otherwise.</returns>
        public bool useCudnn()
        {
            if (engine == EngineParameter.Engine.CAFFE ||
               norm_region == LRNParameter.NormRegion.WITHIN_CHANNEL)    // Currently CuDNN within channel Forward pass returns inconsistent results from Caffe.
                return false;

            return true;
        }

        /// <summary>
        /// Specifies the local size of the normalization window width.
        /// </summary>
        [Description("Specifies the local size of the normalization window width.")]
        public uint local_size
        {
            get { return m_nLocalSize; }
            set { m_nLocalSize = value; }
        }

        /// <summary>
        /// Specifies the alpha value used for variance scaling in the normalization formula.  NOTE: cuDNN uses a default of alpha = 1e-4, whereas Caffe uses a default of alpha = 1.0
        /// </summary>
        [Description("Specifies the alpha value used for variance scaling in the normalization formula.  NOTE: cuDNN uses a default of alpha = 1e-4, whereas CAFFE uses a default of alpha = 1.0.")]
        public double alpha
        {
            get { return m_dfAlpha; }
            set { m_dfAlpha = value; }
        }

        /// <summary>
        /// Specifies the beta value used as the power parameter in the normalization formula.  NOTE: both cuDNN and Caffe use a default of beta = 0.75
        /// </summary>
        [Description("Specifies the beta value used as the power parameter in the normalization formula.  NOTE: both cuDNN and CAFFE use a default of beta = 0.75.")]
        public double beta
        {
            get { return m_dfBeta; }
            set { m_dfBeta = value; }
        }

        /// <summary>
        /// Specifies the region over which to normalize.
        /// </summary>
        [Description("Specifies the region over which to normalize.")]
        public NormRegion norm_region
        {
            get { return m_normRegion; }
            set { m_normRegion = value; }
        }

        /// <summary>
        /// Specifies the k value used by the normalization parameter.  NOTE: cuDNN uses a default of k = 2.0, whereas Caffe uses a default of k = 1.0.
        /// </summary>
        [Description("Specifies the k value used by the normalization parameter.  NOTE: cuDNN uses a default of k = 2.0, whereas CAFFE uses a default of k = 2.0.")]
        public double k
        {
            get { return m_dfK; }
            set { m_dfK = value; }
        }

        /** @copydoc EngineParameter::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            LRNParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc EngineParameter::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is LRNParameter)
            {
                LRNParameter p = (LRNParameter)src;
                m_nLocalSize = p.m_nLocalSize;
                m_dfAlpha = p.m_dfAlpha;
                m_dfBeta = p.m_dfBeta;
                m_normRegion = p.m_normRegion;
                m_dfK = p.m_dfK;
            }
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            LRNParameter p = new LRNParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc EngineParameter::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("engine");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);
            rgChildren.Add("local_size", local_size.ToString());
            rgChildren.Add("alpha", alpha.ToString());
            rgChildren.Add("beta", beta.ToString());
            rgChildren.Add("norm_region", norm_region.ToString());
            rgChildren.Add("k", k.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new LRNParameter FromProto(RawProto rp)
        {
            string strVal;
            LRNParameter p = new LRNParameter();

            ((EngineParameter)p).Copy(EngineParameter.FromProto(rp));

            if ((strVal = rp.FindValue("local_size")) != null)
                p.local_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("alpha")) != null)
                p.alpha = double.Parse(strVal);

            if ((strVal = rp.FindValue("beta")) != null)
                p.beta = double.Parse(strVal);

            if ((strVal = rp.FindValue("norm_region")) != null)
            {
                switch (strVal)
                {
                    case "ACROSS_CHANNELS":
                        p.norm_region = NormRegion.ACROSS_CHANNELS;
                        break;

                    case "WITHIN_CHANNEL":
                        p.norm_region = NormRegion.WITHIN_CHANNEL;
                        break;

                    default:
                        throw new Exception("Unknown 'norm_region' value: " + strVal);
                }
            }

            if ((strVal = rp.FindValue("k")) != null)
                p.k = double.Parse(strVal);

            return p;
        }
    }
}
