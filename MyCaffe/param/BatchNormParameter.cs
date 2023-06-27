using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the BatchNormLayer.
    /// </summary>
    /// <remarks>
    /// @see [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) by Sergey Ioffe and Christian Szegedy, 2015.
    /// @see [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737v2) by Alexander Hermans, Lucas Beyer, and Bastian Leibe, 2017. 
    /// @see [Layer Normalization](https://arxiv.org/abs/1607.06450) by Jimmy Lei Ba,  and Jamie Ryan Kiros, and Geoffrey E. Hinton, 2016.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class BatchNormParameter : EngineParameter
    {
        bool? m_bUseGlobalStats = null;
        double m_dfMovingAverageFraction = 0.999;
        double m_dfEps = 1e-5;
        bool m_bScaleBias = false;
        FillerParameter m_scaleFiller = null;
        FillerParameter m_biasFiller = null;

        /** @copydoc LayerParameterBase */
        public BatchNormParameter()
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
            if (engine != EngineParameter.Engine.CAFFE)
                return true;

            return false;
        }

        /// <summary>
        /// Specifies to use the scale and bias terms, otherwise the scale = 1 and bias = 0
        /// are used to form an identity operation.
        /// </summary>
        /// <remarks>
        /// NOTE: Currently the scale_bias is only used by the CUDNN engine.
        /// </remarks>
        [Description("Specifies to use the scale and bias terms, otherwise the scale = 1 and bias = 0 which performs an identity operation.")]
        public bool scale_bias
        {
            get { return m_bScaleBias; }
            set { m_bScaleBias = value; }
        }

        /// <summary>
        /// Specifies the scale filler used to fill the scale value.  If null, a constant(1) filler is used.
        /// </summary>
        /// <remarks>
        /// NOTE: Currently the scale_bias is only used by the CUDNN engine.
        /// </remarks>
        [Description("Specifies the scale filler used, when null 'constant(1)' is used.")]
        public FillerParameter scale_filler
        {
            get { return m_scaleFiller; }
            set { m_scaleFiller = value; }
        }

        /// <summary>
        /// Specifies the bias filler used to file the bias value.  If null, a constant(0) filler is used.
        /// </summary>
        /// <remarks>
        /// NOTE: Currently the scale_bias is only used by the CUDNN engine.
        /// </remarks>
        [Description("Specifies the bias filler used, when null 'constant(0)' is used.")]
        public FillerParameter bias_filler
        {
            get { return m_biasFiller; }
            set { m_biasFiller = value; }
        }

        /// <summary>
        /// If <i>false</i>, normalization is performed over the current mini-batch
        /// and global statistics are accumulated (but not yet used) by a moving
        /// average.
        /// If <i>true</i>, those accumulated mean and variance values are used for the 
        /// normalization.
        /// By default, this is set to <i>false</i> when the network is in Phase.TRAINING,
        /// and <i>true</i> when the network is in the Phase.TESTING mode.
        /// </summary>
        [Description("If false, accumulate global mean/variance values via a moving average. If true, use those accumulated values instead of computing mean/variance accross the batch.")]
        public bool? use_global_stats
        {
            get { return m_bUseGlobalStats; }
            set { m_bUseGlobalStats = value; }
        }

        /// <summary>
        /// Specifies how much the moving average decays each iteration.  Smaller values
        /// make the moving average decay faster, giving more weight to the recent values.
        /// </summary>
        /// <remarks>
        /// Each iteration updates the moving average @f$_{t-1}@f$ with the current mean
        /// @f$ Y_t @f$ by @f$ S_t = (1-\beta)Y_t + \beta \cdot S_{t-1} @f$, where @f$ \beta @f$
        /// is the moving average fraction parameter.
        /// </remarks>
        [Description("Specifies how much the moving average decays each iteration.")]
        public double moving_average_fraction
        {
            get { return m_dfMovingAverageFraction; }
            set { m_dfMovingAverageFraction = value; }
        }

        /// <summary>
        /// Specifies a small value to add to the variance estimate so that we don't divide by
        /// zero.
        /// </summary>
        [Description("Specifies a small value to add to the variance estimate so that we don't divide by zero.")]
        public double eps
        {
            get { return m_dfEps; }
            set { m_dfEps = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            BatchNormParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is BatchNormParameter)
            {
                BatchNormParameter p = (BatchNormParameter)src;
                m_bUseGlobalStats = p.m_bUseGlobalStats;
                m_dfEps = p.m_dfEps;
                m_dfMovingAverageFraction = p.m_dfMovingAverageFraction;
                m_bScaleBias = p.m_bScaleBias;
                m_biasFiller = p.m_biasFiller;

                if (p.m_scaleFiller != null)
                    m_scaleFiller = p.m_scaleFiller.Clone();
                else
                    m_scaleFiller = null;

                if (p.m_biasFiller != null)
                    m_biasFiller = p.m_biasFiller.Clone();
                else
                    m_biasFiller = null;
            }
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            BatchNormParameter p = new BatchNormParameter();
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

            if (use_global_stats.HasValue)
                rgChildren.Add("use_global_stats", use_global_stats.Value.ToString());

            if (moving_average_fraction != 0.999)
                rgChildren.Add("moving_average_fraction", moving_average_fraction.ToString());

            if (eps != 1e-5)
                rgChildren.Add("eps", eps.ToString());

            if (scale_bias)
            {
                rgChildren.Add("scale_bias", scale_bias.ToString());

                if (scale_filler != null)
                    rgChildren.Add(scale_filler.ToProto("scale_filler"));

                if (bias_filler != null)
                    rgChildren.Add(bias_filler.ToProto("bias_filler"));
            }
            
            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new BatchNormParameter FromProto(RawProto rp)
        {
            string strVal;
            BatchNormParameter p = new BatchNormParameter();

            ((EngineParameter)p).Copy(EngineParameter.FromProto(rp));

            if ((strVal = rp.FindValue("use_global_stats")) != null)
                p.use_global_stats = bool.Parse(strVal);

            if ((strVal = rp.FindValue("moving_average_fraction")) != null)
                p.moving_average_fraction = ParseDouble(strVal);

            if ((strVal = rp.FindValue("eps")) != null)
                p.eps = ParseDouble(strVal);

            if ((strVal = rp.FindValue("scale_bias")) != null)
            {
                p.scale_bias = bool.Parse(strVal);

                RawProto rp1;

                if ((rp1 = rp.FindChild("scale_filler")) != null)
                    p.scale_filler = FillerParameter.FromProto(rp1);

                if ((rp1 = rp.FindChild("bias_filler")) != null)
                    p.bias_filler = FillerParameter.FromProto(rp1);
            }

            return p;
        }
    }
}
