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
    public class BatchNormParameter : LayerParameterBase 
    {
        bool? m_bUseGlobalStats = null;
        double m_dfMovingAverageFraction = 0.999;
        double m_dfEps = 1e-5;

        /** @copydoc LayerParameterBase */
        public BatchNormParameter()
        {
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
            BatchNormParameter p = (BatchNormParameter)src;
            p.m_bUseGlobalStats = m_bUseGlobalStats;
            p.m_dfEps = m_dfEps;
            p.m_dfMovingAverageFraction = m_dfMovingAverageFraction;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            BatchNormParameter p = new BatchNormParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (use_global_stats.HasValue)
                rgChildren.Add("use_global_stats", use_global_stats.Value.ToString());

            if (moving_average_fraction != 0.999)
                rgChildren.Add("moving_average_fraction", moving_average_fraction.ToString());

            if (eps != 1e-5)
                rgChildren.Add("eps", eps.ToString());
            
            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static BatchNormParameter FromProto(RawProto rp)
        {
            string strVal;
            BatchNormParameter p = new BatchNormParameter();

            if ((strVal = rp.FindValue("use_global_stats")) != null)
                p.use_global_stats = bool.Parse(strVal);

            if ((strVal = rp.FindValue("moving_average_fraction")) != null)
                p.moving_average_fraction = double.Parse(strVal);

            if ((strVal = rp.FindValue("eps")) != null)
                p.eps = double.Parse(strVal);

            return p;
        }
    }
}
