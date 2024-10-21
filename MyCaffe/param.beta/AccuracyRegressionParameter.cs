using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the AccuracyRegressionLayer.
    /// </summary>
    /// <remarks>
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class AccuracyRegressionParameter : LayerParameterBase
    {
        ALGORITHM m_alg = ALGORITHM.MAPE;
        double m_dfBucketMin = -2.0;
        double m_dfBucketMax = 2.0;
        int m_nBucketCount = 10;
        double? m_dfIgnoreAbove = null;
        double? m_dfIgnoreBelow = null;

        /// <summary>
        /// Defines the MAPE algorithm to use.
        /// </summary>
        public enum ALGORITHM
        {
            /// <summary>
            /// Defines the Mean Absolute Percentage Error algorithm.
            /// </summary>
            MAPE,
            /// <summary>
            /// Defines the Symmetric Mean Absolute Percentage Error algorithm.
            /// </summary>
            SMAPE,
            /// <summary>
            /// Defines the Bucketed method of calculating accuracy.
            /// </summary>
            BUCKETING
        }

        /** @copydoc LayerParameterBase */
        public AccuracyRegressionParameter()
        {
        }

        /// <summary>
        /// Specifies the algorithm to use: MAPE or SMAPE.
        /// </summary>
        [Description("Specifies algorithm used in the MAPE, SMAPE or BUCKETING calculation.")]
        public ALGORITHM algorithm
        {
            get { return m_alg; }
            set { m_alg = value; }
        }

        /// <summary>
        /// Ignore all scores above this value (default = null).
        /// </summary>
        public double? bucket_ignore_above
        {
            get { return m_dfIgnoreAbove; }
            set { m_dfIgnoreAbove = value; }
        }

        /// <summary>
        /// Ignore all scores below this value (default = null).
        /// </summary>
        public double? bucket_ignore_below
        {
            get { return m_dfIgnoreBelow; }
            set { m_dfIgnoreBelow = value; }
        }

        /// <summary>
        /// Specifies the minimum value of the bucket range used with BUCKETING.
        /// </summary>
        [Description("Specifies the minimum value of the bucket range used with BUCKETING.")]
        public double bucket_min
        {
            get { return m_dfBucketMin; }
            set { m_dfBucketMin = value; }
        }

        /// <summary>
        /// Specifies the maximum value of the bucket range used with BUCKETING.
        /// </summary>
        [Description("Specifies the maximum value of the bucket range used with BUCKETING.")]
        public double bucket_max
        {
            get { return m_dfBucketMax; }
            set { m_dfBucketMax = value; }
        }

        /// <summary>
        /// Specifies the number of buckets to use with BUCKETING.
        /// </summary>
        [Description("Specifies the number of buckets to use with BUCKETING (bucket count must be 2 or greater).")]
        public int bucket_count
        {
            get { return m_nBucketCount; }
            set { m_nBucketCount = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            AccuracyRegressionParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            AccuracyRegressionParameter p = (AccuracyRegressionParameter)src;

            m_alg = p.m_alg;
            m_dfBucketMin = p.m_dfBucketMin;
            m_dfBucketMax = p.m_dfBucketMax;
            m_nBucketCount = p.m_nBucketCount;
            m_dfIgnoreAbove = p.m_dfIgnoreAbove;
            m_dfIgnoreBelow = p.m_dfIgnoreBelow;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            AccuracyRegressionParameter p = new AccuracyRegressionParameter();
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
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("algorithm", algorithm.ToString());
            rgChildren.Add("bucket_min", bucket_min.ToString());
            rgChildren.Add("bucket_max", bucket_max.ToString());
            rgChildren.Add("bucket_count", bucket_count.ToString());

            if (bucket_ignore_below.HasValue)
                rgChildren.Add("bucket_ignore_below", bucket_ignore_below.Value.ToString());
            if (bucket_ignore_above.HasValue)
                rgChildren.Add("bucket_ignore_above", bucket_ignore_above.Value.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static AccuracyRegressionParameter FromProto(RawProto rp)
        {
            string strVal;
            AccuracyRegressionParameter p = new AccuracyRegressionParameter();

            if ((strVal = rp.FindValue("algorithm")) != null)
                p.algorithm = (ALGORITHM)Enum.Parse(typeof(ALGORITHM), strVal, true);

            if ((strVal = rp.FindValue("bucket_min")) != null)
                p.bucket_min = BaseParameter.ParseDouble(strVal);

            if ((strVal = rp.FindValue("bucket_max")) != null)
                p.bucket_max = BaseParameter.ParseDouble(strVal);

            if ((strVal = rp.FindValue("bucket_count")) != null)
                p.bucket_count = int.Parse(strVal);

            if ((strVal = rp.FindValue("bucket_ignore_above")) != null)
                p.bucket_ignore_above = BaseParameter.ParseDouble(strVal);

            if ((strVal = rp.FindValue("bucket_ignore_below")) != null)
                p.bucket_ignore_below = BaseParameter.ParseDouble(strVal);

            return p;
        }
    }
}
