using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the filler parameters used to create each Filler.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class FillerParameter : BaseParameter, ICloneable 
    {
        string m_strType;
        double m_fVal;   // The value in constant filler.
        double m_fMin;   // The min value in uniform filler.
        double m_fMax;   // The max value in uniform filler.
        double m_fMean;  // The mean value in Gaussian filler.
        double m_fStd;   // The std value in Gaussian filler.

        // The expected number of non-zero output weights for a given input in
        // Guassian filler -- the default -1 means don't perform sparsification.
        int m_nSparse = -1;

        // Normalize the filler variance by fan_in, fan_out, or their average.
        // Applies to 'xavier' and 'msra' fillers.

        /// <summary>
        /// Defines the type of filler.
        /// </summary>
        public enum FillerType
        {
            /// <summary>
            /// The constant filler fills a blob with constant values.
            /// </summary>
            CONSTANT,
            /// <summary>
            /// The uniform filler fills a blob with values from a uniform distribution.
            /// </summary>
            /// <remarks>
            /// @see [Uniform Distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)) Wikipedia.
            /// </remarks>
            UNIFORM,
            /// <summary>
            /// The gaussian filler fills a blob with values from a gaussian distribution.
            /// </summary>
            /// <remarks>
            /// @see [Guassian Distribution](https://en.wikipedia.org/wiki/Normal_distribution) Wikipedia.
            /// </remarks>
            GAUSSIAN,
            /// <summary>
            /// The xavier filler fills a blob with values from a xavier distribution.
            /// </summary>
            /// <remarks>
            /// @see [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) by Glorot, Xavier and Bengio, Yoshua, 2010.
            /// </remarks>
            XAVIER,
            /// <summary>
            /// The msra filler fills a blob with values from a msra distribution.
            /// </summary>
            /// <remarks>
            /// @see [Learning hierarchical categories in deep neural networks](http://web.stanford.edu/class/psych209a/ReadingsByDate/02_15/SaxeMcCGanguli13.pdf) by Saxe, Andrew M. and McClelland, James L. and Ganguli, Surya, 2013.
            /// @see [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) by He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian, 2015.
            /// </remarks>
            MSRA,
            /// <summary>
            /// The positive unit ball filler fills a blob with values from a positive unit ball distribution.
            /// </summary>
            POSITIVEUNITBALL,
            /// <summary>
            /// The bilinear filler fills a blob with values from a bilinear distribution.
            /// </summary>
            BILINEAR
        }

        /// <summary>
        /// Defines the variance normalization.
        /// </summary>
        public enum VarianceNorm
        {
            /// <summary>
            /// Specifies a fan-in variance normalization.
            /// </summary>
            FAN_IN = 0,
            /// <summary>
            /// Specifies a fan-out variance normalization.
            /// </summary>
            FAN_OUT = 1,
            /// <summary>
            /// Specifies an average variance normalization.
            /// </summary>
            AVERAGE = 2
        }
        VarianceNorm m_varianceNorm = VarianceNorm.FAN_IN;

        /// <summary>
        /// Filler parameter constructor
        /// </summary>
        /// <remarks>
        /// NOTE: Caffe defaults to 'constant', however this causes models that do not specifically specify
        /// a filler to run with constant 0 filled weights.  Using a 'gaussian' as the default fixes
        /// this and fills the weights with random numbers.
        /// </remarks>
        /// <param name="strType">Optionally, specifies the type of filler to use.  Default = "gaussian"</param>
        /// <param name="dfVal">Optionally, specifies the value.  Default = 0.0</param>
        /// <param name="dfMean">Optionally, specifies the mean.  Default = 0.0</param>
        /// <param name="dfStd">Optionally, specifies the standard deviation.  Default = 1.0</param>
        public FillerParameter(string strType = "gaussian", double dfVal = 0.0, double dfMean = 0.0, double dfStd = 1.0)
        {
            m_strType = strType;
            m_fVal = dfVal;
            m_fMin = 0;
            m_fMax = 1;
            m_fMean = dfMean;
            m_fStd = dfStd;
        }

        /// <summary>
        /// Specifies the type of filler to use.
        /// </summary>
        [Description("Specifies the type of filler to use.")]
        [Browsable(false)]
        public string type
        {
            get { return m_strType; }
            set { m_strType = value; }
        }

#pragma warning disable 1591

        [DisplayName("type")]
        [Description("Specifies the type of filler type to use.")]
        public FillerType FillerTypeMethod /** @private */
        {
            get
            {
                switch (m_strType)
                {
                    case "constant":
                        return FillerType.CONSTANT;

                    case "uniform":
                        return FillerType.UNIFORM;

                    case "gaussian":
                        return FillerType.GAUSSIAN;

                    case "xavier":
                        return FillerType.XAVIER;

                    case "msra":
                        return FillerType.MSRA;

                    case "positive_unitball":
                        return FillerType.POSITIVEUNITBALL;

                    case "bilinear":
                        return FillerType.BILINEAR;

                    default:
                        throw new Exception("Unknown filler type '" + m_strType + "'");
                }
            }

            set { m_strType = FillerParameter.GetFillerName(value); }
        }

#pragma warning restore 1591

        /// <summary>
        /// Queries the filler text name corresponding to the FillerType.
        /// </summary>
        /// <param name="type">Specifies the FillerType.</param>
        /// <returns>The string associated with the FillerType is returned.</returns>
        public static string GetFillerName(FillerType type)
        {
            switch (type)
            {
                case FillerType.CONSTANT:
                    return "constant";

                case FillerType.UNIFORM:
                    return "uniform";

                case FillerType.GAUSSIAN:
                    return "gaussian";

                case FillerType.XAVIER:
                    return "xavier";

                case FillerType.MSRA:
                    return "msra";

                case FillerType.POSITIVEUNITBALL:
                    return "positive_unitball";

                case FillerType.BILINEAR:
                    return "bilinear";

                default:
                    throw new Exception("Unknown filler type '" + type.ToString() + "'");
            }
        }

        /// <summary>
        /// Specifies the value used by 'constant' filler.
        /// </summary>
        [Description("Specifies the value used by 'constant' filler.")]
        public double value
        {
            get { return m_fVal; }
            set { m_fVal = value; }
        }

        /// <summary>
        /// Specifies the minimum value to use with the 'uniform' filler.
        /// </summary>
        [Description("Specifies the minimum value to use with the 'uniform' filler.")]
        public double min
        {
            get { return m_fMin; }
            set { m_fMin = value; }
        }

        /// <summary>
        /// Specifies the maximum value to use with the 'uniform' filler.
        /// </summary>
        [Description("Specifies the maximum value to use with the 'uniform' filler.")]
        public double max
        {
            get { return m_fMax; }
            set { m_fMax = value; }
        }

        /// <summary>
        /// Specifies the mean value to use with the 'gaussian' filler.
        /// </summary>
        [Description("Specifies the mean value to use with the 'gaussian' filler.")]
        public double mean
        {
            get { return m_fMean; }
            set { m_fMean = value; }
        }

        /// <summary>
        /// Specifies the standard deviation value to use with the 'gaussian' filler.
        /// </summary>
        [Description("Specifies the standard deviation value to use with the 'gaussian' filler.")]
        public double std
        {
            get { return m_fStd; }
            set { m_fStd = value; }
        }

        /// <summary>
        /// Specifies the sparcity value to use with the 'guassian' filler.
        /// </summary>
        [Description("Specifies the sparcity value to use with the 'guassian' filler.")]
        public int sparse
        {
            get { return m_nSparse; }
            set { m_nSparse = value; }
        }

        /// <summary>
        /// Specifies the variance normalization method to use with the 'xavier' and 'mrsa' fillers.
        /// </summary>
        [Description("Specifies the variance normalization method to use with the 'xavier' and 'mrsa' fillers.")]
        public VarianceNorm variance_norm
        {
            get { return m_varianceNorm; }
            set { m_varianceNorm = value; }
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public FillerParameter Clone()
        {
            FillerParameter p = new FillerParameter();

            p.m_strType = m_strType;
            p.m_fVal = m_fVal;
            p.m_fMin = m_fMin;
            p.m_fMax = m_fMax;
            p.m_fMean = m_fMean;
            p.m_fStd = m_fStd;
            p.m_nSparse = m_nSparse;
            p.m_varianceNorm = m_varianceNorm;

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

            rgChildren.Add(new RawProto("type", "\"" + type + "\""));

            if (type.ToLower() == "constant")
                rgChildren.Add(new RawProto("value", value.ToString()));

            if (type.ToLower() == "uniform")
            {
                rgChildren.Add(new RawProto("min", min.ToString()));
                rgChildren.Add(new RawProto("max", max.ToString()));
            }

            if (type.ToLower() == "gaussian")
            {
                rgChildren.Add(new RawProto("mean", mean.ToString()));
                rgChildren.Add(new RawProto("std", std.ToString()));

                if (sparse != -1)
                    rgChildren.Add(new RawProto("sparse", sparse.ToString()));
            }

            if (type.ToLower() == "xavier" ||
                type.ToLower() == "mrsa")
                rgChildren.Add(new RawProto("variance_norm", variance_norm.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static FillerParameter FromProto(RawProto rp)
        {
            string strVal;

            if ((strVal = rp.FindValue("type")) == null)
                throw new Exception("Could not find 'type'");

            FillerParameter p = new FillerParameter(strVal);

            if ((strVal = rp.FindValue("value")) != null)
                p.value = double.Parse(strVal);

            if ((strVal = rp.FindValue("min")) != null)
                p.min = double.Parse(strVal);

            if ((strVal = rp.FindValue("max")) != null)
                p.max = double.Parse(strVal);

            if ((strVal = rp.FindValue("mean")) != null)
                p.mean = double.Parse(strVal);

            if ((strVal = rp.FindValue("std")) != null)
                p.std = double.Parse(strVal);

            if ((strVal = rp.FindValue("sparse")) != null)
                p.sparse = int.Parse(strVal);

            if ((strVal = rp.FindValue("variance_norm")) != null)
            {
                switch (strVal)
                {
                    case "FAN_IN":
                        p.variance_norm = VarianceNorm.FAN_IN;
                        break;

                    case "FAN_OUT":
                        p.variance_norm = VarianceNorm.FAN_OUT;
                        break;

                    case "AVERAGE":
                        p.variance_norm = VarianceNorm.AVERAGE;
                        break;
                        
                    default:
                        throw new Exception("Unknown 'variance_norm' value: " + strVal);
                }
            }

            return p;
        }

        object ICloneable.Clone()
        {
            return Clone();
        }
    }
}
