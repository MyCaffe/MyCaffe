using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the MeanErrorLossLayerParameter.
    /// </summary>
    /// <remarks>
    /// Used with regression models, such as those used with time-series prediction.
    /// @see [Methods for forecasts of continuous variables](https://www.cawcr.gov.au/projects/verification/#Methods_for_foreasts_of_continuous_variables) by WCRP, 2017.
    /// @see [MAD vs RMSE vs MAE vs MSLE vs R^2: When to use which?](https://datascience.stackexchange.com/questions/42760/mad-vs-rmse-vs-mae-vs-msle-vs-r%C2%B2-when-to-use-which), StackExchange, 2018.
    /// @see [Mean Absolute Error](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-absolute-error) by Peltarion.
    /// @see [The Proximal Map of the Weighted Mean Absolute Error](https://arxiv.org/abs/2209.13545) by Lukas Baumgärtner, Roland Herzog, Stephan Schmidt, Manuel Weiß, 2022, arXiv 2209.13545
    /// @see [Mean Absolute Error In Machine Learning: What You Need To Know](https://arize.com/blog-course/mean-absolute-error-in-machine-learning-what-you-need-to-know/) by David Burch, 2023, ariz.com
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class MeanErrorLossParameter : LayerParameterBase
    {
        int m_nAxis = 1; // Axis used to calculate the loss normalization.
        MEAN_ERROR m_meanErrorType = MEAN_ERROR.MAE; // default to the Mean Absolute Error.
        bool m_bEnableWeightedLoss = false;
        float m_fMaxWeight = 10.0f;
        int m_nWeightBucketCount = 20;
        int m_nWeightWarmupIerations = 100;
        bool m_bEnableWeightPosNeg = false;

        float m_fAbovePenaltyPortionLambda = 1.0f;
        float? m_fPenalizeValuesAboveThreshold = null;
        float m_fBelowPenaltyPortionLambda = 1.0f;
        float? m_fPenalizeValuesBelowThreshold = null;


        /** @copydoc LayerParameterBase */
        public MeanErrorLossParameter()
        {
        }

        /// <summary>
        /// [\b optional, default = 1] Specifies the axis of the probability.
        /// </summary>
        [Description("Specifies the axis of the probability, default = 1")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// [\b optional, default = MSE] Specifies the type of mean error to use.
        /// </summary>
        [Description("Specifies the type of mean error to use, default = MSE")]
        public MEAN_ERROR mean_error_type
        {
            get { return m_meanErrorType; }
            set { m_meanErrorType = value; }
        }

        /// <summary>
        /// [\b optional, default = false] Specifies to use a weighted loss.
        /// </summary>
        [Description("Specifies to use a weighted loss, default = false")]
        public bool enable_weighted_loss
        {
            get { return m_bEnableWeightedLoss; }
            set { m_bEnableWeightedLoss = value; }
        }

        /// <summary>
        /// [\b optional, default = false] Specifies to enforce equal weighting between positive and negative targets.
        /// </summary>
        [Description("Specifies to enforce equal weighting between positive and negative targets.")]
        public bool enable_weight_posneg
        {
            get { return m_bEnableWeightPosNeg; }
            set { m_bEnableWeightPosNeg = value;}
        }

        /// <summary>
        /// [\b optional, default = 10.0] Specifies the maximum weight to use when using the weighted loss.
        /// </summary>
        [Description("Specifies the maximum weight to use when using the weighted loss, default = 10.0")]
        public float max_weight
        {
            get { return m_fMaxWeight; }
            set { m_fMaxWeight = value; }
        }

        /// <summary>
        /// [\b optional, default = 20] Specifies the number of weight buckets to use when using the weighted loss.
        /// </summary>
        [Description("Specifies the number of weight buckets to use when using the weighted loss, default = 20")]
        public int weight_bucket_count
        {
            get { return m_nWeightBucketCount; }
            set { m_nWeightBucketCount = value; }
        }

        /// <summary>
        /// [\b optional, default = 100] Specifies the number of warmup iterations to use when using the weighted loss.
        /// </summary>
        [Description("Specifies the number of warmup iterations to use when using the weighted loss, default = 100")]
        public int weight_warmup_iterations
        {
            get { return m_nWeightWarmupIerations; }
            set { m_nWeightWarmupIerations = value; }
        }

        /// <summary>
        /// Specifies the portion of the penalty to apply to values above the threshold.  Only used when penalize_values_above_threshold is set.
        /// </summary>
        [Description("Specifies the portion of the penalty to apply to values above the threshold.  Only used when penalize_values_above_threshold is set.")]
        public float above_penalty_portion_lambda
        {
            get { return m_fAbovePenaltyPortionLambda; }
            set { m_fAbovePenaltyPortionLambda = value; }
        }

        /// <summary>
        /// Specifies the threshold above which to penalize values.  If set, the above_penalty_portion_lambda is used to determine the portion of the penalty to apply.
        /// </summary>
        [Description("Specifies the threshold above which to penalize values.  If set, the above_penalty_portion_lambda is used to determine the portion of the penalty to apply.")]
        public float? penalize_values_above_threshold
        {
            get { return m_fPenalizeValuesAboveThreshold; }
            set { m_fPenalizeValuesAboveThreshold = value; }
        }

        /// <summary>
        /// Specifies the portion of the penalty to apply to values below the threshold.  Only used when penalize_values_below_threshold is set.
        /// </summary>
        [Description("Specifies the portion of the penalty to apply to values below the threshold.  Only used when penalize_values_below_threshold is set.")]
        public float below_penalty_portion_lambda
        {
            get { return m_fBelowPenaltyPortionLambda; }
            set { m_fBelowPenaltyPortionLambda = value; }
        }

        /// <summary>
        /// Specifies the threshold below which to penalize values.  If set, the below_penalty_portion_lambda is used to determine the portion of the penalty to apply.
        /// </summary>
        [Description("Specifies the threshold below which to penalize values.  If set, the below_penalty_portion_lambda is used to determine the portion of the penalty to apply.")]
        public float? penalize_values_below_threshold
        {
            get { return m_fPenalizeValuesBelowThreshold; }
            set { m_fPenalizeValuesBelowThreshold = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            MeanErrorLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            MeanErrorLossParameter p = (MeanErrorLossParameter)src;
            m_nAxis = p.m_nAxis;
            m_meanErrorType = p.m_meanErrorType;
            m_fAbovePenaltyPortionLambda = p.m_fAbovePenaltyPortionLambda;
            m_fPenalizeValuesAboveThreshold = p.m_fPenalizeValuesAboveThreshold;
            m_fBelowPenaltyPortionLambda = p.m_fBelowPenaltyPortionLambda;
            m_fPenalizeValuesBelowThreshold = p.m_fPenalizeValuesBelowThreshold;
            m_bEnableWeightedLoss = p.m_bEnableWeightedLoss;
            m_fMaxWeight = p.m_fMaxWeight;
            m_nWeightBucketCount = p.m_nWeightBucketCount;
            m_nWeightWarmupIerations = p.m_nWeightWarmupIerations;
            m_bEnableWeightPosNeg = p.m_bEnableWeightPosNeg;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            MeanErrorLossParameter p = new MeanErrorLossParameter();
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

            rgChildren.Add("axis", axis.ToString());
            rgChildren.Add("mean_error_type", mean_error_type.ToString());
            rgChildren.Add("enable_weighted_loss", enable_weighted_loss.ToString());
            rgChildren.Add("enable_weight_posneg", enable_weight_posneg.ToString());
            rgChildren.Add("max_weight", max_weight.ToString());
            rgChildren.Add("weight_bucket_count", weight_bucket_count.ToString());
            rgChildren.Add("weight_warmup_iterations", weight_warmup_iterations.ToString());

            if (penalize_values_above_threshold.HasValue)
            {
                rgChildren.Add("penalize_values_above_threshold", penalize_values_above_threshold.Value.ToString());
                rgChildren.Add("above_penalty_portion_lambda", above_penalty_portion_lambda.ToString());
            }

            if (penalize_values_below_threshold.HasValue)
            {
                rgChildren.Add("penalize_values_below_threshold", penalize_values_below_threshold.Value.ToString());
                rgChildren.Add("below_penalty_portion_lambda", below_penalty_portion_lambda.ToString());
            }

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static MeanErrorLossParameter FromProto(RawProto rp)
        {
            string strVal;
            MeanErrorLossParameter p = new MeanErrorLossParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("mean_error_type")) != null)
            {
                switch (strVal)
                {
                    case "MSE":
                        p.mean_error_type = MEAN_ERROR.MSE;
                        break;
                    case "MAE":
                        p.mean_error_type = MEAN_ERROR.MAE;
                        break;
                    default:
                        throw new Exception("The mean _error_type parameter must be one of the following: MSE, MAE");
                }
            }

            if ((strVal = rp.FindValue("above_penalty_portion_lambda")) != null)
                p.above_penalty_portion_lambda = BaseParameter.ParseFloat(strVal);

            if ((strVal = rp.FindValue("penalize_values_above_threshold")) != null)
                p.penalize_values_above_threshold = BaseParameter.ParseFloat(strVal);

            if ((strVal = rp.FindValue("below_penalty_portion_lambda")) != null)
                p.below_penalty_portion_lambda = BaseParameter.ParseFloat(strVal);

            if ((strVal = rp.FindValue("penalize_values_below_threshold")) != null)
                p.penalize_values_below_threshold = BaseParameter.ParseFloat(strVal);

            if ((strVal = rp.FindValue("enable_weighted_loss")) != null)
                p.enable_weighted_loss = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_weight_posneg")) != null)
                p.enable_weight_posneg = bool.Parse(strVal);

            if ((strVal = rp.FindValue("max_weight")) != null)
                p.max_weight = BaseParameter.ParseFloat(strVal);

            if ((strVal = rp.FindValue("weight_bucket_count")) != null)
                p.weight_bucket_count = int.Parse(strVal);

            if ((strVal = rp.FindValue("weight_warmup_iterations")) != null)
                p.weight_warmup_iterations = int.Parse(strVal);

            return p;
        }
    }
}
