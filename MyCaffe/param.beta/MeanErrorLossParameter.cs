using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param.beta
{
    /// <summary>
    /// Specifies the parameters for the MeanErrorLossLayerParameter.
    /// </summary>
    /// <remarks>
    /// Used with regression models, such as those used with time-series prediction.
    /// @see [Methods for forecasts of continuous variables](https://www.cawcr.gov.au/projects/verification/#Methods_for_foreasts_of_continuous_variables) by WCRP, 2017.
    /// @see [MAD vs RMSE vs MAE vs MSLE vs R^2: When to use which?](https://datascience.stackexchange.com/questions/42760/mad-vs-rmse-vs-mae-vs-msle-vs-r%C2%B2-when-to-use-which), StackExchange, 2018.
    /// @see [Mean Absolute Error](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-absolute-error) by Peltarion.
    /// </remarks>
    public class MeanErrorLossParameter : LayerParameterBase
    {
        int m_nAxis = 1; // Axis used to calculate the loss normalization.
        MEAN_ERROR m_meanErrorType = MEAN_ERROR.MAE; // default to the Mean Absolute Error.


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
        public MEAN_ERROR mean_error_type
        {
            get { return m_meanErrorType; }
            set { m_meanErrorType = value; }
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
                    case "MSLE":
                        p.mean_error_type = MEAN_ERROR.MSLE;
                        break;
                    case "RMSE":
                        p.mean_error_type = MEAN_ERROR.RMSE;
                        break;
                    case "MAE":
                        p.mean_error_type = MEAN_ERROR.MAE;
                        break;
                    default:
                        throw new Exception("The mean _error_type parameter must be one of the following: MSE, MSLE, RMSE, MAE");
                }
            }

            return p;
        }
    }
}
