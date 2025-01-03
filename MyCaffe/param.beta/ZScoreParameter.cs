using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.beta
{
    /// <summary>
    /// Stores the parameters used by the ZScoreLayer - this layer can be helpful in regression models to normalize the output.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ZScoreParameter : LayerParameterBase
    {
        string m_strSource = "Data";
        string m_strMeanParamPos = "Mean";
        string m_strStdParamPos = "Stdev";
        string m_strMeanParamNeg = "Mean";
        string m_strStdParamNeg = "Stdev";
        bool m_bEnabled = true;
        SCORE_AS_LABEL_NORMALIZATION m_scoreMethod = SCORE_AS_LABEL_NORMALIZATION.Z_SCORE;

        /** @copydoc LayerParameterBase */
        public ZScoreParameter()
            : base()
        {
        }

        /// <summary>
        /// Returns the reason that Caffe version was used instead of [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns></returns>
        public string useCaffeReason()
        {
            return "Currenly only CAFFE supported.";
        }

        /// <summary>
        /// Queries whether or not to use [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns>Returns <i>true</i> when cuDnn is to be used, <i>false</i> otherwise.</returns>
        /// <remarks>Currently, only CAFFE supported.</remarks>
        public bool useCudnn()
        {
            return false;
        }

        /// <summary>
        /// Specifies whether or not the z-score normalization is enabled.
        /// </summary>
        [Description("Specifies whether or not the z-score normalization is enabled.")]
        public bool enabled
        {
            get { return m_bEnabled; }
            set { m_bEnabled = value; }
        }

        /// <summary>
        /// Specifies the data source name that contains an image parameter for the mean and stdev values.
        /// </summary>
        [Description("Specifies the data source name that contains an image parameter for the mean and stdev values.")]
        public string source
        {
            get { return m_strSource; }
            set { m_strSource = value; }
        }

        /// <summary>
        /// Specifies the method used to normalize the scores.
        /// </summary>
        public SCORE_AS_LABEL_NORMALIZATION score_method
        {
            get { return m_scoreMethod; }
            set { m_scoreMethod = value; }
        }

        /// <summary>
        /// Specifies the parameter used to query the data source for the mean value used to normalize positive values (or all values when method = Z_SCORE).
        /// </summary>
        [Description("Specifies the parameter used to query the data source for the mean value used to normalize positive values (or all values when method = Z_SCORE).")] 
        public string mean_pos_param
        {
            get { return m_strMeanParamPos; }
            set { m_strMeanParamPos = value; }
        }

        /// <summary>
        /// Specifies the parameter used to query the data source for the stdev value used to normalize positive values (or all values when method = Z_SCORE).
        /// </summary>
        [Description("Specifies the parameter used to query the data source for the stdev value used to normalize positive values (or all values when method = Z_SCORE).")]
        public string stdev_pos_param
        {
            get { return m_strStdParamPos; }
            set { m_strStdParamPos = value; }
        }

        /// <summary>
        /// Specifies the parameter used to query the data source for the mean value used to normalize negative values.
        /// </summary>
        [Description("Specifies the parameter used to query the data source for the mean value used to normalize negative values.")]
        public string mean_neg_param
        {
            get { return m_strMeanParamNeg; }
            set { m_strMeanParamNeg = value; }
        }

        /// <summary>
        /// Specifies the parameter used to query the data source for the stdev value used to normalize negative values.
        /// </summary>
        [Description("Specifies the parameter used to query the data source for the stdev value used to normalize negative values.")]
        public string stdev_neg_param
        {
            get { return m_strStdParamNeg; }
            set { m_strStdParamNeg = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ZScoreParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc EngineParameter::Copy */
        public override void Copy(LayerParameterBase src)
        {
            if (src is ZScoreParameter)
            {
                ZScoreParameter p = (ZScoreParameter)src;
                m_strSource = p.m_strSource;
                m_strMeanParamPos = p.m_strMeanParamPos;
                m_strStdParamPos = p.m_strStdParamPos;
                m_strMeanParamNeg = p.m_strMeanParamNeg;
                m_strStdParamNeg = p.m_strStdParamNeg;
                m_bEnabled = p.m_bEnabled;
                m_scoreMethod = p.m_scoreMethod;
            }
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            ZScoreParameter p = new ZScoreParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc EngineParameter::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("enabled", enabled.ToString());
            rgChildren.Add("source", source);
            rgChildren.Add("mean_pos_param", mean_pos_param);
            rgChildren.Add("stdev_pos_param", stdev_pos_param);
            rgChildren.Add("mean_neg_param", mean_neg_param);
            rgChildren.Add("stdev_neg_param", stdev_neg_param);
            rgChildren.Add("score_method", score_method.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ZScoreParameter FromProto(RawProto rp)
        {
            string strVal;
            ZScoreParameter p = new ZScoreParameter();

            if ((strVal = rp.FindValue("enabled")) != null)
                p.enabled = bool.Parse(strVal);

            if ((strVal = rp.FindValue("source")) != null)
                p.source = strVal;

            if ((strVal = rp.FindValue("mean_param")) != null)
                p.mean_pos_param = strVal;

            if ((strVal = rp.FindValue("stdev_param")) != null)
                p.stdev_pos_param = strVal;

            if ((strVal = rp.FindValue("mean_pos_param")) != null)
                p.mean_pos_param = strVal;

            if ((strVal = rp.FindValue("stdev_pos_param")) != null)
                p.stdev_pos_param = strVal;

            if ((strVal = rp.FindValue("mean_neg_param")) != null)
                p.mean_neg_param = strVal;

            if ((strVal = rp.FindValue("stdev_neg_param")) != null)
                p.stdev_neg_param = strVal;

            if ((strVal = rp.FindValue("score_method")) != null)
                p.score_method = (SCORE_AS_LABEL_NORMALIZATION)Enum.Parse(typeof(SCORE_AS_LABEL_NORMALIZATION), strVal, true);

            return p;
        }
    }
}
