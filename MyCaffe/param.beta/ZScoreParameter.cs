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
        string m_strMeanParam = "Mean";
        string m_strStdParam = "Stdev";
        bool m_bEnabled = true;

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
        /// Specifies the parameter used to query the data source for the mean value.
        /// </summary>
        [Description("Specifies the parameter used to query the data source for the mean value.")] 
        public string mean_param
        {
            get { return m_strMeanParam; }
            set { m_strMeanParam = value; }
        }

        /// <summary>
        /// Specifies the parameter used to query the data source for the stdev value.
        /// </summary>
        [Description("Specifies the parameter used to query the data source for the stdev value.")]
        public string stdev_param
        {
            get { return m_strStdParam; }
            set { m_strStdParam = value; }
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
                m_strMeanParam = p.m_strMeanParam;
                m_strStdParam = p.m_strStdParam;
                m_bEnabled = p.m_bEnabled;
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
            rgChildren.Add("mean_param", mean_param);
            rgChildren.Add("stdev_param", stdev_param);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new ZScoreParameter FromProto(RawProto rp)
        {
            string strVal;
            ZScoreParameter p = new ZScoreParameter();

            if ((strVal = rp.FindValue("enabled")) != null)
                p.enabled = bool.Parse(strVal);

            if ((strVal = rp.FindValue("source")) != null)
                p.source = strVal;

            if ((strVal = rp.FindValue("mean_param")) != null)
                p.mean_param = strVal;

            if ((strVal = rp.FindValue("stdev_param")) != null)
                p.stdev_param = strVal;

            return p;
        }
    }
}
