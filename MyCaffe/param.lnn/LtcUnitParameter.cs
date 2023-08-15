using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.lnn
{
    /// <summary>
    /// Specifies the parameters for the LtcUnitLayer used by the CfCLayer.
    /// </summary>
    /// <remarks>
    /// @see [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898) by Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus, 2021, arXiv:2106.13898
    /// @see [Closed-form continuous-time neural networks](https://www.nature.com/articles/s42256-022-00556-7) by Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus, 2021, nature machine intelligence
    /// @see [GitHub:raminmh/CfC](https://github.com/raminmh/CfC) by Raminmn, 2021, GitHub (distributed under Apache 2.0 license)
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class LtcUnitParameter : LayerParameterBase
    {
        int m_nInputSize = 0;
        int m_nHiddenSize = 256;
        int m_nOdeUnfolds = 6;
        float m_fEpsilon = 1e-08f;
        float m_fGleakInitMin = 0.001f;
        float m_fGleakInitMax = 1.0f;
        float m_fVleakInitMin = -0.2f;
        float m_fVleakInitMax = 0.2f;
        float m_fCmInitMin = 0.4f;
        float m_fCmInitMax = 0.6f;
        float m_fWInitMin = 0.001f;
        float m_fWInitMax = 1.0f;
        float m_fSigmaInitMin = 3.0f;
        float m_fSigmaInitMax = 8.0f;
        float m_fMuInitMin = 0.3f;
        float m_fMuInitMax = 0.8f;
        float m_fSensoryWInitMin = 0.001f;
        float m_fSensoryWInitMax = 1.0f;
        float m_fSensorySigmaInitMin = 3.0f;
        float m_fSensorySigmaInitMax = 8.0f;
        float m_fSensoryMuInitMin = 0.3f;
        float m_fSensoryMuInitMax = 0.8f;

        /** @copydoc LayerParameterBase */
        public LtcUnitParameter()
        {
        }

        /// <summary>
        /// Specifies the input size.
        /// </summary>
        [Description("Specifies the input size.")]
        public int input_size
        {
            get { return m_nInputSize; }
            set { m_nInputSize = value; }
        }

        /// <summary>
        /// Specifies the number of hidden units (default = 256).
        /// </summary>
        [Description("Specifies the number of hidden units (default = 256).")]
        public int hidden_size
        {
            get { return m_nHiddenSize; }
            set { m_nHiddenSize = value; }
        }

        /// <summary>
        /// Specifies the number of unfolds run by the ode (default = 6).
        /// </summary>
        [Description("Specifies the number of unfolds run by the ode (default = 6).")]
        public int ode_unfolds
        {
            get { return m_nOdeUnfolds; }
            set { m_nOdeUnfolds = value; }
        }

        /// <summary>
        /// Specifies the epsilon used to avoid divide by zero (default = 1e-08).
        /// </summary>
        [Description("Specifies the epsilon used to avoid divide by zero (default = 1e-08).")]
        public float epsilon
        {
            get { return m_fEpsilon; }
            set { m_fEpsilon = value; }
        }

        /// <summary>
        /// Specifies the initial gleak min value (default = 0.001f).
        /// </summary>
        [Description("Specifies the initial gleak min value (default = 0.001f).")]
        public float gleak_init_min
        {
            get { return m_fGleakInitMin; }
            set { m_fGleakInitMin = value; }
        }

        /// <summary>
        /// Specifies the initial gleak max value (default = 1.0f).
        /// </summary>
        [Description("Specifies the initial gleak max value (default = 1.0f).")]
        public float gleak_init_max
        {
            get { return m_fGleakInitMax; }
            set { m_fGleakInitMax = value; }
        }

        /// <summary>
        /// Specifies the initial vleak min value (default = -0.2f).
        /// </summary>
        [Description("Specifies the initial vleak min value (default = -0.2f).")]
        public float vleak_init_min
        {
            get { return m_fVleakInitMin; }
            set { m_fVleakInitMin = value; }
        }

        /// <summary>
        /// Specifies the initial vleak max value (default = 0.2f).
        /// </summary>
        [Description("Specifies the initial vleak max value (default = 0.2f).")]
        public float vleak_init_max
        {
            get { return m_fVleakInitMax; }
            set { m_fVleakInitMax = value; }
        }

        /// <summary>
        /// Specifies the initial cm min value (default = 0.4f).
        /// </summary>
        [Description("Specifies the initial cm min value (default = 0.4f).")]
        public float cm_init_min
        {
            get { return m_fCmInitMin; }
            set { m_fCmInitMin = value; }
        }

        /// <summary>
        /// Specifies the initial cm max value (default = 0.6f).
        /// </summary>
        [Description("Specifies the initial cm max value (default = 0.6f).")]
        public float cm_init_max
        {
            get { return m_fCmInitMax; }
            set { m_fCmInitMax = value; }
        }

        /// <summary>
        /// Specifies the initial w min value (default = 0.001f).
        /// </summary>
        [Description("Specifies the initial w min value (default = 0.001f).")]
        public float w_init_min
        {
            get { return m_fWInitMin; }
            set { m_fWInitMin = value; }
        }

        /// <summary>
        /// Specifies the initial w max value (default = 1.0f).
        /// </summary>
        [Description("Specifies the initial w max value (default = 1.0f).")]
        public float w_init_max
        {
            get { return m_fWInitMax; }
            set { m_fWInitMax = value; }
        }

        /// <summary>
        /// Specifies the initial sigma min value (default = 3.0f).
        /// </summary>
        [Description("Specifies the initial sigma min value (default = 3.0f).")]
        public float sigma_init_min
        {
            get { return m_fSigmaInitMin; }
            set { m_fSigmaInitMin = value; }
        }

        /// <summary>
        /// Specifies the initial sigma max value (default = 8.0f).
        /// </summary>
        [Description("Specifies the initial sigma max value (default = 8.0f).")]
        public float sigma_init_max
        {
            get { return m_fSigmaInitMax; }
            set { m_fSigmaInitMax = value; }
        }

        /// <summary>
        /// Specifies the initial mu min value (default = 0.3f).
        /// </summary>
        [Description("Specifies the initial mu min value (default = 0.3f).")]
        public float mu_init_min
        {
            get { return m_fMuInitMin; }
            set { m_fMuInitMin = value; }
        }

        /// <summary>
        /// Specifies the initial mu max value (default = 0.8f).
        /// </summary>
        [Description("Specifies the initial mu max value (default = 0.8f).")]
        public float mu_init_max
        {
            get { return m_fMuInitMax; }
            set { m_fMuInitMax = value; }
        }

        /// <summary>
        /// Specifies the initial sensory_w min value (default = 0.001f).
        /// </summary>
        [Description("Specifies the initial sensory_w min value (default = 0.001f).")]
        public float sensory_w_init_min
        {
            get { return m_fSensoryWInitMin; }
            set { m_fSensoryWInitMin = value; }
        }

        /// <summary>
        /// Specifies the initial sensory_w max value (default = 1.0f).
        /// </summary>
        [Description("Specifies the initial sensory_w max value (default = 1.0f).")]
        public float sensory_w_init_max
        {
            get { return m_fSensoryWInitMax; }
            set { m_fSensoryWInitMax = value; }
        }

        /// <summary>
        /// Specifies the initial sensory_sigma min value (default = 3.0f).
        /// </summary>
        [Description("Specifies the initial sensory_sigma min value (default = 3.0f).")]
        public float sensory_sigma_init_min
        {
            get { return m_fSensorySigmaInitMin; }
            set { m_fSensorySigmaInitMin = value; }
        }

        /// <summary>
        /// Specifies the initial sensory_sigma max value (default = 8.0f).
        /// </summary>
        [Description("Specifies the initial sensory_sigma max value (default = 8.0f).")]
        public float sensory_sigma_init_max
        {
            get { return m_fSensorySigmaInitMax; }
            set { m_fSensorySigmaInitMax = value; }
        }

        /// <summary>
        /// Specifies the initial sensory_mu min value (default = 0.3f).
        /// </summary>
        [Description("Specifies the initial sensory_mu min value (default = 0.3f).")]
        public float sensory_mu_init_min
        {
            get { return m_fSensoryMuInitMin; }
            set { m_fSensoryMuInitMin = value; }
        }

        /// <summary>
        /// Specifies the initial sensory_mu max value (default = 0.8f).
        /// </summary>
        [Description("Specifies the initial sensory_mu max value (default = 0.8f).")]
        public float sensory_mu_init_max
        {
            get { return m_fSensoryMuInitMax; }
            set { m_fSensoryMuInitMax = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            LtcUnitParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            LtcUnitParameter p = (LtcUnitParameter)src;

            m_nInputSize = p.input_size;
            m_nHiddenSize = p.hidden_size;
            m_nOdeUnfolds = p.ode_unfolds;
            m_fEpsilon = p.epsilon;

            m_fGleakInitMin = p.gleak_init_min;
            m_fGleakInitMax = p.gleak_init_max;
            m_fVleakInitMin = p.vleak_init_min;
            m_fVleakInitMax = p.vleak_init_max;
            m_fCmInitMin = p.cm_init_min;
            m_fCmInitMax = p.cm_init_max;
            m_fWInitMin = p.w_init_min;
            m_fWInitMax = p.w_init_max;
            m_fSigmaInitMin = p.sigma_init_min;
            m_fSigmaInitMax = p.sigma_init_max;
            m_fMuInitMin = p.mu_init_min;
            m_fMuInitMax = p.mu_init_max;
            m_fSensoryWInitMin = p.sensory_w_init_min;
            m_fSensoryWInitMax = p.sensory_w_init_max;
            m_fSensorySigmaInitMin = p.sensory_sigma_init_min;
            m_fSensorySigmaInitMax = p.sensory_sigma_init_max;
            m_fSensoryMuInitMin = p.sensory_mu_init_min;
            m_fSensoryMuInitMax = p.sensory_mu_init_max;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            LtcUnitParameter p = new LtcUnitParameter();
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

            rgChildren.Add("input_size", input_size.ToString());
            rgChildren.Add("hidden_size", hidden_size.ToString());
            rgChildren.Add("ode_unfolds", ode_unfolds.ToString());
            rgChildren.Add("epsilon", epsilon.ToString());

            rgChildren.Add("gleak_init_min", gleak_init_min.ToString());
            rgChildren.Add("gleak_init_max", gleak_init_max.ToString());
            rgChildren.Add("vleak_init_min", vleak_init_min.ToString());
            rgChildren.Add("vleak_init_max", vleak_init_max.ToString());
            rgChildren.Add("cm_init_min", cm_init_min.ToString());
            rgChildren.Add("cm_init_max", cm_init_max.ToString());
            rgChildren.Add("w_init_min", w_init_min.ToString());
            rgChildren.Add("w_init_max", w_init_max.ToString());
            rgChildren.Add("sigma_init_min", sigma_init_min.ToString());
            rgChildren.Add("sigma_init_max", sigma_init_max.ToString());
            rgChildren.Add("mu_init_min", mu_init_min.ToString());
            rgChildren.Add("mu_init_max", mu_init_max.ToString());
            rgChildren.Add("sensory_w_init_min", sensory_w_init_min.ToString());
            rgChildren.Add("sensory_w_init_max", sensory_w_init_max.ToString());
            rgChildren.Add("sensory_sigma_init_min", sensory_sigma_init_min.ToString());
            rgChildren.Add("sensory_sigma_init_max", sensory_sigma_init_max.ToString());
            rgChildren.Add("sensory_mu_init_min", sensory_mu_init_min.ToString());
            rgChildren.Add("sensory_mu_init_max", sensory_mu_init_max.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static LtcUnitParameter FromProto(RawProto rp)
        {
            string strVal;
            LtcUnitParameter p = new LtcUnitParameter();

            if ((strVal = rp.FindValue("input_size")) != null)
                p.input_size = int.Parse(strVal);

            if ((strVal = rp.FindValue("hidden_size")) != null)
                p.hidden_size = int.Parse(strVal);

            if ((strVal = rp.FindValue("ode_unfolds")) != null)
                p.ode_unfolds = int.Parse(strVal);

            if ((strVal = rp.FindValue("epsilon")) != null)
                p.epsilon = ParseFloat(strVal);

            if ((strVal = rp.FindValue("gleak_init_min")) != null)
                p.gleak_init_min = ParseFloat(strVal);

            if ((strVal = rp.FindValue("gleak_init_max")) != null)
                p.gleak_init_max = ParseFloat(strVal);

            if ((strVal = rp.FindValue("vleak_init_min")) != null)
                p.vleak_init_min = ParseFloat(strVal);

            if ((strVal = rp.FindValue("vleak_init_max")) != null)
                p.vleak_init_max = ParseFloat(strVal);

            if ((strVal = rp.FindValue("cm_init_min")) != null)
                p.cm_init_min = ParseFloat(strVal);

            if ((strVal = rp.FindValue("cm_init_max")) != null)
                p.cm_init_max = ParseFloat(strVal);

            if ((strVal = rp.FindValue("w_init_min")) != null)
                p.w_init_min = ParseFloat(strVal);

            if ((strVal = rp.FindValue("w_init_max")) != null)
                p.w_init_max = ParseFloat(strVal);

            if ((strVal = rp.FindValue("sigma_init_min")) != null)
                p.sigma_init_min = ParseFloat(strVal);

            if ((strVal = rp.FindValue("sigma_init_max")) != null)
                p.sigma_init_max = ParseFloat(strVal);

            if ((strVal = rp.FindValue("mu_init_min")) != null)
                p.mu_init_min = ParseFloat(strVal);

            if ((strVal = rp.FindValue("mu_init_max")) != null)
                p.mu_init_max = ParseFloat(strVal);

            if ((strVal = rp.FindValue("sensory_w_init_min")) != null)
                p.sensory_w_init_min = ParseFloat(strVal);

            if ((strVal = rp.FindValue("sensory_w_init_max")) != null)
                p.sensory_w_init_max = ParseFloat(strVal);

            if ((strVal = rp.FindValue("sensory_sigma_init_min")) != null)
                p.sensory_sigma_init_min = ParseFloat(strVal);

            if ((strVal = rp.FindValue("sensory_sigma_init_max")) != null)
                p.sensory_sigma_init_max = ParseFloat(strVal);

            if ((strVal = rp.FindValue("sensory_mu_init_min")) != null)
                p.sensory_mu_init_min = ParseFloat(strVal);

            if ((strVal = rp.FindValue("sensory_mu_init_max")) != null)
                p.sensory_mu_init_max = ParseFloat(strVal);

            return p;
        }
    }
}
