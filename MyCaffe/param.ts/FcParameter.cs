using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.ts
{
    /// <summary>
    /// Specifies the parameters for the FcParameter (used by FcLayer in N-HiTS models).  
    /// </summary>
    /// <remarks>
    /// This layer performs the Linear projection, Activation, Normalization, and Dropout operations.
    /// 
    /// @see [Understanding N-HiTS Time Series Prediction](https://www.signalpop.com/2024/05/29/n-hits/) by Brown, 2024, SignalPop
    /// @see [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting](https://arxiv.org/abs/2201.12886) by Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, and Artur Dubrawski, 2022, arXiv:2201.12886.
    /// @see [Darts: User-Friendly Modern Machine Learning for Time Series](https://jmlr.org/papers/v23/21-1177.html) by Julien Herzen, Francesco Lässig, Samuele Giuliano Piazzetta, Thomas Neuer, Léo Tafti, Guillaume Raille, Tomas Van Pottelbergh, Marek Pasieka, Andrzej Skrodzki, Nicolas Huguenin, Maxime Dumonal, Jan Kościsz, Dennis Bader, Frédérick Gusset, Mounir Benheddi, Camila Williamson, Michal Kosinski, Matej Petrik, and Gaël Grosch, 2022, JMLR
    /// @see [Github - unit8co/darts](https://github.com/unit8co/darts) by unit8co, 2022, GitHub.
    /// 
    /// WORK IN PROGRESS.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class FcParameter : LayerParameterBase
    {
        int m_nAxis = 2;
        int m_nNumOutput = 512;
        bool m_bBiasTerm = true;
        float m_fDropoutRatio = 0.1f;
        bool m_bEnableNormalization = false;
        ACTIVATION m_act = ACTIVATION.RELU;

        /// <summary>
        /// Defines the mode of operation.
        /// </summary>
        public enum ACTIVATION
        {
            /// <summary>
            /// Specifies to ReLU activation.
            /// </summary>
            RELU = 0,
            /// <summary>
            /// Specifies to PRELU activation.
            /// </summary>
            PRELU = 1,
            /// <summary>
            /// Specifies to ELU activation.
            /// </summary>
            ELU = 2,
            /// <summary>
            /// Specifies to SoftPlus activation.
            /// </summary>
            SOFTPLUS = 3,
            /// <summary>
            /// Specifies to TANH activation.
            /// </summary>
            TANH = 4,
            /// <summary>
            /// Specifies to SIGMOID activation.
            /// </summary>
            SIGMOID = 5,
            /// <summary>
            /// Specifies to GELU activation.
            /// </summary>
            GELU = 6,
            /// <summary>
            /// Specifies to SWISH activation.
            /// </summary>
            SWISH = 7,
            /// <summary>
            /// Specifies to MISH activation.
            /// </summary>
            MISH = 8
        }

        /** @copydoc LayerParameterBase */
        public FcParameter()
        {
        }

        /// <summary>
        /// Specifies the axis.
        /// </summary>
        [Description("Specifies the axis.")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Specifies the number of outputs.
        /// </summary>
        [Description("Specifies the number of outputs.")]
        public int num_output
        {
            get { return m_nNumOutput; }
            set { m_nNumOutput = value; }
        }

        /// <summary>
        /// Specifies whether or not to use a bias term (default = true).
        /// </summary>
        [Description("Specifies whether or not to use a bias term (default = true).")]
        public bool bias_term
        {
            get { return m_bBiasTerm; }
            set { m_bBiasTerm = value; }
        }

        /// <summary>
        /// Specifies the ACTIVATION to use (default = ReLU).
        /// </summary>
        [Description("Specifies ACTIVATION to use (default = ReLU).")]
        public ACTIVATION activation
        {
            get { return m_act; }
            set { m_act = value; }
        }

        /// <summary>
        /// Specifies the dropout ratio (default = 0.1).  A value of 0 skips dropout altogether.
        /// </summary>
        [Description("Specifies the dropout ratio (default = 0.1).  A value of 0 skips dropout altogether.")]
        public float dropout_ratio
        {
            get { return m_fDropoutRatio; }
            set { m_fDropoutRatio = value; }
        }

        /// <summary>
        /// Specifies whether or not to enable normalization (default = false).
        /// </summary>
        [Description("Specifies whether or not to enable normalization (default = false).")]
        public bool enable_normalization
        {
            get { return m_bEnableNormalization; }
            set { m_bEnableNormalization = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            FcParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            FcParameter p = (FcParameter)src;

            m_nNumOutput = p.m_nNumOutput;
            m_act = p.m_act;
            m_fDropoutRatio = p.m_fDropoutRatio;
            m_bEnableNormalization = p.m_bEnableNormalization;
            m_nAxis = p.m_nAxis;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            FcParameter p = new FcParameter();
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
            rgChildren.Add("num_output", num_output.ToString());
            rgChildren.Add("activation", activation.ToString());
            rgChildren.Add("dropout_ratio", dropout_ratio.ToString());
            rgChildren.Add("enable_normalization", enable_normalization.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static FcParameter FromProto(RawProto rp)
        {
            string strVal;
            FcParameter p = new FcParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("num_output")) != null)
                p.num_output = int.Parse(strVal);

            if ((strVal = rp.FindValue("activation")) != null)
            {
                if (strVal == ACTIVATION.RELU.ToString())
                    p.activation = ACTIVATION.RELU;
                else
                    throw new Exception("Unknown activation '" + strVal + "'!");
            }

            if ((strVal = rp.FindValue("dropout_ratio")) != null)
                p.dropout_ratio = BaseParameter.ParseFloat(strVal);

            if ((strVal = rp.FindValue("enable_normalization")) != null)
                p.enable_normalization = bool.Parse(strVal);

            return p;
        }
    }
}
