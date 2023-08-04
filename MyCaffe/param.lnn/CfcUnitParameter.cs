using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.lnn
{
    /// <summary>
    /// Specifies the parameters for the CfcUnitLayer used by the CfCLayer.
    /// </summary>
    /// <remarks>
    /// @see [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898) by Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus, 2021, arXiv:2106.13898
    /// @see [Closed-form continuous-time neural networks](https://www.nature.com/articles/s42256-022-00556-7) by Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus, 2021, nature machine intelligence
    /// @see [GitHub:raminmh/CfC](https://github.com/raminmh/CfC) by Raminmn, 2021, GitHub (distributed under Apache 2.0 license)
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class CfcUnitParameter : LayerParameterBase
    {
        bool m_bNoGate = false;
        bool m_bMinimal = false;
        int m_nInputSize = 0;
        int m_nHiddenSize = 0;
        int m_nBackboneUnits = 1;
        int m_nBackboneLayers = 1;
        float m_fBackboneDropout = 0.0f;
        ACTIVATION m_backboneActivation = ACTIVATION.SILU;

        /// <summary>
        /// Defines the activation function used by the backbone.
        /// </summary>
        public enum ACTIVATION
        {
            /// <summary>
            /// Specifies the SILU activation function.
            /// </summary>
            SILU,
            /// <summary>
            /// Specifies the RELU activation function.
            /// </summary>
            RELU,
            /// <summary>
            /// Specifies the TANH activation function.
            /// </summary>
            TANH,
            /// <summary>
            /// Specifies the GELU activation function.
            /// </summary>
            GELU,
            /// <summary>
            /// Specifies the LECUN activation function.
            /// </summary>
            LECUN
        }

        /** @copydoc LayerParameterBase */
        public CfcUnitParameter()
        {
        }

        /// <summary>
        /// Specifies whether to use the gate or not (when true, the no gate mode is used to calculate the forward output).
        /// </summary>
        [Description("Specifies whether to use the gate or not (when true, the no gate mode is used to calculate the forward output).")]
        public bool no_gate
        {
            get { return m_bNoGate; }
            set { m_bNoGate = value; }
        }

        /// <summary>
        /// Specifies whether to use the minimal model or not (when true, the minimal mode is used to calculate the forward output).
        /// </summary>
        [Description("Specifies whether to use the minimal model or not (when true, the minimal mode is used to calculate the forward output).")]
        public bool minimal
        {
            get { return m_bMinimal; }
            set { m_bMinimal = value; }
        }

        /// <summary>
        /// Specifies the input size used to size the backbone units.
        /// </summary>
        [Description("Specifies the input size used to size the backbone units.")]
        public int input_size
        {
            get { return m_nInputSize; }
            set { m_nInputSize = value; }
        }

        /// <summary>
        /// Specifies the hidden size used to size the backbone units and other internal layers.
        /// </summary>
        [Description("Specifies the hidden size used to size the backbone units and other internal layers.")]
        public int hidden_size
        {
            get { return m_nHiddenSize; }
            set { m_nHiddenSize = value; }
        }

        /// <summary>
        /// Specifies the number of backbone units
        /// </summary>
        [Description("Specifies the number of backbone units.")]
        public int backbone_units
        {
            get { return m_nBackboneUnits; }
            set { m_nBackboneUnits = value; }
        }

        /// <summary>
        /// Specifies the number of backbone layers.
        /// </summary>
        [Description("Specifies the number of backbone layers.")]
        public int backbone_layers
        {
            get { return m_nBackboneLayers; }
            set { m_nBackboneLayers = value; }
        }

        /// <summary>
        /// Specifies the backbone dropout ratio.
        /// </summary>
        [Description("Specifies the backbone dropout ratio.")]
        public float backbone_dropout_ratio
        {
            get { return m_fBackboneDropout; }
            set { m_fBackboneDropout = value; }
        }

        /// <summary>
        /// Specifies the backbone activation function.
        /// </summary>
        [Description("Specifies the backbone activation function.")]
        public ACTIVATION backbone_activation
        {
            get { return m_backboneActivation; }
            set { m_backboneActivation = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            CfcUnitParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            CfcUnitParameter p = (CfcUnitParameter)src;

            m_bNoGate = p.no_gate;
            m_bMinimal = p.minimal;
            m_nInputSize = p.input_size;
            m_nHiddenSize = p.hidden_size;
            m_nBackboneUnits = p.backbone_units;
            m_nBackboneLayers = p.backbone_layers;
            m_fBackboneDropout = p.backbone_dropout_ratio;
            m_backboneActivation = p.backbone_activation;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            CfcUnitParameter p = new CfcUnitParameter();
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

            rgChildren.Add("no_gate", no_gate.ToString());
            rgChildren.Add("minimal", minimal.ToString());
            rgChildren.Add("input_size", input_size.ToString());
            rgChildren.Add("hidden_size", hidden_size.ToString());
            rgChildren.Add("backbone_units", backbone_units.ToString());
            rgChildren.Add("backbone_layers", backbone_layers.ToString());
            rgChildren.Add("backbone_dropout_ratio", backbone_dropout_ratio.ToString());
            rgChildren.Add("backbone_activation", backbone_activation.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static CfcUnitParameter FromProto(RawProto rp)
        {
            string strVal;
            CfcUnitParameter p = new CfcUnitParameter();

            if ((strVal = rp.FindValue("no_gate")) != null)
                p.no_gate = bool.Parse(strVal);

            if ((strVal = rp.FindValue("minimal")) != null)
                p.minimal = bool.Parse(strVal);

            if ((strVal = rp.FindValue("input_size")) != null)
                p.input_size = int.Parse(strVal);

            if ((strVal = rp.FindValue("hidden_size")) != null)
                p.hidden_size = int.Parse(strVal);

            if ((strVal = rp.FindValue("backbone_units")) != null)
                p.backbone_units = int.Parse(strVal);

            if ((strVal = rp.FindValue("backbone_layers")) != null)
                p.backbone_layers = int.Parse(strVal);

            if ((strVal = rp.FindValue("backbone_dropout_ratio")) != null)
                p.backbone_dropout_ratio = BaseParameter.ParseFloat(strVal);

            if ((strVal = rp.FindValue("backbone_activation")) != null)
                p.backbone_activation = (ACTIVATION)Enum.Parse(typeof(ACTIVATION), strVal, true);

            return p;
        }
    }
}
