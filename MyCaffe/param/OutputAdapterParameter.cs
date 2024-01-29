using Google.Protobuf.WellKnownTypes;
using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MyCaffe.param.FillerParameter;

namespace MyCaffe.param
{
    /// <summary>
    /// The OutputAdapterParameter specifies the output adapter parameters for all output adapters.
    /// </summary>
    /// <remarks>
    /// When present and enabled, the output adapters are run right after the forward pass of each layer, and before the backward pass of each layer.
    /// </remarks>
    public class OutputAdapterParameter : BaseParameter, ICloneable
    {
        string m_strType;
        bool m_bEnable = false;
        double m_dfAlpha = 1.0;
        uint m_nRank = 4;
        double m_dfDropoutRatio = 0.0;

        /// <summary>
        /// Defines the output adapter type.
        /// </summary>
        public enum OutputAdapterType
        {
            /// <summary>
            /// Specifies the NONE type, no adapter is used.
            /// </summary>
            NONE,
            /// <summary>
            /// Specifies the LoRA (Low Rank Adaptation) type.
            /// </summary>
            /// <remarks>
            /// @see [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, 2023
            /// </remarks>
            LORA
        }

        /// <summary>
        /// Specifies the constuctor.
        /// </summary>
        /// <param name="strType">Specifies the output adapter type.</param>
        /// <param name="bEnable">Specifies to enable the parameter.</param>
        /// <param name="dfAlpha">Specifies the alpha value.</param>
        /// <param name="nRank">Specifies the rank value.</param>
        /// <param name="dfDropoutRatio">Specifies the dropout_ratio.</param>
        public OutputAdapterParameter(string strType = "lora", bool bEnable = false, double dfAlpha = 1.0, uint nRank = 4, double dfDropoutRatio = 0.0)
        {
            m_strType = strType;
            m_bEnable = bEnable;
            m_dfAlpha = dfAlpha;
            m_nRank = nRank;
            m_dfDropoutRatio = dfDropoutRatio;
        }

        /// <summary>
        /// Get/set the output adapter type.    
        /// </summary>
        [Description("Specifies the output adapter type (default = 'lora').")]
        [Browsable(false)]
        public string type
        {
            get { return m_strType; }
            set { m_strType = value; }
        }

        /// <summary>
        /// Get/set whether or not the output adapter is enabled.
        /// </summary>
        [Description("Get/set the enabled state of the parameter (default = false).")]
        public bool enabled
        {
            get { return m_bEnable; }
            set { m_bEnable = value; }
        }

        /// <summary>
        /// Get/set the alpha value for the parameter.
        /// </summary>
        [Description("Get/set the alpha value for the parameter (default = 1.0).")]
        public double alpha
        {
            get { return m_dfAlpha; }
            set { m_dfAlpha = value; }
        }

        /// <summary>
        /// Get/set the rank value for the parameter.
        /// </summary>
        [Description("Get/set the rank value for the parameter (default = 4).")]
        public uint rank
        {
            get { return m_nRank; }
            set { m_nRank = value; }
        }

        /// <summary>
        /// Get/set the dropout ratio for the parameter (default = 0, which disables the dropout).
        /// </summary>
        [Description("Get/set the dropout ratio for the parameter (default = 0, which disables the dropout).")]
        public double dropout_ratio
        {
            get { return m_dfDropoutRatio; }
            set { m_dfDropoutRatio = value; }
        }

#pragma warning disable 1591

        [DisplayName("type")]
        [Description("Specifies the output adapter type.")]
        public OutputAdapterType OutputAdapterTypeMethod /** @private */
        {
            get
            {
                switch (m_strType)
                {
                    case "lora":
                        return OutputAdapterType.LORA;

                    case "none":
                        return OutputAdapterType.NONE;

                    default:
                        throw new Exception("Unknown output adapter type '" + m_strType + "'");
                }
            }
        }

#pragma warning restore 1591

        /// <summary>
        /// Queries the output adapter text name corresponding to the OutputAdapterType.
        /// </summary>
        /// <param name="type">Specifies the OutputAdapterType.</param>
        /// <returns>The string associated with the OutputAdapterType is returned.</returns>
        /// <exception cref="NotImplementedException"></exception>
        public static string GetOutputAdapterName(OutputAdapterType type)
        {
            switch (type)
            {
                case OutputAdapterType.LORA:
                    return "lora";

                case OutputAdapterType.NONE:
                    return "none";

                default:
                    throw new NotImplementedException("The output adapter type '" + type.ToString() + "' is not implemented!");
            }
        }

        /// <summary>
        /// Create a clone of this output adapter parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public OutputAdapterParameter Clone()
        {
            OutputAdapterParameter type = new OutputAdapterParameter();
            type.m_bEnable = m_bEnable;
            type.m_strType = m_strType;
            return type;
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
            rgChildren.Add(new RawProto("enabled", enabled.ToString()));
            rgChildren.Add(new RawProto("alpha", alpha.ToString()));
            rgChildren.Add(new RawProto("rank", rank.ToString()));
            rgChildren.Add(new RawProto("dropout_ratio", dropout_ratio.ToString()));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static OutputAdapterParameter FromProto(RawProto rp)
        {
            string strVal;

            if ((strVal = rp.FindValue("type")) == null)
                throw new Exception("Could not find 'type'");

            OutputAdapterParameter p = new OutputAdapterParameter(strVal);

            if ((strVal = rp.FindValue("enabled")) != null)
                p.enabled = bool.Parse(strVal);

            if ((strVal = rp.FindValue("alpha")) != null)
                p.alpha = ParseDouble(strVal);

            if ((strVal = rp.FindValue("rank")) != null)
                p.rank = uint.Parse(strVal);

            if ((strVal = rp.FindValue("dropout_ratio")) != null)
                p.dropout_ratio = ParseDouble(strVal);

            return p;
        }

        object ICloneable.Clone()
        {
            return Clone();
        }
    }
}
