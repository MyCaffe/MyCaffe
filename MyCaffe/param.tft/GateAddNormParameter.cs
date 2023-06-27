using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using static MyCaffe.param.tft.GluParameter;

namespace MyCaffe.param.tft
{
    /// <summary>
    /// Specifies the parameters for the GetAddNormLayer (Gate Add Norm).  
    /// </summary>
    /// <remarks>
    /// The composite operation includes:
    /// a. Dropout
    /// b. Gating using GLU (Gated Linear Unit)
    /// c. A residual connection to 'earlier' signal from the forward pass of the parent model.
    /// d. Layer Normalization.
    /// 
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L405) by Playtika Research, 2021.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class GateAddNormParameter : LayerParameterBase
    {
        int m_nResidualChannelOffset = 0;

        /** @copydoc LayerParameterBase */
        public GateAddNormParameter()
        {
        }

        /// <summary>
        /// Specifies the residual channel offset used to copy only the latter portions of the residual (default = 0 which uses all of the residual).
        /// </summary>
        [Description("Specifies the residual channel offset used to copy only the latter portions of the residual (default = 0 which uses all of the residual).")]
        public int residual_channel_offset
        {
            get { return m_nResidualChannelOffset; }
            set { m_nResidualChannelOffset = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            GateAddNormParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            GateAddNormParameter p = (GateAddNormParameter)src;

            m_nResidualChannelOffset = p.residual_channel_offset;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            GateAddNormParameter p = new GateAddNormParameter();
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

            rgChildren.Add("residual_channel_offset", residual_channel_offset.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static GateAddNormParameter FromProto(RawProto rp)
        {
            string strVal;
            GateAddNormParameter p = new GateAddNormParameter();

            if ((strVal = rp.FindValue("residual_channel_offset")) != null)
                p.residual_channel_offset = int.Parse(strVal);

            return p;
        }
    }
}
