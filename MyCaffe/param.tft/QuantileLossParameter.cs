using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.tft
{
    /// <summary>
    /// Specifies the parameters for the QuantileLossLayer used in TFT models
    /// </summary>
    /// <remarks>
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch/loss.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/loss.py) by Playtika Research, 2021.
    /// </remarks>
    public class QuantileLossParameter : LayerParameterBase
    {
        List<float> m_rgDesiredQuantiles = new List<float>();

        /** @copydoc LayerParameterBase */
        public QuantileLossParameter()
        {
        }

        /// <summary>
        /// Specifies the desired quantiles.
        /// </summary>
        [Description("Specifies the desired quantiles.")]
        public List<float> desired_quantiles
        {
            get { return m_rgDesiredQuantiles; }
            set { m_rgDesiredQuantiles = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            QuantileLossParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            QuantileLossParameter p = (QuantileLossParameter)src;
            m_rgDesiredQuantiles = Utility.Clone<float>(p.desired_quantiles);
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            QuantileLossParameter p = new QuantileLossParameter();
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

            rgChildren.Add<float>("desired_quantile", desired_quantiles);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static QuantileLossParameter FromProto(RawProto rp)
        {
            QuantileLossParameter p = new QuantileLossParameter();

            p.desired_quantiles = rp.FindArray<float>("desired_quantile");

            return p;
        }
    }
}
