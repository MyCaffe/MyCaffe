using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.tft
{
    /// <summary>
    /// Specifies the parameters for the ReshapeTemporalLayer.  
    /// </summary>
    /// <remarks>
    /// When run using the BEFORE mode, this layer is used to reshape static inputs along time while stacking temporal and time distributed contexts along the batch.
    /// When run using the AFTER mode, this layer reshapes the outputs and weight back into their num samples x temporal steps shape.
    /// 
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L1198) by Playtika Research, 2021.
    /// </remarks>
    public class ReshapeTemporalParameter : LayerParameterBase
    {
        MODE m_mode = MODE.BEFORE;

        /// <summary>
        /// Defines the modulation type.
        /// </summary>
        public enum MODE
        {
            /// <summary>
            /// Specifies to reshape in preparation for a temporal operation.
            /// </summary>
            BEFORE,
            /// <summary>
            /// Specifies to reshape after a temporal operation.
            /// </summary>
            AFTER
        }

        /** @copydoc LayerParameterBase */
        public ReshapeTemporalParameter()
        {
        }

        /// <summary>
        /// Specifies the mode of operation.
        /// </summary>
        [Description("Specifies the mode of operation.")]
        public MODE mode
        {
            get { return m_mode; }
            set { m_mode = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ReshapeTemporalParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            ReshapeTemporalParameter p = (ReshapeTemporalParameter)src;

            m_mode = p.mode;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ReshapeTemporalParameter p = new ReshapeTemporalParameter();
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

            rgChildren.Add("mode", mode.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ReshapeTemporalParameter FromProto(RawProto rp)
        {
            string strVal;
            ReshapeTemporalParameter p = new ReshapeTemporalParameter();

            if ((strVal = rp.FindValue("input_dim")) != null)
            {
                if (strVal == MODE.BEFORE.ToString())
                    p.mode = MODE.BEFORE;
                else if (strVal == MODE.AFTER.ToString())
                    p.mode = MODE.AFTER;
                else
                    throw new Exception("Invalid mode '" + strVal + "'.");
            }

            return p;
        }
    }
}
