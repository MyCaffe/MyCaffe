using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.tft
{
    /// <summary>
    /// Specifies the parameters for the NumericInputTransformationLayer.
    /// </summary>
    /// <remarks>
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L333) by Playtika Research, 2021.
    /// </remarks>
    public class NumericTransformationParameter : LayerParameterBase
    {
        uint m_nNumInput = 0;
        uint m_nStateSize = 0;

        /** @copydoc LayerParameterBase */
        public NumericTransformationParameter()
        {
        }

        /// <summary>
        /// The number of inputs for the layer.
        /// </summary>
        [Description("The number of inputs for the layer.")]
        public uint num_input
        {
            get { return m_nNumInput; }
            set { m_nNumInput = value; }
        }

        /// <summary>
        /// The state size that defines the output vector width.
        /// </summary>
        [Description("The state size that defines the output vector width.")]
        public uint state_size
        {
            get { return m_nStateSize; }
            set { m_nStateSize = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            NumericTransformationParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            NumericTransformationParameter p = (NumericTransformationParameter)src;

            m_nNumInput = p.m_nNumInput;
            m_nStateSize = p.m_nStateSize;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            NumericTransformationParameter p = new NumericTransformationParameter();
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

            rgChildren.Add("num_input", num_input.ToString());
            rgChildren.Add("state_size", state_size.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static NumericTransformationParameter FromProto(RawProto rp)
        {
            string strVal;
            NumericTransformationParameter p = new NumericTransformationParameter();

            if ((strVal = rp.FindValue("num_input")) != null)
                p.num_input = uint.Parse(strVal);

            if ((strVal = rp.FindValue("state_size")) != null)
                p.state_size = uint.Parse(strVal);

            return p;
        }
    }
}
