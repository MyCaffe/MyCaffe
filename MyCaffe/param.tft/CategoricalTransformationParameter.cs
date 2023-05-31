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
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L367) by Playtika Research, 2021.
    /// </remarks>
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class CategoricalTransformationParameter : LayerParameterBase
    {
        uint m_nNumInput = 0;
        uint m_nStateSize = 0;
        List<int> m_rgCardinalities = new List<int>();

        /** @copydoc LayerParameterBase */
        public CategoricalTransformationParameter()
        {
        }

        /// <summary>
        /// The number of categorical inputs for the layer.
        /// </summary>
        [Description("The number of categorical inputs for the layer.")]
        public uint num_input
        {
            get { return m_nNumInput; }
            set { m_nNumInput = value; }
        }

        /// <summary>
        /// The state size that defines the output embedding dimension width.
        /// </summary>
        [Description("The state size that defines the output embedding dimension width.")]
        public uint state_size
        {
            get { return m_nStateSize; }
            set { m_nStateSize = value; }
        }

        /// <summary>
        /// The cardinalities specify the quantity of categories associated with each of the input variables.
        /// </summary>
        [Description("The cardinalities specify the quantity of categories associated with each of the input variables.")]
        public List<int> cardinalities
        {
            get { return m_rgCardinalities; }
            set { m_rgCardinalities = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            CategoricalTransformationParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            CategoricalTransformationParameter p = (CategoricalTransformationParameter)src;

            m_nNumInput = p.m_nNumInput;
            m_nStateSize = p.m_nStateSize;
            m_rgCardinalities = new List<int>(p.m_rgCardinalities);
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            CategoricalTransformationParameter p = new CategoricalTransformationParameter();
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
            rgChildren.Add<int>("cardinality", m_rgCardinalities);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static CategoricalTransformationParameter FromProto(RawProto rp)
        {
            string strVal;
            CategoricalTransformationParameter p = new CategoricalTransformationParameter();

            if ((strVal = rp.FindValue("num_input")) != null)
                p.num_input = uint.Parse(strVal);

            if ((strVal = rp.FindValue("state_size")) != null)
                p.state_size = uint.Parse(strVal);

            p.cardinalities = rp.FindArray<int>("cardinality");

            return p;
        }
    }
}
