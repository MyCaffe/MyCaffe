using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the AccuracyMapeLayer.
    /// </summary>
    /// <remarks>
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class AccuracyMapeParameter : LayerParameterBase
    {
        MAPE_ALGORITHM m_alg = MAPE_ALGORITHM.MAPE;

        /// <summary>
        /// Defines the MAPE algorithm to use.
        /// </summary>
        public enum MAPE_ALGORITHM
        {
            /// <summary>
            /// Defines the Mean Absolute Percentage Error algorithm.
            /// </summary>
            MAPE,
            /// <summary>
            /// Defines the Symmetric Mean Absolute Percentage Error algorithm.
            /// </summary>
            SMAPE
        }

        /** @copydoc LayerParameterBase */
        public AccuracyMapeParameter()
        {
        }

        /// <summary>
        /// Specifies the algorithm to use: MAPE or SMAPE.
        /// </summary>
        [Description("Specifies algorithm used in the MAPE (or SMAPE) calculation.")]
        public MAPE_ALGORITHM algorithm
        {
            get { return m_alg; }
            set { m_alg = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            AccuracyMapeParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            AccuracyMapeParameter p = (AccuracyMapeParameter)src;

            m_alg = p.m_alg;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            AccuracyMapeParameter p = new AccuracyMapeParameter();
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

            rgChildren.Add("algorithm", algorithm.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static AccuracyMapeParameter FromProto(RawProto rp)
        {
            string strVal;
            AccuracyMapeParameter p = new AccuracyMapeParameter();

            if ((strVal = rp.FindValue("algorithm")) != null)
                p.algorithm = (MAPE_ALGORITHM)Enum.Parse(typeof(MAPE_ALGORITHM), strVal, true);

            return p;
        }
    }
}
