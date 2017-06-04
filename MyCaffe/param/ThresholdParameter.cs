using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Stores the parameters used by the ThresholdLayer
    /// </summary>
    /// <remarks>
    /// @see [Neural Networks with Input Specified Thresholds](http://cs231n.stanford.edu/reports/2016/pdfs/118_Report.pdf) by Fei Liu and Junyang Qian, 2016.
    /// </remarks>
    public class ThresholdParameter : LayerParameterBase
    {
        double m_dfThreshold = 0;

        /** @copydoc LayerParameterBase */
        public ThresholdParameter()
        {
        }

        /// <summary>
        /// Specifies the threshold value which must be strictly positive values.
        /// </summary>
        [Description("Specifies the threshold value which must be strictly positive values.")]
        public double threshold
        {
            get { return m_dfThreshold; }
            set { m_dfThreshold = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ThresholdParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            ThresholdParameter p = (ThresholdParameter)src;
            m_dfThreshold = p.m_dfThreshold;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ThresholdParameter p = new ThresholdParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("threshold", threshold.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ThresholdParameter FromProto(RawProto rp)
        {
            string strVal;
            ThresholdParameter p = new ThresholdParameter();

            if ((strVal = rp.FindValue("threshold")) != null)
                p.threshold = double.Parse(strVal);

            return p;
        }
    }
}
