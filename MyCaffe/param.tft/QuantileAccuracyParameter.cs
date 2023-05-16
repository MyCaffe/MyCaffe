﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.tft
{
    /// <summary>
    /// Specifies the parameters for the QuantileAccuracyLayer used in TFT models
    /// </summary>
    /// <remarks>
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// </remarks>
    public class QuantileAccuracyParameter : LayerParameterBase
    {
        List<float> m_rgAccuracyRanges = new List<float>();

        /** @copydoc LayerParameterBase */
        public QuantileAccuracyParameter()
        {
        }

        /// <summary>
        /// Specifies the quantile ranges from center to check for predicted values against target values.  The number of target values falling within the center +/- each quantile accuracy range defines the accuracy.
        /// </summary>
        [Description("Specifies the quantile ranges from center to check for predicted values against target values.  The number of target values falling within the center +/- each quantile accuracy range defines the accuracy.")]
        public List<float> accuracy_ranges
        {
            get { return m_rgAccuracyRanges; }
            set { m_rgAccuracyRanges = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            QuantileAccuracyParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            QuantileAccuracyParameter p = (QuantileAccuracyParameter)src;
            m_rgAccuracyRanges = Utility.Clone<float>(p.accuracy_ranges);
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            QuantileAccuracyParameter p = new QuantileAccuracyParameter();
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

            rgChildren.Add<float>("accuracy_range", accuracy_ranges);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static QuantileAccuracyParameter FromProto(RawProto rp)
        {
            QuantileAccuracyParameter p = new QuantileAccuracyParameter();

            p.accuracy_ranges = rp.FindArray<float>("accuracy_range");

            return p;
        }
    }
}