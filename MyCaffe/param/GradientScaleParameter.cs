using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the GradientScaleLayer.
    /// </summary>
    /// <remarks>
    /// Scaling is performed according to the schedule:
    /// @f$ y = \frac{2 \cdot height} {1 + \exp(-\alpha \cot progress)} - upper\_bound @f$,
    /// where @f$ height = upper\_bound - lower\_bound @f$,
    /// @f$ lower\_bound @f$ is the smallest scaling factor,
    /// @f$ upper\_bound @f$ is the largest scaling factor,
    /// @f$ \alpha @f$ controls how fast the transition occurs between the scaling factors,
    /// @f$ progress = \min(iter / max\_iter, 1) @f$ corresponds to the current transition
    /// state (the @f$ iter @f$ is the current iteration of the solver).
    /// 
    /// The GradientScaleLayer can be used to implement gradient reversals.
    /// 
    /// @see [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818) by Ganin et al., 2015, v4 in 2016.
    /// @see [Github/ddtm/caffe](https://github.com/ddtm/caffe) for original source.
    /// </remarks> 
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class GradientScaleParameter : LayerParameterBase
    {
        double m_dfLowerBound = 0.0;
        double m_dfUpperBound = 1.0;
        double m_dfAlpha = 10.0;
        double m_dfMaxIter = 1;

        /** @copydoc LayerParameterBase */
        public GradientScaleParameter()
        {
        }

        /// <summary>
        /// Specifies the lower bound of the height used for scaling.
        /// </summary>
        [Description("Specifies the lower bound of the height used for scaling.")]
        public double lower_bound
        {
            get { return m_dfLowerBound; }
            set { m_dfLowerBound = value; }
        }

        /// <summary>
        /// Specifies the upper bound of the height used for scaling.
        /// </summary>
        [Description("Specifies the upper bound of the height used for scaling.")]
        public double upper_bound
        {
            get { return m_dfUpperBound; }
            set { m_dfUpperBound = value; }
        }

        /// <summary>
        /// Specifies the alpha value applied to the current iter/max_iter, used when scaling.
        /// </summary>
        [Description("Specifies the alpha value applied to the current iter/max_iter, used when scaling.")]
        public double alpha
        {
            get { return m_dfAlpha; }
            set { m_dfAlpha = value; }
        }

        /// <summary>
        /// Specifies the maximum iteration used when scaling.
        /// </summary>
        [Description("Specifies the maximum iteration used when scaling.")]
        public double max_iter
        {
            get { return m_dfMaxIter; }
            set { m_dfMaxIter = value; }
        }

        /// <summary>
        /// Load the parameter from a binary reader.
        /// </summary>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bNewInstance">When <i>true</i> a new instance is created (the default), otherwise the existing instance is loaded from the binary reader.</param>
        /// <returns>Returns an instance of the parameter.</returns>
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            GradientScaleParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /// <summary>
        /// Copy on parameter to another.
        /// </summary>
        /// <param name="src">Specifies the parameter to copy.</param>
        public override void Copy(LayerParameterBase src)
        {
            GradientScaleParameter p = (GradientScaleParameter)src;
            m_dfLowerBound = p.m_dfLowerBound;
            m_dfUpperBound = p.m_dfUpperBound;
            m_dfAlpha = p.m_dfAlpha;
            m_dfMaxIter = p.m_dfMaxIter;
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public override LayerParameterBase Clone()
        {
            GradientScaleParameter p = new GradientScaleParameter();
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

            rgChildren.Add("lower_bound", lower_bound.ToString());
            rgChildren.Add("upper_bound", upper_bound.ToString());
            rgChildren.Add("alpha", alpha.ToString());
            rgChildren.Add("max_iter", max_iter.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static GradientScaleParameter FromProto(RawProto rp)
        {
            string strVal;
            GradientScaleParameter p = new GradientScaleParameter();

            if ((strVal = rp.FindValue("lower_bound")) != null)
                p.lower_bound = parseDouble(strVal);

            if ((strVal = rp.FindValue("upper_bound")) != null)
                p.upper_bound = parseDouble(strVal);

            if ((strVal = rp.FindValue("alpha")) != null)
                p.alpha = parseDouble(strVal);

            if ((strVal = rp.FindValue("max_iter")) != null)
                p.max_iter = parseDouble(strVal);

            return p;
        }
    }
}
