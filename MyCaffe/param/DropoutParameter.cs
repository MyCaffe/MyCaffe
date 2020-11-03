using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters of the DropoutLayer.
    /// </summary>
    /// <remarks>
    /// @see [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) by Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhavsky, and Ruslan R. Salakhutdinov, 2012.
    /// @see [Information Dropout: Learning Optimal Representations Through Noisy Computation](https://arxiv.org/abs/1611.01353) by Alessandro Achille, and Stevano Soatto, 2016.
    /// </remarks>
    public class DropoutParameter : EngineParameter
    {
        double m_dfDropoutRatio = 0.5;
        long m_lSeed = 0;
        bool m_bActive = true;

        /** @copydoc EngineParameter */
        public DropoutParameter()
            : base()
        {
        }

        /// <summary>
        /// Returns the reason that Caffe version was used instead of [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns></returns>
        public string useCaffeReason()
        {
            if (engine == Engine.CAFFE)
                return "The engine setting is set on CAFFE.";

            if (engine == Engine.DEFAULT)
                return "The engine setting is set to DEFAULT and, unlike other layers, the DropoutLayer defaults to use CAFFE.";

            return "";
        }

        /// <summary>
        /// Queries whether or not to use [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns>Returns <i>true</i> when cuDnn is to be used, <i>false</i> otherwise.</returns>
        public bool useCudnn()
        {
            if (engine == EngineParameter.Engine.CUDNN)
                return true;

            return false;   // DEFAULT = CAFFE
        }

        /// <summary>
        /// Specifies the dropout ratio. (e.g. the probability that values will be dropped out and set to zero.  A value of 0.25 = 25% chance that a value is set to 0, and dropped out.)
        /// </summary>
        [Description("Specifies the dropout ratio. (e.g. the probability that values will be dropped out and set to zero.  A value of 0.25 = 25% chance that a value is set to 0, and dropped out.)")]
        public double dropout_ratio
        {
            get { return m_dfDropoutRatio; }
            set { m_dfDropoutRatio = value; }
        }

        /// <summary>
        /// Specifies the seed used by cuDnn for random number generation.
        /// </summary>
        [Description("Specifies the random number generator seed used with the cuDnn dropout - the default value of '0' uses a random seed.")]
        public long seed
        {
            get { return m_lSeed; }
            set { m_lSeed = value; }
        }

        /// <summary>
        /// Specifies whether or not the dropout is active or not.  When inactive and training, the dropout acts the same as it does during testing and is ignored.
        /// </summary>
        [Description("Specifies whether or not the dropout is active or not.  When inactive and training, the dropout acts the same as it does during testing and is ignored.")]
        public bool active
        {
            get { return m_bActive; }
            set { m_bActive = value; }
        }

        /** @copydoc EngineParameter::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            DropoutParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc EngineParameter::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is DropoutParameter)
            {
                DropoutParameter p = (DropoutParameter)src;
                m_dfDropoutRatio = p.m_dfDropoutRatio;
                m_lSeed = p.m_lSeed;
                m_bActive = p.m_bActive;
            }
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            DropoutParameter p = new DropoutParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc EngineParameter::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("engine");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);

            if (dropout_ratio != 0.5)
                rgChildren.Add("dropout_ratio", dropout_ratio.ToString());

            if (seed != 0)
                rgChildren.Add("seed", seed.ToString());

            if (!active)
                rgChildren.Add("active", active.ToString());
           
            return new RawProto(strName, "", rgChildren);
        }

        /** @copydoc EngineParameter::FromProto */
        public static new DropoutParameter FromProto(RawProto rp)
        {
            string strVal;
            DropoutParameter p = new DropoutParameter();

            ((EngineParameter)p).Copy(EngineParameter.FromProto(rp));

            if ((strVal = rp.FindValue("dropout_ratio")) != null)
                p.dropout_ratio = ParseDouble(strVal);

            if ((strVal = rp.FindValue("seed")) != null)
                p.seed = long.Parse(strVal);

            if ((strVal = rp.FindValue("active")) != null)
                p.active = bool.Parse(strVal);

            return p;
        }
    }
}
