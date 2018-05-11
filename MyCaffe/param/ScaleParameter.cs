using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the ScaleLayer.
    /// </summary>
    public class ScaleParameter : BiasParameter 
    {
        bool m_bBiasTerm = false;
        FillerParameter m_FillerBias = null;

        /** @copydoc LayerParameterBase */
        public ScaleParameter()
        {
        }

        /// <summary>
        /// Whether to also learn a bias (equivalent to a ScalarLayer + BiasLayer, but
        /// may be more efficient).
        /// </summary>
        [Description("Specifies whether to also learn a 'bias' (eqivalent to a ScalarLayer + BiasLayer, but may be more efficient).")]
        public bool bias_term
        {
            get { return m_bBiasTerm; }
            set { m_bBiasTerm = value; }
        }

        /// <summary>
        /// Filler used for bias filling.
        /// </summary>
        [Description("Specifies the filler used for bias filling.")]
        public FillerParameter bias_filler
        {
            get { return m_FillerBias; }
            set { m_FillerBias = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ScaleParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            ScaleParameter p = (ScaleParameter)src;

            base.Copy(src);

            m_bBiasTerm = p.m_bBiasTerm;

            if (p.m_FillerBias != null)
                m_FillerBias = p.m_FillerBias.Clone();
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ScaleParameter p = new ScaleParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (bias_term != false)
            {
                rgChildren.Add("bias_term", bias_term.ToString());

                if (bias_filler != null)
                    rgChildren.Add(bias_filler.ToProto("bias_filler"));
            }

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new ScaleParameter FromProto(RawProto rp)
        {
            string strVal;
            ScaleParameter p = new ScaleParameter();

            if ((strVal = rp.FindValue("bias_term")) != null)
                p.bias_term = bool.Parse(strVal);

            RawProto rpBiasFiller = rp.FindChild("bias_filler");
            if (rpBiasFiller != null)
                p.bias_filler = FillerParameter.FromProto(rpBiasFiller);

            return p;
        }
    }
}
