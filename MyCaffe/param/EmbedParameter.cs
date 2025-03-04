﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters used by the EmbedLayer.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class EmbedParameter : LayerParameterBase 
    {
        uint m_nNumOutput;
        uint m_nInputDim;
        bool m_bBiasTerm = true;
        FillerParameter m_fillerParam_weights = new FillerParameter("xavier");
        FillerParameter m_fillerParam_bias = new FillerParameter("constant", 0.1);
        COMPUTE_TYPE m_backwardComputeType = COMPUTE_TYPE.FAST;

        /// <summary>
        /// Specifies the type of computation to use.
        /// </summary>
        public enum COMPUTE_TYPE
        {
            /// <summary>
            /// Specifies to use the accurate computation which is slower.
            /// </summary>
            ACCUMULATE = 0,
            /// <summary>
            /// Specifies to use the fast computation which is faster but less accurate.
            /// </summary>
            FAST = 1
        }

        /** @copydoc LayerParameterBase */
        public EmbedParameter()
        {
        }

        /// <summary>
        /// Specifies the type of computation to use.
        /// </summary>
        [Description("Specifies the type of computation to use (default=FAST).  ACCUMULATE = 0, FAST = 1")]
        public COMPUTE_TYPE backward_compute_type
        {
            get { return m_backwardComputeType; }
            set { m_backwardComputeType = value; }
        }

        /// <summary>
        ///  Specifies the number of outputs for the layer.
        /// </summary>
        [Description("Specifies the number of outputs for the layer.")]
        public uint num_output
        {
            get { return m_nNumOutput; }
            set { m_nNumOutput = value; }
        }

        /// <summary>
        /// Specifies the input given as integers to be interpreted as one-hot
        /// vector indices with dimension num_input.  Hence num_input should be
        /// 1 greater than the maximum possible input value.
        /// </summary>
        [Description("Specifies the input given as integers to be interpreted as one-hot vector indices with dimension 'num_init'. Hence 'num_input' should be 1 greater than the maximum possible input value.")]
        public uint input_dim
        {
            get { return m_nInputDim; }
            set { m_nInputDim = value; }
        }

        /// <summary>
        /// Specifies whether to use a bias term or not.
        /// </summary>
        [Description("Specifies wheter ot use a bias term or not.")]
        public bool bias_term
        {
            get { return m_bBiasTerm; }
            set { m_bBiasTerm = value; }
        }

        /// <summary>
        /// Specifies the filler for the weights.
        /// </summary>
        [Description("Specifies the filler for the weights.")]
        public FillerParameter weight_filler
        {
            get { return m_fillerParam_weights; }
            set { m_fillerParam_weights = value; }
        }

        /// <summary>
        /// Specifies the filler for the bias.
        /// </summary>
        [Description("Specifies the filler for the bias.")]
        public FillerParameter bias_filler
        {
            get { return m_fillerParam_bias; }
            set { m_fillerParam_bias = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            EmbedParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            EmbedParameter p = (EmbedParameter)src;

            m_nNumOutput = p.m_nNumOutput;
            m_nInputDim = p.m_nInputDim;
            m_bBiasTerm = p.m_bBiasTerm;
            m_backwardComputeType = p.m_backwardComputeType;

            if (p.m_fillerParam_bias != null)
                m_fillerParam_bias = p.m_fillerParam_bias.Clone();

            if (p.m_fillerParam_weights != null)
                m_fillerParam_weights = p.m_fillerParam_weights.Clone();
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            EmbedParameter p = new EmbedParameter();
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

            rgChildren.Add("num_output", num_output.ToString());
            rgChildren.Add("input_dim", input_dim.ToString());

            if (backward_compute_type != COMPUTE_TYPE.FAST)
                rgChildren.Add("backward_compute_type", ((int)backward_compute_type).ToString());

            if (bias_term != true)
                rgChildren.Add("bias_term", bias_term.ToString());

            if (weight_filler != null)
                rgChildren.Add(weight_filler.ToProto("weight_filler"));

            if (bias_term && bias_filler != null)
                rgChildren.Add(bias_filler.ToProto("bias_filler"));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static EmbedParameter FromProto(RawProto rp)
        {
            string strVal;
            EmbedParameter p = new EmbedParameter();

            if ((strVal = rp.FindValue("num_output")) != null)
                p.num_output = uint.Parse(strVal);

            if ((strVal = rp.FindValue("input_dim")) != null)
                p.input_dim = uint.Parse(strVal);

            if ((strVal = rp.FindValue("bias_term")) != null)
                p.bias_term = bool.Parse(strVal);

            if ((strVal = rp.FindValue("backward_compute_type")) != null)
            {
                if (strVal == COMPUTE_TYPE.ACCUMULATE.ToString())
                    p.backward_compute_type = COMPUTE_TYPE.ACCUMULATE;
                else
                    p.backward_compute_type = COMPUTE_TYPE.FAST;
            }

            RawProto rpWeightFiller = rp.FindChild("weight_filler");
            if (rpWeightFiller != null)
                p.weight_filler = FillerParameter.FromProto(rpWeightFiller);

            RawProto rpBiasFiller = rp.FindChild("bias_filler");
            if (rpBiasFiller != null)
                p.bias_filler = FillerParameter.FromProto(rpBiasFiller);

            return p;
        }
    }
}
