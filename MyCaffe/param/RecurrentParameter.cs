﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters used by the RecurrentLayer.
    /// </summary>
    public class RecurrentParameter : EngineParameter
    {
        uint m_nNumOutput = 0;
        FillerParameter m_weight_filler = new FillerParameter("xavier");
        FillerParameter m_bias_filler = new FillerParameter("constant", 0.1);
        bool m_bDebugInfo = false;
        bool m_bExposeHiddenInput = false;
        bool m_bExposeHiddenOutput = false;
        uint m_nNumLayers = 1; // cuDnn only
        double m_dfDropoutRatio = 0.0; // cuDnn only
        long m_lDropoutSeed = 0; // cuDnn only
        bool m_bCudnnEnableTensorCores = false;


        /** @copydoc LayerParameterBase */
        public RecurrentParameter()
        {            
        }

        /// <summary>
        /// Returns the reason that Caffe version was used instead of [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns></returns>
        public string useCaffeReason()
        {
            if (engine == Engine.CAFFE || engine == Engine.DEFAULT)
                return "The engine setting is set on CAFFE or DEFAULT.";

            return "";
        }

        /// <summary>
        /// Queries whether or not to use [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <remarks>
        /// CAFFE is the DEFAULT engine.
        /// </remarks>
        /// <returns>Returns <i>true</i> when cuDnn is to be used, <i>false</i> otherwise.</returns>
        public bool useCudnn()
        {
            if (engine == Engine.CAFFE || engine == Engine.DEFAULT)
                return false;

            return true;
        }


        /// <summary>
        /// The dimension of the output (and usually hidden state) representation --
        /// must be explicitly set to non-zero.
        /// </summary>
        [Description("Specifies the dimension of the output (and usually hidden state) representation -- must be explicitly set to non-zero.")]
        public uint num_output
        {
            get { return m_nNumOutput; }
            set { m_nNumOutput = value; }
        }

        /// <summary>
        /// The filler for the weights.
        /// </summary>
        [Description("Specifies the filler for the weights.")]
        public FillerParameter weight_filler
        {
            get { return m_weight_filler; }
            set { m_weight_filler = value; }
        }

        /// <summary>
        /// The filler for the bias.
        /// </summary>
        [Description("Specifies the filler for the bias.")]
        public FillerParameter bias_filler
        {
            get { return m_bias_filler; }
            set { m_bias_filler = value; }
        }

        /// <summary>
        /// Whether to enable displaying debug info in the unrolled recurrent net.
        /// </summary>
        [Description("Specifies whether to enable displaying debug info in the unrolled recurrent net.")]
        public bool debug_info
        {
            get { return m_bDebugInfo; }
            set { m_bDebugInfo = value; }
        }

        /// <summary>
        /// Whether to add as additional inputs (bottoms) the initial hidden state
        /// blobss.  The number of additional bottom/top blobs required depends on the
        /// recurrent architecture -- e.g., 1 for RNN's, 2 for LSTM's.
        /// </summary>
        [Description("Specifies whether to add as additional inputs (bottoms) the initial hidden state blobs.  The number of additional bottom/top blobs required depends on the recurrent architecture -- e.g., 1 for RNN's, 2 for LSTM's.")]
        public bool expose_hidden_input
        {
            get { return m_bExposeHiddenInput; }
            set { m_bExposeHiddenInput = value; }
        }

        /// <summary>
        /// Whether to add as additional outputs (tops) the final timestep hidden state
        /// blobs.  The number of additional bottom/top blobs required depends on the
        /// recurrent architecture -- e.g., 1 for RNN's, 2 for LSTM's.
        /// </summary>
        [Description("Specifies whether to add as additional outputs (tops) the final timestep hidden state blobs.  The number of additional bottom/top blobs required depends on the recurrent architecture -- e.g., 1 for RNN's, 2 for LSTM's.")]
        public bool expose_hidden_output
        {
            get { return m_bExposeHiddenOutput; }
            set { m_bExposeHiddenOutput = value; }
        }

        /// <summary>
        /// The number of LSTM layers to implement.
        /// </summary>
        /// <remarks>This parameter only applies to cuDnn.</remarks>
        [Description("Specifies the number of LSTM layers to implement (cuDnn only).")]
        public uint num_layers
        {
            get { return m_nNumLayers; }
            set { m_nNumLayers = value; }
        }

        /// <summary>
        /// Specifies the dropout ratio. (e.g. the probability that values will be dropped out and set to zero.  A value of 0.25 = 25% chance that a value is set to 0, and dropped out.)
        /// </summary>
        /// <remarks>The drop-out layer is only used with cuDnn when more than one layer are used.</remarks>
        [Description("Specifies the dropout ratio (cuDnn only). (e.g. the probability that values will be dropped out and set to zero.  A value of 0.25 = 25% chance that a value is set to 0, and dropped out.)")]
        public double dropout_ratio
        {
            get { return m_dfDropoutRatio; }
            set { m_dfDropoutRatio = value; }
        }

        /// <summary>
        /// Specifies the seed used by cuDnn for random number generation.
        /// </summary>
        /// <remarks>The drop-out layer is only used with cuDnn when more than one layer are used.</remarks>
        [Description("Specifies the random number generator seed used with the cuDnn dropout - the default value of '0' uses a random seed (cuDnn only).")]
        public long dropout_seed
        {
            get { return m_lDropoutSeed; }
            set { m_lDropoutSeed = value; }
        }

        /// <summary>
        /// Specifies to enable the CUDA tensor cores when performing the rnn operations which is faster but not supported by all GPU's.
        /// </summary>
        /// <remarks>
        /// When run on GPU's that do not support Tensor cores, the default math (non-tensor core) is used.
        /// </remarks>
        [Description("Specifies to enable CUDA tensor cores when performing the rnn operations which is faster but not supported by all GPU's.  When not supported, the default math is used.")]
        public bool cudnn_enable_tensor_cores
        {
            get { return m_bCudnnEnableTensorCores; }
            set { m_bCudnnEnableTensorCores = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            RecurrentParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            RecurrentParameter p = new RecurrentParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is RecurrentParameter)
            {
                RecurrentParameter p = (RecurrentParameter)src;
                m_nNumOutput = p.num_output;
                m_weight_filler = p.weight_filler.Clone();
                m_bias_filler = p.bias_filler.Clone();
                m_bDebugInfo = p.debug_info;
                m_bExposeHiddenInput = p.expose_hidden_input;
                m_bExposeHiddenOutput = p.expose_hidden_output;
                m_dfDropoutRatio = p.dropout_ratio;
                m_lDropoutSeed = p.dropout_seed;
                m_nNumLayers = p.num_layers;
                m_bCudnnEnableTensorCores = p.m_bCudnnEnableTensorCores;
            }
        }

        /// <summary>
        /// Convert the parameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name to associate with the RawProto.</param>
        /// <returns>The new RawProto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("engine");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);
            rgChildren.Add("num_output", num_output.ToString());

            if (weight_filler != null)
                rgChildren.Add(weight_filler.ToProto("weight_filler"));

            if (bias_filler != null)
                rgChildren.Add(bias_filler.ToProto("bias_filler"));

            rgChildren.Add("debug_info", debug_info.ToString());
            rgChildren.Add("expose_hidden", (expose_hidden_input && expose_hidden_output).ToString());
            rgChildren.Add("expose_hidden_input", expose_hidden_input.ToString());
            rgChildren.Add("expose_hidden_output", expose_hidden_output.ToString());

            if (engine != Engine.CAFFE)
            {
                rgChildren.Add("dropout_ratio", dropout_ratio.ToString());
                rgChildren.Add("dropout_seed", dropout_seed.ToString());
                rgChildren.Add("num_layers", num_layers.ToString());
            }

            rgChildren.Add("cudnn_enable_tensor_cores", cudnn_enable_tensor_cores.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new RecurrentParameter FromProto(RawProto rp)
        {
            string strVal;
            RecurrentParameter p = new RecurrentParameter();

            ((EngineParameter)p).Copy(EngineParameter.FromProto(rp));

            if ((strVal = rp.FindValue("num_output")) != null)
                p.num_output = uint.Parse(strVal);

            RawProto rpWeightFiller = rp.FindChild("weight_filler");
            if (rpWeightFiller != null)
                p.weight_filler = FillerParameter.FromProto(rpWeightFiller);

            RawProto rpBiasFiller = rp.FindChild("bias_filler");
            if (rpBiasFiller != null)
                p.bias_filler = FillerParameter.FromProto(rpBiasFiller);

            if ((strVal = rp.FindValue("debug_info")) != null)
                p.debug_info = bool.Parse(strVal);

            if ((strVal = rp.FindValue("expose_hidden")) != null)
            {
                p.expose_hidden_input = bool.Parse(strVal);
                p.expose_hidden_output = bool.Parse(strVal);
            }

            if ((strVal = rp.FindValue("expose_hidden_input")) != null)
                p.expose_hidden_input = bool.Parse(strVal);

            if ((strVal = rp.FindValue("expose_hidden_output")) != null)
                p.expose_hidden_output = bool.Parse(strVal);

            if ((strVal = rp.FindValue("dropout_ratio")) != null)
                p.dropout_ratio = ParseDouble(strVal);

            if ((strVal = rp.FindValue("dropout_seed")) != null)
                p.dropout_seed = long.Parse(strVal);

            if ((strVal = rp.FindValue("num_layers")) != null)
                p.num_layers = uint.Parse(strVal);

            if ((strVal = rp.FindValue("cudnn_enable_tensor_cores")) != null)
                p.cudnn_enable_tensor_cores = bool.Parse(strVal);

            return p;
        }
    }
}
