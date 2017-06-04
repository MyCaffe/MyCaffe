using System;
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
    public class RecurrentParameter : LayerParameterBase
    {
        uint m_nNumOutput = 0;
        FillerParameter m_weight_filler = new FillerParameter("gaussian");
        FillerParameter m_bias_filler = new FillerParameter("constant", 1.0);
        bool m_bDebugInfo = false;
        bool m_bExposeHidden = false;

        /** @copydoc LayerParameterBase */
        public RecurrentParameter()
        {            
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
        /// blobs, and add as additional outputs (tops) the final timestep hidden state
        /// blobs.  The number of additional bottom/top blobs required depends on the
        /// recurrent architecture -- e.g., 1 for RNN's, 2 for LSTM's.
        /// </summary>
        [Description("Specifies whether to add as additional inputs (bottoms) the initial hidden state blobs, and add as additional outputs (tops) the final timestep hidden state blobs.  The number of additional bottom/top blobs required depends on teh recurrent architecture -- e.g., 1 for RNN's, 2 for LSTM's.")]
        public bool expose_hidden
        {
            get { return m_bExposeHidden; }
            set { m_bExposeHidden = value; }
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
            RecurrentParameter p = (RecurrentParameter)src;
            m_nNumOutput = p.num_output;
            m_weight_filler = p.weight_filler.Clone();
            m_bias_filler = p.bias_filler.Clone();
            m_bDebugInfo = p.debug_info;
            m_bExposeHidden = p.expose_hidden;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("num_output", num_output.ToString());

            if (weight_filler != null)
                rgChildren.Add(weight_filler.ToProto("weight_filler"));

            if (bias_filler != null)
                rgChildren.Add(bias_filler.ToProto("bias_filler"));

            rgChildren.Add("debug_info", debug_info.ToString());
            rgChildren.Add("expose_hidden", expose_hidden.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static RecurrentParameter FromProto(RawProto rp)
        {
            string strVal;
            RecurrentParameter p = new RecurrentParameter();

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
                p.expose_hidden = bool.Parse(strVal);

            return p;
        }
    }
}
