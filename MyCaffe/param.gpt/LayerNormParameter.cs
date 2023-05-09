using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.gpt
{
    /// <summary>
    /// Specifies the parameters for the LayerNormalizationLayer.
    /// </summary>
    /// <remarks>
    /// @see [GitHub:CyberZHG](https://github.com/CyberZHG/torch-layer-normalization/blob/master/torch_layer_normalization/layer_normalization.py) by Zhao HG (MIT Liceense).
    /// @see [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) PyTorch
    /// @see [Understanding and Improving Layer Normalization](https://arxiv.org/abs/1911.07013) by Xu et al., 2019, arXiv:1911.07013
    /// </remarks>
    public class LayerNormParameter : LayerParameterBase 
    {
        double m_dfEpsilon = 1e-10;
        bool m_bEnableCudaImplementation = false;
        bool m_bEnablePassThrough = false;

        /** @copydoc LayerParameterBase */
        public LayerNormParameter()
        {
        }

        /// <summary>
        /// Specifies the epsilon value used to avoid invalid values (default = 1e-10).
        /// </summary>
        [Description("Specifies the epsilon value used to avoid invalid values (default = 1e-10).")]
        public double epsilon
        {
            get { return m_dfEpsilon; }
            set { m_dfEpsilon = value; }
        }

        /// <summary>
        /// Specifies to pass-through the data on the forward and backward pass (e.g. skip the layer norm, used only for debugging. default = false).
        /// </summary>
        [Description("Specifies to pass-through the data on the forward and backward pass (e.g. skip the layer norm, used only for debugging. default = false).")]
        public bool enable_passthrough
        {
            get { return m_bEnablePassThrough; }
            set { m_bEnablePassThrough = value; }
        }

        /// <summary>
        /// Specifies to use the low-level full cuda implementation of LayerNorm (default = false).
        /// </summary>
        /// <remarks>
        /// The cuda implementation runs around 30% faster when using float base types.
        /// </remarks>
        [Description("Specifies to use the low-level full cuda implementation of LayerNorm (default = false).")]
        public bool enable_cuda_impl
        {
            get { return m_bEnableCudaImplementation; }
            set { m_bEnableCudaImplementation = value; }
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
            LayerNormParameter p = FromProto(proto);

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
            LayerNormParameter p = (LayerNormParameter)src;
            m_dfEpsilon = p.epsilon;
            m_bEnableCudaImplementation = p.enable_cuda_impl;
            m_bEnablePassThrough = p.enable_passthrough;
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public override LayerParameterBase Clone()
        {
            LayerNormParameter p = new LayerNormParameter();
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

            rgChildren.Add("epsilon", m_dfEpsilon.ToString());
            rgChildren.Add("enable_cuda_impl", m_bEnableCudaImplementation.ToString());

            if (m_bEnablePassThrough)
                rgChildren.Add("enable_passthrough", m_bEnablePassThrough.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static LayerNormParameter FromProto(RawProto rp)
        {
            string strVal;
            LayerNormParameter p = new LayerNormParameter();

            if ((strVal = rp.FindValue("epsilon")) != null)
                p.m_dfEpsilon = double.Parse(strVal);

            if ((strVal = rp.FindValue("enable_cuda_impl")) != null)
                p.m_bEnableCudaImplementation = bool.Parse(strVal);

            if ((strVal = rp.FindValue("enable_passthrough")) != null)
                p.m_bEnablePassThrough = bool.Parse(strVal);

            return p;
        }
    }
}
