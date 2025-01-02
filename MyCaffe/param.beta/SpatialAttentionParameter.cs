using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the SpatialAttentionLayer.
    /// </summary>
    /// <remarks>
    /// @see [RFAConv: Innovating Spatial Attention and Standard Convolutional Operation](https://arxiv.org/abs/2304.03198) by Xin Zhang, Chen Liu, Degang Yang, Tingting Song, Yichen Ye, Ke Li, Yingze Song, 2023, arXiv:2304.03198
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class SpatialAttentionParameter : LayerParameterBase
    {
        int m_nAxis = 1;
        uint m_nKernelSize = 3;
        ACTIVATION m_activation = ACTIVATION.RELU;

        /// <summary>
        /// Defines the activation types.
        /// </summary>
        public enum ACTIVATION
        {
            /// <summary>
            /// Specifies the ReLU activation.
            /// </summary>
            RELU
        }

        /** @copydoc LayerParameterBase */
        public SpatialAttentionParameter()
        {
        }

        /// <summary>
        /// The axis along which to perform the softmax -- may be negative to index
        /// from the end (e.g., -1 for the last axis).
        /// Any other axes will be evaluated as independent softmaxes.
        /// </summary>
        [Description("Specifies the axis along which to perform the softmax - may be negative to index from the end (e.g., -1 for the last axis).")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Specifies the dim of the kernel size (default = 1)
        /// </summary>
        public uint kernel_size
        {
            get { return m_nKernelSize; }
            set { m_nKernelSize = value; }
        }

        /// <summary>
        /// Specifies the activation function to use.
        /// </summary>
        public ACTIVATION activation
        {
            get { return m_activation; }
            set { m_activation = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            SpatialAttentionParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            SpatialAttentionParameter p = (SpatialAttentionParameter)src;

            m_nKernelSize = p.kernel_size;
            m_nAxis = p.m_nAxis;
            m_activation = p.m_activation;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            SpatialAttentionParameter p = new SpatialAttentionParameter();
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

            rgChildren.Add("axis", axis.ToString());
            rgChildren.Add("kernel_size", kernel_size.ToString());
            rgChildren.Add("activation", m_activation.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static SpatialAttentionParameter FromProto(RawProto rp)
        {
            string strVal;
            SpatialAttentionParameter p = new SpatialAttentionParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("kernel_size")) != null)
                p.kernel_size = uint.Parse(strVal);

            if ((strVal = rp.FindValue("activation")) != null)
            {
                if (strVal == "RELU")
                    p.activation = ACTIVATION.RELU;
                else
                    throw new Exception("Unknown activation '" + strVal + "'!");
            }

            return p;
        }
    }
}
