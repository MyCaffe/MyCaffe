using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the ConcatLayer
    /// </summary>
    /// <remarks>
    /// @see [Deep Image Aesthetics Classification using Inception Modules and Fine-tuning Connected Layer](https://arxiv.org/abs/1610.02256) by Xin Jin, Jingying Chi, Siwei Peng, Yulu Tian, Chaochen Ye, and Xiaodong Li, 2016.
    /// @see [Multi-path Convolutional Neural Networks for Complex Image Classification](https://arxiv.org/abs/1506.04701) by Mingming Wang, 2015.
    /// @see [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) by Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna, 2015.
    /// @see [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) by Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alex Alemi, 2015.
    /// </remarks>
    public class ConcatParameter : LayerParameterBase 
    {
        int m_nAxis = 1;
        uint m_nConcatDim = 0;

        /** @copydoc LayerParameterBase */
        public ConcatParameter()
        {
        }

        /// <summary>
        /// The axis along which to concatenate -- may be negative to index from the
        /// end (e.g., -1 for the last axis).
        /// 
        /// Othe axes must have the same dimension for all the bottom blobs.
        /// By default, ConcatLayer concatentates blobs along the 'channels' axis 1.
        /// </summary>
        [Description("The axis along which to concatenate - may be negative to index from the end (e.g., -1 for the last axis).  Other axes must have the same dimension for all the bottom blobs.  By default, the ConcatLayer concatenates blobs along the 'channel' axis 1.")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// DEPRECIATED: alias for 'axis' -- does not support negative indexing.
        /// </summary>
        [Description("DEPRECIATED - use 'axis' instead.")]
        public uint concat_dim
        {
            get { return m_nConcatDim; }
            set { m_nConcatDim = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ConcatParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            ConcatParameter p = (ConcatParameter)src;
            m_nAxis = p.m_nAxis;
            m_nConcatDim = p.m_nConcatDim;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ConcatParameter p = new ConcatParameter();
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

            if (axis != 1)
                rgChildren.Add("axis", axis.ToString());

            if (concat_dim != 0)
                rgChildren.Add("concat_dim", concat_dim.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ConcatParameter FromProto(RawProto rp)
        {
            string strVal;
            ConcatParameter p = new ConcatParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("concat_dim")) != null)
                p.concat_dim = uint.Parse(strVal);

            return p;
        }
    }
}
