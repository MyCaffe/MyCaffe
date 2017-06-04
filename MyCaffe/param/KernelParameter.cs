using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;
using MyCaffe.common;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the basic kernel parameters (used by convolution and pooling)
    /// </summary>
    public class KernelParameter : EngineParameter 
    {
        List<uint> m_rgPad = new List<uint>();
        List<uint> m_rgStride = new List<uint>();
        List<uint> m_rgKernelSize = new List<uint>();
        List<uint> m_rgDilation = new List<uint>();
        uint? m_nPadH = null;
        uint? m_nPadW = null;
        uint? m_nStrideH = null;
        uint? m_nStrideW = null;
        uint? m_nKernelH = null;
        uint? m_nKernelW = null;


        /** @copydoc EngineParameter */
        public KernelParameter()
        {
        }

        /// <summary>
        /// Pad is given as a single value for equal dimensions in all 
        /// spatial dimensions, or once per spatial dimension.
        /// </summary>
        [Description("Specifies pad given as a single value for equal dimensions in all spatial dimensions, or once per spatial dimension.")]
        public List<uint> pad
        {
            get { return m_rgPad; }
            set { m_rgPad = value; }
        }

        /// <summary>
        /// Stride is given as a single value for equal dimensions in all 
        /// spatial dimensions, or once per spatial dimension.
        /// </summary>
        [Description("Specifies stride given as a single value for equal dimensions in all spatial dimensions, or once per spatial dimension.")]
        public List<uint> stride
        {
            get { return m_rgStride; }
            set { m_rgStride = value; }
        }

        /// <summary>
        /// Kernel size is given as a single value for equal dimensions in all 
        /// spatial dimensions, or once per spatial dimension.
        /// </summary>
        [Description("Specifies kernel size given as a single value for equal dimensions in all spatial dimensions, or once per spatial dimension.")]
        public List<uint> kernel_size
        {
            get { return m_rgKernelSize; }
            set { m_rgKernelSize = value; }
        }

        /// <summary>
        /// Factor used to dilate the kernel, (implicitly) zero-filling the resulting
        /// holes.  (Kernel dilation is sometimes referred to by its use in the
        /// algorithm 'a trous from Holschneider et al. 1987.)
        /// </summary>
        /// <remarks>
        /// Dilation is used by the MyCaffe.ConvolutionLayer and MyCaffe.Im2colLayer.
        /// 
        /// @see [A Real-Time Algorithm for Signal Analysis with the Help of the Wavelet Transform](https://link.springer.com/chapter/10.1007/978-3-642-75988-8_28) by Holschneider, et al., 1990.
        /// @see [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) by Yu, et al., 2015.
        /// @see [Joint Semantic and Motion Segmentation for dynamic scenes using Deep Convolutional Networks](https://arxiv.org/abs/1704.08331) by Haque, et al., 2017. 
        /// </remarks>
        [Description("Specifies dilation given as a single value for equal dimensions in all spatial dimensions, or once per spatial dimension.  When specified, this is used to dilate the kernel, (implicitly) zero-filling the resulting holes. (Kernel dilation is sometimes referred to by its use in the algorithm 'a trous from Holschneider et al. 1987.)")]
        public List<uint> dilation
        {
            get { return m_rgDilation; }
            set { m_rgDilation = value; }
        }

        /// <summary>
        /// The padding height (2D only)
        /// </summary>
        /// <remarks>
        /// For 2D only, the H and W versions may also be used to
        /// specify both spatial dimensions.
        /// </remarks>
        [Description("Specifies padding height (2D only).  'pad_h' and 'pad_w' are used -or- 'pad' is used, but not both.")]
        public uint? pad_h
        {
            get { return m_nPadH; }
            set { m_nPadH = value; }
        }

        /// <summary>
        /// The padding width (2D only)
        /// </summary>
        /// <remarks>
        /// For 2D only, the H and W versions may also be used to
        /// specify both spatial dimensions.
        /// </remarks>
        [Description("Specifies padding width (2D only).  'pad_h' and 'pad_w' are used -or- 'pad' is used, but not both.")]
        public uint? pad_w
        {
            get { return m_nPadW; }
            set { m_nPadW = value; }
        }

        /// <summary>
        /// The stride height (2D only)
        /// </summary>
        /// <remarks>
        /// For 2D only, the H and W versions may also be used to
        /// specify both spatial dimensions.
        /// </remarks>
        [Description("Specifies stride height (2D only).  'stride_h' and 'stride_w' are used -or- 'stride' is used, but not both.")]
        public uint? stride_h
        {
            get { return m_nStrideH; }
            set { m_nStrideH = value; }
        }

        /// <summary>
        /// The stride width (2D only)
        /// </summary>
        /// <remarks>
        /// For 2D only, the H and W versions may also be used to
        /// specify both spatial dimensions.
        /// </remarks>
        [Description("Specifies stride width (2D only).  'stride_h' and 'stride_w' are used -or- 'stride' is used, but not both.")]
        public uint? stride_w
        {
            get { return m_nStrideW; }
            set { m_nStrideW = value; }
        }

        /// <summary>
        /// The kernel height (2D only)
        /// </summary>
        /// <remarks>
        /// For 2D only, the H and W versions may also be used to
        /// specify both spatial dimensions.
        /// </remarks>
        [Description("Specifies kernel size height (2D only).  'kernel_h' and 'kernel_w' are used -or- 'kernel_size' is used, but not both.")]
        public uint? kernel_h
        {
            get { return m_nKernelH; }
            set { m_nKernelH = value; }
        }

        /// <summary>
        /// The kernel width (2D only)
        /// </summary>
        /// <remarks>
        /// For 2D only, the H and W versions may also be used to
        /// specify both spatial dimensions.
        /// </remarks>
        [Description("Specifies kernel size width (2D only).  'kernel_h' and 'kernel_w' are used -or- 'kernel_size' is used, but not both.")]
        public uint? kernel_w
        {
            get { return m_nKernelW; }
            set { m_nKernelW = value; }
        }

        /** @copydoc EngineParameter::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            KernelParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc EngineParameter::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is KernelParameter)
            {
                KernelParameter p = (KernelParameter)src;
                m_rgPad = Utility.Clone<uint>(p.m_rgPad);
                m_rgStride = Utility.Clone<uint>(p.m_rgStride);
                m_rgKernelSize = Utility.Clone<uint>(p.m_rgKernelSize);
                m_rgDilation = Utility.Clone<uint>(p.m_rgDilation);
                m_nPadH = p.m_nPadH;
                m_nPadW = p.m_nPadW;
                m_nStrideH = p.m_nStrideH;
                m_nStrideW = p.m_nStrideW;
                m_nKernelH = p.m_nKernelH;
                m_nKernelW = p.m_nKernelW;
            }
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            KernelParameter p = new KernelParameter();
            p.Copy(this);
            return p;
        }

        /// <summary>
        /// Convert the parameter into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the base name for the raw proto.</param>
        /// <returns>The RawProto is returned.</returns>
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("engine");
            RawProtoCollection rgChildren = new RawProtoCollection();
            KernelParameter p = new KernelParameter();

            rgChildren.Add(rpBase.Children);
            rgChildren.Add<uint>("kernel_size", m_rgKernelSize);
            rgChildren.Add<uint>("stride", m_rgStride);
            rgChildren.Add<uint>("pad", m_rgPad);

            if (m_rgDilation.Count > 0)
                rgChildren.Add<uint>("dilation", m_rgDilation);

            rgChildren.Add("kernel_h", m_nKernelH);
            rgChildren.Add("kernel_w", m_nKernelW);
            rgChildren.Add("stride_h", m_nStrideH);
            rgChildren.Add("stride_w", m_nStrideW);
            rgChildren.Add("pad_h", m_nPadH);
            rgChildren.Add("pad_w", m_nPadW);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parse a RawProto into a new instance of the parameter.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new KernelParameter FromProto(RawProto rp)
        {
            KernelParameter p = new KernelParameter();

            ((EngineParameter)p).Copy(EngineParameter.FromProto(rp));

            p.m_rgPad = rp.FindArray<uint>("pad");
            p.m_rgStride = rp.FindArray<uint>("stride");
            p.m_rgKernelSize = rp.FindArray<uint>("kernel_size");
            p.m_rgDilation = rp.FindArray<uint>("dilation");
            p.m_nPadH = (uint?)rp.FindValue("pad_h", typeof(uint));
            p.m_nPadW = (uint?)rp.FindValue("pad_w", typeof(uint));
            p.m_nStrideH = (uint?)rp.FindValue("stride_h", typeof(uint));
            p.m_nStrideW = (uint?)rp.FindValue("stride_w", typeof(uint));
            p.m_nKernelH = (uint?)rp.FindValue("kernel_h", typeof(uint));
            p.m_nKernelW = (uint?)rp.FindValue("kernel_w", typeof(uint));

            return p;
        }
    }
}
