using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.beta
{
    /// <summary>
    /// Specifies the parameters for the UnPoolingLayer.
    /// </summary>
    /// <remarks>
    /// @see [A Deep Convolutional Auto-Encoder with Pooling - Unpooling Layers in Caffe](https://arxiv.org/abs/1701.04949) by Volodymyr Turchenko, Eric Chalmers, Artur Luczak, 2017.
    /// @see [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) by Vincent Dumoulin and Francesco Visin, 2016.
    /// @see [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150) by Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba, 2015.
    /// @see [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) by Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner, 1998.
    /// @see [Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation](https://arxiv.org/abs/1506.04924) by Seunghoon Hong, Hyeonwoo Noh, and Bohyung Han, 2015.
    /// </remarks>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class UnPoolingParameter : PoolingParameter 
    {
        List<uint> m_rgUnpool = new List<uint>();
        uint? m_nUnPoolH = null;
        uint? m_nUnPoolW = null;

        /** @copydoc PoolingParameter */
        public UnPoolingParameter()
        {
        }

        /// <summary>
        /// Returns the reason that Caffe version was used instead of [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns></returns>
        public new string useCaffeReason()
        {
            return "Not supported by cuDNN.";
        }

        /// <summary>
        /// Queries whether or not to use [NVIDIA's cuDnn](https://developer.nvidia.com/cudnn).
        /// </summary>
        /// <returns>Returns <i>true</i> when cuDnn is to be used, <i>false</i> otherwise.</returns>
        public new bool useCudnn()
        {
            return false;
        }

        /// <summary>
        /// UnPool size is given as a single value for equal dimensions in all 
        /// spatial dimensions, or once per spatial dimension.
        /// </summary>
        [Description("Specifies unpool size override given as a single value for equal dimensions in all spatial dimensions, or once per spatial dimension.")]
        public List<uint> unpool_size
        {
            get { return m_rgUnpool; }
            set { m_rgUnpool = value; }
        }

        /// <summary>
        /// The unpooling height override (2D only)
        /// </summary>
        /// <remarks>
        /// For 2D only, the H and W versions may also be used to
        /// specify both spatial dimensions.
        /// </remarks>
        [Description("Specifies unpooling height override (2D only).  'unpool_h' and 'unpool_w' are used -or- 'unpool_size' is used, but not both.")]
        public uint? unpool_h
        {
            get { return m_nUnPoolH; }
            set { m_nUnPoolH = value; }
        }

        /// <summary>
        /// The unpooling width override (2D only)
        /// </summary>
        /// <remarks>
        /// For 2D only, the H and W versions may also be used to
        /// specify both spatial dimensions.
        /// </remarks>
        [Description("Specifies unpooling width override (2D only).  'unpool_h' and 'unpool_w' are used -or- 'unpool_size' is used, but not both.")]
        public uint? unpool_w
        {
            get { return m_nUnPoolW; }
            set { m_nUnPoolW = value; }
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
            UnPoolingParameter p = FromProto(proto) as UnPoolingParameter;

            if (p == null)
                throw new Exception("Expected UnPoolingParameter type!");

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
            base.Copy(src);

            if (src is UnPoolingParameter)
            {
                UnPoolingParameter p = (UnPoolingParameter)src;

                m_rgUnpool = Utility.Clone<uint>(p.m_rgUnpool);
                m_nUnPoolH = p.m_nUnPoolH;
                m_nUnPoolW = p.m_nUnPoolW;
            }
        }

        /// <summary>
        /// Creates a new copy of this instance of the parameter.
        /// </summary>
        /// <returns>A new instance of this parameter is returned.</returns>
        public override LayerParameterBase Clone()
        {
            UnPoolingParameter p = new UnPoolingParameter();
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
            dilation.Clear();
            RawProto rpBase = base.ToProto("pooling");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);
            rgChildren.Add<uint>("unpool_size", m_rgUnpool);
            rgChildren.Add("unpool_h", m_nUnPoolH);
            rgChildren.Add("unpool_w", m_nUnPoolW);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new UnPoolingParameter FromProto(RawProto rp)
        {
            UnPoolingParameter p = new UnPoolingParameter();           

            ((PoolingParameter)p).Copy(PoolingParameter.FromProto(rp));

            p.m_rgUnpool = rp.FindArray<uint>("unpool_size");
            p.m_nUnPoolH = (uint?)rp.FindValue("unpool_h", typeof(uint));
            p.m_nUnPoolW = (uint?)rp.FindValue("unpool_w", typeof(uint));

            return p;
        }
    }
}
