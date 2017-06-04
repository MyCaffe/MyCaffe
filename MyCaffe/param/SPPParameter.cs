using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// The SPPParameter specifies the parameters for the SPPLayer.
    /// </summary>
    /// <remarks>
    /// @see [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, 2014.
    /// @see [Image-based Localization using Hourglass Networks](https://arxiv.org/abs/1703.07971v1) by Iaroslav Melekhov, Juha Ylioinas, Juho Kannala, and Esa Rahtu, 2017.
    /// @see [Relative Camera Pose Estimation Using Convolutional Neural Networks](https://arxiv.org/abs/1702.01381v2) by Iaroslav Melekhov, Juha Ylioinas, Juho Kannala, Esa Rahtu, 2017.
    /// </remarks>
    public class SPPParameter : EngineParameter 
    {
        PoolingParameter.PoolingMethod m_method = PoolingParameter.PoolingMethod.MAX;
        uint m_nPyramidHeight;

        /** @copydoc EngineParameter */
        public SPPParameter()
            : base()
        {
        }

        /// <summary>
        /// Specifies the pooling method to use.
        /// </summary>
        [Description("Specifies the pooling method to use.")]
        public PoolingParameter.PoolingMethod pool
        {
            get { return m_method; }
            set { m_method = value; }
        }

        /// <summary>
        /// Specifies the pyramid height.
        /// </summary>
        [Description("Specifies the pyramid height.")]
        public uint pyramid_height
        {
            get { return m_nPyramidHeight; }
            set { m_nPyramidHeight = value; }
        }

        /** @copydoc EngineParameter::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            SPPParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc EngineParameter::Copy */
        public override void Copy(LayerParameterBase src)
        {
            base.Copy(src);

            if (src is SPPParameter)
            {
                SPPParameter p = (SPPParameter)src;
                m_method = p.m_method;
                m_nPyramidHeight = p.m_nPyramidHeight;
            }
        }

        /** @copydoc EngineParameter::Clone */
        public override LayerParameterBase Clone()
        {
            SPPParameter p = new SPPParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc EngineParameter::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProto rpBase = base.ToProto("engine");
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(rpBase.Children);
            rgChildren.Add("method", pool.ToString());
            rgChildren.Add("pyramid_height", pyramid_height.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static new SPPParameter FromProto(RawProto rp)
        {
            string strVal;
            SPPParameter p = new SPPParameter();

            p.Copy(EngineParameter.FromProto(rp));

            if ((strVal = rp.FindValue("method")) != null)
            {
                switch (strVal)
                {
                    case "MAX":
                        p.pool = PoolingParameter.PoolingMethod.MAX;
                        break;

                    case "AVE":
                        p.pool = PoolingParameter.PoolingMethod.AVE;
                        break;

                    case "STOCHASTIC":
                        p.pool = PoolingParameter.PoolingMethod.STOCHASTIC;
                        break;

                    default:
                        throw new Exception("Unknown pooling 'method' value: " + strVal);
                }
            }

            if ((strVal = rp.FindValue("pyramid_height")) != null)
                p.pyramid_height = uint.Parse(strVal);

            return p;
        }
    }
}
