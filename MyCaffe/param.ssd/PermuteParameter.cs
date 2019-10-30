using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param.ssd
{
    /// <summary>
    /// Specifies the parameters for the PermuteLayer.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class PermuteParameter : LayerParameterBase 
    {
        List<int> m_rgOrder = new List<int>();

        /** @copydoc LayerParameterBase */
        public PermuteParameter()
        {
        }

        /// <summary>
        /// Specifies the new orders of the axes of data.
        /// </summary>
        /// <remarks>
        /// Notice that the data should be with in the same range as
        /// the input data, and that it starts from 0.
        /// Do not provide a repeated order.
        /// </remarks>
        [Description("Specifies the new orders of the axes of data.  Should be within the same range as the input data starting with 0 and no repeated orders.")]
        public List<int> order
        {
            get { return m_rgOrder; }
            set { m_rgOrder = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            PermuteParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            PermuteParameter p = (PermuteParameter)src;
            m_rgOrder = Utility.Clone<int>(p.order);
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            PermuteParameter p = new PermuteParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            foreach (int nOrder in m_rgOrder)
            {
                rgChildren.Add("order", nOrder.ToString());
            }

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static PermuteParameter FromProto(RawProto rp)
        {
            PermuteParameter p = new PermuteParameter();

            RawProtoCollection rgChildren = rp.FindChildren("order");
            foreach (RawProto rp1 in rgChildren)
            {
                p.order.Add(int.Parse(rp1.Value));
            }

            return p;
        }

        /// <summary>
        /// Calculates the top shape by running the Reshape calculation.
        /// </summary>
        /// <param name="rgOrder">Specifies the ordering to use.</param>
        /// <param name="rgShape">Specifies the original shape to re-order.</param>
        /// <param name="nNumAxes">Specifies the number of axes.</param>
        /// <returns>The new shape based on the ordering is returned.</returns>
        public static List<int> Reshape(List<int> rgOrder, List<int> rgShape, int nNumAxes)
        {
            List<int> rgTopShape = new List<int>();

            for (int i = 0; i < nNumAxes; i++)
            {
                int nOrder = rgOrder[i];
                int nShape = rgShape[nOrder];
                rgTopShape.Add(nShape);
            }

            return rgTopShape;
        }
    }
}
