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
    /// Specifies the parameters for the MyCaffe.CropLayer.
    /// </summary>
    /// <remarks>
    /// To crop, elements of the first bottom are selected to fit the dimensions
    /// of the second, reference bottom.  The crop is configured by 
    /// - the crop 'axis' to pick the diensions for cropping.
    /// - the crop 'offset' to set the shift for all/each dimension.
    /// 
    /// @see [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer, and Trevor Darrell, 2014.
    /// </remarks>
    public class CropParameter : LayerParameterBase 
    {
        int m_nAxis = 2;
        List<uint> m_rgOffset = new List<uint>();

        /** @copydoc LayerParameterBase */
        public CropParameter()
        {
        }

        /// <summary>
        /// The axis along which to crop -- may be negative to index from the
        /// end (e.g., -1 for the last axis). Default is 2 for spatial crop.
        /// </summary>
        /// <remarks>
        /// All dimensions up to but excluding 'axis' are preserved, while the 
        /// dimensions  including the trailing 'axis' are cropped.
        /// </remarks>
        [Description("The axis along which to crop - may be negative to index from the end (e.g., -1 for the last axis).  Other axes must have the same dimension for all the bottom blobs.  By default, the ConcatLayer concatenates blobs along the 'channel' axis 1.")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Specifies the offset to set the shift for all/each dimension. 
        /// </summary>
        /// <remarks>
        /// If only one 'offset is set, then all dimensions are offset by this amount.
        /// Otherwise, the number of offsets must equal the number of cropped axes to
        /// shift the crop in each dimension accordingly.
        /// </remarks>
        [Description("")]
        public List<uint> offset
        {
            get { return m_rgOffset; }
            set { m_rgOffset = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            CropParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            CropParameter p = (CropParameter)src;
            m_nAxis = p.m_nAxis;
            m_rgOffset = Utility.Clone<uint>(p.m_rgOffset);
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            CropParameter p = new CropParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            if (axis != 1)
                rgChildren.Add("axis", axis.ToString());

            rgChildren.Add<uint>("offset", offset);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static CropParameter FromProto(RawProto rp)
        {
            string strVal;
            CropParameter p = new CropParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            p.offset = rp.FindArray<uint>("offset");

            return p;
        }
    }
}
