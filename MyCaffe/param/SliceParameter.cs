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
    /// Specifies the parameters for the SliceLayer.
    /// </summary>
    public class SliceParameter : LayerParameterBase
    {
        int m_nAxis = 1;
        List<uint> m_rgSlicePoint = new List<uint>();
        uint m_nSliceDim = 0;

        /** @copydoc LayerParameterBase */
        public SliceParameter()
        {
        }

        /// <summary>
        /// Specifies the axis along wich to slice -- may be negative to index from the end
        /// (e.g., -1 for the last axis).
        /// By default, SliceLayer concatenates blobs along the 'channels' axis 1.
        /// </summary>
        [Description("Specifies the axis along which to slice - may be negative to index from the end (e.g., -1 for the last axis).")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// Specifies optional slice points which indicate the indexes in the selected dimensions (the number of indices must be equal to the number of top blobs minus one).
        /// </summary>
        [Description("Specifies the optional slice points which indicate the indexes in the selected dimensions (the number of indices must be equal to the number of top blobs minus one).")]
        public List<uint> slice_point
        {
            get { return m_rgSlicePoint; }
            set { m_rgSlicePoint = value; }
        }

        /// <summary>
        /// <b>DEPRECIATED:</b> alias for 'axis' -- does not support negative indexing.
        /// </summary>
        [Description("DEPRECIATED - use 'axis' instead.")]
        [Browsable(false)]
        public uint slice_dim
        {
            get { return m_nSliceDim; }
            set { m_nSliceDim = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            SliceParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            SliceParameter p = (SliceParameter)src;

            m_nAxis = p.m_nAxis;
            m_rgSlicePoint = Utility.Clone<uint>(p.m_rgSlicePoint);
            m_nSliceDim = p.m_nSliceDim;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            SliceParameter p = new SliceParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("axis", axis.ToString());
            rgChildren.Add<uint>("slice_point", slice_point);

            if (slice_dim != 0)
                rgChildren.Add("slice_dim", slice_dim.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static SliceParameter FromProto(RawProto rp)
        {
            string strVal;
            SliceParameter p = new SliceParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            p.slice_point = rp.FindArray<uint>("slice_point");

            if ((strVal = rp.FindValue("slice_dim")) != null)
                p.slice_dim = uint.Parse(strVal);

            return p;
        }
    }
}
