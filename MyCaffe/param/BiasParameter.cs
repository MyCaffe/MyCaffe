using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the BiasLayer
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class BiasParameter : LayerParameterBase
    {
        int m_nAxis = 1;
        int m_nNumAxes = 1;
        FillerParameter m_filler = null;

        /** @copydoc LayerParameterBase */
        public BiasParameter()
        {
        }

        /// <summary>
        /// The first axis of bottom[0] (the first input Blob) along which to apply
        /// bottom[1] (the second input Blob).  May be negative index from end (e.g.,
        /// -1 for the last axis).
        /// </summary>
        /// <remarks>
        /// For example, if bottom[0] is 4D with shape 100x3x40x60, the output
        /// top[0] will have the same shape, and bottom[1] may have any of the
        /// following shapes (for the given value of axis):
        /// 
        ///    (axis == 0 == -4) 100; 100x3; 100x3x40;  100x3x40x60
        ///    (axis == 1 == -3)          3;     3x40;      3x40x60
        ///    (axis == 2 == -2)                   40;        40x60
        ///    (axis == 3 == -1)                                 60
        ///    
        /// Furthermore, bottom[1] may have the empty shape (regardless of the value of
        /// 'axis') -- a scalar bias.
        /// </remarks>
        [Description("Specifies the first axis of the first input Blob (bottom[0]) along which to apply the second input Blob (bottom[1]).  May be negative to index from the end (e.g., -1 for the last axis).")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// (num_axes is ignored unless just one bottom is given and the bias is
        /// a learned parameter of the layer.  Otherwise, num_axes is determined by
        /// the number of axes of the input (bottom[0] covered by the bias
        /// parameter, or -1 to cover all axes of bottom[0] starting from 'axis'.
        /// </summary>
        /// <remarks>
        /// Set num_axes := 0 to add a zero-axis Blob: a scalar.
        /// </remarks>
        [Description("'num_axes' is ignored unless just one bottom is given and the bias is a learned parameter of the layer.  Otherwise, num_axis is determined by the number of axes of the input (bottom[0] covered by the bias parameter, or -1 to cover all axes of bottom[0] starting from 'axis'.  Note: Set 'num_axis' := 0 to add a zero-axis Blob: a scalar.")]
        public int num_axes
        {
            get { return m_nNumAxes; }
            set { m_nNumAxes = value; }
        }

        /// <summary>
        /// (filler is ignored unless just one bottom is given and the bias is
        /// a learned parameter of the layer.)
        /// The initialization for the learned bias parameter.
        /// </summary>
        /// <remarks>
        /// Default is the zero (0) initialization, resulting in teh BiasLayer
        /// initially performing the identity operation.
        /// </remarks>
        [Description("Specifies the filler for the initialization of the learned bias parameter.  'filler' is ignored when more than one bottom is given and the bias is NOT a leanred parameter of the layer.  The default is the zero (0) initialization, resulting in the BiasLayer initially performing the identity operation.")]
        public FillerParameter filler
        {
            get { return m_filler; }
            set { m_filler = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            BiasParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            BiasParameter p = (BiasParameter)src;
            m_nAxis = p.m_nAxis;
            m_nNumAxes = p.m_nNumAxes;

            if (p.m_filler != null)
                m_filler = p.m_filler.Clone();
            else
                m_filler = null;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            BiasParameter p = new BiasParameter();
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

            if (num_axes != 1)
                rgChildren.Add("num_axes", num_axes.ToString());

            if (m_filler != null)
                rgChildren.Add(m_filler.ToProto("filler"));

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static BiasParameter FromProto(RawProto rp)
        {
            string strVal;
            BiasParameter p = new BiasParameter();

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("num_axes")) != null)
                p.num_axes = int.Parse(strVal);

            if ((rp = rp.FindChild("filler")) != null)
                p.m_filler = FillerParameter.FromProto(rp);

            return p;
        }
    }
}
