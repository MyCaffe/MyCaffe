using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.ComponentModel;
using MyCaffe.basecode;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies the parameters for the ReshapeLayer.
    /// </summary>
    public class ReshapeParameter : LayerParameterBase
    {
        BlobShape m_shape = new BlobShape();
        int m_nAxis = 0;
        int m_nNumAxes = -1;

        /** @copydoc LayerParameterBase */
        public ReshapeParameter()
        {
        }

        /// <summary>
        /// Specifies the output dimensions.
        /// </summary>
        /// <remarks>
        /// If some of the dimensions are set to 0,
        /// the corresponding dimension from the bottom layer is used (unchanged).
        /// Exactly one dimension may be set to -1, in which case its value is
        /// inferred from the count of the bottom blob and the remaining dimensions.
        /// 
        /// For example, suppose we want to reshape a 2D blob 'input' with shape 2 x 8:
        /// 
        /// layer {
        ///   type: 'Reshape' bottom: 'input' top: 'output'
        ///   reshape_param { ... }
        /// }
        /// 
        /// If 'input' is 2D with shape 2 x 8, then the following reshape_param
        /// specifications are all equivalent, producing a 3D blob 'output' with shape
        /// 2 x 2 x 4:
        /// 
        ///   reshape_param { shape { dim:  2 dim: 2 dim:  4 } }
        ///   reshape_param { shape { dim:  0 dim: 2 dim:  4 } }
        ///   reshape_param { shape { dim:  0 dim: 2 dim: -1 } }
        ///   reshape_param { shape { dim:  0 dim:-1 dim:  4 } }
        /// </remarks>
        [Description("Specifies the output dimensions.")]
        public BlobShape shape
        {
            get { return m_shape; }
            set { m_shape = value; }
        }

        /// <summary>
        /// Specifies the axis portion of the bottom blob's shape that is
        /// replaced by (included in) the reshape.  By default (axis == 0
        /// and num_axes == -1), the entire bottom blob shape is included 
        /// in the reshape, and hence the shape field must specify the entire 
        /// output shape.
        /// </summary>
        /// <remarks>
        /// axis may be non-zero to retain some portion of the beginning of the 
        /// input shape (and may be negative to index from the end; e.g., -1 to
        /// begin the reshape after the last axis, including nothing in the reshape,
        /// -2 to icnlude only the last axis, etc.).
        /// 
        /// For example, suppose, 'input' is the 2D blob with shape 2 x 8.
        /// Then the following RehsapeLayer specifications are all equivalent,
        /// producing a blob 'output' with shape 2 x 2 x 4:
        /// 
        ///   reshape_param { shape { dim: 2 dim: 2 dim: 4 } }
        ///   resahpe_param { shape { dim: 2 dim: 4 } axis:  1 }
        ///   reshape_param { shape [ dim: 2 dim: 4 } axis: -3 }
        /// </remarks>
        [Description("Specifies the axis portion of the bottom blob's shape that is replaced by (included in) the reshape.  By default (axis == 0 and num_axes == 1), the entire bottom blob is included in the reshape, and hence the shape field must specifiethe entire output shape.")]
        public int axis
        {
            get { return m_nAxis; }
            set { m_nAxis = value; }
        }

        /// <summary>
        /// num_axes specifies the extent of the reshape.
        /// </summary>
        /// <remarks>
        /// If num_axes >= 0 (and axis >= 0), the reshape will be performed only on
        /// input axes in the range [axis, axis+num_axes].
        /// num_axes may also be -1, the default, to include all remaining axes
        /// (starting from axis).
        /// 
        /// For example, suppose, 'input' is the 2D blob with shape 2 x 8.
        /// Then the following RehsapeLayer specifications are all equivalent,
        /// producing a blob 'output' with shape 1 x 2 x 8:
        /// 
        ///   reshape_param { shape { dim: 1 dim: 2 dim: 8 } }
        ///   resahpe_param { shape { dim: 1 dim: 2 } num_axes: 1 }
        ///   reshape_param { shape [ dim: 1 } num_axes: 0 }
        ///   
        /// On the other hand, these would produce output blob shape 2 x 1 x 8:
        /// 
        ///   reshape_param { shape { dim: 2 dim: 1 dim: 8 } }
        ///   resahpe_param { shape { dim: 1 } axis: 1 num_axes: 0 }
        /// </remarks>
        [Description("Specifies the number of axes which specifies the extent of the reshape.")]
        public int num_axes
        {
            get { return m_nNumAxes; }
            set { m_nNumAxes = value; }
        }

        /** @copydoc LayerParameterBase::Load */
        public override object Load(System.IO.BinaryReader br, bool bNewInstance = true)
        {
            RawProto proto = RawProto.Parse(br.ReadString());
            ReshapeParameter p = FromProto(proto);

            if (!bNewInstance)
                Copy(p);

            return p;
        }

        /** @copydoc LayerParameterBase::Copy */
        public override void Copy(LayerParameterBase src)
        {
            ReshapeParameter p = (ReshapeParameter)src;

            if (p.m_shape != null)
                m_shape = p.m_shape.Clone();

            m_nAxis = p.m_nAxis;
            m_nNumAxes = p.m_nNumAxes;
        }

        /** @copydoc LayerParameterBase::Clone */
        public override LayerParameterBase Clone()
        {
            ReshapeParameter p = new ReshapeParameter();
            p.Copy(this);
            return p;
        }

        /** @copydoc LayerParameterBase::ToProto */
        public override RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add(shape.ToProto("shape"));
    
            if (axis != 0)
                rgChildren.Add("axis", axis.ToString());

            if (num_axes != -1)
                rgChildren.Add("num_axes", num_axes.ToString());

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// Parses the parameter from a RawProto.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>A new instance of the parameter is returned.</returns>
        public static ReshapeParameter FromProto(RawProto rp)
        {
            string strVal;
            ReshapeParameter p = new ReshapeParameter();

            RawProto rpShape = rp.FindChild("shape");
            if (rpShape != null)
                p.shape = BlobShape.FromProto(rpShape);

            if ((strVal = rp.FindValue("axis")) != null)
                p.axis = int.Parse(strVal);

            if ((strVal = rp.FindValue("num_axes")) != null)
                p.num_axes = int.Parse(strVal);

            return p;
        }
    }
}
