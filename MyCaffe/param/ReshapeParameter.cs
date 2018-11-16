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

        private static int count(List<int> rg, int nStart, int nEnd)
        {
            int nCount = 1;

            for (int i = nStart; i < nEnd; i++)
            {
                nCount *= rg[i];
            }

            return nCount;
        }

        /// <summary>
        /// Calculate the Copy Axes, inferred axis and constant count.
        /// </summary>
        /// <param name="p">Specifies the layer parameters.</param>
        /// <param name="nInferredAxis">Returns the inferred axis.</param>
        /// <param name="nConstantCount">Returns the constant count.</param>
        /// <param name="log">Optionally, specifies the output log (default = null, which ignores this parameter).</param>
        /// <returns>The copy axes are returned.</returns>
        public static List<int> CalculateCopyAxes(LayerParameter p, out int nInferredAxis, out int nConstantCount, Log log = null)
        {
            List<int> rgCopyAxes = new List<int>();

            nInferredAxis = -1;
            nConstantCount = 1;

            BlobShape top_blob_shape = p.reshape_param.shape;
            int top_num_axes = top_blob_shape.dim.Count();

            for (int i = 0; i < top_num_axes; i++)
            {
                int top_dim = top_blob_shape.dim[i];

                if (top_dim == 0)
                {
                    rgCopyAxes.Add(i);
                }
                else if (top_dim == -1)
                {
                    if (log != null)
                        log.CHECK_EQ(nInferredAxis, -1, "new shape contains multiple -1 dims; at most a single (1) value of -1 may be specified.");

                    nInferredAxis = i;
                }
                else
                {
                    nConstantCount *= top_dim;
                }
            }

            return rgCopyAxes;
        }

        /// <summary>
        /// Calculates the new shape.
        /// </summary>
        /// <param name="p">Specifies the layer parameters.</param>
        /// <param name="rgShape">Specifies the Bottom[0] shape.</param>
        /// <param name="rgCopyAxes">Specifies the copy axes.</param>
        /// <param name="nInferredAxis">Specifies the inferred axis (if any).</param>
        /// <param name="nConstantCount">Specifies the constant count.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <returns>The new top shape is returned.</returns>
        public static List<int> Reshape(LayerParameter p, List<int> rgShape, List<int> rgCopyAxes, int nInferredAxis, int nConstantCount, Log log = null)
        {
            int num_axes1 = rgShape.Count;
            int input_start_axis = p.reshape_param.axis;
            int start_axis = (input_start_axis >= 0) ? input_start_axis : num_axes1 + input_start_axis + 1;

            if (log != null)
            {
                log.CHECK_GE(start_axis, 0, "axis " + input_start_axis.ToString() + " out of range");
                log.CHECK_LE(start_axis, num_axes1, "axis " + input_start_axis.ToString() + " out of range for " + num_axes1.ToString() + "-D input blob");
            }

            int num_axes = p.reshape_param.num_axes;
            if (log != null)
                log.CHECK_GE(num_axes, -1, "num_axes must be >= 0, or -1 for all");

            int end_axis = (num_axes == -1) ? num_axes1 : (start_axis + num_axes);
            if (log != null)
                log.CHECK_LE(end_axis, num_axes1, "end_axis = axis + num_axes is out of range");

            int num_axes_replaced = end_axis - start_axis;
            int num_axes_retained = num_axes1 - num_axes_replaced;
            BlobShape top_blob_shape = p.reshape_param.shape;
            int num_new_axes = top_blob_shape.dim.Count;
            List<int> rgTopShape = new List<int>();
            int top_shape_index = 0;

            for (int i = 0; i < start_axis; i++)
            {
                rgTopShape.Add(rgShape[i]);
                top_shape_index++;
            }

            for (int i = 0; i < num_new_axes; i++)
            {
                rgTopShape.Add(top_blob_shape.dim[i]);
                top_shape_index++;
            }

            for (int i = end_axis; i < num_axes1; i++)
            {
                rgTopShape.Add(rgShape[i]);
                top_shape_index++;
            }

            if (log != null)
                log.CHECK_EQ(top_shape_index, rgTopShape.Count, "The top shape count should equal the top_shape_index.");

            for (int i = 0; i < rgCopyAxes.Count; i++)
            {
                int copy_axis_index = rgCopyAxes[i];

                if (log != null)
                    log.CHECK_GT(num_axes1, start_axis + copy_axis_index, "new shape contains a 0, but there was no corresponding bottom axis to copy");

                rgTopShape[start_axis + copy_axis_index] = rgShape[start_axis + copy_axis_index];
            }

            if (nInferredAxis >= 0)
            {
                // A -1 dim was specified; infer the correct dimension by computing the
                // product of the other dimensions.
                int explicit_count = nConstantCount;
                explicit_count *= count(rgShape, 0, start_axis);
                explicit_count *= count(rgShape, end_axis, rgShape.Count);

                for (int i = 0; i < rgCopyAxes.Count; i++)
                {
                    int copy_axis_index = rgCopyAxes[i];
                    explicit_count *= rgTopShape[start_axis + copy_axis_index];
                }

                int nCount = count(rgShape, 0, rgShape.Count);

                if (log != null)
                    log.CHECK_EQ(0, nCount % explicit_count, "bottom count (" + nCount.ToString() + ") must be divisible by the product of the specified dimensions( " + explicit_count.ToString() + ")");

                int inferred_dim = nCount / explicit_count;
                rgTopShape[start_axis + nInferredAxis] = inferred_dim;
            }

            return rgTopShape;
        }
    }
}
