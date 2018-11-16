using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The ReshapeLayer reshapes the input Blob into an arbitrary-sized output Blob.
    /// This layer is initialized with the MyCaffe.param.ReshapeParameter.
    /// </summary>
    /// <remarks>
    /// Note: similarly to FlattenLayer, this layer does not change the input values
    /// (see FlattenLayer, Blob::ShareData and Blob::ShareDiff).
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ReshapeLayer<T> : Layer<T>
    {
        /// <summary>
        /// Vector of axes indices whos dimensions we'll copy from the bottom.
        /// </summary>
        List<int> m_rgCopyAxes = new List<int>();
        /// <summary>
        /// The index of the axis whose dimension we infer, or -1 if none.
        /// </summary>
        int m_nInferredAxis;
        /// <summary>
        /// The product of the 'constant' output dimensions.
        /// </summary>
        int m_nConstantCount;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides ArgMaxParameter argmax_param
        /// </param>
        public ReshapeLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.RESHAPE;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: reshape
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK(colTop[0] != colBottom[0], type.ToString() + " Layer does not allow in-place computation.");
            m_nInferredAxis = -1;
            m_nConstantCount = 1;
            m_rgCopyAxes = ReshapeParameter.CalculateCopyAxes(m_param, out m_nInferredAxis, out m_nConstantCount);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            //int input_start_axis = m_param.reshape_param.axis;
            //int start_axis = (input_start_axis >= 0) ? input_start_axis : colBottom[0].num_axes + input_start_axis + 1;

            //m_log.CHECK_GE(start_axis, 0, "axis " + input_start_axis.ToString() + " out of range");
            //m_log.CHECK_LE(start_axis, colBottom[0].num_axes, "axis " + input_start_axis.ToString() + " out of range for " + colBottom[0].num_axes.ToString() + "-D input blob");

            //int num_axes = m_param.reshape_param.num_axes;
            //m_log.CHECK_GE(num_axes, -1, "num_axes must be >= 0, or -1 for all");

            //int end_axis = (num_axes == -1) ? colBottom[0].num_axes : (start_axis + num_axes);
            //m_log.CHECK_LE(end_axis, colBottom[0].num_axes, "end_axis = axis + num_axes is out of range");

            //int num_axes_replaced = end_axis - start_axis;
            //int num_axes_retained = colBottom[0].num_axes - num_axes_replaced;
            //BlobShape top_blob_shape = m_param.reshape_param.shape;
            //int num_new_axes = top_blob_shape.dim.Count;
            //List<int> rgTopShape = new List<int>();
            //int top_shape_index = 0;

            //for (int i = 0; i < start_axis; i++)
            //{
            //    rgTopShape.Add(colBottom[0].shape(i));
            //    top_shape_index++;
            //}

            //for (int i = 0; i < num_new_axes; i++)
            //{
            //    rgTopShape.Add(top_blob_shape.dim[i]);
            //    top_shape_index++;
            //}

            //for (int i = end_axis; i < colBottom[0].num_axes; i++)
            //{
            //    rgTopShape.Add(colBottom[0].shape(i));
            //    top_shape_index++;
            //}

            //m_log.CHECK_EQ(top_shape_index, rgTopShape.Count, "The top shape count should equal the top_shape_index.");

            //for (int i = 0; i < m_rgCopyAxes.Count; i++)
            //{
            //    int copy_axis_index = m_rgCopyAxes[i];
            //    m_log.CHECK_GT(colBottom[0].num_axes, start_axis + copy_axis_index, "new shape contains a 0, but there was no corresponding bottom axis to copy");
            //    rgTopShape[start_axis + copy_axis_index] = colBottom[0].shape(start_axis + copy_axis_index);
            //}

            //if (m_nInferredAxis >= 0)
            //{
            //    // A -1 dim was specified; infer the correct dimension by computing the
            //    // product of the other dimensions.
            //    int explicit_count = m_nConstantCount;
            //    explicit_count *= colBottom[0].count(0, start_axis);
            //    explicit_count *= colBottom[0].count(end_axis);

            //    for (int i = 0; i < m_rgCopyAxes.Count; i++)
            //    {
            //        int copy_axis_index = m_rgCopyAxes[i];
            //        explicit_count *= rgTopShape[start_axis + copy_axis_index];
            //    }

            //    m_log.CHECK_EQ(0, colBottom[0].count() % explicit_count, "bottom count (" + colBottom[0].count().ToString() + ") must be divisible by the product of the specified dimensions( " + explicit_count.ToString() + ")");

            //    int inferred_dim = colBottom[0].count() / explicit_count;
            //    rgTopShape[start_axis + m_nInferredAxis] = inferred_dim;
            //}
            List<int> rgTopShape = ReshapeParameter.Reshape(m_param, colBottom[0].shape(), m_rgCopyAxes, m_nInferredAxis, m_nConstantCount, m_log);

            colTop[0].Reshape(rgTopShape);
            m_log.CHECK_EQ(colTop[0].count(), colBottom[0].count(), "output count must match input count");

            colTop[0].ShareData(colBottom[0]);
            colTop[0].ShareDiff(colBottom[0]);
        }

        /// @brief Not implemented - reshape Layers do not perform forward, reshaping is performed in Reshape().
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
        }

        /// @brief Not implemented - reshape Layers do not perform backward.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }
    }
}
