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
