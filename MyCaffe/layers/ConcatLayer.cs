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
    /// The ConcatLayer takes at least two Blob%s and concatentates them along either the num
    /// or channel dimension, outputing the result.
    /// This layer is initialized with the MyCaffe.param.ConcatParameter.
    /// </summary>
    /// <remarks>
    /// @see [Deep Image Aesthetics Classification using Inception Modules and Fine-tuning Connected Layer](https://arxiv.org/abs/1610.02256) by Xin Jin, Jingying Chi, Siwei Peng, Yulu Tian, Chaochen Ye, and Xiaodong Li, 2016.
    /// @see [Multi-path Convolutional Neural Networks for Complex Image Classification](https://arxiv.org/abs/1506.04701) by Mingming Wang, 2015.
    /// @see [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) by Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna, 2015.
    /// @see [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) by Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alex Alemi, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ConcatLayer<T> : Layer<T>
    {
        int m_nNumConcats;
        int m_nConcatInputSize;
        int m_nConcatAxis;

        /// <summary>
        /// The ConcatLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type CONCAT and concat_param.
        /// </param>
        public ConcatLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.CONCAT;
        }

        /// <summary>
        /// Returns the minimum number of required bottom (input) Blobs: input
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: concat
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
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNumAxes = colBottom[0].num_axes;

            if (m_param.concat_param.concat_dim > 0)
            {
                m_nConcatAxis = (int)m_param.concat_param.concat_dim;
                // Don't allow negative indexing for concat_dim, a uint -- almost certainly
                // unintended.
                m_log.CHECK_GE(m_nConcatAxis, 0, "Casting concat_dim from uint to int produced a negative result; concat_dim must be > 0.");
                m_log.CHECK_LT(m_nConcatAxis, nNumAxes, "concat_dim out of range.");
            }
            else
            {
                m_nConcatAxis = colBottom[0].CanonicalAxisIndex(m_param.concat_param.axis);
            }

            // Initialize with the first blob.
            List<int> rgTopShape = Utility.Clone<int>(colBottom[0].shape());
            m_nNumConcats = colBottom[0].count(0, m_nConcatAxis);
            m_nConcatInputSize = colBottom[0].count(m_nConcatAxis + 1);

            int nBottomCountSum = colBottom[0].count();

            for (int i = 1; i < colBottom.Count; i++)
            {
                m_log.CHECK_EQ(nNumAxes, colBottom[i].num_axes, "All inputs must have the same # axes.");

                for (int j = 0; j < nNumAxes; j++)
                {
                    if (j == m_nConcatAxis)
                        continue;

                    m_log.CHECK_EQ(rgTopShape[j], colBottom[i].shape(j), "All inputs must have the same shape, except at concat_axis.  You might try switching between the ONNX(p) and CAFFE(t) type pooling sizing methods.");
                }

                nBottomCountSum += colBottom[i].count();
                rgTopShape[m_nConcatAxis] += colBottom[i].shape(m_nConcatAxis);
            }

            colTop[0].Reshape(rgTopShape);
            m_log.CHECK_EQ(nBottomCountSum, colTop[0].count(), "The bottomCountSums should equal the top[0].count.");

            if (colBottom.Count == 1)
            {
                colTop[0].ShareData(colBottom[0]);
                colTop[0].ShareDiff(colBottom[0]);
            }
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">bottom input blob (length 2+)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x_1 @f$
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x_2 @f$
        ///  -# ...
        ///  - K @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x_K @f$</param>
        /// <param name="colTop">top output blob (length 1)
        ///  -# @f$ (KN \times C \times H \times W) @f$ if axis == 0 or
        ///  -# @f$ (N \times KC \times H \times W) @f$ if axis == 1;
        ///     the concatentation output @f$
        ///       y = [\begin{array}{cccc} x_1 \: x_2 \: ... \: x_k \end{array}]
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (colBottom.Count == 1)
                return;

            long hTopData = colTop[0].mutable_gpu_data;
            int nOffsetConcatAxis = 0;
            int nTopConcatAxis = colTop[0].shape(m_nConcatAxis);

            for (int i = 0; i < colBottom.Count; i++)
            {
                long hBottomData = colBottom[i].gpu_data;
                int nBottomConcatAxis = colBottom[i].shape(m_nConcatAxis);
                int nBottomConcatSize = nBottomConcatAxis * m_nConcatInputSize;
                int nCount = nBottomConcatSize * m_nNumConcats;

                m_cuda.concat_fwd(nCount, hBottomData, m_nNumConcats, m_nConcatInputSize, nTopConcatAxis, nBottomConcatAxis, nOffsetConcatAxis, hTopData);
                nOffsetConcatAxis += nBottomConcatAxis;
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the concatenation inputs.
        /// </summary>
        /// <param name="colTop">top output blob (length 1), providing the error gradient with
        /// respect to the outputs.
        ///  -# @f$ (KN \times C \times H \times W) @f$ if axis == 0, or
        ///     @f$ (N \times KC \times H \times W) @f$ if axis == 1:
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to concatenated outputs y.</param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">input blob vector (length K), into which the top gradient
        ///     @f$ \frac{\partial E}{\partial y} @f$ is deconcatenated back to the
        ///     inputs @f$ 
        ///     \left[ \begin{array}{cccc}
        ///      \frac{\partial E}{\partial x_1} \:
        ///      \frac{\partial E}{\partial x_2} \:
        ///      ... \:
        ///      \frac{\partial E}{\partial x_K}
        ///      \end{array} \right] = 
        ///      \frac{\partial E}{\partial y}
        ///      @f$
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (colBottom.Count == 1)
                return;

            long hTopDiff = colTop[0].gpu_diff;
            int nOffsetConcatAxis = 0;
            int nTopConcatAxis = colTop[0].shape(m_nConcatAxis);

            for (int i = 0; i < colBottom.Count; i++)
            {
                int nBottomConcatAxis = colBottom[i].shape(m_nConcatAxis);

                if (rgbPropagateDown[i])
                {
                    long hBottomDiff = colBottom[i].mutable_gpu_diff;
                    int nBottomConcatSize = nBottomConcatAxis * m_nConcatInputSize;
                    int nCount = nBottomConcatSize * m_nNumConcats;

                    m_cuda.concat_bwd(nCount, hTopDiff, m_nNumConcats, m_nConcatInputSize, nTopConcatAxis, nBottomConcatAxis, nOffsetConcatAxis, hBottomDiff);
                }

                nOffsetConcatAxis += nBottomConcatAxis;
            }
        }
    }
}
