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
    /// The GramLayer computes the Gram matrix used in Neural Style.
    /// </summary>
    /// <remarks>
    /// @see [ftokarev/caffe-neural-style Github](https://github.com/ftokarev/caffe-neural-style) by ftokarev, 2017. 
    /// @see [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, 2015 
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class GramLayer<T> : Layer<T>
    {
        int m_nK;
        int m_nM;
        int m_nN;

        /// <summary>
        /// The GramLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter.</param>
        public GramLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.GRAM;
        }

        /// <summary>
        /// Returns the exact number of bottom blobs (e.g. 1)
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of bottom blobs (e.g. 1)
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
            int nAxis = colBottom[0].CanonicalAxisIndex(m_param.gram_param.axis);

            // Dimensions starting from 'axis' are 'flattened' into a single length 'K' vector.
            m_nK = colBottom[0].count(nAxis);

            // The first 'axis-1' dimensions are independent Gram matrices; the total
            // number of these is 'M' the product over these dimensions.
            m_nM = colBottom[0].count(0, nAxis - 1);

            // Gram matrices will be 'N' by 'N'
            m_nN = colBottom[0].shape(nAxis - 1);

            List<int> rgTopShape = Utility.Clone<int>(colBottom[0].shape(), nAxis + 1);
            rgTopShape[nAxis] = m_nN;

            colTop[0].Reshape(rgTopShape);
        }

        /// <summary>
        /// Computes the Gram matrix values.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs x</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs for the Gram matrix.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            for (int i = 0; i < m_nM; i++)
            {
                m_cuda.gemm(false, true, m_nN, m_nN, m_nK, m_tOne, hBottomData, hBottomData, m_tZero, hTopData, i * m_nK * m_nN, i * m_nK * m_nN, i * m_nN * m_nN);
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the absolute value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            long hTopDiff = colTop[0].gpu_diff;
            long hBottomData = colBottom[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            for (int i = 0; i < m_nM; i++)
            {
                m_cuda.gemm(false, false, m_nN, m_nK, m_nN, m_tOne, hTopDiff, hBottomData, m_tZero, hBottomDiff, i * m_nN * m_nN, i * m_nK * m_nN, i * m_nK * m_nN);
                m_cuda.gemm(true, false, m_nN, m_nK, m_nN, m_tOne, hTopDiff, hBottomData, m_tOne, hBottomDiff, i * m_nN * m_nN, i * m_nK * m_nN, i * m_nK * m_nN);
            }
        }
    }
}
