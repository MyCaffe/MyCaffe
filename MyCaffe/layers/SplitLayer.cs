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
    /// The SplitLayer creates a 'split' path in the network by copying the bottom blob
    /// into multiple top blob's to be used by multiple consuming layers.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SplitLayer<T> : Layer<T>
    {
        int m_nCount;

        /// <summary>
        /// The SplitLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type SPLIT.</param>
        public SplitLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SPLIT;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: split
        /// </summary>
        public override int MinTopBlobs
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
            m_bConvertBottom = false;

            // Copy data during setup through
            // forward pass to ensure variables
            // are passed through.
            Reshape(colBottom, colTop);
            forward(colBottom, colTop);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nCount = colBottom[0].count();

            for (int i = 0; i < colTop.Count; i++)
            {
                // Do not allow in-place computation in the SplitLayer.  Instead, share data by 
                // reference in the forward pass, and keep separeate diff allocations in
                // the backward pass.  (Technically, it should be possible to share the diff
                // blob of the first split output with the input, but this seems to cause
                // some strange effects in practice...)
                m_log.CHECK(colTop[i].gpu_data != colBottom[0].gpu_data, "Layer does not allow in-place computation.");
                colTop[i].ReshapeLike(colBottom[0], colBottom[0].HalfSize);
                m_log.CHECK_EQ(m_nCount, colTop[i].count(), "The count should equal the top[i].count().");
            }
        }

        /// <summary>
        /// Computes the forward calculation copying the bottom Blbos with all of the top Blobs.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1+)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the output.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colBottom[0].count();
            long hBottom = colBottom[0].gpu_data;

            for (int i = 0; i < colTop.Count; i++)
            {
                m_cuda.copy(nCount, hBottom, colTop[i].mutable_gpu_data);
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1+), providing the error gradient
        /// with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            if (colTop.Count == 1)
            {
                m_cuda.copy(m_nCount, colTop[0].gpu_diff, colBottom[0].mutable_gpu_diff);
                return;
            }

            m_cuda.add(m_nCount, colTop[0].gpu_diff, colTop[1].gpu_diff, colBottom[0].mutable_gpu_diff);

            // Add remaining top blob diffs.
            for (int i = 2; i < colTop.Count; i++)
            {
                long hTopDiff = colTop[i].gpu_diff;
                long hBottomDiff = colBottom[0].mutable_gpu_diff;

                m_cuda.axpy(m_nCount, m_tOne, hTopDiff, hBottomDiff);
            }
        }
    }
}
