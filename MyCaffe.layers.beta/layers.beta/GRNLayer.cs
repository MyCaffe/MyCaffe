using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The GRNLayer performs an L2 normalization over the input data.
    /// </summary>
    /// <remarks>
    /// Adapted from original C++ code by Beanfrog at http://research.beenfrog.com/code/2015/04/11/global-response-normalization-L2-layer-in-caffe.html
    /// 
    /// @see [Layer Normalization](https://arxiv.org/abs/1607.06450) by Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class GRNLayer<T> : Layer<T>
    {
        Blob<T> m_blobSumMultiplier;
        Blob<T> m_blobSquare;
        Blob<T> m_blobNorm;
        Blob<T> m_blobTempDot;

        /// <summary>
        /// The GRNLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type GRN.
        /// </param>
        public GRNLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.GRN;
            m_blobSumMultiplier = new common.Blob<T>(cuda, log, false);
            m_blobSquare = new common.Blob<T>(cuda, log, false);
            m_blobNorm = new common.Blob<T>(cuda, log, false);
            m_blobTempDot = new common.Blob<T>(cuda, log, false);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
            dispose(ref m_blobSumMultiplier);
            dispose(ref m_blobSquare);
            dispose(ref m_blobNorm);
            dispose(ref m_blobTempDot);
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobSquare);
            col.Add(m_blobNorm);
            col.Add(m_blobTempDot);
            col.Add(m_blobSumMultiplier);
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: data
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: norm
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
            colTop[0].ReshapeLike(colBottom[0]);

            m_blobSumMultiplier.Reshape(1, colBottom[0].channels, 1, 1);
            m_blobSumMultiplier.SetData(1.0);
            m_blobSquare.ReshapeLike(colBottom[0]);
            m_blobNorm.Reshape(colBottom[0].num, 1, colBottom[0].height, colBottom[0].width);
            m_blobTempDot.Reshape(colBottom[0].num, 1, colBottom[0].height, colBottom[0].width);
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the outputs.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            long hSquareData = m_blobSquare.mutable_gpu_data;
            long hNormData = m_blobNorm.mutable_gpu_data;
            int nCount = colBottom[0].count();
            int nNum = colBottom[0].num;
            int nChannels = colBottom[0].channels;
            int nSpatialDim = colBottom[0].height * colBottom[0].width;

            m_cuda.copy(nCount, hBottomData, hTopData);
            m_cuda.copy(nCount, hBottomData, hSquareData);

            // square
            m_cuda.powx(nCount, hSquareData, 2.0, hSquareData);

            // sum across channel
            m_cuda.channel_sum(nNum * nSpatialDim, nNum, nChannels, nSpatialDim, hSquareData, hNormData);

            // square root
            m_cuda.powx(nNum * nSpatialDim, hNormData, 0.5, hNormData);

            // divide
            m_cuda.channel_div(nNum * nSpatialDim, nNum, nChannels, nSpatialDim, hNormData, hTopData, 2);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopDiff = colTop[0].gpu_diff;
            long hTopData = colTop[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            long hBottomData = colBottom[0].gpu_data;
            long hNormData = m_blobNorm.gpu_data;
            long hTempDotData = m_blobTempDot.mutable_gpu_data;
            long hTempData = m_blobSquare.mutable_gpu_data;
            int nNum = colTop[0].num;
            int nChannels = colTop[0].channels;
            int nSpatialDim = colTop[0].height * colTop[0].width;
            int nCount = colTop[0].count();

            m_cuda.copy(nCount, hTopDiff, hBottomDiff);
            m_cuda.copy(nCount, hBottomData, hTempData);

            // b_diff = t_diff / norm - dot(t_diff, t_data) / (norm)^2 * bottom_data
            // temp_dot_data = dot(t_diff, t_data)
            m_cuda.channel_dot(nNum * nSpatialDim, nNum, nChannels, nSpatialDim, hTopDiff, hTopData, hTempDotData);

            // temp_dot_data /= (norm)^2
            m_cuda.div(nNum * nSpatialDim, hTempDotData, hNormData, hTempDotData);
            m_cuda.div(nNum * nSpatialDim, hTempDotData, hNormData, hTempDotData);

            // bottom_diff = top_diff, bottom_diff /= norm
            m_cuda.channel_div(nNum * nSpatialDim, nNum, nChannels, nSpatialDim, hNormData, hBottomDiff, 2);

            // temp_data = bottom_data, temp_data *= temp_dot_data
            m_cuda.channel_mul(nNum * nSpatialDim, nNum, nChannels, nSpatialDim, hTempDotData, hTempData, 2);

            // bottom_diff += -temp_data
            m_cuda.axpy(nCount, -1.0, hTempData, hBottomDiff);
        }
    }
}
