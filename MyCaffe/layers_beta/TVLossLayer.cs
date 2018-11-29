using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.param;

namespace MyCaffe.layers_beta
{
    /// <summary>
    /// The TVLossLayer computes total variation loss as described by 'Mahendran' et al., and used in Neural Style.
    /// </summary>
    /// <remarks>
    /// @see [ftokarev/caffe-neural-style Github](https://github.com/ftokarev/caffe-neural-style) by ftokarev, 2017. 
    /// @see [Understanding Deep Image Representations by Inverting Them](https://arxiv.org/abs/1412.0035) by A. Mahendran and A. Vedaldi, CVPR, 2015.
    /// @see [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, 2015 
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TVLossLayer<T> : LossLayer<T>
    {
        Blob<T> m_blobTmp;
        Blob<T> m_blobYDiff;
        Blob<T> m_blobXDiff;
        Blob<T> m_blobMask;
        Blob<T> m_blobGradNorm;
        T m_tMinusOne;

        /// <summary>
        /// The TVLossLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter.</param>
        public TVLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.TV_LOSS;

            m_tMinusOne = (T)Convert.ChangeType(-1, typeof(T));
            m_blobTmp = new Blob<T>(cuda, log, false);
            m_blobTmp.Name = p.name + " tmp";
            m_blobYDiff = new Blob<T>(cuda, log, false);
            m_blobYDiff.Name = p.name + " y diff";
            m_blobXDiff = new Blob<T>(cuda, log, false);
            m_blobXDiff.Name = p.name + " x diff";
            m_blobMask = new Blob<T>(cuda, log, false);
            m_blobMask.Name = p.name + " mask";
            m_blobGradNorm = new Blob<T>(cuda, log, false);
            m_blobGradNorm.Name = p.name + " grad norm";
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

        private void create_mask(int nCount, int nH, int nW, Blob<T> mask)
        {
            float[] rgMask = convertF(mask.mutable_cpu_data);
            int nSize = nH * nW;

            for (int i = 0; i < nCount; i++)
            {
                int nUnitPos = i % nSize;

                if (nUnitPos % nW == nW - 1 || nUnitPos / nW == nH - 1)
                    rgMask[i] = 0;
                else
                    rgMask[i] = 1;
            }

            mask.mutable_cpu_data = convert(rgMask);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_blobTmp.ReshapeLike(colBottom[0]);
            m_blobXDiff.ReshapeLike(colBottom[0]);
            m_blobYDiff.ReshapeLike(colBottom[0]);
            m_blobGradNorm.ReshapeLike(colBottom[0]);

            m_blobMask.ReshapeLike(colBottom[0]);
            create_mask(colBottom[0].count(), colBottom[0].shape(-2), colBottom[0].shape(-1), m_blobMask);

            // Loss layers output a scalar; 0 axes.
            List<int> rgLossShape = new List<int>();
            colTop[0].Reshape(rgLossShape);
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
            int nW = colBottom[0].shape(-1);
            int nCount = colBottom[0].count();
            long hBottomData = colBottom[0].gpu_data;

            m_cuda.sub(nCount - 1, hBottomData, hBottomData, m_blobXDiff.mutable_gpu_data, 0, 1, 0);
            m_cuda.mul(nCount, m_blobXDiff.gpu_data, m_blobMask.gpu_data, m_blobXDiff.mutable_gpu_data);

            m_cuda.sub(nCount - nW, hBottomData, hBottomData, m_blobYDiff.mutable_gpu_data, 0, nW, 0);
            m_cuda.mul(nCount, m_blobYDiff.gpu_data, m_blobMask.gpu_data, m_blobYDiff.mutable_gpu_data);

            m_cuda.mul(nCount, m_blobXDiff.gpu_data, m_blobXDiff.gpu_data, m_blobGradNorm.mutable_gpu_data); // X_diff^2
            m_cuda.mul(nCount, m_blobYDiff.gpu_data, m_blobYDiff.gpu_data, m_blobTmp.mutable_gpu_data); // Y_diff^2

            m_cuda.add(nCount, m_blobTmp.gpu_data, m_blobGradNorm.gpu_data, m_blobGradNorm.mutable_gpu_data); // X_diff^2 + Y_diff^2
            m_cuda.powx(nCount, m_blobGradNorm.gpu_data, m_param.tv_loss_param.beta / 2, m_blobTmp.mutable_gpu_data); // (X_diff^2 + Y_diff^2)^(beta/2)

            double dfAsum = convertD(m_blobTmp.asum_data());
            colTop[0].SetData(dfAsum, 0);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the absolute value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$</param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            int nW = colBottom[0].shape(-1);
            int nCount = colBottom[0].count();
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            m_cuda.powx(nCount, m_blobGradNorm.gpu_data, m_param.tv_loss_param.beta / 2 - 1, m_blobGradNorm.mutable_gpu_data);
            m_cuda.scal(nCount, m_param.tv_loss_param.beta / 2, m_blobGradNorm.mutable_gpu_data);

            m_cuda.mul(nCount, m_blobXDiff.gpu_data, m_blobGradNorm.gpu_data, m_blobXDiff.mutable_gpu_data);
            m_cuda.scal(nCount, 2.0, m_blobXDiff.mutable_gpu_data); // dX_diff

            m_cuda.mul(nCount, m_blobYDiff.gpu_data, m_blobGradNorm.gpu_data, m_blobYDiff.mutable_gpu_data);
            m_cuda.scal(nCount, 2.0, m_blobYDiff.mutable_gpu_data); // dY_diff

            m_cuda.axpy(nCount, 1.0, m_blobXDiff.gpu_data, hBottomDiff);
            m_cuda.axpy(nCount, 1.0, m_blobYDiff.gpu_data, hBottomDiff);
            m_cuda.axpy(nCount - 1, m_tMinusOne, m_blobXDiff.gpu_data, hBottomDiff, 0, 1);
            m_cuda.axpy(nCount - nW, m_tMinusOne, m_blobYDiff.gpu_data, hBottomDiff, 0, nW);

            double dfTopDiff = convertD(colTop[0].GetDiff(0));
            m_cuda.scal(nCount, dfTopDiff, hBottomDiff);
        }
    }
}
