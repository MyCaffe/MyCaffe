using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.beta;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The MergeLayer merges two bottom blobs with a specified copy pattern and outputs a single blob result.
    /// </summary>
    /// <remarks>
    /// This layer can be helpful when building encoder/decoder LSTM models.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class MergeLayer<T> : Layer<T>
    {
        int m_nCopyAxis = 0;
        int m_nOrderMajorAxis = 0;
        int m_nCopyCount = 1;
        int m_nSrcStartIdx1 = 0;
        int m_nDstStartIdx1 = 0;
        int m_nCopyDim1 = 1;
        int m_nSrcStartIdx2 = 0;
        int m_nDstStartIdx2 = 0;
        int m_nCopyDim2 = 1;
        int m_nSrcSpatialDim1 = 1;
        int m_nSrcSpatialDim2 = 1;
        int m_nSrcSpatialDimStartIdx1 = 0;
        int m_nDstSpatialDimStartIdx1 = 0;
        int m_nSrcSpatialDimStartIdx2 = 0;
        int m_nDstSpatialDimStartIdx2 = 0;
        int m_nSpatialDimCopyCount = -1;
        int m_nDstSpatialDim = 1;

        /// <summary>
        /// The MergeLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type GRN.
        /// </param>
        public MergeLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.MERGE;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();
                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input1 and input2
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: merge
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
            m_nCopyAxis = m_param.merge_param.copy_axis;
            m_nOrderMajorAxis = m_param.merge_param.order_major_axis;
            m_nCopyCount = m_param.merge_param.copy_count;
            m_nSrcStartIdx1 = Utility.CanonicalAxisIndex(m_param.merge_param.src_start_idx1, colBottom[0].shape()[m_nCopyAxis]);
            m_nDstStartIdx1 = Utility.CanonicalAxisIndex(m_param.merge_param.dst_start_idx1, colBottom[0].shape()[m_nCopyAxis]);
            m_nCopyDim1 = m_param.merge_param.copy_dim1;
            m_nSrcStartIdx2 = Utility.CanonicalAxisIndex(m_param.merge_param.src_start_idx2, colBottom[1].shape()[m_nCopyAxis]);
            m_nDstStartIdx2 = Utility.CanonicalAxisIndex(m_param.merge_param.dst_start_idx2, colBottom[1].shape()[m_nCopyAxis]);
            m_nCopyDim2 = m_param.merge_param.copy_dim2;

            m_nSrcSpatialDim1 = Utility.GetSpatialDim(colBottom[0].shape(), Math.Max(m_nCopyAxis, m_nOrderMajorAxis) + 2);
            m_nSrcSpatialDim2 = Utility.GetSpatialDim(colBottom[1].shape(), Math.Max(m_nCopyAxis, m_nOrderMajorAxis) + 2);
            m_nDstSpatialDim = m_param.merge_param.dst_spatialdim;
            if (m_nDstSpatialDim <= 0)
                m_nDstSpatialDim = 1;

            m_nSrcSpatialDimStartIdx1 = Utility.CanonicalAxisIndex(m_param.merge_param.src_spatialdim_start_idx1, m_nSrcSpatialDim1);
            m_nSrcSpatialDimStartIdx2 = Utility.CanonicalAxisIndex(m_param.merge_param.src_spatialdim_start_idx1, m_nSrcSpatialDim2);

            m_log.CHECK_GT(m_nCopyCount, 0, "The copy_count must be > 0!");

            m_log.CHECK_GT(m_nCopyDim1, 0, "The copy dim must be > 0!");
            m_log.CHECK_GT(m_nCopyDim2, 0, "The copy dim must be > 0!");
            m_log.CHECK_GE(m_nSrcStartIdx1, 0, "The start_idx1 must be >= 0!");
            m_log.CHECK_GE(m_nSrcStartIdx2, 0, "The start_idx1 must be >= 0!");
            m_log.CHECK_GE(m_nDstStartIdx1, 0, "The start_idx1 must be >= 0!");
            m_log.CHECK_GE(m_nDstStartIdx2, 0, "The start_idx1 must be >= 0!");

            m_log.CHECK_GE(m_nCopyAxis, 0, "The copy axis must be >= 0!");
            m_log.CHECK_LE(m_nCopyAxis, 1, "The copy axis must be <= 1!");
            m_log.CHECK_EQ(m_nOrderMajorAxis, 1, "Currently, only order_major_axis=1 is supported.");
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(colBottom.Count, 2, "Two inputs are expected!");
            List<int> rgTopShape = MergeParameter.Reshape(m_log, m_param.merge_param, colBottom[0].shape(), colBottom[1].shape());

            colTop[0].Reshape(rgTopShape);
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
            long hBtmData0 = colBottom[0].gpu_data;
            long hBtmData1 = colBottom[1].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            int nDstStep = colTop[0].shape()[m_nCopyAxis];
            int nSrcStep1 = colBottom[0].shape()[m_nCopyAxis];
            m_cuda.copy_sequence(colBottom[0].count(), hBtmData0, nSrcStep1, m_nSrcStartIdx1, m_nCopyCount, m_nCopyDim1, hTopData, nDstStep, m_nDstStartIdx1, m_nSrcSpatialDim1, m_nDstSpatialDim, m_nSrcSpatialDimStartIdx1, m_nDstSpatialDimStartIdx1, m_nSpatialDimCopyCount);

            int nSrcStep2 = colBottom[1].shape()[m_nCopyAxis];
            m_cuda.copy_sequence(colBottom[1].count(), hBtmData1, nSrcStep2, m_nSrcStartIdx2, m_nCopyCount, m_nCopyDim2, hTopData, nDstStep, m_nDstStartIdx2, m_nSrcSpatialDim2, m_nDstSpatialDim, m_nSrcSpatialDimStartIdx2, m_nDstSpatialDimStartIdx2, m_nSpatialDimCopyCount);
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
            colBottom[0].SetDiff(0);
            long hBtmDiff0 = colBottom[0].mutable_gpu_diff;
            long hBtmDiff1 = colBottom[1].mutable_gpu_diff;
            long hTopData = colTop[0].gpu_diff;

            int nSrcStep = colTop[0].shape()[m_nCopyAxis];
            int nDstStep1 = colBottom[0].shape()[m_nCopyAxis];
            m_cuda.copy_sequence(colTop[0].count(), hTopData, nSrcStep, m_nDstStartIdx1, m_nCopyCount, m_nCopyDim1, hBtmDiff0, nDstStep1, m_nSrcStartIdx1, m_nDstSpatialDim, m_nSrcSpatialDim1, m_nDstSpatialDimStartIdx1, m_nSrcSpatialDimStartIdx1, m_nSpatialDimCopyCount);

            int nDstStep2 = colBottom[1].shape()[m_nCopyAxis];
            m_cuda.copy_sequence(colTop[0].count(), hTopData, nSrcStep, m_nDstStartIdx2, m_nCopyCount, m_nCopyDim2, hBtmDiff1, nDstStep2, m_nSrcStartIdx2, m_nDstSpatialDim, m_nSrcSpatialDim2, m_nDstSpatialDimStartIdx2, m_nSrcSpatialDimStartIdx2, m_nSpatialDimCopyCount);
        }
    }
}
