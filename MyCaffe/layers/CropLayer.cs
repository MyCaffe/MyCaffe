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
    /// The CropLayer takes a Blob and crops it to the shape specified by the second input 
    /// Blob, across all dimensions after the specified axis.
    /// 
    /// This layer is initialized with the MyCaffe.param.CropParameter.
    /// </summary>
    /// <remarks>
    /// @see [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer, and Trevor Darrell, 2014.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class CropLayer<T> : Layer<T>
    {
        Blob<T> m_blobOffsets;
        Blob<T> m_blobSrcStrides;
        Blob<T> m_blobDstStrides;

        /// <summary>
        /// The CropLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type CROP and crop_param.
        /// </param>
        public CropLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.CONCAT;

            m_blobOffsets = new common.Blob<T>(m_cuda, m_log);
            m_blobOffsets.Name = m_param.name + "offsets";
            m_blobSrcStrides = new common.Blob<T>(m_cuda, m_log);
            m_blobSrcStrides.Name = m_param.name + "src strides";
            m_blobDstStrides = new common.Blob<T>(m_cuda, m_log);
            m_blobDstStrides.Name = m_param.name + "dst strides";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobOffsets.Dispose();
            m_blobSrcStrides.Dispose();
            m_blobDstStrides.Dispose();
            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobOffsets);
            col.Add(m_blobSrcStrides);
            col.Add(m_blobDstStrides);
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input, shape
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: crop
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
            m_log.CHECK_EQ(colBottom.Count, 2, "Wrong number of bottom blobs, expected 2.");
            int nInputDim = colBottom[0].num_axes;
            int nStartAxis = colBottom[0].CanonicalAxisIndex(m_param.crop_param.axis);
            m_log.CHECK_LT(nStartAxis, nInputDim, "The crop axis is bigger than the input dim.");

            if (m_param.crop_param.offset.Count > 1)
            {
                // The number of crop values specified must be equal to the number of
                // dimensions following axis.
                m_log.CHECK_EQ(nStartAxis + m_param.crop_param.offset.Count, nInputDim, "The number of offset values specified must be equal to the number of dimensions following axis.");
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nInputDim = colBottom[0].num_axes;
            int nStartAxis = colBottom[0].CanonicalAxisIndex(m_param.crop_param.axis);

            // Initialize offsets to 0 and the new shape to the current shape of the data.
            List<int> rgNewShape = Utility.Clone<int>(colBottom[0].shape());
            List<int> rgOffsetsShape = new List<int>() { nInputDim };

            m_blobOffsets.Reshape(rgOffsetsShape);
            float[] rgOffsetData = convertF(m_blobOffsets.mutable_cpu_data);

            // Determine crop offsets and the new shape post-crop.
            for (int i = 0; i < nInputDim; i++)
            {
                int nCropOffset = 0;
                int nNewSize = colBottom[0].shape(i);

                if (i >= nStartAxis)
                {
                    nNewSize = colBottom[1].shape(i);
                    if (m_param.crop_param.offset.Count == 1)
                    {
                        // If only one offset is given, all crops have the same offset.
                        nCropOffset = (int)m_param.crop_param.offset[0];
                    }
                    else if (m_param.crop_param.offset.Count > 1)
                    {
                        // For several offsets, the number of offsets must be equual to the
                        // number of dimensions to crop, that is dimensions after the axis.
                        nCropOffset = (int)m_param.crop_param.offset[i - nStartAxis]; 
                    }

                    // Check that the crop and offset are within the dimension's bounds.
                    m_log.CHECK_GE(colBottom[0].shape(i) - nCropOffset, colBottom[1].shape(i), "The crop for dimension " + i.ToString() + " is out-of-bounds with size " + colBottom[0].shape(i).ToString() + " and offset " + nCropOffset.ToString());
                }

                rgNewShape[i] = nNewSize;
                rgOffsetData[i] = nCropOffset;
            }

            m_blobOffsets.mutable_cpu_data = convert(rgOffsetData);

            colTop[0].Reshape(rgNewShape);
            
            // Compute strides
            m_blobSrcStrides.Reshape(rgOffsetsShape);
            m_blobDstStrides.Reshape(rgOffsetsShape);

            float[] rgSrcStrides = convertF(m_blobSrcStrides.mutable_cpu_data);
            float[] rgDstStrides = convertF(m_blobDstStrides.mutable_cpu_data);

            for (int i = 0; i < nInputDim; i++)
            {
                rgSrcStrides[i] = colBottom[0].count(i + 1, nInputDim);
                rgDstStrides[i] = colTop[0].count(i + 1, nInputDim);
            }

            m_blobSrcStrides.mutable_cpu_data = convert(rgSrcStrides);
            m_blobDstStrides.mutable_cpu_data = convert(rgDstStrides);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">bottom input blob (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the data 
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the size
        /// </param>
        /// <param name="colTop">top output blob (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the cropped output.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            List<int> rgIndices = Utility.Create<int>(colTop[0].num_axes, 0);
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colTop[0].count();

            m_cuda.crop_fwd(nCount, colBottom[0].num_axes, m_blobSrcStrides.gpu_data, m_blobDstStrides.gpu_data, m_blobOffsets.gpu_data, hBottomData, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the concatenation inputs.
        /// </summary>
        /// <param name="colTop">top output blob (length 1), providing the error gradient with
        /// respect to the outputs.</param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">input blob vector (length 2).</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
            {
                colBottom[0].SetDiff(0);

                long hTopDiff = colTop[0].gpu_diff;
                long hBottomDiff = colBottom[0].mutable_gpu_diff;
                int nCount = colTop[0].count();

                m_cuda.crop_bwd(nCount, colBottom[0].num_axes, m_blobSrcStrides.gpu_data, m_blobDstStrides.gpu_data, m_blobOffsets.gpu_data, hBottomDiff, hTopDiff);
            }
        }
    }
}
