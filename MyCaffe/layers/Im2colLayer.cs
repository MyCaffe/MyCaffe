using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The Im2ColLayer is a helper layer for image operations that rearranges image regions into
    /// column vectors.  
    /// </summary>
    /// <remarks>
    /// Im2col operations are used by the ConvolutionLayer to perform convolution
    /// by matrix multiplication.
    /// 
    /// @see [Fast ConvNets Using Group-wise Brain Damage](https://arxiv.org/abs/1506.02515v2) by Vadim Lebedev, and Victor Lempitsky, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class Im2colLayer<T> : Layer<T>
    {
        /// <summary>
        /// The spatial dimensions of a filter kernel.
        /// </summary>
        Blob<T> m_blobKernelShape;
        /// <summary>
        /// The spatial dimensions of the stride.
        /// </summary>
        Blob<T> m_blobStride;
        /// <summary>
        /// The spatial dimensions of the padding.
        /// </summary>
        Blob<T> m_blobPad;
        /// <summary>
        /// The spatial dimensions of the dilation.
        /// </summary>
        Blob<T> m_blobDilation;
        int m_nNumSpatialAxes;
        int m_nBottomDim;
        int m_nTopDim;
        int m_nChannelAxis;
        int m_nNum;
        int m_nChannels;
        bool m_bForceNDIm2Col;

        /// <summary>
        /// The Im2col constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter of type IM2COL.</param>
        public Im2colLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.IM2COL;
            log.CHECK(p.type == LayerParameter.LayerType.IM2COL, "The layer type should be IM2COL.");

            m_blobKernelShape = new Blob<T>(cuda, log);
            m_blobStride = new Blob<T>(cuda, log);
            m_blobPad = new Blob<T>(cuda, log);
            m_blobDilation = new Blob<T>(cuda, log);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobKernelShape.Dispose();
            m_blobStride.Dispose();
            m_blobPad.Dispose();
            m_blobDilation.Dispose();
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: im2col.
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
            ConvolutionParameter p = m_param.convolution_param;

            m_bForceNDIm2Col = p.force_nd_im2col;
            int nInputNumDims = colBottom[0].shape().Count;
            m_nChannelAxis = colBottom[0].CanonicalAxisIndex(p.axis);
            int nFirstSpatialDim = m_nChannelAxis + 1;
            m_nNumSpatialAxes = nInputNumDims - nFirstSpatialDim;

            m_log.CHECK_GE(m_nNumSpatialAxes, 1, "The spatial axis count must be >= 1.");

            List<int> rgDimBlobShape = new List<int>() { m_nNumSpatialAxes };

            // Setup filter kernel dimensions (kernel_shape_).
            m_blobKernelShape.Reshape(rgDimBlobShape);

            T[] rgKernelShape = m_blobKernelShape.mutable_cpu_data;

            if (p.kernel_h.HasValue || p.kernel_w.HasValue)
            {
                m_log.CHECK_EQ(m_nNumSpatialAxes, 2, "kernel_h & kernel_w can only be used for 2D convolution.");
                m_log.CHECK_EQ(0, p.kernel_size.Count, "Either kernel_size or kernel_h/w should be specified; not both.");
                rgKernelShape[0] = (T)Convert.ChangeType(p.kernel_h.Value, typeof(T));
                rgKernelShape[1] = (T)Convert.ChangeType(p.kernel_w.Value, typeof(T));
            }
            else
            {
                int nNumKernelDims = p.kernel_size.Count;
                m_log.CHECK(nNumKernelDims == 1 || nNumKernelDims == m_nNumSpatialAxes, "kernel_size must be specified once, or once per spatial dimension (kernel_size specified " + nNumKernelDims.ToString() + " times; " + m_nNumSpatialAxes.ToString() + " spatial dims);");

                for (int i = 0; i < m_nNumSpatialAxes; i++)
                {
                    uint nKernel = p.kernel_size[(nNumKernelDims == 1) ? 0 : i];
                    rgKernelShape[i] = (T)Convert.ChangeType(nKernel, typeof(T));
                }
            }

            for (int i = 0; i < m_nNumSpatialAxes; i++)
            {
                int nVal = (int)Convert.ChangeType(rgKernelShape[i], typeof(int));
                m_log.CHECK_GT(nVal, 0, "Filter dimensions must be nonzero.");
            }

            m_blobKernelShape.mutable_cpu_data = rgKernelShape;


            // Setup stride dimensions (stride_).
            m_blobStride.Reshape(rgDimBlobShape);

            T[] rgStrideData = m_blobStride.mutable_cpu_data;

            if (p.stride_h.HasValue || p.stride_w.HasValue)
            {
                m_log.CHECK_EQ(m_nNumSpatialAxes, 2, "stride_h & stride_w can only be used for 2D convolution.");
                m_log.CHECK_EQ(0, p.stride.Count, "Either stride or stride_h/w should be specified; not both.");
                rgStrideData[0] = (T)Convert.ChangeType(p.stride_h.Value, typeof(T));
                rgStrideData[1] = (T)Convert.ChangeType(p.stride_w.Value, typeof(T));
            }
            else
            {
                int nNumStrideDims = p.stride.Count;
                m_log.CHECK(nNumStrideDims == 0 || nNumStrideDims == 1 || nNumStrideDims == m_nNumSpatialAxes, "stride must be specified once, or once per spatial dimension (stride specified " + nNumStrideDims.ToString() + " times; " + m_nNumSpatialAxes.ToString() + " spatial dims);");

                uint nDefaultStride = 1;
                for (int i = 0; i < m_nNumSpatialAxes; i++)
                {
                    uint nStride = (nNumStrideDims == 0) ? nDefaultStride :
                                                           p.stride[(nNumStrideDims == 1) ? 0 : i];

                    rgStrideData[i] = (T)Convert.ChangeType(nStride, typeof(T));
                }
            }

            m_blobStride.mutable_cpu_data = rgStrideData;


            // Setup pad dimensions (pad_).
            m_blobPad.Reshape(rgDimBlobShape);

            T[] rgPadData = m_blobPad.mutable_cpu_data;

            if (p.pad_h.HasValue || p.pad_w.HasValue)
            {
                m_log.CHECK_EQ(m_nNumSpatialAxes, 2, "pad_h & pad_w can only be used for 2D convolution.");
                m_log.CHECK_EQ(0, p.pad.Count, "Either pad or pad_h/w should be specified; not both.");
                rgPadData[0] = (T)Convert.ChangeType(p.pad_h.Value, typeof(T));
                rgPadData[1] = (T)Convert.ChangeType(p.pad_w.Value, typeof(T));
            }
            else
            {
                int nNumPadDims = p.pad.Count;
                m_log.CHECK(nNumPadDims == 0 || nNumPadDims == 1 || nNumPadDims == m_nNumSpatialAxes, "pad must be specified once, or once per spatial dimension (pad specified " + nNumPadDims.ToString() + " times; " + m_nNumSpatialAxes.ToString() + " spatial dims);");

                uint nDefaultPad = 0;
                for (int i = 0; i < m_nNumSpatialAxes; i++)
                {
                    uint nPad = (nNumPadDims == 0) ? nDefaultPad :
                                                     p.pad[(nNumPadDims == 1) ? 0 : i];

                    rgPadData[i] = (T)Convert.ChangeType(nPad, typeof(T));
                }
            }

            m_blobPad.mutable_cpu_data = rgPadData;


            // Setup dilation dimensions (dilation_).
            m_blobDilation.Reshape(rgDimBlobShape);

            T[] rgDilationData = m_blobDilation.mutable_cpu_data;

            int nNumDilationDims = p.dilation.Count;
            m_log.CHECK(nNumDilationDims == 0 || nNumDilationDims == 1 || nNumDilationDims == m_nNumSpatialAxes, "dilation must be specified once, or once per spatial dimension (dilation specified " + nNumDilationDims.ToString() + " times; " + m_nNumSpatialAxes.ToString() + " spatial dims);");

            uint nDefaultDilation = 1;
            for (int i = 0; i < m_nNumSpatialAxes; i++)
            {
                uint nPad = (nNumDilationDims == 0) ? nDefaultDilation :
                                                 p.dilation[(nNumDilationDims == 1) ? 0 : i];

                rgDilationData[i] = (T)Convert.ChangeType(nPad, typeof(T));
            }

            m_blobDilation.mutable_cpu_data = rgDilationData;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            List<int> rgTopShape = Utility.Clone<int>(colBottom[0].shape());
            T[] rgKernelShapeData = m_blobKernelShape.update_cpu_data();
            T[] rgStrideData = m_blobStride.update_cpu_data();
            T[] rgPadData = m_blobPad.update_cpu_data();
            T[] rgDilationData = m_blobDilation.update_cpu_data();

            for (int i = 0; i < m_nNumSpatialAxes; i++)
            {
                int nKernel = val_at(rgKernelShapeData, i);
                int nStride = val_at(rgStrideData, i);
                int nPad = val_at(rgPadData, i);
                int nDilation = val_at(rgDilationData, i);

                rgTopShape[m_nChannelAxis] *= nKernel;
                int nInputDim = colBottom[0].shape()[m_nChannelAxis + i + 1];
                int nKernelExtent = nDilation * (nKernel - 1) + 1;
                int nOutputDim = (nInputDim + 2 * nPad - nKernelExtent) / nStride + 1;
                rgTopShape[m_nChannelAxis + i + 1] = nOutputDim;
            }

            colTop[0].Reshape(rgTopShape);
            m_nNum = colBottom[0].count(0, m_nChannelAxis);
            m_nBottomDim = colBottom[0].count(m_nChannelAxis);
            m_nTopDim = colTop[0].count(m_nChannelAxis);
            m_nChannels = colBottom[0].shape(m_nChannelAxis);
        }

        /// <summary>
        /// Forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the input.</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (S \times C \times H \times W) @f$
        ///     the im2col output 
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            if (!m_bForceNDIm2Col && m_nNumSpatialAxes == 2)
            {
                Size szKernel = size_at(m_blobKernelShape);
                Size szStride = size_at(m_blobStride);
                Size szPad = size_at(m_blobPad);
                Size szDilation = size_at(m_blobDilation);

                for (int n = 0; n < m_nNum; n++)
                {
                    m_cuda.im2col(hBottomData,
                                  n * m_nBottomDim,
                                  m_nChannels,
                                  colBottom[0].shape(m_nChannelAxis + 1),
                                  colBottom[0].shape(m_nChannelAxis + 2),
                                  szKernel.Height, szKernel.Width,
                                  szPad.Height, szPad.Width,
                                  szStride.Height, szStride.Width,
                                  szDilation.Height, szDilation.Width,
                                  hTopData,
                                  n * m_nTopDim);
                }
            }
            else
            {
                int nNumKernels = m_nChannels * colTop[0].count(m_nChannelAxis + 1);
                long hKernelShape = m_blobKernelShape.gpu_data;
                long hStride = m_blobStride.gpu_data;
                long hPad = m_blobPad.gpu_data;
                long hDilation = m_blobDilation.gpu_data;

                for (int n = 0; n < m_nNum; n++)
                {
                    m_cuda.im2col_nd(hBottomData,
                                  n * m_nBottomDim,
                                  m_nNumSpatialAxes,
                                  nNumKernels,
                                  m_nChannelAxis,
                                  colBottom[0].gpu_shape,
                                  colTop[0].gpu_shape,
                                  hKernelShape,
                                  hPad,
                                  hStride,
                                  hDilation,
                                  hTopData,
                                  n * m_nTopDim);
                }
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the forwarded inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1), 
        /// providing the error gradient with respect to the outputs.</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">input Blob vecotor (length 1), into which the top error
        /// gradient is copied.</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            if (!m_bForceNDIm2Col && m_nNumSpatialAxes == 2)
            {
                Size szKernel = size_at(m_blobKernelShape);
                Size szStride = size_at(m_blobStride);
                Size szPad = size_at(m_blobPad);
                Size szDilation = size_at(m_blobDilation);

                for (int n = 0; n < m_nNumSpatialAxes; n++)
                {
                    m_cuda.col2im(hTopDiff,
                                  n * m_nTopDim,
                                  m_nChannels,
                                  colBottom[0].shape()[m_nChannelAxis + 1],
                                  colBottom[0].shape()[m_nChannelAxis + 2],
                                  szKernel.Height, szKernel.Width,
                                  szPad.Height, szPad.Width,
                                  szStride.Height, szStride.Width,
                                  szDilation.Height, szDilation.Width,
                                  hBottomDiff,
                                  n * m_nBottomDim);
                }
            }
            else
            {
                long hKernelShape = m_blobKernelShape.gpu_data;
                long hStride = m_blobStride.gpu_data;
                long hPad = m_blobPad.gpu_data;
                long hDilation = m_blobDilation.gpu_data;

                for (int n = 0; n < m_nNumSpatialAxes; n++)
                {
                    m_cuda.col2im_nd(hTopDiff,
                                  n * m_nTopDim,
                                  m_nNumSpatialAxes,
                                  m_nBottomDim,
                                  m_nChannelAxis,
                                  colBottom[0].gpu_shape,
                                  colTop[0].gpu_shape,
                                  hKernelShape,
                                  hPad,
                                  hStride,
                                  hDilation,
                                  hBottomDiff,
                                  n * m_nBottomDim);
                }
            }
        }
    }
}
