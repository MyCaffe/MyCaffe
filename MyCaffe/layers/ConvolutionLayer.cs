using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using System.Drawing;

namespace MyCaffe.layers
{
    /// <summary>
    /// The ConvolutionLayer convolves the input image with a bank of learned filters, and (optionally) adds biases.
    /// This layer is initialized with the MyCaffe.param.ConvolutionParameter.
    /// </summary>
    /// <remarks>
    /// Caffe convolves by reduction to matrix multiplication.  This achieves
    /// high-throughput and generality of input and filter dimensions but comes at
    /// the cost of memory for matrices.  This makes use of efficiency in BLAS.
    /// 
    /// The input is 'im2col' transformed to a channel 'K x H x W' data matrix
    /// for multiplication with the 'N x K x H x W' filter matrix to yield a
    /// 'N x H x W' output matrix that is then 'col2im' restored.  K is the
    /// input channel * kernel height * kernel width dimension of the unrolled
    /// inputs so that the im2col matrix has a column for each input region to
    /// be filtered.  col2im restores the output spatial structure by unrulling up
    /// the output channel 'N' columns of the output matrix.
    /// 
    /// Note: cuDNN accelerates convolution through forward kernels for filtering and bias
    /// plus backward kernels for gradient w.r.t the filters, biases, and
    /// inputs.  Caffe + cuDNN further speeds up the computation through forward
    /// parallelism across groups and backward parallelism across gradients.
    /// 
    /// The cuDNN engine does not have memory overhead for the matrix buffers.  For many
    /// input and filter regimes the cuDNN engien is faster than the CAFFE engine,
    /// but for fully-convolutional models and large inputs the CAFFE engine can be
    /// faster as long as it fits in memory.
    /// 
    /// @see [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) by Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner, 1998.
    /// @see [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) by Vincent Dumoulin, and Francesco Visin, 2016.
    /// @see [Joint Semantic and Motion Segmentation for dynamic scenes using Deep Convolutional Networks](https://arxiv.org/abs/1704.08331) by Nazrul Haque, N. Dinesh Reddy, and K. Madhava Krishna, 2017. 
    /// @see [A New Convolutional Network-in-Network Structure and Its Applications in Skin Detection, Semantic Segmentation, and Artifact Reduction](https://arxiv.org/abs/1701.06190v1) by Yoonsik Kim, Insung Hwang, and Nam Ik Cho, 2017.
    /// @see [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer, and Trevor Darrell, 2014.
    /// @see [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) by Fisher Yu, and Vladlen Koltun, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ConvolutionLayer<T> : BaseConvolutionLayer<T>
    {
        const int CUDNN_STREAMS_PER_GROUP = 3;

        long[] m_rghCudnn = null;
        long[] m_rghStream = null;
        
        // algorithms for forward and backward convolutions
        CONV_FWD_ALGO[] m_rgfwdAlgo = null;
        CONV_BWD_FILTER_ALGO[] m_rgbwdFilterAlgo = null;
        CONV_BWD_DATA_ALGO[] m_rgbwdDataAlgo = null;

        List<long> m_rghBottomDesc = new List<long>();
        List<long> m_rghTopDesc = new List<long>();
        long m_hBiasDesc = 0;
        long m_hFilterDesc = 0;
        List<long> m_rghConvDesc = new List<long>();
        int m_nBottomOffset = 0;
        int m_nTopOffset = 0;
        int m_nBiasOffset = 0;

        ulong[] m_rglWorkspaceFwdSizes = null;
        ulong[] m_rglWorkspaceBwdFilterSizes = null;
        ulong[] m_rglWorkspaceBwdDataSizes = null;
        ulong[] m_rglWorkspaceFwdOffsets = null; // offsets into workspace fwd data.
        ulong[] m_rglWorkspaceBwdFilterOffsets = null; // offsets into workspace bwd filter data.
        ulong[] m_rglWorkspaceBwdDataOffsets = null; // offsets into workspace bwd data.
        bool m_bUseTensorCores = false;

        /// <summary>
        /// The ConvolutionLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Provides ConvolutionParameter convolution_param with ConvolutionLayer options:
        ///  - num_output. The number of filters.
        ///  
        ///  - kernel_size / kernel_h / kernel_w.  The filter dimensions, given by
        ///  kernel_size for square filters or kernel_h and kernel-w for rectangular 
        ///  filters.
        ///  
        ///  - stride / stride_h / stride_w. (\b optional, default 1).  The filter
        ///  stride, given by stride_size for equal dimensions of stride_h and stride_w
        ///  for different strides.  By default the convolution is dense with stride 1.
        ///  
        ///  - pad / pad_h / pad_w. (\b optional, default 0). The zero-padding for
        ///  convolutions, given by pad for equal dimensions or pad_h and pad_w for
        ///  different padding.  Input padding is computed implicitly instead of 
        ///  actual padding.
        ///  
        ///  - dilation (\b optional, default 1).  The filter
        ///  dilation, given by dilation_size for equal dimensions for different
        ///  dilation.  By default the convolution has dilation 1.
        ///  
        ///  - group (\b optional, default 1).  The number of filter groups.  Group
        ///  convolution is a method for reducing parameterization by selectively
        ///  connecting input and output channels.  The input and output channel dimensions
        ///  must be divisible by the number of groups.  For group = 1, the 
        ///  convolutionjf ilters input and output channels are separeated s.t. each
        ///  group takes 1/group of the input channels and makes 1/group of the
        ///  output channels.  Concretely 4 input channels, 8 output channels, and
        ///  2 groups separate input chanels 1-2 and output channels 1-4 into the
        ///  first group and input channels 3-4 and output channels 5-8 into the xecond
        ///  group.
        ///  
        ///  - bias_term (\b optional, default, true). Whether to have a bias.
        ///  
        ///  - engine: convolution has Engine.CAFFE (matrix multiplication) and Engine.CUDNN (library
        ///  kernels + stream parallelism) engines.
        /// </param>
        public ConvolutionLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.CONVOLUTION;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            for (int i = 0; i < m_rghBottomDesc.Count; i++)
            {
                m_cuda.FreeTensorDesc(m_rghBottomDesc[i]);
                m_cuda.FreeTensorDesc(m_rghTopDesc[i]);
                m_cuda.FreeConvolutionDesc(m_rghConvDesc[i]);
            }

            m_rghBottomDesc.Clear();
            m_rghTopDesc.Clear();
            m_rghConvDesc.Clear();

            if (m_hBiasDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hBiasDesc);
                m_hBiasDesc = 0;
            }

            if (m_hFilterDesc != 0)
            {
                m_cuda.FreeFilterDesc(m_hFilterDesc);
                m_hFilterDesc = 0;
            }

            for (int g = 0; g < (m_nGroup * CUDNN_STREAMS_PER_GROUP); g++)
            {
                if (m_rghStream != null && m_rghStream[g] != 0)
                    m_cuda.FreeStream(m_rghStream[g]);

                if (m_rghCudnn != null && m_rghCudnn[g] != 0)
                    m_cuda.FreeCuDNN(m_rghCudnn[g]);
            }

            m_rghStream = null;
            m_rghCudnn = null;

            base.dispose();
        }

        /// <summary>
        /// Setup the layer for use with both Engine.CAFFE and Engine.CUDNN modes.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            if (!m_param.convolution_param.useCudnn(m_nNumSpatialAxes))
                return;

            // Initialize CUDA streams and cuDNN.
            m_rghStream = new long[m_nGroup * CUDNN_STREAMS_PER_GROUP];
            m_rghCudnn = new long[m_nGroup * CUDNN_STREAMS_PER_GROUP];

            // Initialize algorithm arrays.
            m_rgfwdAlgo = new CONV_FWD_ALGO[colBottom.Count];
            m_rgbwdFilterAlgo = new CONV_BWD_FILTER_ALGO[colBottom.Count];
            m_rgbwdDataAlgo = new CONV_BWD_DATA_ALGO[colBottom.Count];

            // Initialize the size arrays.
            m_rglWorkspaceFwdSizes = new ulong[colBottom.Count];
            m_rglWorkspaceBwdFilterSizes = new ulong[colBottom.Count];
            m_rglWorkspaceBwdDataSizes = new ulong[colBottom.Count];
            m_rglWorkspaceFwdOffsets = new ulong[m_nGroup * CUDNN_STREAMS_PER_GROUP];
            m_rglWorkspaceBwdFilterOffsets = new ulong[m_nGroup * CUDNN_STREAMS_PER_GROUP];
            m_rglWorkspaceBwdDataOffsets = new ulong[m_nGroup * CUDNN_STREAMS_PER_GROUP];

            for (int i = 0; i < colBottom.Count; i++)
            {
                // initialize all to default algorithms.
                m_rgfwdAlgo[i] = (CONV_FWD_ALGO)0;
                m_rgbwdFilterAlgo[i] = (CONV_BWD_FILTER_ALGO)0;
                m_rgbwdDataAlgo[i] = (CONV_BWD_DATA_ALGO)0;

                // default algorithms don't require workspace.
                m_rglWorkspaceFwdSizes[i] = 0;
                m_rglWorkspaceBwdFilterSizes[i] = 0;
                m_rglWorkspaceBwdDataSizes[i] = 0;
            }

            for (int g = 0; g < m_nGroup * CUDNN_STREAMS_PER_GROUP; g++)
            {
                m_rghStream[g] = m_cuda.CreateStream(false, g);
                m_rghCudnn[g] = m_cuda.CreateCuDNN(m_rghStream[g]);
                m_rglWorkspaceFwdOffsets[g] = 0;
                m_rglWorkspaceBwdFilterOffsets[g] = 0;
                m_rglWorkspaceBwdDataOffsets[g] = 0;
            }

            m_bUseTensorCores = m_param.convolution_param.cudnn_enable_tensor_cores;
            if (typeof(T) == typeof(double))
            {
                m_log.WriteLine("WARNING: Tensor cores are only supported with the 'float' base type.  Tensor core use will be disabled for the 'double' base type.");
                m_bUseTensorCores = false;
            }

            // Set the indexing parameters.
            m_nBiasOffset = m_nNumOutput / m_nGroup;

            // Create filter descriptor.
            Size szKernel = size_at(m_blobKernelShape);
            m_hFilterDesc = m_cuda.CreateFilterDesc();
            m_cuda.SetFilterDesc(m_hFilterDesc, m_nNumOutput / m_nGroup, m_nChannels / m_nGroup, szKernel.Height, szKernel.Width, m_bUseHalfSize); 

            // Create tensor descriptor(s) for data and corresponding convolution(s).
            for (int i = 0; i < colBottom.Count; i++)
            {
                m_rghBottomDesc.Add(m_cuda.CreateTensorDesc());
                m_rghTopDesc.Add(m_cuda.CreateTensorDesc());
                m_rghConvDesc.Add(m_cuda.CreateConvolutionDesc());
            }

            // Tensor descriptor for bias.
            if (m_bBiasTerm)
                m_hBiasDesc = m_cuda.CreateTensorDesc();
        }

        /// <summary>
        /// Tests the shapes of both the bottom and top blobs and if they are the same as the previous sizing, returns <i>false</i> indicating that no reshape is needed.
        /// </summary>
        /// <param name="colBottom">Specifies the bottom blobs.</param>
        /// <param name="colTop">Specifies the top blobs.</param>
        /// <param name="bReset">Specifies to reset the test (set to <i>false</i> when using in second derivative classes, e.g. set to true in BaseConvolutionLayer, and false in ConvolutionLayer).</param>
        /// <returns>If a reshape is needed, returns <i>true</i> otherwise returns <i>fasle</i>.</returns>
        protected override bool reshapeNeeded(BlobCollection<T> colBottom, BlobCollection<T> colTop, bool bReset = true)
        {
            // Memory optimizations require reshaping now on each pass.
            if (!bReset)
                return m_bReshapeOnForwardNeeded;

            if (!compareShapes(colBottom, colTop))
            {
                m_bReshapeOnForwardNeeded = true;
                return true;
            }
            else
            {
                m_bReshapeOnForwardNeeded = false;
                return false;
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);
            if (!reshapeNeeded(colBottom, colTop, false))
                return;

            setShapes(colBottom, colTop);

            if (!m_param.convolution_param.useCudnn(m_nNumSpatialAxes))
                return;

            m_log.CHECK_EQ(2, m_nNumSpatialAxes, "cuDNN Convolution input must have 2 spatial axes (e.g., height and width).  Use 'engine: CAFFE' for general ND convolution.");

            m_nBottomOffset = m_nBottomDim / m_nGroup;
            m_nTopOffset = m_nTopDim / m_nGroup;

            int nHeight = colBottom[0].shape(m_nChannelAxis + 1);
            int nWidth = colBottom[0].shape(m_nChannelAxis + 2);
            int nHeightOut = colTop[0].shape(m_nChannelAxis + 1);
            int nWidthOut = colTop[0].shape(m_nChannelAxis + 2);

            Size szPad = size_at(m_blobPad);
            Size szStride = size_at(m_blobStride);
            Size szDilation = size_at(m_blobDilation);

            ulong lWorkspaceLimitBytes = getWorkspaceLimitInBytes(m_bUseTensorCores);

            for (int i = 0; i < colBottom.Count; i++)
            {
                m_cuda.SetTensorDesc(m_rghBottomDesc[i], m_nNum, m_nChannels / m_nGroup, nHeight, nWidth, m_nChannels * nHeight * nWidth, nHeight * nWidth, nWidth, 1, m_bUseHalfSize);
                m_cuda.SetTensorDesc(m_rghTopDesc[i], m_nNum, m_nNumOutput / m_nGroup, nHeightOut, nWidthOut, m_nNumOutput * m_nOutSpatialDim, m_nOutSpatialDim, nWidthOut, 1, m_bUseHalfSize);
                m_cuda.SetConvolutionDesc(m_rghConvDesc[i], szPad.Height, szPad.Width, szStride.Height, szStride.Width, szDilation.Height, szDilation.Width, m_bUseTensorCores, m_bUseHalfSize);

                // Get the algorithms and workspace sizes needed.
                CONV_FWD_ALGO algoFwd = (CONV_FWD_ALGO)0;
                CONV_BWD_FILTER_ALGO algoBwdFilter = (CONV_BWD_FILTER_ALGO)0;
                CONV_BWD_DATA_ALGO algoBwdData = (CONV_BWD_DATA_ALGO)0;
                ulong lWsSizeFwd = 0;
                ulong lWsSizeBwdFilter = 0;
                ulong lWsSizeBwdData = 0;

                m_cuda.GetConvolutionInfo(m_rghCudnn[0], m_rghBottomDesc[i], m_hFilterDesc, m_rghConvDesc[i], m_rghTopDesc[i], lWorkspaceLimitBytes, m_bUseTensorCores, out algoFwd, out lWsSizeFwd, out algoBwdFilter, out lWsSizeBwdFilter, out algoBwdData, out lWsSizeBwdData);
                m_rgfwdAlgo[i] = algoFwd;
                m_rglWorkspaceFwdSizes[i] = lWsSizeFwd;
                m_rgbwdFilterAlgo[i] = algoBwdFilter;
                m_rglWorkspaceBwdFilterSizes[i] = lWsSizeBwdFilter;
                m_rgbwdDataAlgo[i] = algoBwdData;
                m_rglWorkspaceBwdDataSizes[i] = lWsSizeBwdData;
            }

            // reduce over all workspace sizes to get a maximum to allocate / reallocate
            ulong lTotalWsFwd = 0;
            ulong lTotalWsBwdFilter = 0;
            ulong lTotalWsBwdData = 0;

            for (int i = 0; i < colBottom.Count; i++)
            {
                lTotalWsFwd = Math.Max(lTotalWsFwd, m_rglWorkspaceFwdSizes[i]);
                lTotalWsBwdFilter = Math.Max(lTotalWsBwdFilter, m_rglWorkspaceBwdFilterSizes[i]);
                lTotalWsBwdData = Math.Max(lTotalWsBwdData, m_rglWorkspaceBwdDataSizes[i]);
            }

            // Get max over all oeprations.
            ulong lMaxWorkspace = Math.Max(lTotalWsFwd, Math.Max(lTotalWsBwdFilter, lTotalWsBwdData));

            // Ensure all groups have enough workspace.
            ulong lTotalMaxWorkspace = (ulong)lMaxWorkspace * (ulong)m_nGroup * (ulong)CUDNN_STREAMS_PER_GROUP;

            // Initialize the workspace data.
            WorkspaceArgs wsArgs = getWorkspace();

            // This is the total amount of storage needed over all groups + streams.
            setWorkspace(lTotalMaxWorkspace);

            // if we succedd in the allocation, set the offsets for the workspaces.
            for (int g = 0; g < (m_nGroup * CUDNN_STREAMS_PER_GROUP); g++)
            {
                m_rglWorkspaceFwdOffsets[g] = (ulong)g * lTotalWsFwd;
                m_rglWorkspaceBwdFilterOffsets[g] = (ulong)g * lTotalWsBwdFilter;
                m_rglWorkspaceBwdDataOffsets[g] = (ulong)g * lTotalWsBwdData;
            }

            // Tensor descriptor for bias.
            if (m_bBiasTerm)
                m_cuda.SetTensorDesc(m_hBiasDesc, 1, m_nNumOutput / m_nGroup, 1, 1, m_bUseHalfSize);
        }

        /// <summary>
        /// Returns <i>false</i>, for we want convolution, not deconvolution.
        /// </summary>
        /// <returns><i>false</i> is returned.</returns>
        protected override bool reverse_dimensions()
        {
            return false;
        }

        /// <summary>
        /// Computes the output shape used by the BaseConvolutionLayer.
        /// </summary>
        protected override void compute_output_shape()
        {
            T[] rgKernelShapeData = m_blobKernelShape.cpu_data;
            T[] rgStrideData = m_blobStride.cpu_data;
            T[] rgPadData = m_blobPad.cpu_data;
            T[] rgDilationData = m_blobDilation.cpu_data;

            m_rgOutputShape.Clear();

            for (int i = 0; i < m_nNumSpatialAxes; i++)
            {
                int nKernel = val_at(rgKernelShapeData, i);
                int nStride = val_at(rgStrideData, i);
                int nPad = val_at(rgPadData, i);
                int nDilation = val_at(rgDilationData, i);

                // i + 1 to skip channel axis.
                int nInputDim = input_shape(i + 1);
                int nKernelExtent = nDilation * (nKernel - 1) + 1;
                int nOutputDim = (nInputDim + 2 * nPad - nKernelExtent) / nStride + 1;

                if (nOutputDim == 0)
                    nOutputDim = 1;

                m_rgOutputShape.Add(nOutputDim);
            }
        }

        /// <summary>
        /// Run the Forward computation using either the Engine.CAFFE or Engine.CUDNN mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!m_param.convolution_param.useCudnn(m_nNumSpatialAxes))
                forward_cuda(colBottom, colTop);
            else
                forward_cudnn(colBottom, colTop);
        }

        /// <summary>
        /// Run the Backward computation using either the Engine.CAFFE or Engine.CUDNN mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (length 1).</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!m_param.convolution_param.useCudnn(m_nNumSpatialAxes))
                backward_cuda(colTop, rgbPropagateDown, colBottom);
            else
                backward_cudnn(colTop, rgbPropagateDown, colBottom);
        }

        /// <summary>
        /// Run the Forward computation using the Engine.CAFFE mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected void forward_cuda(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hWeight = m_colBlobs[0].gpu_data;

            for (int i = 0; i < colBottom.Count; i++)
            {
                long hBottomData = colBottom[i].gpu_data;
                long hTopData = colTop[i].mutable_gpu_data;

                for (int n = 0; n < m_nNum; n++)
                {
                    forward_gemm(hBottomData, n * m_nBottomDim, hWeight, hTopData, n * m_nTopDim);

                    if (m_bBiasTerm)
                        forward_bias(hTopData, n * m_nTopDim, m_colBlobs[1].gpu_data);
                }
            }
        }

        /// <summary>
        /// Run the Backward computation using the Engine.CAFFE mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (length 1).</param>
        protected void backward_cuda(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hWeight = m_colBlobs[0].gpu_data;
            long hWeightDiff = m_colBlobs[0].mutable_gpu_diff;

            for (int i = 0; i < colTop.Count; i++)
            {
                long hTopDiff = colTop[i].gpu_diff;

                // Bias gradient, if necessary.
                if (m_bBiasTerm && m_rgbParamPropagateDown[1])
                {
                    long hBiasDiff = m_colBlobs[1].mutable_gpu_diff;

                    for (int n = 0; n < m_nNum; n++)
                    {
                        backward_bias(hBiasDiff, hTopDiff, n * m_nTopDim);
                    }
                }

                if (m_rgbParamPropagateDown[0] || rgbPropagateDown[i])
                {
                    long hBottomData = colBottom[i].gpu_data;
                    long hBottomDiff = colBottom[i].mutable_gpu_diff;

                    for (int n = 0; n < m_nNum; n++)
                    {
                        // gradient w.r.t. weight.  Note that we will accumulate diffs.
                        if (m_rgbParamPropagateDown[0])
                            weight_gemm(hBottomData, n * m_nBottomDim, hTopDiff, n * m_nTopDim, hWeightDiff);

                        // gradient w.r.t. bottom data, if necessary.
                        if (rgbPropagateDown[i])
                            backward_gemm(hTopDiff, n * m_nTopDim, hWeight, hBottomDiff, n * m_nBottomDim);
                    }
                }
            }
        }

        /// <summary>
        /// Run the Forward computation using the Engine CUDNN mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected void forward_cudnn(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hWeight = m_colBlobs[0].gpu_data;
            WorkspaceArgs wsArgs = getWorkspace();

            for (int i = 0; i < colBottom.Count; i++)
            {
                long hBottomData = colBottom[i].gpu_data;
                long hTopData = colTop[i].mutable_gpu_data;

                // Forward through cuDNN in parallel over groups.
                for (int g = 0; g < m_nGroup; g++)
                {
                    // Filters.
                    m_cuda.ConvolutionForward(m_rghCudnn[g],
                                              m_tOne,
                                              m_rghBottomDesc[i],
                                              hBottomData, m_nBottomOffset * g,
                                              m_hFilterDesc,
                                              hWeight, m_nWeightOffset * g,
                                              m_rghConvDesc[i],
                                              m_rgfwdAlgo[i],
                                              wsArgs.WorkspaceData, (int)m_rglWorkspaceFwdOffsets[g], m_rglWorkspaceFwdSizes[i],
                                              m_tZero,
                                              m_rghTopDesc[i],
                                              hTopData, m_nTopOffset * g,
                                              false);
                }

                // Synchronize the work across groups, each of which went into its own stream.
                for (int g = 0; g < m_nGroup; g++)
                {
                    m_cuda.SynchronizeStream(m_rghStream[g]);
                }

                // Bias.
                if (m_bBiasTerm)
                {
                    for (int g=0; g<m_nGroup; g++)
                    { 
                        long hBiasData = m_colBlobs[1].gpu_data;

                        m_cuda.AddTensor(m_rghCudnn[g],
                                              m_tOne,
                                              m_hBiasDesc,
                                              hBiasData, m_nBiasOffset * g,
                                              m_tOne,
                                              m_rghTopDesc[i],
                                              hTopData, m_nTopOffset * g);
                    }

                    // Synchronize the work across groups, each of which went into its own stream.
                    for (int g = 0; g < m_nGroup; g++)
                    {
                        m_cuda.SynchronizeStream(m_rghStream[g]);
                    }
                }
            }
        }

        /// <summary>
        /// Run the Backward computation using the Engine CUDNN mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (length 1).</param>
        protected void backward_cudnn(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            WorkspaceArgs wsArgs = getWorkspace();

            // Gradient w.r.t. bias.
            if (m_bBiasTerm && m_rgbParamPropagateDown[1])
            {
                long hBiasDiff = m_colBlobs[1].mutable_gpu_diff;

                for (int i = 0; i < colTop.Count; i++)
                {
                    long hTopDiff = colTop[i].mutable_gpu_diff;

                    // Backward through cuDNN in parallel over groups and gradients.
                    for (int g = 0; g < m_nGroup; g++)
                    {
                        m_cuda.ConvolutionBackwardBias(m_rghCudnn[0 * m_nGroup + g],
                                m_tOne, m_rghTopDesc[i], hTopDiff, m_nTopOffset * g,
                                m_tOne, m_hBiasDesc, hBiasDiff, m_nBiasOffset * g,
                                false);
                    }
                    // Synchronize the work across groups, each of which went into its own stream.
                    for (int g = 0; g < m_nGroup; g++)
                    {
                        m_cuda.SynchronizeStream(m_rghStream[0 * m_nGroup + g]);
                    }
                }
            }

            // Gradient w.r.t weights.
            if (m_rgbParamPropagateDown[0])
            {
                long hWeightDiff = m_colBlobs[0].mutable_gpu_diff;

                for (int i = 0; i < colTop.Count; i++)
                {
                    long hTopDiff = colTop[i].mutable_gpu_diff;
                    long hBottomData = colBottom[i].gpu_data;

                    // Backward through cuDNN in parallel over groups and gradients.
                    for (int g = 0; g < m_nGroup; g++)
                    {
                        m_cuda.ConvolutionBackwardFilter(m_rghCudnn[1 * m_nGroup + g],
                                                       m_tOne,
                                                       m_rghBottomDesc[i], hBottomData, m_nBottomOffset * g,
                                                       m_rghTopDesc[i], hTopDiff, m_nTopOffset * g,
                                                       m_rghConvDesc[i], 
                                                       m_rgbwdFilterAlgo[i], 
                                                       wsArgs.WorkspaceData, (int)m_rglWorkspaceBwdFilterOffsets[1 * m_nGroup + g],
                                                       m_rglWorkspaceBwdFilterSizes[i],
                                                       m_tOne,
                                                       m_hFilterDesc, hWeightDiff, m_nWeightOffset * g,
                                                       false);
                    }
                    // Synchronize the work across groups, each of which went into its own stream.
                    for (int g = 0; g < m_nGroup; g++)
                    {
                        m_cuda.SynchronizeStream(m_rghStream[1 * m_nGroup + g]);
                    }
                }
            }

            // Gradient w.r.t. bottom data.
            long hWeight = m_colBlobs[0].gpu_data;

            for (int i=0; i<colTop.Count; i++)
            { 
                if (rgbPropagateDown[i])
                {
                    long hTopDiff = colTop[i].mutable_gpu_diff;
                    long hBottomDiff = colBottom[i].mutable_gpu_diff;

                    // Backward through cuDNN in parallel over groups and gradients.
                    for (int g = 0; g < m_nGroup; g++)
                    {
                        m_cuda.ConvolutionBackwardData(m_rghCudnn[2 * m_nGroup + g],
                                                      m_tOne,
                                                      m_hFilterDesc, hWeight, m_nWeightOffset * g,
                                                      m_rghTopDesc[i], hTopDiff, m_nTopOffset * g,
                                                      m_rghConvDesc[i],
                                                      m_rgbwdDataAlgo[i],
                                                      wsArgs.WorkspaceData, (int)m_rglWorkspaceBwdDataOffsets[2 * m_nGroup + g],
                                                      m_rglWorkspaceBwdDataSizes[i],
                                                      m_tZero,
                                                      m_rghBottomDesc[i], hBottomDiff, m_nBottomOffset * g,
                                                      false);
                    }
                    // Synchronize the work across groups, each of which went into its own stream.
                    for (int g = 0; g < m_nGroup; g++)
                    {
                        m_cuda.SynchronizeStream(m_rghStream[2 * m_nGroup + g]);
                    }
                }
            }
        }
    }
}
