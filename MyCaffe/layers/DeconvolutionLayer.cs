using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.fillers;
using MyCaffe.common;
using MyCaffe.layers;

namespace MyCaffe.layers
{
    /// <summary>
    /// The DeconvolutionLayer convolves the input with a bank of learned filtered, and (optionally)
    /// add biases, treating filters and convolution parameters in the
    /// opposite sense as ConvolutionLayer.
    /// This layer is initialized with the MyCaffe.param.ConvolutionParameter.
    /// 
    /// ConvolutionLayer computes each output value by dotting an input window with
    /// a filter; DeconvolutionLayer multiplies each input value by a filter
    /// elementwise, and sums over the resulting output windows.  In other words,
    /// DeconvolutionLayer is ConvolutionLayer with the forward and backward passes
    /// reversed.  DeconvolutionLayer reuses ConvolutionParameter for its
    /// parameters, but they take the opposite sense in ConvolutionLayer (so
    /// padding is removed from the output rather than added to the input, and
    /// stride results in upsampling rander than downsampling).
    /// </summary>
    /// <remarks>
    /// @see [Learning Deconvolution Network for Semantic Segmentation](https://arxiv.org/abs/1505.04366) by Hyeonwoo Noh, Seunghoon Hong, and Bohyung Han, 2015.
    /// @see [Learning Fully Convolutional Networks for Iterative Non-blind Deconvolution](https://arxiv.org/abs/1611.06495) by Jiawei Zhang, Wei-Sheng Pan, Rynson Lau, and Ming-Hsuan Yang, 2016. 
    /// @see [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) by Jonathan Long,  Evan Shelhamer, and Trevor Darrell, 2014.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
	public class DeconvolutionLayer<T> : BaseConvolutionLayer<T>
	{
        /// <summary>
        /// The DeconvolutionLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Provides ConvolutionParameter convolution_param with DeconvolutionLayer options:
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
        public DeconvolutionLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.DECONVOLUTION;
        }

        /// <summary>
        /// Returns <i>true</i>, for we want deconvolution, not convolution.
        /// </summary>
        /// <returns><i>true</i> is returned.</returns>
        protected override bool reverse_dimensions()
        {
            return true;
        }

        /// <summary>
        /// Computes the output shape used by the BaseConvolutionLayer.
        /// </summary>
        protected override void compute_output_shape()
        {
            T[] rgKernelShape = m_blobKernelShape.update_cpu_data();
            T[] rgStrideData = m_blobStride.update_cpu_data();
            T[] rgPadData = m_blobPad.update_cpu_data();
            T[] rgDilationData = m_blobDilation.update_cpu_data();

            m_rgOutputShape = new List<int>();

            for (int i = 0; i < m_nNumSpatialAxes; i++)
            {
                int nStride = val_at(rgStrideData, i);
                int nKernel = val_at(rgKernelShape, i);
                int nPad = val_at(rgPadData, i);
                int nDilation = val_at(rgDilationData, i);

                // i+1 to skip channel axis
                int nInputDim = input_shape(i + 1);
                int nKernelExtent = nDilation * (nKernel - 1) + 1;
                int nOutputDim = nStride * (nInputDim - 1) + nKernelExtent - 2 * nPad;
                m_rgOutputShape.Add(nOutputDim);
            }
        }

        /// <summary>
        /// Run the Forward computation (Engine.CAFFE mode supported, only).
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hWeight = m_colBlobs[0].gpu_data;

            for (int i = 0; i < colBottom.Count; i++)
            {
                long hBottomData = colBottom[i].gpu_data;
                long hTopData = colTop[i].mutable_gpu_data;

                for (int n = 0; n < m_nNum; n++)
                {
                    backward_gemm(hBottomData, n * m_nBottomDim, hWeight, hTopData, n * m_nTopDim);

                    if (m_bBiasTerm)
                        forward_bias(hTopData, n * m_nTopDim, m_colBlobs[1].gpu_data);
                }
            }
        }

        /// <summary>
        /// Run the Backward computation using the (Engine.CAFFE mode supported, only).
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (length 1).</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hWeight = m_colBlobs[0].gpu_data;
            long hWeightDiff = m_colBlobs[0].mutable_gpu_diff;

            for (int i = 0; i < colTop.Count; i++)
            {
                long hTopDiff = colTop[i].gpu_diff;
                long hBottomData = colBottom[i].gpu_data;
                long hBottomDiff = colBottom[i].mutable_gpu_diff;

                // Bias gradient if necessary.
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
                    for (int n = 0; n < m_nNum; n++)
                    {
                        // gradient w.r.t. weight.  Note that we will accumulate diffs.
                        if (m_rgbParamPropagateDown[0])
                            weight_gemm(hTopDiff, n * m_nTopDim, hBottomData, n * m_nBottomDim, hWeightDiff);

                        // gradient w.r.t. bottom data, if necessary.
                        if (rgbPropagateDown[i])
                            forward_gemm(hTopDiff, n * m_nTopDim, hWeight, hBottomDiff, n * m_nBottomDim, m_rgbParamPropagateDown[0]);
                    }
                }
            }
        }
    }
}
