using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// <H3>BETA</H3>
    /// 
    /// The UnPoolingLayer performs GPU based unpooling on the network like Zeiler's paper in ECCV 2014.
    /// 
    /// This layer is initialized with the MyCaffe.param.UnPoolingParameter.
    /// </summary>
    /// <remarks>
    /// * Original implementation at: https://github.com/HyeonwooNoh/caffe (merged into https://github.com/mariolew/caffe-unpooling)
    /// 
    /// @see [A Deep Convolutional Auto-Encoder with Pooling - Unpooling Layers in Caffe](https://arxiv.org/abs/1701.04949) by Volodymyr Turchenko, Eric Chalmers, Artur Luczak, 2017.
    /// @see [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) by Matthew D. Zeiler and Rob Fergus, 2013.
    /// @see [Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation](https://arxiv.org/abs/1506.04924) by Seunghoon Hong, Hyeonwoo Noh, and Bohyung Han, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class UnPoolingLayer<T> : Layer<T>
    {
        int m_nKernelH;
        int m_nKernelW;
        int m_nStrideH;
        int m_nStrideW;
        int m_nPadH;
        int m_nPadW;
        int m_nChannels;
        int m_nHeight;
        int m_nWidth;
        int m_nUnPooledHeight = -1;
        int m_nUnPooledWidth = -1;
        bool m_bGlobalPooling;

        /// <summary>
        /// The UnPoolingLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Uses the same PoolingParameter unpooling_param as the PoolingLayer with options:
        ///  - num_output. The number of filters.
        ///  
        ///  - kernel_size / kernel_h / kernel_w.  The pooling dimensions, given by
        ///  kernel_size for square pooling or kernel_h and kernel-w for rectangular 
        ///  pooling.
        ///  
        ///  - stride / stride_h / stride_w. (\b optional, default 1).  The pool
        ///  stride, given by stride_size for equal dimensions of stride_h and stride_w
        ///  for different strides.  By default the pool is dense with stride 1.
        ///  
        ///  - pad / pad_h / pad_w. (\b optional, default 0). The zero-padding for
        ///  pooling, given by pad for equal dimensions or pad_h and pad_w for
        ///  different padding.  Input padding is computed implicitly instead of 
        ///  actual padding.
        ///  
        ///  - global_pooling (\b optional, default, false). Whether to use global
        ///  pooling or not.
        ///  
        ///  - engine: convolution has Engine.CAFFE (matrix multiplication) and Engine.CUDNN (library
        ///  kernels + stream parallelism) engines.
        ///  </param>
        public UnPoolingLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.UNPOOLING;
        }


        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /// <summary>
        /// Returns the minimum number of required bottom (input) Blobs: input
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of required bottom (input) Blobs: input, mask (only when using MAX)
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return (m_param.unpooling_param.pool == PoolingParameter.PoolingMethod.MAX) ? 2 : 1; }
        }

        /// <summary>
        /// Returns the required number of top (output) Blobs: unpool
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer for use with both Engine.CAFFE and Engine.CUDNN modes.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            UnPoolingParameter p = m_param.unpooling_param;

            if (p.global_pooling)
            {
                m_log.CHECK(!(p.kernel_size.Count > 0 || p.kernel_h.HasValue || p.kernel_w.HasValue), "With global pooling = true, Filter size cannot be specified.");
            }
            else
            {
                m_log.CHECK(!(p.kernel_size.Count > 0) != !(p.kernel_h.HasValue && p.kernel_w.HasValue), "Filter size is kernel_size OR kernel_h and kernel_w; not both.");
                m_log.CHECK(p.kernel_size.Count > 0 || (p.kernel_h.HasValue && p.kernel_w.HasValue), "For non-square filters, both kernel_h and kernel_w are required.");
            }

            m_log.CHECK(((p.pad.Count > 0) && p.pad_h.HasValue && p.pad_w.HasValue) || (!p.pad_h.HasValue && !p.pad_w.HasValue), "Pad is pad or pad_h and pad_w are required.");
            m_log.CHECK(((p.stride.Count > 0) && p.stride_h.HasValue && p.stride_w.HasValue) || (!p.stride_h.HasValue && !p.stride_w.HasValue), "Stride is stride or stride_h and stride_w are required.");
            m_bGlobalPooling = p.global_pooling;


            //---- Kernel Size ----

            if (m_bGlobalPooling)
            {
                m_nKernelH = colBottom[0].height;
                m_nKernelW = colBottom[0].width;
            }
            else
            {
                if (p.kernel_size.Count > 0)
                {
                    m_nKernelH = (int)p.kernel_size[0];
                    m_nKernelW = (int)p.kernel_size[0];
                }
                else
                {
                    m_nKernelH = (int)p.kernel_h.Value;
                    m_nKernelW = (int)p.kernel_w.Value;
                }
            }

            m_log.CHECK_GT(m_nKernelH, 0, "Filter dimensions cannot be zero.");
            m_log.CHECK_GT(m_nKernelW, 0, "Filter dimensions cannot be zero.");


            //---- UnPooling Size Override ----

            if (p.unpool_size.Count > 0)
            {
                m_nUnPooledHeight = (int)p.unpool_size[0];
                m_nUnPooledWidth = (int)p.unpool_size[0];
            }
            else
            {
                m_nUnPooledHeight = (p.unpool_h.HasValue) ? (int)p.unpool_h.Value : -1;
                m_nUnPooledWidth = (p.unpool_w.HasValue) ? (int)p.unpool_w.Value : -1;
            }


            //---- Pad ----

            if (p.pad.Count > 0)
            {
                m_nPadH = (int)p.pad[0];
                m_nPadW = (int)p.pad[0];
            }
            else
            {
                m_nPadH = (p.pad_h.HasValue) ? (int)p.pad_h.Value : 0;
                m_nPadW = (p.pad_w.HasValue) ? (int)p.pad_w.Value : 0;
            }


            //---- Stride ----

            if (p.stride.Count > 0)
            {
                m_nStrideH = (int)p.stride[0];
                m_nStrideW = (int)p.stride[0];
            }
            else
            {
                m_nStrideH = (p.stride_h.HasValue) ? (int)p.stride_h.Value : 1;
                m_nStrideW = (p.stride_w.HasValue) ? (int)p.stride_w.Value : 1;
            }

            if (m_bGlobalPooling)
                m_log.CHECK(m_nPadH == 0 && m_nPadW == 0 && m_nStrideH == 1 && m_nStrideW == 1, "With global pooling = true, only pad = 0 and stride = 1 allowed.");

            if (m_nPadH != 0 || m_nPadW != 0)
            {
                m_log.CHECK(m_param.unpooling_param.pool == PoolingParameter.PoolingMethod.MAX, "Padding implemented for MAX unpooling only.");
                m_log.CHECK_LT(m_nPadH, m_nKernelH, "The pad_h must be <= kernel_h.");
                m_log.CHECK_LT(m_nPadW, m_nKernelW, "The pad_w must be <= kernel_w.");
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(4, colBottom[0].num_axes, "Input must have 4 axes, corresponding to (num, channels, height, width)");

            m_nChannels = colBottom[0].channels;
            m_nHeight = colBottom[0].height;
            m_nWidth = colBottom[0].width;

            if (m_bGlobalPooling)
            {
                m_nKernelH = colBottom[0].height;
                m_nKernelW = colBottom[0].width;
            }

            // Given the original pooling calculation:
            // (int)Math.Ceiling((double)((nOriginalHeight + 2 * m_nPadH - m_nKernelH) / (double)m_nStrideH)) + 1;
            // we do not know whether or not the ceiling kicks in for at the unpooling
            // stage, we do not know the original height.  For this reason, the unpooled 
            // height will always be >= the original height.
            // Using the sizing method from HyeonwooNoh at
            // https://github.com/HyeonwooNoh/caffe/blob/master/src/caffe/layers/unpooling_layer.cpp
            if (m_nUnPooledHeight < 0)
            {
                m_nUnPooledHeight = Math.Max((m_nHeight - 1) * m_nStrideH + m_nKernelH - 2 * m_nPadH,
                                              m_nHeight * m_nStrideH - m_nPadH + 1);
            }

            if (m_nUnPooledWidth < 0)
            {
                m_nUnPooledWidth = Math.Max((m_nWidth - 1) * m_nStrideW + m_nKernelW - 2 * m_nPadW,
                                              m_nWidth * m_nStrideW - m_nPadW + 1);
            }

            if (m_nUnPooledHeight <= 0)
            {
                m_nUnPooledHeight = 1;
                m_log.WriteLine("WARNING: unpooling height was 0, setting to 1.");
            }

            if (m_nUnPooledWidth <= 0)
            {
                m_nUnPooledWidth = 1;
                m_log.WriteLine("WARNING: unpooling width was 0, setting to 1.");
            }

            colTop[0].Reshape(colBottom[0].num, m_nChannels, m_nUnPooledHeight, m_nUnPooledWidth);
        }

        /// <summary>
        /// Run the Forward computation using the Engine.CAFFE mode only.
        /// </summary>
        /// <param name="colBottom">Specifies the bottom input Blob vector (length 1-2).</param>
        /// <param name="colTop">Specifies the top output Blob vector (length 1).</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colBottom[0].count();
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;            
            long hBottomMask = 0;   // We'll get the mas from bottom[1] if its of size > 1

            colTop[0].SetData(0);

            switch (m_param.unpooling_param.pool)
            {
                case PoolingParameter.PoolingMethod.MAX:
                    if (colBottom.Count > 1)
                        hBottomMask = colBottom[1].gpu_data;
                    m_cuda.unpooling_fwd(POOLING_METHOD.MAX, nCount, hBottomData, colBottom[0].num, m_nChannels, m_nHeight, m_nWidth, m_nUnPooledHeight, m_nUnPooledWidth, m_nKernelH, m_nKernelW, m_nStrideH, m_nStrideW, m_nPadH, m_nPadW, hTopData, hBottomMask);
                    break;

                case PoolingParameter.PoolingMethod.AVE:
                    m_cuda.unpooling_fwd(POOLING_METHOD.AVE, nCount, hBottomData, colBottom[0].num, m_nChannels, m_nHeight, m_nWidth, m_nUnPooledHeight, m_nUnPooledWidth, m_nKernelH, m_nKernelW, m_nStrideH, m_nStrideW, m_nPadH, m_nPadW, hTopData, 0);
                    break;

                default:
                    m_log.FAIL("Unknown pooling method '" + m_param.unpooling_param.pool.ToString() + "'");
                    break;
            }
        }

        /// <summary>
        /// Run the Backward computation using the Engine.CAFFE mode only.
        /// </summary>
        /// <param name="colTop">Specifies the top output Blob vector (length 1).</param>
        /// <param name="rgbPropagateDown">Specifies whether or not to propagagte down.</param>
        /// <param name="colBottom">Specifies the bottom input Blob vector (length 1-2).</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            int nCount = colBottom[0].count();
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            long hBottomMask = 0;   // We'll get the mas from bottom[1] if its of size > 1

            colBottom[0].SetDiff(0);

            switch (m_param.unpooling_param.pool)
            {
                case PoolingParameter.PoolingMethod.MAX:
                    if (colBottom.Count > 1)
                        hBottomMask = colBottom[1].gpu_data;
                    m_cuda.unpooling_bwd(POOLING_METHOD.MAX, nCount, hTopDiff, colTop[0].num, m_nChannels, m_nHeight, m_nWidth, m_nUnPooledHeight, m_nUnPooledWidth, m_nKernelH, m_nKernelW, m_nStrideH, m_nStrideW, m_nPadH, m_nPadW, hBottomDiff, hBottomMask);
                    break;

                case PoolingParameter.PoolingMethod.AVE:
                    m_cuda.unpooling_bwd(POOLING_METHOD.AVE, nCount, hTopDiff, colTop[0].num, m_nChannels, m_nHeight, m_nWidth, m_nUnPooledHeight, m_nUnPooledWidth, m_nKernelH, m_nKernelW, m_nStrideH, m_nStrideW, m_nPadH, m_nPadW, hBottomDiff, 0);
                    break;

                default:
                    m_log.FAIL("Unknown pooling method '" + m_param.unpooling_param.pool.ToString() + "'");
                    break;
            }
        }
    }
}
