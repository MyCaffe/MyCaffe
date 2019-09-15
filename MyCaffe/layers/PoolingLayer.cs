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
    /// The PoolingLayer pools the input image by taking the max, average, etc. within regions.
    /// This layer is initialized with the MyCaffe.param.PoolingParameter.
    /// </summary>
    /// <remarks>
    /// @see [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) by Vincent Dumoulin and Francesco Visin, 2016.
    /// @see [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150) by Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba, 2015.
    /// @see [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) by Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner, 1998.
    /// </remarks>
    /// <typeparam name="T"></typeparam>
    public class PoolingLayer<T> : Layer<T>
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
        int m_nPooledHeight;
        int m_nPooledWidth;
        bool m_bGlobalPooling;
        Blob<T> m_blobRandIdx;
        Blob<T> m_blobMaxIdx;

        long m_hCudnn = 0;
        long m_hBottomDesc = 0;
        long m_hTopDesc = 0;
        long m_hPoolingDesc = 0;
        PoolingMethod m_method;

        /// <summary>
        /// The PoolingLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Provides PoolingParameter pooling_param with PoolingLayer options:
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
        /// </param>
        public PoolingLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.POOLING;
            m_blobRandIdx = new Blob<T>(cuda, log);
            m_blobRandIdx.Name = m_param.name + " randidx";
            m_blobMaxIdx = new Blob<T>(cuda, log);
            m_blobMaxIdx.Name = m_param.name + " maxidx";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_hPoolingDesc != 0)
            {
                m_cuda.FreePoolingDesc(m_hPoolingDesc);
                m_hPoolingDesc = 0;
            }

            if (m_hTopDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hTopDesc);
                m_hTopDesc = 0;
            }

            if (m_hBottomDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hBottomDesc);
                m_hBottomDesc = 0;
            }

            if (m_hCudnn != 0)
            {
                m_cuda.FreeCuDNN(m_hCudnn);
                m_hCudnn = 0;
            }

            m_blobRandIdx.Dispose();
            m_blobMaxIdx.Dispose();
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                if (!m_param.pooling_param.useCudnn())
                {
                    col.Add(m_blobRandIdx);
                    col.Add(m_blobMaxIdx);
                }

                return col;
            }
        }

        /// <summary>
        /// Returns the required number of bottom (input) Blobs: input
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the required number of top (output) Blobs: pool, mask (Engine.CAFFE only)
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return (m_param.pooling_param.engine == EngineParameter.Engine.CAFFE) ? -1 : 1; }
        }

        /// <summary>
        /// Currentlym Engine.CUDNN does not support the extra top blob.
        /// </summary>
        public override int MinTopBlobs
        {
            get { return (m_param.pooling_param.engine == EngineParameter.Engine.CAFFE) ? 1 : -1; }
        }

        /// <summary>
        /// MAX Pool layers can output an extra top blob for the mask;
        /// others can only output the pooled inputs.
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return (m_param.pooling_param.pool == PoolingParameter.PoolingMethod.MAX) ? 2 : 1; }
        }

        /// <summary>
        /// Setup the layer for use with both Engine.CAFFE and Engine.CUDNN modes.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            PoolingParameter p = m_param.pooling_param;

            if (p.global_pooling)
            {
                if (!(p.kernel_size.Count > 0 || p.kernel_h.HasValue || p.kernel_w.HasValue))
                    m_log.WriteLine("WARNING: With global pooling = true, Filter size cannot be specified, the bottom hxw = '" + colBottom[0].height.ToString() + "x" + colBottom[0].width.ToString() + "' will be used instead for the kernel size.");
            }
            else
            {
                m_log.CHECK(!(p.kernel_size.Count > 0) != !(p.kernel_h.HasValue && p.kernel_w.HasValue), "Filter size is kernel_size OR kernel_h and kernel_w; not both.");
                m_log.CHECK(p.kernel_size.Count > 0 || (p.kernel_h.HasValue && p.kernel_w.HasValue), "For non-square filters, both kernel_h and kernel_w are required.");
            }

            m_log.CHECK(((p.pad.Count == 0) && p.pad_h.HasValue && p.pad_w.HasValue) || (!p.pad_h.HasValue && !p.pad_w.HasValue), "Pad is pad or pad_h and pad_w are required.");
            m_log.CHECK(((p.stride.Count == 0) && p.stride_h.HasValue && p.stride_w.HasValue) || (!p.stride_h.HasValue && !p.stride_w.HasValue), "Stride is stride or stride_h and stride_w are required.");
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
                m_log.CHECK(m_param.pooling_param.pool == PoolingParameter.PoolingMethod.AVE ||
                            m_param.pooling_param.pool == PoolingParameter.PoolingMethod.MAX, "Padding implemented for AVE and MAX pooling only.");
                m_log.CHECK_LT(m_nPadH, m_nKernelH, "The pad_h must be <= kernel_h.");
                m_log.CHECK_LT(m_nPadW, m_nKernelW, "The pad_w must be <= kernel_w.");
            }

            if (!m_param.pooling_param.useCudnn())
                return;


            //---------------------------------------------
            //  cuDnn specific pooling.  
            //
            //  Note only MAX and AVE pooling are supported.
            //---------------------------------------------

            // Setup the convert to half flags used by the Layer just before calling forward and backward.
            m_bUseHalfSize = m_param.use_halfsize;

            if (m_param.pooling_param.pool == PoolingParameter.PoolingMethod.MAX)
                m_method = PoolingMethod.MAX;
            else
                m_method = PoolingMethod.AVE;

            m_hCudnn = m_cuda.CreateCuDNN();
            m_hBottomDesc = m_cuda.CreateTensorDesc();
            m_hTopDesc = m_cuda.CreateTensorDesc();
            m_hPoolingDesc = m_cuda.CreatePoolingDesc();
            m_cuda.SetPoolingDesc(m_hPoolingDesc, m_method, m_nKernelH, m_nKernelW, m_nPadH, m_nPadW, m_nStrideH, m_nStrideW); 
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

            m_nPooledHeight = (int)Math.Ceiling((double)((m_nHeight + 2 * m_nPadH - m_nKernelH) / (double)m_nStrideH)) + 1;
            m_nPooledWidth = (int)Math.Ceiling((double)((m_nWidth + 2 * m_nPadW - m_nKernelW) / (double)m_nStrideW)) + 1;

            if (m_nPooledHeight <= 0)
            {
                m_nPooledHeight = 1;
                m_log.WriteLine("WARNING: pooling height was 0 in layer '" + m_param.name + "', setting to 1.");
            }

            if (m_nPooledWidth <= 0)
            {
                m_nPooledWidth = 1;
                m_log.WriteLine("WARNING: pooling width was 0 in layer '" + m_param.name +"', setting to 1.");
            }

            if (m_nPadH > 0 || m_nPadW > 0)
            {
                // If we have padding, ensure that the last pooling starts strictly
                // inside the image (instead of at the padding); otherwise clip the last.
                if ((m_nPooledHeight - 1) * m_nStrideH >= m_nHeight + m_nPadH)
                    m_nPooledHeight--;

                if ((m_nPooledWidth - 1) * m_nStrideW >= m_nWidth + m_nPadW)
                    m_nPooledWidth--;

                m_log.CHECK_LT((m_nPooledHeight - 1) * m_nStrideH, m_nHeight + m_nPadH, "The pooled height must fit in the image and not overlap onto the padding.");
                m_log.CHECK_LT((m_nPooledWidth - 1) * m_nStrideW, m_nWidth + m_nPadW, "The pooled width must fit in the image and not overlap onto the padding.");
            }

            colTop[0].Reshape(colBottom[0].num, m_nChannels, m_nPooledHeight, m_nPooledWidth, m_bUseHalfSize);

            if (colTop.Count > 1)
                colTop[1].ReshapeLike(colTop[0], m_bUseHalfSize);

            if (!m_param.pooling_param.useCudnn())
            {
                // If max pooling, we will initialize the vector index part.
                if (m_param.pooling_param.pool == PoolingParameter.PoolingMethod.MAX && colTop.Count == 1)
                    m_blobMaxIdx.Reshape(colBottom[0].num, m_nChannels, m_nPooledHeight, m_nPooledWidth);

                // If stochastic pooling, we will initialize the random index part.
                if (m_param.pooling_param.pool == PoolingParameter.PoolingMethod.STOCHASTIC)
                    m_blobRandIdx.Reshape(colBottom[0].num, m_nChannels, m_nPooledHeight, m_nPooledWidth);

                return;
            }

            //---------------------------------------------
            //  cuDnn specific pooling.
            //---------------------------------------------

            m_cuda.SetTensorDesc(m_hBottomDesc, colBottom[0].num, m_nChannels, m_nHeight, m_nWidth, m_bUseHalfSize);
            m_cuda.SetTensorDesc(m_hTopDesc, colBottom[0].num, m_nChannels, m_nPooledHeight, m_nPooledWidth, m_bUseHalfSize);
        }

        /// <summary>
        /// Run the Forward computation using either the Engine.CAFFE or Engine.CUDNN mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!m_param.pooling_param.useCudnn())
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
            if (!m_param.pooling_param.useCudnn())
                backward_cuda(colTop, rgbPropagateDown, colBottom);
            else
                backward_cudnn(colTop, rgbPropagateDown, colBottom);
        }

        /// <summary>
        /// Run the Forward computation using the Engine.CAFFE mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (length 1).</param>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        protected void forward_cuda(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colTop[0].count();

            // We'll output the mask to top[1] if its of size > 1
            bool bUseTopMask = (colTop.Count > 1) ? true : false;
            long hMask = 0;
            long hTopMask = 0;

            switch (m_param.pooling_param.pool)
            {
                case PoolingParameter.PoolingMethod.MAX:
                    if (bUseTopMask)
                        hTopMask = colTop[1].mutable_gpu_data;
                    else
                        hMask = m_blobMaxIdx.mutable_gpu_data;
                    m_cuda.pooling_fwd(POOLING_METHOD.MAX, nCount, hBottomData, colBottom[0].num, m_nChannels, m_nHeight, m_nWidth, m_nPooledHeight, m_nPooledWidth, m_nKernelH, m_nKernelW, m_nStrideH, m_nStrideW, m_nPadH, m_nPadW, hTopData, hMask, hTopMask);
                    break;

                case PoolingParameter.PoolingMethod.AVE:
                    m_cuda.pooling_fwd(POOLING_METHOD.AVE, nCount, hBottomData, colBottom[0].num, m_nChannels, m_nHeight, m_nWidth, m_nPooledHeight, m_nPooledWidth, m_nKernelH, m_nKernelW, m_nStrideH, m_nStrideW, m_nPadH, m_nPadW, hTopData, 0, 0);
                    break;

                case PoolingParameter.PoolingMethod.STOCHASTIC:
                    m_cuda.rng_uniform(nCount, m_tZero, m_tOne, m_blobRandIdx.mutable_gpu_data);
                    if (m_phase == Phase.TRAIN)
                        m_cuda.pooling_fwd(POOLING_METHOD.STO_TRAIN, nCount, hBottomData, colBottom[0].num, m_nChannels, m_nHeight, m_nWidth, m_nPooledHeight, m_nPooledWidth, m_nKernelH, m_nKernelW, m_nStrideH, m_nStrideW, m_nPadH, m_nPadW, hTopData, m_blobRandIdx.gpu_data, 0);
                    else
                        m_cuda.pooling_fwd(POOLING_METHOD.STO_TEST, nCount, hBottomData, colBottom[0].num, m_nChannels, m_nHeight, m_nWidth, m_nPooledHeight, m_nPooledWidth, m_nKernelH, m_nKernelW, m_nStrideH, m_nStrideW, m_nPadH, m_nPadW, hTopData, m_blobRandIdx.gpu_data, 0);
                    break;

                default:
                    m_log.FAIL("Unknown pooling method!");
                    break;
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
            if (!rgbPropagateDown[0])
                return;

            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int nCount = colBottom[0].count();

            // We'll output the mask to top[1] if its of size > 1.
            bool bUseTopMask = (colTop.Count > 1) ? true : false;
            long hMask = 0;
            long hTopMask = 0;

            switch (m_param.pooling_param.pool)
            {
                case PoolingParameter.PoolingMethod.MAX:
                    if (bUseTopMask)
                        hTopMask = colTop[1].gpu_data;
                    else
                        hMask = m_blobMaxIdx.gpu_data;

                    m_cuda.pooling_bwd(POOLING_METHOD.MAX, nCount, hTopDiff, colTop[0].num, m_nChannels, m_nHeight, m_nWidth, m_nPooledHeight, m_nPooledWidth, m_nKernelH, m_nKernelW, m_nStrideH, m_nStrideW, m_nPadH, m_nPadW, hBottomDiff, hMask, hTopMask);
                    break;

                case PoolingParameter.PoolingMethod.AVE:
                    m_cuda.pooling_bwd(POOLING_METHOD.AVE, nCount, hTopDiff, colTop[0].num, m_nChannels, m_nHeight, m_nWidth, m_nPooledHeight, m_nPooledWidth, m_nKernelH, m_nKernelW, m_nStrideH, m_nStrideW, m_nPadH, m_nPadW, hBottomDiff, 0, 0);
                    break;

                case PoolingParameter.PoolingMethod.STOCHASTIC:
                    m_cuda.pooling_bwd(POOLING_METHOD.STO_TRAIN, nCount, hTopDiff, colTop[0].num, m_nChannels, m_nHeight, m_nWidth, m_nPooledHeight, m_nPooledWidth, m_nKernelH, m_nKernelW, m_nStrideH, m_nStrideW, m_nPadH, m_nPadW, hBottomDiff, m_blobRandIdx.gpu_data, 0);
                    break;

                default:
                    m_log.FAIL("Unknown pooling method!");
                    break;
            }
        }

        /// <summary>
        /// Run the Forward computation using the Engine.CUDNN mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected void forward_cudnn(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            m_cuda.PoolingForward(m_hCudnn, m_hPoolingDesc, m_tOne, m_hBottomDesc, hBottomData, m_tZero, m_hTopDesc, hTopData);
        }

        /// <summary>
        /// Run the Backward computation using the Engine.CUDNN mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (length 1).</param>
        protected void backward_cudnn(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopDiff = colTop[0].gpu_diff;
            long hTopData = colTop[0].gpu_data;
            long hBottomData = colBottom[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            m_cuda.PoolingBackward(m_hCudnn, m_hPoolingDesc, m_tOne, m_hTopDesc, hTopData, m_hTopDesc, hTopDiff, m_hBottomDesc, hBottomData, m_tZero, m_hBottomDesc, hBottomDiff);
        }
    }
}
