using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.basecode;
using System.Drawing;

namespace MyCaffe.layers.alpha
{
    /// <summary>
    /// <H3>PRE ALPHA</H3>
    /// 
    /// The UnPoolingLayer1 performs CPU based unpooling on the network like Zeiler's paper in ECCV 2014.
    /// 
    /// This layer is initialized with the Caffe.net.param.PoolingParameter.
    /// </summary>
    /// <remarks>
    /// * Original implementation at: https://github.com/mariolew/caffe-unpooling
    /// 
    /// @see [A Deep Convolutional Auto-Encoder with Pooling - Unpooling Layers in Caffe](https://arxiv.org/abs/1701.04949) by Turchenko, Volodymyr and Chalmers, Eric and Luczak, Artur, 2017.
    /// @see [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) by Zeiler, Matthew D. and Fergus, Rob, 2013.
    /// @see [Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation](https://arxiv.org/abs/1506.04924) by Hong, Seunghoon and Noh, Hyeonwoo and Han, Bohyung, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class UnPoolingLayer1<T> : Layer<T>
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
        int m_nUnPooledHeight;
        int m_nUnPooledWidth;
        bool m_bGlobalPooling;

        /// <summary>
        /// The UnPoolingLayer1 constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Uses the same PoolingParameter pooling_param as the PoolingLayer with options:
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
        public UnPoolingLayer1(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.UNPOOLING1;
        }


        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /// <summary>
        /// Returns the required number of bottom (input) Blobs: pool, mask
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the required number of top (output) Blobs: input
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
            PoolingParameter p = m_param.pooling_param;

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

            m_nUnPooledHeight = (int)((m_nHeight - 1) * m_nStrideH + m_nKernelH - 2 * m_nPadH); 
            m_nUnPooledWidth = (int)((m_nWidth - 1) * m_nStrideW + m_nKernelW - 2 * m_nPadW);

            if (m_nPadH > 0)
            {
                m_nUnPooledHeight -= (m_nHeight % 2 == 0) ? 1 : 0;
                if ((m_nHeight - 1) * m_nStrideH >= m_nUnPooledHeight + m_nPadH)
                    m_nUnPooledHeight++;
            }

            if (m_nPadW > 0)
            {
                m_nUnPooledWidth -= (m_nWidth % 2 == 0) ? 1 : 0;
                if ((m_nWidth - 1) * m_nStrideW >= m_nUnPooledWidth + m_nPadW)
                    m_nUnPooledWidth++;
            }

            if (m_nUnPooledHeight <= 0)
            {
                m_nUnPooledHeight = 1;
//                m_log.WriteLine("WARNING: unpooling height was 0, setting to 1.");
            }

            if (m_nUnPooledWidth <= 0)
            {
                m_nUnPooledWidth = 1;
//                m_log.WriteLine("WARNING: unpooling width was 0, setting to 1.");
            }

            colTop[0].Reshape(colBottom[0].num, m_nChannels, m_nUnPooledHeight, m_nUnPooledWidth);
        }

        /// <summary>
        /// Run the Forward computation using the Engine.CAFFE mode only.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (length 1).</param>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Init the top blob to be all zeros since only special places
            // wouldn't be zero then.
            colTop[0].SetData(0);

            T[] bottom_data = colBottom[0].update_cpu_data();
            T[] mask = colBottom[1].update_cpu_data();
            T[] top_data = colTop[0].mutable_cpu_data;
            int nBottomDataOffset = 0;
            int nTopDataOffset = 0;
            int nMaskOffset = 0;

            switch (m_param.pooling_param.pool)
            {
                case PoolingParameter.PoolingMethod.MAX:
                    // Currently only the CPU version is supported
                    for (int n = 0; n < colBottom[0].num; n++)
                    {
                        for (int c = 0; c < m_nChannels; c++)
                        {
                            for (int ph = 0; ph < m_nHeight; ph++)
                            {
                                for (int pw = 0; pw < m_nWidth; pw++)
                                {
                                    int nIdx = ph * m_nWidth + pw;

#warning TODO: Bug - nMaskOffset + nIdx exceed length on last row of mask.
                                    if (nMaskOffset + nIdx < mask.Length)
                                    {
                                        int nTopIdx = (int)Convert.ChangeType(mask[nMaskOffset + nIdx], typeof(int));

#warning TODO: Bug - nTopDataOffset + nTopIdx exceed length on last row of data.
                                        if (nTopDataOffset + nTopIdx < top_data.Length)
                                            top_data[nTopDataOffset + nTopIdx] = bottom_data[nBottomDataOffset + nIdx];
                                    }
                                }
                            }

                            // switch to the next channel.
                            nTopDataOffset += colTop[0].offset(0, 1);
                            nBottomDataOffset += colBottom[0].offset(0, 1);
                            nMaskOffset += colBottom[0].offset(0, 1);
                        }
                    }
                    colTop[0].mutable_cpu_data = top_data;
                    break;

                case PoolingParameter.PoolingMethod.AVE:
                    throw new NotImplementedException("Unpooling is only supported on the MAX type of pooling.");

                case PoolingParameter.PoolingMethod.STOCHASTIC:
                    throw new NotImplementedException("Unpooling is only supported on the MAX type of pooling.");
            }
        }

        /// @brief Currently, not implemented.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            throw new NotImplementedException("UnPooling does not support the backward operation.");
        }
    }
}
