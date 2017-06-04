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
    /// During training only, sets a random portion of @f$ x @f$ to 0, adjusting
    /// the rest of the vector magnitude accordingly
    /// This layer is initialized with the MyCaffe.param.DropoutParameter.
    /// </summary>
    /// <remarks>
    /// @see [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) by Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhavsky, and Ruslan R. Salakhutdinov, 2012.
    /// @see [Information Dropout: Learning Optimal Representations Through Noisy Computation](https://arxiv.org/abs/1611.01353) by Alessandro Achille, and Stevano Soatto, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DropoutLayer<T> : NeuronLayer<T>
    {
        /// <summary>
        /// When divided by UINT_MAX, the randomly generated values of u\sim U(0,1)
        /// </summary>
        Blob<T> m_blobRand;
        /// <summary>
        /// The probability p of dropping any input.
        /// </summary>
        double m_dfThreshold;
        /// <summary>
        /// The scale of undropped inputs at train time 1/(1-p)
        /// </summary>
        double m_dfScale;
        uint m_uiThreshold;

        long m_hCuda = 0;
        long m_hBottomDesc = 0;
        long m_hStates = 0;
        long m_hReserved = 0;
        long m_hDropoutDesc = 0;
        string m_strBottomSize = null;
        ulong m_ulStates = 0;
        ulong m_ulReserved = 0;

        /// <summary>
        /// The DeconvolutionLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Provides DropoutParameter dropout_param with options:
        ///  - dropout_ratio. The dropout ratio.
        ///  
        ///  - seed.  Optionally, specifies a seed for the random number generator used.
        /// </param>
        public DropoutLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.DROPOUT;
            m_blobRand = new Blob<T>(cuda, log);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_blobRand != null)
            {
                m_blobRand.Dispose();
                m_blobRand = null;
            }

            if (m_hDropoutDesc != 0)
            {
                m_cuda.FreeDropoutDesc(m_hDropoutDesc);
                m_hDropoutDesc = 0;
            }

            if (m_hStates != 0)
            {
                m_cuda.FreeMemory(m_hStates);
                m_hStates = 0;
            }

            if (m_hReserved != 0)
            {
                m_cuda.FreeMemory(m_hReserved);
                m_hReserved = 0;
            }

            if (m_hBottomDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hBottomDesc);
                m_hBottomDesc = 0;
            }

            if (m_hCuda != 0)
            {
                m_cuda.FreeCuDNN(m_hCuda);
                m_hCuda = 0;
            }

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

            m_dfThreshold = m_param.dropout_param.dropout_ratio;
            m_log.CHECK(m_dfThreshold > 0.0, "Threshold should be > 0");
            m_log.CHECK(m_dfThreshold < 1.0, "Threshold should be < 1");
            m_dfScale = 1.0 / (1.0 - m_dfThreshold);
            m_uiThreshold = (uint)(uint.MaxValue * m_dfThreshold);

            if (!m_param.dropout_param.useCudnn())
                return;

            m_hCuda = m_cuda.CreateCuDNN();
            m_hBottomDesc = m_cuda.CreateTensorDesc();
            m_hDropoutDesc = m_cuda.CreateDropoutDesc();
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            // Setup the cache for random number generation
            m_blobRand.ReshapeLike(colBottom[0]);

            if (!m_param.dropout_param.useCudnn())
                return;

            string strBottomSize = colBottom[0].ToSizeString();
            m_log.CHECK(strBottomSize == colTop[0].ToSizeString(), "The bottom[0] and top[0] must have the same size!");

            if (strBottomSize != m_strBottomSize)
            {
                ulong ulStates;
                ulong ulReserved;

                m_cuda.SetTensorDesc(m_hBottomDesc, colBottom[0].num, colBottom[0].channels, colBottom[0].height, colBottom[0].width);
                m_cuda.GetDropoutInfo(m_hCuda, m_hBottomDesc, out ulStates, out ulReserved);

                if (ulStates > m_ulStates)
                {
                    if (m_hStates != 0)
                        m_cuda.FreeMemory(m_hStates);

                    m_hStates = m_cuda.AllocMemory((long)ulStates);
                    m_ulStates = ulStates;
                }

                if (ulReserved > m_ulReserved)
                {
                    if (m_hReserved != 0)
                        m_cuda.FreeMemory(m_hReserved);

                    m_hReserved = m_cuda.AllocMemory((long)ulReserved);
                    m_ulReserved = ulReserved;
                }

                long lSeed = m_param.dropout_param.seed;

                if (lSeed == 0)
                    lSeed = DateTime.Now.Ticks;

                m_cuda.SetDropoutDesc(m_hCuda, m_hDropoutDesc, m_dfThreshold, m_hStates, lSeed);
                m_strBottomSize = strBottomSize;
            }
        }


        /// <summary>
        /// Run the Forward computation using either the Engine.CAFFE or Engine.CUDNN mode as specified in the LayerParameter.
        /// </summary>
        /// <remarks>
        /// Note: during TESTING and RUN, this layer merely acts as a pass-through.
        /// </remarks>
        /// <param name="colBottom">blottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs.  
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!m_param.dropout_param.useCudnn())
                forward_caffe(colBottom, colTop);
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
            if (!m_param.dropout_param.useCudnn())
                backward_caffe(colTop, rgbPropagateDown, colBottom);
            else
                backward_cudnn(colTop, rgbPropagateDown, colBottom);
        }

        /// <summary>
        /// Run the Forward computation using the Engine CAFFE mode as specified in the LayerParameter.
        /// </summary>
        /// <remarks>
        ///     At training time, we have @f$ 
        ///     y_{\mbox{train}} = \left\{
        ///       \begin{array}{ll}
        ///         \frac{x}{1 - p} & \mbox{if } u > p \\
        ///         0 & \mbox{otherwise}
        ///       \end{array} \right.
        ///     @f$, where @f$ u \sim U(0,1) @f$ is generated independently for each
        ///     input at each iteration.  At test time, we simply have
        ///     @f$ y_{\mbox{test}} = \mathbb{E}[y_{\mbox{train}}] = x @f$.
        ///     
        /// Note: during TESTING and RUN, this layer merely acts as a pass-through.
        /// </remarks>
        /// <param name="colBottom">blottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs.  
        /// </param>
        protected void forward_caffe(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();

            if (m_phase == Phase.TRAIN)
            {
                long hMask = m_blobRand.mutable_gpu_data;

                m_cuda.rng_uniform(nCount, m_tZero, convert(uint.MaxValue), hMask);
                // set thresholds
                m_cuda.dropout_fwd(nCount, hBottomData, hMask, m_uiThreshold, convert(m_dfScale), hTopData);
            }
            else
            {
                m_cuda.copy(nCount, hBottomData, hTopData);
            }
        }

        /// <summary>
        /// Run the Backward computation using the Engine CAFFE mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (length 1).</param>
        protected void backward_caffe(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            if (m_phase == Phase.TRAIN)
            {
                long hMask = m_blobRand.gpu_data;
                int nCount = colBottom[0].count();

                m_cuda.dropout_bwd(nCount, hTopDiff, hMask, m_uiThreshold, convert(m_dfScale), hBottomDiff);
            }
            else
            {
                m_cuda.copy(colTop[0].count(), hTopDiff, hBottomDiff);
            }
        }

        /// <summary>
        /// Run the Forward computation using the Engine CUDNN mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected void forward_cudnn(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            if (m_phase == Phase.TRAIN)
                m_cuda.DropoutForward(m_hCuda, m_hDropoutDesc, m_hBottomDesc, hBottomData, m_hBottomDesc, hTopData, m_hReserved);
            else
                m_cuda.copy(colBottom[0].count(), hBottomData, hTopData);
        }

        /// <summary>
        /// Run the Backward computation using the Engine CUDNN mode as specified in the LayerParameter.
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (length 1).</param>
        protected void backward_cudnn(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            if (m_phase == Phase.TRAIN)
                m_cuda.DropoutBackward(m_hCuda, m_hDropoutDesc, m_hBottomDesc, hTopDiff, m_hBottomDesc, hBottomDiff, m_hReserved);
            else
                m_cuda.copy(colTop[0].count(), hTopDiff, hBottomDiff);
        }
    }
}
