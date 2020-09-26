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
    /// The ELULayer computes exponential linear unit non-linearity @f$
    ///     y = \left\{
    ///       \begin{array}{lr}
    ///         x \: \mbox{if} \; x > 0 \\
    ///        \alpha (\exp(x)-1) \: \mbox{if} \; x \le 0
    ///       \end{array} \right. 
    ///     @f$.
    /// This layer is initialized with the MyCaffe.param.EluParameter.
    /// </summary>
    /// <remarks>
    /// @see [Deep Residual Networks with Exponential Linear Unit](https://arxiv.org/abs/1604.04112) by Anish Shah, Eashan Kadam, Hena Shah, Sameer Shinde, and Sandip Shingade, 2016.
    /// @see [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289) by Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ELULayer<T> : NeuronLayer<T>
    {
        long m_hCudnn = 0;
        long m_hBottomDesc = 0;
        long m_hTopDesc = 0;

        /// <summary>
        /// The ELULayer constructor
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides ELUParameter elu_param,
        /// with ELULayer options:
        /// - alpha (\b optional, default 1).
        ///   the values @f$ \alpha @f$ by which controls saturation for negative inputs.
        /// </param>
        public ELULayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.ELU;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_hBottomDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hBottomDesc);
                m_hBottomDesc = 0;
            }

            if (m_hTopDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hTopDesc);
                m_hTopDesc = 0;
            }

            if (m_hCudnn != 0)
            {
                m_cuda.FreeCuDNN(m_hCudnn);
                m_hCudnn = 0;
            }

            base.dispose();
        }

        /// <summary>
        /// Setup the layer to run in either Engine.CAFFE or Engine.CUDNN mode.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!m_param.elu_param.useCudnn())
                return;

            // Initialize CuDNN
            m_hCudnn = m_cuda.CreateCuDNN();
            m_hBottomDesc = m_cuda.CreateTensorDesc();
            m_hTopDesc = m_cuda.CreateTensorDesc();
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

            if (!m_param.elu_param.useCudnn())
                return;

            int nN = colBottom[0].num;
            int nK = colBottom[0].channels;
            int nH = colBottom[0].height;
            int nW = colBottom[0].width;

            m_cuda.SetTensorDesc(m_hBottomDesc, nN, nK, nH, nW);
            m_cuda.SetTensorDesc(m_hTopDesc, nN, nK, nH, nW);
        }

        /// <summary>
        /// Computes the forward calculation using either the Engine.CAFFE or Engine.CUDNN mode.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the computed outputs @f$
        ///     y = \frac{\exp(2x) - 1}{\exp(2x) + 1}
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!m_param.elu_param.useCudnn())
                forward_cuda(colBottom, colTop);
            else
                forward_cudnn(colBottom, colTop);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs using either the Engine.CAFFE or Engine.CUDNN mode.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs @f$ x @f$; Backward fills their diff with 
        ///     gradients @f$ 
        ///     \frac{\partial E}{\partial y} 
        ///         = \frac{\partial E}{\partial y}
        ///           \left(1 - \left[\frac{\exp(2x) - 1}{\exp(2x) + 1} \right]^2 \right)
        ///         = \frac{\partial E}{\partial y} (1 - y^2)
        ///     @f$ if propagate_down[0] == true
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!m_param.elu_param.useCudnn())
                backward_cuda(colTop, rgbPropagateDown, colBottom);
            else
                backward_cudnn(colTop, rgbPropagateDown, colBottom);
        }


        /// <summary>
        /// The forward computation using Cuda.
        /// </summary>
        /// <remarks>
        /// Computes if @f$ x > 0   => y = x @f$
        ///          if @f$ x \leq 0  => y = \alpha (\exp(x)-1) @f$
        /// </remarks>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs @f$ 
        ///         y = \left\{ 
        ///         \begin{array}{lr}
        ///           x \: \mbox{if} \; x > 0 \\
        ///           \alpha (\exp(x)-1) \: \mbox{if} \; x \le 0
        ///         \end{array} \right. 
        ///     @f$.
        /// </param>
        protected void forward_cuda(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colTop[0].count();
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            double dfAlpha = m_param.elu_param.alpha;

            m_cuda.elu_fwd(nCount, hBottomData, hTopData, dfAlpha);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the ELU value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$ 
        ///     with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$; Backward fills their diff with 
        ///     gradients @f$
        ///         \frac{\partial E}{\partial x} = \left\{
        ///         \begin{array}{lr}
        ///           1 \: \mbox{if} \; x > 0 \\
        ///           y + \alpha \: \mbox{if} \; x \le 0
        ///         \end{array} \right. 
        ///     @f$ if propagate_down[0] == true.</param>
        protected void backward_cuda(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            int nCount = colTop[0].count();
            long hTopDiff = colTop[0].gpu_diff;
            long hTopData = colTop[0].gpu_data;
            long hBottomData = colBottom[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            double dfAlpha = m_param.elu_param.alpha;

            m_cuda.elu_bwd(nCount, hTopDiff, hTopData, hBottomData, hBottomDiff, dfAlpha);
        }

        /// <summary>
        /// The forward computation using cuDNN.
        /// </summary>
        /// <remarks>
        /// Computes if @f$ x > 0   => y = x @f$
        ///          if @f$ x \leq 0  => y = \alpha (\exp(x)-1) @f$
        /// </remarks>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs @f$ 
        ///         y = \left\{ 
        ///         \begin{array}{lr}
        ///           x \: \mbox{if} \; x > 0 \\
        ///           \alpha (\exp(x)-1) \: \mbox{if} \; x \le 0
        ///         \end{array} \right. 
        ///     @f$.
        /// </param>
        protected void forward_cudnn(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            m_cuda.EluForward(m_hCudnn, m_tOne, m_hBottomDesc, hBottomData, m_tZero, m_hTopDesc, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the ELU value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$ 
        ///     with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$; Backward fills their diff with 
        ///     gradients @f$
        ///         \frac{\partial E}{\partial x} = \left\{
        ///         \begin{array}{lr}
        ///           1 \: \mbox{if} \; x > 0 \\
        ///           y + \alpha \: \mbox{if} \; x \le 0
        ///         \end{array} \right. 
        ///     @f$ if propagate_down[0] == true.</param>
        protected void backward_cudnn(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            long hTopData = colTop[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomData = colBottom[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            m_cuda.EluBackward(m_hCudnn, m_tOne, m_hTopDesc, hTopData, m_hTopDesc, hTopDiff, m_hBottomDesc, hBottomData, m_tZero, m_hBottomDesc, hBottomDiff);
        }
    }
}
