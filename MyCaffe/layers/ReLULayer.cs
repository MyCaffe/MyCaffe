using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.common;
using MyCaffe.basecode;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The ReLULayer computes the "Rectifier Linear Unit" ReLULayer non-linearity, a classic for neural networks.
    /// This layer is initialized with the MyCaffe.param.ReLUParameter.
    /// </summary>
    /// <remarks>
    /// Computation: @f$ y = (1 + \exp(-x))^{-1} @f$
    /// <br/>
    /// Note that the gradient vanishes as the values move away from 0.
    /// The ReLULayer is often a better choice for this reason.
    /// <br/>
    /// @see [Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853) by Bing Xu, Naiyan Wang, Tianqi Chen, and Mu Li, 2015.
    /// @see [Revise Saturated Activation Functions](https://arxiv.org/abs/1602.05980?context=cs) by Bing Xu, Ruitong Huang, and Mu Li, 2016.
    /// @see [Rectifier Nonlinearities Improve Neural Network Acoustic Models](http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf) by Andrew L. Maas, Awni Y. Hannun, and Andrew Y. Ng, 2013.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ReLULayer<T> : NeuronLayer<T>
    {
        long m_hCudnn = 0;
        long m_hBottomDesc = 0;
        long m_hTopDesc = 0;

        /// <summary>
        /// The ReLULayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type RELU with parameter relu_param,
        /// with options:
        ///   - engine. The engine to use, either Engine.CAFFE, or Engine.CUDNN.
        ///   
        ///   - negative_slope (/b optional, default = 0).  The negative slope.  Allow non-zero slope for negative inputs to speed up optimization. 
        /// </param>
        public ReLULayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.RELU;
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
            if (!m_param.relu_param.useCudnn())
                return;

            // Setup the convert to half flags used by the Layer just before calling forward and backward.
            m_bUseHalfSize = m_param.use_halfsize;

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

            if (!m_param.relu_param.useCudnn())
                return;

            int nN = colBottom[0].num;
            int nK = colBottom[0].channels;
            int nH = colBottom[0].height;
            int nW = colBottom[0].width;

            m_cuda.SetTensorDesc(m_hBottomDesc, nN, nK, nH, nW, m_bUseHalfSize);
            m_cuda.SetTensorDesc(m_hTopDesc, nN, nK, nH, nW, m_bUseHalfSize);
        }

        /// <summary>
        /// Computes the forward calculation using either the Engine.CAFFE or Engine.CUDNN mode.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the computed outputs @f$
        ///         y = (1 + \exp(-x))^{-1}
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!m_param.relu_param.useCudnn())
                forward_cuda(colBottom, colTop);
            else
                forward_cudnn(colBottom, colTop);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs using either the Engine.CAFFE or Engine.CUDNN mode.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.
        ///  -# @f$ (N \times C \times H \times W) @f$ containing error gradients @f$
        ///         \frac{\partial E}{\partial y}
        ///     @f$
        ///     with respect to computed outputs @f$ y @f$.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs @f$ x @f$; 
        ///     Backward fills their diff with gradients @f$
        ///     \frac{\partial E}{\partial y} 
        ///         = \frac{\partial E}{\partial y} y (1 - y)
        ///     @f$ if propagate_down[0] == true
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!m_param.relu_param.useCudnn())
                backward_cuda(colTop, rgbPropagateDown, colBottom);
            else
                backward_cudnn(colTop, rgbPropagateDown, colBottom);
        }

        /// <summary>
        /// Computes the forward calculation using the Engine.CAFFE mode.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the computed outputs @f$
        ///         y = (1 + \exp(-x))^{-1}
        ///     @f$
        /// </param>
        protected void forward_cuda(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();
            T fNegativeSlope = (T)Convert.ChangeType(m_param.relu_param.negative_slope, typeof(T));

            m_cuda.relu_fwd(nCount, hBottomData, hTopData, fNegativeSlope);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs using the Engine.CAFFE mode.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.
        ///  -# @f$ (N \times C \times H \times W) @f$ containing error gradients @f$
        ///         \frac{\partial E}{\partial y}
        ///     @f$
        ///     with respect to computed outputs @f$ y @f$.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs @f$ x @f$; 
        ///     Backward fills their diff with gradients @f$
        ///     \frac{\partial E}{\partial y} 
        ///         = \frac{\partial E}{\partial y} y (1 - y)
        ///     @f$ if propagate_down[0] == true
        /// </param>
        protected void backward_cuda(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopData = colTop[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int nCount = colBottom[0].count();
            T fNegativeSlope = (T)Convert.ChangeType(m_param.relu_param.negative_slope, typeof(T));

            m_cuda.relu_bwd(nCount, hTopDiff, hTopData, hBottomDiff, fNegativeSlope);
        }

        /// <summary>
        /// Computes the forward calculation using the Engine.CUDNN mode.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the computed outputs @f$
        ///         y = (1 + \exp(-x))^{-1}
        ///     @f$
        /// </param>
        protected void forward_cudnn(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            m_cuda.ReLUForward(m_hCudnn, m_tOne, m_hBottomDesc, hBottomData, m_tZero, m_hTopDesc, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs using the Engine.CUDNN mode.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.
        ///  -# @f$ (N \times C \times H \times W) @f$ containing error gradients @f$
        ///         \frac{\partial E}{\partial y}
        ///     @f$
        ///     with respect to computed outputs @f$ y @f$.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs @f$ x @f$; 
        ///     Backward fills their diff with gradients @f$
        ///     \frac{\partial E}{\partial y} 
        ///         = \frac{\partial E}{\partial y} y (1 - y)
        ///     @f$ if propagate_down[0] == true
        /// </param>
        protected void backward_cudnn(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            long hTopData = colTop[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomData = colBottom[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            m_cuda.ReLUBackward(m_hCudnn, m_tOne, m_hTopDesc, hTopData, m_hTopDesc, hTopDiff, m_hBottomDesc, hBottomData, m_tZero, m_hBottomDesc, hBottomDiff);
        }
    }
}
