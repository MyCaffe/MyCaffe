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
    /// The SigmoidLayer is a neuron layer that calculates the sigmoid function,
    /// a classc choice for neural networks.
    /// This layer is initialized with the MyCaffe.param.SigmoidParameter.
    /// </summary>
    /// <remarks>
    /// Computation: @f$ y = (1 + \exp(-x))^{-1} @f$
    /// <br/>
    /// Note that the gradient vanishes as the values move away from 0.
    /// The ReLULayer is often a better choice for this reason.
    /// 
    /// @see [eXpose: A Character-Level Convolutional Neural Network with Embeddings For Detecting Malicious URLs, File Paths and Registry Keys](https://arxiv.org/abs/1702.08568v1) by Joshua Saxe and Konstantin Berlin, 2017. 
    /// @see [Residual Attention Network for Image Classification](https://arxiv.org/abs/1704.06904v1) by Fei Wang, Mengquing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang, and Xiaoou Tang, 2017.
    /// @see [Attention and Localization based on a Deep Convolutional Recurrent Model for Weakly Supervised Audio Tagging](https://arxiv.org/abs/1703.06052v1) by Yong Xu, Qiuqiang Kong, Qiang Huang, Wenwu Wang, and Mark D. Plumbley, 2017.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SigmoidLayer<T> : NeuronLayer<T>
    {
        long m_hCudnn = 0;
        long m_hBottomDesc = 0;
        long m_hTopDesc = 0;

        /// <summary>
        /// The SigmoidLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type SIGMOID with parameter sigmoid_param,
        /// with options:
        ///   - engine. The engine to use, either Engine.CAFFE, or Engine.CUDNN.
        /// </param>
        public SigmoidLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SIGMOID;
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
            if (!m_param.sigmoid_param.useCudnn())
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
            if (!reshapeNeeded(colBottom, colTop, false))
                return;

            if (!m_param.sigmoid_param.useCudnn())
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
        ///     y = (1 + \exp(-x))^{-1}
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!m_param.sigmoid_param.useCudnn())
                forward_cuda(colBottom, colTop);
            else
                forward_cudnn(colBottom, colTop);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the ganh inputs using either the Engine.CAFFE or Engine.CUDNN mode.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.
        ///  -# @f$ (N \times C \times H \times W) @f$ containing error gradients @f$
        ///         \frac{\partial E}{\partial y}
        ///     @f$
        ///     with respect to computed outputs (y).</param>
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
            if (!m_param.sigmoid_param.useCudnn())
                backward_cuda(colTop, rgbPropagateDown, colBottom);
            else
                backward_cudnn(colTop, rgbPropagateDown, colBottom);
        }

        /// <summary>
        /// Computes the forward calculation using the Engine.CAFFE.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the computed outputs @f$
        ///     y = (1 + \exp(-x))^{-1}
        ///     @f$
        /// </param>
        protected void forward_cuda(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();

            m_cuda.sigmoid_fwd(nCount, hBottomData, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the ganh inputs using the Engine.CAFFE.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.
        ///  -# @f$ (N \times C \times H \times W) @f$ containing error gradients @f$
        ///         \frac{\partial E}{\partial y}
        ///     @f$
        ///     with respect to computed outputs (y).</param>
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

            m_cuda.sigmoid_bwd(nCount, hTopDiff, hTopData, hBottomDiff);
        }

        /// <summary>
        /// Computes the forward calculation using the Engine.CUDNN mode.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the computed outputs @f$
        ///     y = (1 + \exp(-x))^{-1}
        ///     @f$
        /// </param>
        protected void forward_cudnn(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            m_cuda.SigmoidForward(m_hCudnn, m_tOne, m_hBottomDesc, hBottomData, m_tZero, m_hTopDesc, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the ganh inputs using the Engine.CUDNN mode.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.
        ///  -# @f$ (N \times C \times H \times W) @f$ containing error gradients @f$
        ///         \frac{\partial E}{\partial y}
        ///     @f$
        ///     with respect to computed outputs (y).</param>
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

            m_cuda.SigmoidBackward(m_hCudnn, m_tOne, m_hTopDesc, hTopData, m_hTopDesc, hTopDiff, m_hBottomDesc, hBottomData, m_tZero, m_hBottomDesc, hBottomDiff);
        }
    }
}
