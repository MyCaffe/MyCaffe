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
    /// The MishLayer provides a novel activation function that tends to work better than ReLU.
    /// This layer is initialized with the MyCaffe.param.MishParameter.
    /// </summary>
    /// <remarks>
    /// Computes the mish non-linearity @f$ y  = x * \tanh(\ln( 1 + \exp(x) )) @f$.
    /// with                            @f$ y' = \frac{\exp(x) * (4*\exp(x) * x + 4*x + 6*\exp(x) + 4*\exp(2x) + \exp(3x) + 4)}{(2*\exp(x) + \exp(2x) + 2)^2} @f$
    /// Note, see Wolfram Alpha with 'derivative of @f$ x * \tanh(\ln(1 + \exp(x))) @f$'                                         
    /// 
    /// @see [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681v1) by Diganta Misra, 2019.
    /// @see [Meet Mish — New State of the Art AI Activation Function. The successor to ReLU?](https://lessw.medium.com/meet-mish-new-state-of-the-art-ai-activation-function-the-successor-to-relu-846a6d93471f) by Less Wright, 2019
    /// @see [Swish Vs Mish: Latest Activation Functions](https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/) by Krutika Bapat, 2020
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class MishLayer<T> : NeuronLayer<T>
    {
        /// <summary>
        /// The MishLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Mish with parameter Mish_param
        /// </param>
        public MishLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.MISH;
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs @f$ 
        ///         y = x \tanh(\ln(1 + \exp(x)))
        ///     @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();

            m_cuda.mish_fwd(nCount, hBottomData, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the Mish value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$; Backward fills their diff with 
        ///     gradients @f$
        ///         \frac{\partial E}{\partial x}
        ///             = \frac{\partial E}{\partial y}\frac{\exp(x) * (4*\exp(x) * x + 4*x + 6*\exp(x) + 4*\exp(2x) + \exp(3x) + 4)}{(2*\exp(x) + \exp(2x) + 2)^2}
        ///     @f$ if propagate_down[0]
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopData = colTop[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            long hBottomData = colBottom[0].gpu_data;
            int nCount = colBottom[0].count();

            m_cuda.mish_bwd(nCount, hTopDiff, hTopData, hBottomDiff, hBottomData);
        }
    }
}
