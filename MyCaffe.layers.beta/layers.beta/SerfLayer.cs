using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The SerfLayer provides a novel activation function that tends to work better than ReLU.
    /// </summary>
    /// <remarks>
    /// Computes the serf non-linearity @f$ y  = x erf(\log( 1 + \exp(x) )) @f$.
    /// with                            @f$ f(x)' = \text{erf}\left(\log \left(e^x+1\right)\right)+\frac{2 x e^{x-\log^2\left(e^x+1\right)}}{\sqrt{\pi } \left(e^x+1\right)} @f$
    /// 
    /// @see [Serf: Towards better training of deep neural networks using log-Softplus ERror activation Function](https://arxiv.org/pdf/2108.09598.pdf) by Sayan Nag and Mayukh Bhattacharyya, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SerfLayer<T> : NeuronLayer<T>
    {
        /// <summary>
        /// The SerfLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Mish with parameter Mish_param
        /// </param>
        public SerfLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SERF;
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
        ///         y  = x erf(\ln( 1 + \exp(x) ))
        ///     @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();

            m_cuda.serf_fwd(nCount, hBottomData, hTopData, m_param.serf_param.threshold);
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
        ///             = \text{erf}\left(\log \left(e^x+1\right)\right)+\frac{2 x e^{x-\log^2\left(e^x+1\right)}}{\sqrt{\pi } \left(e^x+1\right)} @f$
        ///     @f$ if propagate_down[0]
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopData = colTop[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            long hBottomData = colBottom[0].gpu_data;
            int nCount = colBottom[0].count();

            m_cuda.serf_bwd(nCount, hTopDiff, hTopData, hBottomDiff, hBottomData, m_param.serf_param.threshold);
        }
    }
}
