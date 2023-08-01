using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.lnn
{
    /// <summary>
    /// The SiLULayer implements the Sigmoid-weighted Linear Unit (SiLU) activation function 
    /// </summary>
    /// <remarks>
    /// Computes the SiLU non-linearity @f$ y  = x * sigmoid(x) @f$
    ///                                 @f$ y' = sigmoid(x) * (1 + x * (1 - sigmoid(x)) @f$
    /// 
    /// @see [Brief Review - SiLU: Sigmoid-weighted Linear Unit](https://sh-tsang.medium.com/review-silu-sigmoid-weighted-linear-unit-be4bc943624d) by Sik-Ho Tsang, 2022, Medium.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SiLULayer<T> : NeuronLayer<T>
    {
        /// <summary>
        /// The SiLULayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public SiLULayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SILU;
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
        ///         y  =x * sigmoid(x)
        ///     @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();

            m_cuda.silu_fwd(nCount, hBottomData, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the SiLU value inputs.
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
        ///     gradients @f$ y' = sigmoid(x) * (1 + x * (1 - sigmoid(x)) @f$
        ///     if propagate_down[0]
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopData = colTop[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            long hBottomData = colBottom[0].gpu_data;
            int nCount = colBottom[0].count();

            m_cuda.silu_bwd(nCount, hTopDiff, hTopData, hBottomDiff, hBottomData);
        }
    }
}
