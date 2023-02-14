using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The GeluLayer implements the New GELU activation function currently in Google BERT repo (same as OpenAI GPT)
    /// </summary>
    /// <remarks>
    /// Computes the gelu non-linearity @f$ y  = 0.5 * x * (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x^3))) @f$
    ///                                 @f$ y' = 0.5 * tanh(0.797885 * (x + 0.044715 * x^3)) + 
    ///                                          (0.0535161 * x^3 + 0.398942 * x) * sech^2(0.797885 * (x + 0.044715 * x^3)) + 0.5 @f$
    /// Note, see Wolfram Alpha with 'derivative of @f$ d/dx  = 0.5 * x * (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x^3))) @f$                                         
    /// 
    /// @see [Github - Karpathy: NewGELU, line 21](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) by Karpathy, 2022.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class GeluLayer<T> : NeuronLayer<T>
    {
        bool m_bEnableBertVersion;

        /// <summary>
        /// The GeluLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public GeluLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.GELU;
            m_bEnableBertVersion = p.gelu_param.enable_bert_version;
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
        ///         y  = 0.5 * x (1.0 + tanh(sqrt(2.0/PI) * (x + 0.044715 * x^3)))
        ///     @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();

            m_cuda.gelu_fwd(nCount, hBottomData, hTopData, m_bEnableBertVersion);
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
        ///     gradients f$ y' = 0.5 * tanh(0.797885 * (x + 0.044715 * x^3)) + 
        ///                       (0.0535161 * x^3 + 0.398942 * x) * sech^2(0.797885 * (x + 0.044715 * x^3)) + 0.5
        ///     @f$ if propagate_down[0]
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopData = colTop[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            long hBottomData = colBottom[0].gpu_data;
            int nCount = colBottom[0].count();

            m_cuda.gelu_bwd(nCount, hTopDiff, hTopData, hBottomDiff, hBottomData, m_bEnableBertVersion);
        }
    }
}
