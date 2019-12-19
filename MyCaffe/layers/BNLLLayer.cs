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
    /// The Binomial Normal Log Liklihod Layer.
    /// </summary>
    /// <remarks>
    /// Computes @f$ y = x + \log(1 + \exp(-x)) @f$ if @f$ x > 0 @f$;
    ///          @f$ y =     \log(1 + \exp(x)) @f$ otherwise.
    ///          
    /// @see [Likelihood function](https://en.wikipedia.org/wiki/Likelihood_function) Wikipedia.
    /// @see [Imbalance Aware Lithography Hotspot Detection: A Deep Learning Approach](http://www.cse.cuhk.edu.hk/~byu/papers/C55_SPIE2017_CNN.pdf) by Haoyu Yang, Luyang Luo, Jing Su, Chenxi Lin,  and Bei Yu, 2017. 
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class BNLLLayer<T> : NeuronLayer<T>
    {
        /// <summary>
        /// The BNNLLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type BNNL.
        /// </param>
        public BNLLLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.BNLL;
        }

        /// <summary>
        /// Forward compuation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs @f$
        ///     y = \left\{
        ///       \begin{array}{ll}
        ///         x + \log(1 + \exp(-x)) \: \mbox{if } x > 0 \\
        ///             \log(1 + \exp(x)) \: \mbox{otherwise}
        ///       \end{array} \right.
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();

            m_cuda.bnll_fwd(nCount, hBottomData, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the BNLL inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$.
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$; backward fills their diff with gradients
        ///       @f$ 
        ///         \frac{\partial E}{\partial x}
        ///       @f$ if propagate_down[0] == true.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int nCount = colBottom[0].count();

            m_cuda.bnll_bwd(nCount, hTopDiff, hBottomData, hBottomDiff);
        }
    }
}
