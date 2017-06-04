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
    ///     \begin{array}{lr}
    ///         x                  & \mathrm{if} \; x > 0 \\
    ///         \alpha (\exp(x)-1) & \mathrm{if} \; x \le 0
    ///     \end{array} \right
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

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <remarks>
        /// Computes if @f$ x > 0   => y = x @f$
        ///          if @f$ x <= 0  => y = \alpha (\exp(x)-1) @f$
        /// </remarks>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs @f$ 
        ///         y = \left\{ 
        ///         \begin{array}{lr}
        ///             x                  & \mathrm{if} \; x > 0 \\
        ///             \alpha (\exp(x)-1) & \mathrm{if} \; x \le 0
        ///         \end{array} \right
        ///     @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
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
        ///            1            & \mathrm{if} \; x > 0 \\
        ///            y + \alpha   & \mathrm{if} \; x \le 0
        ///         \end{array} \right.
        ///     @f$ if propagate_down[0] == true.</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
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
    }
}
