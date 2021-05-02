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
    /// The ClipLayer provides a neuron layer that clips the data to fit within the 
    /// [min,max] range.
    /// This layer is initialized with the MyCaffe.param.ClipParameter.
    /// </summary>
    /// <remarks>
    /// Computes the clip function @f$ y = x \max(min, \min(max,x)) @f$.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ClipLayer<T> : NeuronLayer<T>
    {
        /// <summary>
        /// The ClipLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Clip with parameter Clip_param,
        /// with options:
        ///     - min the value @f$ \min @f$
        ///     - max the value @f$ \max @f$
        /// </param>
        public ClipLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.CLIP;
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
        ///         y = \max(min, \min(max,x))
        ///     @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();
            double dfMin = m_param.clip_param.min;
            double dfMax = m_param.clip_param.max;

            m_cuda.clip_fwd(nCount, hBottomData, hTopData, (T)Convert.ChangeType(dfMin, typeof(T)), (T)Convert.ChangeType(dfMax, typeof(T)));
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the Clip value inputs.
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
        ///         \frac{\partial E}{\partial x} = \left\{
        ///         \begin{array}{lr}
        ///           0 \: \mathrm{if} \; x < min \vee x > max \\           
        ///           \frac{\partial E}{\partial y} \: \mathrm{if} \; x \ge min \wedge x \le max
        ///         \end{array} \right.
        ///     @f$ 
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
            {
                long hBottomData = colBottom[0].gpu_data;
                long hTopDiff = colTop[0].gpu_diff;
                long hBottomDiff = colBottom[0].mutable_gpu_diff;
                int nCount = colBottom[0].count();
                double dfMin = m_param.clip_param.min;
                double dfMax = m_param.clip_param.max;

                m_cuda.clip_bwd(nCount, hTopDiff, hBottomData, hBottomDiff, (T)Convert.ChangeType(dfMin, typeof(T)), (T)Convert.ChangeType(dfMax, typeof(T)));
            }
        }
    }
}
