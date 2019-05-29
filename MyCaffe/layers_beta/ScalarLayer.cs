using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
using MyCaffe.param;

namespace MyCaffe.layers_beta
{
    /// <summary>
    /// The ScalarLayer computes the operation with the value on the input.
    /// </summary>
    /// <remarks>
    /// Computes @f$ y = val operation x @f$
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ScalarLayer<T> : NeuronLayer<T>
    {
        /// <summary>
        /// The ScalarLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter.</param>
        public ScalarLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SCALAR;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            // Setup the convert to half flags used by the Layer just before calling forward and backward.
            m_bUseHalfSize = m_param.use_halfsize;
        }

        /// <summary>
        /// Computes @f$ y = val operation x @f$
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs x</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs @f$ y = val operation x @f$</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colTop[0].count();
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            m_cuda.copy(nCount, hBottomData, hTopData);

            switch (m_param.scalar_param.operation)
            {
                case ScalarParameter.ScalarOp.MUL:
                    if (m_param.scalar_param.value != 1.0)
                        m_cuda.mul_scalar(nCount, m_param.scalar_param.value, hTopData);
                    break;

                case ScalarParameter.ScalarOp.ADD:
                    if (m_param.scalar_param.value != 0.0)
                        m_cuda.add_scalar(nCount, m_param.scalar_param.value, hTopData);
                    break;

                default:
                    throw new Exception("Unknown scalar operation '" + m_param.scalar_param.operation.ToString());
            }
        }

        /// <summary>
        /// Reverses the previous scalar operation.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$ with
        ///     respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$; Backward fills their diff with gradients @f$
        ///         \frac{\partial E}{\partial x} = \mathrm{sign}{x} \frac{\partial E}{\partial y}
        ///     @f$
        ///     if propagate_down[0] == true.</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            int nCount = colTop[0].count();
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            m_cuda.copy(nCount, hTopDiff, hBottomDiff);

            if (!m_param.scalar_param.passthrough_gradient)
            {
                switch (m_param.scalar_param.operation)
                {
                    case ScalarParameter.ScalarOp.MUL:
                        m_cuda.mul_scalar(nCount, 1.0 / m_param.scalar_param.value, hBottomDiff);
                        break;

                    case ScalarParameter.ScalarOp.ADD:
                        m_cuda.add_scalar(nCount, -m_param.scalar_param.value, hBottomDiff);
                        break;

                    default:
                        throw new Exception("Unknown scalar operation '" + m_param.scalar_param.operation.ToString());
                }
            }
        }
    }
}
