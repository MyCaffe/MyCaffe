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
    /// The PowerLayer computes the power of the input.
    /// This layer is initialized with the MyCaffe.param.PowerParameter.
    /// </summary>
    /// <remarks>
    /// Computes @f$ y = (\alpha x + \beta) ^ \gamma @f$
    /// as specified by the scale @f$ \alpha @f$, shift @f$ \beta @f$, 
    /// and power @f$ \gamma @f$.
    /// 
    /// @see [Optimizing a Shallow Multi-Scale Network for Tiny-Imagenet Classification](http://cs231n.stanford.edu/reports/2015/pdfs/dashb_CS231n_Paper.pdf) by Dash Bodington, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class PowerLayer<T> : NeuronLayer<T>
    {
        double m_dfPower;
        double m_dfScale;
        double m_dfShift;
        double m_dfDiffScale;

        /// <summary>
        /// The PowerLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type LOG with parameter power_param,
        /// with options:
        ///     - scale (\b optional, default 1) the scale @f$ \alpha @f$
        ///     
        ///     - shift (\b optional, default 0) the shift @f$ \beta @f$ 
        ///     
        ///     - power (\b optional, default 1) the power @f$ \gamma @f$</param>
        public PowerLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.POWER;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            m_dfPower = m_param.power_param.power;
            m_dfScale = m_param.power_param.scale;
            m_dfShift = m_param.power_param.shift;
            m_dfDiffScale = m_dfPower * m_dfScale;
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$.
        /// </param>
        /// <param name="colTop">top output blob (length 1)
        ///  -# @f$ (n \times C \times H \times W) @f$
        ///     the computed outputs @f$
        ///       y = (\alpha x + \beta) ^ \gamma
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Special case where we can ignore the input: scale or power are 0.
            if (m_dfDiffScale == 0)
            {
                colTop[0].SetData((m_dfPower == 0) ? 1.0 : Math.Pow(m_dfShift, m_dfPower));
                return;
            }

            int nCount = colBottom[0].count();
            long hTopData = colTop[0].mutable_gpu_data;
            long hBottomData = colBottom[0].gpu_data;

            m_cuda.copy(nCount, hBottomData, hTopData);

            if (m_dfScale != 1.0)
                m_cuda.scal(nCount, convert(m_dfScale), hTopData);

            if (m_dfShift != 0)
                m_cuda.add_scalar(nCount, convert(m_dfShift), hTopData);

            if (m_dfPower != 1.0)
                m_cuda.powx(nCount, hTopData, convert(m_dfPower), hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the power inputs
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$.
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">the input blob (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$; backward fills their diff with
        ///     gradients @f$
        ///       \frac{\partial E}{\partial y} =
        ///          \frac{partial E}{\partial y}
        ///          \alpha \gamma (\alpha x + \beta) ^ {\gamma - 1} =
        ///          \frac{\partial E}{\partial y}
        ///          \frac{\alpha \gamma y}{\alpha x + \beta}
        ///       @f$ if propagate_down[0] == true.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            int nCount = colBottom[0].count();
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            if (m_dfDiffScale == 0 || m_dfPower == 1)
            {
                colBottom[0].SetDiff(m_dfDiffScale);
            }
            else
            {
                long hBottomData = colBottom[0].gpu_data;

                // Compute dx/dy = scale * power * (shift + scale * x)^(power - 1)
                //               = diff_scale * y / (shift + scale * x)
                if (m_dfPower == 2)
                {
                    // Special case for y = (shift + scale * x)^2
                    //           -> dy/dx = 2 * scale * (shift + scale * x)
                    //                    = diff_scale * shift + diff_scale * scale * x 
                    m_cuda.axpby(nCount, convert(m_dfDiffScale * m_dfScale), hBottomData, m_tZero, hBottomDiff);

                    if (m_dfShift != 0)
                        m_cuda.add_scalar(nCount, convert(m_dfDiffScale * m_dfShift), hBottomDiff);
                }
                else if (m_dfShift == 0)
                {
                    // Special case for y = (scale * x)^power
                    //           -> dy/dx = scale * power * (scale * x)^(power - 1)
                    //                    = scale * power * (scale * x)^{power} * (scale * x)^(-1)
                    //                    = power * y / x 
                    long hTopData = colTop[0].gpu_data;
                    m_cuda.div(nCount, hTopData, hBottomData, hBottomDiff);
                    m_cuda.scal(nCount, convert(m_dfPower), hBottomDiff);
                }
                else
                {
                    m_cuda.copy(nCount, hBottomData, hBottomDiff);

                    if (m_dfScale != 1.0)
                        m_cuda.scal(nCount, convert(m_dfScale), hBottomDiff);

                    if (m_dfShift != 0.0)
                        m_cuda.add_scalar(nCount, convert(m_dfShift), hBottomDiff);

                    long hTopData = colTop[0].gpu_data;
                    m_cuda.div(nCount, hTopData, hBottomDiff, hBottomDiff);

                    if (m_dfDiffScale != 1.0)
                        m_cuda.scal(nCount, convert(m_dfDiffScale), hBottomDiff);
                }
            }

            m_cuda.mul(nCount, hTopDiff, hBottomDiff, hBottomDiff);
        }
    }
}
