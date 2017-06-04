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
    /// The LogLayer computes the log of the input.
    /// This layer is initialized with the MyCaffe.param.LogParameter.
    /// </summary>
    /// <remarks>
    /// Computes @f$ y = log_{\gamma}(\alpha x + \beta) @f$,
    /// as specified by the scale @f$ \alpha @f$, shift @f$ \beta @f$,
    /// and base @f$ \gamma @f$.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class LogLayer<T> : NeuronLayer<T>
    {
        double m_dfBaseScale;
        double m_dfInputScale;
        double m_dfInputShift;
        double m_dfBackwardNumScale;

        /// <summary>
        /// The LogLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type LOG with parameter log_param,
        /// with options:
        ///     - scale (\b optional, default 1) the scale @f$ \alpha @f$
        ///     
        ///     - shift (\b optional, default 0) the shift @f$ \beta @f$ 
        ///     
        ///     - base (\b optional, default -1 for a value of @f$ e \approx 2.718 @f$) the base @f$ \gamma @f$</param>
        public LogLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.LOG;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            double dfBase = m_param.log_param.base_val;

            if (dfBase != -1)
                m_log.CHECK_GT(dfBase, 0, "base_val must be strictly positive.");

            // If base == -1, interpret the base as e and set log_base = 1 exactly.
            // Otehrwise, calculate its log explicitly.
            double dfLogBase = (dfBase == -1) ? 1 : Math.Log(dfBase);

            m_log.CHECK(!double.IsNaN(dfLogBase), "NaN result: log(base) == log(" + dfBase.ToString() + ") = " + dfLogBase.ToString());
            m_log.CHECK(!double.IsInfinity(dfLogBase), "Inf result: log(base) == log(" + dfBase.ToString() + ") = " + dfLogBase.ToString());

            m_dfBaseScale = 1.0 / dfLogBase;

            m_log.CHECK(!double.IsNaN(m_dfBaseScale), "NaN result: 1/log(base) == 1/log(" + dfBase.ToString() + ") = " + m_dfBaseScale.ToString());
            m_log.CHECK(!double.IsInfinity(m_dfBaseScale), "Inf result: 1/log(base) == 1/log(" + dfBase.ToString() + ") = " + m_dfBaseScale.ToString());

            m_dfInputScale = m_param.log_param.scale;
            m_dfInputShift = m_param.log_param.shift;
            m_dfBackwardNumScale = m_dfInputScale / dfLogBase;
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colBottom[0].count();
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            if (m_dfInputScale == 1.0 && m_dfInputShift == 0)
            {
                m_cuda.log(nCount, hBottomData, hTopData);
            }
            else
            {
                m_cuda.copy(nCount, hBottomData, hTopData);

                if (m_dfInputScale != 1)
                    m_cuda.scal(nCount, convert(m_dfInputScale), hTopData);

                if (m_dfInputShift != 0)
                    m_cuda.add_scalar(nCount, convert(m_dfInputShift), hTopData);

                m_cuda.log(nCount, hTopData, hTopData);
            }

            if (m_dfBaseScale != 1)
                m_cuda.scal(nCount, convert(m_dfBaseScale), hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the LOG value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$</param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///  </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            int nCount = colBottom[0].count();
            long hBottomData = colBottom[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            m_cuda.copy(nCount, hBottomData, hBottomDiff);

            if (m_dfInputScale != 1.0)
                m_cuda.scal(nCount, convert(m_dfInputScale), hBottomDiff);

            if (m_dfInputShift != 0)
                m_cuda.add_scalar(nCount, convert(m_dfInputShift), hBottomDiff);

            m_cuda.powx(nCount, hBottomDiff, convert(-1.0), hBottomDiff);

            if (m_dfBackwardNumScale != 1.0)
                m_cuda.scal(nCount, convert(m_dfBackwardNumScale), hBottomDiff);

            m_cuda.mul(nCount, hTopDiff, hBottomDiff, hBottomDiff);
        }
    }
}
