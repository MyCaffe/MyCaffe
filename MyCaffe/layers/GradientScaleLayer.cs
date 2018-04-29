using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The GradientScaleLayer which scales the deltas during the backpropagation.
    /// This layer is initialized with the MyCaffe.param.GradientScaleParameter.
    /// </summary>
    /// <remarks>
    /// Scaling is performed according to the schedule:
    /// @f$ y = \frac{2 \cdot height} {1 + \exp(-\alpha \cot progress)} - upper\_bound @f$,
    /// where @f$ height = upper\_bound - lower\_bound @f$,
    /// @f$ lower\_bound @f$ is the smallest scaling factor,
    /// @f$ upper\_bound @f$ is the largest scaling factor,
    /// @f$ \alpha @f$ controls how fast the transition occurs between the scaling factors,
    /// @f$ progress = \min(iter / max\_iter, 1) @f$ corresponds to the current transition
    /// state (@f$ iter @f$ is the current iteration of the solver).
    /// 
    /// The GradientScaleLayer can be used to implement
    /// gradient reversals.
    /// 
    /// @see [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818) by Ganin et al., 2015, v4 in 2016.
    /// @see [Github: ddtm-caffe](https://github.com/ddtm/caffe) for original source.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class GradientScaleLayer<T> : NeuronLayer<T>
    {
        double m_dfLowerBound;
        double m_dfUpperBound;
        double m_dfAlpha;
        double m_dfMaxIter;
        double m_dfCoeff;
        Stopwatch m_swOutput = new Stopwatch();

        /// <summary>
        /// The GradientScaleLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type EXP with parameter exp_param,
        /// with options:
        ///     - lower_bound (\b optional, default 0) the @f$ lower\_bound @f$
        ///     
        ///     - upper_bound (\b optional, default 1) the @f$ upper\_bound @f$ 
        ///     
        ///     - alpha (\b optional, default 10) the @f$ \alpha @f$
        ///     
        ///     - max_iter (\b optional, default 1) the @f$ max\_iter @f$
        /// </param>
        public GradientScaleLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.GRADIENTSCALER;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            GetIterationArgs args = getCurrentIteration();
            if (args != null && args.CurrentPhase != Phase.TRAIN)
                return;

            m_log.CHECK(args != null, "WARNING: The OnGetIteration event is not connected!");

            m_dfLowerBound = m_param.gradient_scale_param.lower_bound;
            m_dfUpperBound = m_param.gradient_scale_param.upper_bound;
            m_dfAlpha = m_param.gradient_scale_param.alpha;
            m_dfMaxIter = m_param.gradient_scale_param.max_iter;
            m_dfCoeff = 1.0; // default adaptation coefficient.

            m_log.CHECK_LE(m_dfLowerBound, m_dfUpperBound, "The lower bound must be <= the upper bound.");
            m_log.CHECK_GE(m_dfAlpha, 0, "The alpha value must be >= 0.0");
            m_log.CHECK_GE(m_dfCoeff, 1, "The max_iter must be >= 1.0");

            int nIteration = (args == null) ? 1 : args.Iteration;
            double dfProgress = Math.Min(1.0, (double)nIteration / m_dfMaxIter);
            double dfHeight = m_dfUpperBound - m_dfLowerBound;

            m_dfCoeff = 2.0 * dfHeight / (1.0 + Math.Exp(-m_dfAlpha * dfProgress)) - dfHeight + m_dfLowerBound;
            m_log.WriteLine("iter = " + nIteration.ToString() + " progress = " + dfProgress.ToString() + " coeff = " + m_dfCoeff.ToString());
            m_swOutput.Start();
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (identity)
        ///  </param>
        /// <param name="colTop">top output Blob vector (identity)
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            colTop[0].ShareData(colBottom[0]);
        }

        /// <summary>
        /// Scales the error gradient w.r.t. the GRADIENTSCALER value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector, providing the error gradient
        /// with respect to outputs
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            GetIterationArgs args = getCurrentIteration();
            if (args != null && args.CurrentPhase != Phase.TRAIN)
                return;

            int nCount = colTop[0].count();
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            int nIteration = (args == null) ? 1 : args.Iteration;
            double dfProgress = Math.Min(1.0, (double)nIteration / m_dfMaxIter);
            double dfHeight = m_dfUpperBound - m_dfLowerBound;

            m_dfCoeff = 2.0 * dfHeight / (1.0 + Math.Exp(-m_dfAlpha * dfProgress)) - dfHeight + m_dfLowerBound;

            if (m_swOutput.Elapsed.TotalMilliseconds > 1000)
            {
                m_log.WriteLine("iter = " + nIteration.ToString() + " progress = " + dfProgress.ToString() + " coeff = " + m_dfCoeff.ToString());
                m_swOutput.Restart();
            }

            m_cuda.scale(nCount, -m_dfCoeff, hTopDiff, hBottomDiff);
        }
    }
}
