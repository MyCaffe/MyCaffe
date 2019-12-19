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
    /// The ThresholdLayer is a neuron layer that tests whether the input exceeds a threshold: outputs 1 for inputs
    /// above threshold; 0 otherwise.
    /// This layer is initialized with the MyCaffe.param.ThresholdParameter.
    /// </summary>
    /// <remarks>
    /// @see [Neural Networks with Input Specified Thresholds](http://cs231n.stanford.edu/reports/2016/pdfs/118_Report.pdf) by Fei Liu and Junyang Qian, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ThresholdLayer<T> : NeuronLayer<T>
    {
        double m_dfThreshold = 0;

        /// <summary>
        /// The ThresholdLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type THRESHOLD with parameter threshold_param,
        /// with options:
        ///   - threshold (\b optional, default 0). The threshold value @f$ x @f$ to which the input values are compared.
        /// </param>
        public ThresholdLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.THRESHOLD;
        }

        /// <summary>
        /// Setup the layer to run in either Engine.CAFFE or Engine.CUDNN mode.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            m_dfThreshold = m_param.threshold_param.threshold;
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$.</param>
        /// <param name="colTop">top output blob (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs @f$
        ///       y = \left\{
        ///       \begin{array}{lr}
        ///         0 \: \mathrm{if} \; x \le t \\
        ///         1 \: \mathrm{if} \; x > t
        ///       \end{array} \right.
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();

            m_cuda.threshold_fwd(nCount, m_dfThreshold, hBottomData, hTopData);
        }

        /// @brief Not implemented (non-diferentiable function)
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            throw new NotImplementedException();
        }
    }
}
