using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;

namespace MyCaffe.fillers
{
    /// <summary>
    /// Fills a Blob with values @f$ x\sim U(-a, a) @f$ where @f$ a @f$ is 
    /// set inversely proportional to number of incoming nodes, outgoing
    /// nodes, or their average.
    /// </summary>
    /// <remarks>
    /// A filler based on paper [Understanding the difficulty of training deep feedfrward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) by Bengio and Glorot, 2010.
    /// 
    /// It fills the incoming matrix by randomly sampling uniform data from @f$ [-scale,
    /// scale] @f$ where @f$ scale = sqrt(3 / n) @f$ where @f$ n @f$ is the fan_in, fan_out, or their
    /// average, depending on the variance_norm option.  You shold make sure the
    /// input blob has shape (num, a, b, c) where @f$ a * b * c = fan_in @f$ and @f$ num * b * c
    /// = fan_out@f$.  Note that this is currently not the case for inner product layers.
    /// </remarks>
    /// <typeparam name="T">The base type <i>float</i> or <i>double</i>.</typeparam>
    public class XavierFiller<T> : Filler<T>
    {
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Instance of CudaDnn - connection to cuda.</param>
        /// <param name="log">Log used for output.</param>
        /// <param name="p">Filler parameter that defines the filler settings.</param>
        public XavierFiller(CudaDnn<T> cuda, Log log, FillerParameter p)
            : base(cuda, log, p)
        {
        }

        /// <summary>
        /// Fill the blob with random numbers from a xavier distribution.
        /// </summary>
        /// <param name="b">Specifies the blob to fill.</param>
        public override void Fill(Blob<T> b)
        {
            int nCount = b.count();
            m_log.CHECK(nCount > 0, "There is no data to fill!");

            int nFanIn = nCount / b.shape(0);
            // Compatibility with ND blobs
            int nFanOut = b.num_axes > 1 ?
                          b.count() / b.shape(1) :
                          b.count();
            double dfN = nFanIn; // default to fan_in

            if (m_param.variance_norm == FillerParameter.VarianceNorm.AVERAGE)
                dfN = (nFanIn + nFanOut) / 2.0;
            else if (m_param.variance_norm == FillerParameter.VarianceNorm.FAN_OUT)
                dfN = nFanOut;

            double dfScale = Math.Sqrt(3.0 / dfN);
            T fPosScale = (T)Convert.ChangeType(dfScale, typeof(T));
            T fNegScale = (T)Convert.ChangeType(-dfScale, typeof(T));
            m_cuda.rng_uniform(nCount, fNegScale, fPosScale, b.mutable_gpu_data);

            m_log.CHECK_EQ(-1, m_param.sparse, "Sparsity not supported by this Filler.");
        }
    }
}
