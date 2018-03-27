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
    /// Fills a Blob with values @f$ x \sim N(0, \sigma^2) @f$ where 
    /// @f$ \sigma^2 @f$ is set inversely proportionla to number of incomming
    /// nodes, outgoing nodes, or their average.
    /// </summary>
    /// <remarks>
    /// A filler based on the paper [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) by He, Zhang, Ren and Sun 2015, 
    /// which specifically accounts for ReLU nonlinearities.
    /// 
    /// Aside: for another perspective on the scaling factor, see the derivation of
    /// [Learning hierarchical categories in deep neural networks](http://web.stanford.edu/class/psych209a/ReadingsByDate/02_15/SaxeMcCGanguli13.pdf) by Saxe, McClelland, and Ganguli 2013 (v3).
    /// 
    /// It fills the incoming matrix by randomly sampling Gaussian data with @f$ std =
    /// sqrt(2 / n) @f$ where @f$ n @f$ is the fan_in, fan_out or their average, depending on 
    /// the variance_norm option.   You should make sure the input blob has shape (num,
    /// a, b, c) where @f$ a * b * c = fan_in @f$ and @f$ num * b * c = fan_out@f$.  Note that this
    /// is currently not the case for inner product layers.
    /// </remarks>
    /// <typeparam name="T">The base type <i>float</i> or <i>double</i>.</typeparam>
    public class MsraFiller<T> : Filler<T>
    {
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Instance of CudaDnn - connection to cuda.</param>
        /// <param name="log">Log used for output.</param>
        /// <param name="p">Filler parameter that defines the filler settings.</param>
        public MsraFiller(CudaDnn<T> cuda, Log log, FillerParameter p)
            : base(cuda, log, p)
        {
        }

        /// <summary>
        /// Fill the blob with random numbers from a MSRA distribution.
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

            double dfStd = Math.Sqrt(2.0 / dfN);
            double dfMean = 0;
            T fStd = (T)Convert.ChangeType(dfStd, typeof(T));
            T fMean = (T)Convert.ChangeType(dfMean, typeof(T));
            m_cuda.rng_gaussian(nCount, fMean, fStd, b.mutable_gpu_data);

            m_log.CHECK_EQ(-1, m_param.sparse, "Sparsity not supported by this Filler.");
        }
    }
}
