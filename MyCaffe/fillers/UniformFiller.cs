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
    /// Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$
    /// </summary>
    /// <remarks>
    /// @see [Uniform Distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)) Wikipedia.
    /// </remarks>
    /// <typeparam name="T">The base type <i>float</i> or <i>double</i>.</typeparam>
    public class UniformFiller<T> : Filler<T>
    {
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Instance of CudaDnn - connection to cuda.</param>
        /// <param name="log">Log used for output.</param>
        /// <param name="p">Filler parameter that defines the filler settings.</param>
        public UniformFiller(CudaDnn<T> cuda, Log log, FillerParameter p)
            : base(cuda, log, p)
        {
        }

        /// <summary>
        /// Fill the memory with random numbers from a uniform distribution.
        /// </summary>
        /// <param name="nCount">Specifies the number of items to fill.</param>
        /// <param name="hMem">Specifies the handle to GPU memory to fill.</param>
        /// <param name="nNumAxes">Optionally, specifies the number of axes (default = 1).</param>
        /// <param name="nNumOutputs">Optionally, specifies the number of outputs (default = 1).</param>
        /// <param name="nNumChannels">Optionally, specifies the number of channels (default = 1).</param>
        /// <param name="nHeight">Optionally, specifies the height (default = 1).</param>
        /// <param name="nWidth">Optionally, specifies the width (default = 1).</param>
        public override void Fill(int nCount, long hMem, int nNumAxes = 1, int nNumOutputs = 1, int nNumChannels = 1, int nHeight = 1, int nWidth = 1)
        {
            m_log.CHECK(nCount > 0, "There is no data to fill!");
            m_cuda.rng_uniform(nCount, m_param.min, m_param.max, hMem);
        }
    }
}
