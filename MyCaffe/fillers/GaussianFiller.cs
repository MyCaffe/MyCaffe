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
    /// Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
    /// </summary>
    /// <remarks>
    /// @see [Guassian Distribution](https://en.wikipedia.org/wiki/Normal_distribution) Wikipedia.
    /// </remarks>
    /// <typeparam name="T">The base type <i>float</i> or <i>double</i>.</typeparam>
    public class GaussianFiller<T> : Filler<T>
    {
        SyncedMemory<T> m_randVec = null;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Instance of CudaDnn - connection to cuda.</param>
        /// <param name="log">Log used for output.</param>
        /// <param name="p">Filler parameter that defines the filler settings.</param>
        public GaussianFiller(CudaDnn<T> cuda, Log log, FillerParameter p)
            : base(cuda, log, p)
        {
            m_randVec = new SyncedMemory<T>(m_cuda, m_log);
        }

        /// <summary>
        /// Fill the memory with random numbers from a guassian distribution.
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
            m_cuda.rng_gaussian(nCount, m_param.mean, m_param.std, hMem);

            int nSparse = m_param.sparse;
            m_log.CHECK_GE(nSparse, -1, "The sparse value should be >= -1.");

            if (nSparse >= 0)
            {
                // Sparse initialization is implemented for 'weight' blobs; i.e. matrices.
                // These have num == channels == 1; width is number of inputs; height is
                // number of outputs.  The 'sparse' variable specifies the mean number
                // of non-zero input weights for a given output.
                m_log.CHECK_GE(nNumAxes, 1, "The blob must have at least one axis.");

                double dfNonZeroProbability = (double)nSparse / (double)nNumOutputs;
                T fNonZeroProbability = (T)Convert.ChangeType(dfNonZeroProbability, typeof(T));

                m_randVec.Allocate(nCount);
                m_cuda.rng_bernoulli(nCount, fNonZeroProbability, m_randVec.gpu_data);
                m_cuda.mul(nCount, hMem, m_randVec.gpu_data, hMem);
            }
        }
    }
}
