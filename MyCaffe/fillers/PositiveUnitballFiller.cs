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
    /// Fills a Blob with values @f$ x \in [0, 1] @f$
    ///     such that @f$ \forall i \sum_j x{ij} = 1 @f$.
    /// </summary>
    /// <typeparam name="T">The base type <i>float</i> or <i>double</i>.</typeparam>
    public class PositiveUnitballFiller<T> : Filler<T>
    {
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Instance of CudaDnn - connection to cuda.</param>
        /// <param name="log">Log used for output.</param>
        /// <param name="p">Filler parameter that defines the filler settings.</param>
        public PositiveUnitballFiller(CudaDnn<T> cuda, Log log, FillerParameter p)
            : base(cuda, log, p)
        {
        }


        /// <summary>
        /// Fill the memory with random numbers from a postive unitball distribution.
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
            m_cuda.rng_uniform(nCount, 0, 1, hMem);

            // We expect the filler to not be called very frequently, so we will
            // just use a simple implementation.

            T[] rgData = m_cuda.GetMemory(hMem);
            int nDim = nCount / nNumOutputs;

            m_log.CHECK_GT(nDim, 0, "The dimension must be greater than 0.");

            for (int i = 0; i < nNumOutputs; i++)
            {
                double dfSum = 0;

                for (int j = 0; j < nDim; j++)
                {
                    int nIdx = i * nDim + j;
                    dfSum += (double)Convert.ChangeType(rgData[nIdx], typeof(double));
                }

                for (int j = 0; j < nDim; j++)
                {
                    int nIdx = i * nDim + j;
                    double dfVal = (double)Convert.ChangeType(rgData[nIdx], typeof(double)) / dfSum;
                    rgData[nIdx] = (T)Convert.ChangeType(dfVal, typeof(T));
                }
            }

            m_cuda.SetMemory(hMem, rgData);

            m_log.CHECK_EQ(-1, m_param.sparse, "Sparsity not supported by this Filler.");
        }
    }
}
