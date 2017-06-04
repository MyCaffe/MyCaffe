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
        /// Fill the blob with random numbers from a postive unitball distribution.
        /// </summary>
        /// <param name="b">Specifies the blob to fill.</param>
        public override void Fill(Blob<T> b)
        {
            int nCount = b.count();
            m_log.CHECK(nCount > 0, "There is no data to fill!");
            m_cuda.rng_uniform(nCount, 0, 1, b.mutable_gpu_data);

            // We expect the filler to not be called very frequently, so we will
            // just use a simple implementation.

            T[] rgData = b.mutable_cpu_data;
            int nDim = nCount / b.num;

            m_log.CHECK_GT(nDim, 0, "The dimension must be greater than 0.");

            for (int i = 0; i < b.num; i++)
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

            b.mutable_cpu_data = rgData;

            m_log.CHECK_EQ(-1, m_param.sparse, "Sparsity not supported by this Filler.");
        }
    }
}
