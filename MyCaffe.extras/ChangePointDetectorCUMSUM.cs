using MyCaffe.basecode;
using MyCaffe.common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.extras
{
    /// <summary>
    /// The ChangePointDetectionCUMSUM class computes the CUMSUM of the input data.
    /// </summary>
    /// <typeparam name="T">Specifies the base type of <i>double</i> or <i>float</i>.</typeparam>
    public class ChangePointDetectorCUMSUM<T> : IDisposable
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public ChangePointDetectorCUMSUM()
        {
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
        }

        private float sum(float[] rg, int nStart, int nEnd)
        {
            float fSum = 0;

            for (int i = nStart; i < nEnd; i++)
            {
                fSum += rg[i];
            }

            return fSum;
        }

        /// <summary>
        /// Compute the CUMSUM of the input data.
        /// </summary>
        /// <param name="blobX">Specifies the input data.</param>
        /// <returns>The S values are returned in a new blob.</returns>
        public Blob<T> ComputeSvalues(Blob<T> blobX)
        {
            float[] rgX = Utility.ConvertVecF<T>(blobX.mutable_cpu_data);
            float[] rgT = new float[rgX.Length];
            float[] rgT1 = new float[rgX.Length];

            Array.Clear(rgT, 0, rgT.Length);

            for (int n = 1; n < rgX.Length; n++)
            {
                Array.Clear(rgT1, 0, rgT1.Length);

                for (int k = 1; k < n; k++)
                {
                    double dfSum1 = sum(rgX, 0, k);
                    double dfSum2 = sum(rgX, k, n);
                    double dfSqrt = Math.Sqrt((double)n * (double)k * (double)(n - k));
                    double dfNum = Math.Abs(((double)n - (double)k) * dfSum1 - k * dfSum2);
                    rgT1[k] = (float)(dfNum / dfSqrt);
                }

                rgT[n] = rgT1.Max();
            }

            Blob<T> blob = new Blob<T>(blobX.Cuda, blobX.Log);
            blob.Name = "CumSum";
            blob.ReshapeLike(blobX);
            blob.mutable_cpu_data = Utility.ConvertVec<T>(rgT);

            return blob;
        }
    }
}
