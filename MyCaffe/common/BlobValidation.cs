using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The BlobValidation class is used to check blob data for nans and inf values - typically this is only used during debugging.
    /// </summary>
    /// <typeparam name="T">Specifies the base type.</typeparam>
    public class BlobValidation<T> : IDisposable
    {
        Blob<T> m_blobWork = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the connection to CUDA.</param>
        /// <param name="log">Specifies the output log.</param>
        public BlobValidation(CudaDnn<T> cuda, Log log)
        {
            m_blobWork = new Blob<T>(cuda, log);
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_blobWork != null)
            {
                m_blobWork.Dispose();
                m_blobWork = null;
            }
        }

        /// <summary>
        /// Validate the specified blob.
        /// </summary>
        /// <param name="strLayerName">Specifies the layer name that owns the blob.</param>
        /// <param name="dir">Specifies the direction FWD or BWD.</param>
        /// <param name="b">Specifies the blob to verify.</param>
        /// <param name="bData">Optionally, specifies to validate the data values (default = true).</param>
        /// <param name="bDiff">Optionally, sppecifies to validate the diff values (default = false).</param>
        /// <param name="bThrowExceptions">Optionally, specifies to throw exceptions (default = true).</param>
        /// <returns>If the blob is valid, true is returned.</returns>
        /// <exception cref="Exception">When bThrowExceptions is true, an exception is thrown when detecting a NAN or INF.</exception>
        public bool Validate(string strLayerName, DIR dir, Blob<T> b, bool bData = true, bool bDiff = false, bool bThrowExceptions = true)
        {
#if DEBUG
            if (bData || bDiff)
                m_blobWork.ReshapeLike(b);

            if (bData)
            {
                Tuple<double, double, double, double> mm_data = b.minmax_data(m_blobWork, true);

                if (mm_data.Item3 > 0 || mm_data.Item4 > 0)
                {
                    if (!bThrowExceptions)
                        return false;
                    throw new Exception("NAN or INF detected in the BOTTOM '" + b.Name + "' Data for layer '" + strLayerName + "' on the " + dir.ToString() + " pass.");
                }
            }

            if (bDiff)
            {
                Tuple<double, double, double, double> mm_diff = b.minmax_diff(m_blobWork, true);

                if (mm_diff.Item3 > 0 || mm_diff.Item4 > 0)
                {
                    if (!bThrowExceptions)
                        return false;
                    throw new Exception("NAN or INF detected in the BOTTOM '" + b.Name + "' Diff for layer '" + strLayerName + "' on the " + dir.ToString() + " pass.");
                }
            }
#endif

            return true;
        }
    }
}
