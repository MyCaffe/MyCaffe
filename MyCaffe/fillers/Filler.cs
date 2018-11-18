using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.param;
using MyCaffe.common;

/// <summary>
/// The MyCaffe.fillers namespace contains all fillers including the Filler class.
/// </summary>
namespace MyCaffe.fillers
{
    /// <summary>
    /// Abstract Filler class used to fill blobs with values.
    /// </summary>
    /// <typeparam name="T">The base type <i>float</i> or <i>double</i>.</typeparam>
    public abstract class Filler<T>
    {
        /// <summary>
        /// Specifies the CudaDnn instance used to communicate to the low-level Cuda Dnn DLL.
        /// </summary>
        protected CudaDnn<T> m_cuda;
        /// <summary>
        /// Specifies the output log.
        /// </summary>
        protected Log m_log;
        /// <summary>
        /// Specifies the filler parameters.
        /// </summary>
        protected FillerParameter m_param;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Instance of CudaDnn - connection to cuda.</param>
        /// <param name="log">Log used for output.</param>
        /// <param name="p">Filler parameter that defines the filler settings.</param>
        public Filler(CudaDnn<T> cuda, Log log, FillerParameter p)
        {
            m_cuda = cuda;
            m_log = log;
            m_param = p;
        }

        /// <summary>
        /// Fill the blob with values based on the actual filler used.
        /// </summary>
        /// <param name="b">Specifies the blob to fill.</param>
        public void Fill(Blob<T> b)
        {
            int nNumChannels = (b.num_axes > 1) ? b.shape(1) : 1;
            int nHeight = (b.num_axes > 2) ? b.shape(2) : 1;
            int nWidth = (b.num_axes > 3) ? b.shape(3) : 1;
            Fill(b.count(), b.mutable_gpu_data, b.num_axes, b.shape(0), nNumChannels, nHeight, nWidth);
        }


        /// <summary>
        /// Fill the memory with values based on the actual filler used.
        /// </summary>
        /// <param name="nCount">Specifies the number of items to fill.</param>
        /// <param name="hMem">Specifies the handle to GPU memory to fill.</param>
        /// <param name="nNumAxes">Optionally, specifies the number of axes (default = 1).</param>
        /// <param name="nNumOutputs">Optionally, specifies the number of outputs (default = 1).</param>
        /// <param name="nNumChannels">Optionally, specifies the number of channels (default = 1).</param>
        /// <param name="nHeight">Optionally, specifies the height (default = 1).</param>
        /// <param name="nWidth">Optionally, specifies the width (default = 1).</param>
        public abstract void Fill(int nCount, long hMem, int nNumAxes = 1, int nNumOutputs = 1, int nNumChannels = 1, int nHeight = 1, int nWidth = 1);

        /// <summary>
        /// Create a new Filler instance.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance.</param>
        /// <param name="log">Specifies the log for output.</param>
        /// <param name="p">Specifies the filler parameter.</param>
        /// <returns></returns>
        public static Filler<T> Create(CudaDnn<T> cuda, Log log, FillerParameter p)
        {
            switch (p.type)
            {
                case "constant":
                    return new ConstantFiller<T>(cuda, log, p);

                case "gaussian":
                    return new GaussianFiller<T>(cuda, log, p);

                case "uniform":
                    return new UniformFiller<T>(cuda, log, p);

                case "positive_unitball":
                    return new PositiveUnitballFiller<T>(cuda, log, p);

                case "xavier":
                    return new XavierFiller<T>(cuda, log, p);

                case "msra":
                    return new MsraFiller<T>(cuda, log, p);

                case "bilinear":
                    return new BilinearFiller<T>(cuda, log, p);

                default:
                    log.FAIL("Unknown filler type: " + p.type);
                    return null;
            }
        }
    }
}
