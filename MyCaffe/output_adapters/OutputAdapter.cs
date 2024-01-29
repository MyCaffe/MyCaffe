using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// The MyCaffe.output_adapters namespace contains all output adapters.
/// </summary>
namespace MyCaffe.output_adapters
{
    /// <summary>
    /// Abstract base class for all output adapters.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class OutputAdapter<T> : IDisposable
    {
        /// <summary>
        /// Specifies the weight blobs of the output adapter.
        /// </summary>
        protected BlobCollection<T> m_colBlobs = new BlobCollection<T>();
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
        protected OutputAdapterParameter m_param;
        /// <summary>
        /// Internal top blobs for internal layer calls.
        /// </summary>
        protected BlobCollection<T> m_colTop = new BlobCollection<T>();
        /// <summary>
        /// Internal bottom blobs for internal layer calls.
        /// </summary>
        protected BlobCollection<T> m_colBtm = new BlobCollection<T>();

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Instance of CudaDnn - connection to cuda.</param>
        /// <param name="log">Log used for output.</param>
        /// <param name="p">OutputAdapter parameter that defines the adapter settings.</param>
        public OutputAdapter(CudaDnn<T> cuda, Log log, OutputAdapterParameter p)
        {
            m_cuda = cuda;
            m_log = log;
            m_param = p;
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            dispose();
            if (m_colBlobs != null)
            {
                m_colBlobs.Dispose();
                m_colBlobs = null;
            }
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        protected virtual void dispose()
        {
        }

        /// <summary>
        /// Add the bottom and top blobs to the internal collections.
        /// </summary>
        /// <param name="blobBtm">Specifies the bottom input blob.</param>
        /// <param name="blobTop">Specifies the top input blob.</param>
        protected void addBtmTop(Blob<T> blobBtm, Blob<T> blobTop)
        {
            m_colBtm.Clear();
            m_colBtm.Add(blobBtm);
            m_colTop.Clear();
            m_colTop.Add(blobTop);
        }

        /// <summary>
        /// Create a new output adapter.
        /// </summary>
        /// <param name="cuda">Specifies the cuda connector to the low-level primitivs.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="p">Specifies the output adapter parameters.</param>
        /// <returns>An instance of a new output adapter is returned.</returns>
        /// <exception cref="Exception">An exception is returned if an unsupported output adapter type is specified in the parameters.</exception>
        public static OutputAdapter<T> Create(CudaDnn<T> cuda, Log log, OutputAdapterParameter p)
        {
            string strType = p.type.ToLower();

            if (strType == "lora")
                return new OutputAdapterLoRA<T>(cuda, log, p);

            throw new Exception("Unknown OutputAdapter type: " + strType);
        }

        /// <summary>
        /// Specifies the weight blobs of the output adapter.
        /// </summary>
        public BlobCollection<T> blobs
        {
            get { return m_colBlobs; }
        }

        /// <summary>
        /// Setup the output adapter. This method is called just after the layer Setup method is called.
        /// </summary>
        /// <param name="p">Specifies the layer parameters.</param>
        /// <param name="blobBottom">Specifies the input data.</param>
        /// <param name="blobTop">Specifies the output data.</param>
        public abstract void Setup(LayerParameter p, BlobCollection<T> blobBottom, BlobCollection<T> blobTop);

        /// <summary>
        /// Reshape the output adapter. This method is called just after the layer's Reshape is called.
        /// </summary>
        /// <param name="colBottom">Specifies the input data.</param>
        /// <param name="colTop">Specifies the output data.</param>
        public abstract void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop);

        /// <summary>
        /// Forward propagate the output adapter. This method is called just after the layer's Forward is called.
        /// </summary>
        /// <param name="colBottom">Specifies the input data (which is the output of the layer's Forward call).</param>
        /// <param name="colTop">Specifies the output data passed to the next layer.</param>
        public abstract void Forward(BlobCollection<T> colBottom, BlobCollection<T> colTop);

        /// <summary>
        /// Backward propagate the output adapter. This method is called just before the layer's Backward is called.
        /// </summary>
        /// <param name="colTop">Specifies the input gradients.</param>
        /// <param name="rgbPropagateDown">Specifies what gets propagated.</param>
        /// <param name="colBottom">Specifies the output gradients (which are then the input gradients to the layer's Backward call).</param>
        public abstract void Backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom);
    }
}
