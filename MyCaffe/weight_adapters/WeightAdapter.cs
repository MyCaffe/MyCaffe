using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.layers;
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
    /// Abstract base class for all weight adapters.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class WeightAdapter<T> : IDisposable
    {
        /// <summary>
        /// Specifies the weight blobs of the weight adapter.
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
        protected WeightAdapterParameter m_param;
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
        public WeightAdapter(CudaDnn<T> cuda, Log log, WeightAdapterParameter p)
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
        /// Attempts to share a parameter Blob if another parameter Blob with the same name and accpetable size is found.
        /// </summary>
        /// <param name="b">Specifies the Blob to share.</param>
        /// <param name="rgMinShape">Specifies the minimum shape requried to share.</param>
        /// <param name="bAllowEndsWithComparison">Optionally, allow name comparison where end of blob 'b' name is compared with the share blob names (default = false).</param>
        /// <returns>If the Blob is shared, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        protected bool shareParameter(Blob<T> b, List<int> rgMinShape, bool bAllowEndsWithComparison = false)
        {
            WeightAdapterParameterEx<T> paramEx = m_param as WeightAdapterParameterEx<T>;
            if (paramEx == null)
                return false;

            if (paramEx.SharedBlobs == null)
                return false;

            return paramEx.SharedBlobs.Share(b, rgMinShape, false, bAllowEndsWithComparison);
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
        public static WeightAdapter<T> Create(CudaDnn<T> cuda, Log log, WeightAdapterParameter p)
        {
            string strType = p.type.ToLower();

            if (strType == "lora")
                return new WeightAdapterLoRA<T>(cuda, log, p);

            throw new Exception("Unknown WeightAdapter type: " + strType);
        }

        /// <summary>
        /// Returns the output adapter parameters.
        /// </summary>
        public WeightAdapterParameter weight_adapter_param
        {
            get { return m_param; }
        }

        /// <summary>
        /// Specifies the weight blobs of the output adapter.
        /// </summary>
        public BlobCollection<T> blobs
        {
            get { return m_colBlobs; }
        }

        /// <summary>
        /// Setup the weight adapter. This method is called just after the layer Setup method is called.
        /// </summary>
        /// <param name="p">Specifies the layer parameters.</param>
        /// <param name="wt">Specifies the input data (which is the output of the layer's Forward call).</param>
        public abstract void Setup(LayerParameter p, Blob<T> wt);

        /// <summary>
        /// Reshape the weight adapter. This method is called just after the layer's Reshape is called.
        /// </summary>
        /// <param name="wt">Specifies the input data (which is the output of the layer's Forward call).</param>
        public abstract void Reshape(Blob<T> wt);

        /// <summary>
        /// Forward propagate the weight adapter. This method is called just after the layer's Forward is called and returns the new weights to use.
        /// </summary>
        /// <param name="wt">Specifies the input data (which is the output of the layer's Forward call).</param>
        /// <returns>A handle to the GPU memory of the altered weights of the same shape as the input 'wt' is returned.</returns>
        public abstract long Forward(Blob<T> wt);

        /// <summary>
        /// Backward propagate the weight adapter. This method is called just before the layer's Backward is called and returns the new weight gradients to use.
        /// </summary>
        /// <param name="colBtm">Specifies the input data (input to the Forward pass).</param>
        /// <param name="colTop">Specifies the output data (output by the Forward pass).</param>
        /// <param name="wt">Specifies the input gradients (which contains the grad before the Backward call).</param>
        /// <returns>A handle to the GPU memory of the altered gradients of the same shape as the input 'wt' is returned.</returns>
        public abstract long Backward(BlobCollection<T> colTop, BlobCollection<T> colBtm, Blob<T> wt);
    }

    /// <summary>
    /// The OutputAdapterParameterEx is used to pass the shared adapted blobs to the output adapter.
    /// </summary>
    /// <typeparam name="T">Specifies the double or float base type.</typeparam>
    public class WeightAdapterParameterEx<T> : WeightAdapterParameter
    {
        BlobCollection<T> m_colSharedBlobs = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="p">Specifies the original parameters.</param>
        /// <param name="colSharedBlobs">Specifies the shared adapted blobs.</param>
        public WeightAdapterParameterEx(WeightAdapterParameter p, BlobCollection<T> colSharedBlobs) : base(p)
        {
            m_colSharedBlobs = colSharedBlobs;
        }

        /// <summary>
        /// Returns the shared parameter Blobs.
        /// </summary>
        public BlobCollection<T> SharedBlobs
        {
            get { return m_colSharedBlobs; }
        }
    }
}
