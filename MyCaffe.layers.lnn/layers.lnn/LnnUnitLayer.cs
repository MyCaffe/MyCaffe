using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.lnn;

namespace MyCaffe.layers.lnn
{
    /// <summary>
    /// The LnnUnitLayer implements the base class to the Cfc and Ltc Unit layers. 
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public abstract class LnnUnitLayer<T> : Layer<T>
    {
        /// <summary>
        /// The LnnUnitLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public LnnUnitLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
        }

        /// <summary>
        /// Create the internal blobs used by the layer for a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <param name="cuda">Specifies the underlying CudaDnn low-level DLL.</param>
        /// <param name="log">Specifies the log.</param>
        /// <returns>The collection of created blobs is returned.</returns>
        public abstract BlobCollection<T> CreateInternalBlobs(int nIdx, CudaDnn<T> cuda, Log log);

        /// <summary>
        /// Set the internal blobs to a set of external blobs.
        /// </summary>
        /// <param name="col">Specifies the blob collection created using CreateInternalBlobs.</param>
        public abstract void SetInternalBlobs(BlobCollection<T> col);
    }
}
