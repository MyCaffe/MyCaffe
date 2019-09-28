using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.db.image;
using MyCaffe.layers;
using MyCaffe.param;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe
{
    /// <summary>
    /// The ILayerCreator interface is implemented by each MyCaffe.layers.x layer extension dll and is used to create new instances
    /// of the layers supported by each layer extension dll.
    /// </summary>
    public interface ILayerCreator
    {
        /// <summary>
        /// Create the layers when using the <i>double</i> base type.
        /// </summary>
        /// <param name="cuda">Specifies the connection to the low-level CUDA interfaces.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="p">Specifies the layer parameter.</param>
        /// <param name="evtCancel">Specifies the cancellation event.</param>
        /// <param name="imgDb">Specifies an interface to the image database, who's use is optional.</param>
        /// <returns>If supported, the layer is returned, otherwise <i>null</i> is returned.</returns>
        Layer<double> CreateDouble(CudaDnn<double> cuda, Log log, LayerParameter p, CancelEvent evtCancel, IXImageDatabase imgDb);

        /// <summary>
        /// Create the layers when using the <i>float</i> base type.
        /// </summary>
        /// <param name="cuda">Specifies the connection to the low-level CUDA interfaces.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="p">Specifies the layer parameter.</param>
        /// <param name="evtCancel">Specifies the cancellation event.</param>
        /// <param name="imgDb">Specifies an interface to the image database, who's use is optional.</param>
        /// <returns>If supported, the layer is returned, otherwise <i>null</i> is returned.</returns>
        Layer<float> CreateSingle(CudaDnn<float> cuda, Log log, LayerParameter p, CancelEvent evtCancel, IXImageDatabase imgDb);
    }
}
