using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.db.image;
using MyCaffe.param;

/// <summary>
/// The MyCaffe.layers.lnn namespace contains all Liquid Neural Network (LNN) related layers.
/// </summary>
namespace MyCaffe.layers.lnn
{
    /// <summary>
    /// The LayerFactor is responsible for creating all layers implemented in the MyCaffe.layers.tft namespace.
    /// </summary>
    public class LayerFactory : ILayerCreator
    {
        /// <summary>
        /// Create the layers when using the <i>double</i> base type.
        /// </summary>
        /// <param name="cuda">Specifies the connection to the low-level CUDA interfaces.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="p">Specifies the layer parameter.</param>
        /// <param name="evtCancel">Specifies the cancellation event.</param>
        /// <param name="db">Specifies an interface to the in-memory database, who's use is optional.</param>
        /// <returns>If supported, the layer is returned, otherwise <i>null</i> is returned.</returns>
        public Layer<double> CreateDouble(CudaDnn<double> cuda, Log log, LayerParameter p, CancelEvent evtCancel, IXDatabaseBase db)
        {
            switch (p.type)
            {
                case LayerParameter.LayerType.CFC:
                    return new CfcLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.CFC_UNIT:
                    return new CfcUnitLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.LTC_UNIT:
                    return new LtcUnitLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.LECUN:
                    return new LeCunLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.SILU:
                    return new SiLULayer<double>(cuda, log, p);

                case LayerParameter.LayerType.SOFTPLUS:
                    return new SoftPlusLayer<double>(cuda, log, p);

                default:
                    return null;
            }
        }

        /// <summary>
        /// Create the layers when using the <i>float</i> base type.
        /// </summary>
        /// <param name="cuda">Specifies the connection to the low-level CUDA interfaces.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="p">Specifies the layer parameter.</param>
        /// <param name="evtCancel">Specifies the cancellation event.</param>
        /// <param name="db">Specifies an interface to the in-memory database, who's use is optional.</param>
        /// <returns>If supported, the layer is returned, otherwise <i>null</i> is returned.</returns>
        public Layer<float> CreateSingle(CudaDnn<float> cuda, Log log, LayerParameter p, CancelEvent evtCancel, IXDatabaseBase db)
        {
            switch (p.type)
            {
                case LayerParameter.LayerType.CFC:
                    return new CfcLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.CFC_UNIT:
                    return new CfcUnitLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.LTC_UNIT:
                    return new LtcUnitLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.LECUN:
                    return new LeCunLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.SILU:
                    return new SiLULayer<float>(cuda, log, p);

                case LayerParameter.LayerType.SOFTPLUS:
                    return new SoftPlusLayer<float>(cuda, log, p);

                default:
                    return null;
            }
        }
    }
}
