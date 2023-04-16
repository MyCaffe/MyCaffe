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
/// The MyCaffe.layers.tft namespace contains all TFT related layers.
/// </summary>
namespace MyCaffe.layers.tft
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
        /// <param name="imgDb">Specifies an interface to the image database, who's use is optional.</param>
        /// <returns>If supported, the layer is returned, otherwise <i>null</i> is returned.</returns>
        public Layer<double> CreateDouble(CudaDnn<double> cuda, Log log, LayerParameter p, CancelEvent evtCancel, IXImageDatabaseBase imgDb)
        {
            switch (p.type)
            {
                case LayerParameter.LayerType.CATEGORICAL_TRANS:
                    return new CategoricalTransformationLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.CHANNEL_EMBEDDING:
                    return new ChannelEmbeddingLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.GATEADDNORM:
                    return new GateAddNormLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.GLU:
                    return new GluLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.GRN:
                    return new GrnLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.NUMERIC_TRANS:
                    return new NumericTransformationLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.MULTIHEAD_ATTENTION_INTERP:
                    return new MultiHeadAttentionInterpLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.VARSELNET:
                    return new VarSetNetLayer<double>(cuda, log, p);

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
        /// <param name="imgDb">Specifies an interface to the image database, who's use is optional.</param>
        /// <returns>If supported, the layer is returned, otherwise <i>null</i> is returned.</returns>
        public Layer<float> CreateSingle(CudaDnn<float> cuda, Log log, LayerParameter p, CancelEvent evtCancel, IXImageDatabaseBase imgDb)
        {
            switch (p.type)
            {
                case LayerParameter.LayerType.CATEGORICAL_TRANS:
                    return new CategoricalTransformationLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.CHANNEL_EMBEDDING:
                    return new ChannelEmbeddingLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.GATEADDNORM:
                    return new GateAddNormLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.GLU:
                    return new GluLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.GRN:
                    return new GrnLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.NUMERIC_TRANS:
                    return new NumericTransformationLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.MULTIHEAD_ATTENTION_INTERP:
                    return new MultiHeadAttentionInterpLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.VARSELNET:
                    return new VarSetNetLayer<float>(cuda, log, p);

                default:
                    return null;
            }
        }
    }
}
