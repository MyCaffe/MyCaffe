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
/// The MyCaffe.layers.gpt namespace contains all GPT related layers.
/// </summary>
namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The LayerFactor is responsible for creating all layers implemented in the MyCaffe.layers.ssd namespace.
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
                case LayerParameter.LayerType.CAUSAL_SELF_ATTENTION:
                    return new CausalSelfAttentionLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.MULTIHEAD_ATTENTION:
                    return new MultiheadAttentionLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.POSITIONAL_ENCODING:
                    return new PositionalEncodingLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.GELU:
                    return new GeluLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.LAYERNORM:
                    return new LayerNormLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.TRANSFORMER_BLOCK:
                    return new TransformerBlockLayer<double>(cuda, log, p);
                    
                case LayerParameter.LayerType.TOKENIZED_DATA:
                    return new TokenizedDataLayer<double>(cuda, log, p, imgDb, evtCancel);

                case LayerParameter.LayerType.TOKENIZED_DATA_PAIRS:
                    return new TokenizedDataPairsLayer<double>(cuda, log, p, imgDb, evtCancel);

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
                case LayerParameter.LayerType.CAUSAL_SELF_ATTENTION:
                    return new CausalSelfAttentionLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.MULTIHEAD_ATTENTION:
                    return new MultiheadAttentionLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.POSITIONAL_ENCODING:
                    return new PositionalEncodingLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.GELU:
                    return new GeluLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.LAYERNORM:
                    return new LayerNormLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.TRANSFORMER_BLOCK:
                    return new TransformerBlockLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.TOKENIZED_DATA:
                    return new TokenizedDataLayer<float>(cuda, log, p, imgDb, evtCancel);

                case LayerParameter.LayerType.TOKENIZED_DATA_PAIRS:
                    return new TokenizedDataPairsLayer<float>(cuda, log, p, imgDb, evtCancel);

                default:
                    return null;
            }
        }
    }
}
