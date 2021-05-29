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
/// The MyCaffe.layers.ssd namespace contains all SSD related layers.
/// </summary>
namespace MyCaffe.layers.beta
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
                case LayerParameter.LayerType.ACCURACY_DECODE:
                    return new AccuracyDecodeLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.ACCURACY_ENCODING:
                    return new AccuracyEncodingLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.ATTENTION:
                    return new AttentionLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.COPY:
                    return new CopyLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.DATA_SEQUENCE:
                    return new DataSequenceLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.DECODE:
                    return new DecodeLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.GATHER:
                    return new GatherLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.GRN:
                    return new GRNLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.KNN:
                    return new KnnLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.LSTM_ATTENTION:
                    return new LSTMAttentionLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.MERGE:
                    return new MergeLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.NORMALIZATION1:
                    return new Normalization1Layer<double>(cuda, log, p);

                case LayerParameter.LayerType.TEXT_DATA:
                    return new TextDataLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.TRANSPOSE:
                    return new TransposeLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.TRIPLET_LOSS:
                    return new TripletLossLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.SQUEEZE:
                    return new SqueezeLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.UNSQUEEZE:
                    return new UnsqueezeLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.UNPOOLING1:
                    return new UnPoolingLayer1<double>(cuda, log, p);

                case LayerParameter.LayerType.UNPOOLING:
                    return new UnPoolingLayer<double>(cuda, log, p);

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
                case LayerParameter.LayerType.ACCURACY_DECODE:
                    return new AccuracyDecodeLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.ACCURACY_ENCODING:
                    return new AccuracyEncodingLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.ATTENTION:
                    return new AttentionLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.COPY:
                    return new CopyLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.DATA_SEQUENCE:
                    return new DataSequenceLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.DECODE:
                    return new DecodeLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.GATHER:
                    return new GatherLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.GRN:
                    return new GRNLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.KNN:
                    return new KnnLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.LSTM_ATTENTION:
                    return new LSTMAttentionLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.MERGE:
                    return new MergeLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.NORMALIZATION1:
                    return new Normalization1Layer<float>(cuda, log, p);

                case LayerParameter.LayerType.TEXT_DATA:
                    return new TextDataLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.TRANSPOSE:
                    return new TransposeLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.TRIPLET_LOSS:
                    return new TripletLossLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.SQUEEZE:
                    return new SqueezeLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.UNSQUEEZE:
                    return new UnsqueezeLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.UNPOOLING1:
                    return new UnPoolingLayer1<float>(cuda, log, p);

                case LayerParameter.LayerType.UNPOOLING:
                    return new UnPoolingLayer<float>(cuda, log, p);

                default:
                    return null;
            }
        }
    }
}
