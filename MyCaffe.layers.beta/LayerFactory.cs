﻿using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

/// <summary>
/// The MyCaffe.layers.beta namespace contains all beta stage layers.
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
        /// <param name="db">Specifies an interface to the in-memory database, who's use is optional.</param>
        /// <returns>If supported, the layer is returned, otherwise <i>null</i> is returned.</returns>
        public Layer<double> CreateDouble(CudaDnn<double> cuda, Log log, LayerParameter p, CancelEvent evtCancel, IXDatabaseBase db)
        {
            switch (p.type)
            {
                case LayerParameter.LayerType.ACCURACY_DECODE:
                    return new AccuracyDecodeLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.ACCURACY_ENCODING:
                    return new AccuracyEncodingLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.ACCURACY_REGRESSION:
                    return new AccuracyRegressionLayer<double>(cuda, log, p, db);

                case LayerParameter.LayerType.ATTENTION:
                    return new AttentionLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS:
                    return new BCEWithLogitsLossLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.CONVOLUTION_OCTAVE:
                    return new ConvolutionOctaveLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.COPY:
                    return new CopyLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.DATA_SEQUENCE:
                    return new DataSequenceLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.DECODE:
                    return new DecodeLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.GATHER:
                    return new GatherLayer<double>(cuda, log, p);
                    
                case LayerParameter.LayerType.GLOBRES_NORM:
                    return new GlobResNormLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.INTERP:
                    return new InterpLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.KNN:
                    return new KnnLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.LSTM_ATTENTION:
                    return new LSTMAttentionLayer<double>(cuda, log, p);
                    
                case LayerParameter.LayerType.MEAN_ERROR_LOSS:
                    return new MeanErrorLossLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.MERGE:
                    return new MergeLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.MISH:
                    return new MishLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.NORMALIZATION1:
                    return new Normalization1Layer<double>(cuda, log, p);

                case LayerParameter.LayerType.MODEL_DATA:
                    return new ModelDataLayer<double>(cuda, log, p, db, evtCancel);

                case LayerParameter.LayerType.PAIRWISE_ACCURACY:
                    return new PairwiseAccuracyLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.PAIRWISE_LOSS:
                    return new PairwiseLossLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.SPATIAL_ATTENTION:
                    return new SpatialAttentionLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.SERF:
                    return new SerfLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.SILU:
                    return new SiLULayer<double>(cuda, log, p);

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

                case LayerParameter.LayerType.Z_SCORE:
                    return new ZScoreLayer<double>(cuda, log, p, db);

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
                case LayerParameter.LayerType.ACCURACY_DECODE:
                    return new AccuracyDecodeLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.ACCURACY_ENCODING:
                    return new AccuracyEncodingLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.ACCURACY_REGRESSION:
                    return new AccuracyRegressionLayer<float>(cuda, log, p, db);

                case LayerParameter.LayerType.ATTENTION:
                    return new AttentionLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS:
                    return new BCEWithLogitsLossLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.CONVOLUTION_OCTAVE:
                    return new ConvolutionOctaveLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.COPY:
                    return new CopyLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.DATA_SEQUENCE:
                    return new DataSequenceLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.DECODE:
                    return new DecodeLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.GATHER:
                    return new GatherLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.GLOBRES_NORM:
                    return new GlobResNormLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.INTERP:
                    return new InterpLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.KNN:
                    return new KnnLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.LSTM_ATTENTION:
                    return new LSTMAttentionLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.MEAN_ERROR_LOSS:
                    return new MeanErrorLossLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.MERGE:
                    return new MergeLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.MISH:
                    return new MishLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.NORMALIZATION1:
                    return new Normalization1Layer<float>(cuda, log, p);

                case LayerParameter.LayerType.MODEL_DATA:
                    return new ModelDataLayer<float>(cuda, log, p, db, evtCancel);

                case LayerParameter.LayerType.PAIRWISE_ACCURACY:
                    return new PairwiseAccuracyLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.PAIRWISE_LOSS:
                    return new PairwiseLossLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.SPATIAL_ATTENTION:
                    return new SpatialAttentionLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.SERF:
                    return new SerfLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.SILU:
                    return new SiLULayer<float>(cuda, log, p);

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

                case LayerParameter.LayerType.Z_SCORE:
                    return new ZScoreLayer<float>(cuda, log, p, db);

                default:
                    return null;
            }
        }
    }
}
