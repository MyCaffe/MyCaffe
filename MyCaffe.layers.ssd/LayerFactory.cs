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
/// The MyCaffe.layers.ssd namespace contains all Single-Shot MultiBox (SSD) related layers.
/// </summary>
/// <remarks>
/// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
/// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
/// </remarks>
namespace MyCaffe.layers.ssd
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
                case LayerParameter.LayerType.ANNOTATED_DATA:
                    return new AnnotatedDataLayer<double>(cuda, log, p, db, evtCancel);

                case LayerParameter.LayerType.DETECTION_EVALUATE:
                    return new DetectionEvaluateLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.DETECTION_OUTPUT:
                    return new DetectionOutputLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.MULTIBOX_LOSS:
                    return new MultiBoxLossLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.NORMALIZATION2:
                    return new Normalization2Layer<double>(cuda, log, p);

                case LayerParameter.LayerType.PERMUTE:
                    return new PermuteLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.PRIORBOX:
                    return new PriorBoxLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.SMOOTHL1_LOSS:
                    return new SmoothL1LossLayer<double>(cuda, log, p);

                case LayerParameter.LayerType.VIDEO_DATA:
                    return new VideoDataLayer<double>(cuda, log, p, db, evtCancel);

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
                case LayerParameter.LayerType.ANNOTATED_DATA:
                    return new AnnotatedDataLayer<float>(cuda, log, p, db, evtCancel);

                case LayerParameter.LayerType.DETECTION_EVALUATE:
                    return new DetectionEvaluateLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.DETECTION_OUTPUT:
                    return new DetectionOutputLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.MULTIBOX_LOSS:
                    return new MultiBoxLossLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.NORMALIZATION2:
                    return new Normalization2Layer<float>(cuda, log, p);

                case LayerParameter.LayerType.PERMUTE:
                    return new PermuteLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.PRIORBOX:
                    return new PriorBoxLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.SMOOTHL1_LOSS:
                    return new SmoothL1LossLayer<float>(cuda, log, p);

                case LayerParameter.LayerType.VIDEO_DATA:
                    return new VideoDataLayer<float>(cuda, log, p, db, evtCancel);

                default:
                    return null;
            }
        }
    }
}
