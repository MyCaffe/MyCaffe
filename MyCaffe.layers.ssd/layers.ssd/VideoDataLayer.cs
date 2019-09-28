using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.data;
using MyCaffe.db.image;
using MyCaffe.param;
using MyCaffe.param.ssd;

/// <summary>
/// The MyCaffe.layers.ssd namespace contains all SSD related layers.
/// </summary>
namespace MyCaffe.layers.ssd
{
    /// <summary>
    /// The VideoDataLayer provides data to the Net from a WebCam or Video file.
    /// This layer is initialized with the MyCaffe.param.VideoDataParameter.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class VideoDataLayer<T> : BasePrefetchingDataLayer<T>
    {
        VideoDataParameter.VideoType m_videoType;
        int m_nSkipFrames;
        int m_nTotalFrames;
        int m_nProcessedFrames;
        List<int> m_rgTopShape;

        /// <summary>
        /// The VideoDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter annotated_data_param.</param>
        /// <param name="db">Specifies the external database to use.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel any pre-fetching operations.</param>
        public VideoDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXImageDatabase db, CancelEvent evtCancel) 
            : base(cuda, log, p, db, evtCancel)
        {
            m_type = LayerParameter.LayerType.VIDEO_DATA;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /// <summary>
        /// No bottom blobs are used by this layer.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 0; }
        }

        /// <summary>
        /// Specifies the minimum number of required top (output) Blobs: data
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the DataLayer by starting up the pre-fetching.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void DataLayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_videoType = m_param.video_data_param.video_type;
            m_nSkipFrames = (int)m_param.video_data_param.skip_frames;

            // Read an image and use it to initialize the top blob.
            if (m_videoType == VideoDataParameter.VideoType.WEBCAM)
            {
            }
            else if (m_videoType == VideoDataParameter.VideoType.VIDEO)
            {
            }
            else
            {
                m_log.FAIL("Unknown video type!");
            }
        }

        /// <summary>
        /// Load a batch of data in the background (this is run on an internal thread within the BasePrefetchingDataLayer class).
        /// </summary>
        /// <param name="batch">Specifies the Batch of data to load.</param>
        protected override void load_batch(Batch<T> batch)
        {
        }
    }
}
