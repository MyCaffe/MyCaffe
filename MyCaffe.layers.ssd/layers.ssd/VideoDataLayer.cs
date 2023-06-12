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
using WebCam;
using DirectX.Capture;
using System.Windows.Forms;
using System.Drawing;
using System.Threading;
using System.IO;

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
        object m_objSync = new object();
        WebCam.WebCam m_webcam = null;
        Bitmap m_bmpSnapshot = null;
        AutoResetEvent m_evtSnapshotReady = new AutoResetEvent(false);
        Filter m_filter = null;
        int m_nSkipFrames;
        int m_nVideoWidth;
        int m_nVideoHeight;
        List<int> m_rgTopShape;
        /// <summary>
        /// Specifies a first timer used to calcualte the batch time.
        /// </summary>
        protected Stopwatch m_swTimerBatch;
        /// <summary>
        /// Specfies a second timer used to calculate the transaction time.
        /// </summary>
        protected Stopwatch m_swTimerTransaction;
        /// <summary>
        /// Specifies the read time.
        /// </summary>
        protected double m_dfReadTime;
        /// <summary>
        /// Specifies the transaction time.
        /// </summary>
        protected double m_dfTransTime;
        private T[] m_rgTopData = null;
        private long m_lDuration = 0;

        /// <summary>
        /// The VideoDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter annotated_data_param.</param>
        /// <param name="db">Specifies the external database to use.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel any pre-fetching operations.</param>
        public VideoDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXDatabaseBase db, CancelEvent evtCancel) 
            : base(cuda, log, p, db, evtCancel)
        {
            m_type = LayerParameter.LayerType.VIDEO_DATA;

            if (p.data_param.display_timing)
            {
                m_swTimerBatch = new Stopwatch();
                m_swTimerTransaction = new Stopwatch();
            }
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            lock (m_objSync)
            {
                if (m_webcam != null)
                {
                    m_webcam.Close();
                    m_webcam.Dispose();
                    m_webcam = null;
                    m_filter = null;
                }
            }
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
        protected override void DataLayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Bitmap bmp = null;

            int nBatchSize = (int)m_param.data_param.batch_size;
            m_videoType = m_param.video_data_param.video_type;
            m_nSkipFrames = (int)m_param.video_data_param.skip_frames;
            m_nVideoWidth = (int)m_param.video_data_param.video_width;
            m_nVideoHeight = (int)m_param.video_data_param.video_height;

            // Read an image and use it to initialize the top blob.
            if (m_videoType == VideoDataParameter.VideoType.WEBCAM)
            {
                m_webcam = new WebCam.WebCam();
                m_webcam.OnSnapshot += m_webcam_OnSnapshot;

                // Default 'source' is a Video File.
                if (m_webcam.VideoInputDevices.Count == 0)
                    m_log.FAIL("Could not find a web-cam!");

                if (m_param.video_data_param.device_id >= m_webcam.VideoInputDevices.Count)
                    m_log.FAIL("The video device_id is greater than the number of web cam devices detected (" + m_webcam.VideoInputDevices.Count.ToString() + ").");

                m_filter = m_webcam.VideoInputDevices[m_param.video_data_param.device_id];
                m_log.WriteLine("Using web-cam '" + m_filter.Name + "' for video input.");

                m_webcam.Open(m_filter, null, null);
                m_webcam.GetImage();
                if (!m_evtSnapshotReady.WaitOne(1000))
                    m_log.FAIL("Failed to get a web-cam snapshot!");

                bmp = m_bmpSnapshot;
            }
            else if (m_videoType == VideoDataParameter.VideoType.VIDEO)
            {
                m_webcam = new WebCam.WebCam();
                m_webcam.OnSnapshot += m_webcam_OnSnapshot;

                if (!File.Exists(m_param.video_data_param.video_file))
                    m_log.FAIL("The video file '" + m_param.video_data_param.video_file + "' does not exist!");

                m_log.WriteLine("Using video source '" + m_param.video_data_param.video_file + "' for video input.");

                m_lDuration = m_webcam.Open(null, null, m_param.video_data_param.video_file);
                m_webcam.GetImage();
                if (!m_evtSnapshotReady.WaitOne(1000))
                    m_log.FAIL("Failed to get a video snapshot!");

                bmp = m_bmpSnapshot;
            }
            else
            {
                m_log.FAIL("Unknown video type!");
            }

            m_log.CHECK(bmp != null, "Could not load an image!");

            // Resize the image if needed.
            if (bmp.Width != m_nVideoWidth || bmp.Height != m_nVideoHeight)
            {
                Bitmap bmpNew = ImageTools.ResizeImage(bmp, m_nVideoWidth, m_nVideoHeight);
                bmp.Dispose();
                bmp = bmpNew;
            }

            // Use data_transformer to infer the expected blob shape from a bitmap.
            m_rgTopShape = m_transformer.InferBlobShape(3, bmp.Width, bmp.Height);
            m_rgTopShape[0] = nBatchSize;
            colTop[0].Reshape(m_rgTopShape);

            for (int i = 0; i < m_rgPrefetch.Length; i++)
            {
                m_rgPrefetch[i].Data.Reshape(m_rgTopShape);
            }

            m_log.WriteLine("Output data size: " + colTop[0].shape_string);

            // Label.
            if (m_bOutputLabels)
            {
                List<int> rgLabelShape = MyCaffe.basecode.Utility.Create<int>(1, nBatchSize);
                colTop[1].Reshape(rgLabelShape);

                for (int i = 0; i < m_rgPrefetch.Length; i++)
                {
                    m_rgPrefetch[i].Label.Reshape(rgLabelShape);
                }
            }
        }

        private void m_webcam_OnSnapshot(object sender, ImageArgs e)
        {
            if (m_bmpSnapshot != null)
                m_bmpSnapshot.Dispose();

            m_bmpSnapshot = e.Image;
            m_evtSnapshotReady.Set();
        }

        /// <summary>
        /// Load a batch of data in the background (this is run on an internal thread within the BasePrefetchingDataLayer class).
        /// </summary>
        /// <param name="batch">Specifies the Batch of data to load.</param>
        protected override void load_batch(Batch<T> batch)
        {
            m_log.CHECK(batch.Data.count() > 0, "There is no space allocated for data!");
            int nBatchSize = (int)m_param.data_param.batch_size;

            // Reshape batch according to the batch size.
            m_rgTopShape[0] = nBatchSize;
            batch.Data.Reshape(m_rgTopShape);

            T[] rgTopLabel = null;
            if (m_bOutputLabels)
                rgTopLabel = batch.Label.mutable_cpu_data;

            if (m_param.data_param.display_timing)
            {
                m_swTimerBatch.Restart();
                m_dfReadTime = 0;
                m_dfTransTime = 0;
            }

            int nSkipFrames = m_nSkipFrames;
            int nDim = 0;
            Bitmap bmp = null;

            for (int i = 0; i < nBatchSize; i++)
            {
                if (m_param.data_param.display_timing)
                    m_swTimerTransaction.Restart();

                lock (m_objSync)
                {
                    if (m_videoType == VideoDataParameter.VideoType.WEBCAM)
                    {
                        if (m_webcam == null)
                            return;

                        m_webcam.GetImage();
                        if (!m_evtSnapshotReady.WaitOne(1000))
                            m_log.FAIL("Failed to get web-cam snapshot!");

                        bmp = m_bmpSnapshot;
                    }
                    else if (m_videoType == VideoDataParameter.VideoType.VIDEO)
                    {
                        if (m_webcam == null)
                            return;

                        if (nSkipFrames > 0)
                            nSkipFrames--;

                        if (nSkipFrames > 0)
                            m_webcam.Step(nSkipFrames);

                        m_webcam.GetImage();

                        if (!m_evtSnapshotReady.WaitOne(1000))
                            m_log.FAIL("Failed to get video file snapshot!");

                        if (m_webcam.IsAtEnd)
                            m_webcam.SetPosition(0);

                        bmp = m_bmpSnapshot;
                    }
                    else
                    {
                        m_log.FAIL("Unknown video type!");
                    }

                    m_log.CHECK(bmp != null, "Could not load image!");

                    // Resize the image if needed.
                    if (bmp.Width != m_nVideoWidth || bmp.Height != m_nVideoHeight)
                    {
                        Bitmap bmpNew = ImageTools.ResizeImage(bmp, m_nVideoWidth, m_nVideoHeight);
                        bmp.Dispose();
                        bmp = bmpNew;
                    }

                    if (m_param.data_param.display_timing)
                    {
                        m_dfReadTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;
                        m_swTimerTransaction.Restart();
                    }

                    SimpleDatum datum = null;

                    if (i == 0)
                    {
                        datum = ImageData.GetImageDataD(bmp, batch.Data.channels, false, 0);

                        // Reshape according to the first datum of each batch
                        // on single input batches allows for inputs of varying dimension.
                        // Use data trabnsformer to infer the expected blob shape for datum.
                        List<int> rgTopShape = m_transformer.InferBlobShape(datum);

                        // Reshape batch according to the batch size.
                        rgTopShape[0] = nBatchSize;
                        batch.Data.Reshape(rgTopShape);

                        nDim = 1;
                        for (int k = 1; k < rgTopShape.Count; k++)
                        {
                            nDim *= rgTopShape[k];
                        }

                        int nTopLen = nDim * nBatchSize;
                        if (m_rgTopData == null || m_rgTopData.Length != nTopLen)
                            m_rgTopData = new T[nTopLen];
                    }

                    nSkipFrames = m_nSkipFrames;

                    if (datum == null)
                        datum = ImageData.GetImageDataD(bmp, batch.Data.channels, false, 0);

                    // Apply transformations (mirror, crop...) to the image.
                    T[] rgTrans = m_transformer.Transform(datum);
                    Array.Copy(rgTrans, 0, m_rgTopData, nDim * i, nDim);

                    // Copy label.
                    if (m_bOutputLabels)
                        rgTopLabel[i] = (T)Convert.ChangeType(datum.Label, typeof(T));

                    if (m_param.data_param.display_timing)
                        m_dfTransTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;
                }
            }

            batch.Data.SetCPUData(m_rgTopData);

            if (m_bOutputLabels)
                batch.Label.SetCPUData(rgTopLabel);

            if (m_param.data_param.display_timing)
            {
                m_swTimerBatch.Stop();
                m_swTimerTransaction.Stop();
                m_log.WriteLine("Prefetch batch: " + m_swTimerBatch.ElapsedMilliseconds.ToString() + " ms.", true);
                m_log.WriteLine("     Read time: " + m_dfReadTime.ToString() + " ms.", true);
                m_log.WriteLine("Transform time: " + m_dfTransTime.ToString() + " ms.", true);
            }
        }
    }
}
