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

namespace MyCaffe.layers.ssd
{
    /// <summary>
    /// The AnnotatedDataLayer provides  annotated data to the Net by assigning top Blobs directly.
    /// This layer is initialized with the MyCaffe.param.AnnotatedDataParameter.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class AnnotatedDataLayer<T> : BasePrefetchingDataLayer<T>
    {
        /// <summary>
        /// Specifies the database.
        /// </summary>
        protected DB m_db;
        /// <summary>
        /// Specifies the database used to traverse through the database.
        /// </summary>
        protected Cursor m_cursor;
        /// <summary>
        /// Specifies the annotation type used if any.
        /// </summary>
        protected SimpleDatum.ANNOTATION_TYPE m_AnnoType = SimpleDatum.ANNOTATION_TYPE.NONE;
        /// <summary>
        /// Specifies the list of batch samplers.
        /// </summary>
        List<BatchSampler> m_rgBatchSamplers = new List<BatchSampler>();
        /// <summary>
        /// Specifies the label map file.
        /// </summary>
        string m_strLabelMapFile;
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

        /// <summary>
        /// The AnnotatedDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter annotated_data_param.</param>
        /// <param name="db">Specifies the external database to use.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel any pre-fetching operations.</param>
        public AnnotatedDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXImageDatabase db, CancelEvent evtCancel) 
            : base(cuda, log, p, db, evtCancel)
        {
            m_type = LayerParameter.LayerType.ANNOTATED_DATA;

            Tuple<IMGDB_LABEL_SELECTION_METHOD, IMGDB_IMAGE_SELECTION_METHOD> kvSel = db.GetSelectionMethod();
            IMGDB_IMAGE_SELECTION_METHOD imgSel = kvSel.Item2;

            if (m_param.data_param.enable_pair_selection.HasValue)
            {
                if (m_param.data_param.enable_pair_selection.Value)
                    imgSel |= IMGDB_IMAGE_SELECTION_METHOD.PAIR;
                else
                    imgSel &= (~IMGDB_IMAGE_SELECTION_METHOD.PAIR);
            }

            if (m_param.data_param.enable_random_selection.HasValue)
            {
                if (m_param.data_param.enable_random_selection.Value)
                    imgSel |= IMGDB_IMAGE_SELECTION_METHOD.RANDOM;
                else
                    imgSel &= (~IMGDB_IMAGE_SELECTION_METHOD.RANDOM);
            }

            db.SetSelectionMethod(null, imgSel);

            m_db = new data.DB(db);
            m_db.Open(p.data_param.source);
            m_cursor = m_db.NewCursor();

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
            int nBatchSize = (int)m_param.data_param.batch_size;

            foreach (BatchSampler sampler in m_param.annotated_data_param.batch_sampler)
            {
                m_rgBatchSamplers.Add(sampler);
            }

            m_strLabelMapFile = m_param.annotated_data_param.label_map_file;

            // Make sure dimension is consistent within batch.
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
