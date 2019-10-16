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
        UInt64 m_nOffset = 0;
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
        private T[] m_rgTopData = null;
        SsdSampler<T> m_sampler = null;
        CryptoRandom m_random = null;

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
            m_random = new CryptoRandom(true, p.transform_param.random_seed.GetValueOrDefault(0));

            if (db == null)
                m_log.FAIL("Currently, the AnnotatedDataLayer requires the MyCaffe Image Database!");

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

            if (!db.GetLoadImageDataCriteria())
                m_log.WriteError(new Exception("The 'Load Image Data Criteria' must be set to TRUE in order to load the Annotation data."));

            db.SetSelectionMethod(null, imgSel);

            m_db = new data.DB(db);
            m_db.Open(p.data_param.source);
            m_cursor = m_db.NewCursor();

            if (p.data_param.display_timing)
            {
                m_swTimerBatch = new Stopwatch();
                m_swTimerTransaction = new Stopwatch();
            }

            m_sampler = new SsdSampler<T>(cuda, log);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            if (m_sampler != null)
            {
                m_sampler.Dispose();
                m_sampler = null;
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
        public override void DataLayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nBatchSize = (int)m_param.data_param.batch_size;

            foreach (BatchSampler sampler in m_param.annotated_data_param.batch_sampler)
            {
                m_rgBatchSamplers.Add(sampler);
            }

            m_strLabelMapFile = m_param.annotated_data_param.label_map_file;

            // Make sure dimension is consistent within batch.
            if (m_param.transform_param.resize_param != null && m_param.transform_param.resize_param.Active)
            {
                if (m_param.transform_param.resize_param.resize_mode == ResizeParameter.ResizeMode.FIT_SMALL_SIZE)
                    m_log.CHECK_EQ(nBatchSize, 1, "The FIT_MSALL_SIZE resize mode only supports a batch size of 1.");
            }

            // Read a data point, and use it to initialize the top blob.
            Datum anno_datum = m_cursor.GetValue();

            // Use data_transformer to infer the expected blob shape from anno_datum.
            List<int> rgTopShape = m_transformer.InferBlobShape(anno_datum);
            // Reshape top[0] and prefetch_data according to the batch_size.
            rgTopShape[0] = nBatchSize;
            colTop[0].Reshape(rgTopShape);

            for (int i = 0; i < m_rgPrefetch.Length; i++)
            {
                m_rgPrefetch[i].Data.Reshape(rgTopShape);
            }

            m_log.WriteLine("Output data size: " + colTop[0].ToSizeString());

            // Label
            if (m_bOutputLabels)
            {
                bool bHasAnnoType = (anno_datum.annotation_type != SimpleDatum.ANNOTATION_TYPE.NONE) && (m_param.annotated_data_param.anno_type != SimpleDatum.ANNOTATION_TYPE.NONE);
                List<int> rgLabelShape = Utility.Create<int>(4, 1);

                if (bHasAnnoType)
                {
                    m_AnnoType = anno_datum.annotation_type;

                    // If anno_type is provided in AnnotatedDataParameter, replace the type stored
                    // in each individual AnnotatedDatum.
                    if (m_param.annotated_data_param.anno_type != SimpleDatum.ANNOTATION_TYPE.NONE)
                    {
                        m_log.WriteLine("WARNING: Annotation type stored in AnnotatedDatum is shadowed.");
                        m_AnnoType = m_param.annotated_data_param.anno_type;
                    }

                    // Infer the label shape from anno_dataum.AnnotationGroup().
                    int nNumBboxes = 0;

                    // Since the number of bboxes can be different for each image,
                    // we store the bbox information in a specific format. In specific:
                    // All bboxes are stored in one spatial plane (num and channels are 1)
                    // and each row contains one and only one box in the following format:
                    // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
                    // Note: Refer to caffe.proto for details about group_label and
                    // instance_id.
                    if (m_AnnoType == SimpleDatum.ANNOTATION_TYPE.BBOX)
                    {
                        for (int g = 0; g < anno_datum.annotation_group.Count; g++)
                        {
                            nNumBboxes += anno_datum.annotation_group[g].annotations.Count;
                        }

                        rgLabelShape[0] = 1;
                        rgLabelShape[1] = 1;
                        // BasePrefetchingDataLayer.LayerSetup() requires to call
                        // cpu_data and gpu_data for consistent prefetch thread, thus
                        // we must make sure there is at least one bbox.
                        rgLabelShape[2] = Math.Max(nNumBboxes, 1);
                        rgLabelShape[3] = 8;
                    }
                    else
                    {
                        m_log.FAIL("Unknown annotation type.");
                    }
                }
                else
                {
                    rgLabelShape[0] = nBatchSize;
                }

                colTop[1].Reshape(rgLabelShape);

                for (int i = 0; i < m_rgPrefetch.Length; i++)
                {
                    m_rgPrefetch[i].Label.Reshape(rgLabelShape);
                }
            }
        }

        /// <summary>
        /// Retrieves the next item from the database and rolls the cursor over once the end 
        /// of the dataset is reached.
        /// </summary>
        protected void Next()
        {
            m_cursor.Next();

            if (!m_cursor.IsValid)
            {
                m_log.WriteLine("Restarting data prefetching from start.");
                m_cursor.SeekToFirst();
            }

            m_nOffset++;
        }

        /// <summary>
        /// Skip to the next value - used when training in a multi-GPU scenario.
        /// </summary>
        /// <returns></returns>
        protected bool Skip()
        {
            UInt64 nSize = (UInt64)m_param.solver_count;
            UInt64 nRank = (UInt64)m_param.solver_rank;
            // In test mode, only rank 0 runs, so avoid skipping.
            bool bKeep = (m_nOffset % nSize) == nRank || m_param.phase == Phase.TEST;

            return !bKeep;
        }

        /// <summary>
        /// Load a batch of data in the background (this is run on an internal thread within the BasePrefetchingDataLayer class).
        /// </summary>
        /// <param name="batch">Specifies the Batch of data to load.</param>
        protected override void load_batch(Batch<T> batch)
        {
            m_log.CHECK(batch.Data.count() > 0, "There is no space allocated for data!");

            if (m_param.data_param.display_timing)
            {
                m_swTimerBatch.Restart();
                m_dfReadTime = 0;
                m_dfTransTime = 0;
            }

            T[] rgTopLabel = null;

            if (m_bOutputLabels)
                rgTopLabel = batch.Label.mutable_cpu_data;

            // Reshape according to the first anno_datum of each batch
            // ont single input batches allows for inputs of varying dimension.
            int nBatchSize = (int)m_param.data_param.batch_size;
            Datum datum;
            int nDim = 0;
            int nNumBboxes = 0;
            Dictionary<int, List<AnnotationGroup>> rgAllAnno = new Dictionary<int, List<AnnotationGroup>>();
            List<int> rgTopShape = null;

            for (int i = 0; i < nBatchSize; i++)
            {
                if (m_param.data_param.display_timing)
                    m_swTimerTransaction.Restart();

                while (Skip())
                {
                    Next();
                }

                datum = m_cursor.GetValue();

                if (m_param.data_param.display_timing)
                {
                    m_dfReadTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;
                    m_swTimerTransaction.Restart();
                }

                if (i == 0)
                {
                    // Reshape according to the first datum of each batch
                    // on single input batches allows for inputs of varying dimension.
                    // Use data transformer to infer the expected blob shape for datum.
                    rgTopShape = m_transformer.InferBlobShape(datum);
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

                SimpleDatum distort_datum = null;
                SimpleDatum expand_datum = null;

                if (m_param.transform_param.distortion_param != null && m_param.transform_param.distortion_param.Active)
                {
                    distort_datum = m_transformer.DistortImage(datum);

                    if (m_param.transform_param.expansion_param != null && m_param.transform_param.expansion_param.Active)
                        expand_datum = m_transformer.ExpandImage(distort_datum);
                    else
                        expand_datum = distort_datum;
                }
                else if (m_param.transform_param.expansion_param != null && m_param.transform_param.expansion_param.Active)
                {
                    expand_datum = m_transformer.ExpandImage(datum);
                }
                else
                {
                    expand_datum = datum;
                }

                SimpleDatum sampled_datum = expand_datum;

                // Generate sampled bboxes from expand_datum.
                if (m_rgBatchSamplers.Count > 0)
                {
                    List<NormalizedBBox> rgSampledBboxes = m_sampler.GenerateBatchSamples(expand_datum, m_rgBatchSamplers);

                    // Randomly pick a sampled bbox and crop the expand_datum.
                    if (rgSampledBboxes.Count > 0)
                    {
                        int nIdx = m_random.Next(rgSampledBboxes.Count);
                        sampled_datum = m_transformer.CropImage(expand_datum, rgSampledBboxes[nIdx]);
                    }
                }

                m_log.CHECK(sampled_datum != null, "The sampled datum cannot be null!");
                List<int> rgShape = m_transformer.InferBlobShape(sampled_datum);

                if (m_param.transform_param.resize_param != null && m_param.transform_param.resize_param.Active)
                {
                    if (m_param.transform_param.resize_param.resize_mode == ResizeParameter.ResizeMode.FIT_SMALL_SIZE)
                        batch.Data.Reshape(rgShape);
                }

                // Apply data transformations (mirror, scale, crop...)
                int nOffset = batch.Data.offset(i);
                List<AnnotationGroup> rgTransformedAnnoVec;

                if (m_bOutputLabels)
                {
                    if (m_AnnoType != SimpleDatum.ANNOTATION_TYPE.NONE)
                    {
                        // Make sure all data have same annoation type.
                        if (m_param.annotated_data_param.anno_type != SimpleDatum.ANNOTATION_TYPE.NONE)
                            sampled_datum.annotation_type = m_AnnoType;
                        else
                            m_log.CHECK_EQ((int)m_AnnoType, (int)sampled_datum.annotation_type, "The sampled datum has a different AnnoationType!");

                        // Transform datum and annotation_group at the same time.
                        bool bMirror;
                        T[] rgTrans = m_transformer.Transform(sampled_datum, out rgTransformedAnnoVec, out bMirror);
                        Array.Copy(rgTrans, 0, m_rgTopData, nDim * i, nDim);

                        // Count the number of bboxes.
                        if (m_AnnoType == SimpleDatum.ANNOTATION_TYPE.BBOX)
                        {
                            for (int g = 0; g < rgTransformedAnnoVec.Count; g++)
                            {
                                nNumBboxes += rgTransformedAnnoVec[g].annotations.Count;
                            }
                        }
                        else
                        {
                            m_log.FAIL("Unknown annotation type.");
                        }

                        rgAllAnno.Add(i, rgTransformedAnnoVec);
                    }
                    else
                    {
                        T[] rgTrans = m_transformer.Transform(sampled_datum);
                        Array.Copy(rgTrans, 0, m_rgTopData, nDim * i, nDim);

                        // Otherwise, store the label from datum.
                        if (m_param.data_param.label_type == DataParameter.LABEL_TYPE.MULTIPLE)
                        {
                            if (datum.DataCriteria == null || datum.DataCriteria.Length == 0)
                                m_log.FAIL("Could not find the multi-label data.  The data source '" + m_param.data_param.source + "' does not appear to have any Image Criteria data.");

                            // Get the number of items and the item size from the end of the data.
                            int nLen = BitConverter.ToInt32(datum.DataCriteria, datum.DataCriteria.Length - (sizeof(int) * 4));
                            int nItemSize = BitConverter.ToInt32(datum.DataCriteria, datum.DataCriteria.Length - (sizeof(int) * 3));
                            int nDstIdx = i * nLen;

                            m_log.CHECK_EQ(nItemSize, 1, "Currently only byte sized labels are supported in multi-label scenarios.");
                            Array.Copy(datum.DataCriteria, 0, rgTopLabel, nDstIdx, nLen);
                        }
                        else
                        {
                            rgTopLabel[i] = (T)Convert.ChangeType(datum.Label, typeof(T));
                        }
                    }
                }

                if (m_param.data_param.display_timing)
                    m_dfTransTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;

                Next();
            }

            batch.Data.SetCPUData(m_rgTopData);

            if (m_bOutputLabels)
                batch.Label.SetCPUData(rgTopLabel);

            // Store 'rich' annotation if needed.
            if (m_bOutputLabels && m_AnnoType != SimpleDatum.ANNOTATION_TYPE.NONE)
            {
                List<int> rgLabelShape = Utility.Create<int>(4, 1);

                if (m_AnnoType == SimpleDatum.ANNOTATION_TYPE.BBOX)
                {
                    rgLabelShape[0] = 1;
                    rgLabelShape[1] = 1;
                    rgLabelShape[3] = 8;

                    if (nNumBboxes == 0)
                    {
                        // Store all -1 in the label.
                        rgLabelShape[2] = 1;
                        batch.Label.Reshape(rgLabelShape);
                        T[] rgLabel = new T[batch.Label.count()];

                        T tNegOne = (T)Convert.ChangeType(-1, typeof(T));
                        for (int i = 0; i < rgLabel.Length; i++)
                        {
                            rgLabel[i] = tNegOne;
                        }

                        batch.Label.SetCPUData(rgLabel);
                    }
                    else
                    {
                        // Reshape the label and store the annotation.
                        rgLabelShape[2] = nNumBboxes;
                        batch.Label.Reshape(rgLabelShape);
                        float[] rgTopLabel1 = convertF(batch.Label.mutable_cpu_data);

                        int nIdx = 0;
                        for (int i = 0; i < nBatchSize; i++)
                        {
                            List<AnnotationGroup> rgAnnoGroups = rgAllAnno[i];

                            for (int g = 0; g < rgAnnoGroups.Count; g++)
                            {
                                AnnotationGroup anno_group = rgAnnoGroups[g];

                                for (int a = 0; a < anno_group.annotations.Count; a++)
                                {
                                    Annotation anno = anno_group.annotations[a];
                                    NormalizedBBox bbox = anno.bbox;

                                    rgTopLabel1[nIdx] = i;
                                    nIdx++;
                                    rgTopLabel1[nIdx] = anno_group.group_label;
                                    nIdx++;
                                    rgTopLabel1[nIdx] = anno.instance_id;
                                    nIdx++;
                                    rgTopLabel1[nIdx] = bbox.xmin;
                                    nIdx++;
                                    rgTopLabel1[nIdx] = bbox.ymin;
                                    nIdx++;
                                    rgTopLabel1[nIdx] = bbox.xmax;
                                    nIdx++;
                                    rgTopLabel1[nIdx] = bbox.ymax;
                                    nIdx++;
                                    rgTopLabel1[nIdx] = (bbox.difficult) ? 1 : 0;
                                    nIdx++;
                                }
                            }
                        }

                        batch.Label.SetCPUData(convert(rgTopLabel1));
                    }
                }
                else
                {
                    m_log.FAIL("Unknown annotation type.");
                }
            }

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
