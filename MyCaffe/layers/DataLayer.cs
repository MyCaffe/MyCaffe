using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Diagnostics;
using MyCaffe.basecode;
using MyCaffe.db.image;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.data;
using MyCaffe.fillers;
using System.IO;
using System.Drawing;

namespace MyCaffe.layers
{
    /// <summary>
    /// The DataLayer loads data from the IXImageDatabase database.
    /// This layer is initialized with the MyCaffe.param.DataParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DataLayer<T> : BasePrefetchingDataLayer<T>
    {
        /// <summary>
        /// Specifies the database.
        /// </summary>
        protected DB<T> m_db;
        /// <summary>
        /// Specifies the database used to traverse through the database.
        /// </summary>
        protected Cursor<T> m_cursor;
        UInt64 m_nOffset = 0;
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
        private T[] m_rgTopLabel = null;
        private int[] m_rgTopShape = null;
        private bool m_bMatchingCycle = true;
        private Datum m_datumNoise = null;
        private LabelCollection m_rgBatchLabels = null;
        private Blob<T> m_blobMask1 = null;
        private Blob<T> m_blobMask = null;
        private Blob<T> m_blobDebug1 = null;
        private int m_nIteration = 0;
        private int m_nBatchCount = 0;
        private SimpleDatum[] m_rgDatum = null;

        /// <summary>
        /// This event fires (only when set) each time a batch is loaded form this dataset.
        /// </summary>
        public event EventHandler<LastBatchLoadedArgs> OnBatchLoad;

        /// <summary>
        /// The DataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter data_param</param>
        /// <param name="db">Specifies the external database to use.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel any pre-fetching operations.</param>
        public DataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXImageDatabaseBase db, CancelEvent evtCancel)
            : base(cuda, log, p, db, evtCancel)
        {
            m_type = LayerParameter.LayerType.DATA;

            if (p.data_param.synchronize_target)
                m_rgBatchLabels = new LabelCollection();

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

            m_db = new data.DB<T>(db);
            m_db.Open(p.data_param.source);

            if (p.data_param.display_timing)
            {
                m_swTimerBatch = new Stopwatch();
                m_swTimerTransaction = new Stopwatch();
            }

            if (m_param.transform_param.mask_param != null && m_param.transform_param.mask_param.Active)
            {
                m_blobMask = new Blob<T>(cuda, log, false);
                m_blobMask1 = new Blob<T>(cuda, log, false);
            }

            if (m_param.data_param.enable_debug_output)
                m_blobDebug1 = new Blob<T>(cuda, log, false);
        }

        /// <summary>
        /// The preStop override is called just before stopping the internal thread managed by the base class.
        /// </summary>
        protected override void preStop()
        {
            base.preStop();

            if (m_rgBatchLabels != null)
                m_rgBatchLabels.Cancel();
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            if (m_rgBatchLabels != null)
            {
                m_rgBatchLabels.Dispose();
                m_rgBatchLabels = null;
            }

            if (m_blobMask != null)
            {
                m_blobMask.Dispose();
                m_blobMask = null;
            }

            if (m_blobMask1 != null)
            {
                m_blobMask1.Dispose();
                m_blobMask1 = null;
            }

            if (m_blobDebug1 != null)
            {
                m_blobDebug1.Dispose();
                m_blobDebug1 = null;
            }
        }

        /// <summary>
        /// Specifies to delay the prefetch when using a synchronized Data Layer.
        /// </summary>
        protected override bool delayPrefetch
        {
            get
            {
                if (m_param.data_param.synchronize_with != null || m_param.data_param.synchronize_target)
                    return true;

                return false;
            }
        }

        /// <summary>
        /// The Connect method connects one Data Layer to another so that they can synchronize.
        /// </summary>
        /// <param name="src">Specifies the source Data Layer whos OnBatchLoad event fires and
        /// is handled by this Data Layer.
        /// </param>
        public void Connect(DataLayer<T> src)
        {
            src.OnBatchLoad += Src_OnBatchLoad;
            m_log.WriteLine("DataLayer '" + m_param.name + "' is now connected to DataLayer '" + src.m_param.name + "'.");

            statupPrefetch();
            src.statupPrefetch();
        }

        /// <summary>
        /// Disconnect any previously connected Data Layers.
        /// </summary>
        public void Disconnect()
        {
            m_rgBatchLabels.Cancel();
        }

        private void Src_OnBatchLoad(object sender, LastBatchLoadedArgs e)
        {
            int nWait = m_rgBatchLabels.WaitProcessing;
            if (nWait == 0)
                return;

            m_rgBatchLabels.Set(e.Labels);
        }

        /// <summary>
        /// Setup the DataLayer by starting up the pre-fetching.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected override void DataLayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nBatchSize = (int)m_param.data_param.batch_size;
            bool bLoadDataCriteria = false;
            m_bMatchingCycle = true;

            m_cursor = m_db.NewCursor(m_transformer, (m_param.data_param.output_image_information) ? m_log : null);

            if (m_bOutputLabels && m_param.data_param.label_type == DataParameter.LABEL_TYPE.MULTIPLE)
                bLoadDataCriteria = true;

            // Read a data point, and use it to initialize the top blob.
            SimpleDatum datum = m_cursor.GetValue(null, bLoadDataCriteria, IMGDB_IMAGE_SELECTION_METHOD.NONE);

            // Use data transformer to infer the expected blob shape from the datum.
            m_rgTopShape = m_transformer.InferBlobShape(datum, m_rgTopShape);

            // When using noise as the secondary image, fill it with noise.
            if (m_param.data_param.enable_noise_for_nonmatch)
                m_datumNoise = createNoisyData(m_rgTopShape, datum);
                
            // Double the channels when loading image pairs where the first image is loaded followed by the second on the channel.
            if (m_param.data_param.images_per_blob > 1)
            {
                m_log.CHECK_EQ(m_param.data_param.images_per_blob, 2, "Currently images_per_blob only supports 2 image pairing and must be set to 2 for image pairing.");
                m_rgTopShape[1] *= m_param.data_param.images_per_blob;
            }

            // Reshape colTop[0] and prefetch data according to the batch size.
            m_rgTopShape[0] = nBatchSize;
            colTop[0].Reshape(m_rgTopShape);

            for (int i = 0; i < m_rgPrefetch.Length; i++)
            {
                m_rgPrefetch[i].Data.Reshape(m_rgTopShape);
            }

            m_log.WriteLine("output data size: " + colTop[0].ToSizeString());

            // Fill out the masks, if used.
            if (m_param.transform_param.mask_param != null && m_param.transform_param.mask_param.Active)
                createMasks(Utility.Clone<int>(m_rgTopShape));

            // Reshape debug data, if used.
            if (m_param.data_param.enable_debug_output)
                createDebug(Utility.Clone<int>(m_rgTopShape));

            // Label
            if (m_bOutputLabels)
            {
                List<int> rgLabelShape = new List<int>() { nBatchSize };

                // When using multi-labels, resize to batch x the number of multiple 
                // labels per image.
                if (m_param.data_param.label_type == DataParameter.LABEL_TYPE.MULTIPLE)
                {
                    if (m_param.data_param.images_per_blob > 1)
                        m_log.FAIL("Image pairing per blob currently only supports the " + DataParameter.LABEL_TYPE.SINGLE.ToString() + " label type.");

                    if (datum.DataCriteria == null || datum.DataCriteria.Length == 0)
                        m_log.FAIL("Could not find the multi-label data.  The data source '" + m_param.data_param.source + "' does not appear to have any Image Criteria data.");

                    if (datum.DataCriteriaFormat == SimpleDatum.DATA_FORMAT.LIST_DOUBLE)
                    {
                        List<double> rg = BinaryData.UnPackDoubleList(datum.DataCriteria, datum.DataCriteriaFormat);
                        int nLen = rg.Count;
                        rgLabelShape.Add(nLen);
                        rgLabelShape.Add(1);
                        rgLabelShape.Add(1);
                    }
                    else if (datum.DataCriteriaFormat == SimpleDatum.DATA_FORMAT.LIST_FLOAT)
                    {
                        List<float> rg = BinaryData.UnPackFloatList(datum.DataCriteria, datum.DataCriteriaFormat);
                        int nLen = rg.Count;
                        rgLabelShape.Add(nLen);
                        rgLabelShape.Add(1);
                        rgLabelShape.Add(1);
                    }
                    else
                    {
                        // Get the number of items and the item size from the end of the data.
                        int nLen = BitConverter.ToInt32(datum.DataCriteria, datum.DataCriteria.Length - (sizeof(int) * 4));
                        int nItemSize = BitConverter.ToInt32(datum.DataCriteria, datum.DataCriteria.Length - (sizeof(int) * 3));

                        rgLabelShape.Add(nLen);
                        m_log.CHECK_EQ(nItemSize, 1, "Currently only byte sized labels are supported in multi-label scenarios.");
                    }
                }
                else 
                {
                    int nChannels = 1;

                    // 1 label comparison + each label in pair order to which they were added to the blob.
                    if (m_param.data_param.images_per_blob > 1)
                    {
                        if (m_param.data_param.output_all_labels)
                            nChannels = m_param.data_param.images_per_blob;
                    }

                    rgLabelShape.Add(nChannels);
                }

                colTop[1].Reshape(rgLabelShape);

                for (int i = 0; i < m_rgPrefetch.Length; i++)
                {
                    m_rgPrefetch[i].Label.Reshape(rgLabelShape);
                }
            }

            m_nIteration = 0;
            m_nBatchCount = 0;
        }

        private void createMasks(int[] rgTopShape)
        {
            int nImgPerBlob = m_param.data_param.images_per_blob;
            m_blobMask.Reshape(rgTopShape);
            rgTopShape[0] = 1;
            rgTopShape[1] /= nImgPerBlob;
            m_blobMask1.Reshape(rgTopShape);
            m_blobMask1.SetData(1);
            float[] rgData = convertF(m_blobMask1.update_cpu_data());
            m_transformer.MaskData(rgTopShape, rgData);
            m_blobMask1.mutable_cpu_data = convert(rgData);

            int nDim = m_blobMask1.count();
            int nOffset = 0;
            for (int n = 0; n < m_blobMask.num; n++)
            {
                for (int j = 0; j < nImgPerBlob; j++)
                {
                    m_cuda.copy(nDim, m_blobMask1.gpu_data, m_blobMask.mutable_gpu_data, 0, nOffset);
                    nOffset += nDim;
                }
            }
        }

        private void createDebug(int[] rgTopShape)
        {
            int nImgPerBlob = m_param.data_param.images_per_blob;
            rgTopShape[0] = 1;
            rgTopShape[1] /= nImgPerBlob;
            m_blobDebug1.Reshape(rgTopShape);
        }

        private Datum createNoisyData(int[] rgTopShape, SimpleDatum datum)
        {
            Blob<T> blobNoise = new Blob<T>(m_cuda, m_log, rgTopShape);

            try
            {
                Filler<T> filler = Filler<T>.Create(m_cuda, m_log, m_param.data_param.data_noise_param.noise_filler);
                filler.Fill(blobNoise);

                if (m_param.data_param.data_noise_param.use_noisy_mean)
                {
                    SimpleDatum sdMean = m_transformer.ImageMean;
                    if (sdMean == null)
                    {
                        if (m_imgdb == null)
                            m_log.FAIL("No 'mean' image is loaded, yet the MyCaffe Image Database = null, and it is required to get the mean image.");

                        sdMean = m_imgdb.GetImageMean(m_src.ID);
                    }

                    if (sdMean == null)
                        m_log.FAIL("The data source '" + m_src.Name + "' does not have a mean image!");

                    if (sdMean.IsRealData)
                    {
                        blobNoise.mutable_cpu_diff = sdMean.GetData<T>();
                    }
                    else
                    {
                        double dfMin = blobNoise.min_data;
                        double dfMax = blobNoise.max_data;

                        if (dfMin < -1.0 || dfMax > 1.0)
                            m_log.WriteLine("WARNING! The noise filler is producing numbers outside of the range [-1,1] which may cause a saturated final noise data image.");

                        blobNoise.mutable_cpu_diff = sdMean.GetData<T>();
                    }

                    m_cuda.mul(blobNoise.count(), blobNoise.gpu_diff, blobNoise.gpu_data, blobNoise.mutable_gpu_data);
                }

                Datum datumNoise;
                if (typeof(T) == typeof(double))
                {
                    double[] rgdf = convertD(blobNoise.update_cpu_data());

                    if (datum.IsRealData)
                    {
                        List<double> rgdf1 = new List<double>(rgdf);
                        datumNoise = new Datum(datum.IsRealData, datum.Channels, datum.Width, datum.Height, m_param.data_param.data_noise_param.noise_data_label, DateTime.MinValue, rgdf1, 1, false, -1);
                    }
                    else
                    {
                        List<byte> rgb = rgdf.Select(p => Math.Min((byte)p, (byte)255)).ToList();
                        datumNoise = new Datum(datum.IsRealData, datum.Channels, datum.Width, datum.Height, m_param.data_param.data_noise_param.noise_data_label, DateTime.MinValue, rgb, 1, false, -1);
                    }
                }
                else
                {
                    float[] rgf = convertF(blobNoise.update_cpu_data());

                    if (datum.IsRealData)
                    {
                        List<float> rgf1 = new List<float>(rgf);
                        datumNoise = new Datum(datum.IsRealData, datum.Channels, datum.Width, datum.Height, m_param.data_param.data_noise_param.noise_data_label, DateTime.MinValue, rgf1, 1, false, -1);
                    }
                    else
                    {
                        List<byte> rgb = rgf.Select(p => Math.Min((byte)p, (byte)255)).ToList();
                        datumNoise = new Datum(datum.IsRealData, datum.Channels, datum.Width, datum.Height, m_param.data_param.data_noise_param.noise_data_label, DateTime.MinValue, rgb, 1, false, -1);
                    }
                }

                if (!string.IsNullOrEmpty(m_param.data_param.data_noise_param.noisy_save_path))
                {
                    if (!Directory.Exists(m_param.data_param.data_noise_param.noisy_save_path))
                        m_log.FAIL("The noisy save path '" + m_param.data_param.data_noise_param.noisy_save_path + "' does not exist!");

                    string strPath = m_param.data_param.data_noise_param.noisy_save_path.TrimEnd('\\');
                    Bitmap bmp = ImageData.GetImage(datumNoise);
                    bmp.Save(m_param.data_param.data_noise_param.noisy_save_path + "\\noisy_data.png");
                    bmp.Dispose();
                }

                return datumNoise;
            }
            finally
            {
                blobNoise.Dispose();
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
        /// Specifies the maximum number of required top (output) Blobs: data, label
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 2; }
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
        /// Provides a final processing step that takes place at the end of the base class forward = this is where we apply the mask if one exists and is enabled.
        /// </summary>
        /// <param name="blobTop">Specifies the top blob just about to be set out the forward operation as the Top[0] blob.</param>
        protected override void final_process(Blob<T> blobTop)
        {
            if (m_param.transform_param.mask_param != null && m_param.transform_param.mask_param.Active)
            {
                int nCount = blobTop.count();
                m_log.CHECK_EQ(m_blobMask.count(), nCount, "The mask must be the same size as the top!");
                m_cuda.mul(nCount, m_blobMask.gpu_data, blobTop.gpu_data, blobTop.mutable_gpu_data);
            }

            if (m_param.data_param.enable_debug_output)
            {
                if (m_nIteration < m_param.data_param.data_debug_param.iterations)
                {
                    int nImgPerBlob = m_param.data_param.images_per_blob;
                    int nChannels = blobTop.channels / nImgPerBlob;

                    if (blobTop.num_axes < 4)
                    { 
                        m_log.WriteLine("WARNING! debug output only supported blobs with 4 or more axes.");
                        return;
                    }

                    string strPath = m_param.data_param.data_debug_param.debug_save_path;
                    if (!Directory.Exists(strPath))
                        Directory.CreateDirectory(strPath);

                    strPath = strPath.TrimEnd('\\');
                    int nDim = nChannels * blobTop.height * blobTop.width;
                    int nCount = m_blobDebug1.count();
                    int nOffset = 0;

                    m_log.CHECK_EQ(nDim, nCount, "The debug data is not sized properly.");

                    for (int n = 0; n < blobTop.num; n++)
                    {
                        for (int j = 0; j < nImgPerBlob; j++)
                        {
                            m_cuda.copy(nCount, blobTop.gpu_data, m_blobDebug1.mutable_gpu_data, nOffset, 0);
                            nOffset += nCount;

                            if (m_param.transform_param.scale != 1)
                            {
                                double dfUnScale = 1.0 / m_param.transform_param.scale;
                                m_blobDebug1.scale_data(dfUnScale);
                            }

                            float[] rgData = convertF(m_blobDebug1.mutable_cpu_data);

                            string strFile = strPath + "\\dbgimg_iter_" + m_nIteration.ToString() + "_num_" + n.ToString() + "_img_" + j.ToString();
                            SimpleDatum sd = new SimpleDatum(nChannels, blobTop.width, blobTop.height, rgData, 0, nDim, false);

                            if (nChannels == 1 || nChannels == 3)
                            {
                                Bitmap bmp = ImageData.GetImage(sd);
                                bmp.Save(strFile + ".png");
                                bmp.Dispose();
                            }
                        }
                    }
                }
            }

            m_nIteration++;
        }

        /// <summary>
        /// Load a batch of data in the background (this is run on an internal thread within the BasePrefetchingDataLayer class).
        /// </summary>
        /// <param name="batch">Specifies the Batch of data to load.</param>
        protected override void load_batch(Batch<T> batch)
        {
            m_log.CHECK(batch.Data.count() > 0, "There is no space allocated for data!");
            int nBatchSize = (int)m_param.data_param.batch_size;
            bool bLoadDataCriteria = false;

            if (m_bOutputLabels && m_param.data_param.label_type == DataParameter.LABEL_TYPE.MULTIPLE)
                bLoadDataCriteria = true;

            if (m_bOutputLabels)
            {
                int nCount = batch.Label.count();
                m_log.CHECK_GT(nCount, 0, "The label count cannot be zero!");

                if (m_rgTopLabel == null || m_rgTopLabel.Length < nCount)
                    m_rgTopLabel = new T[nCount];
            }

            if (m_param.data_param.display_timing)
            {
                m_swTimerBatch.Restart();
                m_dfReadTime = 0;
                m_dfTransTime = 0;
            }

            SimpleDatum datum;
            int nDim = 0;
            List<int> rgLabels = null;
            List<int> rgTargetLabels = null;

            if (OnBatchLoad != null)
                rgLabels = new List<int>();

            // If we are synced with another dataset, wait for it to load the initial data set.
            if (m_param.data_param.synchronize_target)
            {
                m_log.CHECK_EQ(m_param.data_param.images_per_blob, 1, "DataLayer synchronize targets are not supported when loading more than 1 image per blob.");

                int nWait = m_rgBatchLabels.WaitReady;
                if (nWait == 0)
                    return;

                rgTargetLabels = m_rgBatchLabels.Get();
                m_log.CHECK_EQ(nBatchSize, m_rgBatchLabels.Count, "The batch label count (previously loaded by the primary dataset) does not match the batch size '" + m_param.data_param.batch_size.ToString() + "' of this layer!");
            }

            for (int i = 0; i < nBatchSize; i++)
            {
                if (m_param.data_param.display_timing)
                    m_swTimerTransaction.Restart();

                while (Skip())
                {
                    if (m_evtCancel.WaitOne(0))
                        return;
                    Next();
                }

                if (rgTargetLabels == null)
                {
                    datum = m_cursor.GetValue(null, bLoadDataCriteria);

                    if (m_param.data_param.images_per_blob > 1)
                    {
                        if (m_rgDatum == null || m_rgDatum.Length != m_param.data_param.images_per_blob - 1)
                            m_rgDatum = new SimpleDatum[m_param.data_param.images_per_blob - 1];

                        for (int j = 0; j < m_param.data_param.images_per_blob - 1; j++)
                        {
                            Next();

                            while (Skip())
                            {
                                if (m_evtCancel.WaitOne(0))
                                    return;
                                Next();
                            }

                            if (m_param.data_param.balance_matches)
                            {
                                if (m_bMatchingCycle)
                                {
                                    m_rgDatum[j] = getNextPair(true, datum, bLoadDataCriteria);
                                }
                                else
                                {
                                    if (m_param.data_param.enable_noise_for_nonmatch)
                                        m_rgDatum[j] = m_datumNoise;
                                    else
                                        m_rgDatum[j] = getNextPair(false, datum, bLoadDataCriteria);
                                }
                            }
                            else
                            {
                                if (m_param.data_param.enable_noise_for_nonmatch)
                                    m_rgDatum[j] = m_datumNoise;
                                else
                                    m_rgDatum[j] = m_cursor.GetValue(null, bLoadDataCriteria);
                            }
                        }

                        m_bMatchingCycle = !m_bMatchingCycle;
                    }
                }
                else
                {
                    datum = m_cursor.GetValue(rgTargetLabels[i], bLoadDataCriteria);
                }

                // When debug output is enabled, output information each image loaded.
                if (m_param.data_param.enable_debug_output)
                {
                    saveImageInfo(m_param.data_param.data_debug_param, datum, i, 0);

                    if (m_rgDatum != null)
                    {
                        for (int n = 0; n < m_rgDatum.Length; n++)
                        {
                            saveImageInfo(m_param.data_param.data_debug_param, m_rgDatum[n], i, n + 1);
                        }
                    }
                }

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
                    m_rgTopShape = m_transformer.InferBlobShape(datum, m_rgTopShape);

                    // Double the channels when loading image pairs where the first image is loaded followed by the second on the channel.
                    if (m_rgDatum != null)
                        m_rgTopShape[1] *= (m_rgDatum.Length + 1);

                    // Reshape batch according to the batch size.
                    m_rgTopShape[0] = nBatchSize;
                    batch.Data.Reshape(m_rgTopShape);

                    nDim = 1;
                    for (int k = 1; k < m_rgTopShape.Length; k++)
                    {
                        nDim *= m_rgTopShape[k];
                    }

                    int nTopLen = nDim * nBatchSize;
                    if (m_rgTopData == null || m_rgTopData.Length != nTopLen)
                        m_rgTopData = new T[nTopLen];
                }

                // Apply data transformations (mirrow, scaling, crop, etc)
                int nDimCount = nDim;

                if (m_rgDatum != null)
                    nDimCount /= (m_rgDatum.Length + 1);

                T[] rgTrans = m_transformer.Transform(datum);
                Array.Copy(rgTrans, 0, m_rgTopData, nDim * i, nDimCount);

                // When using load_image_pairs, stack the additional images right after the first.
                if (m_rgDatum != null)
                {
                    for (int j = 0; j < m_rgDatum.Length; j++)
                    {
                        rgTrans = m_transformer.Transform(m_rgDatum[j]);
                        int nOffset = (nDim * i) + (nDimCount * (j + 1));
                        Array.Copy(rgTrans, 0, m_rgTopData, nOffset, nDimCount);
                    }
                }

                // Copy label.
                if (m_bOutputLabels)
                {
                    if (m_param.data_param.label_type == DataParameter.LABEL_TYPE.MULTIPLE)
                    {
                        if (m_param.data_param.images_per_blob > 1)
                            m_log.FAIL("Loading image pairs (images_per_blob > 1) currently only supports the " + DataParameter.LABEL_TYPE.SINGLE.ToString() + " label type.");

                        if (m_param.transform_param.label_mapping.Active)
                            m_log.FAIL("Label mapping is not supported on labels of type 'MULTIPLE'.");

                        if (datum.DataCriteria == null || datum.DataCriteria.Length == 0)
                            m_log.FAIL("Could not find the multi-label data.  The data source '" + m_param.data_param.source + "' does not appear to have any Image Criteria data.");

                        // Get the number of items and the item size from the end of the data.
                        int nLen = BitConverter.ToInt32(datum.DataCriteria, datum.DataCriteria.Length - (sizeof(int) * 4));
                        int nItemSize = BitConverter.ToInt32(datum.DataCriteria, datum.DataCriteria.Length - (sizeof(int) * 3));
                        int nDstIdx = i * nLen;

                        m_log.CHECK_EQ(nItemSize, 1, "Currently only byte sized labels are supported in multi-label scenarios.");
                        Array.Copy(datum.DataCriteria, 0, m_rgTopLabel, nDstIdx, nLen);
                    }
                    else
                    {
                        // When using image pairs, the label is set to 1 when the labels are the same and 0 when they are different.
                        if (m_rgDatum != null)
                        {
                            if (m_rgDatum.Length == 1)
                            {
                                int nLabelDim = 1;

                                if (m_param.data_param.output_all_labels)
                                    nLabelDim = m_param.data_param.images_per_blob;

                                if (m_param.data_param.output_all_labels)
                                {
                                    int nLabel = datum.Label;
                                    if (m_param.data_param.forced_primary_label >= 0)
                                        nLabel = m_param.data_param.forced_primary_label;

                                    m_rgTopLabel[i * nLabelDim] = (T)Convert.ChangeType(nLabel, typeof(T));

                                    for (int j = 0; j < m_rgDatum.Length; j++)
                                    {
                                        m_rgTopLabel[i * nLabelDim + 1 + j] = (T)Convert.ChangeType(m_rgDatum[j].Label, typeof(T));
                                    }
                                }
                                else
                                {
                                    if (datum.Label == m_rgDatum[0].Label)
                                        m_rgTopLabel[i * nLabelDim] = m_tOne;
                                    else
                                        m_rgTopLabel[i * nLabelDim] = m_tZero;
                                }
                            }
                            else
                                m_log.FAIL("Currently image pairing only supports up to 2 images per blob.");
                        }
                        else
                        {
                            m_rgTopLabel[i] = (T)Convert.ChangeType(datum.Label, typeof(T));
                        }
                    }
                }

                if (m_param.data_param.display_timing)
                    m_dfTransTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;

                if (rgLabels != null)
                    rgLabels.Add(datum.Label);

                Next();

                if (m_evtCancel.WaitOne(0))
                    return;
            }

            m_nBatchCount++;
            batch.Data.SetCPUData(m_rgTopData);

            if (m_bOutputLabels)
                batch.Label.SetCPUData(m_rgTopLabel);

            if (m_param.data_param.display_timing)
            {
                m_swTimerBatch.Stop();
                m_swTimerTransaction.Stop();
                m_log.WriteLine("Prefetch batch: " + m_swTimerBatch.ElapsedMilliseconds.ToString() + " ms.", true);
                m_log.WriteLine("     Read time: " + m_dfReadTime.ToString() + " ms.", true);
                m_log.WriteLine("Transform time: " + m_dfTransTime.ToString() + " ms.", true);
            }

            if (m_param.data_param.synchronize_target)
            {
                if (m_rgBatchLabels != null)
                    m_rgBatchLabels.Done();
            }

            if (OnBatchLoad != null)
                OnBatchLoad(this, new LastBatchLoadedArgs(rgLabels));
        }

        private SimpleDatum getNextPair(bool bMatching, SimpleDatum d, bool bLoadDataCriteria)
        {
            int nRetries = 10;
            int nIdx = 0;
            SimpleDatum dNew = null;
            string strType = null;

            if (bMatching)
            {
                int? nLabel = null;

                if (d.Label == d.OriginalLabel)
                    nLabel = d.Label;

                dNew = m_cursor.GetValue(nLabel, bLoadDataCriteria);

                while (dNew.Label != d.Label && nIdx < nRetries)
                {
                    Next();
                    dNew = m_cursor.GetValue(nLabel, bLoadDataCriteria);
                    nIdx++;
                }

                if (dNew.Label != d.Label)
                    strType = "match";
            }
            else
            {
                dNew = m_cursor.GetValue(null, bLoadDataCriteria);

                while (dNew.Label == d.Label && nIdx < nRetries)
                {
                    Next();
                    dNew = m_cursor.GetValue(null, bLoadDataCriteria);
                    nIdx++;
                }

                if (dNew.Label == d.Label)
                    strType = "non-match";
            }

            if (strType != null)
                m_log.WriteLine("WARNING: The secondary pairing " + strType + " could not be found after " + nRetries.ToString() + " retries!");

            return dNew;
        }

        private void saveImageInfo(DataDebugParameter p, SimpleDatum d, int nNum, int nImg)
        {
            string strFile = p.debug_save_path.TrimEnd('\\') + "\\dbgimg_iter_" + m_nBatchCount.ToString() + "_num_" + nNum.ToString() + "_img_" + nImg.ToString();
            d.SaveInfo(strFile + ".txt");
        }
    }

    class LabelCollection : IDisposable /** @private */
    {
        ManualResetEvent m_evtReady = new ManualResetEvent(false);
        ManualResetEvent m_evtDone = new ManualResetEvent(true);
        AutoResetEvent m_evtCancel = new AutoResetEvent(false);
        List<int> m_rgLabels = new List<int>();
        object m_sync = new object();

        public LabelCollection()
        {
        }

        public void Dispose()
        {
            if (m_evtReady != null)
            {
                m_evtReady.Dispose();
                m_evtReady = null;
            }

            if (m_evtDone != null)
            {
                m_evtDone.Dispose();
                m_evtDone = null;
            }

            if (m_evtCancel != null)
            {
                m_evtCancel.Dispose();
                m_evtCancel = null;
            }
        }

        public void Cancel()
        {
            m_evtCancel.Set();
        }

        public int WaitReady
        {
            get
            {
                List<WaitHandle> rgWait = new List<WaitHandle>() { m_evtCancel, m_evtReady };
                return WaitHandle.WaitAny(rgWait.ToArray());
            }
        }

        public int WaitProcessing
        {
            get
            {
                List<WaitHandle> rgWait = new List<WaitHandle>() { m_evtCancel, m_evtDone };
                return WaitHandle.WaitAny(rgWait.ToArray());
            }
        }

        public void Done()
        {
            m_evtDone.Set();
        }

        public void Set(List<int> rg)
        {
            lock (m_sync)
            {
                m_rgLabels = new List<int>(rg);
                m_evtReady.Set();
            }
        }

        public int Count
        {
            get
            {
                lock (m_sync)
                {
                    return m_rgLabels.Count;
                }
            }
        }

        public List<int> Get()
        {
            lock (m_sync)
            {
                List<int> rg = new List<int>(m_rgLabels);
                m_evtReady.Reset();
                m_evtDone.Reset();
                return rg;
            }
        }
    }


    /// <summary>
    /// Specifies the arguments sent to the OnBatchLoad event used when synchronizing between Data Layers.
    /// </summary>
    public class LastBatchLoadedArgs : EventArgs
    {
        List<int> m_rgLabels;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="rgLabels">Specifies the labels loaded.</param>
        public LastBatchLoadedArgs(List<int> rgLabels)
        {
            m_rgLabels = new List<int>(rgLabels);
        }

        /// <summary>
        /// Returns the labels loaded.
        /// </summary>
        public List<int> Labels
        {
            get { return m_rgLabels; }
        }
    }
}
