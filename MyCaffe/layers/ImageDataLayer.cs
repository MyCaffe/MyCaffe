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
    /// The ImageDataLayer loads data from the image files located in the root directory specified.
    /// This layer is initialized with the MyCaffe.param.ImageDataParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ImageDataLayer<T> : BasePrefetchingDataLayer<T>
    {
        CryptoRandom m_random = new CryptoRandom();
        List<Tuple<string, int>> m_rgLines = new List<Tuple<string, int>>();
        int m_nLinesId = 0;
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

        /// <summary>
        /// The ImageDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter data_param</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel any pre-fetching operations.</param>
        public ImageDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, CancelEvent evtCancel)
            : base(cuda, log, p, null, evtCancel)
        {
            m_type = LayerParameter.LayerType.IMAGE_DATA;
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
        /// Specifies the exact number of top blobs as 2 for this layer: data, label.
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Shuffle the images so that they are loaded in a random order.
        /// </summary>
        protected virtual void shuffleImages()
        {
            List<Tuple<string, int>> rgLines = new List<Tuple<string, int>>();

            while (m_rgLines.Count > 0)
            {
                int nIdx = m_random.Next(m_rgLines.Count);
                rgLines.Add(m_rgLines[nIdx]);
                m_rgLines.RemoveAt(nIdx);

                if (m_rgLines.Count == 1)
                {
                    rgLines.Add(m_rgLines[0]);
                    m_rgLines.Clear();
                }
            }

            m_rgLines = rgLines;
        }

        private void loadFileList()
        {
            if (m_rgLines.Count > 0)
                return;

            // Read the file with filenames and labels.
            string strSource = m_param.image_data_param.source;
            m_log.WriteLine("INFO: Opening file '" + strSource + "'.");

            using (StreamReader sr = new StreamReader(strSource))
            {
                string strLine = sr.ReadLine();

                while (strLine != null)
                {
                    if (!string.IsNullOrEmpty(strLine))
                    {
                        int nPos = strLine.LastIndexOf(' ');
                        if (nPos < 0)
                        {
                            nPos = strLine.LastIndexOf(',');
                            if (nPos < 0)
                                nPos = strLine.LastIndexOf(';');
                        }

                        m_log.CHECK_GT(nPos, 0, "The separator character of ' ' or ',' or ';' could not be found!");
                        string strFile = strLine.Substring(0, nPos);
                        string strLabel = strLine.Substring(nPos + 1);
                        int nLabel = int.Parse(strLabel);

                        m_rgLines.Add(new Tuple<string, int>(strFile, nLabel));
                    }

                    strLine = sr.ReadLine();
                }
            }

            m_log.CHECK_GT(m_rgLines.Count, 0, "The file is empty!");
        }

        private string getRootFolder()
        {
            if (string.IsNullOrEmpty(m_param.image_data_param.root_folder))
                return "";

            return m_param.image_data_param.root_folder.TrimEnd('\\', '/') + "\\";
        }

        /// <summary>
        /// Allows any derivative classes to pre-initialize the m_src which is used in LayerSetup before the DataLayerSetup.
        /// </summary>
        /// <returns>When used this method should return <i>true</i>, otherwise <i>false</i> is returned by default.</returns>
        protected override bool setupSourceDescriptor()
        {
            string strRootFolder = getRootFolder();
            int nH = (int)m_param.image_data_param.new_height;
            int nW = (int)m_param.image_data_param.new_width;
            int nC = (m_param.image_data_param.is_color) ? 3 : 1;

            if (nH == 0 && nW == 0)
            {
                loadFileList();
                Datum datum = loadImage(strRootFolder, m_rgLines[0], m_param.image_data_param.is_color, nH, nW);
                nH = datum.height;
                nW = datum.width;
            }

            m_src = new basecode.descriptors.SourceDescriptor(0, "Internal", nW, nH, nC, false, false);

            return true;
        }

        /// <summary>
        /// Setup the ImageDataLayer by starting up the pre-fetching.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        protected override void DataLayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nBatchSize = (int)m_param.image_data_param.batch_size;
            int nNewHeight = (int)m_param.image_data_param.new_height;
            int nNewWidth = (int)m_param.image_data_param.new_width;
            bool bIsColor = m_param.image_data_param.is_color;
            string strRootFolder = getRootFolder();

            m_log.CHECK((nNewHeight == 0 && nNewWidth == 0) || (nNewHeight > 0 && nNewWidth > 0), "Current implementation requires new_height and new_width to be set at the same time.");

            // Read the file with filenames and labels.
            loadFileList();

            // Randomly shuffle the images.
            if (m_param.image_data_param.shuffle)
            {
                shuffleImages();
            }
            else if (m_param.image_data_param.rand_skip == 0)
            {
                LayerParameterEx<T> layer_param = m_param as LayerParameterEx<T>;
                if (layer_param != null && layer_param.solver_rank > 0)
                    m_log.WriteLine("WARNING: Shuffling or skipping recommended for multi-GPU.");
            }

            m_log.WriteLine("A total of " + m_rgLines.Count.ToString("N0") + " images.");

            m_nLinesId = 0;
            // Check if we would need to randomly skip a few data points.
            if (m_param.image_data_param.rand_skip > 0)
            {
                int nSkip = m_random.Next((int)m_param.image_data_param.rand_skip);
                m_log.WriteLine("Skipping first " + nSkip.ToString() + " data points.");
                m_log.CHECK_GT(m_rgLines.Count, nSkip, "Not enough data points to skip.");
                m_nLinesId = nSkip;
            }

            // Read an image and use it to initialize the top blob.
            Datum datum = loadImage(strRootFolder, m_rgLines[m_nLinesId], bIsColor, nNewHeight, nNewWidth);
            // Use data_transofrmer to infer the expected blob shape from the image.
            List<int> rgTopShape = m_transformer.InferBlobShape(datum);

            // Reshape colTop[0] and prefetch data according to the batch size.
            rgTopShape[0] = nBatchSize;
            colTop[0].Reshape(rgTopShape);

            for (int i = 0; i < m_rgPrefetch.Length; i++)
            {
                m_rgPrefetch[i].Data.Reshape(rgTopShape);
            }

            m_log.WriteLine("output data size: " + colTop[0].ToSizeString());

            // label.
            List<int> rgLabelShape = new List<int>() { nBatchSize };

            colTop[1].Reshape(rgLabelShape);

            for (int i = 0; i < m_rgPrefetch.Length; i++)
            {
                m_rgPrefetch[i].Label.Reshape(rgLabelShape);
            }
        }

        private Datum loadImage(string strRootFolder, Tuple<string, int> item, bool bIsColor, int nNewHeight, int nNewWidth)
        {
            string strFile = strRootFolder + item.Item1;
            int nLabel = m_rgLines[m_nLinesId].Item2;
            m_log.CHECK(File.Exists(strFile), "Could not find the file '" + strFile + "'!");
            Bitmap bmp = new Bitmap(strFile);

            // Resize the image if needed.
            if ((nNewWidth > 0 && nNewHeight > 0) && (bmp.Width != nNewWidth || bmp.Height != nNewHeight))
            {
                Bitmap bmpNew = ImageTools.ResizeImage(bmp, nNewWidth, nNewHeight);
                bmp.Dispose();
                bmp = bmpNew;
            }

            Datum data;
            int nChannels = (bIsColor) ? 3 : 1;

            if (typeof(T) == typeof(double))
                data = ImageData.GetImageDataD(bmp, nChannels, false, nLabel);
            else
                data = ImageData.GetImageDataF(bmp, nChannels, false, nLabel);

            bmp.Dispose();

            return data;
        }

        /// <summary>
        /// Load a batch of data in the background (this is run on an internal thread within the BasePrefetchingDataLayer class).
        /// </summary>
        /// <param name="batch">Specifies the Batch of data to load.</param>
        protected override void load_batch(Batch<T> batch)
        {
            m_log.CHECK(batch.Data.count() > 0, "There is no space allocated for data!");
            int nBatchSize = (int)m_param.image_data_param.batch_size;
            int nNewHeight = (int)m_param.image_data_param.new_height;
            int nNewWidth = (int)m_param.image_data_param.new_width;
            bool bIsColor = m_param.image_data_param.is_color;
            string strRootFolder = getRootFolder();

            T[] rgTopLabel = null;
            int nCount = batch.Label.count();
            m_log.CHECK_GT(nCount, 0, "The label count cannot be zero!");
            rgTopLabel = new T[nCount];

            if (m_param.image_data_param.display_timing)
            {
                m_swTimerBatch.Restart();
                m_dfReadTime = 0;
                m_dfTransTime = 0;
            }

            Datum datum;
            int nDim = 0;

            for (int i = 0; i < nBatchSize; i++)
            {
                if (m_param.data_param.display_timing)
                    m_swTimerTransaction.Restart();

                m_log.CHECK_GT(m_rgLines.Count, m_nLinesId, "The lines ID is too small!");
                datum = loadImage(strRootFolder, m_rgLines[m_nLinesId], bIsColor, nNewHeight, nNewWidth);

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

                // Apply data transformations (mirrow, scaling, crop, etc)
                int nDimCount = nDim;
                T[] rgTrans = m_transformer.Transform(datum);
                Array.Copy(rgTrans, 0, m_rgTopData, nDim * i, nDimCount);

                rgTopLabel[i] = (T)Convert.ChangeType(datum.Label, typeof(T));

                if (m_param.data_param.display_timing)
                    m_dfTransTime += m_swTimerTransaction.Elapsed.TotalMilliseconds;

                if (m_evtCancel.WaitOne(0))
                    return;

                batch.Data.SetCPUData(m_rgTopData);
                batch.Label.SetCPUData(rgTopLabel);

                if (m_param.data_param.display_timing)
                {
                    m_swTimerBatch.Stop();
                    m_swTimerTransaction.Stop();
                    m_log.WriteLine("Prefetch batch: " + m_swTimerBatch.ElapsedMilliseconds.ToString() + " ms.", true);
                    m_log.WriteLine("     Read time: " + m_dfReadTime.ToString() + " ms.", true);
                    m_log.WriteLine("Transform time: " + m_dfTransTime.ToString() + " ms.", true);
                }

                // Go to the next iter.
                m_nLinesId++;

                // We have reached the end, restart from the first.
                if (m_nLinesId == m_rgLines.Count)
                {
                    m_nLinesId = 0;
                    if (m_param.image_data_param.shuffle)
                        shuffleImages();
                }
            }
        }
    }
}
