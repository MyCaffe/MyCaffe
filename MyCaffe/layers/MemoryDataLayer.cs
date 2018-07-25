using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Diagnostics;
using MyCaffe.basecode;
using MyCaffe.imagedb;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.data;

namespace MyCaffe.layers
{
    /// <summary>
    /// The MemoryDataLayer provides data to the Net from memory.
    /// This layer is initialized with the MyCaffe.param.MemoryDataParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class MemoryDataLayer<T> : BaseDataLayer<T>
    {
        int m_nBatchSize;
        int m_nChannels;
        int m_nHeight;
        int m_nWidth;
        int m_nLabelChannels;
        int m_nLabelHeight;
        int m_nLabelWidth;
        int m_nDataSize;
        int m_nLabelSize;
        Blob<T> m_blobData;
        Blob<T> m_blobLabel;
        bool m_bHasNewData;
        int m_nPos = 0;
        int m_nN = 1;

        public event EventHandler<MemoryDataLayerGetDataArgs> OnGetData;

        /// <summary>
        /// The BaseDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type MEMORYDATA with memorydata_param options:
        ///   - batch_size. The batch size of the data.
        ///   
        ///   - channels. The number of channels in the data.
        ///   
        ///   - height. The height of the data.
        ///   
        ///   - width. The width of the data.
        /// </param>
        public MemoryDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p, null)
        {
            m_type = LayerParameter.LayerType.MEMORYDATA;
            m_blobData = new Blob<T>(cuda, log);
            m_blobLabel = new Blob<T>(cuda, log);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /// <summary>
        /// Setup the MemoryDataLayer.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void DataLayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nBatchSize = (int)m_param.memory_data_param.batch_size;
            m_nChannels = (int)m_param.memory_data_param.channels;
            m_nHeight = (int)m_param.memory_data_param.height;
            m_nWidth = (int)m_param.memory_data_param.width;
            m_nDataSize = m_nChannels * m_nHeight * m_nWidth;

            m_log.CHECK_GT(m_nBatchSize * m_nDataSize, 0, "batch_size, channels, height, and width must be specified and positive in memory_data_param.");

            m_nLabelChannels = (int)m_param.memory_data_param.label_channels;
            m_nLabelHeight = (int)m_param.memory_data_param.label_height;
            m_nLabelWidth = (int)m_param.memory_data_param.label_width;
            m_nLabelSize = m_nLabelChannels * m_nLabelHeight * m_nLabelWidth;

            if (m_param.memory_data_param.label_type == LayerParameterBase.LABEL_TYPE.MULTIPLE)
                m_log.CHECK_GT(m_nBatchSize * m_nLabelSize, 0, "batch_size, label_channels, label_height, and label_width must be specified and positive in memory_data_param when using label_type = MULTIPLE.");

            if (OnGetData != null)
                OnGetData(this, new MemoryDataLayerGetDataArgs(true));

            m_log.CHECK_EQ(m_nLabelChannels, m_blobLabel.channels, "The actual label channels (" + m_blobLabel.channels.ToString() + ") do not match the 'memory_data_param.label_channels' setting of " + m_nLabelChannels.ToString() + ".");
            m_log.CHECK_EQ(m_nLabelHeight, m_blobLabel.height, "The actual label channels (" + m_blobLabel.height.ToString() + ") do not match the 'memory_data_param.label_channels' setting of " + m_nLabelHeight.ToString() + ".");
            m_log.CHECK_EQ(m_nLabelWidth, m_blobLabel.width, "The actual label channels (" + m_blobLabel.width.ToString() + ") do not match the 'memory_data_param.label_channels' setting of " + m_nLabelWidth.ToString() + ".");

            colTop[0].Reshape(m_nBatchSize, m_nChannels, m_nHeight, m_nWidth);
            colTop[1].Reshape(m_nBatchSize, m_nLabelChannels, m_nLabelHeight, m_nLabelWidth);

            m_blobData.update_cpu_data();
            m_blobLabel.update_cpu_data();
        }

        /// <summary>
        /// No bottom blobs are used by this layer.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 0; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs.
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// This method is used to add a list of Datums to the memory.
        /// </summary>
        /// <param name="rgData">The list of Data Datums to add.</param>
        public virtual void AddDatumVector(List<Datum> rgData)
        {
            m_log.CHECK(!m_bHasNewData, "Can't add data until current data has been consumed.");
            int nNum = rgData.Count;
            m_log.CHECK_GT(nNum, 0, "There are no datum to add.");
            m_log.CHECK_EQ(nNum % m_nBatchSize, 0, "The added data must be a multiple of the batch size.");

            m_blobData.Reshape(nNum, m_nChannels, m_nHeight, m_nWidth);

            // Reshape label blob depending on label type.
            if (m_param.memory_data_param.label_type == LayerParameterBase.LABEL_TYPE.MULTIPLE)
            {
                List<float> rgLbl = BinaryData.UnPackFloatList(rgData[0].DataCriteria, rgData[0].DataCriteriaFormat);
                m_blobLabel.Reshape(nNum, rgLbl.Count, 1, 1);
            }
            else
            {
                m_blobLabel.Reshape(nNum, 1, 1, 1);
            }

            // Apply data transformations (mirror, scale, crop...)
            m_transformer.Transform(rgData, m_blobData, m_cuda, m_log);

            // Copy labels - use DataCriteria for MULTIPLE labels
            if (m_param.memory_data_param.label_type == LayerParameterBase.LABEL_TYPE.MULTIPLE)
            {
                T[] rgLabels = m_blobLabel.mutable_cpu_data;
                int nIdx = 0;

                for (int i = 0; i < nNum; i++)
                {
                    List<float> rgLbl = BinaryData.UnPackFloatList(rgData[i].DataCriteria, rgData[i].DataCriteriaFormat);
                    for (int j = 0; j < rgLbl.Count; j++)
                    {
                        rgLabels[nIdx] = (T)Convert.ChangeType(rgLbl[j], typeof(T));
                        nIdx++;
                    }
                }
                m_blobLabel.mutable_cpu_data = rgLabels;
            }
            // Copy labels - use standard Datum label for SINGLE labels.
            else
            {
                T[] rgLabels = m_blobLabel.mutable_cpu_data;
                for (int i = 0; i < nNum; i++)
                {
                    rgLabels[i] = (T)Convert.ChangeType(rgData[i].label, typeof(T));
                }
                m_blobLabel.mutable_cpu_data = rgLabels;
            }

            m_bHasNewData = true;
            m_nN = nNum;
        }

        /// <summary>
        /// Resets the data by copying the internal data to the parameters specified.
        /// </summary>
        /// <param name="data">Specifies the data Blob that will receive a copy of the internal data.</param>
        /// <param name="labels">Specifies the label Blob that will receive a copy of the internal lables.</param>
        /// <param name="n">Specifies the number runs to perform on each batch.</param>
        public void Reset(Blob<T> data, Blob<T> labels, int n)
        {
            m_log.CHECK_GT(m_blobData.count(), 0, "There is no data.");
            m_log.CHECK_GT(m_blobLabel.count(), 0, "There are no lables.");
            m_log.CHECK_EQ(n % m_nBatchSize, 0, "'n' must be a multiple of batch size.");
            m_nN = n;

            m_blobData.ReshapeLike(data);
            m_blobLabel.ReshapeLike(labels);

            m_cuda.copy(m_blobData.count(), data.gpu_data, m_blobData.mutable_gpu_data);
            m_cuda.copy(m_blobLabel.count(), labels.gpu_data, m_blobLabel.mutable_gpu_data);
        }

        /// <summary>
        /// Returns the batch size.
        /// </summary>
        public int batch_size
        {
            get { return (int)m_nBatchSize; }
            set
            {
                m_log.CHECK(!m_bHasNewData, "Can't change the batch size until current data has been consumed.");
                m_nBatchSize = value;
                m_blobData.Reshape(m_nBatchSize, m_nChannels, m_nHeight, m_nWidth);
                m_blobLabel.Reshape(m_nBatchSize, m_nLabelChannels, m_nLabelHeight, m_nLabelWidth);
            }
        }

        /// <summary>
        /// Returns the data channels.
        /// </summary>
        public int channels
        {
            get { return m_nChannels; }
        }

        /// <summary>
        /// Returns the data height.
        /// </summary>
        public int height
        {
            get { return m_nHeight; }
        }

        /// <summary>
        /// Returns the data width.
        /// </summary>
        public int width
        {
            get { return m_nWidth; }
        }

        /// <summary>
        /// The forward computation which loads the data into the top (output) Blob%s.
        /// </summary>
        /// <param name="colBottom">Not used.</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     The data.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nSrcOffset;

            m_log.CHECK_GT(m_blobData.count(), 0, "MemoryDataLayer needs to be initialized by calling Reset or AddDatumVector.");

            List<int> rgLabelShape = Utility.Clone<int>(m_blobLabel.shape());
            if (rgLabelShape.Count == 0)
                rgLabelShape.Add(m_nBatchSize);
            else
                rgLabelShape[0] = m_nBatchSize;

            colTop[0].Reshape(m_nBatchSize, m_nChannels, m_nHeight, m_nWidth);
            colTop[1].Reshape(rgLabelShape);

            nSrcOffset = m_nPos * m_nDataSize;
            m_cuda.copy(colTop[0].count(), m_blobData.gpu_data, colTop[0].mutable_gpu_data, nSrcOffset, 0);

            nSrcOffset = m_nPos * m_nLabelSize;
            m_cuda.copy(colTop[1].count(), m_blobLabel.gpu_data, colTop[1].mutable_gpu_data, nSrcOffset, 0);
            m_nPos = (m_nPos + m_nBatchSize) % m_nN;

            if (m_nPos == 0)
            {
                m_bHasNewData = false;
                if (OnGetData != null)
                    OnGetData(this, new MemoryDataLayerGetDataArgs(false));
            }
        }
    }

    public class MemoryDataLayerGetDataArgs : EventArgs
    {
        bool m_bInitialization = true;

        public MemoryDataLayerGetDataArgs(bool bInit)
        {
            m_bInitialization = bInit;
        }

        public bool Initialization
        {
            get { return m_bInitialization; }
        }
    }
}
