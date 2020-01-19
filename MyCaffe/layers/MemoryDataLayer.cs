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
        int m_nLabelChannels = 0;
        int m_nLabelHeight = 0;
        int m_nLabelWidth = 0;
        int m_nDataSize;
        int m_nLabelSize = 0;
        int m_nClipSize1 = 0;
        int m_nClipSize2 = 0;
        int m_nLabelIdx = -1;
        int m_nClipIdx = -1;
        Blob<T> m_blobData;
        Blob<T> m_blobLabel;
        Blob<T> m_blobClip = null;
        bool m_bHasNewData;
        int m_nPos = 0;
        int m_nN = 1;

        /// <summary>
        /// The OnGetData event fires on the DataLayerSetup call and each time the data wraps around (e.g. all data as already fed through) during the forward call.
        /// </summary>
        public event EventHandler<MemoryDataLayerGetDataArgs> OnGetData;
        /// <summary>
        /// The OnDataPack event fires from within the AddDatumVector method and is used to pack the data into a specific ordering needed by the layers connected
        /// downstream of the MemoryDataLayer, such as an LSTM layer.
        /// </summary>
        public event EventHandler<MemoryDataLayerPackDataArgs<T>> OnDataPack;

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
            m_blobData.Name = m_param.name + " data";
            m_blobLabel = new Blob<T>(cuda, log);
            m_blobLabel.Name = m_param.name + " label";
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
        protected override void DataLayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nBatchSize = (int)m_param.memory_data_param.batch_size;
            m_nChannels = (int)m_param.memory_data_param.channels;
            m_nHeight = (int)m_param.memory_data_param.height;
            m_nWidth = (int)m_param.memory_data_param.width;
            m_nDataSize = m_nChannels * m_nHeight * m_nWidth;

            m_log.CHECK_GT(m_nBatchSize * m_nDataSize, 0, "batch_size, channels, height, and width must be specified and positive in memory_data_param.");

            if (m_param.memory_data_param.label_type != LayerParameterBase.LABEL_TYPE.NONE)
            {
                m_nLabelChannels = (int)m_param.memory_data_param.label_channels;
                m_nLabelHeight = (int)m_param.memory_data_param.label_height;
                m_nLabelWidth = (int)m_param.memory_data_param.label_width;
                m_nLabelSize = m_nLabelChannels * m_nLabelHeight * m_nLabelWidth;

                if (m_param.memory_data_param.label_type == LayerParameterBase.LABEL_TYPE.MULTIPLE)
                    m_log.CHECK_GT(m_nBatchSize * m_nLabelSize, 0, "batch_size, label_channels, label_height, and label_width must be specified and positive in memory_data_param when using label_type = MULTIPLE.");
            }

            if (m_param.memory_data_param.clip_length1 > 0)
                m_nClipSize1 = (int)m_param.memory_data_param.clip_length1;

            if (m_param.memory_data_param.clip_length2 > 0)
                m_nClipSize2 = (int)m_param.memory_data_param.clip_length2;

            if (m_nClipSize1 > 1 && m_nClipSize2 == 0)
                m_nClipSize2 = 1;

            if (m_nClipSize2 > 1 && m_nClipSize1 == 0)
                m_nClipSize1 = 1;

            if (m_nClipSize1 > 0 || m_nClipSize2 > 0)
            {
                m_blobClip = new Blob<T>(m_cuda, m_log);
                m_blobClip.Name = m_param.name + " clip";
                m_blobClip.Reshape(new List<int>() { m_nClipSize1, m_nClipSize2 });
            }

            if (OnGetData != null)
            {
                OnGetData(this, new MemoryDataLayerGetDataArgs(true));

                if (m_param.memory_data_param.label_type != LayerParameterBase.LABEL_TYPE.NONE)
                {
                    m_log.CHECK_EQ(m_nLabelChannels, m_blobLabel.channels, "The actual label channels (" + m_blobLabel.channels.ToString() + ") do not match the 'memory_data_param.label_channels' setting of " + m_nLabelChannels.ToString() + ".");
                    m_log.CHECK_EQ(m_nLabelHeight, m_blobLabel.height, "The actual label channels (" + m_blobLabel.height.ToString() + ") do not match the 'memory_data_param.label_channels' setting of " + m_nLabelHeight.ToString() + ".");
                    m_log.CHECK_EQ(m_nLabelWidth, m_blobLabel.width, "The actual label channels (" + m_blobLabel.width.ToString() + ") do not match the 'memory_data_param.label_channels' setting of " + m_nLabelWidth.ToString() + ".");
                }
            }
            else
            {
                m_blobData.Reshape(m_nBatchSize, m_nChannels, m_nHeight, m_nWidth);

                if (m_param.memory_data_param.label_type != LayerParameterBase.LABEL_TYPE.NONE)
                {
                    if (m_param.memory_data_param.label_type == LayerParameterBase.LABEL_TYPE.SINGLE && m_blobClip != null)
                        m_blobLabel.Reshape(1, m_nLabelChannels, m_nLabelHeight, m_nLabelWidth);
                    else
                        m_blobLabel.Reshape(m_nBatchSize, m_nLabelChannels, m_nLabelHeight, m_nLabelWidth);
                }
            }

            colTop[0].Reshape(m_nBatchSize, m_nChannels, m_nHeight, m_nWidth);
            m_blobData.update_cpu_data();

            int nLabelIdx = -1;
            int nClipIdx = -1;

            for (int i = 0; i < m_param.top.Count; i++)
            {
                if (m_param.top[i].ToLower() == "label")
                    nLabelIdx = i;

                if (m_param.top[i].ToLower() == "clip")
                    nClipIdx = i;
            }

            if (m_param.memory_data_param.label_type != LayerParameterBase.LABEL_TYPE.NONE)
            {
                if (nLabelIdx == -1)
                    nLabelIdx = (nClipIdx == -1) ? 1 : nClipIdx + 1;

                m_nLabelIdx = nLabelIdx;
                colTop[nLabelIdx].ReshapeLike(m_blobLabel);
                m_blobLabel.update_cpu_data();
            }

            if (m_blobClip != null)
            {
                if (nClipIdx == -1)
                    throw new Exception("Could not find a top named 'clip'!");

                m_nClipIdx = nClipIdx;
                colTop[nClipIdx].ReshapeLike(m_blobClip);
                m_blobClip.update_cpu_data();
            }
        }

        /// <summary>
        /// Reshape the internal data and outputs.
        /// </summary>
        /// <param name="colBottom">Specifies the inputs.</param>
        /// <param name="colTop">Specifies teh outputs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_blobData.Reshape(m_nBatchSize, m_nChannels, m_nHeight, m_nWidth);

            colTop[0].Reshape(m_nBatchSize, m_nChannels, m_nHeight, m_nWidth);
            m_blobData.update_cpu_data();

            if (m_param.memory_data_param.label_type != LayerParameterBase.LABEL_TYPE.NONE)
            {
                if (m_param.memory_data_param.label_type == LayerParameterBase.LABEL_TYPE.SINGLE && m_blobClip != null)
                    m_blobLabel.Reshape(1, m_nLabelChannels, m_nLabelHeight, m_nLabelWidth);
                else
                    m_blobLabel.Reshape(m_nBatchSize, m_nLabelChannels, m_nLabelHeight, m_nLabelWidth);

                colTop[m_nLabelIdx].ReshapeLike(m_blobLabel);
                m_blobLabel.update_cpu_data();
            }

            if (m_blobClip != null)
            {
                m_blobClip.Reshape(new List<int>() { m_nClipSize1, m_nClipSize2 });
                colTop[m_nClipIdx].ReshapeLike(m_blobClip);
                m_blobClip.update_cpu_data();
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
        /// Returns the exact number of required top (output) Blobs.
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the minimum number of top blobs: data
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of top blobs: data, clip, label
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 3; }
        }

        /// <summary>
        /// This method is used to add a single Datum to the memory.
        /// </summary>
        /// <param name="sd">The Data Datum to add.</param>
        /// <param name="nLblAxis">Optionally, specifies the axis on which the multi-label data is placed.  This field is not used on SINGLE label types.</param>
        /// <param name="bReset">Optionally, specifies to force reset the internal data.</param>
        /// <param name="bResizeBatch">Optionally, specifies whether or not to size the batch to the number of rgData.</param>
        public virtual void AddDatum(SimpleDatum sd, int nLblAxis = 1, bool bReset = false, bool bResizeBatch = false)
        {
            List<Datum> rgData = new List<Datum>();
            rgData.Add(new Datum(sd));
            AddDatumVector(rgData, null, nLblAxis, bReset, bResizeBatch);
        }

        /// <summary>
        /// This method is used to add a list of Datums to the memory.
        /// </summary>
        /// <param name="rgData">The list of Data Datums to add.</param>
        /// <param name="rgClip">Optionally, specifies the clip data, if any exits.</param>
        /// <param name="nLblAxis">Optionally, specifies the axis on which the multi-label data is placed.  This field is not used on SINGLE label types.</param>
        /// <param name="bReset">Optionally, specifies to force reset the internal data.</param>
        /// <param name="bResizeBatch">Optionally, specifies whether or not to size the batch to the number of rgData.</param>
        /// <remarks>
        /// The batch size is only resized when the clip data is null, otherwise, regardless of the 'bResizeBatch' setting, the batch size is not changed.</remarks>
        public virtual void AddDatumVector(Datum[] rgData, Datum[] rgClip = null, int nLblAxis = 1, bool bReset = false, bool bResizeBatch = false)
        {
            AddDatumVector(rgData.ToList(), (rgClip == null) ? null : rgClip.ToList(), nLblAxis, bReset, bResizeBatch);
        }

        /// <summary>
        /// This method is used to add a list of Datums to the memory.
        /// </summary>
        /// <param name="rgData">The list of Data Datums to add.</param>
        /// <param name="rgClip">Optionally, specifies the clip data, if any exits.</param>
        /// <param name="nLblAxis">Optionally, specifies the axis on which the multi-label data is placed.  This field is not used on SINGLE label types.</param>
        /// <param name="bReset">Optionally, specifies to force reset the internal data.</param>
        /// <param name="bResizeBatch">Optionally, specifies whether or not to size the batch to the number of rgData.</param>
        /// <remarks>
        /// The batch size is only resized when the clip data is null, otherwise, regardless of the 'bResizeBatch' setting, the batch size is not changed.</remarks>
        public virtual void AddDatumVector(List<Datum> rgData, List<Datum> rgClip = null, int nLblAxis = 1, bool bReset = false, bool bResizeBatch = false)
        {
            if (bReset)
                m_bHasNewData = false;

            m_log.CHECK(!m_bHasNewData, "Can't add data until current data has been consumed.");
            int nNum = rgData.Count;
            m_log.CHECK_GT(nNum, 0, "There are no datum to add.");

            if (rgClip == null)
            {
                if (bResizeBatch)
                    m_nBatchSize = rgData.Count;

                int nNumAligned = (int)Math.Floor((double)rgData.Count / (double)m_nBatchSize) * m_nBatchSize;
                m_log.CHECK_GT(nNumAligned, 0, "Three are not enough datum to add.");

                if (nNumAligned < nNum)
                {
                    m_log.WriteLine("WARNING: Clipping batch to batch aligned count of " + nNumAligned.ToString() + ".");

                    for (int i = nNumAligned; i < nNum; i++)
                    {
                        rgData.RemoveAt(rgData.Count - 1);
                    }
                }

                nNum = nNumAligned;
                m_log.CHECK_EQ(nNum % m_nBatchSize, 0, "The added data must be a multiple of the batch size.");

                m_blobData.Reshape(nNum, m_nChannels, m_nHeight, m_nWidth);

                // Apply data transformations (mirror, scale, crop...)
                m_transformer.Transform(rgData, m_blobData, m_cuda, m_log);

                if (m_param.memory_data_param.label_type != LayerParameterBase.LABEL_TYPE.NONE)
                {
                    List<int> rgLblShape = new List<int>();
                    int nLabelNum = nNum;

                    if (m_param.memory_data_param.label_type == LayerParameterBase.LABEL_TYPE.ONLY_ONE)
                        nLabelNum = 1;

                    rgLblShape.Add(nLabelNum);
                    rgLblShape.Add(1);
                    rgLblShape.Add(1);
                    rgLblShape.Add(1);

                    // Reshape label blob depending on label type.
                    if (m_param.memory_data_param.label_type == LayerParameterBase.LABEL_TYPE.MULTIPLE)
                    {
                        m_log.CHECK_GE(nLblAxis, 1, "The label axis must be greater than or equal to 1.");
                        m_log.CHECK_LE(nLblAxis, 4, "The label axis must be less than 4.");

                        List<float> rgLbl = BinaryData.UnPackFloatList(rgData[0].DataCriteria, rgData[0].DataCriteriaFormat);
                        rgLblShape[nLblAxis] = rgLbl.Count;
                    }

                    m_blobLabel.Reshape(rgLblShape);

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
                    else
                    {
                        T[] rgLabels = m_blobLabel.mutable_cpu_data;

                        // For recurrent layers, single labels only use the last label in the sequence.
                        if (rgClip != null)
                        {
                            rgLabels[0] = (T)Convert.ChangeType(rgData[nNum - 1].label, typeof(T));
                        }
                        else
                        {
                            // Copy labels - use standard Datum label for SINGLE labels.
                            for (int i = 0; i < nNum; i++)
                            {
                                rgLabels[i] = (T)Convert.ChangeType(rgData[i].label, typeof(T));
                            }
                        }

                        m_blobLabel.mutable_cpu_data = rgLabels;
                    }
                }
            }
            else
            {
                if (OnDataPack == null)
                    throw new Exception("To properly handle data packing, you must connect the OnDataPack event and properly fill out the data and clip blobs with the ordering expected by the recurrent layers.");

                MemoryDataLayerPackDataArgs<T> args = new MemoryDataLayerPackDataArgs<T>(m_blobData, m_blobClip, m_blobLabel, rgData, rgClip);
                OnDataPack(this, args);
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
            m_log.CHECK_EQ(n % m_nBatchSize, 0, "'n' must be a multiple of batch size.");
            m_nN = n;

            m_log.CHECK_GT(m_blobData.count(), 0, "There is no data.");
            m_blobData.ReshapeLike(data);
            m_cuda.copy(m_blobData.count(), data.gpu_data, m_blobData.mutable_gpu_data);

            if (m_param.memory_data_param.label_type != LayerParameterBase.LABEL_TYPE.NONE)
            {
                m_log.CHECK_GT(m_blobLabel.count(), 0, "There are no lables.");
                m_blobLabel.ReshapeLike(labels);
                m_cuda.copy(m_blobLabel.count(), labels.gpu_data, m_blobLabel.mutable_gpu_data);
            }
        }

        /// <summary>
        /// Copy the data by copying the src alyer data and label to the parameters specified.
        /// </summary>
        /// <param name="src">Specifies the source layer.</param>
        public void Copy(MemoryDataLayer<T> src)
        {
            m_nN = src.m_nN;

            m_blobData.ReshapeLike(src.m_blobData);
            m_cuda.copy(src.m_blobData.count(), src.m_blobData.gpu_data, m_blobData.mutable_gpu_data);

            if (m_param.memory_data_param.label_type != LayerParameterBase.LABEL_TYPE.NONE)
            {
                m_blobLabel.ReshapeLike(src.m_blobLabel);
                m_cuda.copy(src.m_blobLabel.count(), src.m_blobLabel.gpu_data, m_blobLabel.mutable_gpu_data);
            }

            if (m_blobClip != null)
            {
                m_blobClip.ReshapeLike(src.m_blobClip);
                m_cuda.copy(src.m_blobClip.count(), src.m_blobClip.gpu_data, m_blobClip.mutable_gpu_data);
            }

            m_bHasNewData = true;
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
        /// Returns the clip 1 size, if any exists.
        /// </summary>
        public int clip_size1
        {
            get { return m_nClipSize1; }
        }

        /// <summary>
        /// Returns the clip 2 size, if any exists.
        /// </summary>
        public int clip_size2
        {
            get { return m_nClipSize2; }
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

            Blob<T> blobData = colTop[0];
            Blob<T> blobClip = null;
            Blob<T> blobLabel = null;

            for (int i=1; i<colTop.Count; i++)
            {
                if (blobClip == null && (colTop[i].type == BLOB_TYPE.CLIP || colTop[i].Name.ToLower().Contains("clip")))
                    blobClip = colTop[i];
                else
                    blobLabel = colTop[i];
            }

            blobData.Reshape(m_nBatchSize, m_nChannels, m_nHeight, m_nWidth);

            nSrcOffset = m_nPos * m_nDataSize;
            m_cuda.copy(blobData.count(), m_blobData.gpu_data, blobData.mutable_gpu_data, nSrcOffset, 0);


            if (m_param.memory_data_param.label_type != LayerParameterBase.LABEL_TYPE.NONE)
            {
                if (blobLabel == null)
                    m_log.WriteError(new Exception("Could not find the MemoryDataLayer 'label' top!"));

                int nLabelSize = m_nBatchSize;
                if (m_param.memory_data_param.label_type == LayerParameterBase.LABEL_TYPE.ONLY_ONE)
                    nLabelSize = 1;

                List<int> rgLabelShape = Utility.Clone<int>(m_blobLabel.shape());
                if (rgLabelShape.Count == 0)
                    rgLabelShape.Add(nLabelSize);
                else
                    rgLabelShape[0] = nLabelSize;

                blobLabel.Reshape(rgLabelShape);

                nSrcOffset = m_nPos * m_nLabelSize;
                m_cuda.copy(blobLabel.count(), m_blobLabel.gpu_data, blobLabel.mutable_gpu_data, nSrcOffset, 0);
            }

            if (m_blobClip != null)
            {
                if (blobClip == null)
                    m_log.WriteError(new Exception("Could not find the MemoryDataLayer 'clip' top!"));

                blobClip.CopyFrom(m_blobClip, false, true);
            }

            m_nPos = (m_nPos + m_nBatchSize) % m_nN;

            if (m_nPos == 0)
                m_bHasNewData = false;
        }
    }

    /// <summary>
    /// The MemoryDataLayerGetDataArgs class is passed to the OnGetData event.
    /// </summary>
    public class MemoryDataLayerGetDataArgs : EventArgs
    {
        bool m_bInitialization = true;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bInit">Set to <i>true</i> when the event fires from within the DataLayerSetup and <i>false</i> otherwise.</param>
        public MemoryDataLayerGetDataArgs(bool bInit)
        {
            m_bInitialization = bInit;
        }

        /// <summary>
        /// Returns whether the event was fired during the DataLayerSetup call or not.
        /// </summary>
        public bool Initialization
        {
            get { return m_bInitialization; }
        }
    }

    /// <summary>
    /// The MemoryDataLayerPackDataArgs is passed to the OnDataPack event which fires each time the data received in AddDatumVector needs to be packed 
    /// into a specific ordering as is the case when using an LSTM network.
    /// </summary>
    /// <typeparam name="T">Specifies the base datatype of <i>float</i> or <i>double</i>.</typeparam>
    public class MemoryDataLayerPackDataArgs<T> : EventArgs
    {
        Blob<T> m_blobData;
        Blob<T> m_blobClip;
        Blob<T> m_blobLabel;
        List<Datum> m_rgData;
        List<Datum> m_rgClip;
        LayerParameter.LayerType m_lstmType = LayerParameter.LayerType.LSTM;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="blobData">Specifies the blob data to fill with the ordered data.</param>
        /// <param name="blobClip">Specifies the clip data to fill with the ordered data.</param>
        /// <param name="blobLabel">Specifies the labeld ata to fill with ordered data.</param>
        /// <param name="rgData">Specifies the raw data to use to fill.</param>
        /// <param name="rgClip">Specifies the raw clip data to use to fill.</param>
        /// <param name="type">Specifies the LSTM type.</param>
        public MemoryDataLayerPackDataArgs(Blob<T> blobData, Blob<T> blobClip, Blob<T> blobLabel, List<Datum> rgData, List<Datum> rgClip, LayerParameter.LayerType type = LayerParameter.LayerType.LSTM)
        {
            m_blobData = blobData;
            m_blobClip = blobClip;
            m_blobLabel = blobLabel;
            m_rgData = rgData;
            m_rgClip = rgClip;
            m_lstmType = type;
        }

        /// <summary>
        /// Returns the LSTM type.
        /// </summary>
        public LayerParameter.LayerType LstmType
        {
            get { return m_lstmType; }
        }

        /// <summary>
        /// Returns the blob data to fill with ordered data.
        /// </summary>
        public Blob<T> Data
        {
            get { return m_blobData; }
        }

        /// <summary>
        /// Returns the clip data to fill with ordered data for clipping.
        /// </summary>
        public Blob<T> Clip
        {
            get { return m_blobClip; }
        }

        /// <summary>
        /// Returns the label data to fill with ordered label information.
        /// </summary>
        public Blob<T> Label
        {
            get { return m_blobLabel; }
        }

        /// <summary>
        /// Returns the raw data items to use to fill.
        /// </summary>
        public List<Datum> DataItems
        {
            get { return m_rgData; }
        }

        /// <summary>
        /// Returns the raw clip items to use to fill.
        /// </summary>
        public List<Datum> ClipItems
        {
            get { return m_rgClip; }
        }
    }
}
