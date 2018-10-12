using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.basecode.descriptors;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.data;
using MyCaffe.db.image;

namespace MyCaffe.layers
{
    /// <summary>
    /// The BaseDataLayer is the base class for data Layers that feed Blobs of data into the Net.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public abstract class BaseDataLayer<T> : Layer<T>
    {
        /// <summary>
        /// Specifies the DataTransformer used to transform each data item as it loaded.
        /// </summary>
        protected DataTransformer<T> m_transformer;
        /// <summary>
        /// Specifies whether or not the Layer should output labels.
        /// </summary>
        protected bool m_bOutputLabels;
        /// <summary>
        /// Specifies the CaffeImageDatabase.
        /// </summary>
        protected IXImageDatabase m_imgdb;
        /// <summary>
        /// Specifies the SourceDescriptor of the data source.
        /// </summary>
        protected SourceDescriptor m_src;
        /// <summary>
        /// Specifies the SimpleDatum that optionally contains the image Mean for data centering.
        /// </summary>
        protected SimpleDatum m_imgMean = null;

        /// <summary>
        /// The BaseDataLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter</param>
        /// <param name="db">Specifies the external database to use.</param>
        public BaseDataLayer(CudaDnn<T> cuda, Log log, LayerParameter p, IXImageDatabase db)
            : base(cuda, log, p)
        {
            if (db != null)
            {
                m_imgdb = db;

                if (p.type == LayerParameter.LayerType.DATA ||
                    p.type == LayerParameter.LayerType.TRIPLET_DATA)
                    m_src = m_imgdb.GetSourceByName(p.data_param.source);

                if (p.transform_param.use_imagedb_mean)
                {
                    if (db != null)
                        m_imgMean = db.GetImageMean(m_src.ID);
                    else
                        m_log.WriteLine("WARNING: The image database is NULL, and therefore no mean image can not be acquired.");
                }
            }
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
        }

        /// <summary>
        /// Get/set the image mean.
        /// </summary>
        public SimpleDatum ImageMean
        {
            get { return m_imgMean; }
            set { m_imgMean = value; }
        }

        /// <summary>
        /// Implements common data layer setup functionality, and calls 
        /// DataLayerSetUp to do special data layer setup for individual layer types.
        /// </summary>
        /// <remarks>
        /// This method may not be overridden except by BasePrefetchingDataLayer.
        /// </remarks>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (colTop.Count == 1)
                m_bOutputLabels = false;
            else
                m_bOutputLabels = true;

            m_transformer = new DataTransformer<T>(m_log, m_param.transform_param, m_param.phase, m_imgMean);
            m_transformer.InitRand();

            // The subclasses should setup the size of bottom and top.
            DataLayerSetUp(colBottom, colTop);
        }

        /// <summary>
        /// Override this method to perform the actual data loading.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public abstract void DataLayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop);

        /// <summary>
        /// Data layers have no bottoms, so reshaping is trivial.
        /// </summary>
        /// <param name="colBottom">not used.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
        }

        /// @brief Not implemented - data Layers do not perform backward.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }

        /// <summary>
        /// Returns the data transformer used.
        /// </summary>
        public DataTransformer<T> Transformer
        {
            get { return m_transformer; }
        }
    }
}
