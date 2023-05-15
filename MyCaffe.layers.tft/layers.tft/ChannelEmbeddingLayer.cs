using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.tft
{
    /// <summary>
    /// The ChannelEmbeddingLayer implements the transforming/embeddings for both the numeric and categorical data of an input channel.  This
    /// layer manages both a NumericTransformationLayer and CategoricalTransformationLayer.
    /// 
    /// Both the numeric_trans_param and categorical_trans_param's should be filled out for this layer.
    /// </summary>
    /// <remarks>
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L249) by Playtika Research, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ChannelEmbeddingLayer<T> : Layer<T>
    {
        Layer<T> m_numericLayer = null;
        Layer<T> m_categoricalLayer = null;
        BlobCollection<T> m_colBtm = new BlobCollection<T>();
        BlobCollection<T> m_colNumericTop = new BlobCollection<T>();
        BlobCollection<T> m_colCategoricalTop = new BlobCollection<T>();
        List<int> m_rgShape = new List<int>(4);

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public ChannelEmbeddingLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.CHANNEL_EMBEDDING;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_numericLayer);
            dispose(ref m_categoricalLayer);

            m_colNumericTop.Dispose();
            m_colCategoricalTop.Dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;
        }

        /// <summary>
        /// Returns the min number of required bottom (input) Blobs: numeric data or categorical data (determined based on param num_inputs)
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the max number of required bottom (input) Blobs: numeric data, categorical data
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: norm
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        private void getBlobs(BlobCollection<T> colBottom, out Blob<T> blobNumeric, out Blob<T> blobCategorical)
        {
            blobNumeric = null;
            blobCategorical = null;

            m_log.CHECK_EQ(colBottom.Count, 2, "The bottom must have a count = 2.");

            if (m_param.numeric_trans_param.num_input > 0)
            {
                m_log.CHECK_GT(colBottom[0].count(), 0, "The bottom(0) must have a count > 0!");
                blobNumeric = colBottom[0];
            }
            if (m_param.categorical_trans_param.num_input > 0)
            {
                m_log.CHECK_GT(colBottom[1].count(), 0, "The bottom(1) must have a count > 0!");
                blobCategorical = colBottom[1];
            }
            
            if (blobNumeric == null && blobCategorical == null)
                m_log.FAIL("At least one of the numeric or categorical num_input must be > 0.");
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobNumeric = null;
            Blob<T> blobCategorical = null;

            m_log.CHECK_EQ(m_param.numeric_trans_param.state_size, m_param.categorical_trans_param.state_size, "The numeric and categorical parameters must have the smae sate_size.");

            getBlobs(colBottom, out blobNumeric, out blobCategorical);

            if (blobNumeric != null)
            {
                m_colNumericTop.Clear();
                for (int i=0; i<m_param.numeric_trans_param.num_input; i++)
                {
                    Blob<T> blobTop = new Blob<T>(m_cuda, m_log);
                    m_colNumericTop.Add(blobTop);
                }

                LayerParameter p = new LayerParameter(LayerParameter.LayerType.NUMERIC_TRANS, m_param.name + ".numeric");
                p.numeric_trans_param.Copy(m_param.numeric_trans_param);
                m_numericLayer = Layer<T>.Create(m_cuda, m_log, convertLayerParam(p, m_param), null) as NumericTransformationLayer<T>;

                m_colBtm.Clear();
                m_colBtm.Add(blobNumeric);
                m_numericLayer.LayerSetUp(m_colBtm, m_colNumericTop);
                blobs.Add(m_numericLayer.blobs);
            }

            if (blobCategorical != null)
            {
                m_colCategoricalTop.Clear();
                for (int i = 0; i < m_param.categorical_trans_param.num_input; i++)
                {
                    Blob<T> blobTop = new Blob<T>(m_cuda, m_log);
                    m_colCategoricalTop.Add(blobTop);
                }

                LayerParameter p = new LayerParameter(LayerParameter.LayerType.CATEGORICAL_TRANS, m_param.name + ".categorical");
                p.categorical_trans_param.Copy(m_param.categorical_trans_param);
                m_categoricalLayer = Layer<T>.Create(m_cuda, m_log, convertLayerParam(p, m_param), null) as CategoricalTransformationLayer<T>;

                m_colBtm.Clear();
                m_colBtm.Add(blobCategorical);
                m_categoricalLayer.LayerSetUp(m_colBtm, m_colCategoricalTop);
                blobs.Add(m_categoricalLayer.blobs);
            }
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobNumeric = null;
            Blob<T> blobCategorical = null;
            int nN = 0;
            int nC = 0;
            int? nH = null;

            getBlobs(colBottom, out blobNumeric, out blobCategorical);

            if (blobNumeric != null)
            {
                nN = colBottom[0].num;
                m_colBtm.Clear();
                m_colBtm.Add(blobNumeric);
                m_numericLayer.Reshape(m_colBtm, m_colNumericTop);

                if (colBottom[0].num_axes > 2)
                {
                    nC = colBottom[0].channels;
                    nH = (int)m_param.numeric_trans_param.num_input;
                }
                else
                {
                    nC += (int)m_param.numeric_trans_param.num_input;
                }
            }

            if (blobCategorical != null)
            {
                if (nN != 0)
                    m_log.CHECK_EQ(colBottom[1].num, nN, "The bottom(0).num and bottom(1).num must be equal!");

                nN = colBottom[1].num;
                m_colBtm.Clear();
                m_colBtm.Add(blobCategorical);
                m_categoricalLayer.Reshape(m_colBtm, m_colCategoricalTop);

                if (colBottom[0].num_axes > 2)
                {
                    if (nC != 0)
                        m_log.CHECK_EQ(colBottom[1].channels, nC, "The bottom(0).channels and bottom(1).channels must be equal!");

                    nC = colBottom[1].channels;

                    if (!nH.HasValue)
                        nH = 0;
                    nH += (int)m_param.categorical_trans_param.num_input;
                }
                else
                {
                    nC += (int)m_param.categorical_trans_param.num_input;
                }
            }

            int nEmb = (int)m_param.categorical_trans_param.state_size;

            m_rgShape.Clear();
            m_rgShape.Add(nN);

            if (!nH.HasValue)
            {
                m_rgShape.Add(nC * nEmb);
            }
            else
            {
                m_rgShape.Add(nC);
                m_rgShape.Add(nH.Value * nEmb);
            }

            colTop[0].Reshape(m_rgShape);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H numeric \times 1) @f$ 
        ///     the numeric inputs @f$ x @f$
        ///  -# @f$ (N \times C \times H categorical \times 1) @f$ 
        ///     the categorical inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector)
        ///  -# @f$ (N \times C \times H numeric + H categorical \times Emb size) @f$
        ///     the computed outputs 
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobNumeric = null;
            Blob<T> blobCategorical = null;
            int nCount = 0;
            int nBlocks = (int)m_param.numeric_trans_param.num_input + (int)m_param.categorical_trans_param.num_input;
            int nEmb = (int)m_param.numeric_trans_param.state_size;
            int nIdx = 0;

            getBlobs(colBottom, out blobNumeric, out blobCategorical);

            if (blobNumeric != null)
            {
                m_colBtm.Clear();
                m_colBtm.Add(blobNumeric);
                m_numericLayer.Forward(m_colBtm, m_colNumericTop);
                nCount = m_colNumericTop[0].count();

                for (int i=0; i<m_colNumericTop.Count; i++)
                {
                    m_cuda.channel_copy(nCount, m_colNumericTop[0].num, 1, nBlocks, nEmb, nIdx, colTop[0].mutable_gpu_data, m_colNumericTop[i].gpu_data, DIR.BWD);
                    nIdx++;
                }
            }

            if (blobCategorical != null)
            {
                m_colBtm.Clear();
                m_colBtm.Add(blobCategorical);
                m_categoricalLayer.Forward(m_colBtm, m_colCategoricalTop);
                nCount = m_colCategoricalTop[0].count();

                for (int i = 0; i < m_colCategoricalTop.Count; i++)
                {
                    m_cuda.channel_copy(nCount, m_colCategoricalTop[0].num, 1, nBlocks, nEmb, nIdx, colTop[0].mutable_gpu_data, m_colCategoricalTop[i].gpu_data, DIR.BWD);
                    nIdx++;
                }
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the stacked embedding numeric and categorical value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H numeric + H categorical \times Emb size) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times 1) @f$
        ///     the numeric inputs @f$ x @f$;  
        ///  -# @f$ (N \times C \times H \times 1) @f$
        ///     the categorical inputs @f$ x @f$;  
        ///  NOTE: gradients are only passed to the internal embedding layers and not the bottom inputs
        ///  for they contain data values.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            Blob<T> blobNumeric = null;
            Blob<T> blobCategorical = null;
            int nCount;
            int nBlocks = (int)m_param.numeric_trans_param.num_input + (int)m_param.categorical_trans_param.num_input;
            int nEmb = (int)m_param.numeric_trans_param.state_size;
            int nIdx = 0;

            getBlobs(colBottom, out blobNumeric, out blobCategorical);

            if (blobNumeric != null)
            {
                nCount = m_colNumericTop[0].count();

                for (int i = 0; i < m_colNumericTop.Count; i++)
                {
                    m_cuda.channel_copy(nCount, m_colNumericTop[0].num, 1, nBlocks, nEmb, nIdx, colTop[0].gpu_diff, m_colNumericTop[i].mutable_gpu_diff, DIR.FWD);
                    nIdx++;
                }

                m_colBtm.Clear();
                m_colBtm.Add(blobNumeric);
                m_numericLayer.Backward(m_colNumericTop, new List<bool>() { true }, m_colBtm);
            }

            if (blobCategorical != null)
            {
                nCount = m_colCategoricalTop[0].count();

                for (int i = 0; i < m_colCategoricalTop.Count; i++)
                {
                    m_cuda.channel_copy(nCount, m_colCategoricalTop[0].num, 1, nBlocks, nEmb, nIdx, colTop[0].gpu_diff, m_colCategoricalTop[i].mutable_gpu_diff, DIR.FWD);
                    nIdx++;
                }

                m_colBtm.Clear();
                m_colBtm.Add(blobCategorical);
                m_categoricalLayer.Backward(m_colCategoricalTop, new List<bool>() { true }, m_colBtm);
            }
        }
    }
}
