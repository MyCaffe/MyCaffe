using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.tft
{
    /// <summary>
    /// The CategoricalTransformationLayer implements the transforming/embeddings for the set of categorical input variables from a single input channel.
    /// Each input variable is projected using a dedicated embedding layer to a vector of width state_size.  The result of applying this module
    /// is a list, with length num_inputs, that contians the embedding of each input variable for all observations and time steps.
    /// </summary>
    /// <remarks>
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L367) by Playtika Research, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class CategoricalTransformationLayer<T> : Layer<T>
    {
        List<Layer<T>> m_rgEmbLayers = new List<Layer<T>>();
        BlobCollection<T> m_rgBtm = new BlobCollection<T>();
        BlobCollection<T> m_rgEmbBtm = new BlobCollection<T>();
        BlobCollection<T> m_rgEmbTop = new BlobCollection<T>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public CategoricalTransformationLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.CATEGORICAL_TRANS;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: data
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: norm
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return (int)m_param.categorical_trans_param.num_input; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nOffset = (colBottom[0].num_axes == 2) ? 1 : 2;
            int nDim = colBottom[0].count(0, nOffset);
            int nNumInput = m_param.categorical_trans_param.cardinalities.Count;
            int nSpatialDim = colBottom[0].count(colBottom[0].num_axes-1) / nNumInput;
            List<int> rgShape = new List<int>() { nDim, nSpatialDim };
            Blob<T> blobBtm = null;

            m_log.CHECK_EQ(m_param.categorical_trans_param.num_input, m_param.categorical_trans_param.cardinalities.Count, "The num_input must match the number of cardinalities!");

            m_rgEmbBtm.Clear();
            m_rgEmbBtm.Add(blobBtm);
            m_rgEmbTop.Clear();
            m_rgEmbTop.Add(colTop[0]);

            for (int i = 0; i < nNumInput; i++)
            {
                blobBtm = new Blob<T>(m_cuda, m_log);
                blobBtm.Reshape(rgShape);
                m_rgBtm.Add(blobBtm);

                m_rgEmbBtm[0] = m_rgBtm[i];
                m_rgEmbTop[0] = colTop[i];

                int nCardinality = m_param.categorical_trans_param.cardinalities[i];
                LayerParameter p = new LayerParameter(LayerParameter.LayerType.EMBED, m_param.name + ".emb" + i.ToString());
                p.embed_param.num_output = m_param.categorical_trans_param.state_size;
                p.embed_param.input_dim = (uint)nCardinality;
                p.embed_param.bias_term = false;

                Layer<T> emb_layer = Layer<T>.Create(m_cuda, m_log, convertLayerParam(p, m_param), null);
                m_rgEmbLayers.Add(emb_layer);

                emb_layer.LayerSetUp(m_rgEmbBtm, m_rgEmbTop);
                blobs.Add(emb_layer.blobs);
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNumInput = m_param.categorical_trans_param.cardinalities.Count;
            for (int i = 0; i < nNumInput; i++)
            {
                m_rgEmbBtm[0] = m_rgBtm[i];
                m_rgEmbTop[0] = colTop[i];
                m_rgEmbLayers[i].Reshape(m_rgEmbBtm, m_rgEmbTop);
            }
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times 1) @f$ 
        ///     the inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length len(cardinalities))
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the computed outputs 
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNumInput = m_param.categorical_trans_param.cardinalities.Count;
            for (int i = 0; i < nNumInput; i++)
            {
                int nCount = m_rgBtm[i].count();
                m_cuda.channel_copy(nCount, nCount, 1, nNumInput, 1, i, colBottom[0].gpu_data, m_rgBtm[i].mutable_gpu_data, DIR.FWD);

                m_rgEmbBtm[0] = m_rgBtm[i];
                m_rgEmbTop[0] = colTop[i];
                m_rgEmbLayers[i].Forward(m_rgEmbBtm, m_rgEmbTop);
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the cardinality value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$; Backward fills their diff 
        ///     if propagate_down[0]
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            int nNumInput = m_param.categorical_trans_param.cardinalities.Count;
            for (int i = 0; i < nNumInput; i++)
            {
                m_rgEmbBtm[0] = m_rgBtm[i];
                m_rgEmbTop[0] = colTop[i];
                m_rgEmbLayers[i].Backward(m_rgEmbTop, rgbPropagateDown, m_rgEmbBtm);

                // data fields do not have gradients so no gradients are output.
                //int nCount = m_rgEmbBtm[0].count();
                //m_cuda.channel_copy(nCount, nCount, 1, nNumInput, 1, i, colBottom[0].mutable_gpu_diff, m_rgEmbBtm[0].gpu_diff, DIR.BWD);
            }
        }
    }
}
