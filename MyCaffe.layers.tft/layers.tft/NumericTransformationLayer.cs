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
    /// The NumericTransformationLayer implements the transforming/embeddings for the set of numeric input variables from a single input channel.
    /// Each input variable is projected using a dedicated inner product layer to a vector of width state_size.  The result of applying this module
    /// is a list, with length num_inputs, that contians the embedding of each input variable for all observations and time steps.
    /// </summary>
    /// <remarks>
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L333) by Playtika Research, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class NumericTransformationLayer<T> : Layer<T>
    {
        List<Layer<T>> m_rgIpLayers = new List<Layer<T>>();
        BlobCollection<T> m_rgBtm = new BlobCollection<T>();
        BlobCollection<T> m_rgIpBtm = new BlobCollection<T>();
        BlobCollection<T> m_rgIpTop = new BlobCollection<T>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public NumericTransformationLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.NUMERIC_TRANS;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_rgIpLayers != null)
            {
                foreach (Layer<T> layer in m_rgIpLayers)
                {
                    layer.Dispose();
                }
                m_rgIpLayers = null;
            }

            if (m_rgBtm != null)
            {
                m_rgBtm.Dispose();
                m_rgBtm = null;
            }
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
            get { return (int)m_param.numeric_trans_param.num_input; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nDim = colBottom[0].count(0, 2);
            int nSpatialDim = 1;
            List<int> rgShape = new List<int>() { nDim, nSpatialDim };
            Blob<T> blobBtm = null;

            m_rgIpBtm.Clear();
            m_rgIpBtm.Add(blobBtm);
            m_rgIpTop.Clear();
            m_rgIpTop.Add(colTop[0]);

            for (int i = 0; i < m_param.numeric_trans_param.num_input; i++)
            {
                blobBtm = new Blob<T>(m_cuda, m_log);
                blobBtm.Reshape(rgShape);
                m_rgBtm.Add(blobBtm);

                m_rgIpBtm[0] = m_rgBtm[i];
                m_rgIpTop[0] = colTop[i];

                LayerParameter p = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT);
                p.inner_product_param.num_output = m_param.numeric_trans_param.state_size;
                p.inner_product_param.axis = 1;

                Layer<T> ip_layer = Layer<T>.Create(m_cuda, m_log, p, null);
                m_rgIpLayers.Add(ip_layer);

                ip_layer.LayerSetUp(m_rgIpBtm, m_rgIpTop);
                blobs.Add(ip_layer.blobs);
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            for (int i = 0; i < m_param.numeric_trans_param.num_input; i++)
            {
                m_rgIpBtm[0] = m_rgBtm[i];
                m_rgIpTop[0] = colTop[i];
                m_rgIpLayers[i].Reshape(m_rgIpBtm, m_rgIpTop);
            }
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times num_input \times 1) @f$ 
        ///     the inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length num_input)
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the computed outputs
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            for (int i = 0; i < m_param.numeric_trans_param.num_input; i++)
            {
                int nCount = m_rgBtm[i].count();
                m_cuda.channel_copy(nCount, nCount, 1, (int)m_param.numeric_trans_param.num_input, 1, i, colBottom[0].gpu_data, m_rgBtm[i].mutable_gpu_data, DIR.FWD);

                m_rgIpBtm[0] = m_rgBtm[i];
                m_rgIpTop[0] = colTop[i];
                m_rgIpLayers[i].Forward(m_rgIpBtm, m_rgIpTop);
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the numeric value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length num_input), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times num_input \times 1) @f$
        ///     the inputs @f$ x @f$; Backward fills their diff
        ///     @f$ if propagate_down[0]
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            for (int i = 0; i < m_param.numeric_trans_param.num_input; i++)
            {
                m_rgIpBtm[0] = m_rgBtm[i];
                m_rgIpTop[0] = colTop[i];
                m_rgIpLayers[i].Backward(m_rgIpTop, rgbPropagateDown, m_rgIpBtm);

                int nCount = m_rgIpBtm[0].count();
                m_cuda.channel_copy(nCount, nCount, 1, (int)m_param.numeric_trans_param.num_input, 1, i, colBottom[0].mutable_gpu_diff, m_rgIpBtm[0].gpu_diff, DIR.BWD);
            }
        }
    }
}
