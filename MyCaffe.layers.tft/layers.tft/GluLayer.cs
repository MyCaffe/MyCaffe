using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.tft
{
    /// <summary>
    /// The GluLayer implements the Gated Linear Unit layer.
    /// </summary>
    /// <remarks>
    /// The output of the layer is a linear projection (X * W + b) modulated by the gates **sigmoid** (X * V + c).  These
    /// gates multiply each element of the matrix X * W + b and control the information passed in.  The simplified gating
    /// mechanism in this layer is for non-deterministic gates that reduce the vanishing gradient problem, by having linear
    /// units couypled to the gates.  This retains the non-linear capabilities of the layer while allowing the gradient
    /// to propagate through the linear unit without scaling.
    /// 
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L11) by Playtika Research, 2021.
    /// @see ["Language modeling with gated convolution networks](https://arxiv.org/abs/1612.08083) by Dauphin, Yann N., et al., International conference on machine learning, PMLR, 2017
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class GluLayer<T> : Layer<T>
    {
        Layer<T> m_ip1Layer;
        Layer<T> m_ip2Layer;
        Layer<T> m_modLayer;
        Blob<T> m_blobIp1;
        Blob<T> m_blobIp2;
        Blob<T> m_blobMod;
        Blob<T> m_blobBtm;
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public GluLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.GLU;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobIp1);
            dispose(ref m_blobIp2);
            dispose(ref m_blobMod);
            dispose(ref m_blobBtm);

            dispose(ref m_ip1Layer);
            dispose(ref m_ip2Layer);
            dispose(ref m_modLayer);
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: x
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: y
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        private void addBtmTop(Blob<T> btm, Blob<T> top)
        {
            m_colBtm.Clear();
            m_colBtm.Add(btm);
            m_colTop.Clear();
            m_colTop.Add(top);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_blobBtm = new Blob<T>(m_cuda, m_log);
            m_blobBtm.ReshapeLike(colBottom[0]);

            if (m_ip1Layer == null)
            {
                LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, m_param.name + ".fc1");
                ip1.inner_product_param.num_output = (uint)m_param.glu_param.input_dim;
                ip1.inner_product_param.axis = m_param.glu_param.axis;
                ip1.inner_product_param.bias_term = m_param.glu_param.bias_term;
                ip1.inner_product_param.enable_noise = m_param.glu_param.enable_noise;
                ip1.inner_product_param.sigma_init = m_param.glu_param.sigma_init;
                ip1.inner_product_param.bias_filler = m_param.glu_param.bias_filler;
                ip1.inner_product_param.weight_filler = m_param.glu_param.weight_filler;

                m_blobIp1 = new Blob<T>(m_cuda, m_log);
                m_ip1Layer = Layer<T>.Create(m_cuda, m_log, ip1, null);

                addBtmTop(colBottom[0], m_blobIp1);
                m_ip1Layer.Setup(m_colBtm, m_colTop);
                blobs.Add(m_ip1Layer.blobs);
            }

            if (m_modLayer == null)
            {
                if (m_param.glu_param.modulation == param.tft.GluParameter.MODULATION.SIGMOID)
                {
                    LayerParameter mod = new LayerParameter(LayerParameter.LayerType.SIGMOID, m_param.name + ".mod");
                    mod.sigmoid_param.engine = EngineParameter.Engine.DEFAULT;

                    m_blobMod = new Blob<T>(m_cuda, m_log);
                    m_modLayer = Layer<T>.Create(m_cuda, m_log, mod, null);

                    addBtmTop(m_blobIp1, m_blobMod);
                    m_modLayer.Setup(m_colBtm, m_colTop);
                }
                else
                {
                    m_log.FAIL("Unknown modulation type '" + m_param.glu_param.modulation.ToString() + "'");
                }
            }

            if (m_ip2Layer == null)
            {
                LayerParameter ip2 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, m_param.name + "fc2");
                ip2.inner_product_param.num_output = (uint)m_param.glu_param.input_dim;
                ip2.inner_product_param.axis = m_param.glu_param.axis;
                ip2.inner_product_param.bias_term = m_param.glu_param.bias_term;
                ip2.inner_product_param.enable_noise = m_param.glu_param.enable_noise;
                ip2.inner_product_param.sigma_init = m_param.glu_param.sigma_init;
                ip2.inner_product_param.bias_filler = m_param.glu_param.bias_filler;
                ip2.inner_product_param.weight_filler = m_param.glu_param.weight_filler;

                m_blobIp2 = new Blob<T>(m_cuda, m_log);
                m_ip2Layer = Layer<T>.Create(m_cuda, m_log, ip2, null);

                addBtmTop(colBottom[0], m_blobIp2);
                m_ip2Layer.Setup(m_colBtm, m_colTop);
                blobs.Add(m_ip2Layer.blobs);

                colTop[0].ReshapeLike(m_blobIp2);
            }
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_blobBtm.ReshapeLike(colBottom[0]);

            addBtmTop(colBottom[0], m_blobIp1);
            m_ip1Layer.Reshape(m_colBtm, m_colTop);

            addBtmTop(m_blobIp1, m_blobMod);
            m_modLayer.Reshape(m_colBtm, m_colTop);

            addBtmTop(colBottom[0], m_blobIp2);
            m_ip2Layer.Reshape(m_colBtm, m_colTop);

            colTop[0].ReshapeLike(m_blobIp2);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the numeric inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector)
        ///  -# @f$ (N \times C \times H \times W size) @f$
        ///     the computed outputs @f$ y @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_blobBtm.CopyFrom(colBottom[0]);

            addBtmTop(colBottom[0], m_blobIp1);
            m_ip1Layer.Forward(m_colBtm, m_colTop); // x1 = fc1(x)

            addBtmTop(m_blobIp1, m_blobMod);
            m_modLayer.Forward(m_colBtm, m_colTop);  // sig = sigmoid(x1)

            addBtmTop(colBottom[0], m_blobIp2);
            m_ip2Layer.Forward(m_colBtm, m_colTop);   // x2 = fc2(x)

            m_cuda.mul(colTop[0].count(), m_blobIp2.gpu_data, m_blobMod.gpu_data, colTop[0].mutable_gpu_data);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the stacked embedding numeric and categorical value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$;  
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            // sig grad = y grad * x2
            m_cuda.mul(colTop[0].count(), colTop[0].gpu_diff, m_blobIp2.gpu_data, m_blobMod.mutable_gpu_diff);

            // x2 grad = y grad * sig
            m_cuda.mul(colTop[0].count(), colTop[0].gpu_diff, m_blobMod.gpu_data, m_blobIp2.mutable_gpu_diff);

            addBtmTop(m_blobBtm, m_blobIp2);
            m_ip2Layer.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            addBtmTop(m_blobIp1, m_blobMod);
            m_modLayer.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            addBtmTop(colBottom[0], m_blobIp1);
            m_ip1Layer.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            // Add gradients from x1 and x2
            m_cuda.add(colBottom[0].count(), m_blobBtm.gpu_diff, colBottom[0].gpu_diff, colBottom[0].mutable_gpu_diff);
        }
    }
}
