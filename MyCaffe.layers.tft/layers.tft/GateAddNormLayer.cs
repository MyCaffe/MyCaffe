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
    /// The GateAddNormLayer implements the Dropout, Gated Linear Unit layer, LayerNorm while adding in the residual.
    /// </summary>
    /// <remarks>
    /// The composite operation includes:
    /// a. Dropout
    /// b. Gating using GLU (Gated Linear Unit)
    /// c. A residual connection to 'earlier' signal from the forward pass of the parent model.
    /// d. Layer Normalization.
    /// 
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L405) by Playtika Research, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class GateAddNormLayer<T> : Layer<T>
    {
        Layer<T> m_dropout = null;
        Layer<T> m_gate = null;
        Layer<T> m_layerNorm = null;
        Blob<T> m_blobDrop = null;
        Blob<T> m_blobGate = null;
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public GateAddNormLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.GATEADDNORM;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobGate);
            dispose(ref m_blobDrop);

            if (m_dropout != null)
            {
                m_dropout.Dispose();
                m_dropout = null;
            }

            if (m_gate != null)
            {
                m_gate.Dispose();
                m_gate = null;
            }

            if (m_layerNorm != null)
            {
                m_layerNorm.Dispose();
                m_layerNorm = null;
            }
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            if (m_blobDrop != null)
                col.Add(m_blobDrop);
            col.Add(m_blobGate);
        }

        /// <summary>
        /// Returns the min number of required bottom (input) Blobs: x
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the max number of required bottom (input) Blobs: x, residual
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 2; }
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

        private List<int> reshapeAcrossTime(params Blob<T>[] rgb)
        {
            List<int> rgOriginalShape = Utility.Clone<int>(rgb[0].shape());
            int nOuter = rgb[0].count(0, rgb[0].num_axes - 1);
            int nInner = rgb[0].shape().Last();
            List<int> rgShape = new List<int>() { nOuter, nInner };

            foreach (Blob<T> b in rgb)
            {
                b.Reshape(rgShape);
            }

            return rgOriginalShape;
        }

        private void reshape(List<int> rgShape, params Blob<T>[] rgb)
        {
            foreach (Blob<T> b in rgb)
            {
                b.Reshape(rgShape);
            }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            LayerParameter p;
            Blob<T> blobBtm = colBottom[0];

            if (m_param.dropout_param != null && m_param.dropout_param.dropout_ratio > 0)
            {
                p = new LayerParameter(LayerParameter.LayerType.DROPOUT, "drop");
                p.dropout_param.Copy(m_param.dropout_param);
                m_dropout = Layer<T>.Create(m_cuda, m_log, p, null);
                m_blobDrop = new Blob<T>(m_cuda, m_log);

                addBtmTop(colBottom[0], m_blobDrop);
                m_dropout.Setup(m_colBtm, m_colTop);
                blobBtm = m_blobDrop;
            }

            p = new LayerParameter(LayerParameter.LayerType.GLU, "glu");
            p.glu_param.Copy(m_param.glu_param);
            m_gate = Layer<T>.Create(m_cuda, m_log, p, null);
            m_blobGate = new Blob<T>(m_cuda, m_log);

            List<int> rgShape = reshapeAcrossTime(blobBtm, m_blobGate);
            addBtmTop(blobBtm, m_blobGate);
            m_gate.Setup(m_colBtm, m_colTop);
            reshape(rgShape, blobBtm, m_blobGate);
            blobs.Add(m_gate.blobs);

            p = new LayerParameter(LayerParameter.LayerType.LAYERNORM, "layernorm");
            p.layer_norm_param.Copy(m_param.layer_norm_param);
            m_layerNorm = Layer<T>.Create(m_cuda, m_log, p, null);
            addBtmTop(m_blobGate, colTop[0]);
            m_layerNorm.Setup(m_colBtm, m_colTop);
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobBtm = colBottom[0];

            if (m_dropout != null)
            {
                addBtmTop(colBottom[0], m_blobDrop);
                m_dropout.Reshape(m_colBtm, m_colTop);
                blobBtm = m_blobDrop;
            }

            List<int> rgShape = reshapeAcrossTime(blobBtm, m_blobGate);
            addBtmTop(blobBtm, m_blobGate);
            m_gate.Reshape(m_colBtm, m_colTop);
            reshape(rgShape, blobBtm, m_blobGate);

            addBtmTop(m_blobGate, colTop[0]);
            m_layerNorm.Reshape(m_colBtm, m_colTop);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector)
        ///  -# @f$ (N \times C \times H \times W size) @f$
        ///     the computed outputs @f$ y @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobBtm = colBottom[0];

            if (m_dropout != null)
            {
                addBtmTop(colBottom[0], m_blobDrop);
                m_dropout.Forward(m_colBtm, m_colTop);
                blobBtm = m_blobDrop;
            }

            List<int> rgShape = reshapeAcrossTime(blobBtm, m_blobGate);
            addBtmTop(blobBtm, m_blobGate);
            m_gate.Forward(m_colBtm, m_colTop);
            reshape(rgShape, blobBtm, m_blobGate);

            if (colBottom.Count > 1)
                m_cuda.add(m_blobGate.count(), m_blobGate.gpu_data, colBottom[1].gpu_data, m_blobGate.mutable_gpu_data);

            addBtmTop(m_blobGate, colTop[0]);
            m_layerNorm.Forward(m_colBtm, m_colTop);
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
            addBtmTop(m_blobGate, colTop[0]);
            m_layerNorm.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            if (colBottom.Count > 1)
                colBottom[1].CopyFrom(m_blobGate, true);

            Blob<T> blobBtm = colBottom[0];
            if (m_dropout != null)
                blobBtm = m_blobDrop;

            List<int> rgShape = reshapeAcrossTime(blobBtm, m_blobGate);
            addBtmTop(blobBtm, m_blobGate);
            m_gate.Backward(m_colTop, rgbPropagateDown, m_colBtm);
            reshape(rgShape, blobBtm, m_blobGate);

            if (m_dropout != null)
            {
                addBtmTop(m_blobDrop, colBottom[0]);
                m_dropout.Backward(m_colTop, rgbPropagateDown, m_colBtm);
            }
        }
    }
}
