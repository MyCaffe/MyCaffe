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
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L405) by Playtika Research, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class GateAddNormLayer<T> : Layer<T>
    {
        int m_nBlocks;
        Layer<T> m_dropout = null;
        Layer<T> m_gate = null;
        Layer<T> m_layerNorm = null;
        Blob<T> m_blobResidual = null;
        Blob<T> m_blobDrop = null;
        Blob<T> m_blobGate = null;
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();
        List<int> m_rgShape = new List<int>(4);

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

            if (m_param.dropout_param != null && m_param.dropout_param.dropout_ratio > 0)
            {
                m_blobDrop = new Blob<T>(cuda, log);
                m_blobDrop.Name = p.name + ".drop";
            }

            m_blobResidual = new Blob<T>(cuda, log);
            m_blobResidual.Name = p.name + ".residual";
            m_blobGate = new Blob<T>(cuda, log);
            m_blobGate.Name = p.name + ".gate";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobResidual);
            dispose(ref m_blobGate);
            dispose(ref m_blobDrop);

            dispose(ref m_dropout);
            dispose(ref m_gate);
            dispose(ref m_layerNorm);
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            if (m_blobDrop != null)
                col.Add(m_blobDrop);
            col.Add(m_blobGate);
            col.Add(m_blobResidual);
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

            if (m_param.gateaddnorm_param.residual_channel_offset > 0)
            {
                int nDiff = colBottom[1].channels - m_param.gateaddnorm_param.residual_channel_offset;
                if (colBottom[1].channels % nDiff != 0)
                    m_log.FAIL("The number bottom(1).channels must be divisible by the bottom(1).channels - the residual channel offset. For example if bottom(1).channels = 120 and redidual_channel_offset = 90, the difference = 30 which is a factor of both 120 and 90.");
            }

            if (m_param.dropout_param != null && m_param.dropout_param.dropout_ratio > 0)
            {
                p = new LayerParameter(LayerParameter.LayerType.DROPOUT, "drop");
                p.dropout_param.Copy(m_param.dropout_param);
                m_dropout = Layer<T>.Create(m_cuda, m_log, p, null);

                addBtmTop(colBottom[0], m_blobDrop);
                m_dropout.Setup(m_colBtm, m_colTop);
                blobBtm = m_blobDrop;
            }

            p = new LayerParameter(LayerParameter.LayerType.GLU, "glu");
            p.glu_param.Copy(m_param.glu_param);
            m_gate = Layer<T>.Create(m_cuda, m_log, p, null);

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

            setup_internal_blobs(m_colInternalBlobs);
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobBtm = colBottom[0];

            if (colBottom.Count > 1)
            {
                if (m_param.gateaddnorm_param.residual_channel_offset > 0)
                {
                    int nDiff = colBottom[1].channels - m_param.gateaddnorm_param.residual_channel_offset;
                    m_log.CHECK_EQ(colBottom[1].channels % nDiff, 0, "The bottom(1).channels must be divisible by the bottom(1).channels - residual_channel_offset!");
                    m_nBlocks = colBottom[1].channels / nDiff;

                    int nQTimeSteps = nDiff;
                    m_rgShape.Clear();
                    m_rgShape.Add(colBottom[0].num);
                    m_rgShape.Add(nQTimeSteps);
                    m_rgShape.Add(colBottom[0].count(2));
                    m_blobResidual.Reshape(m_rgShape);
                }
                else
                {
                    m_blobResidual.ReshapeLike(colBottom[1]);
                }
            }

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

        private void copy_to_fwd(BlobCollection<T> colBtm, Blob<T> bTop)
        {
            if (colBtm.Count < 2)
                return;

            Blob<T> bBtm = colBtm[1];

            if (m_param.gateaddnorm_param.residual_channel_offset > 0)
            {
                // Copy just the future items to the top, so if future = 30,
                // with input shape is btm(256,120,64) just the last (256,30,64) are copied to top 
                int nOuterNum = bBtm.num;
                int nChannels = m_nBlocks;
                int nInnerNum = (bBtm.channels / m_nBlocks) * bBtm.count(2);
                m_cuda.channel_copy(bTop.count(), nOuterNum, nChannels, m_nBlocks, nInnerNum, m_nBlocks-1, bBtm.gpu_data, bTop.mutable_gpu_data, DIR.FWD);
            }
            else
            {
                bTop.CopyFrom(bBtm);
            }
        }

        private void copy_to_bwd(BlobCollection<T> colBtm, Blob<T> bTop)
        {
            if (colBtm.Count < 2)
                return;

            Blob<T> bBtm = colBtm[1];

            if (m_param.gateaddnorm_param.residual_channel_offset > 0)
            {
                // Copy just the future items to the top, so if future = 30,
                // with input shape is btm(256,120,64) just the last (256,30,64) are copied to top 
                int nOuterNum = bBtm.num;
                int nChannels = m_nBlocks;
                int nInnerNum = (bBtm.channels / m_nBlocks) * bBtm.count(2);
                m_cuda.channel_add(bBtm.count(), nOuterNum, nChannels, m_nBlocks, nInnerNum, m_nBlocks-1, bBtm.mutable_gpu_diff, bTop.gpu_diff, DIR.BWD);
            }
            else
            {
                bTop.CopyFrom(bBtm, true);
            }
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
            copy_to_fwd(colBottom, m_blobResidual);
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

            if (colBottom.Count > 1)
                m_cuda.add(m_blobGate.count(), m_blobGate.gpu_data, m_blobResidual.gpu_data, m_blobGate.mutable_gpu_data);

            addBtmTop(m_blobGate, colTop[0]);
            m_layerNorm.Forward(m_colBtm, m_colTop);

            reshape(rgShape, blobBtm, m_blobGate);
            colTop[0].ReshapeLike(m_blobGate);
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
            List<int> rgShape = reshapeAcrossTime(colTop[0], m_blobGate);

            addBtmTop(m_blobGate, colTop[0]);
            m_layerNorm.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            copy_to_bwd(colBottom, m_blobResidual);

            addBtmTop(colBottom[0], m_blobGate);
            m_gate.Backward(m_colTop, rgbPropagateDown, m_colBtm);
            reshape(rgShape, colBottom[0], m_blobGate);

            if (m_dropout != null)
            {
                addBtmTop(m_blobDrop, colBottom[0]);
                m_dropout.Backward(m_colTop, rgbPropagateDown, m_colBtm);
                colBottom[0].CopyFrom(m_blobDrop, true);
            }
        }
    }
}
