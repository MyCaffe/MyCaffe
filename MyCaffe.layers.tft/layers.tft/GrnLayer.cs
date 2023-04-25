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
    /// The GrnLayer implements the Gated Linear Unit layer.
    /// </summary>
    /// <remarks>
    /// This layer takes as input a primary input 'x' and optional context vector 'c'.  A GLU (Gated Linear Unit) is used
    /// for controlling the extent to which the module will contribute to the original input 'x', potentially skipping
    /// over the layer entirely as the GLU outputs could all be close to zero by the GLU supressing.  In cases where
    /// no context vector is used, the GRN treats the context input as zero.  During training dropout is applied before
    /// gating.
    /// 
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L44) by Playtika Research, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class GrnLayer<T> : Layer<T>
    {
        Layer<T> m_ipSkipLayer = null;
        Layer<T> m_ipFc1 = null;
        Layer<T> m_ipContext = null;
        Layer<T> m_act = null;
        Layer<T> m_ipFc2 = null;
        Layer<T> m_dropout = null;
        Layer<T> m_gate = null;
        Layer<T> m_layerNorm = null;
        Blob<T> m_blobResidual = null;
        Blob<T> m_blobIp1 = null;
        Blob<T> m_blobContext = null;
        Blob<T> m_blobContextAdd = null;
        Blob<T> m_blobIp2 = null;
        Blob<T> m_blobGate = null;
        Blob<T> m_blobGatePlusResidual = null;
        Blob<T> m_blobBtm = null;
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public GrnLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.GRN;

            if (m_param.grn_param.input_dim != m_param.grn_param.output_dim)
                m_blobResidual = new Blob<T>(cuda, log);

            m_blobIp1 = new Blob<T>(cuda, log);
            m_blobIp1.Name = p.name + ".ip1";
            m_blobIp2 = new Blob<T>(cuda, log);
            m_blobIp2.Name = p.name + ".ip2";
            m_blobGate = new Blob<T>(cuda, log);
            m_blobGate.Name = p.name + ".gate";
            m_blobGatePlusResidual = new Blob<T>(cuda, log);
            m_blobGatePlusResidual.Name = p.name + ".gate_p_res";
            m_blobBtm = new Blob<T>(cuda, log);
            m_blobBtm.Name = p.name + ".btm";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobResidual);
            dispose(ref m_blobIp1);
            dispose(ref m_blobContext);
            dispose(ref m_blobContextAdd);
            dispose(ref m_blobIp2);
            dispose(ref m_blobGate);
            dispose(ref m_blobGatePlusResidual);
            dispose(ref m_blobBtm);

            dispose(ref m_ipSkipLayer);
            dispose(ref m_ipFc1);
            dispose(ref m_ipContext);
            dispose(ref m_act);
            dispose(ref m_ipFc2);
            dispose(ref m_dropout);
            dispose(ref m_gate);
            dispose(ref m_layerNorm);
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            if (m_blobContext != null)
                col.Add(m_blobContext);
            if (m_blobContextAdd != null)
                col.Add(m_blobContextAdd);
            if (m_blobResidual != null)
                col.Add(m_blobResidual);
            col.Add(m_blobIp1);
            col.Add(m_blobIp2);
            col.Add(m_blobGate);
            col.Add(m_blobGatePlusResidual);
        }

        /// <summary>
        /// Returns the min number of required bottom (input) Blobs: x
        /// </summary>
        public override int MinBottomBlobs 
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the max number of required bottom (input) Blobs: x, context
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

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            //-------------------------------------------------------
            // Input conditioning components (Eq.4 in original paper)
            //-------------------------------------------------------

            // For a direct residual connection, the dimension of the input must match the dimension of the output,
            // otherwise we need to project the input for creating the residual connection.
            if (m_param.grn_param.input_dim != m_param.grn_param.output_dim)
            {
                LayerParameter ip = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, m_param.name + ".skip");
                ip.inner_product_param.num_output = (uint)m_param.grn_param.output_dim;
                ip.inner_product_param.axis = m_param.grn_param.axis;
                ip.inner_product_param.weight_filler = m_param.grn_param.weight_filler;
                ip.inner_product_param.bias_filler = m_param.grn_param.bias_filler;
                m_ipSkipLayer = Layer<T>.Create(m_cuda, m_log, ip, null);

                addBtmTop(colBottom[0], m_blobResidual);
                m_ipSkipLayer.Setup(m_colBtm, m_colTop);
                blobs.Add(m_ipSkipLayer.blobs);
            }

            // Create the linear layer for projecting the primary input (across time if necessary)
            LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, m_param.name + ".fc1");
            ip1.inner_product_param.num_output = (uint)m_param.grn_param.hidden_dim;
            ip1.inner_product_param.axis = m_param.grn_param.axis;
            ip1.inner_product_param.weight_filler = m_param.grn_param.weight_filler;
            ip1.inner_product_param.bias_filler = m_param.grn_param.bias_filler;
            m_ipFc1 = Layer<T>.Create(m_cuda, m_log, ip1, null);
            m_blobIp1 = new Blob<T>(m_cuda, m_log);

            addBtmTop(colBottom[0], m_blobIp1);
            m_ipFc1.Setup(m_colBtm, m_colTop);
            blobs.Add(m_ipFc1.blobs);
            Blob<T> blobIp1 = m_blobIp1;

            // If a context input exists, project the context as well.
            if (colBottom.Count > 1)
            {
                LayerParameter ip = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, m_param.name + ".context");
                ip.inner_product_param.num_output = (uint)m_param.grn_param.hidden_dim;
                ip.inner_product_param.axis = m_param.grn_param.axis;
                ip.inner_product_param.weight_filler = m_param.grn_param.weight_filler;
                ip.inner_product_param.bias_term = false;
                m_ipContext = Layer<T>.Create(m_cuda, m_log, ip, null);
                m_blobContext = new Blob<T>(m_cuda, m_log);
                m_blobContext.Name = m_param.name + ".ctx";
                m_blobContextAdd = new Blob<T>(m_cuda, m_log);
                m_blobContextAdd.Name = m_param.name + ".ctx_add";

                addBtmTop(colBottom[1], m_blobContext);
                m_ipContext.Setup(m_colBtm, m_colTop);
                blobs.Add(m_ipContext.blobs);

                m_cuda.add(m_blobContext.count(), m_blobContext.gpu_data, m_blobIp1.gpu_data, m_blobContext.mutable_gpu_data);
                blobIp1 = m_blobContext;
            }

            // non-linear activation function applied to the sum of projections.
            LayerParameter act = new LayerParameter(LayerParameter.LayerType.ELU, m_param.name + ".act");
            act.elu_param.engine = EngineParameter.Engine.CAFFE;
            act.elu_param.alpha = 1.0;
            m_act = Layer<T>.Create(m_cuda, m_log, act, null);

            addBtmTop(blobIp1, blobIp1);
            m_act.Setup(m_colBtm, m_colTop);

            //-------------------------------------------------------
            // Further projection components (Eq.3 in original paper)
            //-------------------------------------------------------

            // Create the linear layer for projecting top of the activation function
            LayerParameter ip2 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, m_param.name + ".fc2");
            ip2.inner_product_param.num_output = (uint)m_param.grn_param.output_dim;
            ip2.inner_product_param.axis = m_param.grn_param.axis;
            ip2.inner_product_param.weight_filler = m_param.grn_param.weight_filler;
            ip2.inner_product_param.bias_filler = m_param.grn_param.bias_filler;
            m_ipFc2 = Layer<T>.Create(m_cuda, m_log, ip2, null);

            addBtmTop(blobIp1, m_blobIp2);
            m_ipFc2.Setup(m_colBtm, m_colTop);
            blobs.Add(m_ipFc2.blobs);

            //-------------------------------------------------------
            // Output gating components (Eq.2 in original paper)
            //-------------------------------------------------------

            if (m_param.grn_param.dropout > 0)
            {
                LayerParameter drop = new LayerParameter(LayerParameter.LayerType.DROPOUT, m_param.name + ".drop");
                drop.dropout_param.dropout_ratio = m_param.grn_param.dropout;
                m_dropout = Layer<T>.Create(m_cuda, m_log, drop, null);

                addBtmTop(m_blobIp2, m_blobIp2);
                m_dropout.Setup(m_colBtm, m_colTop);
            }

            LayerParameter gate = new LayerParameter(LayerParameter.LayerType.GLU, m_param.name + ".gate");
            gate.glu_param.input_dim = m_param.grn_param.output_dim;
            gate.glu_param.axis = m_param.grn_param.axis;
            gate.glu_param.weight_filler = m_param.grn_param.weight_filler;
            gate.glu_param.bias_filler = m_param.grn_param.bias_filler;
            m_gate = Layer<T>.Create(m_cuda, m_log, gate, null);

            addBtmTop(m_blobIp2, m_blobGate);
            m_gate.Setup(m_colBtm, m_colTop);
            blobs.Add(m_gate.blobs);

            LayerParameter layerNorm = new LayerParameter(LayerParameter.LayerType.LAYERNORM, m_param.name + ".layernorm");
            layerNorm.layer_norm_param.epsilon = 1e-10;
            m_layerNorm = Layer<T>.Create(m_cuda, m_log, layerNorm, null);

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
            if (m_ipSkipLayer != null)
            {
                addBtmTop(colBottom[0], m_blobResidual);
                m_ipSkipLayer.Reshape(m_colBtm, m_colTop);
            }

            addBtmTop(colBottom[0], m_blobIp1);
            m_ipFc1.Reshape(m_colBtm, m_colTop);
            Blob<T> blobIp1 = m_blobIp1;

            if (colBottom.Count > 1)
            {
                addBtmTop(colBottom[1], m_blobContext);
                m_ipContext.Reshape(m_colBtm, m_colTop);
                m_blobContextAdd.ReshapeLike(m_blobContext);
                blobIp1 = m_blobContext;
            }

            addBtmTop(blobIp1, blobIp1);
            m_act.Reshape(m_colBtm, m_colTop);

            addBtmTop(blobIp1, m_blobIp2);
            m_ipFc2.Reshape(m_colBtm, m_colTop);

            if (m_dropout != null)
            {
                addBtmTop(m_blobIp2, m_blobIp2);
                m_dropout.Reshape(m_colBtm, m_colTop);
            }

            addBtmTop(m_blobIp2, m_blobGate);
            m_gate.Reshape(m_colBtm, m_colTop);

            m_blobGatePlusResidual.ReshapeLike(m_blobGate);

            addBtmTop(m_blobGate, colTop[0]);
            m_layerNorm.Reshape(m_colBtm, m_colTop);

            m_blobBtm.ReshapeLike(colBottom[0]);
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
            Blob<T> blobResidual = colBottom[0];

            if (m_ipSkipLayer != null)
            {
                addBtmTop(colBottom[0], m_blobResidual);
                m_ipSkipLayer.Forward(m_colBtm, m_colTop);
                blobResidual = m_blobResidual;
            }

            addBtmTop(colBottom[0], m_blobIp1);
            m_ipFc1.Forward(m_colBtm, m_colTop);
            Blob<T> blobIp1 = m_blobIp1;

            if (colBottom.Count > 1)
            {
                addBtmTop(colBottom[1], m_blobContext);
                m_ipContext.Forward(m_colBtm, m_colTop);

                m_cuda.add(m_blobContext.count(), m_blobIp1.gpu_data, m_blobContext.gpu_data, m_blobContextAdd.gpu_data);
                blobIp1 = m_blobContextAdd;
            }

            // act
            addBtmTop(blobIp1, blobIp1);
            m_act.Forward(m_colBtm, m_colTop);

            // Fc2
            addBtmTop(blobIp1, m_blobIp2);
            m_ipFc2.Forward(m_colBtm, m_colTop);

            // dropout
            if (m_dropout != null)
            {
                addBtmTop(m_blobIp2, m_blobIp2);
                m_dropout.Forward(m_colBtm, m_colTop);
            }

            // gate
            addBtmTop(m_blobIp2, m_blobGate);
            m_gate.Forward(m_colBtm, m_colTop);

            // add residual
            m_cuda.add(m_blobGatePlusResidual.count(), m_blobGate.gpu_data, blobResidual.gpu_data, m_blobGatePlusResidual.mutable_gpu_data);

            // layernorm
            addBtmTop(m_blobGatePlusResidual, colTop[0]);
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
            // layernorm
            addBtmTop(m_blobGatePlusResidual, colTop[0]);
            m_layerNorm.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            // add residual
            if (m_ipSkipLayer != null)
            {
                m_blobResidual.CopyFrom(m_blobGatePlusResidual, true);
            }
            else
            {
                colBottom[0].CopyFrom(m_blobGatePlusResidual, true, false, 0, true);
            }

            m_blobGate.CopyFrom(m_blobGatePlusResidual, true, false, 0, true);

            // gate
            addBtmTop(m_blobIp2, m_blobGate);
            m_gate.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            // dropout
            if (m_dropout != null)
            {
                addBtmTop(m_blobIp2, m_blobIp2);
                m_dropout.Backward(m_colTop, rgbPropagateDown, m_colBtm);
            }

            // Fc2
            addBtmTop(m_blobIp1, m_blobIp2);
            m_ipFc2.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            // act
            addBtmTop(m_blobIp1, m_blobIp1);
            m_act.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            if (m_ipContext != null)
            {
                m_blobContext.CopyFrom(m_blobIp1, true);
                addBtmTop(colBottom[1], m_blobContext);
                m_ipContext.Backward(m_colTop, rgbPropagateDown, m_colBtm);
            }

            addBtmTop(m_blobBtm, m_blobIp1);
            m_ipFc1.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            m_cuda.add(colBottom[0].count(), colBottom[0].gpu_diff, m_blobBtm.gpu_diff, colBottom[0].mutable_gpu_diff);

            if (m_ipSkipLayer != null)
            {
                addBtmTop(m_blobBtm, m_blobResidual);
                m_ipSkipLayer.Backward(m_colTop, rgbPropagateDown, m_colBtm);
                m_cuda.add(colBottom[0].count(), colBottom[0].gpu_diff, m_blobBtm.gpu_diff, colBottom[0].mutable_gpu_diff);
            }
        }
    }
}
