using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using System.Diagnostics;

namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The CausalSelfAttention provides a vanilla multi-head self-attention layer with projection at the end.
    /// </summary>
    /// <remarks>
    /// @see [GitHub:model:CausalSelfAttention](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) by Karpathy, 2022, GitHub:Karpathy
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class CausalSelfAttentionLayer2<T> : Layer<T>
    {
        // Causal mask to ensure that atttention is only applied to the left in the input sequence.
        Blob<T> m_blobBias;
        Layer<T> m_mh_att = null;

        BlobCollection<T> m_colInternalBottom = new BlobCollection<T>();
        BlobCollection<T> m_colInternalTop = new BlobCollection<T>();

        /// <summary>
        /// The CausalSelfAttention constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter inner_product_param, with options:
        /// </param>
        public CausalSelfAttentionLayer2(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.CAUSAL_SELF_ATTENTION;

            LayerParameter p1 = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION);
            p1.multihead_attention_param.heads = p.causal_self_attention_param.heads;
            p1.multihead_attention_param.embed = p.causal_self_attention_param.embed;
            p1.multihead_attention_param.block_size = p.causal_self_attention_param.block_size;
            p1.multihead_attention_param.attn_dropout = p.causal_self_attention_param.attn_dropout;
            p1.multihead_attention_param.resid_dropout = p.causal_self_attention_param.resid_dropout;
            p1.multihead_attention_param.weight_init = param.gpt.MultiheadAttentionParameter.WEIGHT_INIT.GPT;
            m_mh_att = new MultiheadAttentionLayer<T>(m_cuda, m_log, p1);

            // Causal mask to ensure that atttention is only applied to the left in the input sequence.
            m_blobBias = new Blob<T>(cuda, log);
            m_blobBias.Name = m_param.name + " bias";

            List<int> rgShape = new List<int>() { 1, 1, (int)p.causal_self_attention_param.block_size, (int)p.causal_self_attention_param.block_size };
            shareLayerBlob(m_blobBias, rgShape);
            m_blobBias.Reshape(rgShape);
            fillBias(m_blobBias);

            setup_internal_blobs(m_colInternalBlobs);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_mh_att);
            dispose(ref m_blobBias);

            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobBias);

            col.Add(m_mh_att.internal_blobs);
        }

        private void fillBias(Blob<T> b)
        {
            b.SetData(1.0);

            float[] rgBiasData = convertF(b.mutable_cpu_data);

            for (int i = 0; i<b.height; i++)
            {
                for (int j = i + 1; j < b.width; j++)
                {
                    rgBiasData[i * b.width + j] = 0;
                }
            }

            b.mutable_cpu_data = convert(rgBiasData);
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: attn
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Re-initialize the parameters of the layer.
        /// </summary>
        /// <param name="target">Specifies the weights to target (e.g. weights, bias or both).</param>
        /// <returns>When handled, this method returns <i>true</i>, otherwise <i>false</i>.</returns>
        public override bool ReInitializeParameters(WEIGHT_TARGET target)
        {
            base.ReInitializeParameters(target);

            m_mh_att.ReInitializeParameters(target);

            return true;
        }

        private void addInternal(Blob<T> bottom, Blob<T> top)
        {
            m_colInternalBottom.Clear();
            m_colInternalBottom.Add(bottom);

            m_colInternalTop.Clear();
            m_colInternalTop.Add(top);
        }

        private void addInternal(List<Blob<T>> rgBottom, Blob<T> top)
        {
            m_colInternalBottom.Clear();

            for (int i=0; i<rgBottom.Count; i++)
            {
                m_colInternalBottom.Add(rgBottom[i]);
            }

            m_colInternalTop.Clear();
            m_colInternalTop.Add(top);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobX = colBottom[0];

            addInternal(new List<Blob<T>> { blobX, blobX, blobX, m_blobBias }, colTop[0]);
            m_mh_att.LayerSetUp(m_colInternalBottom, m_colInternalTop);

            blobs.Add(m_mh_att.blobs);

            foreach (Blob<T> blob in blobs)
            {
                if (!blob.Name.StartsWith(m_param.name + "_"))
                    blob.Name = m_param.name + "_" + blob.Name;
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobX = colBottom[0];

            addInternal(new List<Blob<T>> { blobX, blobX, blobX, m_blobBias }, colTop[0]);
            m_mh_att.Reshape(m_colInternalBottom, m_colInternalTop);
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed causal self attention.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobX = colBottom[0];

            addInternal(new List<Blob<T>> { blobX, blobX, blobX, m_blobBias }, colTop[0]);
            m_mh_att.Forward(m_colInternalBottom, m_colInternalTop);
        }

        /// <summary>
        /// Computes the loss error gradient w.r.t the outputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///   -# @f$ (N \times K \times H \times W) @f$.
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            // Gradient with respect to state then data.
            if (rgbPropagateDown[0])
            {
                List<bool> rgbPropagate = new List<bool>() { true, true };

                Blob<T> blobX = colBottom[0];

                addInternal(new List<Blob<T>> { blobX, blobX, blobX, m_blobBias }, colTop[0]);
                m_mh_att.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
            }
        }
    }
}
