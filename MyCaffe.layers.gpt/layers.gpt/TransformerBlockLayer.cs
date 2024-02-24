using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using System.Diagnostics;
using MyCaffe.param.gpt;
using System.Runtime.InteropServices.WindowsRuntime;

namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The TransformerBlock provides a generic transformer block
    /// </summary>
    /// <remarks>
    /// @see [GitHub:model:TransformerBlock](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py) by Karpathy, 2022, GitHub:Karpathy
    /// @see [GitHub:devjwsong:transformer-translator-pytorch](https://github.com/devjwsong/transformer-translator-pytorch/blob/master/src/layers.py) by Song, 2021, GitHub:devjwsong
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TransformerBlockLayer<T> : Layer<T>
    {
        Blob<T> m_blobNorm1;
        Blob<T> m_blobAttn1;
        Blob<T> m_blobNorm2;
        Blob<T> m_blobAttn2 = null;
        Blob<T> m_blobNorm3 = null;
        Blob<T> m_blobNormA = null;
        Blob<T> m_blobMlp1;
        Blob<T> m_blobMlp2 = null;
        Blob<T> m_blobMlp3 = null;
        Blob<T> m_blobMlpOut;
        Blob<T> m_blobX = null;
        Layer<T> m_norm1;          // Input layer normalization.
        Layer<T> m_attn1;        // Attention block used with encoder and decoder        
        Layer<T> m_norm2;          // Layer normalization after the first attention block
        Layer<T> m_attn2 = null; // Attention block used with decoder only.
        Layer<T> m_norm3 = null;   // Layer normalization after second attention block, used with decoder only.
        // MLP block
        Layer<T> m_fc1;          // FF initial linear
        Layer<T> m_fc2 = null;
        Layer<T> m_proj;         // FF projection
        Layer<T> m_act;          // FF activation
        Layer<T> m_dropout = null; // resid dropout

        BlobCollection<T> m_colInternalBottom = new BlobCollection<T>();
        BlobCollection<T> m_colInternalTop = new BlobCollection<T>();

        /// <summary>
        /// The TransformerBlock constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter inner_product_param, with options:
        /// </param>
        public TransformerBlockLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            CancelEvent evtCancel = new CancelEvent();

            m_type = LayerParameter.LayerType.TRANSFORMER_BLOCK;

            m_blobNorm1 = new Blob<T>(cuda, log);
            m_blobNorm1.Name = m_param.name + ".norm1";
            m_blobAttn1 = new Blob<T>(cuda, log);
            m_blobAttn1.Name = m_param.name + ".attn1";
            m_blobNorm2 = new Blob<T>(cuda, log);
            m_blobNorm2.Name = m_param.name + ".norm2";
            m_blobMlp1 = new Blob<T>(cuda, log);
            m_blobMlp1.Name = m_param.name + ".mlp1";
            if (p.transformer_block_param.enable_llama_style_head)
            {
                m_blobMlp2 = new Blob<T>(cuda, log);
                m_blobMlp2.Name = m_param.name + ".mlp2";
                m_blobMlp3 = new Blob<T>(cuda, log);
                m_blobMlp3.Name = m_param.name + ".mlp3";
                m_blobNormA = new Blob<T>(cuda, log, false);
                m_blobNormA.Name = m_param.name + ".normA";
            }
            m_blobMlpOut = new Blob<T>(cuda, log);
            m_blobMlpOut.Name = m_param.name + ".mlp_out";
            m_blobX = new Blob<T>(cuda, log);
            m_blobX.Name = m_param.name + ".xB";

            if (p.transformer_block_param.normalization_type == TransformerBlockParameter.NORMALIZATION.RMS_NORM)
            {
                LayerParameter rms1 = new LayerParameter(LayerParameter.LayerType.RMSNORM, p.name + ".rms1", m_phase, p.freeze_learning);
                rms1.rms_norm_param.axis = 2;
                m_norm1 = Layer<T>.Create(cuda, log, convertLayerParam(rms1, p), evtCancel) as Layer<T>;

                LayerParameter rms2 = new LayerParameter(LayerParameter.LayerType.RMSNORM, p.name + ".rms2", m_phase, p.freeze_learning);
                rms2.rms_norm_param.axis = 2;
                m_norm2 = Layer<T>.Create(cuda, log, convertLayerParam(rms2, p), evtCancel) as Layer<T>;
            }
            else
            {
                LayerParameter ln1 = new LayerParameter(LayerParameter.LayerType.LAYERNORM, p.name + ".ln1", m_phase, p.freeze_learning);
                ln1.layer_norm_param.enable_cuda_impl = p.transformer_block_param.enable_layernorm_cuda_impl;
                m_norm1 = Layer<T>.Create(cuda, log, convertLayerParam(ln1, p), evtCancel) as Layer<T>;

                LayerParameter ln2 = new LayerParameter(LayerParameter.LayerType.LAYERNORM, p.name + ".ln2", m_phase, p.freeze_learning);
                ln2.layer_norm_param.enable_cuda_impl = p.transformer_block_param.enable_layernorm_cuda_impl;
                m_norm2 = Layer<T>.Create(cuda, log, convertLayerParam(ln2, p), evtCancel) as Layer<T>;
            }

            if (p.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION)
            {
                LayerParameter attn = new LayerParameter(LayerParameter.LayerType.CAUSAL_SELF_ATTENTION, p.name + ".attn", m_phase, p.freeze_learning);
                attn.causal_self_attention_param.block_size = p.transformer_block_param.block_size;
                attn.causal_self_attention_param.embed = p.transformer_block_param.embed;
                attn.causal_self_attention_param.heads = p.transformer_block_param.heads;
                attn.causal_self_attention_param.attn_dropout = p.transformer_block_param.attn_dropout;
                attn.causal_self_attention_param.resid_dropout = p.transformer_block_param.resid_dropout;
                attn.causal_self_attention_param.layers = p.transformer_block_param.layers;
                attn.parameters.Add((m_param.parameters.Count > 0) ? m_param.parameters[0] : new ParamSpec(1.0, 1.0));
                attn.parameters.Add((m_param.parameters.Count > 1) ? m_param.parameters[1] : new ParamSpec(1.0, 0.0));
                m_attn1 = Layer<T>.Create(cuda, log, convertLayerParam(attn, p), evtCancel);
            }
            else if (p.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION2)
            {
                LayerParameter attn = new LayerParameter(LayerParameter.LayerType.CAUSAL_SELF_ATTENTION, p.name + ".attn", m_phase, p.freeze_learning);
                attn.causal_self_attention_param.block_size = p.transformer_block_param.block_size;
                attn.causal_self_attention_param.embed = p.transformer_block_param.embed;
                attn.causal_self_attention_param.heads = p.transformer_block_param.heads;
                attn.causal_self_attention_param.attn_dropout = p.transformer_block_param.attn_dropout;
                attn.causal_self_attention_param.resid_dropout = p.transformer_block_param.resid_dropout;
                attn.causal_self_attention_param.layers = p.transformer_block_param.layers;
                attn.causal_self_attention_param.output_adapter_q = p.transformer_block_param.output_adapter_q;
                attn.causal_self_attention_param.output_adapter_k = p.transformer_block_param.output_adapter_k;
                attn.causal_self_attention_param.output_adapter_v = p.transformer_block_param.output_adapter_v;
                attn.causal_self_attention_param.output_adapter_out = p.transformer_block_param.output_adapter_out;
                attn.causal_self_attention_param.enable_rotary_positional_embedding = p.transformer_block_param.enable_rotary_positional_embedding;
                attn.causal_self_attention_param.bias_term = p.transformer_block_param.bias_term;
                attn.parameters.Add((m_param.parameters.Count > 0) ? m_param.parameters[0] : new ParamSpec(1.0, 1.0));
                attn.parameters.Add((m_param.parameters.Count > 1) ? m_param.parameters[1] : new ParamSpec(1.0, 0.0));
                m_attn1 = new CausalSelfAttentionLayer2<T>(cuda, log, convertLayerParam(attn, p));
            }
            else if (p.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.ENCODER)
            {
                LayerParameter attn = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION, p.name + ".attn", m_phase, p.freeze_learning);
                attn.multihead_attention_param.block_size = p.transformer_block_param.block_size;
                attn.multihead_attention_param.embed = p.transformer_block_param.embed;
                attn.multihead_attention_param.heads = p.transformer_block_param.heads;
                attn.multihead_attention_param.attn_dropout = p.transformer_block_param.attn_dropout;
                attn.multihead_attention_param.resid_dropout = p.transformer_block_param.resid_dropout;
                attn.multihead_attention_param.layers = p.transformer_block_param.layers;
                attn.multihead_attention_param.output_adapter_q = p.transformer_block_param.output_adapter_q;
                attn.multihead_attention_param.output_adapter_k = p.transformer_block_param.output_adapter_k;
                attn.multihead_attention_param.output_adapter_v = p.transformer_block_param.output_adapter_v;
                attn.multihead_attention_param.output_adapter_out = p.transformer_block_param.output_adapter_out;
                attn.multihead_attention_param.weight_init = MultiheadAttentionParameter.WEIGHT_INIT.ENCODER_DECODER;
                attn.multihead_attention_param.enable_rotary_positional_embedding = p.transformer_block_param.enable_rotary_positional_embedding;
                attn.multihead_attention_param.bias_term = p.transformer_block_param.bias_term;
                attn.parameters.Add((m_param.parameters.Count > 0) ? m_param.parameters[0] : new ParamSpec(1.0, 1.0));
                attn.parameters.Add((m_param.parameters.Count > 1) ? m_param.parameters[1] : new ParamSpec(1.0, 0.0));
                m_attn1 = Layer<T>.Create(cuda, log, convertLayerParam(attn, p), evtCancel);
            }
            else if (p.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.DECODER)
            {
                m_blobAttn2 = new Blob<T>(cuda, log);
                m_blobAttn2.Name = m_param.name + " attn2";
                m_blobNorm3 = new Blob<T>(cuda, log);
                m_blobNorm3.Name = m_param.name + " norm3";

                if (p.transformer_block_param.normalization_type == TransformerBlockParameter.NORMALIZATION.RMS_NORM)
                {
                    LayerParameter rms3 = new LayerParameter(LayerParameter.LayerType.RMSNORM, p.name + ".rms3", m_phase, p.freeze_learning);
                    rms3.rms_norm_param.axis = 2;
                    m_norm3 = Layer<T>.Create(cuda, log, convertLayerParam(rms3, p), evtCancel) as Layer<T>;
                }
                else
                {
                    LayerParameter ln3 = new LayerParameter(LayerParameter.LayerType.LAYERNORM, p.name + ".ln3", m_phase, p.freeze_learning);
                    ln3.layer_norm_param.enable_cuda_impl = p.transformer_block_param.enable_layernorm_cuda_impl;
                    m_norm3 = Layer<T>.Create(cuda, log, convertLayerParam(ln3, p), evtCancel) as Layer<T>;
                }

                LayerParameter attn1 = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION, p.name + ".attn1", m_phase, p.freeze_learning);
                attn1.multihead_attention_param.block_size = p.transformer_block_param.block_size;
                attn1.multihead_attention_param.embed = p.transformer_block_param.embed;
                attn1.multihead_attention_param.heads = p.transformer_block_param.heads;
                attn1.multihead_attention_param.attn_dropout = p.transformer_block_param.attn_dropout;
                attn1.multihead_attention_param.resid_dropout = p.transformer_block_param.resid_dropout;
                attn1.multihead_attention_param.layers = p.transformer_block_param.layers;
                attn1.multihead_attention_param.output_adapter_q = p.transformer_block_param.output_adapter_q;
                attn1.multihead_attention_param.output_adapter_k = p.transformer_block_param.output_adapter_k;
                attn1.multihead_attention_param.output_adapter_v = p.transformer_block_param.output_adapter_v;
                attn1.multihead_attention_param.output_adapter_out = p.transformer_block_param.output_adapter_out;
                attn1.multihead_attention_param.weight_init = MultiheadAttentionParameter.WEIGHT_INIT.ENCODER_DECODER;
                attn1.multihead_attention_param.enable_rotary_positional_embedding = p.transformer_block_param.enable_rotary_positional_embedding;
                attn1.multihead_attention_param.bias_term = p.transformer_block_param.bias_term;
                attn1.parameters.Add((m_param.parameters.Count > 0) ? m_param.parameters[0] : new ParamSpec(1.0, 1.0));
                attn1.parameters.Add((m_param.parameters.Count > 1) ? m_param.parameters[1] : new ParamSpec(1.0, 0.0));
                m_attn1 = Layer<T>.Create(cuda, log, convertLayerParam(attn1, p), evtCancel);

                LayerParameter attn2 = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION, p.name + ".attn2", m_phase, p.freeze_learning);
                attn2.multihead_attention_param.block_size = p.transformer_block_param.block_size;
                attn2.multihead_attention_param.embed = p.transformer_block_param.embed;
                attn2.multihead_attention_param.heads = p.transformer_block_param.heads;
                attn2.multihead_attention_param.attn_dropout = p.transformer_block_param.attn_dropout;
                attn2.multihead_attention_param.resid_dropout = p.transformer_block_param.resid_dropout;
                attn2.multihead_attention_param.layers = p.transformer_block_param.layers;
                attn2.multihead_attention_param.output_adapter_q = p.transformer_block_param.output_adapter_q;
                attn2.multihead_attention_param.output_adapter_k = p.transformer_block_param.output_adapter_k;
                attn2.multihead_attention_param.output_adapter_v = p.transformer_block_param.output_adapter_v;
                attn2.multihead_attention_param.output_adapter_out = p.transformer_block_param.output_adapter_out;
                attn2.multihead_attention_param.weight_init = MultiheadAttentionParameter.WEIGHT_INIT.ENCODER_DECODER;
                attn2.multihead_attention_param.enable_rotary_positional_embedding = p.transformer_block_param.enable_rotary_positional_embedding;
                attn2.multihead_attention_param.bias_term = p.transformer_block_param.bias_term;
                attn2.parameters.Add((m_param.parameters.Count > 0) ? m_param.parameters[0] : new ParamSpec(1.0, 1.0));
                attn2.parameters.Add((m_param.parameters.Count > 1) ? m_param.parameters[1] : new ParamSpec(1.0, 0.0));
                m_attn2 = Layer<T>.Create(cuda, log, convertLayerParam(attn2, p), evtCancel);
            }
            else
            {
                throw new Exception("The block type '" + p.transformer_block_param.block_type.ToString() + "' is not supported!");
            }

            // Calculate the hidden dimension if not specified.
            uint nHiddenDim = p.transformer_block_param.hidden_dim;
            if (nHiddenDim == 0)
            {
                nHiddenDim = p.transformer_block_param.embed * 4;

                if (m_param.transformer_block_param.enable_llama_style_head)
                {
                    nHiddenDim = (uint)(2 * nHiddenDim / 3);
                    nHiddenDim = m_param.transformer_block_param.multiple_of * ((nHiddenDim + m_param.transformer_block_param.multiple_of - 1) / m_param.transformer_block_param.multiple_of);
                }
            }

            // w1 Linear
            LayerParameter fc1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, p.name + ".fc1", m_phase, p.freeze_learning);
            fc1.inner_product_param.axis = 2;
            fc1.inner_product_param.bias_term = p.transformer_block_param.bias_term;
            fc1.inner_product_param.num_output = nHiddenDim;
            if (p.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION ||
                p.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION2)
            {
                fc1.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
                fc1.inner_product_param.bias_filler = new FillerParameter("constant", 0.0);
            }
            else
            {
                fc1.inner_product_param.weight_filler = new FillerParameter("xavier");
                fc1.inner_product_param.bias_filler = new FillerParameter("xavier");
            }
            fc1.parameters.Add((m_param.parameters.Count > 0) ? m_param.parameters[0] : new ParamSpec(1.0, 1.0));
            fc1.parameters.Add((m_param.parameters.Count > 1) ? m_param.parameters[1] : new ParamSpec(1.0, 0.0));
            m_fc1 = Layer<T>.Create(cuda, log, convertLayerParam(fc1, p), evtCancel);

            // w2 Linear
            LayerParameter proj = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, p.name + ".proj", m_phase, p.freeze_learning);
            proj.inner_product_param.axis = 2;
            proj.inner_product_param.bias_term = p.transformer_block_param.bias_term;
            proj.inner_product_param.num_output = (uint)p.transformer_block_param.embed;
            if (p.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION ||
                p.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION2)
            {
                // apply special scaled init to the residual projections, per GPT-2 paper
                proj.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02/Math.Sqrt(2 * m_param.transformer_block_param.layers)); 
                proj.inner_product_param.bias_filler = new FillerParameter("constant", 0.0);
            }
            else
            {
                proj.inner_product_param.weight_filler = new FillerParameter("xavier");
                proj.inner_product_param.bias_filler = new FillerParameter("xavier");
            }
            proj.parameters.Add((m_param.parameters.Count > 0) ? m_param.parameters[0] : new ParamSpec(1.0, 1.0));
            proj.parameters.Add((m_param.parameters.Count > 1) ? m_param.parameters[1] : new ParamSpec(1.0, 0.0));
            m_proj = Layer<T>.Create(cuda, log, convertLayerParam(proj, p), evtCancel);

            if (p.transformer_block_param.enable_llama_style_head)
            {
                // w3 Linear
                LayerParameter fc2 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, p.name + ".fc2", m_phase, p.freeze_learning);
                fc2.inner_product_param.axis = 2;
                fc2.inner_product_param.bias_term = p.transformer_block_param.bias_term;
                fc2.inner_product_param.num_output = nHiddenDim;
                if (p.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION ||
                    p.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION2)
                {
                    fc2.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
                    fc2.inner_product_param.bias_filler = new FillerParameter("constant", 0.0);
                }
                else
                {
                    fc2.inner_product_param.weight_filler = new FillerParameter("xavier");
                    fc2.inner_product_param.bias_filler = new FillerParameter("xavier");
                }
                fc2.parameters.Add((m_param.parameters.Count > 0) ? m_param.parameters[0] : new ParamSpec(1.0, 1.0));
                fc2.parameters.Add((m_param.parameters.Count > 1) ? m_param.parameters[1] : new ParamSpec(1.0, 0.0));
                m_fc2 = Layer<T>.Create(cuda, log, convertLayerParam(fc2, p), evtCancel);
            }

            // ReLU has a very similar curve, and is faster.
            LayerParameter.LayerType actType = LayerParameter.LayerType.RELU;
            bool? bEnableBert = null;

            if (p.transformer_block_param.activation == param.gpt.TransformerBlockParameter.ACTIVATION.GELU_BERT)
            {
                actType = LayerParameter.LayerType.GELU;
                bEnableBert = true;
            }
            else if (p.transformer_block_param.activation == param.gpt.TransformerBlockParameter.ACTIVATION.GELU)
            {
                actType = LayerParameter.LayerType.GELU;
                bEnableBert = false;
            }
            else if (p.transformer_block_param.activation == param.gpt.TransformerBlockParameter.ACTIVATION.SILU)
            {
                actType = LayerParameter.LayerType.SILU;
            }

            LayerParameter act = new LayerParameter(actType, p.name + ".act", m_phase, p.freeze_learning);   
            if (bEnableBert.HasValue)
                act.gelu_param.enable_bert_version = bEnableBert.Value;
            
            m_act = Layer<T>.Create(cuda, log, convertLayerParam(act, p), evtCancel);

            if (p.transformer_block_param.resid_dropout > 0)
            {
                LayerParameter dropout = new LayerParameter(LayerParameter.LayerType.DROPOUT, p.name + ".drop", m_phase, p.freeze_learning);
                dropout.dropout_param.dropout_ratio = p.transformer_block_param.resid_dropout;
                m_dropout = Layer<T>.Create(cuda, log, convertLayerParam(dropout, p), evtCancel);
            }

            setup_internal_blobs(m_colInternalBlobs);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobNorm1);
            dispose(ref m_blobAttn1);
            dispose(ref m_blobNorm2);
            dispose(ref m_blobAttn2);
            dispose(ref m_blobNorm3);
            dispose(ref m_blobNormA);
            dispose(ref m_blobMlp1);
            dispose(ref m_blobMlp2);
            dispose(ref m_blobMlp3);
            dispose(ref m_blobMlpOut);
            dispose(ref m_blobX);

            dispose(ref m_norm1);
            dispose(ref m_attn1);
            dispose(ref m_norm2);
            dispose(ref m_attn2);
            dispose(ref m_norm3);
            dispose(ref m_fc1);
            dispose(ref m_fc2);
            dispose(ref m_proj);
            dispose(ref m_act);
            dispose(ref m_dropout);
            
            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobNorm1);
            col.Add(m_blobAttn1);
            col.Add(m_blobNorm2);
            if (m_blobAttn2 != null)
                col.Add(m_blobAttn2);
            col.Add(m_blobX);
            if (m_blobNorm3 != null)
                col.Add(m_blobNorm3);
            if (m_blobNormA != null)
                col.Add(m_blobNormA);
            col.Add(m_blobMlp1);
            if (m_blobMlp2 != null)
                col.Add(m_blobMlp2);
            col.Add(m_blobMlpOut);

            col.Add(m_norm1.internal_blobs);
            col.Add(m_attn1.internal_blobs);
            col.Add(m_norm2.internal_blobs);
            if (m_attn2 != null)
                col.Add(m_attn2.internal_blobs);
            if (m_norm3 != null)
                col.Add(m_norm3.internal_blobs);
            col.Add(m_fc1.internal_blobs);
            col.Add(m_act.internal_blobs);
            if (m_fc2 != null)
                col.Add(m_fc2.internal_blobs);
            col.Add(m_proj.internal_blobs);
            if (m_dropout != null)
                col.Add(m_dropout.internal_blobs);
        }

        /// <summary>
        /// Returns the minimum number of required bottom (input) Blobs: input
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of required bottom (input) Blobs: input, e_mask (when ENCODER,DECODER), d_mask (when DECODER)
        /// </summary>
        public override int MaxBottomBlobs
        {
            get
            {
                switch (m_param.transformer_block_param.block_type)
                {
                    case TransformerBlockParameter.BLOCK_TYPE.ENCODER:
                        return 2;

                    case TransformerBlockParameter.BLOCK_TYPE.DECODER:
                        return 4;

                    default:
                        return 1;
                }
            }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: trans
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

            m_norm1.ReInitializeParameters(target);
            m_attn1.ReInitializeParameters(target);
            m_norm2.ReInitializeParameters(target);
            if (m_attn2 != null)
                m_attn2.ReInitializeParameters(target);
            if (m_norm3 != null)
                m_norm3.ReInitializeParameters(target);
            m_fc1.ReInitializeParameters(target);
            if (m_fc2 != null)
                m_fc2.ReInitializeParameters(target);
            m_proj.ReInitializeParameters(target);

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
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.
        ///           CAUSAL_SELF
        ///                 colBottom[0] = input
        ///           ENCODER
        ///                 colBottom[0] = input
        ///                 colBottom[1] = encoder mask
        ///           DECODER
        ///                 colBottom[0] = decoder input
        ///                 colBottom[1] = decoder mask
        ///                 colBottom[2] = last encoder output
        ///                 colBottom[3] = encoder mask
        /// </param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            colTop[0].ReshapeLike(colBottom[0]);

            shareLayerBlob(m_blobNorm1, colBottom[0].shape());
            m_blobNorm1.ReshapeLike(colBottom[0]);
            shareLayerBlob(m_blobAttn1, colBottom[0].shape());
            m_blobAttn1.ReshapeLike(colBottom[0]);
            shareLayerBlob(m_blobNorm2, colBottom[0].shape());
            m_blobNorm2.ReshapeLike(colBottom[0]);
            if (m_blobNormA != null)
            {
                shareLayerBlob(m_blobNormA, colBottom[0].shape());
                m_blobNormA.ReshapeLike(colBottom[0]);
            }
            shareLayerBlob(m_blobX, colBottom[0].shape());
            m_blobX.ReshapeLike(colBottom[0]);

            if (m_blobAttn2 != null)
            {
                shareLayerBlob(m_blobAttn2, colBottom[0].shape());
                m_blobAttn2.ReshapeLike(colBottom[0]);
            }

            if (m_blobNorm3 != null)
            {
                shareLayerBlob(m_blobNorm3, colBottom[0].shape());
                m_blobNorm3.ReshapeLike(colBottom[0]);
            }

            shareLayerBlob(m_blobMlp1, colBottom[0].shape());
            m_blobMlp1.ReshapeLike(colBottom[0]);
            if (m_blobMlp2 != null)
            {
                shareLayerBlob(m_blobMlp2, colBottom[0].shape());
                m_blobMlp2.ReshapeLike(colBottom[0]);
            }
            if (m_blobMlp3 != null)
            {
                shareLayerBlob(m_blobMlp3, colBottom[0].shape());
                m_blobMlp3.ReshapeLike(colBottom[0]);
            }
            shareLayerBlob(m_blobMlpOut, colBottom[0].shape());
            m_blobMlpOut.ReshapeLike(colBottom[0]);

            addInternal(colBottom[0], m_blobNorm1);
            m_norm1.LayerSetUp(m_colInternalBottom, m_colInternalTop);

            if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION ||
                m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION2)
            {
                // self.attn(self.ln_1(x))            
                addInternal(m_blobNorm1, m_blobAttn1);
                m_attn1.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            }
            else if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.ENCODER)
            {
                // self.attn(x_1, x_1, x_1, e_mask)
                addInternal(new List<Blob<T>>() { m_blobNorm1, m_blobNorm1, m_blobNorm1, colBottom[1] }, m_blobAttn1);
                m_attn1.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            }
            else if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.DECODER)
            {
                // self.attn1(x_1, x_1, x_1, d_mask)
                addInternal(new List<Blob<T>>() { m_blobNorm1, m_blobNorm1, m_blobNorm1, colBottom[1] }, m_blobAttn1);
                m_attn1.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            }
            else
            {
                throw new Exception("Unknown block type '" + m_param.transformer_block_param.block_type.ToString() + "'!");
            }

            addInternal(colTop[0], m_blobNorm2);
            m_norm2.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            Blob<T> blobNorm = m_blobNorm2;

            if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.DECODER)
            {
                // self.attn2(x_2, e_output, e_output, e_mask)
                addInternal(new List<Blob<T>>() { m_blobNorm2, colBottom[2], colBottom[2], colBottom[3] }, m_blobAttn2);
                m_attn2.LayerSetUp(m_colInternalBottom, m_colInternalTop);

                addInternal(m_blobAttn2, m_blobNorm3);
                m_norm3.LayerSetUp(m_colInternalBottom, m_colInternalTop);
                blobNorm = m_blobNorm3;
            }

            addInternal(blobNorm, m_blobMlp1);
            m_fc1.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            addInternal(m_blobNorm2, m_blobMlp1);
            m_fc1.Reshape(m_colInternalBottom, m_colInternalTop);
            addInternal(m_blobMlp1, m_blobMlp1);
            m_act.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            if (m_fc2 != null)
            {
                addInternal(blobNorm, m_blobMlp2);
                m_fc2.LayerSetUp(m_colInternalBottom, m_colInternalTop);
                m_fc2.Reshape(m_colInternalBottom, m_colInternalTop);
                addInternal(m_blobMlp2, m_blobMlpOut);
            }
            else
            {
                addInternal(m_blobMlp1, m_blobMlpOut);
            }
            m_proj.LayerSetUp(m_colInternalBottom, m_colInternalTop);

            if (m_dropout != null)
            {
                addInternal(m_blobMlpOut, m_blobMlpOut);
                m_dropout.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            }

            colTop[0].ReshapeLike(m_blobMlpOut);

            blobs.Add(m_norm1.blobs);
            blobs.Add(m_attn1.blobs);
            if (m_attn2 != null)
                blobs.Add(m_attn2.blobs);
            blobs.Add(m_norm2.blobs);
            blobs.Add(m_fc1.blobs);
            if (m_fc2 != null)
                blobs.Add(m_fc2.blobs);
            blobs.Add(m_proj.blobs);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            colTop[0].ReshapeLike(colBottom[0]);

            m_blobNorm1.ReshapeLike(colBottom[0]);
            m_blobAttn1.ReshapeLike(colBottom[0]);
            m_blobNorm2.ReshapeLike(colBottom[0]);
            m_blobX.ReshapeLike(colBottom[0]);

            if (m_blobAttn2 != null)
                m_blobAttn2.ReshapeLike(colBottom[0]);

            if (m_blobNorm3 != null)
                m_blobNorm3.ReshapeLike(colBottom[0]);

            m_blobMlp1.ReshapeLike(colBottom[0]);
            m_blobMlpOut.ReshapeLike(colBottom[0]);
            
            addInternal(colBottom[0], m_blobNorm1);
            m_norm1.Reshape(m_colInternalBottom, m_colInternalTop);

            if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION ||
                m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION2)
            {
                // self.attn(self.ln_1(x))            
                addInternal(m_blobNorm1, m_blobAttn1);
                m_attn1.Reshape(m_colInternalBottom, m_colInternalTop);
            }
            else if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.ENCODER)
            {
                // self.attn(x_1, x_1, x_1, e_mask)
                addInternal(new List<Blob<T>>() { m_blobNorm1, m_blobNorm1, m_blobNorm1, colBottom[1] }, m_blobAttn1);
                m_attn1.Reshape(m_colInternalBottom, m_colInternalTop);
            }
            else if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.DECODER)
            {
                // self.attn1(x_1, x_1, x_1, d_mask)
                addInternal(new List<Blob<T>>() { m_blobNorm1, m_blobNorm1, m_blobNorm1, colBottom[1] }, m_blobAttn1);
                m_attn1.Reshape(m_colInternalBottom, m_colInternalTop);
            }
            else
            {
                throw new Exception("Unknown block type '" + m_param.transformer_block_param.block_type.ToString() + "'!");
            }

            addInternal(colTop[0], m_blobNorm2);
            m_norm2.Reshape(m_colInternalBottom, m_colInternalTop);
            Blob<T> blobNorm = m_blobNorm2;

            if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.DECODER)
            {
                // self.attn2(x_2, e_output, e_output, e_mask)
                addInternal(new List<Blob<T>>() { m_blobNorm2, colBottom[2], colBottom[2], colBottom[3] }, m_blobAttn2);
                m_attn2.Reshape(m_colInternalBottom, m_colInternalTop);

                addInternal(m_blobAttn2, m_blobNorm3);
                m_norm3.Reshape(m_colInternalBottom, m_colInternalTop);
                blobNorm = m_blobNorm3;
            }

            addInternal(blobNorm, m_blobMlp1);
            m_fc1.Reshape(m_colInternalBottom, m_colInternalTop);
            addInternal(m_blobMlp1, m_blobMlp1);
            m_act.Reshape(m_colInternalBottom, m_colInternalTop);
            if (m_fc2 != null)
            {
                addInternal(blobNorm, m_blobMlp2);
                m_fc2.Reshape(m_colInternalBottom, m_colInternalTop);
                addInternal(m_blobMlp2, m_blobMlpOut);
                m_blobMlp3.ReshapeLike(m_blobMlp2);
            }
            else
            {
                addInternal(m_blobMlp1, m_blobMlpOut);
            }
            m_proj.Reshape(m_colInternalBottom, m_colInternalTop);

            if (m_dropout != null)
            {
                addInternal(m_blobMlpOut, m_blobMlpOut);
                m_dropout.Reshape(m_colInternalBottom, m_colInternalTop);
            }

            colTop[0].ReshapeLike(colBottom[0]);
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.
        /// CAUSAL_SELF
        ///       colBottom[0] = @f$ (N \times C \times H \times W) @f$ input
        /// ENCODER
        ///       colBottom[0] = @f$ (N \times C \times H \times W) @f$ input
        ///       colBottom[1] = @f$ (N \times C \times H \times W) @f$ encoder mask
        /// DECODER
        ///       colBottom[0] = @f$ (N \times C \times H \times W) @f$ decoder input
        ///       colBottom[1] = @f$ (N \times C \times H \times W) @f$ decoder mask
        ///       colBottom[2] = @f$ (N \times C \times H \times W) @f$ last encoder output
        ///       colBottom[3] = @f$ (N \times C \times H \times W) @f$ encoder mask
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed transformer block.
        /// </param>
        /// <remarks>
        /// The encoder and decoder masks use the following formats.
        /// 
        /// Encoder Mask:
        /// shape = (batch, seq_len, 1)
        /// The sequence length is filled with 1 where data exists in each sequence, and
        /// 0 otherwise.  For example, when using a sequence length of 4 and batch = 3, 
        /// the following input:
        /// <code>
        ///  encoder input                encoder mask
        ///  shape = (3,4)                (3,4)
        ///  [33, 44, 22, 55]             [  1,  1,  1,  1]
        ///  [44, 33, 0,  0 ] has mask -> [  1,  1,  0,  0]
        ///  [88, 99, 22, 0 ]             [  1,  1,  1,  0]
        /// </code>
        /// 
        /// Decoder Mask:
        /// shape (batch, seq_len, seq_len)
        /// The decoder mask is first filled with a mask similar to the encoder mask, whre each
        /// sequence for each entry is duplicated for the number of sequences high to create an
        /// initial mask like the following. Next a triangular mask is anded to avoid right side info.
        /// <code>
        ///  decoder input                encoder like mask        triangular mask     final decoder mask
        ///  shape = (3,4)                (3,4,4)                  (3,4,4)             (3,4,4)
        ///  [33, 44, 22, 55]             [  1,  1,  1,  1]        [  1,  0,  0,  0]   [  1,  0,  0,  0]
        ///                               [  1,  1,  1,  1]        [  1,  1,  0,  0]   [  1,  1,  0,  0]
        ///                               [  1,  1,  1,  1] -and-> [  1,  1,  1,  0] = [  1,  1,  1,  0]
        ///                               [  1,  1,  1,  1]        [  1,  1,  1,  1]   [  1,  1,  1,  1]
        ///  [44, 33, 0,  0 ] has mask -> [  1,  1,  0,  0]        [  1,  0,  0,  0]   [  1,  0,  0,  0]
        ///                               [  1,  1,  0,  0]        [  1,  1,  0,  0]   [  1,  1,  0,  0]
        ///                               [  1,  1,  0,  0] -and-> [  1,  1,  1,  0] = [  1,  1,  0,  0]
        ///                               [  1,  1,  0,  0]        [  1,  1,  1,  1]   [  1,  1,  0,  0]
        ///  [88, 99, 22, 0 ]             [  1,  1,  1,  0]        [  1,  0,  0,  0]   [  1,  0,  0,  0]
        ///                               [  1,  1,  1,  0]        [  1,  1,  0,  0]   [  1,  1,  0,  0]
        ///                               [  1,  1,  1,  0] -and-> [  1,  1,  1,  0] = [  1,  1,  1,  0]
        ///                               [  1,  1,  1,  0]        [  1,  1,  1,  1]   [  1,  1,  1,  0]
        /// </code>                              
        /// </remarks>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colBottom[0].count();
            Blob<T> blobX = colBottom[0];
            Blob<T> blobXMask = (colBottom.Count > 1) ? colBottom[1] : null;
            Blob<T> blobEncOut = (colBottom.Count > 3) ? colBottom[2] : null;
            Blob<T> blobEncMask = (colBottom.Count > 3) ? colBottom[3] : null;

            
            //-------------------------------------------
            // x = x + self.attn(self.ln_1(x))

            // x_1 = self.ln_1(x)
            addInternal(blobX, m_blobNorm1);            
            m_norm1.Forward(m_colInternalBottom, m_colInternalTop);

            if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION ||
                m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION2)
            {
                // attn1 = self.attn(self.ln_1(x))            
                addInternal(m_blobNorm1, m_blobAttn1);
                m_attn1.Forward(m_colInternalBottom, m_colInternalTop);
            }
            else if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.ENCODER)
            {
                // attn1 = self.attn(x_1, x_1, x_1, e_mask)
                addInternal(new List<Blob<T>>() { m_blobNorm1, m_blobNorm1, m_blobNorm1, blobXMask }, m_blobAttn1);
                m_attn1.Forward(m_colInternalBottom, m_colInternalTop);
            }
            else if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.DECODER)
            {
                // attn1 = self.attn1(x_1, x_1, x_1, d_mask)
                addInternal(new List<Blob<T>>() { m_blobNorm1, m_blobNorm1, m_blobNorm1, blobXMask }, m_blobAttn1);
                m_attn1.Forward(m_colInternalBottom, m_colInternalTop);
            }
            else
            {
                throw new Exception("Unknown block type '" + m_param.transformer_block_param.block_type.ToString() + "'!");
            }
            
            // xB = x + self.attn1(self.ln_1(x))
            m_cuda.add(nCount, blobX.gpu_data, m_blobAttn1.gpu_data, m_blobX.mutable_gpu_data);
            
            // x_2 = self.ln_2(xB) 
            addInternal(m_blobX, m_blobNorm2);
            m_norm2.Forward(m_colInternalBottom, m_colInternalTop);
            Blob<T> blobNorm = m_blobNorm2;

            if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.DECODER)
            {
                // attn2 = self.attn2(x_2, e_output, e_output, e_mask)
                addInternal(new List<Blob<T>>() { m_blobNorm2, blobEncOut, blobEncOut, blobEncMask }, m_blobAttn2);
                m_attn2.Forward(m_colInternalBottom, m_colInternalTop);

                // xC = xB + self.attn2(self.ln_2(x))
                m_cuda.add(nCount, m_blobX.gpu_data, m_blobAttn2.gpu_data, m_blobX.mutable_gpu_data);

                // x_3 = self.ln3(xC)
                addInternal(m_blobX, m_blobNorm3);
                m_norm3.Forward(m_colInternalBottom, m_colInternalTop);
                blobNorm = m_blobNorm3;
            }

            // CSA | ENCODER: ff = self.mlpf(self.ln_2(x_2)),
            // DECODER:       ff = self.mlpf(self.ln_3(x_3))
            addInternal(blobNorm, m_blobMlp1);
            m_fc1.Forward(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobMlp1, m_blobMlp1);
            m_act.Forward(m_colInternalBottom, m_colInternalTop);

            Blob<T> blobMlp = m_blobMlp1;
            if (m_param.transformer_block_param.enable_llama_style_head)
            {
                addInternal(blobNorm, m_blobMlp2);
                m_fc2.Forward(m_colInternalBottom, m_colInternalTop);
                m_cuda.mul(m_blobMlp1.count(), m_blobMlp1.gpu_data, m_blobMlp2.gpu_data, m_blobMlp3.mutable_gpu_data);
                blobMlp = m_blobMlp3;
            }

            addInternal(blobMlp, m_blobMlpOut);
            m_proj.Forward(m_colInternalBottom, m_colInternalTop);

            if (m_dropout != null)
            {
                addInternal(m_blobMlpOut, m_blobMlpOut);
                m_dropout.Forward(m_colInternalBottom, m_colInternalTop);
            }

            // CSA | ENCODER: xC = xB + self.mlpf(self.ln_2(x_2)),
            // DECODER:       xD = xC + self.mlpf(self.ln_3(x_3))
            m_cuda.add(nCount, m_blobX.gpu_data, m_blobMlpOut.gpu_data, colTop[0].mutable_gpu_data);
        }

        /// <summary>
        /// Computes the loss error gradient w.r.t the outputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///   -# @f$ (N \times K \times H \times W) @f$.
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.
        /// CAUSAL_SELF
        ///       colBottom[0] = @f$ (N \times C \times H \times W) @f$ input
        /// ENCODER
        ///       colBottom[0] = @f$ (N \times C \times H \times W) @f$ input
        ///       colBottom[1] = @f$ (N \times C \times H \times W) @f$ encoder mask
        /// DECODER
        ///       colBottom[0] = @f$ (N \times C \times H \times W) @f$ decoder input
        ///       colBottom[1] = @f$ (N \times C \times H \times W) @f$ decoder mask
        ///       colBottom[2] = @f$ (N \times C \times H \times W) @f$ last encoder output
        ///       colBottom[3] = @f$ (N \times C \times H \times W) @f$ encoder mask
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            int nCount = colBottom[0].count();
            Blob<T> blobX = colBottom[0];
            Blob<T> blobXMask = (colBottom.Count > 1) ? colBottom[1] : null;
            Blob<T> blobEncOut = (colBottom.Count > 3) ? colBottom[2] : null;
            Blob<T> blobEncMask = (colBottom.Count > 3) ? colBottom[3] : null;

            // Gradient with respect to state then data.
            if (rgbPropagateDown[0])
            {
                List<bool> rgbPropagate = new List<bool>() { true, true };

                // CSA | ENCODER Gradient for xC = xB + self.mlpf(self.ln_2(x_2))
                // DECODER Gradient for       xD = xC + self.mlpf(self.ln_3(x_3))
                // xD -> ff (decoder), otherwise xC -> ff (encoder)
                m_cuda.copy(nCount, colTop[0].gpu_diff, m_blobMlpOut.mutable_gpu_diff);
                // xD -> xC (decoder), otherwise xC -> xB (encoder)
                m_cuda.copy(nCount, colTop[0].gpu_diff, m_blobX.mutable_gpu_diff); // xB, xC

                if (m_dropout != null)
                {
                    addInternal(m_blobMlpOut, m_blobMlpOut);
                    m_dropout.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                }

                // Gradient for MLP
                Blob<T> blobMlp = m_blobMlp1;
                if (m_param.transformer_block_param.enable_llama_style_head)
                    blobMlp = m_blobMlp3;

                addInternal(blobMlp, m_blobMlpOut);
                m_proj.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                Blob<T> blobNorm = (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.DECODER) ? m_blobNorm3 : m_blobNorm2;
                if (m_param.transformer_block_param.enable_llama_style_head)
                {
                    m_cuda.mul(m_blobMlp1.count(), m_blobMlp3.gpu_diff, m_blobMlp2.gpu_data, m_blobMlp1.mutable_gpu_diff);
                    m_cuda.mul(m_blobMlp1.count(), m_blobMlp3.gpu_diff, m_blobMlp1.gpu_data, m_blobMlp2.mutable_gpu_diff);

                    addInternal(blobNorm, m_blobMlp2);
                    m_fc2.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                    m_cuda.copy(blobNorm.count(), blobNorm.gpu_diff, m_blobNormA.mutable_gpu_data);
                }

                addInternal(m_blobMlp1, m_blobMlp1);
                m_act.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // ff -> x_3 (decoder), otherwise x_2 (encoder)
                addInternal(blobNorm, m_blobMlp1);
                m_fc1.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                if (m_param.transformer_block_param.enable_llama_style_head)
                    m_cuda.add(blobNorm.count(), m_blobNormA.gpu_data, blobNorm.gpu_diff, blobNorm.mutable_gpu_diff);

                if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.DECODER)
                {
                    // x_3 = self.ln3(xC)
                    // x_3 -> xC1
                    m_blobAttn2.CopyFrom(m_blobX, true);
                    addInternal(m_blobX, m_blobNorm3);
                    m_norm3.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                    // x_3 + xC1 -> xC
                    // xC -> xB (implied)
                    m_cuda.add(nCount, m_blobAttn2.gpu_diff, m_blobX.gpu_diff, m_blobX.mutable_gpu_diff);
                    // xC -> attn2
                    m_blobAttn2.CopyFrom(m_blobX, true);

                    // attn2 = self.attn2(x_2, e_output, e_output, e_mask)
                    // attn2 -> x_2 (ln2), e_output1, e_output2
                    addInternal(new List<Blob<T>>() { m_blobNorm2, blobEncOut, blobEncOut, blobEncMask }, m_blobAttn2);
                    m_attn2.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                }

                // x_2 = self.ln_2(xB) 
                // x_2 -> xB1
                m_blobAttn1.CopyFrom(m_blobX, true);
                addInternal(m_blobX, m_blobNorm2);
                m_norm2.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // xC + xB1 -> xB
                // xB -> x (implied)
                m_cuda.add(nCount, m_blobAttn1.gpu_diff, m_blobX.gpu_diff, m_blobX.mutable_gpu_diff);
                // xB -> attn1
                m_blobAttn1.CopyFrom(m_blobX, true);

                if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION ||
                    m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION2)
                {
                    // Gradient for self.attn(self.ln_1(x))
                    addInternal(m_blobNorm1, m_blobAttn1);
                    m_attn1.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                }
                else if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.ENCODER ||
                         m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.DECODER)
                {
                    // Gradient for self.attn(x_1, x_1, x_1, e_mask)
                    addInternal(new List<Blob<T>>() { m_blobNorm1, m_blobNorm1, m_blobNorm1, blobXMask }, m_blobAttn1);
                    m_attn1.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                }
                else
                {
                    throw new Exception("Unknown block type '" + m_param.transformer_block_param.block_type.ToString() + "'!");
                }

                // x_1 = ln1(x)
                // x_1 -> x1
                addInternal(blobX, m_blobNorm1);
                m_norm1.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Accumulate attention gradient with others in bottom[0].
                // x1 + xB -> x
                m_cuda.add(nCount, blobX.gpu_diff, m_blobX.gpu_diff, blobX.mutable_gpu_diff);
            }
        }
    }
}
