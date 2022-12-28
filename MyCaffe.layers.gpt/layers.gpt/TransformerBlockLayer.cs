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
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TransformerBlockLayer<T> : Layer<T>
    {
        Blob<T> m_blobLn1;
        Blob<T> m_blobAttn;
        Blob<T> m_blobLn2;
        Blob<T> m_blobMlp;
        Blob<T> m_blobMlpOut;
        // Initial input Layer normalization
        Layer<T> m_ln1;
        // Layer normalization after the first attention block
        Layer<T> m_ln2;
        // Attention block
        Layer<T> m_attn1;
        // MLP block
        Layer<T> m_fc;      // initial linear
        Layer<T> m_proj;    // projection
        Layer<T> m_act;     // activation
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

            m_blobLn1 = new Blob<T>(cuda, log);
            m_blobAttn = new Blob<T>(cuda, log);
            m_blobLn2 = new Blob<T>(cuda, log);
            m_blobMlp = new Blob<T>(cuda, log);
            m_blobMlpOut = new Blob<T>(cuda, log);

            LayerParameter ln1 = new LayerParameter(LayerParameter.LayerType.LAYERNORM, "ln1");
            m_ln1 = Layer<T>.Create(cuda, log, ln1, evtCancel) as Layer<T>;
            
            LayerParameter ln2 = new LayerParameter(LayerParameter.LayerType.LAYERNORM, "ln2");
            m_ln2 = Layer<T>.Create(cuda, log, ln1, evtCancel) as Layer<T>;

            if (p.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION)
            {
                LayerParameter attn = new LayerParameter(LayerParameter.LayerType.CAUSAL_SELF_ATTENTION, "attn");
                attn.causal_self_attention_param.block_size = p.transformer_block_param.block_size;
                attn.causal_self_attention_param.embed = p.transformer_block_param.embed;
                attn.causal_self_attention_param.heads = p.transformer_block_param.heads;
                attn.causal_self_attention_param.attn_dropout = p.transformer_block_param.attn_dropout;
                attn.causal_self_attention_param.resid_dropout = p.transformer_block_param.resid_dropout;
                attn.causal_self_attention_param.layers = p.transformer_block_param.layers;
                m_attn1 = Layer<T>.Create(cuda, log, attn, evtCancel);
            }
            else if (p.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.ENCODER)
            {
                LayerParameter attn = new LayerParameter(LayerParameter.LayerType.MULTIHEAD_ATTENTION, "attn");
                attn.multihead_attention_param.block_size = p.transformer_block_param.block_size;
                attn.multihead_attention_param.embed = p.transformer_block_param.embed;
                attn.multihead_attention_param.heads = p.transformer_block_param.heads;
                attn.multihead_attention_param.attn_dropout = p.transformer_block_param.attn_dropout;
                attn.multihead_attention_param.resid_dropout = p.transformer_block_param.resid_dropout;
                attn.multihead_attention_param.layers = p.transformer_block_param.layers;
                m_attn1 = Layer<T>.Create(cuda, log, attn, evtCancel);
            }
            else
            {
                throw new Exception("The block type '" + p.transformer_block_param.block_type.ToString() + "' is not supported!");
            }

            LayerParameter fc = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "fc");
            fc.inner_product_param.axis = 2;
            fc.inner_product_param.bias_term = true;
            fc.inner_product_param.num_output = (uint)(p.transformer_block_param.embed * 4);
            fc.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);  
            fc.inner_product_param.bias_filler = new FillerParameter("constant", 0.0); 
            fc.parameters.Add(new ParamSpec(1.0, 1.0));
            fc.parameters.Add(new ParamSpec(1.0, 0.0));
            m_fc = Layer<T>.Create(cuda, log, fc, evtCancel);

            LayerParameter proj = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "proj");
            proj.inner_product_param.axis = 2;
            proj.inner_product_param.bias_term = true;
            proj.inner_product_param.num_output = (uint)p.transformer_block_param.embed;
            // apply special scaled init to the residual projections, per GPT-2 paper
            proj.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02/Math.Sqrt(2 * m_param.transformer_block_param.layers)); 
            proj.inner_product_param.bias_filler = new FillerParameter("constant", 0.0);  
            proj.parameters.Add(new ParamSpec(1.0, 1.0));
            proj.parameters.Add(new ParamSpec(1.0, 0.0));
            m_proj = Layer<T>.Create(cuda, log, proj, evtCancel);

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
                
            LayerParameter act = new LayerParameter(actType, "act");   
            if (bEnableBert.HasValue)
                act.gelu_param.enable_bert_version = bEnableBert.Value;
            
            m_act = Layer<T>.Create(cuda, log, act, evtCancel);

            if (p.transformer_block_param.resid_dropout > 0)
            {
                LayerParameter dropout = new LayerParameter(LayerParameter.LayerType.DROPOUT, "dropout");
                dropout.dropout_param.dropout_ratio = p.transformer_block_param.resid_dropout;
                m_dropout = Layer<T>.Create(cuda, log, dropout, evtCancel);
            }
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobLn1);
            dispose(ref m_blobAttn);
            dispose(ref m_blobLn2);
            dispose(ref m_blobMlp);
            dispose(ref m_blobMlpOut);

            dispose(ref m_ln1);
            dispose(ref m_ln2);
            dispose(ref m_attn1);
            dispose(ref m_fc);
            dispose(ref m_proj);
            dispose(ref m_act);
            dispose(ref m_dropout);
            
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                return col;
            }
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
                        return 3;

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

            m_ln1.ReInitializeParameters(target);
            m_ln2.ReInitializeParameters(target);
            m_attn1.ReInitializeParameters(target);
            m_fc.ReInitializeParameters(target);
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
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            colTop[0].ReshapeLike(colBottom[0]);

            m_blobLn1.ReshapeLike(colBottom[0]);
            m_blobAttn.ReshapeLike(colBottom[0]);
            m_blobLn2.ReshapeLike(colBottom[0]);
            m_blobMlp.ReshapeLike(colBottom[0]);
            m_blobMlpOut.ReshapeLike(colBottom[0]);

            addInternal(colBottom[0], m_blobLn1);
            m_ln1.LayerSetUp(m_colInternalBottom, m_colInternalTop);

            if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION)
            {
                // self.attn(self.ln_1(x))            
                addInternal(m_blobLn1, m_blobAttn);
                m_attn1.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            }
            else if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.ENCODER)
            {
                // self.attn(x_1, x_1, x_1, e_mask)
                addInternal(new List<Blob<T>>() { m_blobLn1, m_blobLn1, m_blobLn1, colBottom[1] }, m_blobAttn);
                m_attn1.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            }
            else
            {
                throw new Exception("Unknown block type '" + m_param.transformer_block_param.block_type.ToString() + "'!");
            }

            addInternal(colTop[0], m_blobLn2);
            m_ln2.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            
            addInternal(m_blobLn2, m_blobMlp);
            m_fc.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            addInternal(m_blobLn2, m_blobMlp);
            m_fc.Reshape(m_colInternalBottom, m_colInternalTop);
            addInternal(m_blobMlp, m_blobMlp);
            m_act.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            addInternal(m_blobMlp, m_blobMlpOut);
            m_proj.LayerSetUp(m_colInternalBottom, m_colInternalTop);

            if (m_dropout != null)
            {
                addInternal(m_blobMlpOut, m_blobMlpOut);
                m_dropout.LayerSetUp(m_colInternalBottom, m_colInternalTop);
            }

            colTop[0].ReshapeLike(m_blobMlpOut);

            blobs.Add(m_attn1.blobs);
            blobs.Add(m_fc.blobs);
            blobs.Add(m_proj.blobs);

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
            if (!reshapeNeeded(colBottom, colTop))
                return;

            colTop[0].ReshapeLike(colBottom[0]);

            m_blobLn1.ReshapeLike(colBottom[0]);
            m_blobAttn.ReshapeLike(colBottom[0]);
            m_blobLn2.ReshapeLike(colBottom[0]);
            m_blobMlp.ReshapeLike(colBottom[0]);
            m_blobMlpOut.ReshapeLike(colBottom[0]);

            addInternal(colBottom[0], m_blobLn1);
            m_ln1.Reshape(m_colInternalBottom, m_colInternalTop);

            if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION)
            {
                // self.attn(self.ln_1(x))            
                addInternal(m_blobLn1, m_blobAttn);
                m_attn1.Reshape(m_colInternalBottom, m_colInternalTop);
            }
            else if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.ENCODER)
            {
                // self.attn(x_1, x_1, x_1, e_mask)
                addInternal(new List<Blob<T>>() { m_blobLn1, m_blobLn1, m_blobLn1, colBottom[1] }, m_blobAttn);
                m_attn1.Reshape(m_colInternalBottom, m_colInternalTop);
            }
            else
            {
                throw new Exception("Unknown block type '" + m_param.transformer_block_param.block_type.ToString() + "'!");
            }

            addInternal(colTop[0], m_blobLn2);
            m_ln2.Reshape(m_colInternalBottom, m_colInternalTop);

            addInternal(m_blobLn2, m_blobMlp);
            m_fc.Reshape(m_colInternalBottom, m_colInternalTop);
            addInternal(m_blobMlp, m_blobMlp);
            m_act.Reshape(m_colInternalBottom, m_colInternalTop);
            addInternal(m_blobMlp, m_blobMlpOut);
            m_proj.Reshape(m_colInternalBottom, m_colInternalTop);

            if (m_dropout != null)
            {
                addInternal(m_blobMlpOut, m_blobMlpOut);
                m_dropout.Reshape(m_colInternalBottom, m_colInternalTop);
            }
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed transformer block.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colBottom[0].count();

            //-------------------------------------------
            // x = x + self.attn(self.ln_1(x))

            // self.ln_1(x)
            addInternal(colBottom[0], m_blobLn1);            
            m_ln1.Forward(m_colInternalBottom, m_colInternalTop);

            if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION)
            {
                // self.attn(self.ln_1(x))            
                addInternal(m_blobLn1, m_blobAttn);
                m_attn1.Forward(m_colInternalBottom, m_colInternalTop);
            }
            else if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.ENCODER)
            {
                // self.attn(x_1, x_1, x_1, e_mask)
                addInternal(new List<Blob<T>>() { m_blobLn1, m_blobLn1, m_blobLn1, colBottom[1] }, m_blobAttn);
                m_attn1.Forward(m_colInternalBottom, m_colInternalTop);
            }
            else
            {
                throw new Exception("Unknown block type '" + m_param.transformer_block_param.block_type.ToString() + "'!");
            }

            // x = x + self.attn(self.ln_1(x))
            m_cuda.add(nCount, colBottom[0].gpu_data, m_blobAttn.gpu_data, colTop[0].mutable_gpu_data);

            //-------------------------------------------
            // x = x + self.mlpf(self.ln_2(x))

            // self.ln_2(x) 
            addInternal(colTop[0], m_blobLn2);
            m_ln2.Forward(m_colInternalBottom, m_colInternalTop);
            
            // self.mlpf(self.ln_2(x))
            addInternal(m_blobLn2, m_blobMlp);
            m_fc.Forward(m_colInternalBottom, m_colInternalTop);
            addInternal(m_blobMlp, m_blobMlp);
            m_act.Forward(m_colInternalBottom, m_colInternalTop);
            addInternal(m_blobMlp, m_blobMlpOut);
            m_proj.Forward(m_colInternalBottom, m_colInternalTop);

            if (m_dropout != null)
            {
                addInternal(m_blobMlpOut, m_blobMlpOut);
                m_dropout.Forward(m_colInternalBottom, m_colInternalTop);
            }
            
            // x = x + self.mlpf(self.ln_2(x))
            m_cuda.add(nCount, colTop[0].gpu_data, m_blobMlpOut.gpu_data, colTop[0].mutable_gpu_data);
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
            int nCount = colBottom[0].count();

            // Gradient with respect to state then data.
            if (rgbPropagateDown[0])
            {
                List<bool> rgbPropagate = new List<bool>() { true, true };

                // Gradient for x = x + self.mlpf(self.ln_2(x))
                m_cuda.copy(nCount, colTop[0].gpu_diff, m_blobMlpOut.mutable_gpu_diff);
                m_cuda.copy(nCount, colTop[0].gpu_diff, colBottom[0].mutable_gpu_diff); // temporary holding dx2

                // Gradient for self.mlpf(self.ln_2(x))
                if (m_dropout != null)
                {
                    addInternal(m_blobMlpOut, m_blobMlpOut);
                    m_dropout.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                }

                addInternal(m_blobMlp, m_blobMlpOut);
                m_proj.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                addInternal(m_blobMlp, m_blobMlp);
                m_act.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                addInternal(m_blobLn2, m_blobMlp);
                m_fc.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Gradient for self.ln_2(x) 
                addInternal(m_blobAttn, m_blobLn2);
                m_ln2.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Gradient for x = x + self.attn(self.ln_1(x))
                m_cuda.add(nCount, colBottom[0].gpu_diff, m_blobAttn.gpu_diff, m_blobAttn.mutable_gpu_diff);

                if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.CAUSAL_SELF_ATTENTION)
                {
                    // Gradient for self.attn(self.ln_1(x))
                    addInternal(m_blobLn1, m_blobAttn);
                    m_attn1.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                }
                else if (m_param.transformer_block_param.block_type == TransformerBlockParameter.BLOCK_TYPE.ENCODER)
                {
                    // Gradient for self.attn(x_1, x_1, x_1, e_mask)
                    addInternal(new List<Blob<T>>() { m_blobLn1, m_blobLn1, m_blobLn1, colBottom[1] }, m_blobAttn);
                    m_attn1.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                }
                else
                {
                    throw new Exception("Unknown block type '" + m_param.transformer_block_param.block_type.ToString() + "'!");
                }

                // Gradient for self.ln_1(x)
                addInternal(colBottom[0], m_blobLn1);
                m_ln1.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                m_cuda.add(nCount, colBottom[0].gpu_diff, m_blobAttn.gpu_diff, colBottom[0].mutable_gpu_diff);
            }
        }
    }
}
