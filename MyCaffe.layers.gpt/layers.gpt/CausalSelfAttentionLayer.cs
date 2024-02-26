﻿using System;
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
    public class CausalSelfAttentionLayer<T> : Layer<T>
    {
        List<int> m_rgShape = new List<int>() { 1, 1, 1, 1 };
        // Key, query, value projections for all heads, but in a batch.
        Layer<T> m_c_attn = null;
        // Output projection.
        Layer<T> m_c_proj = null;
        // Regularization
        Layer<T> m_attn_dropout = null;
        Layer<T> m_resid_dropout = null;
        // Transpose
        Layer<T> m_transpose;
        Layer<T> m_transposeQ;
        // Softmax
        Layer<T> m_softmax = null;
        // Causal mask to ensure that atttention is only applied to the left in the input sequence.
        Blob<T> m_blobMask;
        Blob<T> m_blobQ;
        Blob<T> m_blobK;
        Blob<T> m_blobV;
        Blob<T> m_blobQt;
        Blob<T> m_blobKt;
        Blob<T> m_blobKt1;
        Blob<T> m_blobVt;
        Blob<T> m_blobWork;
        Blob<T> m_blobAttA;
        Blob<T> m_blobAttB;
        Blob<T> m_blobIpAttn;
        Blob<T> m_blobY;
        int[] m_rgYShape = new int[4];
        int[] m_rgWorkShape = new int[4];
        long m_hRope = 0;
        long m_hCudnn = 0;
        long m_hFlashAttention = 0;
        // The number of heads.
        int m_nHeads;
        int m_nEmbed;
        int m_nBlockSize;
        double m_dfAttnDropout;
        double m_dfResidDropout;
        double m_dfIgnoreVal = -1e+29;

        int m_nSize;
        int m_nDataSize;
        int m_nB;
        int m_nT;
        int m_nC;

        BlobCollection<T> m_colInternalBottom = new BlobCollection<T>();
        BlobCollection<T> m_colInternalTop = new BlobCollection<T>();

        /// <summary>
        /// The CausalSelfAttention constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter inner_product_param, with options:
        /// </param>
        public CausalSelfAttentionLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.CAUSAL_SELF_ATTENTION;

            m_nHeads = (int)p.causal_self_attention_param.heads;
            m_nEmbed = (int)p.causal_self_attention_param.embed;
            m_nBlockSize = (int)p.causal_self_attention_param.block_size;
            m_dfAttnDropout = p.causal_self_attention_param.attn_dropout;
            m_dfResidDropout = p.causal_self_attention_param.resid_dropout;

            log.CHECK_EQ(m_nEmbed % m_nHeads, 0, "The embedding size must be divisible by the number of heads.");

            // Key, query, value projections for all heads, but in a batch.
            // input features = m_nHeads
            LayerParameter ipAttn = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, m_param.name + ".c_attn", m_phase, p.freeze_learning);
            ipAttn.inner_product_param.num_output = (uint)(3 * m_nEmbed);
            ipAttn.inner_product_param.bias_term = p.causal_self_attention_param.bias_term;
            ipAttn.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02); 
            ipAttn.inner_product_param.bias_filler = new FillerParameter("constant", 0.0); 
            ipAttn.inner_product_param.axis = 2;
            ipAttn.output_adapter = p.causal_self_attention_param.output_adapter_q;
            ipAttn.parameters.Add(new ParamSpec(1.0, 1.0));
            ipAttn.parameters.Add(new ParamSpec(1.0, 0.0));
            m_c_attn = Layer<T>.Create(cuda, log, convertLayerParam(ipAttn, p), null);

            // Output projection.
            // input features = m_nEmbed
            LayerParameter ipProj = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, m_param.name + ".c_proj", m_phase, p.freeze_learning);
            ipProj.inner_product_param.num_output = (uint)m_nEmbed;
            ipProj.inner_product_param.bias_term = p.causal_self_attention_param.bias_term;
            ipProj.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02 / Math.Sqrt(2 * m_param.causal_self_attention_param.layers)); 
            ipProj.inner_product_param.bias_filler = new FillerParameter("constant", 0.0); 
            ipProj.inner_product_param.axis = 2;
            ipProj.output_adapter = p.causal_self_attention_param.output_adapter_out;
            ipProj.parameters.Add(new ParamSpec(1.0, 1.0));
            ipProj.parameters.Add(new ParamSpec(1.0, 0.0));
            m_c_proj = Layer<T>.Create(cuda, log, convertLayerParam(ipProj, p), null);

            // Regularization
            if (m_dfAttnDropout > 0)
            {
                LayerParameter dropoutAttn = new LayerParameter(LayerParameter.LayerType.DROPOUT, m_param.name + ".drop.attn", m_phase, p.freeze_learning);
                dropoutAttn.dropout_param.dropout_ratio = m_dfAttnDropout;
                m_attn_dropout = Layer<T>.Create(cuda, log, convertLayerParam(dropoutAttn, p), null);
            }

            if (m_dfResidDropout > 0)
            {
                LayerParameter dropoutResid = new LayerParameter(LayerParameter.LayerType.DROPOUT, m_param.name + ".drop.res", m_phase, p.freeze_learning);
                dropoutResid.dropout_param.dropout_ratio = m_dfResidDropout;
                m_resid_dropout = Layer<T>.Create(cuda, log, convertLayerParam(dropoutResid, p), null);
            }

            // Transpose
            LayerParameter transpose = new LayerParameter(LayerParameter.LayerType.TRANSPOSE, m_param.name + ".trans", m_phase, p.freeze_learning);
            transpose.transpose_param.dim[1] = 2;
            transpose.transpose_param.dim[2] = 1;
            m_transpose = Layer<T>.Create(cuda, log, convertLayerParam(transpose, p), null);

            LayerParameter transposeK = new LayerParameter(LayerParameter.LayerType.TRANSPOSE, m_param.name + ".trans.k", m_phase, p.freeze_learning);
            transposeK.transpose_param.dim[2] = 3;
            transposeK.transpose_param.dim[3] = 2;
            m_transposeQ = Layer<T>.Create(cuda, log, convertLayerParam(transposeK, p), null);

            // Softmax
            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX, m_param.name + ".smx", m_phase, p.freeze_learning);
            softmax.softmax_param.axis = -1;
            softmax.softmax_param.engine = EngineParameter.Engine.CAFFE;
            m_softmax = Layer<T>.Create(cuda, log, convertLayerParam(softmax, p), null);

            // Causal mask to ensure that atttention is only applied to the left in the input sequence.
            m_blobMask = new Blob<T>(cuda, log);
            m_blobMask.Name = m_param.name + ".mask";

            List<int> rgShape = new List<int>() { 1, 1, m_nBlockSize, m_nBlockSize };
            m_blobMask.Reshape(rgShape);
            fillMask(m_blobMask);

            m_blobQ = new Blob<T>(cuda, log);
            m_blobQ.Name = m_param.name + ".Q";
            m_blobK = new Blob<T>(cuda, log);
            m_blobK.Name = m_param.name + ".K";
            m_blobV = new Blob<T>(cuda, log);
            m_blobV.Name = m_param.name + ".V";
            m_blobQt = new Blob<T>(cuda, log);
            m_blobQt.Name = m_param.name + ".Qt";
            m_blobKt = new Blob<T>(cuda, log);
            m_blobKt.Name = m_param.name + ".Kt";
            m_blobKt1 = new Blob<T>(cuda, log);
            m_blobKt1.Name = m_param.name + ".Kt1";
            m_blobVt = new Blob<T>(cuda, log);
            m_blobVt.Name = m_param.name + ".Vt";
            m_blobAttA = new Blob<T>(cuda, log);
            m_blobAttA.Name = m_param.name + ".AttA";
            m_blobAttB = new Blob<T>(cuda, log);
            m_blobAttB.Name = m_param.name + ".AttB";
            m_blobWork = new Blob<T>(cuda, log);
            m_blobWork.Name = m_param.name + ".Work";

            m_blobIpAttn = new Blob<T>(cuda, log);
            m_blobIpAttn.Name = m_param.name + ".IpAttn";
            m_blobY = new Blob<T>(cuda, log);
            m_blobY.Name = m_param.name + ".Y";

            setup_internal_blobs(m_colInternalBlobs);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_c_attn);
            dispose(ref m_c_proj);
            dispose(ref m_attn_dropout);
            dispose(ref m_resid_dropout);
            dispose(ref m_transpose);
            dispose(ref m_transposeQ);
            dispose(ref m_softmax);

            dispose(ref m_blobMask);
            dispose(ref m_blobQ);
            dispose(ref m_blobK);
            dispose(ref m_blobV);
            dispose(ref m_blobQt);
            dispose(ref m_blobKt);
            dispose(ref m_blobKt1);
            dispose(ref m_blobVt);
            dispose(ref m_blobAttA);
            dispose(ref m_blobAttB);
            dispose(ref m_blobWork);
            dispose(ref m_blobIpAttn);
            dispose(ref m_blobY);

            if (m_hFlashAttention != 0)
            {
                m_cuda.FreeAttn(m_hFlashAttention);
                m_hFlashAttention = 0;
            }

            if (m_hRope != 0)
            {
                m_cuda.FreeRope(m_hRope);
                m_hRope = 0;
            }

            if (m_hCudnn != 0)
            {
                m_cuda.FreeCuDNN(m_hCudnn);
                m_hCudnn = 0;
            }

            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobIpAttn);
            col.Add(m_blobQ);
            col.Add(m_blobK);
            col.Add(m_blobV);
            col.Add(m_blobQt);
            col.Add(m_blobKt);
            col.Add(m_blobVt);
            col.Add(m_blobKt1);
            col.Add(m_blobAttA);
            col.Add(m_blobMask);
            col.Add(m_blobAttB);
            col.Add(m_blobWork);
            col.Add(m_blobY);

            col.Add(m_c_attn.internal_blobs);
            col.Add(m_transpose.internal_blobs);
            col.Add(m_transposeQ.internal_blobs);
            col.Add(m_softmax.internal_blobs);
            if (m_attn_dropout != null)
                col.Add(m_attn_dropout.internal_blobs);
            col.Add(m_c_proj.internal_blobs);
            if (m_resid_dropout != null)
                col.Add(m_resid_dropout.internal_blobs);
        }

        private void fillMask(Blob<T> b)
        {
            b.SetData(1.0);

            float[] rgMaskData = convertF(b.mutable_cpu_data);

            for (int i = 0; i<b.height; i++)
            {
                for (int j = i + 1; j < b.width; j++)
                {
                    rgMaskData[i * b.width + j] = 0;
                }
            }

            b.mutable_cpu_data = convert(rgMaskData);
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
            
            m_c_attn.ReInitializeParameters(target);
            m_c_proj.ReInitializeParameters(target);

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

            m_nB = blobX.num;         // batch size
            m_nT = blobX.channels;    // sequence length
            m_nC = blobX.height;      // embedding dim (m_nEmbed)
            m_nSize = m_nC / (int)m_nHeads;

            m_nDataSize = blobX.count(3);

            m_nSize *= m_nDataSize;

            addInternal(blobX, m_blobIpAttn);
            m_c_attn.Setup(m_colInternalBottom, m_colInternalTop);

            blobs.Add(m_c_attn.blobs[0]);
            if (m_param.causal_self_attention_param.bias_term)
                blobs.Add(m_c_attn.blobs[1]);

            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nT;
            m_rgShape[2] = m_nHeads;
            m_rgShape[3] = m_nSize;

            shareLayerBlob(m_blobQ, m_rgShape);
            m_blobQ.Reshape(m_rgShape);
            addInternal(m_blobQ, m_blobQt);
            m_transpose.Setup(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)

            shareLayerBlob(m_blobAttA, blobX.shape());
            m_blobAttA.ReshapeLike(blobX);
            shareLayerBlob(m_blobAttB, blobX.shape());
            m_blobAttB.ReshapeLike(blobX);
            addInternal(m_blobAttA, m_blobAttB);
            m_softmax.Setup(m_colInternalBottom, m_colInternalTop);

            if (m_attn_dropout != null)
            {
                addInternal(m_blobAttB, m_blobAttB);
                m_attn_dropout.Setup(m_colInternalBottom, m_colInternalTop);
            }

            if (m_param.causal_self_attention_param.enable_rotary_positional_embedding)
                m_hRope = m_cuda.CreateRope(m_cuda.GetDeviceID(), colBottom[0].count(), m_nB, m_nT, m_nSize * m_nHeads);

            if (m_param.causal_self_attention_param.enable_flash_scaled_dot_product_attention)
            {
                m_hCudnn = m_cuda.CreateCuDNN();
                m_hFlashAttention = m_cuda.CreateAttn();
                m_cuda.SetAttn(m_hCudnn, m_hFlashAttention, m_cuda.GetDeviceID(), (m_phase == Phase.TRAIN) ? true : false, m_nB, m_nT, m_nHeads, m_nEmbed / m_nHeads, (float)m_param.causal_self_attention_param.attn_dropout);
            }

            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nT;
            m_rgShape[2] = m_nC;
            m_rgShape[3] = m_nDataSize;

            shareLayerBlob(m_blobY, m_rgShape);
            m_blobY.Reshape(m_rgShape);

            addInternal(m_blobY, colTop[0]);
            m_c_proj.Setup(m_colInternalBottom, m_colInternalTop);

            blobs.Add(m_c_proj.blobs[0]);
            if (m_param.causal_self_attention_param.bias_term)
                blobs.Add(m_c_proj.blobs[1]);

            if (m_resid_dropout != null)
            {
                addInternal(colTop[0], colTop[0]);
                m_resid_dropout.Setup(m_colInternalBottom, m_colInternalTop);
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

            m_nB = blobX.num;         // batch size
            m_nT = blobX.channels;    // sequence length
            m_nC = blobX.height;      // embedding dim (m_nEmbed)
            m_nSize = m_nC / m_nHeads;

            m_nDataSize = blobX.count(3);

            m_nSize *= m_nDataSize;

            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nT;
            m_rgShape[2] = m_nHeads;
            m_rgShape[3] = m_nSize;

            shareLayerBlob(m_blobK, m_rgShape);
            m_blobK.Reshape(m_rgShape);
            shareLayerBlob(m_blobKt1, m_rgShape);
            m_blobKt1.Reshape(m_rgShape);
            shareLayerBlob(m_blobKt, m_rgShape);

            addInternal(m_blobK, m_blobKt);
            m_transpose.Reshape(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)
            m_blobKt1.ReshapeLike(m_blobKt);

            shareLayerBlob(m_blobQ, m_rgShape);
            m_blobQ.Reshape(m_rgShape);
            shareLayerBlob(m_blobQt, m_rgShape);

            addInternal(m_blobQ, m_blobQt);
            m_transpose.Reshape(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)

            shareLayerBlob(m_blobV, m_rgShape);
            m_blobV.Reshape(m_rgShape);
            shareLayerBlob(m_blobVt, m_rgShape);

            m_blobV.Reshape(m_nB, m_nT, m_nHeads, m_nSize);
            addInternal(m_blobV, m_blobVt);
            m_transpose.Reshape(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)

            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nHeads;
            m_rgShape[2] = m_nT;
            m_rgShape[3] = m_nT;

            shareLayerBlob(m_blobAttA, m_rgShape);
            m_blobAttA.Reshape(m_rgShape);
            shareLayerBlob(m_blobAttB, m_rgShape);
            m_blobAttB.Reshape(m_rgShape);

            m_rgShape[0] = m_blobVt.num;
            m_rgShape[1] = m_blobVt.channels;
            m_rgShape[2] = m_blobVt.width;  // col major
            m_rgShape[3] = m_blobVt.height;

            shareLayerBlob(m_blobWork, m_rgShape);
            m_blobWork.Reshape(m_rgShape); // col major
            
            addInternal(m_blobWork, m_blobY);
            m_transposeQ.Reshape(m_colInternalBottom, m_colInternalTop);

            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nT;
            m_rgShape[2] = m_nC;
            m_rgShape[3] = m_nDataSize;

            shareLayerBlob(m_blobY, m_rgShape);
            m_blobY.Reshape(m_rgShape);
            
            addInternal(m_blobY, colTop[0]);
            m_c_proj.Reshape(m_colInternalBottom, m_colInternalTop);
            
            if (m_resid_dropout != null)
            {
                addInternal(colTop[0], colTop[0]);
                m_resid_dropout.Reshape(m_colInternalBottom, m_colInternalTop);
            }

            if (m_blobMask.height != m_nT || m_blobMask.width != m_nT)
            {
                List<int> rgShape = new List<int>() { 1, 1, m_nT, m_nT };
                m_blobMask.Reshape(rgShape);
                fillMask(m_blobMask);
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
        ///     the computed causal self attention.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobX = colBottom[0];

            // Calculate query, key, values for all heads in batch and move head forward to be the batch dim.
            // q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
            addInternal(blobX, m_blobIpAttn);
            m_c_attn.Forward(m_colInternalBottom, m_colInternalTop);

            // Split IP output (3 * nEmbed) into query, key, values.
            int nCount = m_blobQ.count();            
            m_cuda.channel_copy(nCount, m_blobIpAttn.num, m_blobIpAttn.channels, 3, m_nEmbed, 0, m_blobIpAttn.gpu_data, m_blobQ.mutable_gpu_data, DIR.FWD);
            m_cuda.channel_copy(nCount, m_blobIpAttn.num, m_blobIpAttn.channels, 3, m_nEmbed, 1, m_blobIpAttn.gpu_data, m_blobK.mutable_gpu_data, DIR.FWD);
            m_cuda.channel_copy(nCount, m_blobIpAttn.num, m_blobIpAttn.channels, 3, m_nEmbed, 2, m_blobIpAttn.gpu_data, m_blobV.mutable_gpu_data, DIR.FWD);

            // When using rope, apply the rotary positional embedding.
            if (m_hRope != 0)
            {
                m_cuda.RopeForward(m_hRope, m_blobQ.count(), m_blobQ.gpu_data, m_blobQ.mutable_gpu_data);
                m_cuda.RopeForward(m_hRope, m_blobK.count(), m_blobK.gpu_data, m_blobK.mutable_gpu_data);
            }

            // Transpose query, key and values along axes 1 & 2
            // q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            // k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            // v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            addInternal(m_blobQ, m_blobQt);
            m_transpose.Forward(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)
            addInternal(m_blobK, m_blobKt);
            m_transpose.Forward(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)
            addInternal(m_blobV, m_blobVt);
            m_transpose.Forward(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)

            // Perform Self Attention forward pass
            if (m_param.causal_self_attention_param.enable_flash_scaled_dot_product_attention)
            {
                m_blobWork.Reshape(m_blobVt.num, m_blobVt.channels, m_blobVt.height, m_blobVt.width);
                m_cuda.AttnScaledDotProductForward(m_hCudnn, m_hFlashAttention, m_blobQt.gpu_data, m_blobKt.gpu_data, m_blobVt.gpu_data, m_blobMask.gpu_data, m_blobWork.mutable_gpu_data);
            }
            else
            {
                // Multiply query and key(T) matrices and scale
                // att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

                addInternal(m_blobKt, m_blobKt1);
                m_transposeQ.Forward(m_colInternalBottom, m_colInternalTop);

                double dfScale = 1.0 / Math.Sqrt(m_nSize);
                m_blobAttA.MatMul(m_blobQt, m_blobKt1);
                m_blobAttA.scale_data(dfScale);

                // Apply mask to attention matrix
                // att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                m_cuda.mask(m_blobAttA.count(), m_blobMask.count(), convert(0.0), convert(m_dfIgnoreVal), m_blobAttA.gpu_data, m_blobMask.gpu_data, m_blobAttA.mutable_gpu_data); // all masked items set to -inf.

                // Take softmax of attention along the last axis.
                // att = F.softmax(att, dim = -1)
                addInternal(m_blobAttA, m_blobAttB);
                m_softmax.Forward(m_colInternalBottom, m_colInternalTop);

                // Apply attention dropout.
                // att = self.attn_dropout(att)
                if (m_attn_dropout != null)
                {
                    addInternal(m_blobAttB, m_blobAttB);
                    m_attn_dropout.Forward(m_colInternalBottom, m_colInternalTop);
                }

                m_blobWork.Reshape(m_blobVt.num, m_blobVt.channels, m_blobVt.height, m_blobVt.width);

                // Multiply attention matrix with values
                // y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                m_blobWork.MatMul(m_blobAttB, m_blobVt);
            }

            // Reassemble all head outputs side by side.
            // y = y.transpose(1, 2).contiguous().view(B, T, C) 
            addInternal(m_blobWork, m_blobY);
            m_transpose.Forward(m_colInternalBottom, m_colInternalTop); 
            m_blobY.Reshape(m_nB, m_nT, m_nC, m_nDataSize);

            // Apply output projection.
            // y = self.resid_dropout(self.c_proj(y))
            addInternal(m_blobY, colTop[0]);
            m_c_proj.Forward(m_colInternalBottom, m_colInternalTop);

            // Apply resid dropout
            if (m_resid_dropout != null)
            {
                addInternal(colTop[0], colTop[0]);
                m_resid_dropout.Forward(m_colInternalBottom, m_colInternalTop);
            }

            m_rgYShape[0] = m_blobY.num;
            m_rgYShape[1] = m_blobY.channels;
            m_rgYShape[2] = m_blobY.height;
            m_rgYShape[3] = m_blobY.width;

            m_rgWorkShape[0] = m_blobWork.num;
            m_rgWorkShape[1] = m_blobWork.channels;
            m_rgWorkShape[2] = m_blobWork.height;
            m_rgWorkShape[3] = m_blobWork.width;
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
            m_blobY.Reshape(m_rgYShape);
            m_blobWork.Reshape(m_rgWorkShape);

            // Gradient with respect to state then data.
            if (rgbPropagateDown[0])
            {
                List<bool> rgbPropagate = new List<bool>() { true, true };

                // Apply resid dropout
                if (m_resid_dropout != null)
                {
                    addInternal(colTop[0], colTop[0]);
                    m_resid_dropout.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                }

                // Apply output projection.
                // y = self.resid_dropout(self.c_proj(y))
                addInternal(m_blobY, colTop[0]);
                m_c_proj.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Reassemble all head outputs side by side.
                // y = y.transpose(1, 2).contiguous().view(B, T, C) 
                addInternal(m_blobWork, m_blobY);
                m_transpose.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Perform Self Attention backward pass
                if (m_param.causal_self_attention_param.enable_flash_scaled_dot_product_attention)
                {
                    m_blobY.CopyFrom(m_blobWork, true, true);
                    m_cuda.AttnScaledDotProductBackward(m_hCudnn, m_hFlashAttention, m_blobQt.gpu_data, m_blobQt.mutable_gpu_diff, m_blobKt.gpu_data, m_blobKt.mutable_gpu_diff, m_blobVt.gpu_data, m_blobVt.mutable_gpu_diff, 0, m_blobY.gpu_data, m_blobY.gpu_diff);
                }
                else
                {
                    // Multiply attention matrix with values
                    // y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                    m_blobY.CopyFrom(m_blobWork, true, true);

                    // Multiply attention matrix with values
                    // y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                    // Gradient with respect to att
                    // att' = y' @ v^T 
                    // Gradient with respect to vt
                    // vt' = att^T @ y' 
                    m_blobY.MatMulGrad(m_blobAttB, m_blobVt, m_blobWork);

                    // Apply attention dropout.
                    // att = self.attn_dropout(att)
                    if (m_attn_dropout != null)
                    {
                        addInternal(m_blobAttB, m_blobAttB);
                        m_attn_dropout.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                    }

                    // Take softmax of attention along the last axis.
                    // att = F.softmax(att, dim = -1)
                    addInternal(m_blobAttA, m_blobAttB);
                    m_softmax.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                    // Multiply qt with kt^T to create attention matrix
                    // att = qt @ kt^T
                    // Gradient with respect to qt
                    // qt' = att' @ kt
                    // Gradient with respect to qt
                    // qt' = att' @ kt
                    double dfScale = 1.0 / Math.Sqrt(m_nSize);
                    m_blobAttA.MatMulGrad(m_blobQt, m_blobKt1, m_blobWork, dfScale);

                    // Transpose Kt1 back to Kt
                    addInternal(m_blobKt, m_blobKt1);
                    m_transposeQ.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                }

                // Transpose query, key and values along axes 1 & 2
                // k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
                // q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
                // v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
                addInternal(m_blobK, m_blobKt);
                m_transpose.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom); // (B, nh, T, hs)
                addInternal(m_blobQ, m_blobQt);
                m_transpose.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom); // (B, nh, T, hs)
                addInternal(m_blobV, m_blobVt);
                m_transpose.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom); // (B, nh, T, hs)

                // When using rope, handle the rotary positional embedding.
                if (m_hRope != 0)
                {
                    m_cuda.RopeBackward(m_hRope, m_blobQ.count(), m_blobQ.gpu_data, m_blobQ.gpu_diff, m_blobQ.mutable_gpu_diff);
                    m_cuda.RopeBackward(m_hRope, m_blobK.count(), m_blobK.gpu_data, m_blobK.gpu_diff, m_blobK.mutable_gpu_diff);
                }

                // Split IP output (3 * nEmbed) into query, key, values.
                int nCount = m_blobQ.count();
                m_cuda.channel_copy(nCount, m_blobIpAttn.num, m_blobIpAttn.channels, 3, m_nEmbed, 0, m_blobIpAttn.mutable_gpu_diff, m_blobQ.gpu_diff, DIR.BWD);
                m_cuda.channel_copy(nCount, m_blobIpAttn.num, m_blobIpAttn.channels, 3, m_nEmbed, 1, m_blobIpAttn.mutable_gpu_diff, m_blobK.gpu_diff, DIR.BWD);
                m_cuda.channel_copy(nCount, m_blobIpAttn.num, m_blobIpAttn.channels, 3, m_nEmbed, 2, m_blobIpAttn.mutable_gpu_diff, m_blobV.gpu_diff, DIR.BWD);

                // Calculate query, key, values for all heads in batch and move head forward to be the batch dim.
                // q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
                addInternal(colBottom[0], m_blobIpAttn);
                m_c_attn.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
            }
        }
    }
}
