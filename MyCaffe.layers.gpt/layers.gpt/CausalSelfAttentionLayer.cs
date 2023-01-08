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
        Blob<T> m_blobBias;
        Blob<T> m_blobQ;
        Blob<T> m_blobK;
        Blob<T> m_blobV;
        Blob<T> m_blobQt;
        Blob<T> m_blobQt1;
        Blob<T> m_blobKt;
        Blob<T> m_blobKt1;
        Blob<T> m_blobVt;
        Blob<T> m_blobVt1;
        Blob<T> m_blobWork;
        Blob<T> m_blobAtt;
        Blob<T> m_blobIpAttn;
        Blob<T> m_blobY;
        // The number of heads.
        int m_nHeads;
        int m_nEmbed;
        int m_nBlockSize;
        double m_dfAttnDropout;
        double m_dfResidDropout;

        int m_nSize;
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
            LayerParameter ipAttn = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "c_attn");
            ipAttn.inner_product_param.num_output = (uint)(3 * m_nEmbed);
            ipAttn.inner_product_param.bias_term = true;
            ipAttn.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02); 
            ipAttn.inner_product_param.bias_filler = new FillerParameter("constant", 0.0); 
            ipAttn.inner_product_param.axis = 2;
            ipAttn.parameters.Add(new ParamSpec(1.0, 1.0));
            ipAttn.parameters.Add(new ParamSpec(1.0, 0.0));
            m_c_attn = Layer<T>.Create(cuda, log, convertLayerParam(ipAttn, p), null);

            // Output projection.
            // input features = m_nEmbed
            LayerParameter ipProj = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "c_proj");
            ipProj.inner_product_param.num_output = (uint)m_nEmbed;
            ipProj.inner_product_param.bias_term = true;
            ipProj.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02 / Math.Sqrt(2 * m_param.causal_self_attention_param.layers)); 
            ipProj.inner_product_param.bias_filler = new FillerParameter("constant", 0.0); 
            ipProj.inner_product_param.axis = 2;            
            ipProj.parameters.Add(new ParamSpec(1.0, 1.0));
            ipProj.parameters.Add(new ParamSpec(1.0, 0.0));
            m_c_proj = Layer<T>.Create(cuda, log, convertLayerParam(ipProj, p), null);

            // Regularization
            if (m_dfAttnDropout > 0)
            {
                LayerParameter dropoutAttn = new LayerParameter(LayerParameter.LayerType.DROPOUT);
                dropoutAttn.dropout_param.dropout_ratio = m_dfAttnDropout;
                m_attn_dropout = Layer<T>.Create(cuda, log, convertLayerParam(dropoutAttn, p), null);
            }

            if (m_dfResidDropout > 0)
            {
                LayerParameter dropoutResid = new LayerParameter(LayerParameter.LayerType.DROPOUT);
                dropoutResid.dropout_param.dropout_ratio = m_dfResidDropout;
                m_resid_dropout = Layer<T>.Create(cuda, log, convertLayerParam(dropoutResid, p), null);
            }

            // Transpose
            LayerParameter transpose = new LayerParameter(LayerParameter.LayerType.TRANSPOSE);
            transpose.transpose_param.dim[1] = 2;
            transpose.transpose_param.dim[2] = 1;
            m_transpose = Layer<T>.Create(cuda, log, convertLayerParam(transpose, p), null);

            LayerParameter transposeK = new LayerParameter(LayerParameter.LayerType.TRANSPOSE);
            transposeK.transpose_param.dim[2] = 3;
            transposeK.transpose_param.dim[3] = 2;
            m_transposeQ = Layer<T>.Create(cuda, log, convertLayerParam(transposeK, p), null);

            // Softmax
            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            softmax.softmax_param.axis = -1;
            softmax.softmax_param.engine = EngineParameter.Engine.CUDNN;
            m_softmax = Layer<T>.Create(cuda, log, convertLayerParam(softmax, p), null);

            // Causal mask to ensure that atttention is only applied to the left in the input sequence.
            m_blobBias = new Blob<T>(cuda, log);
            m_blobBias.Name = m_param.name + " bias";

            List<int> rgShape = new List<int>() { 1, 1, m_nBlockSize, m_nBlockSize };
            shareLayerBlob(m_blobBias, rgShape);
            m_blobBias.Reshape(rgShape);
            fillBias(m_blobBias);
           
            m_blobQ = new Blob<T>(cuda, log);
            m_blobQ.Name = m_param.name + " Q";
            m_blobK = new Blob<T>(cuda, log);
            m_blobK.Name = m_param.name + " K";
            m_blobV = new Blob<T>(cuda, log);
            m_blobV.Name = m_param.name + " V";
            m_blobQt = new Blob<T>(cuda, log);
            m_blobQt.Name = m_param.name + " Qt";
            m_blobQt1 = new Blob<T>(cuda, log, false);
            m_blobQt1.Name = m_param.name + " Qt1";
            m_blobKt = new Blob<T>(cuda, log);
            m_blobKt.Name = m_param.name + " Kt";
            m_blobKt1 = new Blob<T>(cuda, log);
            m_blobKt1.Name = m_param.name + " Kt1";
            m_blobVt = new Blob<T>(cuda, log);
            m_blobVt.Name = m_param.name + " Vt";
            m_blobVt1 = new Blob<T>(cuda, log, false);
            m_blobVt1.Name = m_param.name + " Vt1";
            m_blobAtt = new Blob<T>(cuda, log);
            m_blobAtt.Name = m_param.name + " Att";
            m_blobWork = new Blob<T>(cuda, log);
            m_blobWork.Name = m_param.name + " Work";

            m_blobIpAttn = new Blob<T>(cuda, log);
            m_blobIpAttn.Name = m_param.name + " IpAttn";
            m_blobY = new Blob<T>(cuda, log);
            m_blobY.Name = m_param.name + " Y";

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

            dispose(ref m_blobBias);
            dispose(ref m_blobQ);
            dispose(ref m_blobK);
            dispose(ref m_blobV);
            dispose(ref m_blobQt);
            dispose(ref m_blobQt1);
            dispose(ref m_blobKt);
            dispose(ref m_blobKt1);
            dispose(ref m_blobVt);
            dispose(ref m_blobVt1);
            dispose(ref m_blobAtt);
            dispose(ref m_blobWork);
            dispose(ref m_blobIpAttn);
            dispose(ref m_blobY);

            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobBias);
            col.Add(m_blobQ);
            col.Add(m_blobK);
            col.Add(m_blobV);
            col.Add(m_blobQt);
            col.Add(m_blobQt1);
            col.Add(m_blobKt);
            col.Add(m_blobKt1);
            col.Add(m_blobVt);
            col.Add(m_blobVt1);
            col.Add(m_blobAtt);
            col.Add(m_blobWork);
            col.Add(m_blobIpAttn);
            col.Add(m_blobY);

            col.Add(m_c_attn.internal_blobs);
            col.Add(m_c_proj.internal_blobs);
            if (m_attn_dropout != null)
                col.Add(m_attn_dropout.internal_blobs);
            if (m_resid_dropout != null)
                col.Add(m_resid_dropout.internal_blobs);
            col.Add(m_transpose.internal_blobs);
            col.Add(m_transposeQ.internal_blobs);
            col.Add(m_softmax.internal_blobs);
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

            addInternal(blobX, m_blobIpAttn);
            m_c_attn.Setup(m_colInternalBottom, m_colInternalTop);

            blobs.Add(m_c_attn.blobs[0]);
            blobs.Add(m_c_attn.blobs[1]);

            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nT;
            m_rgShape[2] = m_nHeads;
            m_rgShape[3] = m_nSize;

            shareLayerBlob(m_blobQ, m_rgShape);
            m_blobQ.Reshape(m_rgShape);
            addInternal(m_blobQ, m_blobQt);
            m_transpose.Setup(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)

            shareLayerBlob(m_blobAtt, blobX.shape());
            m_blobAtt.ReshapeLike(blobX);
            addInternal(m_blobAtt, m_blobAtt);
            m_softmax.Setup(m_colInternalBottom, m_colInternalTop);

            if (m_attn_dropout != null)
            {
                addInternal(m_blobAtt, m_blobAtt);
                m_attn_dropout.Setup(m_colInternalBottom, m_colInternalTop);
            }
            
            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nT;
            m_rgShape[2] = m_nC;
            m_rgShape[3] = 1;

            shareLayerBlob(m_blobY, m_rgShape);
            m_blobY.Reshape(m_rgShape);

            addInternal(m_blobY, colTop[0]);
            m_c_proj.Setup(m_colInternalBottom, m_colInternalTop);

            blobs.Add(m_c_proj.blobs[0]);
            blobs.Add(m_c_proj.blobs[1]);

            if (m_resid_dropout != null)
            {
                addInternal(colTop[0], colTop[0]);
                m_resid_dropout.Setup(m_colInternalBottom, m_colInternalTop);
            }

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

            Blob<T> blobX = colBottom[0];

            m_nB = blobX.num;         // batch size
            m_nT = blobX.channels;    // sequence length
            m_nC = blobX.height;      // embedding dim (m_nEmbed)
            m_nSize = m_nC / m_nHeads;

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
            shareLayerBlob(m_blobQt1, m_rgShape);
            m_blobQt1.Reshape(m_rgShape);
            shareLayerBlob(m_blobQt, m_rgShape);

            addInternal(m_blobQ, m_blobQt);
            m_transpose.Reshape(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)
            m_blobQt1.ReshapeLike(m_blobQt);

            shareLayerBlob(m_blobV, m_rgShape);
            m_blobV.Reshape(m_rgShape);
            shareLayerBlob(m_blobVt1, m_rgShape);
            m_blobVt1.Reshape(m_rgShape);
            shareLayerBlob(m_blobVt, m_rgShape);

            m_blobV.Reshape(m_nB, m_nT, m_nHeads, m_nSize);
            addInternal(m_blobV, m_blobVt);
            m_transpose.Reshape(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)
            m_blobVt1.ReshapeLike(m_blobVt);

            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nHeads;
            m_rgShape[2] = m_nT;
            m_rgShape[3] = m_nT;

            shareLayerBlob(m_blobAtt, m_rgShape);
            m_blobAtt.Reshape(m_rgShape);

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
            m_rgShape[3] = 1;

            shareLayerBlob(m_blobY, m_rgShape);
            m_blobY.Reshape(m_rgShape);
            
            addInternal(m_blobY, colTop[0]);
            m_c_proj.Reshape(m_colInternalBottom, m_colInternalTop);
            
            if (m_resid_dropout != null)
            {
                addInternal(colTop[0], colTop[0]);
                m_resid_dropout.Reshape(m_colInternalBottom, m_colInternalTop);
            }
        }

        private void gemm_fwd(double dfScale, Blob<T> blobA, Blob<T> blobB, Blob<T> blobC)
        {
            int nAxis = 2;
            int nM = blobA.height;
            int nN = blobB.width;
            int nK = blobB.height;

            int nOuterDim = blobA.count(0, nAxis);
            uint lda = (uint)nN;
            uint ldb = (uint)nK;
            uint ldc = (uint)nN;
            uint strideb = (uint)(nM * nK);
            uint stridea = (uint)(nK * nN);
            uint stridec = (uint)(nM * nN);

            // cuBlas performs gemm in col-maj, performing Kt1(rm) x Qt(rm) = Att(rm), (e.g. reverse of att = q @ k)
            // @see [How to transpose a matrix in CUDA/cublas](https://stackoverflow.com/questions/13782012/how-to-transpose-a-matrix-in-cuda-cublas)
            m_cuda.gemm(false, false, nN, nM, nK, dfScale, blobB.gpu_data, blobA.gpu_data, 0.0, blobC.mutable_gpu_data, lda, ldb, ldc, stridea, strideb, stridec, (uint)nOuterDim);
        }

        private void gemm_bwd1(double dfScale, Blob<T> blobA, Blob<T> blobB, Blob<T> blobC)
        {
            int nM = blobA.height;
            int nN = blobB.width;
            int nK = blobB.height;

            int nAxis = 2;
            int nOuterDim = blobA.count(0, nAxis);
            uint lda = (uint)nN;
            uint ldb = (uint)nK;
            uint ldc = (uint)nN;
            uint strideb = (uint)(nM * nK);
            uint stridea = (uint)(nK * nN);
            uint stridec = (uint)(nM * nN);

            // cuBlas performs gemm in col-maj, performing B(rm) x A'(rm) = C'(rm), (e.g. reverse of c' = a' @ b)
            // @see [How to transpose a matrix in CUDA/cublas](https://stackoverflow.com/questions/13782012/how-to-transpose-a-matrix-in-cuda-cublas)
            m_cuda.gemm(false, false, nN, nM, nK, dfScale, blobB.gpu_data, blobA.gpu_diff, 0.0, blobC.mutable_gpu_diff, lda, ldb, ldc, stridea, strideb, stridec, (uint)nOuterDim);
        }

        private void gemm_bwd2(double dfScale, bool bTransposeA, Blob<T> blobA, Blob<T> blobB, Blob<T> blobC)
        {
            int nM = blobA.height;
            int nN = blobB.width;
            int nK = blobB.height;

            int nAxis = 2;
            int nOuterDim = blobA.count(0, nAxis);
            uint lda = (uint)nN;
            uint ldb = (uint)nK;
            uint ldc = (uint)nN;
            uint strideb = (uint)(nM * nK);
            uint stridea = (uint)(nK * nN);
            uint stridec = (uint)(nM * nN);

            // cuBlas performs gemm in col-maj, performing B'(rm) x A(rm)^T = C'(rm), (e.g. reverse of c' = a^T @ b')
            // @see [How to transpose a matrix in CUDA/cublas](https://stackoverflow.com/questions/13782012/how-to-transpose-a-matrix-in-cuda-cublas)
            m_cuda.gemm(false, bTransposeA, nN, nM, nK, dfScale, blobB.gpu_diff, blobA.gpu_data, 0.0, blobC.mutable_gpu_diff, lda, ldb, ldc, stridea, strideb, stridec, (uint)nOuterDim);
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

            // Transpose query, key and values along axes 1 & 2
            // k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            // q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            // v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            addInternal(m_blobK, m_blobKt);
            m_transpose.Forward(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)
            addInternal(m_blobQ, m_blobQt);
            m_transpose.Forward(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)
            addInternal(m_blobV, m_blobVt);
            m_transpose.Forward(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)

            // Multiply query and key(T) matrices and scale
            // att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            addInternal(m_blobKt, m_blobKt1);
            m_transposeQ.Forward(m_colInternalBottom, m_colInternalTop);
            
            double dfScale = 1.0 / Math.Sqrt(m_nSize);
            gemm_fwd(dfScale, m_blobQt, m_blobKt1, m_blobAtt);

            // Apply mask to attention matrix
            // att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            m_cuda.mask(m_blobAtt.count(), m_blobBias.count(), convert(0.0), convert(double.NegativeInfinity), m_blobAtt.gpu_data, m_blobBias.gpu_data, m_blobAtt.mutable_gpu_data); // all masked items set to -inf.

            // Take softmax of attention along the last axis.
            // att = F.softmax(att, dim = -1)
            addInternal(m_blobAtt, m_blobAtt);
            m_softmax.Forward(m_colInternalBottom, m_colInternalTop);

            // Apply attention dropout.
            // att = self.attn_dropout(att)
            if (m_attn_dropout != null)
            {
                addInternal(m_blobAtt, m_blobAtt);
                m_attn_dropout.Forward(m_colInternalBottom, m_colInternalTop);
            }

            m_blobWork.Reshape(m_blobVt.num, m_blobVt.channels, m_blobVt.height, m_blobVt.width);

            // Multiply attention matrix with values
            // y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            gemm_fwd(1.0, m_blobAtt, m_blobVt, m_blobWork);

            // Reassemble all head outputs side by side.
            // y = y.transpose(1, 2).contiguous().view(B, T, C) 
            addInternal(m_blobWork, m_blobY);
            m_transpose.Forward(m_colInternalBottom, m_colInternalTop); 
            m_blobY.Reshape(m_nB, m_nT, m_nC, 1);

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

                // Multiply attention matrix with values
                // y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                m_blobY.CopyFrom(m_blobWork, true, true);

                // Transpose Vt
                addInternal(m_blobVt, m_blobVt1);
                m_transposeQ.Forward(m_colInternalBottom, m_colInternalTop);

                // Multiply attention matrix with values
                // y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

                // Gradient with respect to att
                // att' = y' @ v^T 
                gemm_bwd1(1.0, m_blobY, m_blobVt1, m_blobAtt);

                // Gradient with respect to vt
                // vt' = att^T @ y' 
                gemm_bwd2(1.0, true, m_blobAtt, m_blobY, m_blobVt);

                // Apply attention dropout.
                // att = self.attn_dropout(att)
                if (m_attn_dropout != null)
                {
                    addInternal(m_blobAtt, m_blobAtt);
                    m_attn_dropout.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                }

                // Take softmax of attention along the last axis.
                // att = F.softmax(att, dim = -1)
                addInternal(m_blobAtt, m_blobAtt);
                m_softmax.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Multiply qt with kt^T to create attention matrix
                // att = qt @ kt^T
                double dfScale = 1.0 / Math.Sqrt(m_nSize);

                // Gradient with respect to qt
                // qt' = att' @ kt
                gemm_bwd1(dfScale, m_blobAtt, m_blobKt, m_blobQt);
                
                // Transpose Qt
                addInternal(m_blobQt, m_blobQt1);
                m_transposeQ.Forward(m_colInternalBottom, m_colInternalTop);

                // Gradient with respect to kt^T
                // kt^T' = qt^T @ att'
                gemm_bwd2(dfScale, false, m_blobQt1, m_blobAtt, m_blobKt1);

                // Transpose Kt1 back to Kt
                addInternal(m_blobKt, m_blobKt1);
                m_transposeQ.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

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
