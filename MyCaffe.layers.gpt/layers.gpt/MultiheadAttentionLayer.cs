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

/// WORK IN PROGRESS
namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The MultiheadAttention provides a vanilla multi-head layer.
    /// </summary>
    /// <remarks>
    /// @see [GitHub:devjwsong:transformer-translator-pytorch](https://github.com/devjwsong/transformer-translator-pytorch/blob/master/src/layers.py) by Song, 2021, GitHub:devjwsong
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class MultiheadAttentionLayer<T> : Layer<T>
    {
        List<int> m_rgShape = new List<int>() { 1, 1, 1, 1 };
        // Key, query, value projections for all heads, but in a batch.
        Layer<T> m_c_attnQ = null;
        Layer<T> m_c_attnK = null;
        Layer<T> m_c_attnV = null;
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
        Blob<T> m_blobX0 = null;
        Blob<T> m_blobX1 = null;
        Blob<T> m_blobX2 = null;
        Blob<T> m_blobQ = null;
        Blob<T> m_blobK = null;
        Blob<T> m_blobV = null;
        Blob<T> m_blobQt;
        Blob<T> m_blobKt;
        Blob<T> m_blobKt1;
        Blob<T> m_blobVt;
        Blob<T> m_blobWork;
        Blob<T> m_blobAttA;
        Blob<T> m_blobAttB;
        Blob<T> m_blobY;
        Blob<T> m_blobKeyCache = null;
        Blob<T> m_blobValCache = null;
        Blob<T> m_blobWork2 = null;
        Blob<T> m_blobWork3 = null;
        long m_hRope = 0;
        long m_hCudnn = 0;
        long m_hCudaAttention = 0;
        // The number of heads.
        int m_nHeads;
        int m_nEmbed;
        int m_nBlockSize;
        double m_dfAttnDropout;
        double m_dfResidDropout;
        double m_dfIgnoreVal = -1e+29;
        int[] m_rgYShape = new int[4];
        int[] m_rgWorkShape = new int[4];

        int m_nSize;
        int m_nB;
        int m_nT;
        int m_nC;

        BlobCollection<T> m_colInternalBottom = new BlobCollection<T>();
        BlobCollection<T> m_colInternalTop = new BlobCollection<T>();

        /// <summary>
        /// The MultiheadAttention constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter inner_product_param, with options:
        /// </param>
        public MultiheadAttentionLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.MULTIHEAD_ATTENTION;

            m_nHeads = (int)p.multihead_attention_param.heads;
            m_nEmbed = (int)p.multihead_attention_param.embed;
            m_nBlockSize = (int)p.multihead_attention_param.block_size;
            m_dfAttnDropout = p.multihead_attention_param.attn_dropout;
            m_dfResidDropout = p.multihead_attention_param.resid_dropout;

            log.CHECK_EQ(m_nEmbed % m_nHeads, 0, "The embedding size must be divisible by the number of heads.");

            // Query projection for all heads, but in a batch.
            // input features = m_nHeads
            LayerParameter ipAttnQ = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, p.name + ".c_attnQ", m_phase, p.freeze_learning);
            ipAttnQ.inner_product_param.num_output = (uint)m_nEmbed;
            ipAttnQ.inner_product_param.bias_term = m_param.multihead_attention_param.bias_term;
            if (m_param.multihead_attention_param.weight_init == MultiheadAttentionParameter.WEIGHT_INIT.ENCODER_DECODER)
            {
                ipAttnQ.inner_product_param.weight_filler = new FillerParameter("xavier");
                ipAttnQ.inner_product_param.bias_filler = new FillerParameter("xavier");
            }
            else
            {
                ipAttnQ.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
                ipAttnQ.inner_product_param.bias_filler = new FillerParameter("constant", 0.0);
            }
            ipAttnQ.inner_product_param.axis = 2;
            ipAttnQ.weight_adapter = p.multihead_attention_param.weight_adapter_q;
            ipAttnQ.parameters.Add((m_param.parameters.Count > 0) ? m_param.parameters[0] : new ParamSpec(1.0, 1.0));
            ipAttnQ.parameters.Add((m_param.parameters.Count > 1) ? m_param.parameters[1] : new ParamSpec(1.0, 0.0));
            m_c_attnQ = Layer<T>.Create(cuda, log, convertLayerParam(ipAttnQ, p), null);

            // Key projection for all heads, but in a batch.
            // input features = m_nHeads
            LayerParameter ipAttnK = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, p.name + ".c_attnK", m_phase, p.freeze_learning);
            ipAttnK.inner_product_param.num_output = (uint)m_nEmbed;
            ipAttnK.inner_product_param.bias_term = m_param.multihead_attention_param.bias_term; ;
            if (m_param.multihead_attention_param.weight_init == MultiheadAttentionParameter.WEIGHT_INIT.ENCODER_DECODER)
            {
                ipAttnK.inner_product_param.weight_filler = new FillerParameter("xavier");
                ipAttnK.inner_product_param.bias_filler = new FillerParameter("xavier");
            }
            else
            {
                ipAttnK.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
                ipAttnK.inner_product_param.bias_filler = new FillerParameter("constant", 0.0);
            }
            ipAttnK.inner_product_param.axis = 2;
            ipAttnK.weight_adapter = p.multihead_attention_param.weight_adapter_k;
            ipAttnK.parameters.Add((m_param.parameters.Count > 0) ? m_param.parameters[0] : new ParamSpec(1.0, 1.0));
            ipAttnK.parameters.Add((m_param.parameters.Count > 1) ? m_param.parameters[1] : new ParamSpec(1.0, 0.0));
            m_c_attnK = Layer<T>.Create(cuda, log, convertLayerParam(ipAttnK, p), null);

            // Value projection for all heads, but in a batch.
            // input features = m_nHeads
            LayerParameter ipAttnV = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, p.name + ".c_attnV", m_phase, p.freeze_learning);
            ipAttnV.inner_product_param.num_output = (uint)m_nEmbed;
            ipAttnV.inner_product_param.bias_term = m_param.multihead_attention_param.bias_term; ;
            if (m_param.multihead_attention_param.weight_init == MultiheadAttentionParameter.WEIGHT_INIT.ENCODER_DECODER)
            {
                ipAttnV.inner_product_param.weight_filler = new FillerParameter("xavier");
                ipAttnV.inner_product_param.bias_filler = new FillerParameter("xavier");
            }
            else
            {
                ipAttnV.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02);
                ipAttnV.inner_product_param.bias_filler = new FillerParameter("constant", 0.0);
            }
            ipAttnV.inner_product_param.axis = 2;
            ipAttnV.weight_adapter = p.multihead_attention_param.weight_adapter_v;
            ipAttnV.parameters.Add((m_param.parameters.Count > 0) ? m_param.parameters[0] : new ParamSpec(1.0, 1.0));
            ipAttnV.parameters.Add((m_param.parameters.Count > 1) ? m_param.parameters[1] : new ParamSpec(1.0, 0.0));
            m_c_attnV = Layer<T>.Create(cuda, log, convertLayerParam(ipAttnV, p), null);

            // Output projection.
            // input features = m_nEmbed
            LayerParameter ipProj = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, p.name + ".c_proj", m_phase, p.freeze_learning);
            ipProj.inner_product_param.num_output = (uint)m_nEmbed;
            ipProj.inner_product_param.bias_term = m_param.multihead_attention_param.bias_term;
            if (m_param.multihead_attention_param.weight_init == MultiheadAttentionParameter.WEIGHT_INIT.ENCODER_DECODER)
            {
                ipProj.inner_product_param.weight_filler = new FillerParameter("xavier");
                ipProj.inner_product_param.bias_filler = new FillerParameter("xavier");
            }
            else
            {
                ipProj.inner_product_param.weight_filler = new FillerParameter("gaussian", 0, 0, 0.02 / Math.Sqrt(2 * m_param.multihead_attention_param.layers));
                ipProj.inner_product_param.bias_filler = new FillerParameter("constant", 0.0);
            }
            ipProj.inner_product_param.axis = 2;
            ipProj.weight_adapter = p.multihead_attention_param.weight_adapter_out;
            ipProj.parameters.Add((m_param.parameters.Count > 0) ? m_param.parameters[0] : new ParamSpec(1.0, 1.0));
            ipProj.parameters.Add((m_param.parameters.Count > 1) ? m_param.parameters[1] : new ParamSpec(1.0, 0.0));
            m_c_proj = Layer<T>.Create(cuda, log, convertLayerParam(ipProj, p), null);

            // Regularization
            if (m_dfAttnDropout > 0)
            {
                LayerParameter dropoutAttn = new LayerParameter(LayerParameter.LayerType.DROPOUT, p.name + ".drop.attn", m_phase, p.freeze_learning);
                dropoutAttn.dropout_param.dropout_ratio = m_dfAttnDropout;
                m_attn_dropout = Layer<T>.Create(cuda, log, convertLayerParam(dropoutAttn, p), null);
            }

            if (m_dfResidDropout > 0)
            {
                LayerParameter dropoutResid = new LayerParameter(LayerParameter.LayerType.DROPOUT, p.name + ".drop.resid", m_phase, p.freeze_learning);
                dropoutResid.dropout_param.dropout_ratio = m_dfResidDropout;
                m_resid_dropout = Layer<T>.Create(cuda, log, convertLayerParam(dropoutResid, p), null);
            }

            // Transpose
            LayerParameter transpose = new LayerParameter(LayerParameter.LayerType.TRANSPOSE, p.name + ".trans", m_phase, p.freeze_learning);
            transpose.transpose_param.dim[1] = 2;
            transpose.transpose_param.dim[2] = 1;
            m_transpose = Layer<T>.Create(cuda, log, convertLayerParam(transpose, p), null);

            LayerParameter transposeQ = new LayerParameter(LayerParameter.LayerType.TRANSPOSE, p.name + ".transQ", m_phase, p.freeze_learning);
            transposeQ.transpose_param.dim[2] = 3;
            transposeQ.transpose_param.dim[3] = 2;
            m_transposeQ = Layer<T>.Create(cuda, log, convertLayerParam(transposeQ, p), null);

            // Softmax
            LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX, p.name + ".softmax", m_phase, p.freeze_learning);
            softmax.softmax_param.axis = -1;
            softmax.softmax_param.engine = EngineParameter.Engine.CAFFE;
            m_softmax = Layer<T>.Create(cuda, log, convertLayerParam(softmax, p), null);

            m_blobX0 = createIntraLayerBlob("mth_x0");
            m_blobX1 = createIntraLayerBlob("mth_x0");
            m_blobX2 = createIntraLayerBlob("mth_x0");

            //m_blobX0 = new Blob<T>(cuda, log);
            //m_blobX0.Name = m_param.name + " x0";
            //m_blobX1 = new Blob<T>(cuda, log);
            //m_blobX1.Name = m_param.name + " x1";
            //m_blobX2 = new Blob<T>(cuda, log);
            //m_blobX2.Name = m_param.name + " x2";
            m_blobQ = new Blob<T>(cuda, log);
            m_blobQ.Name = m_param.name + " Q";
            m_blobK = new Blob<T>(cuda, log);
            m_blobK.Name = m_param.name + " K";
            m_blobV = createIntraLayerBlob("mth_v");
            //m_blobV = new Blob<T>(cuda, log);
            //m_blobV.Name = m_param.name + " V";
            m_blobQt = new Blob<T>(cuda, log);
            m_blobQt.Name = m_param.name + " Qt";
            m_blobKt = createIntraLayerBlob("mth_kt");
            //m_blobKt = new Blob<T>(cuda, log);
            //m_blobKt.Name = m_param.name + " Kt";
            m_blobKt1 = new Blob<T>(cuda, log);
            m_blobKt1.Name = m_param.name + " Kt1";
            m_blobVt = new Blob<T>(cuda, log);
            m_blobVt.Name = m_param.name + " Vt";
            m_blobAttA = createIntraLayerBlob("mth_attA");
            //m_blobAttA = new Blob<T>(cuda, log);
            //m_blobAttA.Name = m_param.name + " AttA";
            m_blobAttB = new Blob<T>(cuda, log);
            m_blobAttB.Name = m_param.name + " AttB";
            m_blobWork = new Blob<T>(cuda, log);
            m_blobWork.Name = m_param.name + " Work";
            m_blobY = createIntraLayerBlob("mth_y");
            //m_blobY = new Blob<T>(cuda, log);
            //m_blobY.Name = m_param.name + " Y";

            if (m_param.multihead_attention_param.enable_key_value_cache)
            {
                m_blobKeyCache = new Blob<T>(cuda, log, false);
                m_blobKeyCache.Name = m_param.name + ".KeyCache";
                m_blobValCache = new Blob<T>(cuda, log, false);
                m_blobValCache.Name = m_param.name + ".ValCache";
                m_blobWork2 = createIntraLayerBlob("mth_work2", false);
                m_blobWork3 = createIntraLayerBlob("mth_work3", false);
            }
            
            setup_internal_blobs(m_colInternalBlobs);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_c_attnQ);
            dispose(ref m_c_attnK);
            dispose(ref m_c_attnV);
            dispose(ref m_c_proj);
            dispose(ref m_attn_dropout);
            dispose(ref m_resid_dropout);
            dispose(ref m_transpose);
            dispose(ref m_transposeQ);
            dispose(ref m_softmax);

            dispose(ref m_blobX0);
            dispose(ref m_blobX1);
            dispose(ref m_blobX2);
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
            dispose(ref m_blobY);

            dispose(ref m_blobKeyCache);
            dispose(ref m_blobValCache);
            dispose(ref m_blobWork2);
            dispose(ref m_blobWork3);

            if (m_hCudaAttention != 0)
            {
                m_cuda.FreeAttn(m_hCudaAttention);
                m_hCudaAttention = 0;
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

            col.Add(m_blobX0);
            col.Add(m_blobX1);
            col.Add(m_blobX2);
            col.Add(m_blobQ);
            col.Add(m_blobK);
            col.Add(m_blobV);
            col.Add(m_blobQt);
            col.Add(m_blobKt);
            col.Add(m_blobVt);
            col.Add(m_blobKt1);
            col.Add(m_blobAttA);
            col.Add(m_blobAttB);
            col.Add(m_blobWork);
            col.Add(m_blobY);

            if (m_blobKeyCache != null)
                col.Add(m_blobKeyCache);

            if (m_blobValCache != null)
                col.Add(m_blobValCache);

            if (m_blobWork2 != null)
                col.Add(m_blobWork2);
            if (m_blobWork3 != null)
                col.Add(m_blobWork3);

            col.Add(m_c_attnQ.internal_blobs);
            col.Add(m_c_attnK.internal_blobs);
            col.Add(m_c_attnV.internal_blobs);

            col.Add(m_transpose.internal_blobs);
            col.Add(m_transposeQ.internal_blobs);
            col.Add(m_softmax.internal_blobs);
            if (m_attn_dropout != null)
                col.Add(m_attn_dropout.internal_blobs);
            col.Add(m_c_proj.internal_blobs);
            if (m_resid_dropout != null)
                col.Add(m_resid_dropout.internal_blobs);
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: q, k, v, mask
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 4; }
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
            
            m_c_attnQ.ReInitializeParameters(target);
            m_c_attnK.ReInitializeParameters(target);
            m_c_attnV.ReInitializeParameters(target);
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

        private void reshape(Blob<T> b, params int[] rgShape)
        {
            m_rgShape.Clear();
            m_rgShape.AddRange(rgShape);
            b.Reshape(m_rgShape);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nB = colBottom[0].num;         // batch size
            m_nT = colBottom[0].channels;    // sequence length

            m_nC = colBottom[0].height;      // embedding dim (m_nEmbed)
            m_nSize = m_nC / m_nHeads;

            shareLayerBlob(m_blobX0, colBottom[0].shape());
            m_blobX0.ReshapeLike(colBottom[0]);
            shareLayerBlob(m_blobX1, colBottom[1].shape());
            m_blobX1.ReshapeLike(colBottom[1]);
            shareLayerBlob(m_blobX2, colBottom[2].shape());
            m_blobX2.ReshapeLike(colBottom[2]);

            reshape(m_blobX0, colBottom[0].num, colBottom[0].channels, 1, colBottom[0].count(2));
            reshape(m_blobX1, colBottom[1].num, colBottom[1].channels, 1, colBottom[1].count(2));
            reshape(m_blobX2, colBottom[2].num, colBottom[2].channels, 1, colBottom[2].count(2));

            addInternal(m_blobX0, m_blobQ);
            m_c_attnQ.Setup(m_colInternalBottom, m_colInternalTop);
            addInternal(m_blobX1, m_blobK);
            m_c_attnK.Setup(m_colInternalBottom, m_colInternalTop);
            addInternal(m_blobX2, m_blobV);
            m_c_attnV.Setup(m_colInternalBottom, m_colInternalTop);

            blobs.Add(m_c_attnQ.blobs);
            blobs.Add(m_c_attnK.blobs);
            blobs.Add(m_c_attnV.blobs);

            blobs_adapted.Add(m_c_attnQ.blobs_adapted);
            blobs_adapted.Add(m_c_attnK.blobs_adapted);
            blobs_adapted.Add(m_c_attnV.blobs_adapted);

            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nHeads;
            m_rgShape[2] = m_nT;
            m_rgShape[3] = m_nSize;

            shareLayerBlob(m_blobQ, m_rgShape);
            m_blobQ.Reshape(m_rgShape);

            addInternal(m_blobQ, m_blobQt);
            m_transpose.Setup(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)

            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nHeads;
            m_rgShape[2] = m_nBlockSize;
            m_rgShape[3] = (m_param.multihead_attention_param.enable_key_value_cache) ? 1 : m_nBlockSize;

            shareLayerBlob(m_blobAttA, m_rgShape);
            m_blobAttA.Reshape(m_rgShape);
            shareLayerBlob(m_blobAttB, m_rgShape);
            m_blobAttB.Reshape(m_rgShape);

            addInternal(m_blobAttA, m_blobAttB);
            m_softmax.Setup(m_colInternalBottom, m_colInternalTop);

            if (m_attn_dropout != null)
            {
                addInternal(m_blobAttB, m_blobAttB);
                m_attn_dropout.Setup(m_colInternalBottom, m_colInternalTop);
            }

            if (m_param.multihead_attention_param.enable_rotary_positional_embedding)
            {
                int nGpuID = m_cuda.GetDeviceID();
                m_hRope = m_cuda.CreateRope(m_param.multihead_attention_param.rope_shared_index, nGpuID, colBottom[0].count(), m_nB, (int)m_param.multihead_attention_param.block_size, m_nHeads, m_nC / m_nHeads);
            }

            if (m_param.multihead_attention_param.enable_cuda_scaled_dot_product_attention)
            {
                m_hCudnn = m_cuda.CreateCuDNN();
                m_hCudaAttention = m_cuda.CreateAttn();
                m_cuda.SetAttn(m_hCudnn, m_hCudaAttention, m_cuda.GetDeviceID(), (m_phase == Phase.TRAIN) ? true : false, m_nB, (int)m_param.multihead_attention_param.block_size, m_nHeads, m_nEmbed/m_nHeads, (float)m_param.multihead_attention_param.attn_dropout);
            }

            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nT;
            m_rgShape[2] = m_nC;
            m_rgShape[3] = 1;

            shareLayerBlob(m_blobY, m_rgShape);
            m_blobY.Reshape(m_rgShape);

            addInternal(m_blobY, colTop[0]);
            m_c_proj.Setup(m_colInternalBottom, m_colInternalTop);

            blobs.Add(m_c_proj.blobs);
            blobs_adapted.Add(m_c_proj.blobs_adapted);

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
            m_nB = colBottom[0].num;         // batch size
            m_nT = colBottom[0].channels;    // sequence length

            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nT;
            m_rgShape[2] = m_nHeads;
            m_rgShape[3] = m_nSize;

            m_nC = colBottom[0].height;      // embedding dim (m_nEmbed)
            m_nSize = m_nC / m_nHeads;

            reshape(m_blobX0, colBottom[0].num, colBottom[0].channels, 1, colBottom[0].count(2));
            reshape(m_blobX1, colBottom[1].num, colBottom[1].channels, 1, colBottom[1].count(2));
            reshape(m_blobX2, colBottom[2].num, colBottom[2].channels, 1, colBottom[2].count(2));

            shareLayerBlob(m_blobK, m_rgShape);
            m_blobK.Reshape(m_rgShape);
            shareLayerBlob(m_blobQ, m_rgShape);
            m_blobQ.Reshape(m_rgShape);
            shareLayerBlob(m_blobV, m_rgShape);
            m_blobV.Reshape(m_rgShape);

            shareLayerBlob(m_blobKt1, m_rgShape);
            m_blobKt1.ReshapeLike(m_blobK);
            shareLayerBlob(m_blobKt, m_rgShape);

            addInternal(m_blobK, m_blobKt);
            m_transpose.Reshape(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)
            m_blobKt1.ReshapeLike(m_blobKt);

            shareLayerBlob(m_blobQt, m_rgShape);

            addInternal(m_blobQ, m_blobQt);
            m_transpose.Reshape(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)

            shareLayerBlob(m_blobVt, m_rgShape);

            addInternal(m_blobV, m_blobVt);
            m_transpose.Reshape(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)

            m_rgShape[0] = m_nB;
            m_rgShape[1] = m_nHeads;
            m_rgShape[2] = m_nT;
            m_rgShape[3] = (m_param.multihead_attention_param.enable_key_value_cache) ? 1 : m_nT;

            shareLayerBlob(m_blobAttA, m_rgShape);
            m_blobAttA.Reshape(m_rgShape);
            shareLayerBlob(m_blobAttB, m_rgShape);
            m_blobAttB.Reshape(m_rgShape);

            if (m_param.multihead_attention_param.enable_key_value_cache)
            {
                m_rgShape[0] = m_nB;
                m_rgShape[1] = m_nHeads;
                m_rgShape[2] = m_nT;
                m_rgShape[3] = m_nSize;

                shareLayerBlob(m_blobKeyCache, m_rgShape);
                m_blobKeyCache.Reshape(m_rgShape);
                shareLayerBlob(m_blobValCache, m_rgShape);
                m_blobValCache.Reshape(m_rgShape);

                m_rgShape[2] = 1;
                m_rgShape[3] = 1;
                shareLayerBlob(m_blobWork2, m_rgShape);
                m_blobWork2.Reshape(m_rgShape);
                shareLayerBlob(m_blobWork3, m_rgShape);
                m_blobWork3.Reshape(m_rgShape);
            }

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

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 4) q, k, v, mask
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        /// <param name="colTop">top output blob vector (length 1) y
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed causal self attention.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blobMask = colBottom[3];
            int nPos = m_options.GetPropertyAsInt("position", -1);

            m_blobX0.CopyFrom(colBottom[0], false, false, 0, true);
            m_blobX1.CopyFrom(colBottom[1], false, false, 0, true);
            m_blobX2.CopyFrom(colBottom[2], false, false, 0, true);

            // Calculate query, for all heads in batch and move head forward to be the batch dim.
            // q  = self.c_attnQ(x1)
            addInternal(m_blobX0, m_blobQ);
            m_c_attnQ.Forward(m_colInternalBottom, m_colInternalTop);

            // Calculate key, for all heads in batch and move head forward to be the batch dim.
            // k  = self.c_attnK(x2)
            addInternal(m_blobX1, m_blobK);
            m_c_attnK.Forward(m_colInternalBottom, m_colInternalTop);

            // Calculate value, for all heads in batch and move head forward to be the batch dim.
            // v  = self.c_attnK(x3)
            addInternal(m_blobX2, m_blobV);
            m_c_attnV.Forward(m_colInternalBottom, m_colInternalTop);

            // Transpose query, key and values along axes 1 & 2
            // k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            // q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            // v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            m_blobQ.Reshape(m_nB, m_nT, m_nHeads, m_nSize);
            m_blobK.Reshape(m_nB, m_nT, m_nHeads, m_nSize);
            m_blobV.Reshape(m_nB, m_nT, m_nHeads, m_nSize);

            // When using rope, apply the rotary positional embedding.
            if (m_hRope != 0)
            {
                int nFreqOffset = -1;
                if (nPos > 0)
                    nFreqOffset = nPos * m_nSize / 2;

                m_cuda.RopeForward(m_hRope, m_blobQ.count(), m_blobQ.gpu_data, m_blobQ.mutable_gpu_data, nFreqOffset);
                m_cuda.RopeForward(m_hRope, m_blobK.count(), m_blobK.gpu_data, m_blobK.mutable_gpu_data, nFreqOffset);
            }

            addInternal(m_blobQ, m_blobQt);
            m_transpose.Forward(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)
            addInternal(m_blobK, m_blobKt);
            m_transpose.Forward(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)
            addInternal(m_blobV, m_blobVt);
            m_transpose.Forward(m_colInternalBottom, m_colInternalTop); // (B, nh, T, hs)

            // Perform Self Attention forward pass
            if (m_param.multihead_attention_param.enable_cuda_scaled_dot_product_attention)
            {
                m_blobWork.Reshape(m_blobVt.num, m_blobVt.channels, m_blobVt.height, m_blobVt.width);
                m_cuda.AttnScaledDotProductForward(m_hCudnn, m_hCudaAttention, colBottom[0].channels, m_blobQt.gpu_data, m_blobKt.gpu_data, m_blobVt.gpu_data, blobMask.gpu_data, m_blobWork.mutable_gpu_data);
            }
            else
            {
                // Multiply query and key(T) matrices and scale
                // att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                addInternal(m_blobKt, m_blobKt1);
                m_transposeQ.Forward(m_colInternalBottom, m_colInternalTop);

                double dfScale = 1.0 / Math.Sqrt(m_nSize);

                if (m_param.multihead_attention_param.enable_key_value_cache)
                {
                    m_blobKeyCache.Reshape(nPos + 1, m_nHeads, m_nSize, 1);
                    m_blobValCache.Reshape(nPos + 1, m_nHeads, 1, m_nSize);

                    m_cuda.copy(m_blobKt1.count(), m_blobKt1.gpu_data, m_blobKeyCache.mutable_gpu_data, 0, nPos * m_blobKt1.count());
                    m_cuda.copy(m_blobVt.count(), m_blobVt.gpu_data, m_blobValCache.mutable_gpu_data, 0, nPos * m_blobVt.count());

                    m_blobAttA.Reshape(1, m_nHeads, 1, nPos + 1);

                    for (int i=0; i<nPos+1; i++)
                    {
                        m_cuda.copy(m_blobKt1.count(), m_blobKeyCache.gpu_data, m_blobKt1.mutable_gpu_data, i * m_blobKt1.count(), 0);
                        m_blobWork2.MatMul(m_blobQt, m_blobKt1, true);
                        m_blobWork2.scale_data(dfScale);
                        m_cuda.channel_copy(m_blobWork2.count(), 1, m_nHeads, nPos + 1, 1, i, m_blobAttA.mutable_gpu_data, m_blobWork2.gpu_data, DIR.BWD);
                    }
                }
                else
                {
                    m_blobAttA.MatMul(m_blobQt, m_blobKt1);
                    m_blobAttA.scale_data(dfScale);
                    // Apply mask to attention matrix
                    // att = att.masked_fill(self.bias[:,:,:T,:T] == 0, -1e+29)
                    m_cuda.mask(m_blobAttA.count(), blobMask.count(), convert(0.0), convert(m_dfIgnoreVal), m_blobAttA.gpu_data, blobMask.gpu_data, m_blobAttA.mutable_gpu_data); // all masked items set to -inf.
                }

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

                // Multiply attention matrix with values
                // y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
                if (m_param.multihead_attention_param.enable_key_value_cache)
                {
                    m_blobWork.Reshape(1, m_nHeads, 1, m_nSize);
                    m_blobWork3.Reshape(1, m_nHeads, 1, 1);
                    m_blobWork2.Reshape(1, m_nHeads, 1, m_nSize);

                    for (int i = 0; i < nPos + 1; i++)
                    {
                        m_cuda.copy(m_blobVt.count(), m_blobValCache.gpu_data, m_blobVt.mutable_gpu_data, i * m_blobVt.count(), 0);
                        m_cuda.channel_copy(m_blobWork3.count(), 1, m_nHeads, nPos + 1, 1, i, m_blobAttB.gpu_data, m_blobWork3.mutable_gpu_data, DIR.FWD);

                        m_blobWork2.MatMul(m_blobWork3, m_blobVt, true);

                        if (i == 0)
                            m_cuda.copy(m_blobWork.count(), m_blobWork2.gpu_data, m_blobWork.mutable_gpu_data);
                        else
                            m_cuda.add(m_blobWork.count(), m_blobWork2.gpu_data, m_blobWork.gpu_data, m_blobWork.mutable_gpu_data);
                    }
                }
                else
                {
                    m_blobWork.MatMul(m_blobAttB, m_blobVt, true);
                }
            }

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
                // y = self.w_0(concat_output)
                addInternal(m_blobY, colTop[0]);
                m_c_proj.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Reassemble all head outputs side by side.
                // y = y.transpose(1, 2).contiguous().view(B, T, C) 
                addInternal(m_blobWork, m_blobY);
                m_transpose.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Perform Self Attention backward pass
                if (m_param.multihead_attention_param.enable_cuda_scaled_dot_product_attention)
                {
                    m_blobY.CopyFrom(m_blobWork, true, true);
                    m_cuda.AttnScaledDotProductBackward(m_hCudnn, m_hCudaAttention, m_blobQt.gpu_data, m_blobQt.mutable_gpu_diff, m_blobKt.gpu_data, m_blobKt.mutable_gpu_diff, m_blobVt.gpu_data, m_blobVt.mutable_gpu_diff, 0, m_blobY.gpu_data, m_blobY.gpu_diff);
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
                    m_blobY.MatMulGrad(m_blobAttB, m_blobVt, 1);

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
                    m_blobAttA.MatMulGrad(m_blobQt, m_blobKt1, dfScale);

                    // Transpose Kt1 back to Kt
                    addInternal(m_blobKt, m_blobKt1);
                    m_transposeQ.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);
                }

                // Transpose query, key and values along axes 1 & 2
                // k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
                // q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
                // v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
                m_blobQ.Reshape(m_nB, m_nT, m_nHeads, m_nSize);
                m_blobK.Reshape(m_nB, m_nT, m_nHeads, m_nSize);
                m_blobV.Reshape(m_nB, m_nT, m_nHeads, m_nSize);

                addInternal(m_blobQ, m_blobQt);
                m_transpose.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom); // (B, nh, T, hs)
                addInternal(m_blobK, m_blobKt);
                m_transpose.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom); // (B, nh, T, hs)
                addInternal(m_blobV, m_blobVt);
                m_transpose.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom); // (B, nh, T, hs)

                // When using rope, handle the rotary positional embedding.
                if (m_hRope != 0)
                {
                    m_cuda.RopeBackward(m_hRope, m_blobQ.count(), m_blobQ.gpu_data, m_blobQ.gpu_diff, m_blobQ.mutable_gpu_diff);
                    m_cuda.RopeBackward(m_hRope, m_blobK.count(), m_blobK.gpu_data, m_blobK.gpu_diff, m_blobK.mutable_gpu_diff);
                }

                m_rgShape[0] = m_nB;
                m_rgShape[1] = m_nT;
                m_rgShape[2] = m_nHeads * m_nSize;
                m_rgShape.RemoveAt(3);

                m_blobQ.Reshape(m_rgShape);
                m_blobK.Reshape(m_rgShape);
                m_blobV.Reshape(m_rgShape);

                m_rgShape.Add(1);

                // Calculate query for all heads in batch and move head forward to be the batch dim.
                // q = self.c_attnQ(x1)
                addInternal(m_blobX0, m_blobQ);
                m_c_attnQ.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Calculate query for all heads in batch and move head forward to be the batch dim.
                // k = self.c_attnK(x2)
                addInternal(m_blobX1, m_blobK);
                m_c_attnK.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                // Calculate query for all heads in batch and move head forward to be the batch dim.
                // v = self.c_attnV(x3)
                addInternal(m_blobX2, m_blobV);
                m_c_attnV.Backward(m_colInternalTop, rgbPropagate, m_colInternalBottom);

                if (colBottom[0].gpu_diff == colBottom[1].gpu_diff && colBottom[0].gpu_diff == colBottom[2].gpu_diff)
                {
                    m_cuda.add(m_blobX0.count(), m_blobX0.gpu_diff, m_blobX1.gpu_diff, m_blobX2.gpu_diff, colBottom[0].mutable_gpu_diff);
                }
                else if (colBottom[1].gpu_diff == colBottom[2].gpu_diff)
                {
                    colBottom[0].CopyFrom(m_blobX0, true);
                    m_cuda.add(m_blobX1.count(), m_blobX1.gpu_diff, m_blobX2.gpu_diff, colBottom[1].mutable_gpu_diff);
                }
                else
                {
                    colBottom[0].CopyFrom(m_blobX0, true);
                    colBottom[1].CopyFrom(m_blobX1, true);
                    colBottom[2].CopyFrom(m_blobX2, true);
                }
            }
        }
    }
}
