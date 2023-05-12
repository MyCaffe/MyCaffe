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
    /// The MultiHeadAttentionInterpLayer implements the Multi-head Attention Interpretive Layer
    /// </summary>
    /// <remarks>
    /// The Multi-Headed Attention layer learns long-term relationships across different time-steps.  This version of 
    /// the layer is modified to enhance explainability.  On this modification, the 'values' signal is shared across
    /// all heads - the additive aggregation is employed across all heads.  According to the paper by Lim et al., each
    /// head can learn different temporal patterns, while attending to a common set of input features which can be
    /// interpreted as a simple ensemble over attention weights into a combined matrix, which compared to the  original
    /// multi-head attention matrix, yields an increased representation capacity in an efficient way.
    /// 
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L443) by Playtika Research, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class MultiHeadAttentionInterpLayer<T> : Layer<T>
    {
        List<int> m_rgShapeQ;
        List<int> m_rgShapeK;
        List<int> m_rgShapeV;
        List<int> m_rgShapeMask;
        int m_nNumHeads;
        int m_nDModel;
        int m_nAllHeadsDim;
        int m_nNumFut = 0;
        int m_nNumHist = 0;
        int m_nBlocks = 0;
        double m_dfScale;
        Layer<T> m_ipQLayer;
        Layer<T> m_ipKLayer;
        Layer<T> m_ipVLayer;
        Layer<T> m_transpose;
        Layer<T> m_softmax;
        Layer<T> m_ipOutLayer;
        Blob<T> m_blobQ;
        Blob<T> m_blobK;
        Blob<T> m_blobV;
        Blob<T> m_blobIpQ;
        Blob<T> m_blobIpK;
        Blob<T> m_blobIpV;
        Blob<T> m_blobMask;
        Blob<T> m_blobIpVfull;
        Blob<T> m_blobIpQt;
        Blob<T> m_blobIpKt;
        Blob<T> m_blobIpKt1;
        Blob<T> m_blobIpVt;
        Blob<T> m_blobAttnScores1;
        Blob<T> m_blobAttnScoresAllHeads;
        Blob<T> m_blobAttnOutputAllHeads;
        Blob<T> m_blobWork;
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();
        List<int> m_rgShape = new List<int>(4);

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public MultiHeadAttentionInterpLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.MULTIHEAD_ATTENTION_INTERP;

            m_blobQ = new Blob<T>(cuda, log);
            m_blobQ.Name = p.name + ".q";
            m_blobK = new Blob<T>(cuda, log);
            m_blobQ.Name = p.name + ".k";
            m_blobV = new Blob<T>(cuda, log);
            m_blobV.Name = p.name + ".v";
            m_blobIpQ = new Blob<T>(cuda, log);
            m_blobIpQ.Name = p.name + ".ipq";
            m_blobIpK = new Blob<T>(cuda, log);
            m_blobIpK.Name = p.name + ".ipk";
            m_blobIpV = new Blob<T>(cuda, log);
            m_blobIpV.Name = p.name + ".ipv";
            m_blobMask = new Blob<T>(cuda, log, false);
            m_blobMask.Name = p.name + ".mask";
            m_blobIpVfull = new Blob<T>(cuda, log);
            m_blobIpVfull.Name = p.name + ".ipvfull";
            m_blobIpQt = new Blob<T>(cuda, log);
            m_blobIpQt.Name = p.name + ".ipqt";
            m_blobIpKt = new Blob<T>(cuda, log);
            m_blobIpKt.Name = p.name + ".ipkt";
            m_blobIpKt1 = new Blob<T>(cuda, log);
            m_blobIpKt1.Name = p.name + ".ipkt1";
            m_blobIpVt = new Blob<T>(cuda, log);
            m_blobIpVt.Name = p.name + ".ipvt";
            m_blobAttnScores1 = new Blob<T>(cuda, log);
            m_blobAttnScores1.Name = p.name + ".attn_scores";
            m_blobAttnScoresAllHeads = new Blob<T>(cuda, log);
            m_blobAttnScoresAllHeads.Name = p.name + ".attn_scr_allhd";
            m_blobAttnOutputAllHeads = new Blob<T>(cuda, log);
            m_blobAttnOutputAllHeads.Name = p.name + ".attn_out_allhd";
            m_blobWork = new Blob<T>(cuda, log);
            m_blobWork.Name = p.name + ".work";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobQ);
            dispose(ref m_blobK);
            dispose(ref m_blobV);
            dispose(ref m_blobIpQ);
            dispose(ref m_blobIpK);
            dispose(ref m_blobIpV);
            dispose(ref m_blobMask);
            dispose(ref m_blobIpVfull);
            dispose(ref m_blobIpQt);
            dispose(ref m_blobIpKt);
            dispose(ref m_blobIpKt1);
            dispose(ref m_blobIpVt);
            dispose(ref m_blobAttnScores1);
            dispose(ref m_blobAttnScoresAllHeads);
            dispose(ref m_blobAttnOutputAllHeads);
            dispose(ref m_blobWork);

            dispose(ref m_ipQLayer);
            dispose(ref m_ipKLayer);
            dispose(ref m_ipVLayer);
            dispose(ref m_transpose);
            dispose(ref m_softmax);
            dispose(ref m_ipOutLayer);
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobQ);
            col.Add(m_blobK);
            col.Add(m_blobV);
            col.Add(m_blobIpQ);
            col.Add(m_blobIpK);
            col.Add(m_blobIpV);
            col.Add(m_blobMask);
            col.Add(m_blobIpVfull);
            col.Add(m_blobIpQt);
            col.Add(m_blobIpKt);
            col.Add(m_blobIpKt1);
            col.Add(m_blobIpVt);
            col.Add(m_blobAttnScores1);
            col.Add(m_blobAttnScoresAllHeads);
            col.Add(m_blobAttnOutputAllHeads);
            col.Add(m_blobWork);
        }

        /// <summary>
        /// Returns the min number of required bottom (input) Blobs: input -> q,k,v, mask is generated
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the max number of required bottom (input) Blobs: q, k, v, mask
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 4; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: y, attn_out, attn_scores
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 3; }
        }

        private void addBtmTop(Blob<T> btm, Blob<T> top)
        {
            m_colBtm.Clear();
            m_colBtm.Add(btm);
            m_colTop.Clear();
            m_colTop.Add(top);
        }

        private void reshapeRepeat(Blob<T> b, List<int> rgShape, int nRepeat)
        {
            m_rgShape.Clear();
            m_rgShape.AddRange(rgShape);
            m_rgShape[3] *= nRepeat;
            b.Reshape(m_rgShape);
        }

        private void reshapeFwd(Blob<T> b, int nNumHeads, List<int> rgShape = null)
        {
            m_rgShape.Clear();

            if (rgShape == null)
                rgShape = b.shape();

            m_rgShape.Add(rgShape[0]);
            m_rgShape.Add(rgShape[1]);
            m_rgShape.Add(nNumHeads);
            m_rgShape.Add(rgShape[2] / nNumHeads);
            b.Reshape(m_rgShape);
        }

        private void reshapeBwd(Blob<T> b, int nNumHeads, List<int> rgShape = null)
        {
            m_rgShape.Clear();

            if (rgShape == null)
                rgShape = b.shape();

            m_rgShape.Add(rgShape[0]);
            m_rgShape.Add(rgShape[1]);
            m_rgShape.Add(rgShape[2] * rgShape[3]);
            b.Reshape(m_rgShape);
        }

        private void reshapeSansHead(Blob<T> b, List<int> rgShape)
        {
            m_rgShape.Clear();
            m_rgShape.AddRange(rgShape);
            m_rgShape.RemoveAt(1);
            b.Reshape(m_rgShape);
        }

        private void calculateChannelMeanAcrossChannelsFwd(Blob<T> bBtm, Blob<T> bTop)
        {
            int nN = bBtm.num;
            int nC = bBtm.channels;
            int nSpatialDim = bBtm.count(2);
            int nSpatialDimDst = bTop.count(1);

            m_log.CHECK_EQ(bTop.num, nN, "Both src and dst must have same 'num'.");
            m_log.CHECK_EQ(nSpatialDim, bTop.count(1), "Both src and dst must have the same spatial dim.");

            bTop.SetData(0);
            m_blobWork.ReshapeLike(bTop);

            for (int i = 0; i < nC; i++)
            {
                m_cuda.channel_copy(m_blobWork.count(), nN, 1, nC, nSpatialDim, i, bBtm.gpu_data, m_blobWork.gpu_data, DIR.FWD);
                m_cuda.add(m_blobWork.count(), m_blobWork.gpu_data, bTop.gpu_data, bTop.mutable_gpu_data);
            }

            bTop.scale_data(1.0 / nC);
        }

        private void calculateChannelMeanAcrossChannelsBwd(Blob<T> bBtm, Blob<T> bTop)
        {
            int nN = bBtm.num;
            int nC = bBtm.channels;
            int nSpatialDim = bBtm.count(2);

            m_log.CHECK_EQ(bTop.num, nN, "Both src and dst must have same 'num'.");
            m_log.CHECK_EQ(nSpatialDim, bTop.count(1), "Both src and dst must have the same spatial dim.");

            bBtm.SetDiff(0);

            for (int i = 0; i < nC; i++)
            {
                m_cuda.channel_copy(bTop.count(), nN, 1, nC, nSpatialDim, i, bBtm.gpu_diff, bTop.gpu_diff, DIR.BWD);
            }

            bBtm.scale_diff(1.0 / nC);
        }

        private void generate_mask(Blob<T> mask)
        {
            m_rgShape.Clear();
            m_rgShape.Add(m_nNumFut);
            m_rgShape.Add(m_nNumFut + m_nNumHist);
            mask.Reshape(m_rgShape);

            int nRow = m_nNumFut + m_nNumHist;
            int nOutSeqLen = m_nNumFut; //- m_nTargetWindowStartIdx;  not used
            float[] rgData = new float[mask.count()];

            for (int i = 0; i < m_nNumFut; i++)
            {
                for (int j = 0; j < m_nNumHist + nOutSeqLen; j++)
                {
                    int nIdx = i * nRow + j;

                    if (j > m_nNumHist && j-m_nNumHist > i)
                        rgData[nIdx] = 1;
                }
            }

            mask.mutable_cpu_data = convert(rgData);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_param.multihead_attention_interp_param.enable_self_attention)
                m_log.CHECK_EQ(colBottom.Count, 1, "When using self-attention, there should only be one bottom.");
            else
                m_log.CHECK_EQ(colBottom.Count, 3, "When not using self-attention, there should be three bottom values: q, k, v");

            m_nNumHeads = (int)m_param.multihead_attention_interp_param.num_heads;
            m_nDModel = (int)m_param.multihead_attention_interp_param.embed_dim;
            m_nAllHeadsDim = m_nNumHeads * m_nDModel;
            m_dfScale = 1.0 / Math.Sqrt(m_nDModel);

            m_log.CHECK(colBottom.Count == 1 || colBottom.Count == 4, "The bottom count must be 1 (input ->q,k,v, mask generated) or 4 for q,k,q,mask");

            m_nNumFut = (int)m_param.multihead_attention_interp_param.num_future_steps;
            m_log.CHECK_GT(m_nNumFut, 0, "The number of future steps must be greater than zero.");
            m_nNumHist = (int)m_param.multihead_attention_interp_param.num_historical_steps;
            m_log.CHECK_GT(m_nNumHist, 0, "The number of historical steps must be greater than zero.");
            m_log.CHECK_EQ(m_nNumFut + m_nNumHist, colBottom[0].channels, "The number of future + historical steps must equal the bottom(0).channels.");
            m_log.CHECK_EQ(m_nNumHist % m_nNumFut, 0, "The historical steps must be a multiple of the future steps!  For example, historical steps = 90 and future steps = 30.");
            m_nBlocks = (m_nNumHist + m_nNumFut) / m_nNumFut;

            if (colBottom.Count == 1)
                generate_mask(m_blobMask);
            else
                m_blobMask.ShareData(colBottom[3]);

            if (m_ipQLayer == null)
            {
                LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, m_param.name + ".ipQ");
                ip1.inner_product_param.num_output = (uint)m_nAllHeadsDim;
                ip1.inner_product_param.axis = 2;
                ip1.inner_product_param.bias_term = true;
                ip1.inner_product_param.enable_noise = m_param.multihead_attention_interp_param.enable_noise;
                ip1.inner_product_param.sigma_init = m_param.multihead_attention_interp_param.sigma_init;
                ip1.inner_product_param.bias_filler = m_param.multihead_attention_interp_param.bias_filler;
                ip1.inner_product_param.weight_filler = m_param.multihead_attention_interp_param.weight_filler;
                ip1.inner_product_param.bias_grad_scale = 1000000.0; // helps improve bias gradient accuracy.

                m_ipQLayer = Layer<T>.Create(m_cuda, m_log, ip1, null);

                if (colBottom.Count == 1)
                {
                    m_rgShape.Clear();
                    m_rgShape.Add(colBottom[0].num);
                    m_rgShape.Add(m_nNumFut);
                    m_rgShape.Add(colBottom[0].count(2));
                    m_blobQ.Reshape(m_rgShape);
                }
                else
                {
                    m_blobQ.ReshapeLike(colBottom[0]);
                }

                addBtmTop(m_blobQ, m_blobIpQ);
                m_ipQLayer.Setup(m_colBtm, m_colTop);
                blobs.Add(m_ipQLayer.blobs);
            }

            if (m_ipKLayer == null)
            {
                LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, m_param.name + ".ipK");
                ip1.inner_product_param.num_output = (uint)m_nAllHeadsDim;
                ip1.inner_product_param.axis = 2;
                ip1.inner_product_param.bias_term = true;
                ip1.inner_product_param.enable_noise = m_param.multihead_attention_interp_param.enable_noise;
                ip1.inner_product_param.sigma_init = m_param.multihead_attention_interp_param.sigma_init;
                ip1.inner_product_param.bias_filler = m_param.multihead_attention_interp_param.bias_filler;
                ip1.inner_product_param.weight_filler = m_param.multihead_attention_interp_param.weight_filler;
                ip1.inner_product_param.bias_grad_scale = 1000000.0; // helps improve bias gradient accuracy.

                m_ipKLayer = Layer<T>.Create(m_cuda, m_log, ip1, null);
                m_blobK.ReshapeLike((colBottom.Count == 1) ? colBottom[0] : colBottom[1]);

                addBtmTop(m_blobK, m_blobIpK);
                m_ipKLayer.Setup(m_colBtm, m_colTop);
                blobs.Add(m_ipKLayer.blobs);
            }

            if (m_ipVLayer == null)
            {
                LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, m_param.name + ".ipV");
                ip1.inner_product_param.num_output = (uint)m_param.multihead_attention_interp_param.embed_dim;
                ip1.inner_product_param.axis = 2;
                ip1.inner_product_param.bias_term = true;
                ip1.inner_product_param.enable_noise = m_param.multihead_attention_interp_param.enable_noise;
                ip1.inner_product_param.sigma_init = m_param.multihead_attention_interp_param.sigma_init;
                ip1.inner_product_param.bias_filler = m_param.multihead_attention_interp_param.bias_filler;
                ip1.inner_product_param.weight_filler = m_param.multihead_attention_interp_param.weight_filler;

                m_ipVLayer = Layer<T>.Create(m_cuda, m_log, ip1, null);
                m_blobV.ReshapeLike((colBottom.Count == 1) ? colBottom[0] : colBottom[1]);

                addBtmTop(m_blobV, m_blobIpV);
                m_ipVLayer.Setup(m_colBtm, m_colTop);
                blobs.Add(m_ipVLayer.blobs);
            }

            // Transpose
            if (m_transpose == null)
            {
                // Reshape q, k, v projections to the following sizes
                // queries tensor - q: [num_samples x num_future_steps x state_size]
                // keys tensor    - k: [num_samples x num_total_steps x state_size]
                // values tensor  - v: [num_samples x num_total_steps x state_size]
                reshapeFwd(m_blobIpQ, m_nNumHeads);
                reshapeFwd(m_blobIpK, m_nNumHeads);
                reshapeFwd(m_blobIpV, m_nNumHeads);
                reshapeRepeat(m_blobIpVfull, m_blobIpV.shape(), m_nNumHeads);

                LayerParameter transpose = new LayerParameter(LayerParameter.LayerType.TRANSPOSE, m_param.name + ".trans");
                transpose.transpose_param.dim[1] = 2;
                transpose.transpose_param.dim[2] = 1;
                m_transpose = Layer<T>.Create(m_cuda, m_log, convertLayerParam(transpose, m_param), null);

                addBtmTop(m_blobIpQ, m_blobIpQt);
                m_transpose.Setup(m_colBtm, m_colTop);
                addBtmTop(m_blobIpK, m_blobIpKt);
                m_transpose.Setup(m_colBtm, m_colTop);
                addBtmTop(m_blobIpVfull, m_blobIpVt);
                m_transpose.Setup(m_colBtm, m_colTop);
            }

            // Transpose
            if (m_blobIpKt1.count() == 0)
            { 
                List<int> rgShape = Utility.Clone<int>(m_blobIpKt.shape());
                int nTemp = rgShape[2];
                rgShape[2] = rgShape[3];
                rgShape[3] = nTemp;
                m_blobIpKt1.Reshape(rgShape);

                m_blobAttnScores1.MatMul(m_blobIpQt, m_blobIpKt1, true);
            }

            // Softmax
            if (m_softmax == null)
            {
                LayerParameter softmax = new LayerParameter(LayerParameter.LayerType.SOFTMAX, m_param.name + ".softmax");
                softmax.softmax_param.axis = -1;
                softmax.softmax_param.engine = EngineParameter.Engine.CUDNN;
                m_softmax = Layer<T>.Create(m_cuda, m_log, convertLayerParam(softmax, m_param), null);

                addBtmTop(m_blobAttnScores1, m_blobAttnScoresAllHeads);
                m_softmax.Setup(m_colBtm, m_colTop);

                m_blobAttnOutputAllHeads.MatMul(m_blobAttnScoresAllHeads, m_blobIpVt, true);
            }

            if (m_ipOutLayer == null)
            {
                LayerParameter ip1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, m_param.name + ".ipOut");
                ip1.inner_product_param.num_output = (uint)m_param.multihead_attention_interp_param.embed_dim;
                ip1.inner_product_param.axis = 2;
                ip1.inner_product_param.bias_term = true;
                ip1.inner_product_param.enable_noise = m_param.multihead_attention_interp_param.enable_noise;
                ip1.inner_product_param.sigma_init = m_param.multihead_attention_interp_param.sigma_init;
                ip1.inner_product_param.bias_filler = m_param.multihead_attention_interp_param.bias_filler;
                ip1.inner_product_param.weight_filler = m_param.multihead_attention_interp_param.weight_filler;

                m_ipOutLayer = Layer<T>.Create(m_cuda, m_log, ip1, null);

                reshapeSansHead(colTop[1], m_blobAttnOutputAllHeads.shape());
                reshapeSansHead(colTop[2], m_blobAttnScoresAllHeads.shape());

                addBtmTop(colTop[1], colTop[0]);
                m_ipOutLayer.Setup(m_colBtm, m_colTop);
                blobs.Add(m_ipOutLayer.blobs);
            }
        }

        /// <summary>
        /// Determines if a reshape is needed or not.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        /// <param name="bReset">Forces a reshape when true.</param>
        /// <returns>True is returned when a reshape is needed.</returns>
        protected override bool reshapeNeeded(BlobCollection<T> colBottom, BlobCollection<T> colTop, bool bReset = false)
        {
            if (bReset)
                return true;

            bool bShapeQDirty = m_rgShapeQ == null || !colBottom[0].CompareShape(m_rgShapeQ);
            bool bShapeKDirty = (colBottom.Count == 1) ? bShapeQDirty : m_rgShapeK == null || !colBottom[1].CompareShape(m_rgShapeK);
            bool bShapeVDirty = (colBottom.Count == 1) ? bShapeQDirty : m_rgShapeV == null || !colBottom[2].CompareShape(m_rgShapeV);
            bool bShapeMaskDirty = false;

            m_rgShapeQ = Utility.Clone<int>(colBottom[0].shape());
            m_rgShapeK = m_rgShapeQ;
            m_rgShapeV = m_rgShapeQ;

            if (colBottom.Count > 1)
                m_rgShapeK = Utility.Clone<int>(colBottom[1].shape());
            if (colBottom.Count > 2)
                m_rgShapeV = Utility.Clone<int>(colBottom[2].shape());

            if (colBottom.Count > 3)
            {
                bShapeMaskDirty = m_rgShapeMask == null || !colBottom[3].CompareShape(m_rgShapeMask);
                m_rgShapeMask = Utility.Clone<int>(colBottom[3].shape());
            }

            if (bShapeQDirty || bShapeKDirty || bShapeVDirty || bShapeMaskDirty)
                return true;

            return false;
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!reshapeNeeded(colBottom, colTop))
                return;

            if (colBottom.Count == 1)
            {
                m_rgShape.Clear();
                m_rgShape.Add(colBottom[0].num);
                m_rgShape.Add(m_nNumFut);
                m_rgShape.Add(colBottom[0].count(2));
                m_blobQ.Reshape(m_rgShape);
            }
            else
            {
                m_blobQ.ReshapeLike(colBottom[0]);
            }

            addBtmTop(m_blobQ, m_blobIpQ);
            m_ipQLayer.Reshape(m_colBtm, m_colTop);

            m_blobK.ReshapeLike((colBottom.Count == 1) ? colBottom[0] : colBottom[1]);
            addBtmTop(m_blobK, m_blobIpK);
            m_ipKLayer.Reshape(m_colBtm, m_colTop);

            m_blobV.ReshapeLike((colBottom.Count == 1) ? colBottom[0] : colBottom[2]);
            addBtmTop(m_blobV, m_blobIpV);
            m_ipVLayer.Reshape(m_colBtm, m_colTop);

            // Reshape q, k, v projections to the following sizes
            // queries tensor - q: [num_samples x num_future_steps x state_size]
            // keys tensor    - k: [num_samples x num_total_steps x state_size]
            // values tensor  - v: [num_samples x num_total_steps x state_size]
            reshapeFwd(m_blobIpQ, m_nNumHeads);
            reshapeFwd(m_blobIpK, m_nNumHeads);
            reshapeFwd(m_blobIpV, m_nNumHeads);
            reshapeRepeat(m_blobIpVfull, m_blobIpV.shape(), m_nNumHeads);

            addBtmTop(m_blobIpQ, m_blobIpQt);
            m_transpose.Reshape(m_colBtm, m_colTop);

            addBtmTop(m_blobIpK, m_blobIpKt);
            m_transpose.Reshape(m_colBtm, m_colTop);

            addBtmTop(m_blobIpVfull, m_blobIpVt);
            m_transpose.Reshape(m_colBtm, m_colTop);

            List<int> rgShape = Utility.Clone<int>(m_blobIpKt.shape());
            int nTemp = rgShape[2];
            rgShape[2] = rgShape[3];
            rgShape[3] = nTemp;
            m_blobIpKt1.Reshape(rgShape);

            m_blobAttnScores1.MatMul(m_blobIpQt, m_blobIpKt1, true);

            addBtmTop(m_blobAttnScores1, m_blobAttnScoresAllHeads);
            m_softmax.Reshape(m_colBtm, m_colTop);

            colTop[1].MatMul(m_blobAttnScoresAllHeads, m_blobIpVt, true);
            m_blobWork.ReshapeLike(colTop[1]);

            reshapeSansHead(colTop[1], m_blobAttnOutputAllHeads.shape());
            reshapeSansHead(colTop[2], m_blobAttnScoresAllHeads.shape());

            addBtmTop(colTop[1], colTop[0]);
            m_ipOutLayer.Reshape(m_colBtm, m_colTop);
        }

        private void copy_to_q_fwd(int nCount, Blob<T> bBtm, Blob<T> bTop)
        {
            if (nCount == 1)
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

        private void copy_to_q_bwd(int nCount, Blob<T> bBtm, Blob<T> bTop)
        {
            if (nCount == 1)
            {
                // Copy just the future items to the top, so if future = 30,
                // with input shape is btm(256,120,64) just the last (256,30,64) are copied to top 
                int nOuterNum = bBtm.num;
                int nChannels = m_nBlocks;
                int nInnerNum = (bBtm.channels / m_nBlocks) * bBtm.count(2);
                m_cuda.channel_add(bTop.count(), nOuterNum, nChannels, m_nBlocks, nInnerNum, m_nBlocks-1, bBtm.mutable_gpu_diff, bTop.gpu_diff, DIR.BWD);
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
        ///     the numeric inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector)
        ///  -# @f$ (N \times C \times H \times W size) @f$
        ///     the computed outputs @f$ y @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Calculate q, k, v projections
            copy_to_q_fwd(colBottom.Count, colBottom[0], m_blobQ);

            addBtmTop(m_blobQ, m_blobIpQ);
            m_ipQLayer.Forward(m_colBtm, m_colTop);

            m_blobK.CopyFrom((colBottom.Count == 1) ? colBottom[0] : colBottom[1]);

            addBtmTop(m_blobK, m_blobIpK);
            m_ipKLayer.Forward(m_colBtm, m_colTop);

            m_blobV.CopyFrom((colBottom.Count == 1) ? colBottom[0] : colBottom[2]);

            addBtmTop(m_blobV, m_blobIpV);
            m_ipVLayer.Forward(m_colBtm, m_colTop);

            // Reshape q, k, v projections to the following sizes
            // queries tensor - q: [num_samples x num_future_steps x num_heads x state_size]
            // keys tensor    - k: [num_samples x num_total_steps x num_heads x state_size]
            // values tensor  - v: [num_samples x num_total_steps x num_heads x state_size]
            reshapeFwd(m_blobIpQ, m_nNumHeads);
            reshapeFwd(m_blobIpK, m_nNumHeads);
            reshapeFwd(m_blobIpV, m_nNumHeads);
            reshapeRepeat(m_blobIpVfull, m_blobIpV.shape(), m_nNumHeads);

            // repeat blobIpV width to V full.
            int nInnerNum = m_blobIpV.count(2);
            for (int i = 0; i < m_nNumHeads; i++)
            {
                m_cuda.channel_copy(m_blobIpV.count(), m_blobIpV.num, m_blobIpV.channels, m_nNumHeads, nInnerNum, i, m_blobIpVfull.mutable_gpu_data, m_blobIpV.gpu_data, DIR.BWD);
            }

            // Transpose to get the new shapes
            // queries tensor - q: [num_samples x num_heads x num_future_steps x state_size]
            // keys tensor    - k: [num_samples x num_heads x num_total_steps x state_size]
            // values tensor  - v: [num_samples x num_heads x num_total_steps x state_size]

            addBtmTop(m_blobIpQ, m_blobIpQt);
            m_transpose.Forward(m_colBtm, m_colTop);

            addBtmTop(m_blobIpK, m_blobIpKt);
            m_transpose.Forward(m_colBtm, m_colTop);

            addBtmTop(m_blobIpVfull, m_blobIpVt);
            m_transpose.Forward(m_colBtm, m_colTop);

            //-----------------------------------------
            // Calculate the attention
            //-----------------------------------------
            {
                // Apply the scaled dot product
                m_blobIpKt1.CopyFromAndTransposeHeightWidth(m_blobIpKt);
                m_blobAttnScores1.MatMul(m_blobIpQt, m_blobIpKt1, true);
                m_blobAttnScores1.scale_data(m_dfScale);

                // Decoder masking is applied to the multi-head attention layer to ensure that each temporal dimension can
                // only attend to the preceding features.
                if (m_blobMask != null)
                { 
                    // Apply mask to attention matrix
                    // att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                    float fInf = 1e29f;
                    // all masked items set to -inf.
                    m_cuda.mask_batch(m_blobAttnScores1.count(), 1, m_blobMask.count(), convert(1.0), convert(-1 * fInf), m_blobAttnScores1.gpu_data, m_blobMask.gpu_data, m_blobAttnScores1.mutable_gpu_data);
                }

                // Calculate the softmax to find the most imporant parts of the data (e.g. where to focus the attention)
                addBtmTop(m_blobAttnScores1, m_blobAttnScoresAllHeads);
                m_softmax.Forward(m_colBtm, m_colTop);

                // Multiply the softmax with the values to get the attention outputs.
                m_blobAttnOutputAllHeads.MatMul(m_blobAttnScoresAllHeads, m_blobIpVt, true);

                // attention scores -> colTop[2], shape [num_samples x num_heads x num_future_steps x num_total_steps]
                // attention output -> colTop[1], shape [num_samples x num_heads x num_future_steps x state_size]
            }

            // Average along all heads.
            calculateChannelMeanAcrossChannelsFwd(m_blobAttnOutputAllHeads, colTop[1]);
            calculateChannelMeanAcrossChannelsFwd(m_blobAttnScoresAllHeads, colTop[2]);

            // Weight the attention outputs (in colTop[1]) placing the results in colTop[0]
            addBtmTop(colTop[1], colTop[0]);
            m_ipOutLayer.Forward(m_colBtm, m_colTop);
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
            // Calculate grad for the attention output weights (colTop[0] grad -> colTop[1] attn output grad)
            addBtmTop(colTop[1], colTop[0]);
            m_ipOutLayer.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            // Average along all heads.
            calculateChannelMeanAcrossChannelsBwd(m_blobAttnOutputAllHeads, colTop[1]);

            //-----------------------------------------
            // Calculate the attention gradients
            //-----------------------------------------
            {
                // Multiply the softmax with the values to get the attention outputs.
                m_blobAttnOutputAllHeads.MatMulGrad(m_blobAttnScoresAllHeads, m_blobIpVt, m_blobWork);

                // Calculate the softmax gradient for the most imporant parts of the data (e.g. where to focus the attention)
                addBtmTop(m_blobAttnScores1, m_blobAttnScoresAllHeads);
                m_softmax.Backward(m_colTop, rgbPropagateDown, m_colBtm);

                // Calculate the Qt and Kt1 gradients.
                m_blobAttnScores1.MatMulGrad(m_blobIpQt, m_blobIpKt1, m_blobWork, m_dfScale);

                // Transform the gradients back to Kt.
                m_blobIpKt.CopyFromAndTransposeHeightWidth(m_blobIpKt1, true);
            }

            // Transpose the gradients back to Q, K and V
            addBtmTop(m_blobIpQ, m_blobIpQt);
            m_transpose.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            addBtmTop(m_blobIpK, m_blobIpKt);
            m_transpose.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            addBtmTop(m_blobIpVfull, m_blobIpVt);
            m_transpose.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            // Copy each IpVFull block to IpV
            m_blobIpV.SetDiff(0);

            int nOuterNum = m_blobIpVfull.count(0, 2);
            m_cuda.channel_copy(m_blobIpV.count(), nOuterNum, 1, m_nNumHeads, m_blobIpVfull.width, 0, m_blobIpVfull.gpu_diff, m_blobIpV.mutable_gpu_diff, DIR.FWD);

            for (int i = 1; i < m_nNumHeads; i++)
            {
                m_cuda.channel_add(m_blobIpV.count(), nOuterNum, 1, m_nNumHeads, m_blobIpVfull.width, i, m_blobIpVfull.gpu_diff, m_blobIpV.mutable_gpu_diff, DIR.FWD);
            }

            // Reshape back to original q, k, v projection shapes
            // queries tensor - q: [num_samples x num_future_steps x state_size]
            // keys tensor    - k: [num_samples x num_total_steps x state_size]
            // values tensor  - v: [num_samples x num_total_steps x state_size]            
            reshapeBwd(m_blobIpQ, m_nNumHeads);
            reshapeBwd(m_blobIpK, m_nNumHeads);
            reshapeBwd(m_blobIpV, m_nNumHeads);

            // Calculate q, k, v projection gradients
            addBtmTop(m_blobQ, m_blobIpQ);
            m_ipQLayer.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            addBtmTop(m_blobK, m_blobIpK);
            m_ipKLayer.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            addBtmTop(m_blobV, m_blobIpV);
            m_ipVLayer.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            if (colBottom.Count == 1)
            {
                colBottom[0].SetDiff(0);
                copy_to_q_bwd(colBottom.Count, colBottom[0], m_blobQ);
                m_cuda.add(colBottom[0].count(), colBottom[0].gpu_diff, m_blobK.gpu_diff, colBottom[0].mutable_gpu_diff);
                m_cuda.add(colBottom[0].count(), colBottom[0].gpu_diff, m_blobV.gpu_diff, colBottom[0].mutable_gpu_diff);
            }
            else
            {
                colBottom[0].CopyFrom(m_blobQ, true);
                colBottom[1].CopyFrom(m_blobK, true);
                colBottom[2].CopyFrom(m_blobV, true);
            }
        }
    }
}
