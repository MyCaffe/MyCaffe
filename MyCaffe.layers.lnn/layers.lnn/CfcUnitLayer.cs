using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.lnn;

namespace MyCaffe.layers.lnn
{
    /// <summary>
    /// The CfcUnitLayer implements the Closed form Continuous Cell (CfcCell) layer. 
    /// </summary>
    /// <remarks>
    /// @see [GitHub:raminmh/CfC](https://github.com/raminmh/CfC) by raminmh, 2021, GitHub (distributed under Apache 2.0).
    /// @see [Closed-form continuous-time neural networks](https://www.nature.com/articles/s42256-022-00556-7) by Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl and Daniela Rus, 2022, Nature Machine Intelligence, 4, 992-1003
    /// @see [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898) by Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus, 2021, arXiv 2106.13898
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class CfcUnitLayer<T> : LnnUnitLayer<T>
    {
        Layer<T> m_cat;
        Layer<T>[] m_rgLinearLayers = null;
        Layer<T>[] m_rgActivationLayers = null;
        Layer<T>[] m_rgDropoutLayers = null;
        BlobCollection<T> m_rgLinearBtms = new BlobCollection<T>();
        BlobCollection<T> m_rgLinearTops = new BlobCollection<T>();
        BlobCollection<T> m_rgActivationBtms = new BlobCollection<T>();
        BlobCollection<T> m_rgActivationTops = new BlobCollection<T>();
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();
        Layer<T> m_tanh;
        Layer<T> m_sigmoid;
        Layer<T> m_ff1;
        Layer<T> m_ff2;
        Layer<T> m_timeA;
        Layer<T> m_timeB;
        Blob<T> m_blobFF1;
        Blob<T> m_blobFF2;
        Blob<T> m_blobTimeA;
        Blob<T> m_blobTimeB;
        Blob<T> m_blobTInterp;
        Blob<T> m_blobTInterp1;
        Blob<T> m_blobTInterpInv;
        Blob<T> m_blobTInterpOnes;
        Blob<T> m_blobTs;
        Blob<T> m_blobX;
        Blob<T> m_blobTop1;
        Blob<T> m_blobTop2;
        int m_nNumLayers;
        int m_nNumUnits;

        /// <summary>
        /// The CfcUnitLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public CfcUnitLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.CFC_UNIT;

            m_nNumLayers = m_param.cfc_unit_param.backbone_layers;
            m_nNumUnits = m_param.cfc_unit_param.backbone_units;

            if (m_nNumLayers < 1)
                m_nNumLayers = 1;

            LayerParameter concat = new LayerParameter(LayerParameter.LayerType.CONCAT, "concat");
            concat.concat_param.axis = 1;
            m_cat = Layer<T>.Create(m_cuda, m_log, convertLayerParam(concat, p), null);

            Blob<T> blobBtm = new Blob<T>(m_cuda, m_log);
            blobBtm.Name = "bb";

            for (int i = 0; i < m_nNumLayers; i++)
            {
                // Linear Layer
                m_rgLinearBtms.Add(blobBtm);

                Blob<T> blobTop = new Blob<T>(m_cuda, m_log);
                blobTop.Name = "bb_" + i.ToString();

                m_rgLinearTops.Add(blobTop);

                // Activation Layer
                blobBtm = blobTop;
                m_rgActivationBtms.Add(blobBtm);

                blobTop = new Blob<T>(m_cuda, m_log);
                blobTop.Name = "bb_act_" + i.ToString();

                m_rgActivationTops.Add(blobTop);
                blobBtm = blobTop;
            }

            m_rgLinearLayers = new Layer<T>[m_nNumLayers];
            m_rgActivationLayers = new Layer<T>[m_nNumLayers];

            if (m_param.cfc_unit_param.backbone_dropout_ratio > 0)
                m_rgDropoutLayers = new Layer<T>[m_nNumLayers];

            for (int i = 0; i < m_nNumLayers; i++)
            {
                LayerParameter ip = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "bb_" + i.ToString());
                ip.inner_product_param.num_output = (uint)m_nNumUnits;
                ip.inner_product_param.bias_term = true;
                ip.inner_product_param.weight_filler = new FillerParameter("xavier", 0.0, 0.01);
                ip.inner_product_param.bias_filler = new FillerParameter("constant", 0.1);
                m_rgLinearLayers[i] = Layer<T>.Create(m_cuda, m_log, convertLayerParam(ip, p), null);

                LayerParameter act;
                switch (m_param.cfc_unit_param.backbone_activation)
                {
                    case CfcUnitParameter.ACTIVATION.SILU:
                        act = new LayerParameter(LayerParameter.LayerType.SILU, "bb_act_" + i.ToString());
                        break;

                    case CfcUnitParameter.ACTIVATION.RELU:
                        act = new LayerParameter(LayerParameter.LayerType.RELU, "bb_act_" + i.ToString());
                        break;

                    case CfcUnitParameter.ACTIVATION.TANH:
                        act = new LayerParameter(LayerParameter.LayerType.TANH, "bb_act_" + i.ToString());
                        break;

                    case CfcUnitParameter.ACTIVATION.GELU:
                        act = new LayerParameter(LayerParameter.LayerType.GELU, "bb_act_" + i.ToString());
                        break;

                    case CfcUnitParameter.ACTIVATION.LECUN:
                        act = new LayerParameter(LayerParameter.LayerType.LECUN, "bb_act_" + i.ToString());
                        break;

                    default:
                        throw new Exception("Unknown activation type: " + m_param.cfc_unit_param.backbone_activation.ToString());
                }

                m_rgActivationLayers[i] = Layer<T>.Create(m_cuda, m_log, convertLayerParam(act, p), null);

                if (i > 0 && m_rgDropoutLayers != null)
                {
                    LayerParameter drop = new LayerParameter(LayerParameter.LayerType.DROPOUT, "bb_drop_" + i.ToString());
                    drop.dropout_param.dropout_ratio = m_param.cfc_unit_param.backbone_dropout_ratio;

                    m_rgDropoutLayers[i] = Layer<T>.Create(m_cuda, m_log, convertLayerParam(drop, p), null);
                }
            }

            m_blobX = new Blob<T>(m_cuda, m_log);
            m_blobX.Name = "x";

            // FF1 Layer
            LayerParameter ff1 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "ff1");
            ff1.inner_product_param.num_output = (uint)m_param.cfc_unit_param.hidden_size;
            ff1.inner_product_param.bias_term = true;
            ff1.inner_product_param.weight_filler = new FillerParameter("xavier", 0.0, 0.01);
            ff1.inner_product_param.bias_filler = new FillerParameter("constant", 0.1);
            m_ff1 = Layer<T>.Create(m_cuda, m_log, convertLayerParam(ff1, p), null);

            m_blobFF1 = new Blob<T>(m_cuda, m_log);
            m_blobFF1.Name = "ff1";

            // Tanh Layer
            LayerParameter tanh = new LayerParameter(LayerParameter.LayerType.TANH, "tanh");
            m_tanh = Layer<T>.Create(m_cuda, m_log, convertLayerParam(tanh, p), null);

            // FF2 Layer
            LayerParameter ff2 = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "ff2");
            ff2.inner_product_param.num_output = (uint)m_param.cfc_unit_param.hidden_size;
            ff2.inner_product_param.bias_term = true;
            ff2.inner_product_param.weight_filler = new FillerParameter("xavier", 0.0, 0.01);
            ff2.inner_product_param.bias_filler = new FillerParameter("constant", 0.1);
            m_ff2 = Layer<T>.Create(m_cuda, m_log, convertLayerParam(ff2, p), null);

            m_blobFF2 = new Blob<T>(m_cuda, m_log);
            m_blobFF2.Name = "ff2";

            // Time A Layer
            LayerParameter timeA = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "time_a");
            timeA.inner_product_param.num_output = (uint)m_param.cfc_unit_param.hidden_size;
            timeA.inner_product_param.bias_term = true;
            timeA.inner_product_param.weight_filler = new FillerParameter("xavier", 0.0, 0.01);
            timeA.inner_product_param.bias_filler = new FillerParameter("constant", 0.1);
            m_timeA = Layer<T>.Create(m_cuda, m_log, convertLayerParam(timeA, p), null);

            m_blobTimeA = new Blob<T>(m_cuda, m_log);
            m_blobTimeA.Name = "time_a";

            // Time B Layer
            LayerParameter timeB = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "time_b");
            timeB.inner_product_param.num_output = (uint)m_param.cfc_unit_param.hidden_size;
            timeB.inner_product_param.bias_term = true;
            timeB.inner_product_param.weight_filler = new FillerParameter("xavier", 0.0, 0.01);
            timeB.inner_product_param.bias_filler = new FillerParameter("constant", 0.1);
            m_timeB = Layer<T>.Create(m_cuda, m_log, convertLayerParam(timeB, p), null);

            m_blobTimeB = new Blob<T>(m_cuda, m_log);
            m_blobTimeB.Name = "time_b";

            // Sigmoid Layer
            LayerParameter sigmoid = new LayerParameter(LayerParameter.LayerType.SIGMOID, "sigmoid");
            m_sigmoid = Layer<T>.Create(m_cuda, m_log, convertLayerParam(sigmoid, p), null);

            // T-Interp
            m_blobTInterp = new Blob<T>(m_cuda, m_log);
            m_blobTInterp.Name = "t-interp";

            m_blobTInterpInv = new Blob<T>(m_cuda, m_log);
            m_blobTInterpInv.Name = "t-interpinv";

            m_blobTInterp1 = new Blob<T>(m_cuda, m_log);
            m_blobTInterp1.Name = "t-interp1";
            m_blobTInterpOnes = new Blob<T>(m_cuda, m_log, true);
            m_blobTInterpOnes.Name = "t_interp_ones";

            m_blobTs = new Blob<T>(m_cuda, m_log);
            m_blobTs.Name = "ts";

            m_blobTop1 = new Blob<T>(m_cuda, m_log);
            m_blobTop1.Name = "top1";
            m_blobTop2 = new Blob<T>(m_cuda, m_log);
            m_blobTop2.Name = "top2";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            if (m_rgLinearLayers != null)
            {
                for (int i = 0; i < m_rgLinearLayers.Length; i++)
                {
                    m_rgLinearLayers[i].Dispose();
                }
                m_rgLinearLayers = null;
            }

            if (m_bOwnInternalBlobs)
                dispose_internal_blobs();

            dispose(ref m_blobTInterpOnes);

            dispose(ref m_cat);
            dispose(ref m_tanh);
            dispose(ref m_sigmoid);
            dispose(ref m_ff1);
            dispose(ref m_ff2);
            dispose(ref m_timeA);
            dispose(ref m_timeB);
        }

        private void dispose_internal_blobs()
        {
            dispose(ref m_rgLinearBtms);
            dispose(ref m_rgLinearTops);
            dispose(ref m_rgActivationBtms);
            dispose(ref m_rgActivationTops);

            dispose(ref m_blobFF1);
            dispose(ref m_blobFF2);
            dispose(ref m_blobTimeA);
            dispose(ref m_blobTimeB);
            dispose(ref m_blobTInterp);
            dispose(ref m_blobTInterp1);
            dispose(ref m_blobTInterpInv);
            dispose(ref m_blobTs);
            dispose(ref m_blobX);
            dispose(ref m_blobTop1);
            dispose(ref m_blobTop2);
        }

        /// <summary>
        /// Create the internal shared blobs used by the layer for a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <param name="cuda">Specifies the underlying CudaDnn low-level DLL.</param>
        /// <param name="log">Specifies the log.</param>
        /// <returns>The collection of created blobs is returned.</returns>
        /// <remarks>
        /// Note when creating shared blobs, existing internal blobs are disposed.
        /// </remarks>
        public override BlobCollection<T> CreateInternalSharedBlobs(int nIdx, CudaDnn<T> cuda, Log log)
        { 
            BlobCollection<T> col = new BlobCollection<T>();

            dispose_internal_blobs();

            Blob<T> blobFF1 = new Blob<T>(cuda, log);
            blobFF1.Name = "ff1_" + nIdx.ToString();
            col.Add(blobFF1);

            Blob<T> blobFF2 = new Blob<T>(cuda, log);
            blobFF2.Name = "ff2_" + nIdx.ToString();
            col.Add(blobFF2);

            Blob<T> blobTimeA = new Blob<T>(cuda, log);
            blobTimeA.Name = "timeA_" + nIdx.ToString();
            col.Add(blobTimeA);

            Blob<T> blobTimeB = new Blob<T>(cuda, log);
            blobTimeB.Name = "timeB_" + nIdx.ToString();
            col.Add(blobTimeB);

            Blob<T> blobTInterp = new Blob<T>(cuda, log);
            blobTInterp.Name = "tInterp_" + nIdx.ToString();
            col.Add(blobTInterp);

            Blob<T> blobTInterp1 = new Blob<T>(cuda, log);
            blobTInterp1.Name = "tInterp1_" + nIdx.ToString();
            col.Add(blobTInterp1);

            Blob<T> blobTInterpInv = new Blob<T>(cuda, log);
            blobTInterpInv.Name = "tInterpInv_" + nIdx.ToString();
            col.Add(blobTInterpInv);

            Blob<T> blobTs = new Blob<T>(cuda, log);
            blobTs.Name = "ts_" + nIdx.ToString();
            col.Add(blobTs);

            Blob<T> blobX = new Blob<T>(cuda, log);
            blobX.Name = "x_" + nIdx.ToString();
            col.Add(blobX);

            Blob<T> blobTop1 = new Blob<T>(cuda, log);
            blobTop1.Name = "top1_" + nIdx.ToString();
            col.Add(blobTop1);

            Blob<T> blobTop2 = new Blob<T>(cuda, log);
            blobTop2.Name = "top2_" + nIdx.ToString();
            col.Add(blobTop2);

            Blob<T> blobBb = new Blob<T>(cuda, log);
            blobBb.Name = "bb_" + nIdx.ToString();
            col.Add(blobBb);

            for (int i = 0; i < m_param.cfc_unit_param.backbone_layers; i++)
            {
                Blob<T> blobFc = new Blob<T>(cuda, log);
                blobFc.Name = "bb_fc" + (i + 1).ToString() + "_" + nIdx.ToString();
                col.Add(blobFc);

                Blob<T> blobAct = new Blob<T>(cuda, log);
                blobAct.Name = "bb_act" + (i + 1).ToString() + "_" + nIdx.ToString();
                col.Add(blobAct);
            }

            return col;
        }

        /// <summary>
        /// Set the internal shared blobs to a set of external blobs.
        /// </summary>
        /// <param name="col">Specifies the blob collection created using CreateInternalBlobs.</param>
        public override void SetInternalSharedBlobs(BlobCollection<T> col)
        {
            int nIdx = 0;

            m_bOwnInternalBlobs = false;
            m_blobFF1 = col[nIdx];
            nIdx++;

            m_blobFF2 = col[nIdx];
            nIdx++;

            m_blobTimeA = col[nIdx];
            nIdx++;

            m_blobTimeB = col[nIdx];
            nIdx++;

            m_blobTInterp = col[nIdx];
            nIdx++;

            m_blobTInterp1 = col[nIdx];
            nIdx++;

            m_blobTInterpInv = col[nIdx];
            nIdx++;

            m_blobTs = col[nIdx];
            nIdx++;

            m_blobX = col[nIdx];
            nIdx++;

            m_blobTop1 = col[nIdx];
            nIdx++;

            m_blobTop2 = col[nIdx];
            nIdx++;

            BlobCollection<T> colLin = new BlobCollection<T>();
            while (nIdx < col.Count)
            {
                colLin.Add(col[nIdx]);
                nIdx++;
            }

            nIdx = 0;
            for (int i=0; i<m_param.cfc_unit_param.backbone_layers; i++)
            {
                m_rgLinearBtms[i] = colLin[nIdx];
                nIdx++;
                m_rgLinearTops[i] = colLin[nIdx];
                m_rgActivationBtms[i] = colLin[nIdx];
                nIdx++;
                m_rgActivationTops[i] = colLin[nIdx];
            }

            colLin.Clear();
        }

        private void addBtmTop(Blob<T> btm, Blob<T> top)
        {
            m_colBtm.Clear();
            m_colBtm.Add(btm);
            m_colTop.Clear();
            m_colTop.Add(top);
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input, hx, ts
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 3; }
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
            return true;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            addBtmTop(colBottom[0], m_rgLinearBtms[0]);
            m_colBtm.Add(colBottom[1]);
            m_cat.Setup(m_colBtm, m_colTop);

            for (int i = 0; i < m_nNumLayers; i++)
            {
                addBtmTop(m_rgLinearBtms[i], m_rgLinearTops[i]);
                m_rgLinearLayers[i].Setup(m_colBtm, m_colTop);
                blobs.Add(m_rgLinearLayers[i].blobs);

                addBtmTop(m_rgActivationBtms[i], m_rgActivationTops[i]);
                m_rgActivationLayers[i].Setup(m_colBtm, m_colTop);

                if (i > 0 && m_rgDropoutLayers != null)
                    m_rgDropoutLayers[i].Setup(m_colBtm, m_colTop);
            }

            Blob<T> blobX = m_rgActivationTops[m_nNumLayers - 1];
            m_blobX.ReshapeLike(blobX);

            // FF1 Layer
            addBtmTop(blobX, m_blobFF1);
            m_ff1.Setup(m_colBtm, m_colTop);
            blobs.Add(m_ff1.blobs);

            // Tanh Layer
            addBtmTop(m_blobFF1, m_blobFF1);
            m_tanh.Setup(m_colBtm, m_colTop);

            // FF2 Layer
            addBtmTop(blobX, m_blobFF2);
            m_ff2.Setup(m_colBtm, m_colTop);
            blobs.Add(m_ff2.blobs);

            addBtmTop(m_blobFF2, m_blobFF2);
            m_tanh.Setup(m_colBtm, m_colTop);

            // Time A Layer
            addBtmTop(blobX, m_blobTimeA);
            m_timeA.Setup(m_colBtm, m_colTop);
            blobs.Add(m_timeA.blobs);

            // Time B Layer
            addBtmTop(blobX, m_blobTimeB);
            m_timeB.Setup(m_colBtm, m_colTop);
            blobs.Add(m_timeB.blobs);

            // T-Interp
            m_blobTInterp.ReshapeLike(m_blobTimeA);
            m_blobTInterpInv.ReshapeLike(m_blobTimeA);
            m_blobTInterp1.ReshapeLike(m_blobTimeA);
            m_blobTInterpOnes.ReshapeLike(m_blobTimeA);

            addBtmTop(m_blobTInterp, colTop[0]);
            m_sigmoid.Setup(m_colBtm, m_colTop);

            m_blobTs.ReshapeLike(m_blobTimeA);
            m_blobTop1.ReshapeLike(colTop[0]);
            m_blobTop2.ReshapeLike(colTop[0]);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            addBtmTop(colBottom[0], m_rgLinearBtms[0]);
            m_colBtm.Add(colBottom[1]);
            m_cat.Reshape(m_colBtm, m_colTop);

            for (int i = 0; i < m_rgLinearLayers.Length; i++)
            {
                addBtmTop(m_rgLinearBtms[i], m_rgLinearTops[i]);
                m_rgLinearLayers[i].Reshape(m_colBtm, m_colTop);

                addBtmTop(m_rgActivationBtms[i], m_rgActivationTops[i]);
                m_rgActivationLayers[i].Reshape(m_colBtm, m_colTop);

                if (m_rgDropoutLayers != null)
                    m_rgDropoutLayers[i].Reshape(m_colBtm, m_colTop);
            }

            Blob<T> blobX = m_rgActivationTops[m_rgLinearLayers.Length - 1];
            m_blobX.ReshapeLike(blobX);

            // FF1 Layer
            addBtmTop(blobX, m_blobFF1);
            m_ff1.Reshape(m_colBtm, m_colTop);

            // Tanh Layer
            addBtmTop(m_blobFF1, m_blobFF1);
            m_tanh.Reshape(m_colBtm, m_colTop);

            // FF2 Layer
            addBtmTop(blobX, m_blobFF2);
            m_ff2.Reshape(m_colBtm, m_colTop);

            addBtmTop(m_blobFF2, m_blobFF2);
            m_tanh.Reshape(m_colBtm, m_colTop);

            // Time A Layer
            addBtmTop(blobX, m_blobTimeA);
            m_timeA.Reshape(m_colBtm, m_colTop);
            m_blobTs.ReshapeLike(m_blobTimeA);

            m_blobTInterp.ReshapeLike(m_blobTimeA);
            m_blobTInterpInv.ReshapeLike(m_blobTimeA);
            m_blobTInterp1.ReshapeLike(m_blobTimeA);
            m_blobTInterpOnes.ReshapeLike(m_blobTimeA);
            m_blobTInterpOnes.SetData(1.0);

            // Time B Layer
            addBtmTop(blobX, m_blobTimeB);
            m_timeB.Reshape(m_colBtm, m_colTop);

            // Sigmoid Layer
            addBtmTop(m_blobTInterp, colTop[0]);
            m_sigmoid.Reshape(m_colBtm, m_colTop);

            m_blobTop1.ReshapeLike(colTop[0]);
            m_blobTop2.ReshapeLike(colTop[0]);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs @f$ 
        ///         y
        ///     @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            addBtmTop(colBottom[0], m_rgLinearBtms[0]);
            m_colBtm.Add(colBottom[1]);
            m_cat.Forward(m_colBtm, m_colTop);

            for (int i = 0; i < m_rgLinearLayers.Length; i++)
            {
                addBtmTop(m_rgLinearBtms[i], m_rgLinearTops[i]);
                m_rgLinearLayers[i].Forward(m_colBtm, m_colTop);

                addBtmTop(m_rgActivationBtms[i], m_rgActivationTops[i]);
                m_rgActivationLayers[i].Forward(m_colBtm, m_colTop);

                if (m_rgDropoutLayers != null)
                    m_rgDropoutLayers[i].Forward(m_colBtm, m_colTop);
            }

            Blob<T> blobX = m_rgActivationTops[m_rgLinearLayers.Length - 1];
            m_blobX.CopyFrom(blobX, false);

            // FF1 Layer
            addBtmTop(blobX, m_blobFF1);
            m_ff1.Forward(m_colBtm, m_colTop);

            // Tanh Layer
            addBtmTop(m_blobFF1, m_blobFF1);
            m_tanh.Forward(m_colBtm, m_colTop);

            // FF2 Layer
            addBtmTop(blobX, m_blobFF2);
            m_ff2.Forward(m_colBtm, m_colTop);

            addBtmTop(m_blobFF2, m_blobFF2);
            m_tanh.Forward(m_colBtm, m_colTop);

            // Time A Layer
            addBtmTop(blobX, m_blobTimeA);
            m_timeA.Forward(m_colBtm, m_colTop);

            // Time B Layer
            addBtmTop(blobX, m_blobTimeB);
            m_timeB.Forward(m_colBtm, m_colTop);

            // Calculate the t-interpolation factor.
            m_cuda.channel_fillfrom(m_blobTs.count(), m_blobTs.num, 1, m_blobTs.channels, colBottom[2].gpu_data, m_blobTs.mutable_gpu_data, DIR.FWD);
            // t_a * ts
            m_cuda.mul(m_blobTInterp.count(), m_blobTimeA.gpu_data, m_blobTs.gpu_data, m_blobTInterp.mutable_gpu_data);

            // t_interp = t_a * ts + t_b
            m_cuda.add(m_blobTInterp.count(), m_blobTimeB.gpu_data, m_blobTInterp.gpu_data, m_blobTInterp.mutable_gpu_data);

            // Sigmoid Layer
            addBtmTop(m_blobTInterp, m_blobTInterp);
            m_sigmoid.Forward(m_colBtm, m_colTop);

            if (m_param.cfc_unit_param.no_gate)
            {
                // t_interp * ff2
                m_cuda.mul(m_blobTop1.count(), m_blobTInterp.gpu_data, m_blobFF2.gpu_data, m_blobTop1.mutable_gpu_data);
                // ff1 + t_interp * ff2
                m_cuda.add(m_blobTop2.count(), m_blobFF1.gpu_data, m_blobTop1.gpu_data, m_blobTop2.mutable_gpu_data);
                colTop[0].CopyFrom(m_blobTop2);
            }
            else
            {
                // 1.0 - t_interp
                m_blobTInterpInv.SetData(1.0);
                m_cuda.sub(m_blobTInterpInv.count(), m_blobTInterpInv.gpu_data, m_blobTInterp.gpu_data, m_blobTInterpInv.mutable_gpu_data);
                // ff1 * (1.0 - t_interp)
                m_cuda.mul(m_blobTInterpInv.count(), m_blobTInterpInv.gpu_data, m_blobFF1.gpu_data, m_blobTop1.mutable_gpu_data);
                // t_interp * ff2
                m_cuda.mul(colTop[0].count(), m_blobTInterp.gpu_data, m_blobFF2.gpu_data, m_blobTop2.mutable_gpu_data);
                // ff1 * (1.0 - t_interp) + t_interp * ff2
                m_cuda.add(colTop[0].count(), m_blobTop1.gpu_data, m_blobTop2.gpu_data, colTop[0].mutable_gpu_data);
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the CfcUnit value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs @f$ x @f$; Backward fills their diff with 
        ///     gradients @f$ y' @f$
        ///     if propagate_down[0]
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (m_param.cfc_unit_param.no_gate)
            {
                // grad = top.grad * ones
                m_blobFF1.CopyFrom(colTop[0], true);

                // grad = top.grad * t-interp
                m_cuda.mul(m_blobFF2.count(), colTop[0].gpu_diff, m_blobTInterp.gpu_data, m_blobFF2.mutable_gpu_diff);

                // grad = top.grad * ff2
                m_cuda.mul(m_blobTInterp.count(), colTop[0].gpu_diff, m_blobFF2.gpu_data, m_blobTInterp1.mutable_gpu_diff);
            }
            else
            {
                // ff1 grad = top.trad * (1.0 - t_interp)
                m_cuda.sub(m_blobFF1.count(), m_blobTInterpOnes.gpu_data, m_blobTInterp.gpu_data, m_blobFF1.mutable_gpu_diff);
                m_cuda.mul(m_blobFF1.count(), m_blobFF1.gpu_diff, colTop[0].gpu_diff, m_blobFF1.mutable_gpu_diff);

                // ff2 grad = top.grad * t_interp
                m_cuda.mul(m_blobFF2.count(), colTop[0].gpu_diff, m_blobTInterp.gpu_data, m_blobFF2.mutable_gpu_diff);

                // ti grad = top.grad * (ff2 - ff1)
                m_cuda.sub(m_blobTInterp1.count(), m_blobFF2.gpu_data, m_blobFF1.gpu_data, m_blobTInterp1.mutable_gpu_diff);
                m_cuda.mul(m_blobTInterp1.count(), m_blobTInterp1.gpu_diff, colTop[0].gpu_diff, m_blobTInterp1.mutable_gpu_diff);
            }

            // Sigmoid Grad
            m_blobTInterp1.CopyFrom(m_blobTInterp);
            addBtmTop(m_blobTInterp, m_blobTInterp1);
            m_sigmoid.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            // t_b grad = t-interp grad * 1.0
            m_blobTimeB.CopyFrom(m_blobTInterp, true);

            // t_a grad = t-interp grad * ts
            m_cuda.mul(m_blobTimeA.count(), m_blobTInterp.gpu_diff, m_blobTs.gpu_data, m_blobTimeA.mutable_gpu_diff);

            // ts grad = t-interp grad * t_a
            m_cuda.mul(m_blobTs.count(), m_blobTInterp.gpu_diff, m_blobTimeA.gpu_data, m_blobTs.mutable_gpu_diff);
            m_cuda.channel_sum(m_blobTs.count(), 1, m_blobTs.num, m_blobTs.channels, m_blobTs.gpu_diff, colBottom[2].mutable_gpu_diff, false);

            Blob<T> blobX = m_rgActivationTops[m_rgActivationTops.Count-1];
            blobX.SetDiff(0);

            // time_b grad
            addBtmTop(m_blobX, m_blobTimeB);
            m_timeB.Backward(m_colTop, rgbPropagateDown, m_colBtm);
            m_cuda.add(m_blobX.count(), m_blobX.gpu_diff, blobX.gpu_diff, blobX.mutable_gpu_diff);

            // time_a grad
            addBtmTop(m_blobX, m_blobTimeA);
            m_timeA.Backward(m_colTop, rgbPropagateDown, m_colBtm);
            m_cuda.add(m_blobX.count(), m_blobX.gpu_diff, blobX.gpu_diff, blobX.mutable_gpu_diff);

            // ff2 grad
            addBtmTop(m_blobFF2, m_blobFF2);
            m_tanh.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            addBtmTop(m_blobX, m_blobFF2);
            m_ff2.Backward(m_colTop, rgbPropagateDown, m_colBtm);
            m_cuda.add(m_blobX.count(), m_blobX.gpu_diff, blobX.gpu_diff, blobX.mutable_gpu_diff);

            // ff1 grad
            addBtmTop(m_blobFF1, m_blobFF1);
            m_tanh.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            addBtmTop(m_blobX, m_blobFF1);
            m_ff1.Backward(m_colTop, rgbPropagateDown, m_colBtm);
            m_cuda.add(m_blobX.count(), m_blobX.gpu_diff, blobX.gpu_diff, blobX.mutable_gpu_diff);

            // Backbone grad
            for (int i = m_rgLinearLayers.Length - 1; i >= 0; i--)
            {
                addBtmTop(m_rgActivationBtms[i], m_rgActivationTops[i]);

                if (m_rgDropoutLayers != null)
                    m_rgDropoutLayers[i].Backward(m_colTop, rgbPropagateDown, m_colBtm);

                m_rgActivationLayers[i].Backward(m_colTop, rgbPropagateDown, m_colBtm);

                addBtmTop(m_rgLinearBtms[i], m_rgLinearTops[i]);
                m_rgLinearLayers[i].Backward(m_colTop, rgbPropagateDown, m_colBtm);
            }

            addBtmTop(colBottom[0], m_rgLinearBtms[0]);
            m_colBtm.Add(colBottom[1]);
            m_cat.Backward(m_colTop, new List<bool>() { true, true }, m_colBtm);
        }
    }
}
