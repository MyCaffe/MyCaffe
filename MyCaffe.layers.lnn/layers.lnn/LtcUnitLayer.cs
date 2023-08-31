using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.param;
using MyCaffe.param.lnn;

namespace MyCaffe.layers.lnn
{
    /// <summary>
    /// The LtcUnitLayer implements the liquid time constant with ODE solver (LTCCell) layer. 
    /// </summary>
    /// <remarks>
    /// @see [GitHub:raminmh/CfC](https://github.com/raminmh/CfC) by raminmh, 2021, GitHub (distributed under Apache 2.0).
    /// @see [Closed-form continuous-time neural networks](https://www.nature.com/articles/s42256-022-00556-7) by Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl and Daniela Rus, 2022, Nature Machine Intelligence, 4, 992-1003
    /// @see [Closed-form Continuous-time Neural Models](https://arxiv.org/abs/2106.13898) by Ramin Hasani, Mathias Lechner, Alexander Amini, Lucas Liebenwein, Aaron Ray, Max Tschaikowski, Gerald Teschl, Daniela Rus, 2021, arXiv 2106.13898
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class LtcUnitLayer<T> : LnnUnitLayer<T>
    {
        Blob<T> m_blobInputs = null;
        Blob<T> m_blobMues = null;
        Blob<T> m_blobX = null;
        Blob<T> m_blobSensorySigmoidW = null;
        Blob<T> m_blobSensoryActivationW = null;
        Blob<T> m_blobSensoryActivationW1 = null;
        Blob<T> m_blobSensoryActivationRev = null;
        Blob<T> m_blobSensoryNumeratorW = null;
        Blob<T> m_blobSensoryDenominatorW = null;
        Blob<T> m_blobSensoryNumeratorW1 = null;
        Blob<T> m_blobSensoryDenominatorW1 = null;
        Blob<T> m_blobTs = null;
        Blob<T> m_blobCmt = null;
        Blob<T> m_blobWork = null;
        Blob<T> m_blobVPre = null;
        BlobCollection<T> m_colWtsAccum = new BlobCollection<T>();
        BlobCollection<T> m_colCmt = new BlobCollection<T>();
        BlobCollection<T> m_colVPre = new BlobCollection<T>();
        BlobCollection<T> m_colMues = new BlobCollection<T>();
        BlobCollection<T> m_colSigmoidW = new BlobCollection<T>();
        BlobCollection<T> m_colActivationW = new BlobCollection<T>();
        BlobCollection<T> m_colActivationW1 = new BlobCollection<T>();
        BlobCollection<T> m_colActivationRev = new BlobCollection<T>();
        BlobCollection<T> m_colNumeratorW = new BlobCollection<T>();
        BlobCollection<T> m_colDenominatorW = new BlobCollection<T>();
        BlobCollection<T> m_colNumerator1 = new BlobCollection<T>();
        BlobCollection<T> m_colNumerator2 = new BlobCollection<T>();
        BlobCollection<T> m_colNumerator = new BlobCollection<T>();
        BlobCollection<T> m_colDenominator = new BlobCollection<T>();
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();
        Layer<T> m_sigmoid = null;
        int[] m_rgShape = new int[] { 1, 1, 1, 1, };

        /// <summary>
        /// Defines the type of weight in the blobs.
        /// </summary>
        public enum WEIGHT
        {
            /// <summary>
            /// Specifies the gleak weight
            /// </summary>
            GLEAK,
            /// <summary>
            /// Specifies the vleak weight
            /// </summary>
            VLEAK,
            /// <summary>
            /// Specifies the cm weight
            /// </summary>
            CM,
            /// <summary>
            /// Specifies the sigma weight
            /// </summary>
            SIGMA,
            /// <summary>
            /// Specifies the mu weight
            /// </summary>
            MU,
            /// <summary>
            /// Specifies the w weight
            /// </summary>
            W,
            /// <summary>
            /// Specifies the erev weight
            /// </summary>
            EREV,
            /// <summary>
            /// Specifies the sensory sigma weight
            /// </summary>
            SENSORY_SIGMA,
            /// <summary>
            /// Specifies the sensory mu weight
            /// </summary>
            SENSORY_MU,
            /// <summary>
            /// Specifies the sensory w weight
            /// </summary>
            SENSORY_W,
            /// <summary>
            /// Specifies the sensory erev weight
            /// </summary>
            SENSORY_EREV,
            /// <summary>
            /// Specifies the input weight
            /// </summary>
            INPUT_WT,
            /// <summary>
            /// Specifies the input bias
            /// </summary>
            INPUT_BIAS
        }


        /// <summary>
        /// The LtcUnitLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public LtcUnitLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.LTC_UNIT;

            int nSensorySize = m_param.ltc_unit_param.input_size;
            int nStateSize = m_param.ltc_unit_param.hidden_size;

            List<int> rgShape = new List<int>() { nStateSize };
            addWeight(blobs, m_colWtsAccum, "gleak", rgShape, m_param.ltc_unit_param.gleak_init_min, m_param.ltc_unit_param.gleak_init_max);
            addWeight(blobs, m_colWtsAccum, "vleak", rgShape, m_param.ltc_unit_param.vleak_init_min, m_param.ltc_unit_param.vleak_init_max);
            addWeight(blobs, m_colWtsAccum, "cm", rgShape, m_param.ltc_unit_param.cm_init_min, m_param.ltc_unit_param.cm_init_max);

            rgShape.Add(nStateSize);
            addWeight(blobs, m_colWtsAccum, "sigma", rgShape, m_param.ltc_unit_param.sigma_init_min, m_param.ltc_unit_param.sigma_init_max);
            addWeight(blobs, m_colWtsAccum, "mu", rgShape, m_param.ltc_unit_param.mu_init_min, m_param.ltc_unit_param.mu_init_max);
            addWeight(blobs, m_colWtsAccum, "w", rgShape, m_param.ltc_unit_param.w_init_min, m_param.ltc_unit_param.w_init_max);
            addWeight(blobs, m_colWtsAccum, "erev", rgShape);

            rgShape[0] = nSensorySize;
            addWeight(blobs, m_colWtsAccum, "sensory_sigma", rgShape, m_param.ltc_unit_param.sensory_sigma_init_min, m_param.ltc_unit_param.sensory_sigma_init_max);
            addWeight(blobs, m_colWtsAccum, "sensory_mu", rgShape, m_param.ltc_unit_param.sensory_mu_init_min, m_param.ltc_unit_param.sensory_mu_init_max);
            addWeight(blobs, m_colWtsAccum, "sensory_w", rgShape, m_param.ltc_unit_param.sensory_w_init_min, m_param.ltc_unit_param.sensory_w_init_max);
            addWeight(blobs, m_colWtsAccum, "sensory_erev", rgShape);

            addWeight(blobs, m_colWtsAccum, "input_w", nSensorySize, 1.0);
            addWeight(blobs, m_colWtsAccum, "input_b", nSensorySize, 0.0);

            m_blobVPre = new Blob<T>(cuda, log);
            m_blobVPre.Name = m_param.name + ".vpre";
            m_blobWork = new Blob<T>(cuda, log);
            m_blobWork.Name = m_param.name + ".work";
            m_blobInputs = new Blob<T>(m_cuda, m_log);
            m_blobInputs.Name = m_param.name + ".inputs";
            m_blobMues = new Blob<T>(m_cuda, m_log);
            m_blobMues.Name = m_param.name + ".mues";
            m_blobX = new Blob<T>(m_cuda, m_log);
            m_blobX.Name = m_param.name + ".x";
            m_blobSensorySigmoidW = new Blob<T>(m_cuda, m_log);
            m_blobSensorySigmoidW.Name = m_param.name + ".sensory_sigmoid_w";
            m_blobSensoryActivationW = new Blob<T>(m_cuda, m_log);
            m_blobSensoryActivationW.Name = m_param.name + ".sensory_activation_w";
            m_blobSensoryActivationW1 = new Blob<T>(m_cuda, m_log);
            m_blobSensoryActivationW1.Name = m_param.name + ".sensory_activation_w1";
            m_blobSensoryActivationRev = new Blob<T>(m_cuda, m_log);
            m_blobSensoryActivationRev.Name = m_param.name + ".sensory_activation_erev";
            m_blobSensoryNumeratorW = new Blob<T>(m_cuda, m_log);
            m_blobSensoryNumeratorW.Name = m_param.name + ".sensory_numerator_w";
            m_blobSensoryDenominatorW = new Blob<T>(m_cuda, m_log);
            m_blobSensoryDenominatorW.Name = m_param.name + ".sensory_denominator_w";
            m_blobSensoryNumeratorW1 = new Blob<T>(m_cuda, m_log);
            m_blobSensoryNumeratorW1.Name = m_param.name + ".sensory_numerator_w1";
            m_blobSensoryDenominatorW1 = new Blob<T>(m_cuda, m_log);
            m_blobSensoryDenominatorW1.Name = m_param.name + ".sensory_denominator_w1";
            m_blobCmt = new Blob<T>(m_cuda, m_log);
            m_blobCmt.Name = m_param.name + ".cm_t";
            m_blobTs = new Blob<T>(m_cuda, m_log);
            m_blobTs.Name = m_param.name + ".ts";

            for (int i=0; i<m_param.ltc_unit_param.ode_unfolds; i++)
            {
                m_colVPre.Add(new Blob<T>(cuda, log));
                m_colVPre[i].Name = m_param.name + ".vpre." + i.ToString();
                m_colCmt.Add(new Blob<T>(cuda, log));
                m_colCmt[i].Name = m_param.name + ".cmt." + i.ToString();

                m_colMues.Add(new Blob<T>(cuda, log));
                m_colMues[i].Name = m_param.name + ".mues." + i.ToString();
                m_colSigmoidW.Add(new Blob<T>(cuda, log));
                m_colSigmoidW[i].Name = m_param.name + ".sigmoid_w." + i.ToString();
                m_colActivationW.Add(new Blob<T>(cuda, log));
                m_colActivationW[i].Name = m_param.name + ".activation_w." + i.ToString();
                m_colActivationW1.Add(new Blob<T>(cuda, log));
                m_colActivationW1[i].Name = m_param.name + ".activation_w1." + i.ToString();
                m_colActivationRev.Add(new Blob<T>(cuda, log));
                m_colActivationRev[i].Name = m_param.name + ".activation_rev." + i.ToString();
                m_colNumeratorW.Add(new Blob<T>(cuda, log));
                m_colNumeratorW[i].Name = m_param.name + ".numerator_w." + i.ToString();
                m_colDenominatorW.Add(new Blob<T>(cuda, log));
                m_colDenominatorW[i].Name = m_param.name + ".denominator_w." + i.ToString();
                m_colNumerator1.Add(new Blob<T>(cuda, log));
                m_colNumerator1[i].Name = m_param.name + ".numerator1." + i.ToString();
                m_colNumerator2.Add(new Blob<T>(cuda, log));
                m_colNumerator2[i].Name = m_param.name + ".numerator2." + i.ToString();
                m_colNumerator.Add(new Blob<T>(cuda, log));
                m_colNumerator[i].Name = m_param.name + ".numerator." + i.ToString();
                m_colDenominator.Add(new Blob<T>(cuda, log));
                m_colDenominator[i].Name = m_param.name + ".denominator." + i.ToString();
            }

            LayerParameter sigmoid_param = new LayerParameter(LayerParameter.LayerType.SIGMOID);
            sigmoid_param.name = m_param.name + ".sigmoid";
            sigmoid_param.sigmoid_param.engine = EngineParameter.Engine.CAFFE;
            m_sigmoid = Layer<T>.Create(cuda, log, sigmoid_param, null);
        }

        private void addWeight(BlobCollection<T> blobs1, BlobCollection<T> blobsAcc, string strName, List<int> rgShape, float fMin, float fMax)
        {
            Blob<T> blob = new Blob<T>(m_cuda, m_log, rgShape);
            blob.Name = strName;

            Blob<T> blobAcc = new Blob<T>(m_cuda, m_log, rgShape, false);
            blobAcc.Name = strName + "_acc";

            FillerParameter fp = new FillerParameter("uniform");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            fp.min = fMin;
            fp.max = fMax;

            filler.Fill(blob);

            blobAcc.SetData(0);

            blobs1.Add(blob);
            blobsAcc.Add(blobAcc);
        }

        private void addWeight(BlobCollection<T> blobs1, BlobCollection<T> blobsAcc, string strName, List<int> rgShape)
        {
            Blob<T> blob = new Blob<T>(m_cuda, m_log, rgShape);
            blob.Name = strName;

            Blob<T> blobAcc = new Blob<T>(m_cuda, m_log, rgShape, false);
            blobAcc.Name = strName + "_acc";

            FillerParameter fp = new FillerParameter("uniform");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            fp.min = -1;
            fp.max = 1;

            filler.Fill(blob);

            blobAcc.SetData(0);

            blobs1.Add(blob);
            blobsAcc.Add(blobAcc);
        }

        private void addWeight(BlobCollection<T> blobs1, BlobCollection<T> blobsAcc, string strName, int nSize, double dfVal)
        {
            List<int> rgShape = new List<int>() { nSize };

            Blob<T> blob = new Blob<T>(m_cuda, m_log, rgShape);
            blob.Name = strName;

            Blob<T> blobAcc = new Blob<T>(m_cuda, m_log, rgShape, false);
            blobAcc.Name = strName + "_acc";

            FillerParameter fp = new FillerParameter("constant", dfVal);
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(blob);

            blobAcc.SetData(0);

            blobs1.Add(blob);
            blobsAcc.Add(blobAcc);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            if (m_bOwnInternalBlobs)
                dispose_internal_blobs();

            dispose(ref m_colWtsAccum);
            dispose(ref m_sigmoid);
        }

        private void dispose_internal_blobs(bool bSetToNull = true)
        {
            dispose(ref m_blobVPre);
            dispose(ref m_blobInputs);
            dispose(ref m_blobMues);
            dispose(ref m_blobX);
            dispose(ref m_blobSensorySigmoidW);
            dispose(ref m_blobSensoryActivationW);
            dispose(ref m_blobSensoryActivationW1);
            dispose(ref m_blobSensoryActivationRev);
            dispose(ref m_blobSensoryNumeratorW);
            dispose(ref m_blobSensoryDenominatorW);
            dispose(ref m_blobSensoryNumeratorW1);
            dispose(ref m_blobSensoryDenominatorW1);
            dispose(ref m_blobCmt);
            dispose(ref m_blobTs);
            dispose(ref m_blobWork);

            dispose(ref m_colVPre, bSetToNull);
            dispose(ref m_colCmt, bSetToNull);
            dispose(ref m_colMues, bSetToNull);
            dispose(ref m_colSigmoidW, bSetToNull);
            dispose(ref m_colActivationW, bSetToNull);
            dispose(ref m_colActivationW1, bSetToNull);
            dispose(ref m_colActivationRev, bSetToNull);
            dispose(ref m_colNumeratorW, bSetToNull);
            dispose(ref m_colDenominatorW, bSetToNull);
            dispose(ref m_colNumerator, bSetToNull);
            dispose(ref m_colNumerator1, bSetToNull);
            dispose(ref m_colNumerator2, bSetToNull);
            dispose(ref m_colDenominator, bSetToNull);
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

            dispose_internal_blobs(false);

            Blob<T> blobVPre = new Blob<T>(cuda, log);
            blobVPre.Name = m_param.name + ".vpre";
            col.Add(blobVPre);

            Blob<T> blobWork = new Blob<T>(cuda, log);
            blobWork.Name = m_param.name + ".work";
            col.Add(blobWork);

            Blob<T> blobInputs = new Blob<T>(m_cuda, m_log);
            blobInputs.Name = m_param.name + ".inputs";
            col.Add(blobInputs);

            Blob<T> blobMues = new Blob<T>(m_cuda, m_log);
            blobMues.Name = m_param.name + ".mues";
            col.Add(blobMues);

            Blob<T> blobX = new Blob<T>(m_cuda, m_log);
            blobX.Name = m_param.name + ".x";
            col.Add(blobX);

            Blob<T> blobSensorySigmoidW = new Blob<T>(m_cuda, m_log);
            blobSensorySigmoidW.Name = m_param.name + ".sensory_sigmoid_w";
            col.Add(blobSensorySigmoidW);

            Blob<T> blobSensoryActivationW = new Blob<T>(m_cuda, m_log);
            blobSensoryActivationW.Name = m_param.name + ".sensory_activation_w";
            col.Add(blobSensoryActivationW);

            Blob<T> blobSensoryActivationW1 = new Blob<T>(m_cuda, m_log);
            blobSensoryActivationW1.Name = m_param.name + ".sensory_activation_w1";
            col.Add(blobSensoryActivationW1);

            Blob<T> blobSensoryActivationRev = new Blob<T>(m_cuda, m_log);
            blobSensoryActivationRev.Name = m_param.name + ".sensory_activation_erev";
            col.Add(blobSensoryActivationRev);

            Blob<T> blobSensoryNumeratorW = new Blob<T>(m_cuda, m_log);
            blobSensoryNumeratorW.Name = m_param.name + ".sensory_numerator_w";
            col.Add(blobSensoryNumeratorW);

            Blob<T> blobSensoryDenominatorW = new Blob<T>(m_cuda, m_log);
            blobSensoryDenominatorW.Name = m_param.name + ".sensory_denominator_w";
            col.Add(blobSensoryDenominatorW);

            Blob<T> blobSensoryNumeratorW1 = new Blob<T>(m_cuda, m_log);
            blobSensoryNumeratorW1.Name = m_param.name + ".sensory_numerator_w1";
            col.Add(blobSensoryNumeratorW1);

            Blob<T> blobSensoryDenominatorW1 = new Blob<T>(m_cuda, m_log);
            blobSensoryDenominatorW1.Name = m_param.name + ".sensory_denominator_w1";
            col.Add(blobSensoryDenominatorW1);

            Blob<T> blobCmt = new Blob<T>(m_cuda, m_log);
            blobCmt.Name = m_param.name + ".cm_t";
            col.Add(blobCmt);

            Blob<T> blobTs = new Blob<T>(m_cuda, m_log);
            blobTs.Name = m_param.name + ".ts";
            col.Add(blobTs);

            for (int i = 0; i < m_param.ltc_unit_param.ode_unfolds; i++)
            {
                Blob<T> blobVPre_a = new Blob<T>(cuda, log);
                blobVPre_a.Name = m_param.name + ".vpre." + i.ToString();
                col.Add(blobVPre_a);

                Blob<T> blobCmt_a = new Blob<T>(cuda, log);
                blobCmt_a.Name = m_param.name + ".cmt." + i.ToString();
                col.Add(blobCmt_a);

                Blob<T> blobMues_a = new Blob<T>(cuda, log);
                blobMues_a.Name = m_param.name + ".mues." + i.ToString();
                col.Add(blobMues_a);

                Blob<T> blobSigmoidW_a = new Blob<T>(cuda, log);
                blobSigmoidW_a.Name = m_param.name + ".sigmoid_w." + i.ToString();
                col.Add(blobSigmoidW_a);

                Blob<T> blobActivationW_a = new Blob<T>(cuda, log);
                blobActivationW_a.Name = m_param.name + ".activation_w." + i.ToString();
                col.Add(blobActivationW_a);

                Blob<T> blobActivationW1_a = new Blob<T>(cuda, log);
                blobActivationW1_a.Name = m_param.name + ".activation_w1." + i.ToString();
                col.Add(blobActivationW1_a);

                Blob<T> blobActivationRev_a = new Blob<T>(cuda, log);
                blobActivationRev_a.Name = m_param.name + ".activation_rev." + i.ToString();
                col.Add(blobActivationRev_a);

                Blob<T> blobNumeratorW_a = new Blob<T>(cuda, log);
                blobNumeratorW_a.Name = m_param.name + ".numerator_w." + i.ToString();
                col.Add(blobNumeratorW_a);

                Blob<T> blobDenominatorW_a = new Blob<T>(cuda, log);
                blobDenominatorW_a.Name = m_param.name + ".denominator_w." + i.ToString();
                col.Add(blobDenominatorW_a);

                Blob<T> blobNumerator1_a = new Blob<T>(cuda, log);
                blobNumerator1_a.Name = m_param.name + ".numerator1." + i.ToString();
                col.Add(blobNumerator1_a);

                Blob<T> blobNumerator2_a = new Blob<T>(cuda, log);
                blobNumerator2_a.Name = m_param.name + ".numerator2." + i.ToString();
                col.Add(blobNumerator2_a);

                Blob<T> blobNumerator_a = new Blob<T>(cuda, log);
                blobNumerator_a.Name = m_param.name + ".numerator." + i.ToString();
                col.Add(blobNumerator_a);

                Blob<T> blobDenominator_a = new Blob<T>(cuda, log);
                blobDenominator_a.Name = m_param.name + ".denominator." + i.ToString();
                col.Add(blobDenominator_a);
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

            m_blobVPre = col[nIdx];
            nIdx++;

            m_blobWork = col[nIdx];
            nIdx++;

            m_blobInputs = col[nIdx];
            nIdx++;

            m_blobMues = col[nIdx];
            nIdx++;

            m_blobX = col[nIdx];
            nIdx++;

            m_blobSensorySigmoidW = col[nIdx];
            nIdx++;

            m_blobSensoryActivationW = col[nIdx];
            nIdx++;

            m_blobSensoryActivationW1 = col[nIdx];
            nIdx++;

            m_blobSensoryActivationRev = col[nIdx];
            nIdx++;

            m_blobSensoryNumeratorW = col[nIdx];
            nIdx++;

            m_blobSensoryDenominatorW = col[nIdx];
            nIdx++;

            m_blobSensoryNumeratorW1 = col[nIdx];
            nIdx++;

            m_blobSensoryDenominatorW1 = col[nIdx];
            nIdx++;

            m_blobCmt = col[nIdx];
            nIdx++;

            m_blobTs = col[nIdx];
            nIdx++;

            m_colVPre.Clear();
            m_colCmt.Clear();
            m_colMues.Clear();
            m_colSigmoidW.Clear();
            m_colActivationW.Clear();
            m_colActivationW1.Clear();
            m_colActivationRev.Clear();
            m_colNumeratorW.Clear();
            m_colDenominatorW.Clear();
            m_colNumerator1.Clear();
            m_colNumerator2.Clear();
            m_colNumerator.Clear();
            m_colDenominator.Clear();

            for (int i = 0; i < m_param.ltc_unit_param.ode_unfolds; i++)
            {
                m_colVPre.Add(col[nIdx]);
                nIdx++;

                m_colCmt.Add(col[nIdx]);
                nIdx++;

                m_colMues.Add(col[nIdx]);
                nIdx++;

                m_colSigmoidW.Add(col[nIdx]);
                nIdx++;

                m_colActivationW.Add(col[nIdx]);
                nIdx++;

                m_colActivationW1.Add(col[nIdx]);
                nIdx++;

                m_colActivationRev.Add(col[nIdx]);
                nIdx++;

                m_colNumeratorW.Add(col[nIdx]);
                nIdx++;

                m_colDenominatorW.Add(col[nIdx]);
                nIdx++;

                m_colNumerator1.Add(col[nIdx]);
                nIdx++;

                m_colNumerator2.Add(col[nIdx]);
                nIdx++;

                m_colNumerator.Add(col[nIdx]);
                nIdx++;

                m_colDenominator.Add(col[nIdx]);
                nIdx++;
            }
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
            addBtmTop(m_blobInputs, m_blobSensoryActivationW);
            m_sigmoid.Setup(m_colBtm, m_colTop);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_blobInputs.ReshapeLike(colBottom[0]);
            m_blobMues.Reshape(colBottom[0].num, colBottom[0].channels, m_param.ltc_unit_param.hidden_size, 1);
            m_colMues.Reshape(colBottom[0].num, colBottom[0].channels, m_param.ltc_unit_param.hidden_size, 1);
            m_blobX.Reshape(colBottom[0].num, colBottom[0].channels, m_param.ltc_unit_param.hidden_size, 1);

            m_blobVPre.ReshapeLike(colBottom[1]);

            m_rgShape[0] = m_blobInputs.num;
            m_rgShape[1] = m_blobInputs.channels;
            m_rgShape[2] = m_param.ltc_unit_param.hidden_size;
            m_blobSensorySigmoidW.Reshape(m_rgShape);
            m_blobSensoryActivationW.Reshape(m_rgShape);
            m_blobSensoryActivationW1.Reshape(m_rgShape);
            m_blobSensoryActivationRev.Reshape(m_rgShape);

            m_rgShape[0] = m_blobSensoryActivationW.num;
            m_rgShape[1] = m_blobSensoryActivationW.height;
            m_rgShape[2] = m_blobSensoryActivationW.width;
            m_blobSensoryNumeratorW.Reshape(m_rgShape);
            m_blobSensoryDenominatorW.Reshape(m_rgShape);
            m_blobSensoryNumeratorW1.Reshape(m_rgShape);
            m_blobSensoryDenominatorW1.Reshape(m_rgShape);
            m_blobCmt.Reshape(m_rgShape);
            m_blobTs.ReshapeLike(colBottom[2]);

            m_rgShape[1] = m_param.ltc_unit_param.hidden_size;
            m_rgShape[2] = m_param.ltc_unit_param.hidden_size;
            m_colSigmoidW.Reshape(m_rgShape);
            m_colActivationW.Reshape(m_rgShape);
            m_colActivationW1.Reshape(m_rgShape);
            m_colActivationRev.Reshape(m_rgShape);

            m_rgShape[2] = 1;
            m_colNumeratorW.Reshape(m_rgShape);
            m_colDenominatorW.Reshape(m_rgShape);
            m_colNumerator.Reshape(m_rgShape);
            m_colNumerator1.Reshape(m_rgShape);
            m_colDenominator.Reshape(m_rgShape);

            m_rgShape[0] = m_param.ltc_unit_param.hidden_size;
            m_rgShape[1] = 1;
            m_colNumerator2.Reshape(m_rgShape);

            m_colCmt.ReshapeLike(m_blobCmt);
            m_colVPre.ReshapeLike(m_blobVPre);

            colTop[0].ReshapeLike(m_blobVPre);
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
            map_inputs_fwd(colBottom[0], m_blobInputs);

            addBtmTop(m_blobInputs, colTop[0]);
            m_colBtm.Add(colBottom[1]);
            m_colBtm.Add(colBottom[2]);
            ode_solver_fwd(m_colBtm, m_colTop);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the LtcUnit value inputs.
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
            addBtmTop(m_blobInputs, colTop[0]);
            m_colBtm.Add(colBottom[1]);
            m_colBtm.Add(colBottom[2]);
            ode_solver_bwd(m_colBtm, m_colTop);

            map_inputs_bwd(colBottom[0], m_blobInputs);
        }

        private void op_fwd(OP op, Blob<T> btm1, Blob<T> btm2, Blob<T> top, int nC = 0, int nN1 = 0, int nSD1 = 0, int nN2 = 0, int nSD2 = 0)
        {
            if (nC == 0)
                nC = btm1.channels;

            if (nN1 == 0)
                nN1 = btm1.num;
            
            if (nSD1 == 0)
                nSD1 = (btm1.num_axes < 2) ? 1 : btm1.count(2);
            
            if (nN2 == 0)
                nN2 = (btm2.num_axes == 1) ? 1 : btm2.num;
            
            if (nSD2 == 0)  
                nSD2 = (btm2.num_axes < 2) ? 1 : btm2.count(2);
            
            int nN = Math.Max(nN1, nN2);
            int nSD = Math.Max(nSD1, nSD2);
            int nCount = nN * nC * nSD;

            if (nCount != top.count())
                top.Reshape(nN, nC, nSD, 1);

            m_cuda.channel_op_fwd(op, top.count(), nC, nN1, nSD1, nN2, nSD2, btm1.gpu_data, btm2.gpu_data, top.mutable_gpu_data);
        }

        private void op_bwd_local(OP op, Blob<T> btm1, Blob<T> btm2, Blob<T> top, int nC = 0, int nN1 = 0, int nSD1 = 0, int nN2 = 0, int nSD2 = 0)
        {
            if (nC == 0)
                nC = btm1.channels;

            if (nN1 == 0)
                nN1 = btm1.num;

            if (nSD1 == 0)
                nSD1 = (btm1.num_axes < 2) ? 1 : btm1.count(2);

            if (nN2 == 0)
                nN2 = (btm2.num_axes == 1) ? 1 : btm2.num;

            if (nSD2 == 0)
                nSD2 = (btm2.num_axes < 2) ? 1 : btm2.count(2);

            int nN = Math.Max(nN1, nN2);
            int nSD = Math.Max(nSD1, nSD2);
            int nCount = nN * nC * nSD;

            if (nCount != top.count())
                top.Reshape(nN, nC, nSD, 1);

            if (nCount != m_blobWork.count())
                m_blobWork.ReshapeLike(top);

            //m_cuda.channel_op_bwd(op, top.count(), nC, nN1, nSD1, nN2, nSD2, btm1.gpu_data, btm2.gpu_data, top.gpu_data, btm1.mutable_gpu_diff, btm2.mutable_gpu_diff, top.gpu_diff, m_blobWork.mutable_gpu_data);

            // Gradient for btm1
            if (top.gpu_diff != btm1.gpu_diff)
            {
                m_blobWork.SetDiff(0);
                if (op == OP.MUL || op == OP.DIV)
                {
                    if (top.count() == btm1.count())
                        m_cuda.channel_op_fwd(op, nCount, nC, nN1, nSD1, nN2, nSD2, top.gpu_diff, btm2.gpu_data, btm1.mutable_gpu_diff);
                    else
                        m_cuda.channel_op_fwd(op, nCount, nC, nN1, nSD1, nN2, nSD2, top.gpu_diff, btm2.gpu_data, m_blobWork.mutable_gpu_diff);
                }
                else
                {
                    if (top.count() == btm1.count())
                        btm1.CopyFrom(top, true);
                    else
                        m_blobWork.CopyFrom(top, true);
                }

                if (nSD1 < nSD2)
                {
                    int nNa = nN1 * nC * nSD1;
                    int nCa = nSD2 / nSD1;
                    int nSDa = 1;
                    m_cuda.channel_sum(nCount, nNa, nCa, nSDa, m_blobWork.gpu_diff, btm1.mutable_gpu_diff, true);
                }
                else if (nN1 < nN2)
                {
                    int nNa = nN1;
                    int nCa = nN2 / nN1;
                    int nSDa = nC * nSD1;
                    m_cuda.channel_sum(nCount, nNa, nCa, nSDa, m_blobWork.gpu_diff, btm1.mutable_gpu_diff, true);
                }
            }

            // Gradient for btm2
            if (top.gpu_diff != btm2.gpu_diff)
            {
                m_blobWork.SetDiff(0);
                if (op == OP.DIV)
                {
                    m_cuda.powx(btm2.count(), btm2.gpu_data, 2.0, btm2.mutable_gpu_diff);
                    m_cuda.channel_op_fwd(op, nCount, nC, nN1, nSD1, nN2, nSD2, btm1.gpu_data, btm2.gpu_diff, m_blobWork.mutable_gpu_diff);
                    m_blobWork.scale_diff(-1);

                    int nCc = m_blobWork.channels;
                    int nNc = m_blobWork.num;
                    int nSDc = top.count(2);

                    if (top.count() == btm2.count())
                        m_cuda.channel_op_fwd(OP.MUL, nCount, nCc, nNc, nSDc, nNc, nSDc, m_blobWork.gpu_diff, top.gpu_diff, btm2.mutable_gpu_diff);
                    else
                        m_cuda.channel_op_fwd(OP.MUL, nCount, nCc, nNc, nSDc, nNc, nSDc, m_blobWork.gpu_diff, top.gpu_diff, m_blobWork.mutable_gpu_diff);
                }
                else if (op == OP.MUL)
                {
                    if (top.count() == btm2.count())
                        m_cuda.channel_op_fwd(op, nCount, nC, nN1, nSD1, nN2, nSD2, top.gpu_diff, btm1.gpu_data, btm2.mutable_gpu_diff);
                    else
                        m_cuda.channel_op_fwd(op, nCount, nC, nN1, nSD1, nN1, nSD2, top.gpu_diff, btm1.gpu_data, m_blobWork.mutable_gpu_diff);
                }
                else if (op == OP.SUB)
                {
                    if (top.count() == btm1.count())
                        m_cuda.scale(top.count(), -1, top.gpu_diff, btm2.mutable_gpu_diff);
                    else
                        m_cuda.scale(top.count(), -1, top.gpu_diff, m_blobWork.mutable_gpu_diff);
                }
                else
                {
                    if (top.count() == btm2.count())
                        btm2.CopyFrom(top, true);
                    else
                        m_blobWork.CopyFrom(top, true);
                }

                if (nSD2 < nSD1)
                {
                    int nNb = Math.Max(nN1, nN2);
                    int nCb = nSD1 / nSD2;
                    int nSDb = 1;
                    m_cuda.channel_sum(nCount, nNb, nCb, nSDb, m_blobWork.gpu_diff, btm2.mutable_gpu_diff, true);
                }
                else if (nN2 < nN1)
                {
                    int nNb = nN2;
                    int nCb = nN1 / nN2;
                    int nSDb = nC * nSD2;
                    m_cuda.channel_sum(nCount, nNb, nCb, nSDb, m_blobWork.gpu_diff, btm2.mutable_gpu_diff, true);
                }
            }
        }

        private void op_bwd(OP op, Blob<T> btm1, Blob<T> btm2, Blob<T> top, int nC = 0, int nN1 = 0, int nSD1 = 0, int nN2 = 0, int nSD2 = 0, int nCy = 0, int nSDy = 0)
        {
            if (nC == 0)
                nC = btm1.channels;

            if (nN1 == 0)
                nN1 = btm1.num;

            if (nSD1 == 0)
                nSD1 = (btm1.num_axes < 2) ? 1 : btm1.count(2);

            if (nN2 == 0)
                nN2 = (btm2.num_axes == 1) ? 1 : btm2.num;

            if (nSD2 == 0)
                nSD2 = (btm2.num_axes < 2) ? 1 : btm2.count(2);

            if (nCy == 0)
                nCy = top.channels;

            if (nSDy == 0)
                nSDy = top.count(2);

            int nN = Math.Max(nN1, nN2);
            int nSD = Math.Max(nSD1, nSD2);
            int nCount = nN * nC * nSD;

            if (nCount != top.count())
                top.Reshape(nN, nC, nSD, 1);

            if (nCount != m_blobWork.count())
                m_blobWork.ReshapeLike(top);

            m_cuda.channel_op_bwd(op, top.count(), nC, nN1, nSD1, nN2, nSD2, nCy, nSDy, btm1.gpu_data, btm2.gpu_data, top.gpu_data, btm1.mutable_gpu_diff, btm2.mutable_gpu_diff, top.gpu_diff, m_blobWork.mutable_gpu_data);
        }

        private void sigmoid_fwd(BlobCollection<T> colBtm, BlobCollection<T> colTop, int t = -1)
        {
            Blob<T> blobPre = colBtm[0];
            Blob<T> blobMu = colBtm[1];
            Blob<T> blobSigma = colBtm[2];
            Blob<T> blobTop = colTop[0];
            Blob<T> blobMues = m_blobMues;

            if (t >= 0)
                blobMues = m_colMues[t];

            op_fwd(OP.SUB, blobPre, blobMu, blobMues, blobPre.channels, blobPre.num, blobPre.count(2), 1, blobMu.channels);
            op_fwd(OP.MUL, blobMues, blobSigma, m_blobX, blobMues.channels, blobMues.num, blobMues.count(2), 1, blobSigma.channels);

            addBtmTop(m_blobX, blobTop);
            m_sigmoid.Forward(m_colBtm, m_colTop);
        }

        private void sigmoid_bwd(BlobCollection<T> colBtm, BlobCollection<T> colTop, int t = -1)
        {
            Blob<T> blobPre = colBtm[0];
            Blob<T> blobMu = colBtm[1];
            Blob<T> blobSigma = colBtm[2];
            Blob<T> blobTop = colTop[0];
            Blob<T> blobMues = m_blobMues;

            if (t >= 0)
                blobMues = m_colMues[t];

            addBtmTop(m_blobX, blobTop);
            m_sigmoid.Backward(m_colTop, new List<bool>() { true }, m_colBtm);

            op_bwd_local(OP.MUL, blobMues, blobSigma, m_blobX, blobMues.channels, blobMues.num, blobMues.count(2), 1, blobSigma.channels);
            op_bwd_local(OP.SUB, blobPre, blobMu, blobMues, blobPre.channels, blobPre.num, blobPre.count(2), 1, blobMu.channels);
        }


        private void map_inputs_fwd(Blob<T> btm, Blob<T> top)
        {
            op_fwd(OP.MUL, btm, blobs[(int)WEIGHT.INPUT_WT], top);
            op_fwd(OP.ADD, top, blobs[(int)WEIGHT.INPUT_BIAS], top);
        }

        private void map_inputs_bwd(Blob<T> btm, Blob<T> top)
        {
            op_bwd(OP.ADD, top, blobs[(int)WEIGHT.INPUT_BIAS], top);
            op_bwd(OP.MUL, btm, blobs[(int)WEIGHT.INPUT_WT], top);
        }

        private void ode_solver_fwd(BlobCollection<T> colBtm, BlobCollection<T> colTop)
        {
            Blob<T> blobInputs = colBtm[0];
            Blob<T> blobVPre = colBtm[1];
            Blob<T> blobTs = colBtm[2];
            Blob<T> blobTop = colTop[0];
            int nN = blobInputs.num;
            int nC = blobInputs.channels;
            int nSD = m_param.ltc_unit_param.hidden_size;

            int nCount = blobInputs.count();

            m_blobVPre.CopyFrom(blobVPre);

            // Pre-compute the effect of the sensory inputs.
            addBtmTop(blobInputs, m_blobSensorySigmoidW);
            m_colBtm.Add(blobs[(int)WEIGHT.SENSORY_MU]);
            m_colBtm.Add(blobs[(int)WEIGHT.SENSORY_SIGMA]);
            sigmoid_fwd(m_colBtm, m_colTop);

            op_fwd(OP.MUL, m_blobSensorySigmoidW, blobs[(int)WEIGHT.SENSORY_W], m_blobSensoryActivationW, nC, nN, nSD, 1, nSD);
            op_fwd(OP.MUL, m_blobSensoryActivationW, blobs[(int)WEIGHT.SENSORY_EREV], m_blobSensoryActivationRev, nC, nN, nSD, 1, nSD);

            // Reduce over dim=1 (source sensory neurons)
            m_cuda.channel_sum(nCount, m_blobSensoryActivationRev.num, m_blobSensoryActivationRev.channels, m_blobSensoryActivationRev.count(2), m_blobSensoryActivationRev.gpu_data, m_blobSensoryNumeratorW.mutable_gpu_data, true);  
            m_cuda.channel_sum(nCount, m_blobSensoryActivationW.num, m_blobSensoryActivationW.channels, m_blobSensoryActivationW.count(2), m_blobSensoryActivationW.gpu_data, m_blobSensoryDenominatorW.mutable_gpu_data, true);

            // cm/t is loop invariant, so we can compute it once here.
            m_blobTs.CopyFrom(blobTs);
            m_blobTs.add_scalar(1.0);
            m_blobTs.scale_data(1.0 / m_param.ltc_unit_param.ode_unfolds);
            op_fwd(OP.DIV, blobs[(int)WEIGHT.CM], m_blobTs, m_blobCmt, 1, 1, nSD, nN, 1);

            // Unfold the multi-step ODE solver into one RNN step.
            for (int t = 0; t < m_param.ltc_unit_param.ode_unfolds; t++)
            {
                m_colVPre[t].CopyFrom(m_blobVPre);
                m_colCmt[t].CopyFrom(m_blobCmt);

                // Compute the W activation
                addBtmTop(m_colVPre[t], m_colSigmoidW[t]);
                m_colBtm.Add(blobs[(int)WEIGHT.MU]);
                m_colBtm.Add(blobs[(int)WEIGHT.SIGMA]);
                sigmoid_fwd(m_colBtm, m_colTop, t);
                op_fwd(OP.MUL, m_colSigmoidW[t], blobs[(int)WEIGHT.W], m_colActivationW[t], nSD, nN, nSD, 1, nSD);

                // Compute the Rev activation
                op_fwd(OP.MUL, m_colActivationW[t], blobs[(int)WEIGHT.EREV], m_colActivationRev[t], nSD, nN, nSD, 1, nSD);

                // Reduce over dim=1 (source neurons)
                m_cuda.channel_sum(nCount, m_colActivationRev[t].num, m_colActivationRev[t].channels, m_colActivationRev[t].count(2), m_colActivationRev[t].gpu_data, m_colNumeratorW[t].mutable_gpu_data, true);
                m_cuda.channel_sum(nCount, m_colActivationW[t].num, m_colActivationW[t].channels, m_colActivationW[t].count(2), m_colActivationW[t].gpu_data, m_colDenominatorW[t].mutable_gpu_data, true);
                // Add sensory input
                op_fwd(OP.ADD, m_colNumeratorW[t], m_blobSensoryNumeratorW, m_colNumeratorW[t]);
                op_fwd(OP.ADD, m_colDenominatorW[t], m_blobSensoryDenominatorW, m_colDenominatorW[t]);

                // Compute the numerator
                op_fwd(OP.MUL, m_colCmt[t], m_colVPre[t], m_colNumerator1[t]);
                op_fwd(OP.MUL, blobs[(int)WEIGHT.GLEAK], blobs[(int)WEIGHT.VLEAK], m_colNumerator2[t], nSD, 1, 1, 1, 1);
                op_fwd(OP.ADD, m_colNumerator1[t], m_colNumerator2[t], m_colNumerator[t], nSD, nN, 1, 1, 1);
                op_fwd(OP.ADD, m_colNumerator[t], m_colNumeratorW[t], m_colNumerator[t]);

                // Compute the denominator
                op_fwd(OP.ADD, m_colCmt[t], blobs[(int)WEIGHT.GLEAK], m_colDenominator[t], nSD, nN, 1, 1, 1);
                op_fwd(OP.ADD, m_colDenominator[t], m_colDenominatorW[t], m_colDenominator[t]);
                m_colDenominator[t].add_scalar(m_param.ltc_unit_param.epsilon);

                // Compute the output
                op_fwd(OP.DIV, m_colNumerator[t], m_colDenominator[t], m_blobVPre);
            }

            blobTop.CopyFrom(m_blobVPre);
        }

        private void accumulateGrad(BlobCollection<T> src, BlobCollection<T> dst, WEIGHT wt)
        {
            m_cuda.add(dst[(int)wt].count(), src[(int)wt].gpu_diff, dst[(int)wt].gpu_data, dst[(int)wt].mutable_gpu_data);
        }

        private void copyGrad(BlobCollection<T> src, BlobCollection<T> dst, WEIGHT wt)
        {
            m_cuda.copy(dst[(int)wt].count(), src[(int)wt].gpu_data, dst[(int)wt].mutable_gpu_diff);
        }

        private void ode_solver_bwd(BlobCollection<T> colBtm, BlobCollection<T> colTop)
        {
            Blob<T> blobInputs = colBtm[0];
            Blob<T> blobVPre = colBtm[1];
            Blob<T> blobTs = colBtm[2];
            Blob<T> blobTop = colTop[0];
            int nN = blobInputs.num;
            int nC = blobInputs.channels;
            int nSD = m_param.ltc_unit_param.hidden_size;

            int nCount = blobInputs.count();

            m_blobCmt.SetDiff(0);

            foreach (Blob<T> blob in m_colWtsAccum)
            {
                blob.SetDiff(0);
            }

            // Unfold the multi-step ODE solver into one RNN step.
            for (int t = m_param.ltc_unit_param.ode_unfolds-1; t >= 0; t--)
            {
                // Compute the output
                if (t == m_param.ltc_unit_param.ode_unfolds - 1)
                    op_bwd_local(OP.DIV, m_colNumerator[t], m_colDenominator[t], colTop[0]);
                else
                    op_bwd_local(OP.DIV, m_colNumerator[t], m_colDenominator[t], m_blobVPre);

                m_blobVPre.SetDiff(0);

                // Compute the denominator
                op_bwd(OP.ADD, m_colDenominator[t], m_colDenominatorW[t], m_colDenominator[t]);
                op_bwd(OP.ADD, m_colCmt[t], blobs[(int)WEIGHT.GLEAK], m_colDenominator[t], nSD, nN, 1, 1, 1);
                accumulateGrad(blobs, m_colWtsAccum, WEIGHT.GLEAK);
                m_cuda.add(m_blobCmt.count(), m_colCmt[t].gpu_diff, m_blobCmt.gpu_diff, m_blobCmt.mutable_gpu_diff);

                // Compute the numerator
                op_bwd(OP.ADD, m_colNumerator[t], m_colNumeratorW[t], m_colNumerator[t]);
                op_bwd(OP.ADD, m_colNumerator1[t], m_colNumerator2[t], m_colNumerator[t], nSD, nN, 1, 1, 1);
                op_bwd(OP.MUL, blobs[(int)WEIGHT.GLEAK], blobs[(int)WEIGHT.VLEAK], m_colNumerator2[t], nSD, 1, 1, 1, 1);
                accumulateGrad(blobs, m_colWtsAccum, WEIGHT.GLEAK);
                accumulateGrad(blobs, m_colWtsAccum, WEIGHT.VLEAK);

                op_bwd(OP.MUL, m_colCmt[t], m_colVPre[t], m_colNumerator1[t]);
                m_cuda.add(m_blobCmt.count(), m_colCmt[t].gpu_diff, m_blobCmt.gpu_diff, m_blobCmt.mutable_gpu_diff);
                m_cuda.add(m_blobVPre.count(), m_colVPre[t].gpu_diff, m_blobVPre.gpu_diff, m_blobVPre.mutable_gpu_diff);

                // Add sensory input
                op_bwd(OP.ADD, m_colDenominatorW[t], m_blobSensoryDenominatorW1, m_colDenominatorW[t]);
                m_cuda.add(m_blobSensoryDenominatorW.count(), m_blobSensoryDenominatorW1.gpu_diff, m_blobSensoryDenominatorW.gpu_diff, m_blobSensoryDenominatorW.mutable_gpu_diff);
                op_bwd(OP.ADD, m_colNumeratorW[t], m_blobSensoryNumeratorW1, m_colNumeratorW[t]);
                m_cuda.add(m_blobSensoryNumeratorW.count(), m_blobSensoryNumeratorW1.gpu_diff, m_blobSensoryNumeratorW.gpu_diff, m_blobSensoryNumeratorW.mutable_gpu_diff);

                // Reduce over dim=1 (source neurons)
                m_cuda.channel_sum(m_colActivationRev[t].count(), m_colActivationRev[t].num, m_colActivationRev[t].channels, m_colActivationRev[t].count(2), m_colActivationRev[t].mutable_gpu_diff, m_colNumeratorW[t].gpu_diff, true, DIR.BWD);
                m_cuda.channel_sum(m_colActivationW1[t].count(), m_colActivationW1[t].num, m_colActivationW1[t].channels, m_colActivationW1[t].count(2), m_colActivationW1[t].mutable_gpu_diff, m_colDenominatorW[t].gpu_diff, true, DIR.BWD);

                // Compute the Rev activation
                op_bwd_local(OP.MUL, m_colActivationW[t], blobs[(int)WEIGHT.EREV], m_colActivationRev[t], nSD, nN, nSD, 1, nSD);
                accumulateGrad(blobs, m_colWtsAccum, WEIGHT.EREV);
                // Accumulate the gradient
                m_cuda.add(m_colActivationW[t].count(), m_colActivationW1[t].gpu_diff, m_colActivationW[t].gpu_diff, m_colActivationW1[t].mutable_gpu_diff);

                // Compute the W activation
                op_bwd_local(OP.MUL, m_colSigmoidW[t], blobs[(int)WEIGHT.W], m_colActivationW1[t], nSD, nN, nSD, 1, nSD);
                accumulateGrad(blobs, m_colWtsAccum, WEIGHT.W);

                addBtmTop(m_colVPre[t], m_colSigmoidW[t]);
                m_colBtm.Add(blobs[(int)WEIGHT.MU]);
                m_colBtm.Add(blobs[(int)WEIGHT.SIGMA]);
                sigmoid_bwd(m_colBtm, m_colTop, t);

                m_cuda.add(m_blobVPre.count(), m_colVPre[t].gpu_diff, m_blobVPre.gpu_diff, m_blobVPre.mutable_gpu_diff);
                accumulateGrad(blobs, m_colWtsAccum, WEIGHT.MU);
                accumulateGrad(blobs, m_colWtsAccum, WEIGHT.SIGMA);
            }

            copyGrad(m_colWtsAccum, blobs, WEIGHT.GLEAK);
            copyGrad(m_colWtsAccum, blobs, WEIGHT.VLEAK);
            copyGrad(m_colWtsAccum, blobs, WEIGHT.EREV);
            copyGrad(m_colWtsAccum, blobs, WEIGHT.W);
            copyGrad(m_colWtsAccum, blobs, WEIGHT.MU);
            copyGrad(m_colWtsAccum, blobs, WEIGHT.SIGMA);

            m_cuda.debug();
            // cm/t is loop invariant, so we can compute it once here.
            op_bwd(OP.DIV, blobs[(int)WEIGHT.CM], m_blobTs, m_blobCmt, 1, 1, nSD, nN, 1, m_blobCmt.channels, m_blobCmt.count(2));
            m_blobTs.scale_diff(1.0 / m_param.ltc_unit_param.ode_unfolds);
            blobTs.CopyFrom(m_blobTs, true);

            // Reduce over dim=1 (source sensory neurons)
            m_cuda.channel_sum(m_blobSensoryActivationRev.count(), m_blobSensoryActivationRev.num, m_blobSensoryActivationRev.channels, m_blobSensoryActivationRev.count(2), m_blobSensoryActivationRev.gpu_diff, m_blobSensoryNumeratorW.mutable_gpu_diff, true, DIR.BWD, 1);
            m_cuda.channel_sum(m_blobSensoryActivationW1.count(), m_blobSensoryActivationW1.num, m_blobSensoryActivationW1.channels, m_blobSensoryActivationW1.count(2), m_blobSensoryActivationW1.gpu_diff, m_blobSensoryDenominatorW.mutable_gpu_diff, true, DIR.BWD, 1);

            // Pre-compute the effect of the sensory inputs.
            op_bwd_local(OP.MUL, m_blobSensoryActivationW, blobs[(int)WEIGHT.SENSORY_EREV], m_blobSensoryActivationRev, nC, nN, nSD, 1, nSD);
            m_cuda.add(m_blobSensoryActivationW.count(), m_blobSensoryActivationW.gpu_diff, m_blobSensoryActivationW1.gpu_diff, m_blobSensoryActivationW.mutable_gpu_diff);
            op_bwd_local(OP.MUL, m_blobSensorySigmoidW, blobs[(int)WEIGHT.SENSORY_W], m_blobSensoryActivationW, nC, nN, nSD, 1, nSD);

            addBtmTop(blobInputs, m_blobSensorySigmoidW);
            m_colBtm.Add(blobs[(int)WEIGHT.SENSORY_MU]);
            m_colBtm.Add(blobs[(int)WEIGHT.SENSORY_SIGMA]);
            sigmoid_bwd(m_colBtm, m_colTop);

            blobVPre.CopyFrom(m_blobVPre, true);
        }
    }
}
