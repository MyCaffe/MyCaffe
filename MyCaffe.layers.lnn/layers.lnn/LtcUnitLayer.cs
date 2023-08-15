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
    public class LtcUnitLayer<T> : Layer<T>
    {
        Blob<T> m_blobInputs = null;
        Blob<T> m_blobMues = null;
        Blob<T> m_blobX = null;
        Blob<T> m_blobSensoryActivationW = null;
        Blob<T> m_blobSensoryActivationRev = null;
        Blob<T> m_blobSensoryNumeratorW = null;
        Blob<T> m_blobSensoryDenominatorW = null;
        Blob<T> m_blobTs = null;
        Blob<T> m_blobCmt = null;
        Blob<T> m_blobActivationW = null;
        Blob<T> m_blobActivationRev = null;
        Blob<T> m_blobNumeratorW = null;
        Blob<T> m_blobDenominatorW = null;
        Blob<T> m_blobNumerator1 = null;
        Blob<T> m_blobNumerator2 = null;
        Blob<T> m_blobNumerator = null;
        Blob<T> m_blobDenominator = null;
        Blob<T> m_blobWork = null;
        Blob<T> m_blobVPre = null;
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
            addWeight(blobs, "gleak", rgShape, m_param.ltc_unit_param.gleak_init_min, m_param.ltc_unit_param.gleak_init_max);
            addWeight(blobs, "vleak", rgShape, m_param.ltc_unit_param.vleak_init_min, m_param.ltc_unit_param.vleak_init_max);
            addWeight(blobs, "cm", rgShape, m_param.ltc_unit_param.cm_init_min, m_param.ltc_unit_param.cm_init_max);

            rgShape.Add(nStateSize);
            addWeight(blobs, "sigma", rgShape, m_param.ltc_unit_param.sigma_init_min, m_param.ltc_unit_param.sigma_init_max);
            addWeight(blobs, "mu", rgShape, m_param.ltc_unit_param.mu_init_min, m_param.ltc_unit_param.mu_init_max);
            addWeight(blobs, "w", rgShape, m_param.ltc_unit_param.w_init_min, m_param.ltc_unit_param.w_init_max);
            addWeight(blobs, "erev", rgShape);

            rgShape[0] = nSensorySize;
            addWeight(blobs, "sensory_sigma", rgShape, m_param.ltc_unit_param.sensory_sigma_init_min, m_param.ltc_unit_param.sensory_sigma_init_max);
            addWeight(blobs, "sensory_mu", rgShape, m_param.ltc_unit_param.sensory_mu_init_min, m_param.ltc_unit_param.sensory_mu_init_max);
            addWeight(blobs, "sensory_w", rgShape, m_param.ltc_unit_param.sensory_w_init_min, m_param.ltc_unit_param.sensory_w_init_max);
            addWeight(blobs, "sensory_erev", rgShape);

            addWeight(blobs, "input_w", nSensorySize, 1.0);
            addWeight(blobs, "input_b", nSensorySize, 0.0);

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
            m_blobSensoryActivationW = new Blob<T>(m_cuda, m_log);
            m_blobSensoryActivationW.Name = m_param.name + ".sensory_activation_w";
            m_blobSensoryActivationRev = new Blob<T>(m_cuda, m_log);
            m_blobSensoryActivationRev.Name = m_param.name + ".sensory_activation_erev";
            m_blobSensoryNumeratorW = new Blob<T>(m_cuda, m_log);
            m_blobSensoryNumeratorW.Name = m_param.name + ".sensory_numerator_w";
            m_blobSensoryDenominatorW = new Blob<T>(m_cuda, m_log);
            m_blobSensoryDenominatorW.Name = m_param.name + ".sensory_denominator_w";
            m_blobActivationW = new Blob<T>(m_cuda, m_log);
            m_blobActivationW.Name = m_param.name + ".activation_w";
            m_blobActivationRev = new Blob<T>(m_cuda, m_log);
            m_blobActivationRev.Name = m_param.name + ".activation_rev";
            m_blobNumeratorW = new Blob<T>(m_cuda, m_log);
            m_blobNumeratorW.Name = m_param.name + ".numerator_w";
            m_blobDenominatorW = new Blob<T>(m_cuda, m_log);
            m_blobDenominatorW.Name = m_param.name + ".denominator_w";
            m_blobNumerator = new Blob<T>(m_cuda, m_log);
            m_blobNumerator.Name = m_param.name + ".numerator";
            m_blobNumerator1 = new Blob<T>(m_cuda, m_log);
            m_blobNumerator1.Name = m_param.name + ".numerator1";
            m_blobNumerator2 = new Blob<T>(m_cuda, m_log);
            m_blobNumerator2.Name = m_param.name + ".numerator2";
            m_blobDenominator = new Blob<T>(m_cuda, m_log);
            m_blobDenominator.Name = m_param.name + ".denominator";
            m_blobCmt = new Blob<T>(m_cuda, m_log);
            m_blobCmt.Name = m_param.name + ".cm_t";
            m_blobTs = new Blob<T>(m_cuda, m_log);
            m_blobTs.Name = m_param.name + ".ts";

            LayerParameter sigmoid_param = new LayerParameter(LayerParameter.LayerType.SIGMOID);
            sigmoid_param.name = m_param.name + ".sigmoid";
            m_sigmoid = new SigmoidLayer<T>(cuda, log, sigmoid_param);
        }

        private void addWeight(BlobCollection<T> blobs1, string strName, List<int> rgShape, float fMin, float fMax)
        {
            Blob<T> blob = new Blob<T>(m_cuda, m_log, rgShape);
            blob.Name = strName;

            FillerParameter fp = new FillerParameter("uniform");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            fp.min = fMin;
            fp.max = fMax;

            filler.Fill(blob);

            blobs1.Add(blob);
        }

        private void addWeight(BlobCollection<T> blobs1, string strName, List<int> rgShape)
        {
            Blob<T> blob = new Blob<T>(m_cuda, m_log, rgShape);
            blob.Name = strName;

            FillerParameter fp = new FillerParameter("uniform");
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            fp.min = -1;
            fp.max = 1;

            filler.Fill(blob);

            blobs1.Add(blob);
        }

        private void addWeight(BlobCollection<T> blobs1, string strName, int nSize, double dfVal)
        {
            List<int> rgShape = new List<int>() { nSize };

            Blob<T> blob = new Blob<T>(m_cuda, m_log, rgShape);
            blob.Name = strName;

            FillerParameter fp = new FillerParameter("constant", dfVal);
            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);

            filler.Fill(blob);

            blobs1.Add(blob);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            dispose(ref m_blobVPre);
            dispose(ref m_blobInputs);
            dispose(ref m_blobMues);
            dispose(ref m_blobX);
            dispose(ref m_blobSensoryActivationW);
            dispose(ref m_blobSensoryActivationRev);
            dispose(ref m_blobSensoryNumeratorW);
            dispose(ref m_blobSensoryDenominatorW);
            dispose(ref m_blobActivationW);
            dispose(ref m_blobActivationRev);
            dispose(ref m_blobNumeratorW);
            dispose(ref m_blobDenominatorW);
            dispose(ref m_blobNumerator);
            dispose(ref m_blobNumerator1);
            dispose(ref m_blobNumerator2);
            dispose(ref m_blobDenominator);
            dispose(ref m_blobCmt);
            dispose(ref m_blobTs);

            dispose(ref m_sigmoid);
        }

        /// <summary>
        /// Set the internal blobs to a set of external blobs.
        /// </summary>
        /// <param name="blobs1">Specifies internal weight blobs.</param>
        public void SetInternalBlobs(BlobCollection<T> blobs1)
        {
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
            m_blobX.Reshape(colBottom[0].num, colBottom[0].channels, m_param.ltc_unit_param.hidden_size, 1);

            m_blobVPre.ReshapeLike(colBottom[1]);

            m_rgShape[0] = m_blobInputs.num;
            m_rgShape[1] = m_blobInputs.channels;
            m_rgShape[2] = m_param.ltc_unit_param.hidden_size;
            m_blobSensoryActivationW.Reshape(m_rgShape);
            m_blobSensoryActivationRev.Reshape(m_rgShape);

            m_rgShape[0] = m_blobSensoryActivationW.num;
            m_rgShape[1] = m_blobSensoryActivationW.height;
            m_rgShape[2] = m_blobSensoryActivationW.width;
            m_blobSensoryNumeratorW.Reshape(m_rgShape);
            m_blobSensoryDenominatorW.Reshape(m_rgShape);
            m_blobCmt.Reshape(m_rgShape);
            m_blobTs.ReshapeLike(colBottom[2]);

            m_rgShape[1] = m_param.ltc_unit_param.hidden_size;
            m_rgShape[2] = m_param.ltc_unit_param.hidden_size;
            m_blobActivationW.Reshape(m_rgShape);
            m_blobActivationRev.Reshape(m_rgShape);

            m_rgShape[2] = 1;
            m_blobNumeratorW.Reshape(m_rgShape);
            m_blobDenominatorW.Reshape(m_rgShape);
            m_blobNumerator.Reshape(m_rgShape);
            m_blobNumerator1.Reshape(m_rgShape);
            m_blobDenominator.Reshape(m_rgShape);

            m_rgShape[0] = m_param.ltc_unit_param.hidden_size;
            m_rgShape[1] = 1;
            m_blobNumerator2.Reshape(m_rgShape);

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
                top.Reshape(Math.Max(nN1, nN2), nC, Math.Max(nSD1, nSD2), 1);

            m_cuda.channel_op(op, top.count(), nC, nN1, nSD1, nN2, nSD2, btm1.gpu_data, btm2.gpu_data, top.mutable_gpu_data, DIR.FWD);
        }

        private void sigmoid_fwd(BlobCollection<T> colBtm, BlobCollection<T> colTop)
        {
            Blob<T> blobPre = colBtm[0];
            Blob<T> blobMu = colBtm[1];
            Blob<T> blobSigma = colBtm[2];
            Blob<T> blobTop = colTop[0];

            op_fwd(OP.SUB, blobPre, blobMu, m_blobMues, blobPre.channels, blobPre.num, blobPre.count(2), 1, blobMu.channels);
            op_fwd(OP.MUL, m_blobMues, blobSigma, m_blobX, m_blobMues.channels, m_blobMues.num, m_blobMues.count(2), 1, blobSigma.channels);

            addBtmTop(m_blobX, blobTop);
            m_sigmoid.Forward(m_colBtm, m_colTop);
        }

        private void map_inputs_fwd(Blob<T> btm, Blob<T> top)
        {
            op_fwd(OP.MUL, btm, blobs[(int)WEIGHT.INPUT_WT], top);
            op_fwd(OP.ADD, top, blobs[(int)WEIGHT.INPUT_BIAS], top);
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
            addBtmTop(blobInputs, m_blobSensoryActivationW);
            m_colBtm.Add(blobs[(int)WEIGHT.SENSORY_MU]);
            m_colBtm.Add(blobs[(int)WEIGHT.SENSORY_SIGMA]);
            sigmoid_fwd(m_colBtm, m_colTop);

            op_fwd(OP.MUL, m_blobSensoryActivationW, blobs[(int)WEIGHT.SENSORY_W], m_blobSensoryActivationW, nC, nN, nSD, 1, nSD);
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
                // Compute the W activation
                addBtmTop(m_blobVPre, m_blobActivationW);
                m_colBtm.Add(blobs[(int)WEIGHT.MU]);
                m_colBtm.Add(blobs[(int)WEIGHT.SIGMA]);
                sigmoid_fwd(m_colBtm, m_colTop);
                op_fwd(OP.MUL, m_blobActivationW, blobs[(int)WEIGHT.W], m_blobActivationW, nSD, nN, nSD, 1, nSD);

                // Compute the Rev activation
                op_fwd(OP.MUL, m_blobActivationW, blobs[(int)WEIGHT.EREV], m_blobActivationRev, nSD, nN, nSD, 1, nSD);

                // Reduce over dim=1 (source neurons)
                m_cuda.channel_sum(nCount, m_blobActivationRev.num, m_blobActivationRev.channels, m_blobActivationRev.count(2), m_blobActivationRev.gpu_data, m_blobNumeratorW.mutable_gpu_data, true);
                m_cuda.channel_sum(nCount, m_blobActivationW.num, m_blobActivationW.channels, m_blobActivationW.count(2), m_blobActivationW.gpu_data, m_blobDenominatorW.mutable_gpu_data, true);
                // Add sensory input
                op_fwd(OP.ADD, m_blobNumeratorW, m_blobSensoryNumeratorW, m_blobNumeratorW);
                op_fwd(OP.ADD, m_blobDenominatorW, m_blobSensoryDenominatorW, m_blobDenominatorW);

                // Compute the numerator
                op_fwd(OP.MUL, m_blobCmt, m_blobVPre, m_blobNumerator1);
                op_fwd(OP.MUL, blobs[(int)WEIGHT.GLEAK], blobs[(int)WEIGHT.VLEAK], m_blobNumerator2, nSD, 1, 1, 1, 1);
                op_fwd(OP.ADD, m_blobNumerator1, m_blobNumerator2, m_blobNumerator, nSD, nN, 1, 1, 1);
                op_fwd(OP.ADD, m_blobNumerator, m_blobNumeratorW, m_blobNumerator);

                // Compute the denominator
                op_fwd(OP.ADD, m_blobCmt, blobs[(int)WEIGHT.GLEAK], m_blobDenominator, nSD, nN, 1, 1, 1);
                op_fwd(OP.ADD, m_blobDenominator, m_blobDenominatorW, m_blobDenominator);
                m_blobDenominator.add_scalar(m_param.ltc_unit_param.epsilon);

                // Compute the output
                op_fwd(OP.DIV, m_blobNumerator, m_blobDenominator, m_blobVPre);
            }

            blobTop.CopyFrom(m_blobVPre);
        }

        /// <summary>
        /// WORK IN PROGRESS Computes the error gradient w.r.t. the LtcUnit value inputs.
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
        }
    }
}
