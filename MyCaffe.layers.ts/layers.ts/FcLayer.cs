using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.ts
{
    /// <summary>
    /// The FcLayer performs the FC layer functionality used by the N-HiTS model.
    /// </summary>
    /// <remarks>
    /// This layer performs the Linear projection, Activation, Normalization, and Dropout operations.
    /// 
    /// @see [Understanding N-HiTS Time Series Prediction](https://www.signalpop.com/2024/05/29/n-hits/) by Brown, 2024, SignalPop
    /// @see [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting](https://arxiv.org/abs/2201.12886) by Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, and Artur Dubrawski, 2022, arXiv:2201.12886.
    /// @see [Darts: User-Friendly Modern Machine Learning for Time Series](https://jmlr.org/papers/v23/21-1177.html) by Julien Herzen, Francesco Lässig, Samuele Giuliano Piazzetta, Thomas Neuer, Léo Tafti, Guillaume Raille, Tomas Van Pottelbergh, Marek Pasieka, Andrzej Skrodzki, Nicolas Huguenin, Maxime Dumonal, Jan Kościsz, Dennis Bader, Frédérick Gusset, Mounir Benheddi, Camila Williamson, Michal Kosinski, Matej Petrik, and Gaël Grosch, 2022, JMLR
    /// @see [Github - unit8co/darts](https://github.com/unit8co/darts) by unit8co, 2022, GitHub.
    /// 
    /// WORK IN PROGRESS.
    /// </remarks>
    public class FcLayer<T> : Layer<T>
    {
        Blob<T> m_blobIp = null;
        Layer<T> m_layerIp = null;
        Blob<T> m_blobAct = null;
        Layer<T> m_layerAct = null;
        Blob<T> m_blobNorm = null;
        Layer<T> m_layerNorm = null;
        Blob<T> m_blobDrop = null;
        Layer<T> m_layerDrop = null;
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public FcLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.FC;

            m_blobIp = new Blob<T>(cuda, log);
            m_blobIp.Name = m_param.name + " ip";
            m_blobAct = new Blob<T>(cuda, log);
            m_blobAct.Name = m_param.name + " act";

            if (p.fc_param.enable_normalization)
            {
                m_blobNorm = new Blob<T>(cuda, log);
                m_blobNorm.Name = m_param.name + " norm";
            }

            if (p.fc_param.dropout_ratio > 0)
            {
                m_blobDrop = new Blob<T>(cuda, log);
                m_blobDrop.Name = m_param.name + " drop";
            }
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobIp);
            dispose(ref m_blobAct);
            dispose(ref m_blobNorm);
            dispose(ref m_blobDrop);
            dispose(ref m_layerDrop);
            dispose(ref m_layerNorm);
            dispose(ref m_layerAct);
            dispose(ref m_layerIp);
            base.dispose();
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

            col.Add(m_blobIp);
            col.Add(m_blobAct);
            if (m_blobNorm != null)
                col.Add(m_blobNorm);
            if (m_blobDrop != null)
                col.Add(m_blobDrop);
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: x
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: y
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            LayerParameter ip = new LayerParameter(LayerParameter.LayerType.INNERPRODUCT, "ip", m_phase);
            ip.inner_product_param.axis = m_param.fc_param.axis;
            ip.inner_product_param.num_output = (uint)m_param.fc_param.num_output;
            ip.inner_product_param.bias_term = m_param.fc_param.bias_term;
            m_layerIp = new InnerProductLayer<T>(m_cuda, m_log, ip);

            addBtmTop(colBottom[0], m_blobIp);
            m_layerIp.Setup(m_colBtm, m_colTop);

            LayerParameter act = null;

            switch (m_param.fc_param.activation)
            {
                case param.ts.FcParameter.ACTIVATION.RELU:
                    act = new LayerParameter(LayerParameter.LayerType.RELU, "act", m_phase);
                    break;

                case param.ts.FcParameter.ACTIVATION.PRELU:
                    act = new LayerParameter(LayerParameter.LayerType.PRELU, "act", m_phase);
                    break;

                case param.ts.FcParameter.ACTIVATION.ELU:
                    act = new LayerParameter(LayerParameter.LayerType.ELU, "act", m_phase);
                    break;

                case param.ts.FcParameter.ACTIVATION.SOFTPLUS:
                    act = new LayerParameter(LayerParameter.LayerType.SOFTPLUS, "act", m_phase);
                    break;

                case param.ts.FcParameter.ACTIVATION.TANH:
                    act = new LayerParameter(LayerParameter.LayerType.TANH, "act", m_phase);
                    break;

                case param.ts.FcParameter.ACTIVATION.SIGMOID:
                    act = new LayerParameter(LayerParameter.LayerType.SIGMOID, "act", m_phase);
                    break;

                case param.ts.FcParameter.ACTIVATION.GELU:
                    act = new LayerParameter(LayerParameter.LayerType.GELU, "act", m_phase);
                    break;

                case param.ts.FcParameter.ACTIVATION.SWISH:
                    act = new LayerParameter(LayerParameter.LayerType.SWISH, "act", m_phase);
                    break;

                case param.ts.FcParameter.ACTIVATION.MISH:
                    act = new LayerParameter(LayerParameter.LayerType.MISH, "act", m_phase);
                    break;

                default:
                    m_log.FAIL("Unknown activation type '" + m_param.fc_param.activation.ToString() + "'!");
                    break;
            }
            m_layerAct = new ReLULayer<T>(m_cuda, m_log, act);

            addBtmTop(m_blobIp, m_blobAct);
            m_layerAct.Setup(m_colBtm, m_colTop);

            Blob<T> blobBtm = m_blobAct;

            if (m_param.fc_param.enable_normalization)
            {
                LayerParameter norm = new LayerParameter(LayerParameter.LayerType.BATCHNORM, "norm", m_phase);
                m_layerNorm = new BatchNormLayer<T>(m_cuda, m_log, norm);
                addBtmTop(blobBtm, m_blobNorm);
                m_layerNorm.Setup(m_colBtm, m_colTop);
                blobBtm = m_blobNorm;
            }

            if (m_param.fc_param.dropout_ratio > 0)
            {
                LayerParameter dp = new LayerParameter(LayerParameter.LayerType.DROPOUT, "drop", m_phase);
                dp.dropout_param.dropout_ratio = m_param.fc_param.dropout_ratio;
                m_layerDrop = new DropoutLayer<T>(m_cuda, m_log, dp);
                addBtmTop(blobBtm, m_blobDrop);
                m_layerDrop.Setup(m_colBtm, m_colTop);
                blobBtm = m_blobDrop;
            }

            colTop[0].ReshapeLike(blobBtm);
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            addBtmTop(colBottom[0], m_blobIp);
            m_layerIp.Reshape(m_colBtm, m_colTop);

            addBtmTop(m_blobIp, m_blobAct);
            m_layerAct.Reshape(m_colBtm, m_colTop);

            Blob<T> blobBtm = m_blobAct;

            if (m_param.fc_param.enable_normalization)
            {
                addBtmTop(blobBtm, m_blobNorm);
                m_layerNorm.Reshape(m_colBtm, m_colTop);
                blobBtm = m_blobNorm;
            }

            if (m_param.fc_param.dropout_ratio > 0)
            {
                addBtmTop(blobBtm, m_blobDrop);
                m_layerDrop.Reshape(m_colBtm, m_colTop);
                blobBtm = m_blobDrop;
            }

            colTop[0].ReshapeLike(blobBtm);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times T \times H \times W) @f$ 
        ///     the numeric inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector)
        ///  -# @f$ (N \times T \times H \times W size) @f$
        ///     the computed outputs @f$ y @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            addBtmTop(colBottom[0], m_blobIp);
            m_layerIp.Forward(m_colBtm, m_colTop);

            addBtmTop(m_blobIp, m_blobAct);
            m_layerAct.Forward(m_colBtm, m_colTop);

            Blob<T> blobBtm = m_blobAct;

            if (m_param.fc_param.enable_normalization)
            {
                addBtmTop(blobBtm, m_blobNorm);
                m_layerNorm.Forward(m_colBtm, m_colTop);
                blobBtm = m_blobNorm;
            }

            if (m_param.fc_param.dropout_ratio > 0)
            {
                addBtmTop(blobBtm, m_blobDrop);
                m_layerDrop.Forward(m_colBtm, m_colTop);
                blobBtm = m_blobDrop;
            }

            colTop[0].CopyFrom(blobBtm);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times T \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times T \times H \times W) @f$
        ///     the inputs @f$ x @f$;  
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            bool bCopyTop = true;

            if (m_param.fc_param.dropout_ratio > 0)
            {
                Blob<T> blobBtm = (m_param.fc_param.enable_normalization) ? m_blobNorm : m_blobAct;
                if (bCopyTop)
                {
                    m_blobDrop.CopyFrom(colTop[0], true);
                    bCopyTop = false;
                }
                addBtmTop(blobBtm, m_blobDrop);
                m_layerDrop.Backward(m_colTop, rgbPropagateDown, m_colBtm);
            }

            if (m_param.fc_param.enable_normalization)
            {
                if (bCopyTop)
                {
                    m_blobNorm.CopyFrom(colTop[0], true);
                    bCopyTop = false;
                }
                addBtmTop(m_blobAct, m_blobNorm);
                m_layerNorm.Backward(m_colTop, rgbPropagateDown, m_colBtm);
            }

            if (bCopyTop)
            {
                m_blobAct.CopyFrom(colTop[0], true);
                bCopyTop = false;
            }

            addBtmTop(m_blobIp, m_blobAct);
            m_layerAct.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            addBtmTop(colBottom[0], m_blobIp);
            m_layerIp.Backward(m_colTop, rgbPropagateDown, m_colBtm);
        }
    }
}
