using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

// WORK IN PROGRESS
namespace MyCaffe.layers.tft
{
    /// <summary>
    /// The VarSetNetLayer implements the Variable Selection Network
    /// </summary>
    /// <remarks>
    /// The VSN enables instance-wise variable selection and is applied to both the static covariates and time-dependent covariates as the
    /// specific contribution of each input to the output is typically unknown.  The VSN provides insights into which variables contribute
    /// the most for the prediction problem and allows the model to remove unnecessarily noisy inputs which could negatively impact the
    /// performance.
    /// 
    /// @see [Github - PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch) by Playtika Research, 2021.
    /// @see [Github - PlaytikaOSS/tft-torch tft.py](https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py#L149) by Playtika Research, 2021.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class VarSetNetLayer<T> : Layer<T>
    {
        Layer<T> m_grnFlatten;
        Blob<T> m_blobSparseWts;
        Layer<T> m_softmax;
        Blob<T> m_blobSparseWtsSmx;
        Blob<T> m_blobGrn1;
        Blob<T> m_blobProcessedInputs;
        List<Layer<T>> m_rgSingleVarGrn = new List<Layer<T>>();
        BlobCollection<T> m_colSingleVarGrn = new BlobCollection<T>();
        BlobCollection<T> m_colTop = new BlobCollection<T>();
        BlobCollection<T> m_colBtm = new BlobCollection<T>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public VarSetNetLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.VARSELNET;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobSparseWts);
            dispose(ref m_blobSparseWtsSmx);
            dispose(ref m_blobGrn1);
            dispose(ref m_blobProcessedInputs);

            if (m_colSingleVarGrn != null)
            {
                m_colSingleVarGrn.Dispose();
                m_colSingleVarGrn = null;
            }

            dispose(ref m_grnFlatten);

            if (m_rgSingleVarGrn != null)
            {
                foreach (Layer<T> layer in m_rgSingleVarGrn)
                {
                    layer.Dispose();
                }
                m_rgSingleVarGrn = null;
            }
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;
        }

        /// <summary>
        /// Returns the min number of required bottom (input) Blobs: flattened_embedding
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the min number of required bottom (input) Blobs: flattened_embedding, context
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: outputs_sum, sparse_wts
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 2; }
        }

        private void addBtmTop(Blob<T> btm, Blob<T> top)
        {
            m_colBtm.Clear();
            m_colBtm.Add(btm);
            m_colTop.Clear();
            m_colTop.Add(top);
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // This GRN is applied on the flat concatenation of the input representation (all inputs together),
            // possibly provided with context information.
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRN);
            p.grn_param.axis = m_param.varselnet_param.axis;
            p.grn_param.batch_first = m_param.varselnet_param.batch_first;
            p.grn_param.bias_filler = m_param.varselnet_param.bias_filler;
            p.grn_param.weight_filler = m_param.varselnet_param.weight_filler;
            p.grn_param.input_dim = m_param.varselnet_param.input_dim;
            p.grn_param.hidden_dim = m_param.varselnet_param.hidden_dim;
            p.grn_param.output_dim = m_param.varselnet_param.hidden_dim;
            m_grnFlatten = Layer<T>.Create(m_cuda, m_log, p, null);
            m_blobSparseWts = new Blob<T>(m_cuda, m_log);

            addBtmTop(colBottom[0], m_blobSparseWts);
            if (colBottom.Count > 1)
                m_colBtm.Add(colBottom[1]);
            m_grnFlatten.Setup(m_colBtm, m_colTop);

            // Activation for transforming the GRN output to weights.
            p = new LayerParameter(LayerParameter.LayerType.SOFTMAX);
            p.softmax_param.axis = m_param.varselnet_param.axis;
            p.softmax_param.engine = EngineParameter.Engine.DEFAULT;
            m_softmax = Layer<T>.Create(m_cuda, m_log, p, null);
            m_blobSparseWtsSmx = new Blob<T>(m_cuda, m_log);

            addBtmTop(m_blobSparseWts, m_blobSparseWtsSmx);
            m_softmax.Setup(m_colBtm, m_colTop);

            // Each input variable (after transformation into its wide represenation) goes through its own GRN
            m_blobGrn1 = new Blob<T>(m_cuda, m_log);
            List<int> rgShape = new List<int>();
            rgShape.Add(colBottom[0].num);
            rgShape.Add(colBottom[0].channels / m_param.varselnet_param.num_inputs);
            m_blobGrn1.Reshape(rgShape);

            for (int i = 0; i < m_param.varselnet_param.num_inputs; i++)
            {
                p = new LayerParameter(LayerParameter.LayerType.GRN);
                p.grn_param.axis = m_param.varselnet_param.axis;
                p.grn_param.batch_first = m_param.varselnet_param.batch_first;
                p.grn_param.bias_filler = m_param.varselnet_param.bias_filler;
                p.grn_param.weight_filler = m_param.varselnet_param.weight_filler;
                p.grn_param.input_dim = m_param.varselnet_param.input_dim;
                p.grn_param.hidden_dim = m_param.varselnet_param.hidden_dim;
                p.grn_param.output_dim = m_param.varselnet_param.hidden_dim;
                Layer<T> grn = Layer<T>.Create(m_cuda, m_log, p, null);

                Blob<T> blobGrn = new Blob<T>(m_cuda, m_log);
                blobGrn.ReshapeLike(m_blobGrn1);

                m_rgSingleVarGrn.Add(grn);
                m_colSingleVarGrn.Add(blobGrn);

                addBtmTop(m_blobGrn1, m_colSingleVarGrn[i]);
                m_rgSingleVarGrn[i].Setup(m_colBtm, m_colTop);
            }

            m_blobProcessedInputs.ReshapeLike(colBottom[0]);
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            addBtmTop(colBottom[0], m_blobSparseWts);
            if (colBottom.Count > 1)
                m_colBtm.Add(colBottom[1]);
            m_grnFlatten.Reshape(m_colBtm, m_colTop);

            addBtmTop(m_blobSparseWts, m_blobSparseWtsSmx);
            m_softmax.Setup(m_colBtm, m_colTop);

            List<int> rgShape = new List<int>();
            rgShape.Add(colBottom[0].num);
            rgShape.Add(colBottom[0].channels / m_param.varselnet_param.num_inputs);
            m_blobGrn1.Reshape(rgShape);

            for (int i = 0; i < m_param.varselnet_param.num_inputs; i++)
            {
                m_colSingleVarGrn[i].ReshapeLike(m_blobGrn1);

                addBtmTop(m_blobGrn1, m_colSingleVarGrn[i]);
                m_rgSingleVarGrn[i].Setup(m_colBtm, m_colTop);
            }

            m_blobProcessedInputs.ReshapeLike(colBottom[0]);
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
            addBtmTop(colBottom[0], m_blobSparseWts);
            if (colBottom.Count > 1)
                m_colBtm.Add(colBottom[1]);
            m_grnFlatten.Forward(m_colBtm, m_colTop);

            addBtmTop(m_blobSparseWts, m_blobSparseWtsSmx);
            m_softmax.Forward(m_colBtm, m_colTop);

            for (int i = 0; i < m_param.varselnet_param.num_inputs; i++)
            {
                addBtmTop(m_blobGrn1, m_colSingleVarGrn[i]);
                m_rgSingleVarGrn[i].Forward(m_colBtm, m_colTop);
            }
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
        }
    }
}
