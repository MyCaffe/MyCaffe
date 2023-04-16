using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http.Headers;
using System.Reflection;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

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
    /// @see [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363) by Bryan Lim, Sercan O. Arik, Nicolas Loeff, and Tomas Pfister, 2019, arXiv 1912.09363
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
        Layer<T> m_transpose;
        Blob<T> m_blobSparseWtsSmxT;
        Blob<T> m_blobGrn1;
        Blob<T> m_blobProcessedInputs;
        Blob<T> m_blobProcessedInputs1;
        Blob<T> m_blobBtm;
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
            dispose(ref m_blobSparseWtsSmxT);
            dispose(ref m_blobGrn1);
            dispose(ref m_blobProcessedInputs);
            dispose(ref m_blobProcessedInputs1);
            dispose(ref m_blobBtm);

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
            List<int> rgShape = new List<int>();

            m_blobBtm = new Blob<T>(m_cuda, m_log);

            // This GRN is applied on the flat concatenation of the input representation (all inputs together),
            // possibly provided with context information.
            LayerParameter p = new LayerParameter(LayerParameter.LayerType.GRN, m_param.name + ".flat");
            p.grn_param.axis = m_param.varselnet_param.axis;
            p.grn_param.batch_first = m_param.varselnet_param.batch_first;
            p.grn_param.bias_filler = m_param.varselnet_param.bias_filler;
            p.grn_param.weight_filler = m_param.varselnet_param.weight_filler;
            p.grn_param.input_dim = m_param.varselnet_param.num_inputs * m_param.varselnet_param.input_dim;
            p.grn_param.hidden_dim = m_param.varselnet_param.hidden_dim;
            p.grn_param.output_dim = m_param.varselnet_param.num_inputs;
            p.grn_param.context_dim = m_param.varselnet_param.context_dim;
            m_grnFlatten = Layer<T>.Create(m_cuda, m_log, p, null);
            m_blobSparseWts = new Blob<T>(m_cuda, m_log);

            addBtmTop(colBottom[0], m_blobSparseWts);
            if (colBottom.Count > 1)
                m_colBtm.Add(colBottom[1]);
            m_grnFlatten.Setup(m_colBtm, m_colTop);
            blobs.Add(m_grnFlatten.blobs);

            // Activation for transforming the GRN output to weights.
            p = new LayerParameter(LayerParameter.LayerType.SOFTMAX, m_param.name + ".smx");
            p.softmax_param.axis = m_param.varselnet_param.axis;
            p.softmax_param.engine = EngineParameter.Engine.DEFAULT;
            m_softmax = Layer<T>.Create(m_cuda, m_log, p, null);
            m_blobSparseWtsSmx = new Blob<T>(m_cuda, m_log);

            addBtmTop(m_blobSparseWts, m_blobSparseWtsSmx);
            m_softmax.Setup(m_colBtm, m_colTop);

            rgShape = Utility.Clone<int>(m_blobSparseWtsSmx.shape());
            rgShape.Add(1);
            rgShape.Add(1);
            m_blobSparseWtsSmx.Reshape(rgShape);

            // Setup transpose applied to smx.
            p = new LayerParameter(LayerParameter.LayerType.TRANSPOSE, m_param.name + ".trfm");
            p.transpose_param.dim[1] = 2;
            p.transpose_param.dim[2] = 1;
            m_transpose = Layer<T>.Create(m_cuda, m_log, p, null);
            m_blobSparseWtsSmxT = new Blob<T>(m_cuda, m_log);

            addBtmTop(m_blobSparseWtsSmx, m_blobSparseWtsSmxT);
            m_transpose.Setup(m_colBtm, m_colTop);

            // Each input variable (after transformation into its wide represenation) goes through its own GRN
            m_blobGrn1 = new Blob<T>(m_cuda, m_log);
            rgShape.Clear();
            rgShape.Add(colBottom[0].num);
            rgShape.Add(colBottom[0].channels / m_param.varselnet_param.num_inputs);
            m_blobGrn1.Reshape(rgShape);

            for (int i = 0; i < m_param.varselnet_param.num_inputs; i++)
            {
                p = new LayerParameter(LayerParameter.LayerType.GRN, m_param.name + ".grn" + i.ToString());
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
                blobs.Add(grn.blobs);
            }

            rgShape.Clear();
            rgShape.Add(colBottom[0].num);
            rgShape.Add(colBottom[0].channels / m_param.varselnet_param.num_inputs);
            rgShape.Add(m_param.varselnet_param.num_inputs);
            m_blobProcessedInputs = new Blob<T>(m_cuda, m_log);
            m_blobProcessedInputs.Reshape(rgShape);
            m_blobProcessedInputs1 = new Blob<T>(m_cuda, m_log);
            m_blobProcessedInputs1.Reshape(rgShape);
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            List<int> rgShape;

            m_blobBtm.ReshapeLike(colBottom[0]);

            addBtmTop(colBottom[0], m_blobSparseWts);
            if (colBottom.Count > 1)
                m_colBtm.Add(colBottom[1]);
            m_grnFlatten.Reshape(m_colBtm, m_colTop);

            addBtmTop(m_blobSparseWts, m_blobSparseWtsSmx);
            m_softmax.Reshape(m_colBtm, m_colTop);

            rgShape = Utility.Clone<int>(m_blobSparseWtsSmx.shape());
            rgShape.Add(1);
            rgShape.Add(1);
            m_blobSparseWtsSmx.Reshape(rgShape);

            addBtmTop(m_blobSparseWtsSmx, m_blobSparseWtsSmxT);
            m_transpose.Reshape(m_colBtm, m_colTop);

            rgShape.Clear();
            rgShape.Add(colBottom[0].num);
            rgShape.Add(colBottom[0].channels / m_param.varselnet_param.num_inputs);
            m_blobGrn1.Reshape(rgShape);

            for (int i = 0; i < m_param.varselnet_param.num_inputs; i++)
            {
                m_colSingleVarGrn[i].ReshapeLike(m_blobGrn1);

                addBtmTop(m_blobGrn1, m_colSingleVarGrn[i]);
                m_rgSingleVarGrn[i].Reshape(m_colBtm, m_colTop);
            }

            rgShape.Clear();
            rgShape.Add(colBottom[0].num);
            rgShape.Add(colBottom[0].channels / m_param.varselnet_param.num_inputs);
            rgShape.Add(m_param.varselnet_param.num_inputs);
            m_blobProcessedInputs.Reshape(rgShape);
            m_blobProcessedInputs1.Reshape(rgShape);

            rgShape.RemoveAt(rgShape.Count - 1);
            colTop[0].Reshape(rgShape);
            colTop[1].ReshapeLike(m_blobSparseWts);
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
            m_blobBtm.CopyFrom(colBottom[0]);

            // Infer variable selection weights using flattened embedding run through GRN.  The flattened embedding
            // should have shape [(num_samples * num_temporal_steps) x (num_inputs x input_dim)] where the 
            // input_dim represents the model_dim or the state_dim.  With static variable selection, num_temporal_Steps
            // is set to 1.
            addBtmTop(colBottom[0], m_blobSparseWts);
            if (colBottom.Count > 1)
                m_colBtm.Add(colBottom[1]);
            m_grnFlatten.Forward(m_colBtm, m_colTop);

            // Sparse weights are of shape [(num_samples * num_temporal_steps) x num_inputs x 1]
            addBtmTop(m_blobSparseWts, m_blobSparseWtsSmx);
            m_softmax.Forward(m_colBtm, m_colTop);

            // Unsqueeze by 2
            List<int> rgShape = Utility.Clone<int>(m_blobSparseWtsSmx.shape());
            rgShape.Add(1);
            rgShape.Add(1);
            m_blobSparseWtsSmx.Reshape(rgShape);

            // Weigh the processed inputs with the weights viewed as [(num_samples * num_temporal_steps) x 1 x num_inputs]
            // so that the weight given to each variable (for each time-step/observation) multiplies the entire state
            // vector representing the specific input variable on the specific time-step
            addBtmTop(m_blobSparseWtsSmx, m_blobSparseWtsSmxT);
            m_transpose.Forward(m_colBtm, m_colTop);

            // Before weighting the variables, a GRN is applied ot each transformed input.
            for (int i = 0; i < m_param.varselnet_param.num_inputs; i++)
            {
                // Copy the variable specific data to the GRN input.
                m_cuda.channel_copy(m_blobGrn1.count(), m_blobGrn1.num, 1, m_param.varselnet_param.num_inputs, m_blobGrn1.channels, i, colBottom[0].gpu_data, m_blobGrn1.mutable_gpu_data, DIR.FWD);

                // Each element in the resulting list is of size [(num_samples * num_temporal_steps) x state_size],
                // and each element corresponds to a single input variable.
                addBtmTop(m_blobGrn1, m_colSingleVarGrn[i]);
                m_rgSingleVarGrn[i].Forward(m_colBtm, m_colTop);

                // Combine the outputs of the state var GRNs along an additional axis with 
                // dimension [(num_samples * num_temporal_steps) x state_size x num_inputs]
                m_cuda.channel_copy(m_blobGrn1.count(), m_blobGrn1.num, m_blobGrn1.channels, m_param.varselnet_param.num_inputs, 1, i, m_blobProcessedInputs.mutable_gpu_data, m_colSingleVarGrn[i].gpu_data, DIR.BWD);
            }

            // Apply the transposed smx weightings to the processed inputs
            int nInnerNum = m_blobProcessedInputs.count(2);
            m_cuda.channel_mulv(m_blobProcessedInputs.count(), m_blobProcessedInputs.num, m_blobProcessedInputs.channels, nInnerNum, m_blobProcessedInputs.gpu_data, m_blobSparseWtsSmxT.gpu_data, m_blobProcessedInputs1.mutable_gpu_data);

            // Sum up the weights to create a weighted sum representation of width state_size for each time-step and
            // dimension [(num_samples * num_temporal_steps) x state_size x num_inputs]
            m_cuda.channel_sum(m_blobProcessedInputs1.count(), m_blobProcessedInputs1.num, m_blobProcessedInputs1.channels, nInnerNum, m_blobProcessedInputs1.gpu_data, colTop[0].mutable_gpu_data, false);
            colTop[1].CopyFrom(m_blobSparseWts);
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
            m_blobSparseWts.CopyFrom(colTop[1], true);

            // Expand the top(0) diff to each channel in the processed inputs.
            int nInnerNum = m_blobProcessedInputs.count(2);
            m_cuda.channel_fillfrom(m_blobProcessedInputs1.count(), m_blobProcessedInputs1.num, m_blobProcessedInputs1.channels, nInnerNum, colTop[0].gpu_diff, m_blobProcessedInputs1.mutable_gpu_diff, DIR.FWD);

            // Apply the transposed smx weightings to the processed inputs.
            m_cuda.channel_mulv(m_blobProcessedInputs.count(), m_blobProcessedInputs.num, m_blobProcessedInputs.channels, nInnerNum, m_blobProcessedInputs1.gpu_diff, m_blobSparseWtsSmxT.gpu_data, m_blobProcessedInputs.mutable_gpu_diff);

            // GRN is applied ot each transformed input.
            for (int i = 0; i < m_param.varselnet_param.num_inputs; i++)
            {
                // Combine the outputs of the state var GRNs along an additional axis with 
                // dimension [(num_samples * num_temporal_steps) x state_size x num_inputs]
                m_cuda.channel_copy(m_blobGrn1.count(), m_blobGrn1.num, m_blobGrn1.channels, m_param.varselnet_param.num_inputs, 1, i, m_blobProcessedInputs.mutable_gpu_diff, m_colSingleVarGrn[i].gpu_diff, DIR.FWD);

                // Each element in the resulting list is of size [(num_samples * num_temporal_steps) x state_size],
                // and each element corresponds to a single input variable.
                addBtmTop(m_blobGrn1, m_colSingleVarGrn[i]);
                m_rgSingleVarGrn[i].Backward(m_colTop, rgbPropagateDown, m_colBtm);

                // Copy the variable specific data to the GRN input.
                m_cuda.channel_copy(m_blobGrn1.count(), m_blobGrn1.num, 1, m_param.varselnet_param.num_inputs, m_blobGrn1.channels, i, m_blobBtm.gpu_diff, m_blobGrn1.mutable_gpu_diff, DIR.BWD);
            }

            // Weigh the processed inputs with the weights viewed as [(num_samples * num_temporal_steps) x 1 x num_inputs]
            // so that the weight given to each variable (for each time-step/observation) multiplies the entire state
            // vector representing the specific input variable on the specific time-step
            addBtmTop(m_blobSparseWtsSmx, m_blobSparseWtsSmxT);
            m_transpose.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            // Sparse weights are of shape [(num_samples * num_temporal_steps) x num_inputs x 1]
            addBtmTop(m_blobSparseWts, m_blobSparseWtsSmx);
            m_softmax.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            // Infer variable selection weights using flattened embedding run through GRN.  The flattened embedding
            // should have shape [(num_samples * num_temporal_steps) x (num_inputs x input_dim)] where the 
            // input_dim represents the model_dim or the state_dim.  With static variable selection, num_temporal_Steps
            // is set to 1.
            addBtmTop(colBottom[0], m_blobSparseWts);
            if (colBottom.Count > 1)
                m_colBtm.Add(colBottom[1]);
            m_grnFlatten.Backward(m_colTop, rgbPropagateDown, m_colBtm);

            // Add gradient accumulation from individual variable GRN's.
            m_cuda.add(colBottom[0].count(), colBottom[0].gpu_data, m_blobBtm.gpu_data, colBottom[0].mutable_gpu_data);                        
        }
    }
}
