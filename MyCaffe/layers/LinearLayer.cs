using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;
using MyCaffe.fused_ops;
using System.Diagnostics;

namespace MyCaffe.layers
{
    /// <summary>
    /// The LinearLayer, also know as a 'fully-connected' layer, computes the inner product
    /// with a set of learned weights, and (optionally) adds biases and is designed to 
    /// produce campatible results with the PyTorch Linear layer.
    /// 
    /// This layer is initialized with the MyCaffe.param.LinearParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class LinearLayer<T> : Layer<T>
    {
        int m_nM;
        int m_nK;
        int m_nN;
        bool m_bBiasTerm;
        bool m_bTranspose;
        Blob<T> m_blobBiasMultiplier;
        Blob<T> m_blobWork = null;
        MatMulOp<T> m_matmul = null;
        List<int> m_rgOriginalShape;
        List<int> m_rgBtmShape = new List<int>(4);
        List<int> m_rgTopShape = new List<int>(4);
        List<int> m_rgShape = new List<int>() { 1, 1 };

        /// <summary>
        /// The LinearLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter linear_param, with options:
        ///   - num_output. The number of outputs.
        ///   
        ///   - bias_term (/b optional, default = true).  Whether or not to include bias.
        ///   
        ///   - weight_filler (/b optional, default = "gaussian").  The filler used to initialize the weights.
        ///   
        ///   - bias_filler (/b optional, default = "constant, 1.0").  The filler used to initialize the bias.
        ///   
        ///   - axis (/b optional, default = 1). The axis to be lumped into a single inner-product computation.
        ///   
        ///   - transpose (/b optional, default = false).  Whether or not to transpose the weight matrix or not.
        /// </param>
        public LinearLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.LINEAR;
            m_blobBiasMultiplier = new Blob<T>(cuda, log);
            m_blobBiasMultiplier.Name = m_param.name + ".biasmult";
            m_bTranspose = p.linear_param.transpose;

            if (m_bTranspose)
                m_blobWork = createIntraLayerBlob("ip.work", false);

            setup_internal_blobs(m_colInternalBlobs);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobBiasMultiplier);
            dispose(ref m_blobWork);

            if (m_matmul != null)
            {
                m_matmul.Dispose();
                m_matmul = null;
            }

            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            if (m_param.linear_param.bias_term)
                col.Add(m_blobBiasMultiplier);
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input.
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input.
        /// </summary>
        /// <remarks>
        /// When specified, the input_dim overrides the m_param.linear_param.num_output.
        /// </remarks>
        public override int MaxBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: ip
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

            if (target == WEIGHT_TARGET.BOTH || target == WEIGHT_TARGET.WEIGHTS)
            {
                Filler<T> weight_filler = Filler<T>.Create(m_cuda, m_log, m_param.linear_param.weight_filler);
                weight_filler.Fill(m_colBlobs[0]);
            }

            if (m_param.linear_param.bias_term && m_colBlobs.Count > 1 && (target == WEIGHT_TARGET.BOTH || target == WEIGHT_TARGET.BIAS))
            {
                Filler<T> bias_filler = Filler<T>.Create(m_cuda, m_log, m_param.linear_param.bias_filler);
                bias_filler.Fill(m_colBlobs[1]);
            }

            return true;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (colBottom.Count > 1)
                m_param.linear_param.num_output = (uint)convertF(colBottom[1].GetData(0));

            int nNumOutput = (int)m_param.linear_param.num_output;
            m_bBiasTerm = m_param.linear_param.bias_term;
            m_nN = nNumOutput;

            int nAxis = colBottom[0].CanonicalAxisIndex(m_param.linear_param.axis);

            List<int> rgShape = colBottom[0].shape();
            int nShapeCount = rgShape.Count;
            for (int i = nShapeCount; i <= nAxis; i++)
            {
                rgShape.Add(1);
            }

            if (nShapeCount != rgShape.Count)
                colBottom[0].Reshape(rgShape);

            // Dimensions starting from 'axis' are 'flattened' into a single
            // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
            // and axis == 1, N inner products with dimension CHW are preformed.
            m_nK = colBottom[0].count(nAxis);

            // Check if we need to set up the weights.
            if (m_colBlobs.Count > 0)
            {
                m_log.WriteLine("Skipping parameter initialization.");
            }
            else
            {
                // Initialize the weight.
                List<int> rgWeightShape = Utility.Create<int>(2, 0);

                if (m_bTranspose)
                {
                    rgWeightShape[0] = m_nK;
                    rgWeightShape[1] = m_nN;
                }
                else
                {
                    rgWeightShape[0] = m_nN;
                    rgWeightShape[1] = m_nK;
                }

                Blob<T> blobWeight = new Blob<T>(m_cuda, m_log, !layer_param.freeze_learning);
                blobWeight.Name = m_param.name + " weights";
                blobWeight.blob_type = BLOB_TYPE.IP_WEIGHT;

                if (!shareParameter(blobWeight, rgWeightShape, true))
                {
                    blobWeight.Reshape(rgWeightShape);
                    Filler<T> weight_filler = Filler<T>.Create(m_cuda, m_log, m_param.linear_param.weight_filler);
                    weight_filler.Fill(blobWeight);
                }
                m_colBlobs.Add(blobWeight);
                if (m_blobWork != null)
                    m_blobWork.ReshapeLike(blobWeight);

                // If necessary, initialize and fill the bias term.
                if (m_bBiasTerm)
                {
                    List<int> rgBiasShape = Utility.Create<int>(1, m_nN);
                    Blob<T> blobBias = new Blob<T>(m_cuda, m_log, !layer_param.freeze_learning);
                    blobBias.Name = m_param.name + " bias";
                    blobBias.blob_type = BLOB_TYPE.IP_WEIGHT;

                    if (!shareParameter(blobBias, rgBiasShape, true))
                    {
                        blobBias.Reshape(rgBiasShape);
                        Filler<T> bias_filler = Filler<T>.Create(m_cuda, m_log, m_param.linear_param.bias_filler);
                        bias_filler.Fill(blobBias);
                    }
                    m_colBlobs.Add(blobBias);
                }
            }

            m_rgbParamPropagateDown = new DictionaryMap<bool>(m_colBlobs.Count, true);

            if (m_weightAdapter != null)
                m_weightAdapter.Setup(layer_param, m_colBlobs[0]);

            copyShape(colBottom[0], m_rgBtmShape);

            // The first 'axis' dimensions are independent of inner products; the total
            // number of these is M_, the product over these dimensions.
            m_nM = colBottom[0].count(0, nAxis);

            m_rgShape[0] = m_nM;
            m_rgShape[1] = m_nK;
            colBottom[0].Reshape(m_rgShape);

            m_matmul = new MatMulOp<T>(m_cuda, m_log, 2, m_param.linear_param.enable_fused_comp);
            m_matmul.Create(colBottom[0], m_colBlobs[0], colTop[0], false, m_bTranspose);

            colBottom[0].Reshape(m_rgBtmShape);
            Reshape(colBottom, colTop);

            m_rgOriginalShape = Utility.Clone<int>(colBottom[0].shape());
        }

        private void copyShape(Blob<T> b, List<int> rgShape)
        {
            rgShape.Clear();
            rgShape.AddRange(b.shape());
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            copyShape(colBottom[0], m_rgBtmShape);
            copyShape(colTop[0], m_rgTopShape);

            if (colBottom[0].CompareShape(m_rgOriginalShape))
                return;

            // The first 'axis' dimensions are independent of inner products; the total
            // number of these is M_, the product over these dimensions.
            int nAxis = colBottom[0].CanonicalAxisIndex(m_param.linear_param.axis);
            m_nM = colBottom[0].count(0, nAxis);

            List<int> rgShape = new List<int>(colBottom[0].shape());

            // Figure out the dimensions
            while (rgShape.Count <= m_param.linear_param.axis)
            {
                rgShape.Add(1);
            }

            colBottom[0].Reshape(rgShape);

            int nNewK = colBottom[0].count(nAxis);

            m_log.CHECK_EQ(m_nK, nNewK, "Input size incompatible with inner product parameters.");

            // The top shape will be the bottom shape with the flattened axes dropped,
            // and replaced by a single axis with dimensions num_output (N_).
            List<int> rgTopShape = Utility.Clone<int>(colBottom[0].shape(), nAxis + 1);
            rgTopShape[nAxis] = m_nN;

            colTop[0].Reshape(rgTopShape);
            if (m_param.linear_param.output_contains_predictions)
                colTop[0].blob_type = BLOB_TYPE.PREDICTION;

            // Set up the bias multiplier
            if (m_bBiasTerm)
            {
                List<int> rgBiasShape = Utility.Create<int>(1, m_nM);
                m_blobBiasMultiplier.Reshape(rgBiasShape);
                m_blobBiasMultiplier.SetData(1.0);
            }

            if (m_weightAdapter != null)
                m_weightAdapter.Reshape(m_colBlobs[0]);

            m_rgShape[0] = m_nM;
            m_rgShape[1] = m_nK;
            colBottom[0].Reshape(m_rgShape);

            m_matmul.Reshape(colBottom[0], m_colBlobs[0], colTop[0], false, m_bTranspose);

            colBottom[0].Reshape(m_rgBtmShape);
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times K \times 1 \times 1) @f$
        ///     the computed inner product with the weights, where
        ///     @f$ K @f$ equals <i>num_output</i>.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            long hWeight = m_colBlobs[0].gpu_data;
            long hBias = (m_bBiasTerm) ? m_colBlobs[1].gpu_data : 0;

            if (m_weightAdapter != null)
                hWeight = m_weightAdapter.Forward(m_colBlobs[0]);

            m_matmul.Run(hBottomData, hWeight, hTopData);

            if (m_bBiasTerm)
                m_cuda.gemm(false, false, m_nM, m_nN, 1, m_tOne, m_blobBiasMultiplier.gpu_data, hBias, m_tOne, hTopData);
        }

        /// <summary>
        /// Computes the Linear loss error gradient w.r.t the outputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///   -# @f$ (N \times K \times 1 \times 1) @f$, where @f$ K @f$ is equal to <i>num_output</i>.
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            Blob<T> blobWeight = m_colBlobs[0];

            if (m_weightAdapter != null)
                blobWeight = m_weightAdapter.Weight;

            colBottom[0].Reshape(1, 1, m_nM, m_nK);
            colTop[0].Reshape(1, 1, m_nM, m_nN);

            colTop[0].MatMulGrad(colBottom[0], blobWeight);

            colBottom[0].Reshape(m_rgBtmShape);
            colTop[0].Reshape(m_rgTopShape);

            if (m_bTranspose)
            {
                m_cuda.transposeHW(1, 1, blobWeight.num, blobWeight.channels, blobWeight.gpu_diff, m_blobWork.mutable_gpu_data);
                m_cuda.copy(blobWeight.count(), m_blobWork.gpu_data, blobWeight.mutable_gpu_diff);
            }

            if (m_weightAdapter != null)
                m_weightAdapter.Backward(colTop, colBottom, blobWeight);
        }
    }
}
