using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;

namespace MyCaffe.layers
{
    /// <summary>
    /// The InnerProductLayer, also know as a 'fully-connected' layer, computes the inner product
    /// with a set of learned weights, and (optionally) adds biases.
    /// This layer is initialized with the MyCaffe.param.InnerProductParameter.
    /// </summary>
    /// <remarks>
    /// @see [Product-based Neural Networks for User Response Prediction](https://arxiv.org/abs/1611.00144) by Yanru Qu, Kan Cai, Weinan Zhang, Yong Yu, Ying Wen, and Jun Wang, 2016. 
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class InnerProductLayer<T> : Layer<T>
    {
        int m_nM;
        int m_nK;
        int m_nN;
        bool m_bBiasTerm;
        Blob<T> m_blobBiasMultiplier;
        bool m_bTranspose;
        bool m_bEnableNoise = false;
        double m_dfSigmaInit = 0;
        Blob<T> m_blobEpsilonWeight = null;
        Blob<T> m_blobEpsilonBias = null;
        Filler<T> m_fillerEpsilon = null;
        Filler<T> m_fillerEpsilonBias = null;

        /// <summary>
        /// The InnerProductLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LayerParameter inner_product_param, with options:
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
        public InnerProductLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.INNERPRODUCT;
            m_blobBiasMultiplier = new Blob<T>(cuda, log);
            m_blobBiasMultiplier.Name = m_param.name + " biasmult";

            if (p.inner_product_param.enable_noise)
            {
                m_blobEpsilonWeight = new Blob<T>(cuda, log);
                m_blobEpsilonWeight.Name = "epsilon_wt";

                if (p.inner_product_param.bias_term)
                {
                    m_blobEpsilonBias = new Blob<T>(cuda, log);
                    m_blobEpsilonBias.Name = "epsilon_bias";
                }
            }
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobBiasMultiplier.Dispose();

            if (m_blobEpsilonBias != null)
            {
                m_blobEpsilonBias.Dispose();
                m_blobEpsilonBias = null;
            }

            if (m_blobEpsilonWeight != null)
            {
                m_blobEpsilonWeight.Dispose();
                m_blobEpsilonWeight = null;
            }

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobBiasMultiplier);

                if (m_blobEpsilonWeight != null)
                    col.Add(m_blobEpsilonWeight);

                if (m_blobEpsilonBias != null)
                    col.Add(m_blobEpsilonBias);

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input.
        /// </summary>
        public override int ExactNumBottomBlobs
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
        /// <returns>When handled, this method returns <i>true</i>, otherwise <i>false</i>.</returns>
        public override bool ReInitializeParameters()
        {
            base.ReInitializeParameters();

            Filler<T> weight_filler = Filler<T>.Create(m_cuda, m_log, m_param.inner_product_param.weight_filler);
            weight_filler.Fill(m_colBlobs[0]);

            if (m_param.inner_product_param.bias_term && m_colBlobs.Count > 1)
            {
                Filler<T> bias_filler = Filler<T>.Create(m_cuda, m_log, m_param.inner_product_param.bias_filler);
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
            int nNumOutput = (int)m_param.inner_product_param.num_output;
            m_bBiasTerm = m_param.inner_product_param.bias_term;
            m_bTranspose = m_param.inner_product_param.transpose;
            m_bEnableNoise = m_param.inner_product_param.enable_noise;
            m_dfSigmaInit = m_param.inner_product_param.sigma_init;
            m_nN = nNumOutput;

            List<int> rgShape = colBottom[0].shape();
            int nShapeCount = rgShape.Count;
            for (int i = nShapeCount; i <= m_param.inner_product_param.axis; i++)
            {
                rgShape.Add(1);
            }

            if (nShapeCount != rgShape.Count)
                colBottom[0].Reshape(rgShape);

            int nAxis = colBottom[0].CanonicalAxisIndex(m_param.inner_product_param.axis);

            // Dimensions starting from 'axis' are 'flattened' into a single
            // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
            // and axis == 1, N inner products with dimension CHW are preformed..
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

                Blob<T> blobWeight = new Blob<T>(m_cuda, m_log);
                blobWeight.Name = m_param.name + " weights";
                blobWeight.type = Blob<T>.BLOB_TYPE.IP_WEIGHT;

                if (!shareParameter(blobWeight, rgWeightShape))
                {
                    blobWeight.Reshape(rgWeightShape);
                    Filler<T> weight_filler = Filler<T>.Create(m_cuda, m_log, m_param.inner_product_param.weight_filler);
                    weight_filler.Fill(blobWeight);
                }
                m_colBlobs.Add(blobWeight);

                // If necessary, initialize and fill the bias term.
                if (m_bBiasTerm)
                {
                    List<int> rgBiasShape = Utility.Create<int>(1, 0);
                    rgBiasShape[0] = m_nN;

                    Blob<T> blobBias = new Blob<T>(m_cuda, m_log);
                    blobBias.Name = m_param.name + " bias";
                    blobBias.type = Blob<T>.BLOB_TYPE.IP_WEIGHT;

                    if (!shareParameter(blobBias, rgBiasShape))
                    {
                        blobBias.Reshape(rgBiasShape);
                        Filler<T> bias_filler = Filler<T>.Create(m_cuda, m_log, m_param.inner_product_param.bias_filler);
                        bias_filler.Fill(blobBias);
                    }
                    m_colBlobs.Add(blobBias);
                }

                // Add Noise sigma weight and bias
                if (m_bEnableNoise)
                {
                    FillerParameter fp = new FillerParameter("uniform");
                    fp.min = -1;
                    fp.max = 1;
                    m_fillerEpsilon = Filler<T>.Create(m_cuda, m_log, fp);

                    Blob<T> blobSigmaWeight = new Blob<T>(m_cuda, m_log);
                    blobSigmaWeight.Name = m_param.name + " sigma_wt";
                    blobSigmaWeight.type = Blob<T>.BLOB_TYPE.WEIGHT;
                    blobSigmaWeight.ReshapeLike(m_colBlobs[0]);
                    blobSigmaWeight.SetData(m_dfSigmaInit / Math.Sqrt(blobSigmaWeight.shape(1)));
                    m_colBlobs.Add(blobSigmaWeight);
                    m_blobEpsilonWeight.ReshapeLike(blobSigmaWeight);

                    if (m_bBiasTerm)
                    {
                        Blob<T> blobSigmaBias = new Blob<T>(m_cuda, m_log);
                        blobSigmaBias.Name = m_param.name + " sigma_bias";
                        blobSigmaBias.type = Blob<T>.BLOB_TYPE.WEIGHT;
                        blobSigmaBias.ReshapeLike(m_colBlobs[1]);
                        blobSigmaBias.SetData(m_dfSigmaInit / Math.Sqrt(blobSigmaBias.shape(0)));
                        m_colBlobs.Add(blobSigmaBias);
                        m_blobEpsilonBias.ReshapeLike(blobSigmaBias);
                    }

                    ResetNoise();
                }
            }

            m_rgbParamPropagateDown = new DictionaryMap<bool>(m_colBlobs.Count, true);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Figure out the dimensions
            int nAxis = colBottom[0].CanonicalAxisIndex(m_param.inner_product_param.axis);
            int nNewK = colBottom[0].count(nAxis);

            m_log.CHECK_EQ(m_nK, nNewK, "Input size incompatible with inner product parameters.");

            // The first 'axis' dimensions are independent of inner products; the total
            // number of these is M_, the product over these dimensions.
            m_nM = colBottom[0].count(0, nAxis);

            // The top shape will be the bottom shape with the flattened axes dropped,
            // and replaced by a single axis with dimensions num_output (N_).
            List<int> rgTopShape = Utility.Clone<int>(colBottom[0].shape(), nAxis + 1);
            rgTopShape[nAxis] = m_nN;

            // Deconvolution Layer requires min_top_axes = 4
            for (int i = rgTopShape.Count; i < m_param.inner_product_param.min_top_axes; i++)
            {
                rgTopShape.Add(1);
            }

            colTop[0].Reshape(rgTopShape);

            // Set up the bias multiplier
            if (m_bBiasTerm)
            {
                List<int> rgBiasShape = Utility.Create<int>(1, 0);
                rgBiasShape[0] = m_nM;
                m_blobBiasMultiplier.Reshape(rgBiasShape);
                m_blobBiasMultiplier.SetData(1.0);
            }
        }

        /// <summary>
        /// Resample the noise for both weights and bias (if used).
        /// </summary>
        public void ResetNoise()
        {
            if (m_bEnableNoise)
            {
                // Resamples the noise vector.
                resetNoise(m_blobEpsilonWeight);

                if (m_bBiasTerm)
                {
                    // Resample the noise vector
                    resetNoise(m_blobEpsilonBias);
                }
            }
        }

        private void resetNoise(Blob<T> b)
        {
            m_fillerEpsilon.Fill(b);
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

            if (m_bEnableNoise)
            {
                // Multiply the sigma weight by the noise vector.
                m_cuda.mul(m_colBlobs[2].count(), m_colBlobs[2].gpu_data, m_blobEpsilonWeight.gpu_data, m_blobEpsilonWeight.mutable_gpu_diff);
                // Add the sigma noise to the weights.
                m_cuda.add(m_colBlobs[0].count(), m_colBlobs[0].gpu_data, m_blobEpsilonWeight.gpu_diff, m_colBlobs[0].mutable_gpu_data);

                if (m_bBiasTerm)
                {
                    // Multiply the sigma bias by the noise vector.
                    m_cuda.mul(m_colBlobs[3].count(), m_colBlobs[3].gpu_data, m_blobEpsilonBias.gpu_data, m_blobEpsilonBias.mutable_gpu_diff);
                    // Add the sigma noise to the bias.
                    m_cuda.add(m_colBlobs[1].count(), m_colBlobs[1].gpu_data, m_blobEpsilonBias.gpu_diff, m_colBlobs[1].mutable_gpu_data);
                }
            }

            if (m_nM == 1)
            {
                m_cuda.gemv(false, m_nN, m_nK, m_tOne, hWeight, hBottomData, m_tZero, hTopData);

                if (m_bBiasTerm)
                    m_cuda.axpy(m_nN, m_blobBiasMultiplier.GetData(0), m_colBlobs[1].gpu_data, hTopData);
            }
            else
            {
                m_cuda.gemm(false, (m_bTranspose) ? false : true, m_nM, m_nN, m_nK, m_tOne, hBottomData, hWeight, m_tZero, hTopData);

                if (m_bBiasTerm)
                    m_cuda.gemm(false, false, m_nM, m_nN, 1, m_tOne, m_blobBiasMultiplier.gpu_data, m_colBlobs[1].gpu_data, m_tOne, hTopData);
            }
        }

        /// <summary>
        /// Computes the inner product loss error gradient w.r.t the outputs.
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
            long hTopDiff = colTop[0].gpu_diff;

            // Gradient with respect to weight.
            if (m_rgbParamPropagateDown[0])
            {
                long hBottomData = colBottom[0].gpu_data;

                if (m_bTranspose)
                    m_cuda.gemm(true, false, m_nK, m_nN, m_nM, m_tOne, hBottomData, hTopDiff, m_tOne, m_colBlobs[0].mutable_gpu_diff);
                else
                    m_cuda.gemm(true, false, m_nN, m_nK, m_nM, m_tOne, hTopDiff, hBottomData, m_tOne, m_colBlobs[0].mutable_gpu_diff);
            }

            // Gradient with respect to bias.
            if (m_bBiasTerm && m_rgbParamPropagateDown[1])
            {
                m_cuda.gemv(true, m_nM, m_nN, m_tOne, hTopDiff, m_blobBiasMultiplier.gpu_data, m_tOne, m_colBlobs[1].mutable_gpu_diff);
            }

            // Gradient with respect to bottom data.
            if (rgbPropagateDown[0])
            {
                if (m_bTranspose)
                    m_cuda.gemm(false, true, m_nM, m_nK, m_nN, m_tOne, hTopDiff, m_colBlobs[0].gpu_data, m_tZero, colBottom[0].mutable_gpu_diff);
                else
                    m_cuda.gemm(false, false, m_nM, m_nK, m_nN, m_tOne, hTopDiff, m_colBlobs[0].gpu_data, m_tZero, colBottom[0].mutable_gpu_diff);
            }

            if (m_bEnableNoise)
            {
                // Gradient with respect to the sigma weight
                m_cuda.mul(m_colBlobs[2].count(), m_colBlobs[0].gpu_diff, m_blobEpsilonWeight.gpu_data, m_colBlobs[2].mutable_gpu_diff);

                if (m_bBiasTerm)
                {
                    // Gradient with respect to the sigma bais
                    m_cuda.mul(m_colBlobs[3].count(), m_colBlobs[1].gpu_diff, m_blobEpsilonBias.gpu_data, m_colBlobs[3].mutable_gpu_diff);
                }
            }
        }
    }
}
