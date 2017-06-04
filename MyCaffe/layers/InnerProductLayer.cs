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
            m_blobBiasMultiplier.Name = "ip_biasmult";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobBiasMultiplier.Dispose();
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobBiasMultiplier);

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
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNumOutput = (int)m_param.inner_product_param.num_output;
            m_bBiasTerm = m_param.inner_product_param.bias_term;
            m_bTranspose = m_param.inner_product_param.transpose;
            m_nN = nNumOutput;
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

                    if (!shareParameter(blobBias, rgBiasShape))
                    {
                        blobBias.Reshape(rgBiasShape);
                        Filler<T> bias_filler = Filler<T>.Create(m_cuda, m_log, m_param.inner_product_param.bias_filler);
                        bias_filler.Fill(blobBias);
                    }
                    m_colBlobs.Add(blobBias);
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
        }
    }
}
