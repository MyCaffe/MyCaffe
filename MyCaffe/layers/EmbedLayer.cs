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
    /// The EmbedLayer is a layer for learning 'embeddings' of one-hot vector input.
    /// This layer is initialized with the MyCaffe.param.EmbedParameter.
    /// </summary>
    /// <remarks>
    /// Equivalent to an InnerProductLayer with one-hot vectors as input, but
    /// for efficiency the input is the 'hot' index of each column itself.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class EmbedLayer<T> : Layer<T>
    {
        int m_nM;
        int m_nK;
        int m_nN;
        bool m_bBiasTerm;
        Blob<T> m_blobBiasMultiplier;

        /// <summary>
        /// The EmbedLayer constructor
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides EmbedLayer embed_param,
        /// with EmbedLayer options:
        /// - num_output. The number of outputs for the Layer.
        /// 
        /// - input_dim. The input given as an integer to be interpreted as one-hot
        /// vector indices with dimension num_input.  
        /// 
        /// - bias_term (/bdefault = true).  Whether or not to use bias.
        /// 
        /// - weight_filler. The weight filler to use.
        /// 
        /// - bias_filler.  The bias filler to use.
        /// </param>
        public EmbedLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.EMBED;
            m_blobBiasMultiplier = new common.Blob<T>(cuda, log);
            m_blobBiasMultiplier.Name = m_param.name + " biasmult";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_blobBiasMultiplier != null)
            {
                m_blobBiasMultiplier.Dispose();
                m_blobBiasMultiplier = null;
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

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (intput) Blobs: input.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: embed
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

            Filler<T> weight_filler = Filler<T>.Create(m_cuda, m_log, m_param.embed_param.weight_filler);
            weight_filler.Fill(m_colBlobs[0]);

            if (m_param.embed_param.bias_term && m_colBlobs.Count > 1)
            {
                Filler<T> bias_filler = Filler<T>.Create(m_cuda, m_log, m_param.embed_param.bias_filler);
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
            m_nN = (int)m_param.embed_param.num_output;
            m_log.CHECK_GT(m_nN, 0, "EmbedLayer num_output must be positive.");

            m_nK = (int)m_param.embed_param.input_dim;
            m_log.CHECK_GT(m_nK, 0, "EmbedLayer input_dim must be positive.");

            m_bBiasTerm = m_param.embed_param.bias_term;

            if (m_colBlobs.Count > 0)
            {
                m_log.WriteLine("Skipping parameter initialization.");
            }
            else
            {
                m_colBlobs.Clear();

                // Initialize the weights --
                // transposed from InnerProductLayer for spacial locality.
                List<int> rgWeightShape = new List<int>() { m_nK, m_nN };
                Blob<T> blobWeight = new Blob<T>(m_cuda, m_log);
                blobWeight.Name = m_param.name + " weights";
                blobWeight.type = BLOB_TYPE.WEIGHT;

                if (!shareParameter(blobWeight, rgWeightShape))
                {
                    blobWeight.Reshape(rgWeightShape);

                    // fill the weights
                    Filler<T> weight_filler = Filler<T>.Create(m_cuda, m_log, m_param.embed_param.weight_filler);
                    weight_filler.Fill(blobWeight);
                }
                m_colBlobs.Add(blobWeight);


                // If necessary, initialize and fill the bias term
                if (m_bBiasTerm)
                {
                    List<int> rgBiasShape = new List<int>() { m_nN };
                    Blob<T> blobBias = new Blob<T>(m_cuda, m_log);
                    blobBias.Name = m_param.name + " bias";
                    blobBias.type = BLOB_TYPE.WEIGHT;

                    if (!shareParameter(blobBias, rgBiasShape))
                    {
                        blobBias.Reshape(rgBiasShape);
                        Filler<T> bias_filler = Filler<T>.Create(m_cuda, m_log, m_param.embed_param.bias_filler);
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
            m_nM = colBottom[0].count();
            List<int> rgTopShape = Utility.Clone<int>(colBottom[0].shape());
            rgTopShape.Add(m_nN);
            colTop[0].Reshape(rgTopShape);

            // Set up the bias multiplier
            if (m_bBiasTerm)
            {
                List<int> rgBiasShape = new List<int>() { m_nM };
                m_blobBiasMultiplier.Reshape(rgBiasShape);
                m_blobBiasMultiplier.SetData(1.0);
            }
        }

        /// <summary>
        /// The Forward computation.
        /// </summary>
        /// <param name="colBottom">input blob vector (length 1)</param>
        /// <param name="colTop">output blob vector (length 1)</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            long hWeight = m_colBlobs[0].gpu_data;
            int nCount = colTop[0].count();

            m_cuda.embed_fwd(nCount, hBottomData, hWeight, m_nM, m_nN, m_nK, hTopData);

            if (m_bBiasTerm)
                m_cuda.gemm(false, false, m_nM, m_nN, 1, 1.0, m_blobBiasMultiplier.gpu_data, m_colBlobs[1].gpu_data, 1.0, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the input.
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (length 1).</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            m_log.CHECK(!rgbPropagateDown[0], "Can't backpropagate to EmbedLayer input.");

            if (m_rgbParamPropagateDown[0])
            {
                int nTopCount = colTop[0].count();
                long hTopDiff = colTop[0].gpu_diff;
                long hBottomData = colBottom[0].gpu_data;
                long hWeightDiff = m_colBlobs[0].mutable_gpu_diff;
                m_cuda.embed_bwd(nTopCount, hBottomData, hTopDiff, m_nM, m_nN, m_nK, hWeightDiff);                
            }

            if (m_bBiasTerm && m_rgbParamPropagateDown[1])
            {
                long hTopDiff = colTop[0].gpu_diff;
                long hBiasDiff = m_colBlobs[1].mutable_gpu_diff;
                m_cuda.gemv(true, m_nM, m_nN, 1.0, hTopDiff, m_blobBiasMultiplier.gpu_data, 1.0, hBiasDiff);
            }
        }
    }
}
