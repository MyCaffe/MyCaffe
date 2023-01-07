using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The "Mean-Variance Normalization" MVNLayer normalizes the input to have 0-mean and/or unit (1) variance.
    /// This layer is initialized with the MyCaffe.param.MVNParameter.
    /// </summary>
    /// <remarks>
    /// @see [Layer Normalization](https://arxiv.org/abs/1607.06450) by Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton, 2016.
    /// @see [Learning weakly supervised multimodal phoneme embeddings](https://arxiv.org/abs/1704.06913v1) by Rahma Chaabouni, Ewan Dunbar, Neil Zeghidour, and Emmanuel Dupoux, 2017. 
    /// @see [Estimating Phoneme Class Conditional Probabilities from Raw Speech Signal using Convolutional Neural Networks](https://arxiv.org/abs/1304.1018) by Dimitri Palaz, Ronan Collobert, and Mathew Magimai-Doss, 2013.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class MVNLayer<T> : Layer<T>
    {
        Blob<T> m_blobMean;
        Blob<T> m_blobVariance;
        Blob<T> m_blobTemp;
        // Sum_multiplier is used to carry out sum using BLAS
        Blob<T> m_blobSumMultiplier;
        double m_dfEps;

        /// <summary>
        /// The MVNLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LossParameter mvn_param, with options:
        ///   - normalize_variance (\b optional, default = true). Whether or not to normalize the variance.
        ///   
        ///   - across_channels (\b optional, default = false). Whether or not to normalize across channels.
        ///   
        ///   - eps (\b optional, default = 1e-9). A small value to avoid divide by zero.
        /// </param>
        public MVNLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.MVN;
            m_blobMean = new common.Blob<T>(cuda, log);
            m_blobMean.Name = m_param.name + " mean";
            m_blobVariance = new common.Blob<T>(cuda, log);
            m_blobVariance.Name = m_param.name + " variance";
            m_blobTemp = new Blob<T>(cuda, log);
            m_blobTemp.Name = m_param.name + " temp";
            m_blobSumMultiplier = new Blob<T>(cuda, log);
            m_blobSumMultiplier.Name = m_param.name + " summult";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_blobMean != null)
            {
                m_blobMean.Dispose();
                m_blobMean = null;
            }

            if (m_blobVariance != null)
            {
                m_blobVariance.Dispose();
                m_blobVariance = null;
            }

            if (m_blobTemp != null)
            {
                m_blobTemp.Dispose();
                m_blobTemp = null;
            }

            if (m_blobSumMultiplier != null)
            {
                m_blobSumMultiplier.Dispose();
                m_blobSumMultiplier = null;
            }

            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobMean);
            col.Add(m_blobVariance);
            col.Add(m_blobTemp);
            col.Add(m_blobSumMultiplier);
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: mvn
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
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            colTop[0].ReshapeLike(colBottom[0]);

            m_blobMean.Reshape(colBottom[0].num, colBottom[0].channels, 1, 1);
            m_blobVariance.Reshape(colBottom[0].num, colBottom[0].channels, 1, 1);
            m_blobTemp.Reshape(colBottom[0].num, colBottom[0].channels, colBottom[0].height, colBottom[0].width);

            if (m_param.mvn_param.across_channels)
                m_blobSumMultiplier.Reshape(1, colBottom[0].channels, colBottom[0].height, colBottom[0].width);
            else
                m_blobSumMultiplier.Reshape(1, 1, colBottom[0].height, colBottom[0].width);

            m_blobSumMultiplier.SetData(1.0);
            m_dfEps = m_param.mvn_param.eps;
        }

        /// <summary>
        /// The forward computation that computes the normalization.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     The input data.
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     The normalized output data.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nNum = colBottom[0].num;

            if (!m_param.mvn_param.across_channels)
                nNum *= colBottom[0].channels;

            int nDim = colBottom[0].count() / nNum;

            // subtract mean
            m_cuda.gemv(false, nNum, nDim, 1.0 / nDim, hBottomData, m_blobSumMultiplier.gpu_data, 0.0, m_blobMean.mutable_gpu_data); //EX
            m_cuda.gemm(false, false, nNum, nDim, 1, -1.0, m_blobMean.gpu_data, m_blobSumMultiplier.gpu_data, 0.0, m_blobTemp.mutable_gpu_data);
            m_cuda.add(m_blobTemp.count(), hBottomData, m_blobTemp.gpu_data, hTopData); // X-EX

            if (m_param.mvn_param.normalize_variance)
            {
                // compute variance using var(X) = E((X-EX)^2)
                m_cuda.powx(colBottom[0].count(), hTopData, 2.0, m_blobTemp.mutable_gpu_data); // (X-EX)^2
                m_cuda.gemv(false, nNum, nDim, 1.0 / nDim, m_blobTemp.gpu_data, m_blobSumMultiplier.gpu_data, 0.0, m_blobVariance.mutable_gpu_data); // E((X-EX)^2)

                // normalize variance
                m_cuda.powx(m_blobVariance.count(), m_blobVariance.gpu_data, 0.5, m_blobVariance.mutable_gpu_data);
                m_cuda.add_scalar(m_blobVariance.count(), m_dfEps, m_blobVariance.mutable_gpu_data);
                m_cuda.gemm(false, false, nNum, nDim, 1, 1.0, m_blobVariance.gpu_data, m_blobSumMultiplier.gpu_data, 0.0, m_blobTemp.mutable_gpu_data);
                m_cuda.div(m_blobTemp.count(), hTopData, m_blobTemp.gpu_data, hTopData);
            }
        }

        /// <summary>
        /// Computes the mvn error gradient w.r.t the output.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the outputs.
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            long hTopDiff = colTop[0].gpu_diff;
            long hTopData = colTop[0].gpu_data;
            long hBottomData = colBottom[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int nNum = colBottom[0].num;

            if (!m_param.mvn_param.across_channels)
                nNum *= colBottom[0].channels;

            int nDim = colBottom[0].count() / nNum;

            if (m_param.mvn_param.normalize_variance)
            {
                m_cuda.mul(m_blobTemp.count(), hTopData, hTopDiff, hBottomDiff);
                m_cuda.gemv(false, nNum, nDim, 1.0, hBottomDiff, m_blobSumMultiplier.gpu_data, 0.0, m_blobMean.mutable_gpu_data);
                m_cuda.gemm(false, false, nNum, nDim, 1, 1.0, m_blobMean.gpu_data, m_blobSumMultiplier.gpu_data, 0.0, hBottomDiff);
                m_cuda.mul(m_blobTemp.count(), hTopData, hBottomDiff, hBottomDiff);

                m_cuda.gemv(false, nNum, nDim, 1.0, hTopDiff, m_blobSumMultiplier.gpu_data, 0.0, m_blobMean.mutable_gpu_data);
                m_cuda.gemm(false, false, nNum, nDim, 1, 1.0, m_blobMean.gpu_data, m_blobSumMultiplier.gpu_data, 1.0, hBottomDiff);

                m_cuda.axpby(m_blobTemp.count(), 1.0, hTopDiff, -1.0 / nDim, hBottomDiff);

                // put the squares of bottom into temp_
                m_cuda.powx(m_blobTemp.count(), hBottomData, 2.0, m_blobTemp.mutable_gpu_data);
                m_cuda.gemm(false, false, nNum, nDim, 1, 1.0, m_blobVariance.gpu_data, m_blobSumMultiplier.gpu_data, 0.0, m_blobTemp.mutable_gpu_data);
                m_cuda.div(m_blobTemp.count(), hBottomDiff, m_blobTemp.gpu_data, hBottomDiff);
            }
            else
            {
                m_cuda.gemv(false, nNum, nDim, 1.0 / nDim, hTopDiff, m_blobSumMultiplier.gpu_data, 0.0, m_blobMean.mutable_gpu_data);
                m_cuda.gemm(false, false, nNum, nDim, 1, -1.0, m_blobMean.gpu_data, m_blobSumMultiplier.gpu_data, 0.0, m_blobTemp.mutable_gpu_data);
                m_cuda.add(m_blobTemp.count(), hTopDiff, m_blobTemp.gpu_data, hBottomDiff);
            }
        }
    }
}
