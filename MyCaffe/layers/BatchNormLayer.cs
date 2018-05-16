using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The BatchNormLayer normalizes the input to have 0-mean and/or unit (1) variance across
    /// the batch.
    /// This layer is initialized with the BatchNormParameter.
    /// </summary>
    /// <remarks>
    /// This layer computes Batch Normalization as described in [1]. For each channel
    /// in the data (i.e. axis 1), it subtracts the mean and divides by the variance, 
    /// where both statistics are computed across both spatial dimensions and across 
    /// the different examples in the batch.
    /// 
    /// By default, during training time, the network its computing global 
    /// mean/variance statistics via a running average, which is then used at test
    /// time to allow deterministic outputs for each input.  You can manually
    /// toggle whether the network is accumulating or using the statistics via the
    /// use_global_stats option.  For reference, these statistics are kept int the
    /// layer's three blobs: (0) mean, (1) variance, and (2) moving average factor.
    /// IMPORTANT: for this feature to work, you MUST set the learning rate to zero
    /// for all three parameter blobs, i.e., param {lr_mult: 0} three times in the
    /// layer definition.
    /// 
    /// Note that the original papaer also included a per-channel learned bias and
    /// scaling factor.  To implement this in Caffe, define a 'ScaleLayer' configured
    /// with 'bias_term: true' after each 'BatchNormLayer' to handle both the bias
    /// and scaling factor.
    /// 
    /// [1] S. Ioffe and C. Szegedy, [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). arXiv preprint
    ///     arXiv:1502.03167 (2015).
    ///     
    /// @see [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737v2) by Alexander Hermans, Lucas Beyer, and Bastian Leibe, 2017. 
    /// @see [Layer Normalization](https://arxiv.org/abs/1607.06450) by Jimmy Lei Ba,  and Jamie Ryan Kiros, and Geoffrey E. Hinton, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class BatchNormLayer<T> : Layer<T>
    {
        Blob<T> m_blobMean;     // also used as save mean with cuDNN
        Blob<T> m_blobVariance; // also used as save var with cuDNN
        Blob<T> m_blobTemp;     
        Blob<T> m_blobXNorm;    
        bool m_bUseGlobalStats;
        double m_dfMovingAverageFraction;
        int m_nChannels;
        double m_dfEps;

        // extra temporary variables used to carry out sums/broadcasting using BLAS
        Blob<T> m_blobBatchSumMultiplier;
        Blob<T> m_blobNumByChans;
        Blob<T> m_blobSpaitalSumMultiplier;

        // cuDNN support
        bool m_bScaleBias = false;
        long m_hCuDnn = 0;
        long m_hFwdBottomDesc = 0;
        long m_hFwdTopDesc = 0;
        long m_hBwdBottomDesc = 0;
        long m_hBwdTopDesc = 0;
        long m_hFwdScaleBiasMeanVarDesc = 0;
        long m_hBwdScaleBiasMeanVarDesc = 0;
        BATCHNORM_MODE m_mode = BATCHNORM_MODE.SPATIAL_PERSISTENT;
        Blob<T> m_blobScaleOnes = null;
        Blob<T> m_blobBiasZeros = null;
        Blob<T> m_blobPrivateTop = null;
        Blob<T> m_blobPrivateBottom = null;
        const double CUDNN_BN_MIN_EPSILON = 1e-5;
        int m_nIteration = 0;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides BatchNormParam batch_norm_param.</param>
        public BatchNormLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.BATCHNORM;
            m_blobMean = new common.Blob<T>(cuda, log);
            m_blobMean.Name = "bn_mean";
            m_blobVariance = new common.Blob<T>(cuda, log);
            m_blobVariance.Name = "bn_variance";
            m_blobTemp = new common.Blob<T>(cuda, log);
            m_blobTemp.Name = "bn_temp";
            m_blobXNorm = new common.Blob<T>(cuda, log);
            m_blobXNorm.Name = "bn_xnorm";
            m_blobBatchSumMultiplier = new common.Blob<T>(cuda, log);
            m_blobBatchSumMultiplier.Name = "bn_summult";
            m_blobNumByChans = new common.Blob<T>(cuda, log);
            m_blobNumByChans.Name = "bn_numbychan";
            m_blobSpaitalSumMultiplier = new common.Blob<T>(cuda, log);
            m_blobSpaitalSumMultiplier.Name = "bn_spatialsummult";

            if (p.batch_norm_param.useCudnn())
            {
                m_blobMean.Name = "save mean";
                m_blobVariance.Name = "save var";

                m_blobPrivateTop = new Blob<T>(cuda, log);
                m_blobPrivateTop.Name = "private top";
                m_blobPrivateBottom = new Blob<T>(cuda, log);
                m_blobPrivateBottom.Name = "private bottom";
                m_blobScaleOnes = new Blob<T>(cuda, log);
                m_blobScaleOnes.Name = "scale ones";
                m_blobBiasZeros = new Blob<T>(cuda, log);
                m_blobBiasZeros.Name = "bias zeros";
            }
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobMean.Dispose();
            m_blobVariance.Dispose();
            m_blobTemp.Dispose();
            m_blobXNorm.Dispose();
            m_blobBatchSumMultiplier.Dispose();
            m_blobNumByChans.Dispose();
            m_blobSpaitalSumMultiplier.Dispose();


            // CuDnn Cleanup
            if (m_blobPrivateTop != null)
            {
                m_blobPrivateTop.Dispose();
                m_blobPrivateTop = null;
            }

            if (m_blobPrivateBottom != null)
            {
                m_blobPrivateBottom.Dispose();
                m_blobPrivateBottom = null;
            }

            if (m_blobScaleOnes != null)
            {
                m_blobScaleOnes.Dispose();
                m_blobScaleOnes = null;
            }

            if (m_blobBiasZeros != null)
            {
                m_blobBiasZeros.Dispose();
                m_blobBiasZeros = null;
            }

            if (m_hBwdBottomDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hBwdBottomDesc);
                m_hBwdBottomDesc = 0;
            }

            if (m_hBwdScaleBiasMeanVarDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hBwdScaleBiasMeanVarDesc);
                m_hBwdScaleBiasMeanVarDesc = 0;
            }

            if (m_hBwdTopDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hBwdTopDesc);
                m_hBwdTopDesc = 0;
            }

            if (m_hFwdBottomDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hFwdBottomDesc);
                m_hFwdBottomDesc = 0;
            }

            if (m_hFwdScaleBiasMeanVarDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hFwdScaleBiasMeanVarDesc);
                m_hFwdScaleBiasMeanVarDesc = 0;
            }

            if (m_hFwdTopDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hFwdTopDesc);
                m_hFwdTopDesc = 0;
            }

            if (m_hCuDnn != 0)
            {
                m_cuda.FreeCuDNN(m_hCuDnn);
                m_hCuDnn = 0;
            }

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobMean);
                col.Add(m_blobVariance);

                if (m_param.batch_norm_param.useCudnn())
                {
                    col.Add(m_blobPrivateBottom);
                    col.Add(m_blobPrivateTop);

                    if (!m_bScaleBias)
                    {
                        col.Add(m_blobScaleOnes);
                        col.Add(m_blobBiasZeros);
                    }
                }
                else
                {
                    col.Add(m_blobTemp);
                    col.Add(m_blobXNorm);
                    col.Add(m_blobBatchSumMultiplier);
                    col.Add(m_blobNumByChans);
                    col.Add(m_blobSpaitalSumMultiplier);
                }

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of bottom (input) Blobs required: input
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of top (output) Blobs required: batchnorm
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

            for (int i = 0; i < 3; i++)
            {
                m_colBlobs[i].SetData(0);
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
            bool bUseCuDnn = m_param.batch_norm_param.useCudnn();

            m_dfMovingAverageFraction = m_param.batch_norm_param.moving_average_fraction;
            m_bUseGlobalStats = (m_phase == Phase.TEST || m_phase == Phase.RUN) ? true : false;

            if (m_param.batch_norm_param.use_global_stats.HasValue)
                m_bUseGlobalStats = m_param.batch_norm_param.use_global_stats.Value;

            if (colBottom[0].num_axes == 1)
                m_nChannels = 1;
            else
                m_nChannels = colBottom[0].shape(1);

            m_dfEps = m_param.batch_norm_param.eps;

            m_bScaleBias = m_param.batch_norm_param.scale_bias;  // by default = false;
            if (m_param.batch_norm_param.scale_filler != null || // implicit set.
                m_param.batch_norm_param.bias_filler != null)
                m_bScaleBias = true;

            if (m_bScaleBias && !bUseCuDnn)
                m_bScaleBias = false;

            if (m_colBlobs.Count > 0)
            {
                m_log.WriteLine("Skipping parameter initialization.");
            }
            else
            {
                List<int> rgSize = new List<int>();
                rgSize.Add(m_nChannels);

                m_colBlobs.Clear(true);

                m_colBlobs.Add(new Blob<T>(m_cuda, m_log, rgSize));  // global mean
                m_colBlobs[0].Name = "global mean";
                m_colBlobs[0].SetData(0.0);
                m_colBlobs.Add(new Blob<T>(m_cuda, m_log, rgSize));  // glboal var
                m_colBlobs[1].Name = "global variance";
                m_colBlobs[1].SetData(0.0);
                m_colBlobs.Add(new Blob<T>(m_cuda, m_log, rgSize));  // variance correction
                m_colBlobs[2].Name = "var correction";
                m_colBlobs[2].SetData(1.0);

                if (m_bScaleBias)
                {
                    m_colBlobs.Add(new Blob<T>(m_cuda, m_log, rgSize)); // scale
                    m_colBlobs[3].Name = "scale";

                    FillerParameter fpScale = m_param.batch_norm_param.scale_filler;
                    if (fpScale == null)
                        fpScale = new FillerParameter("constant", 1.0);

                    Filler<T> fillerScale = Filler<T>.Create(m_cuda, m_log, fpScale);
                    fillerScale.Fill(m_colBlobs[3]);

                    m_colBlobs.Add(new Blob<T>(m_cuda, m_log, rgSize)); // bias
                    m_colBlobs[4].Name = "bias";

                    FillerParameter fpBias = m_param.batch_norm_param.bias_filler;
                    if (fpBias == null)
                        fpBias = new FillerParameter("constant", 0.0);

                    Filler<T> fillerBias = Filler<T>.Create(m_cuda, m_log, fpBias);
                    fillerBias.Fill(m_colBlobs[4]);
                }

                m_nIteration = 0;
            }

            // Mask statistics from optimization by setting local learning rates
            // for mean, variance, and variance correction to zero.
            for (int i = 0; i < 3; i++)
            {
                if (m_param.parameters.Count == i)
                {
                    m_param.parameters.Add(new ParamSpec(0.0, 0.0));
                }
                else
                {
                    m_param.parameters[i].lr_mult = 0;
                    m_param.parameters[i].decay_mult = 0;
                }
            }

            // Set lr for scale and bias to 1
            if (m_bScaleBias)
            {
                for (int i = 3; i < 5; i++)
                {
                    if (m_param.parameters.Count == i)
                    {
                        m_param.parameters.Add(new ParamSpec(1.0, 1.0));
                    }
                    else
                    {
                        m_param.parameters[i].lr_mult = 1;
                        m_param.parameters[i].decay_mult = 1;
                    }
                }
            }

            if (!m_param.batch_norm_param.useCudnn())
                return;

            //-----------------------------------
            // Handle cuDNN setup
            //-----------------------------------

            int nChannels = colBottom[0].channels;
            List<int> rgShape = new List<int>() { 1, nChannels, 1, 1 };

            if (!m_bScaleBias)
            {
                m_blobScaleOnes.Reshape(rgShape);
                m_blobScaleOnes.SetData(1.0);
                m_blobBiasZeros.Reshape(rgShape);
                m_blobBiasZeros.SetData(0.0);
            }

            m_hCuDnn = m_cuda.CreateCuDNN();
            m_hFwdBottomDesc = m_cuda.CreateTensorDesc();
            m_hFwdTopDesc = m_cuda.CreateTensorDesc();
            m_hFwdScaleBiasMeanVarDesc = m_cuda.CreateTensorDesc();
            m_hBwdBottomDesc = m_cuda.CreateTensorDesc();
            m_hBwdTopDesc = m_cuda.CreateTensorDesc();
            m_hBwdScaleBiasMeanVarDesc = m_cuda.CreateTensorDesc();
            m_mode = BATCHNORM_MODE.SPATIAL_PERSISTENT;
            m_dfEps = Math.Min(m_dfEps, CUDNN_BN_MIN_EPSILON);

            m_blobMean.Reshape(rgShape);
            m_blobVariance.Reshape(rgShape);

            if (colBottom[0] == colTop[0]) // CuDNN BN does not support in-place.
            {
                m_blobPrivateTop.ReshapeLike(colTop[0]);
                m_blobPrivateBottom.ReshapeLike(colBottom[0]);
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (colBottom[0].num_axes >= 1)
                m_log.CHECK_EQ(colBottom[0].shape(1), m_nChannels, "The colBottom[0].shape(1) should equal the channel count '" + m_nChannels.ToString() + "'.");

            colTop[0].ReshapeLike(colBottom[0]);

            List<int> rgSize = new List<int>();
            rgSize.Add(m_nChannels);

            m_blobMean.Reshape(rgSize);
            m_blobVariance.Reshape(rgSize);

            if (!m_param.batch_norm_param.useCudnn())
            {
                m_blobTemp.ReshapeLike(colBottom[0]);
                m_blobXNorm.ReshapeLike(colBottom[0]);

                rgSize[0] = colBottom[0].shape(0);
                m_blobBatchSumMultiplier.Reshape(rgSize);

                int nSpatialDim = colBottom[0].count() / (m_nChannels * colBottom[0].shape(0));
                if (m_blobSpaitalSumMultiplier.num_axes == 0 ||
                    m_blobSpaitalSumMultiplier.shape(0) != nSpatialDim)
                {
                    rgSize[0] = nSpatialDim;
                    m_blobSpaitalSumMultiplier.Reshape(rgSize);
                    m_blobSpaitalSumMultiplier.SetData(1);
                }

                int nNumByChans = m_nChannels * colBottom[0].shape(0);
                if (m_blobNumByChans.num_axes == 0 ||
                    m_blobNumByChans.shape(0) != nNumByChans)
                {
                    rgSize[0] = nNumByChans;
                    m_blobNumByChans.Reshape(rgSize);
                    m_blobBatchSumMultiplier.SetData(1);
                }

                return;
            }

            //-----------------------------------
            // Handle cuDNN setup
            //-----------------------------------
            int N = colBottom[0].num;
            int C = colBottom[0].channels;
            int H = colBottom[0].height;
            int W = colBottom[0].width;

            // Setup the main tensors.
            m_cuda.SetTensorDesc(m_hFwdBottomDesc, N, C, H, W);
            m_cuda.SetTensorDesc(m_hFwdTopDesc, N, C, H, W);
            m_cuda.SetTensorDesc(m_hBwdBottomDesc, N, C, H, W);
            m_cuda.SetTensorDesc(m_hBwdTopDesc, N, C, H, W);

            // Setup auxilary tensors for caching mean and inVar for forward and backard pass.
            m_blobMean.Reshape(1, C, 1, 1);
            m_blobVariance.Reshape(1, C, 1, 1);

            if (!m_param.batch_norm_param.scale_bias)
            {
                if (m_blobScaleOnes.channels != C)    // scale
                {
                    m_blobScaleOnes.Reshape(1, C, 1, 1);
                    m_blobScaleOnes.SetData(1.0);
                }

                if (m_blobBiasZeros.channels != C)    // bias
                {
                    m_blobBiasZeros.Reshape(1, C, 1, 1);
                    m_blobBiasZeros.SetData(0.0);
                }
            }

            m_cuda.DeriveBatchNormDesc(m_hFwdScaleBiasMeanVarDesc, m_hFwdBottomDesc, m_hBwdScaleBiasMeanVarDesc, m_hBwdBottomDesc, m_mode);

            if (colTop[0] == colBottom[0])
            {
                m_blobPrivateTop.ReshapeLike(colTop[0]);
                m_blobPrivateBottom.ReshapeLike(colBottom[0]);
            }
        }

        /// <summary>
        /// Perform the forward compuation.
        /// </summary>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_param.batch_norm_param.useCudnn())
                forward_cudnn(colBottom, colTop);
            else
                forward_cuda(colBottom, colTop);
        }

        /// <summary>
        /// Perform the backward computation.
        /// </summary>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (m_param.batch_norm_param.useCudnn())
                backward_cudnn(colTop, rgbPropagateDown, colBottom);
            else
                backward_cuda(colTop, rgbPropagateDown, colBottom);
        }

        /// <summary>
        /// Perform the forward compuation using the native Cuda version.
        /// </summary>
        protected void forward_cuda(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nNum = colBottom[0].shape(0);
            int nSpatialDim = colBottom[0].count() / (m_nChannels * colBottom[0].shape(0));

            if (colBottom[0] != colTop[0])
                m_cuda.copy(colBottom[0].count(), hBottomData, hTopData);

            if (m_bUseGlobalStats)
            {
                // use the stored mean/variance estimates
                double dfScaleFactor = convertD(m_colBlobs[2].GetData(0));

                if (dfScaleFactor != 0)
                    dfScaleFactor = 1.0 / dfScaleFactor;

                int nCount = m_blobVariance.count();

                m_cuda.scale(nCount, dfScaleFactor, m_colBlobs[0].gpu_data, m_blobMean.mutable_gpu_data);
                m_cuda.scale(nCount, dfScaleFactor, m_colBlobs[1].gpu_data, m_blobVariance.mutable_gpu_data);
            }
            else
            {
                // compute mean
                m_cuda.gemv(false, m_nChannels * nNum, nSpatialDim, 1.0 / (nNum * nSpatialDim), hBottomData, m_blobSpaitalSumMultiplier.gpu_data, 0.0, m_blobNumByChans.mutable_gpu_data);
                m_cuda.gemv(true, nNum, m_nChannels, 1.0, m_blobNumByChans.gpu_data, m_blobBatchSumMultiplier.gpu_data, 0.0, m_blobMean.mutable_gpu_data);
            }

            // subtract mean
            m_cuda.gemm(false, false, nNum, m_nChannels, 1, 1.0, m_blobBatchSumMultiplier.gpu_data, m_blobMean.gpu_data, 0.0, m_blobNumByChans.mutable_gpu_data);
            m_cuda.gemm(false, false, m_nChannels * nNum, nSpatialDim, 1, -1.0, m_blobNumByChans.gpu_data, m_blobSpaitalSumMultiplier.gpu_data, 1.0, hTopData);

            if (!m_bUseGlobalStats)
            {
                // compute variance using var(x) = E((X-EX)^2)
                m_cuda.mul(colTop[0].count(), hTopData, hTopData, m_blobTemp.mutable_gpu_data); // (X-EX)^2
                m_cuda.gemv(false, m_nChannels * nNum, nSpatialDim, 1.0 / (nNum * nSpatialDim), m_blobTemp.gpu_data, m_blobSpaitalSumMultiplier.gpu_data, 0.0, m_blobNumByChans.mutable_gpu_data);
                m_cuda.gemv(true, nNum, m_nChannels, 1.0, m_blobNumByChans.gpu_data, m_blobSpaitalSumMultiplier.gpu_data, 0.0, m_blobVariance.mutable_gpu_data); // E((X-EX)^2) 

                // compute and save moving average
                double dfVal = convertD(m_colBlobs[2].GetData(0));
                dfVal *= m_dfMovingAverageFraction;
                dfVal += 1.0;
                m_colBlobs[2].SetData(dfVal, 0);

                m_cuda.axpby(m_blobMean.count(), 1.0, m_blobMean.gpu_data, m_dfMovingAverageFraction, m_colBlobs[0].mutable_gpu_data);
                int nM = colBottom[0].count() / m_nChannels;
                double dfBiasCorrectionFactor = (nM > 1) ? ((double)nM / (double)(nM - 1)) : 1.0;
                m_cuda.axpby(m_blobVariance.count(), dfBiasCorrectionFactor, m_blobVariance.gpu_data, m_dfMovingAverageFraction, m_colBlobs[1].mutable_gpu_data);
            }

            // normalize variance
            m_cuda.add_scalar(m_blobVariance.count(), m_dfEps, m_blobVariance.mutable_gpu_data);
            m_cuda.sqrt(m_blobVariance.count(), m_blobVariance.gpu_data, m_blobVariance.mutable_gpu_data);

            // replicate variance to input size
            m_cuda.gemm(false, false, nNum, m_nChannels, 1, 1.0, m_blobBatchSumMultiplier.gpu_data, m_blobVariance.gpu_data, 0.0, m_blobNumByChans.mutable_gpu_data);
            m_cuda.gemm(false, false, m_nChannels * nNum, nSpatialDim, 1, 1.0, m_blobNumByChans.gpu_data, m_blobSpaitalSumMultiplier.gpu_data, 0.0, m_blobTemp.mutable_gpu_data);
            m_cuda.div(m_blobTemp.count(), hTopData, m_blobTemp.gpu_data, hTopData);
            // The caching is only needed because later in-place layers
            //  might clobber the data.  Can we skip this if they won't?
            m_cuda.copy(m_blobXNorm.count(), hTopData, m_blobXNorm.mutable_gpu_data);
        }

        /// <summary>
        /// Perform the backward computation using the native Cuda version.
        /// </summary>
        protected void backward_cuda(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopDiff = 0;

            if (colBottom[0] != colTop[0])
            {
                hTopDiff = colTop[0].gpu_diff;
            }
            else
            {
                m_cuda.copy(m_blobXNorm.count(), colTop[0].gpu_diff, m_blobXNorm.mutable_gpu_diff);
                hTopDiff = m_blobXNorm.gpu_diff;
            }

            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            if (m_bUseGlobalStats)
            {
                m_cuda.div(m_blobTemp.count(), hTopDiff, m_blobTemp.gpu_data, hBottomDiff);
                return;
            }

            long hTopData = m_blobXNorm.gpu_data;
            int nNum = colBottom[0].shape()[0];
            int nSpatialDim = colBottom[0].count() / (m_nChannels * colBottom[0].shape(0));
            // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
            //
            // dE(y)/dX =
            //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
            //     ./ sqrt(var(X) + eps)
            //
            // where \cdot and ./ are hadamard product and elementwise division,
            // respectively, dE/dY is the top diff, and mean/var/sum are all computed
            // along all dimensions except the channels dimension.  In the above
            // equation, the operations allow for expansion (i.e. broadcast) along all
            // dimensions except the channels dimension where required.

            // sum(dE/dY \cdot Y)
            m_cuda.mul(m_blobTemp.count(), hTopData, hTopDiff, hBottomDiff);
            m_cuda.gemv(false, m_nChannels * nNum, nSpatialDim, 1.0, hBottomDiff, m_blobSpaitalSumMultiplier.gpu_data, 0.0, m_blobNumByChans.mutable_gpu_data);
            m_cuda.gemv(true, nNum, m_nChannels, 1.0, m_blobNumByChans.gpu_data, m_blobBatchSumMultiplier.gpu_data, 0.0, m_blobMean.mutable_gpu_data);

            // reshape (broadcast) the above
            m_cuda.gemm(false, false, nNum, m_nChannels, 1, 1.0, m_blobBatchSumMultiplier.gpu_data, m_blobMean.gpu_data, 0.0, m_blobNumByChans.mutable_gpu_data);
            m_cuda.gemm(false, false, m_nChannels * nNum, nSpatialDim, 1, 1.0, m_blobNumByChans.gpu_data, m_blobSpaitalSumMultiplier.gpu_data, 0.0, hBottomDiff);

            // sum(dE/dY \cdot Y) \cdot Y
            m_cuda.mul(m_blobTemp.count(), hTopData, hBottomDiff, hBottomDiff);

            // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
            m_cuda.gemv(false, m_nChannels * nNum, nSpatialDim, 1.0, hTopDiff, m_blobSpaitalSumMultiplier.gpu_data, 0.0, m_blobNumByChans.mutable_gpu_data);
            m_cuda.gemv(true, nNum, m_nChannels, 1.0, m_blobNumByChans.gpu_data, m_blobBatchSumMultiplier.gpu_data, 0.0, m_blobMean.mutable_gpu_data);

            // reshape (broadcast) the above to make
            // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
            m_cuda.gemm(false, false, nNum, m_nChannels, 1, 1.0, m_blobBatchSumMultiplier.gpu_data, m_blobMean.gpu_data, 0.0, m_blobNumByChans.mutable_gpu_data);
            m_cuda.gemm(false, false, nNum * m_nChannels, nSpatialDim, 1, 1.0, m_blobNumByChans.gpu_data, m_blobSpaitalSumMultiplier.gpu_data, 1.0, hBottomDiff);

            // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
            m_cuda.axpby(m_blobTemp.count(), 1.0, hTopDiff, -1.0 / (double)(nNum * nSpatialDim), hBottomDiff);

            // Note: blobTemp still contains sqrt(var(X) + eps), computed during the forward
            // pass.
            m_cuda.div(m_blobTemp.count(), hBottomDiff, m_blobTemp.gpu_data, hBottomDiff);
        }

        /// <summary>
        /// Perform the forward compuation using cuDNN.
        /// </summary>
        protected void forward_cudnn(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            if (colTop[0] == colBottom[0])
                hTopData = m_blobPrivateTop.mutable_gpu_data;

            double dfEps = m_dfEps;
            long hGlobalMean = m_colBlobs[0].gpu_data;
            long hGlobalVar = m_colBlobs[1].gpu_data;
            long hScaleData = (m_bScaleBias) ? m_colBlobs[3].gpu_data : m_blobScaleOnes.gpu_data;
            long hBiasData = (m_bScaleBias) ? m_colBlobs[4].gpu_data : m_blobBiasZeros.gpu_data;

            if (!m_bUseGlobalStats)
            {
                long hSaveMean = m_blobMean.mutable_gpu_data;
                long hSaveVar = m_blobVariance.mutable_gpu_data;

                hGlobalMean = m_colBlobs[0].mutable_gpu_data;
                hGlobalVar = m_colBlobs[1].mutable_gpu_data;

                double dfFactor = 1.0;

                if (m_nIteration > 0)
                    dfFactor = 1 - m_dfMovingAverageFraction;

                m_cuda.BatchNormForward(m_hCuDnn, m_mode, m_tOne, m_tZero, 
                                        m_hFwdBottomDesc, hBottomData, 
                                        m_hFwdTopDesc, hTopData, 
                                        m_hFwdScaleBiasMeanVarDesc, hScaleData, hBiasData, 
                                        dfFactor, hGlobalMean, hGlobalVar, dfEps, hSaveMean, hSaveVar, true);
            }
            else
            {
                m_cuda.BatchNormForward(m_hCuDnn, BATCHNORM_MODE.SPATIAL, m_tOne, m_tZero, 
                                        m_hFwdBottomDesc, hBottomData, 
                                        m_hFwdTopDesc, hTopData, 
                                        m_hFwdScaleBiasMeanVarDesc, hScaleData, hBiasData, 
                                        1.0, hGlobalMean, hGlobalVar, dfEps, 0, 0, false);
            }

            if (colTop[0] == colBottom[0])
            {
                m_blobPrivateBottom.CopyFrom(colBottom[0]);
                colTop[0].CopyFrom(m_blobPrivateTop);
            }

            m_nIteration++;
        }

        /// <summary>
        /// Perform the backward computation using cuDNN.
        /// </summary>
        protected void backward_cudnn(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomData = colBottom[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            double dfEps = m_dfEps;
            long hSaveMean = m_blobMean.gpu_data;
            long hSaveVar = m_blobVariance.gpu_data;
            long hScaleData = (m_bScaleBias) ? m_colBlobs[3].gpu_data : m_blobScaleOnes.gpu_data;
            long hScaleDiff = (m_bScaleBias) ? m_colBlobs[3].mutable_gpu_diff : m_blobScaleOnes.mutable_gpu_diff;
            long hBiasDiff = (m_bScaleBias) ? m_colBlobs[4].mutable_gpu_diff : m_blobBiasZeros.mutable_gpu_diff;

            if (colTop[0] == colBottom[0])
            {
                // copy diff from top to private top.
                m_blobPrivateTop.CopyFrom(colTop[0], true);
                hTopDiff = m_blobPrivateTop.gpu_diff;
                hBottomData = m_blobPrivateBottom.gpu_data;
            }

            m_cuda.BatchNormBackward(m_hCuDnn, m_mode, m_tOne, m_tZero, m_tOne, m_tOne,
                                        m_hBwdBottomDesc, hBottomData,
                                        m_hBwdBottomDesc, hTopDiff,
                                        m_hBwdBottomDesc, hBottomDiff,
                                        m_hBwdScaleBiasMeanVarDesc, hScaleData, hScaleDiff, hBiasDiff,
                                        dfEps, hSaveMean, hSaveVar);
        }
    }
}
