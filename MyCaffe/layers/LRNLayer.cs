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
    /// The "Local Response Normalization" LRNLayer is used to normalize the input in a local region across or within feature maps.
    /// This layer is initialized with the MyCaffe.param.LRNParameter.
    /// </summary>
    /// <remarks>
    /// @see [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) by Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R. Salakhutdinov, 2012.
    /// @see [Layer Normalization](https://arxiv.org/abs/1607.06450) by Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class LRNLayer<T> : Layer<T>
    {
        int m_nSize;
        int m_nPrePad;
        double m_dfAlpha;
        double m_dfBeta;
        double m_dfK;
        int m_nNum;
        int m_nChannels;
        int m_nHeight;
        int m_nWidth;

        // Fields used for normalization ACROSS_CHANNELS
        // scale_ stores the intermediate summing results.
        Blob<T> m_blobScale;

        // Fields for normalization WITHIN_CHANNEL
        SplitLayer<T> m_splitLayer;
        BlobCollection<T> m_colSplitTopVec = new BlobCollection<T>();
        PowerLayer<T> m_squareLayer;
        Blob<T> m_blobSquareInput;
        Blob<T> m_blobSquareOutput;
        BlobCollection<T> m_colSquareBottomVec = new BlobCollection<T>();
        BlobCollection<T> m_colSquareTopVec = new BlobCollection<T>();
        PoolingLayer<T> m_poolLayer;
        Blob<T> m_blobPoolOutput;
        BlobCollection<T> m_colPoolTopVec = new BlobCollection<T>();
        PowerLayer<T> m_powerLayer;
        Blob<T> m_blobPowerOutput;
        BlobCollection<T> m_colPowerTopVec = new BlobCollection<T>();
        EltwiseLayer<T> m_productLayer;
        Blob<T> m_blobProductInput;
        BlobCollection<T> m_colProductBottomVec = new BlobCollection<T>();

        // cuDnn - lrn
        long m_hCuDnn = 0;
        long m_hNormDesc = 0;
        long m_hBottomDesc = 0;
        long m_hTopDesc = 0;

        // cuDnn - lcn
        int m_nTempDataSize;
        long m_hTempData1;
        long m_hTempData2;

        /// <summary>
        /// The LRNLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type LRN with parameter lrn_param,
        /// with options:
        ///     - engine (\b optional, default Engine.CUDNN for ACROSS_CHANNELS, Engine.CAFFE for WITHIN_CHANNELS). The engine (Engine.CUDNN or Engine.CAFFE) to use.
        ///     
        ///     - norm (\b optional, default ACROSS_CHANNELS). The region to normalize 
        ///     
        ///     - local_size (\b optional, default 5). The local size of the normalization window.
        ///     
        ///     - alpha (\b optional, default 1e-4).  The alpha value used for variance scaling.  Note: cuDnn uses a default of 1e-4, whereas Caffe uses a default of 1.0.
        ///     
        ///     - beta (\b optional, default 0.75). The beta value used as the power parameter.
        ///     
        ///     - k (\b optional, default 2.0). The k value used during normalization.  Note: cuDnn uses a default of 2.0, whereas Caffe uses a default of 1.0.
        /// </param>
        public LRNLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.LRN;
            m_blobScale = new Blob<T>(cuda, log);
            m_blobScale.Name = "lrn_scale";
            m_blobSquareInput = new Blob<T>(cuda, log);
            m_blobSquareInput.Name = "lrn_sqin";
            m_blobSquareOutput = new Blob<T>(cuda, log);
            m_blobSquareOutput.Name = "lrn_sqout";
            m_blobPoolOutput = new Blob<T>(cuda, log);
            m_blobPoolOutput.Name = "lrn_poolout";
            m_blobPowerOutput = new Blob<T>(cuda, log);
            m_blobPowerOutput.Name = "lrn_powout";
            m_blobProductInput = new Blob<T>(cuda, log);
            m_blobProductInput.Name = "lrn_prodin";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobScale.Dispose();
            m_blobSquareInput.Dispose();
            m_blobSquareOutput.Dispose();
            m_blobPoolOutput.Dispose();
            m_blobPowerOutput.Dispose();
            m_blobProductInput.Dispose();

            if (m_splitLayer != null)
            {
                m_splitLayer.Dispose();
                m_splitLayer = null;
            }

            if (m_squareLayer != null)
            {
                m_squareLayer.Dispose();
                m_squareLayer = null;
            }

            if (m_poolLayer != null)
            {
                m_poolLayer.Dispose();
                m_poolLayer = null;
            }

            if (m_powerLayer != null)
            {
                m_powerLayer.Dispose();
                m_powerLayer = null;
            }

            if (m_powerLayer != null)
            {
                m_productLayer.Dispose();
                m_powerLayer = null;
            }

            if (m_hNormDesc != 0)
            {
                m_cuda.FreeLRNDesc(m_hNormDesc);
                m_hNormDesc = 0;
            }

            if (m_hBottomDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hBottomDesc);
                m_hBottomDesc = 0;
            }

            if (m_hTopDesc != 0)
            {
                m_cuda.FreeTensorDesc(m_hTopDesc);
                m_hTopDesc = 0;
            }

            if (m_hCuDnn != 0)
            {
                m_cuda.FreeCuDNN(m_hCuDnn);
                m_hCuDnn = 0;
            }

            if (m_hTempData1 != 0)
            {
                m_cuda.FreeMemory(m_hTempData1);
                m_hTempData1 = 0;
            }

            if (m_hTempData2 != 0)
            {
                m_cuda.FreeMemory(m_hTempData2);
                m_hTempData2 = 0;
            }

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                if (!m_param.lrn_param.useCudnn())
                {
                    if (m_param.lrn_param.norm_region == LRNParameter.NormRegion.ACROSS_CHANNELS)
                    {
                        col.Add(m_blobScale);
                    }
                    else
                    {
                        col.Add(m_blobSquareInput);
                        col.Add(m_blobSquareOutput);
                        col.Add(m_blobPoolOutput);
                        col.Add(m_blobPowerOutput);
                        col.Add(m_blobProductInput);
                    }
                }

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: lrn
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer for both Engine.CUDNN and Engine.CAFFE modes.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nSize = (int)m_param.lrn_param.local_size;
            m_log.CHECK_EQ(m_nSize % 2, 1, "LRN only supports odd values for local_size.");
            m_nPrePad = (m_nSize - 1) / 2;
            m_dfAlpha = m_param.lrn_param.alpha;
            m_dfBeta = m_param.lrn_param.beta;
            m_dfK = m_param.lrn_param.k;

            if (m_param.lrn_param.norm_region == LRNParameter.NormRegion.WITHIN_CHANNEL)
            {
                // Set up split_layer to use in the numerator and denominator.
                m_colSplitTopVec = new BlobCollection<T>();
                m_colSplitTopVec.Add(m_blobProductInput);
                m_colSplitTopVec.Add(m_blobSquareInput);
                LayerParameter split_param = new LayerParameter(LayerParameter.LayerType.SPLIT, "split");
                m_splitLayer = new SplitLayer<T>(m_cuda, m_log, split_param);
                m_splitLayer.Setup(colBottom, m_colSplitTopVec);

                // Set up square_layer to square teh inputs.
                m_colSquareBottomVec = new BlobCollection<T>();
                m_colSquareTopVec = new BlobCollection<T>();
                m_colSquareBottomVec.Add(m_blobSquareInput);
                m_colSquareTopVec.Add(m_blobSquareOutput);
                LayerParameter square_param = new LayerParameter(LayerParameter.LayerType.POWER, "square");
                square_param.power_param.power = 2.0;
                m_squareLayer = new PowerLayer<T>(m_cuda, m_log, square_param);
                m_squareLayer.Setup(m_colSquareBottomVec, m_colSquareTopVec);

                // Set up pool_layer to sum over square neighborhoods of the input.
                m_colPoolTopVec = new BlobCollection<T>();
                m_colPoolTopVec.Add(m_blobPoolOutput);
                LayerParameter pool_param = new LayerParameter(LayerParameter.LayerType.POOLING, "pool");
                pool_param.pooling_param.pool = PoolingParameter.PoolingMethod.AVE;
                pool_param.pooling_param.pad.Add((uint)m_nPrePad);
                pool_param.pooling_param.kernel_size.Add((uint)m_nSize);
                m_poolLayer = new PoolingLayer<T>(m_cuda, m_log, pool_param);
                m_poolLayer.Setup(m_colSquareTopVec, m_colPoolTopVec);

                // Set up power_layer to compute (1 + alpha/N^2 s)^-beta, where s is
                // the sum of the squared neighborhood (the output of pool_layer)
                m_colPowerTopVec = new BlobCollection<T>();
                m_colPowerTopVec.Add(m_blobPowerOutput);
                LayerParameter power_param = new LayerParameter(LayerParameter.LayerType.POWER, "power");
                power_param.power_param.power = -m_dfBeta;
                power_param.power_param.scale = m_dfAlpha;
                power_param.power_param.shift = 1.0;
                m_powerLayer = new PowerLayer<T>(m_cuda, m_log, power_param);
                m_powerLayer.Setup(m_colPoolTopVec, m_colPowerTopVec);

                // Set up a product_layer to compute outputs by multiplying inputs by the
                // inverse denominator computed by the power layer.
                m_colProductBottomVec = new BlobCollection<T>();
                m_colProductBottomVec.Add(m_blobProductInput);
                m_colProductBottomVec.Add(m_blobPowerOutput);
                LayerParameter product_param = new LayerParameter(LayerParameter.LayerType.ELTWISE, "product");
                product_param.eltwise_param.operation = EltwiseParameter.EltwiseOp.PROD;
                m_productLayer = new EltwiseLayer<T>(m_cuda, m_log, product_param);
                m_productLayer.Setup(m_colProductBottomVec, colTop);
            }

            if (!m_param.lrn_param.useCudnn())
                return;

            m_hCuDnn = m_cuda.CreateCuDNN();
            m_hNormDesc = m_cuda.CreateLRNDesc();
            m_hBottomDesc = m_cuda.CreateTensorDesc();
            m_hTopDesc = m_cuda.CreateTensorDesc();
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(4, colBottom[0].num_axes, "Input must have 4 axes, corresponding to (num, channels, height, width)");
            m_nNum = colBottom[0].num;
            m_nChannels = colBottom[0].channels;
            m_nHeight = colBottom[0].height;
            m_nWidth = colBottom[0].width;

            switch (m_param.lrn_param.norm_region)
            {
                case LRNParameter.NormRegion.ACROSS_CHANNELS:
                    colTop[0].Reshape(m_nNum, m_nChannels, m_nHeight, m_nWidth);
                    if (!m_param.lrn_param.useCudnn())
                        m_blobScale.Reshape(m_nNum, m_nChannels, m_nHeight, m_nWidth);
                    break;

                case LRNParameter.NormRegion.WITHIN_CHANNEL:
                    m_splitLayer.Reshape(colBottom, m_colSplitTopVec);
                    m_squareLayer.Reshape(m_colSquareBottomVec, m_colSquareTopVec);
                    m_poolLayer.Reshape(m_colSquareTopVec, m_colPoolTopVec);
                    m_powerLayer.Reshape(m_colPoolTopVec, m_colPowerTopVec);
                    m_productLayer.Reshape(m_colProductBottomVec, colTop);
                    break;
            }

            if (!m_param.lrn_param.useCudnn())
                return;

            m_cuda.SetTensorDesc(m_hBottomDesc, m_nNum, m_nChannels, m_nHeight, m_nWidth);
            m_cuda.SetTensorDesc(m_hTopDesc, m_nNum, m_nChannels, m_nHeight, m_nWidth);
            m_cuda.SetLRNDesc(m_hNormDesc, (uint)m_nSize, m_dfAlpha, m_dfBeta, m_dfK);

            if (m_param.lrn_param.norm_region == LRNParameter.NormRegion.WITHIN_CHANNEL)
            {
                int nTotalSize = m_nNum * m_nChannels * m_nHeight * m_nWidth;

                if (nTotalSize > m_nTempDataSize)
                {
                    if (m_hTempData1 != 0)
                    {
                        m_cuda.FreeMemory(m_hTempData1);
                        m_hTempData1 = 0;
                    }

                    if (m_hTempData2 != 0)
                    {
                        m_cuda.FreeMemory(m_hTempData2);
                        m_hTempData2 = 0;
                    }

                    m_hTempData1 = m_cuda.AllocMemory(nTotalSize);
                    m_hTempData2 = m_cuda.AllocMemory(nTotalSize);
                    m_nTempDataSize = nTotalSize;
                }
            }
        }

        /// <summary>
        /// Forward computation using either the Engine.CUDNN or Engine.CAFFE mode depending on the 
        /// <i>engine</i> parameter setting.
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     normalized outputs.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!m_param.lrn_param.useCudnn())
                forward_cuda(colBottom, colTop);
            else
                forward_cudnn(colBottom, colTop);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the inputs using either the Engine.CUDNN or Engine.CAFFE mode depending on the 
        /// <i>engine</i> parameter setting.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///  </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!m_param.lrn_param.useCudnn())
                backward_cuda(colTop, rgbPropagateDown, colBottom);
            else
                backward_cudnn(colTop, rgbPropagateDown, colBottom);
        }

        /// <summary>
        /// Forward computation using the Engine.CAFFE mode.
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs.
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     normalized outputs.
        /// </param>
        protected void forward_cuda(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            switch (m_param.lrn_param.norm_region)
            {
                case LRNParameter.NormRegion.ACROSS_CHANNELS:
                    CrossChannelForward(colBottom, colTop);
                    break;

                case LRNParameter.NormRegion.WITHIN_CHANNEL:
                    WithinChannelForward(colBottom, colTop);
                    break;

                default:
                    m_log.FAIL("Unknown normalization region.");
                    break;
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the inputs using the Engine.CAFFE mode.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the error gradients.
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        /// </param>
        protected void backward_cuda(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            switch (m_param.lrn_param.norm_region)
            {
                case LRNParameter.NormRegion.ACROSS_CHANNELS:
                    CrossChannelBackward(colTop, rgbPropagateDown, colBottom);
                    break;

                case LRNParameter.NormRegion.WITHIN_CHANNEL:
                    WithinChannelBackward(colTop, rgbPropagateDown, colBottom);
                    break;

                default:
                    m_log.FAIL("Unknown normalization region.");
                    break;
            }
        }

        /// <summary>
        /// Forward computation using the Engine.CUDNN mode.
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     normalized outputs.
        /// </param>
        protected void forward_cudnn(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            if (m_param.lrn_param.norm_region == LRNParameter.NormRegion.WITHIN_CHANNEL)
                m_cuda.DivisiveNormalizationForward(m_hCuDnn, m_hNormDesc, m_tOne, m_hBottomDesc, hBottomData, m_hTempData1, m_hTempData2, m_tZero, m_hTopDesc, hTopData);
            else
                m_cuda.LRNCrossChannelForward(m_hCuDnn, m_hNormDesc, m_tOne, m_hBottomDesc, hBottomData, m_tZero, m_hTopDesc, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the inputs using the Engine.CUDNN mode.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$</param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///  </param>
        protected void backward_cudnn(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopDiff = colTop[0].gpu_diff;
            long hTopData = colTop[0].gpu_data;
            long hBottomData = colBottom[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            if (m_param.lrn_param.norm_region == LRNParameter.NormRegion.WITHIN_CHANNEL)
                m_cuda.DivisiveNormalizationBackward(m_hCuDnn, m_hNormDesc, m_tOne, m_hBottomDesc, hBottomData, hTopDiff, m_hTempData1, m_hTempData2, m_tZero, m_hBottomDesc, hBottomDiff);
            else
                m_cuda.LRNCrossChannelBackward(m_hCuDnn, m_hNormDesc, m_tOne, m_hTopDesc, hTopData, m_hTopDesc, hTopDiff, m_hBottomDesc, hBottomData, m_tZero, m_hBottomDesc, hBottomDiff);
        }

        private void CrossChannelForward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // First, compute scale
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            long hScaleData = m_blobScale.mutable_gpu_data;

            // We will launch one kernel for each pixel location, and have the kernel
            // go through all of the channels.
            int nThreads = m_nNum * m_nHeight * m_nWidth;
            m_cuda.lrn_fillscale(nThreads, hBottomData, m_nNum, m_nChannels, m_nHeight, m_nWidth, m_nSize, convert(m_dfAlpha / m_nSize), convert(m_dfK), hScaleData);

            nThreads = colBottom[0].count();
            m_cuda.lrn_computeoutput(nThreads, hBottomData, hScaleData, convert(-m_dfBeta), hTopData);
        }

        private void WithinChannelForward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_splitLayer.Forward(colBottom, m_colSplitTopVec);
            m_squareLayer.Forward(m_colSquareBottomVec, m_colSquareTopVec);
            m_poolLayer.Forward(m_colSquareTopVec, m_colPoolTopVec);
            m_powerLayer.Forward(m_colPoolTopVec, m_colPowerTopVec);
            m_productLayer.Forward(m_colProductBottomVec, colTop);
        }

        private void CrossChannelBackward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            int nThreads = m_nNum * m_nHeight * m_nWidth;
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].gpu_data;
            long hScaleData = m_blobScale.gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            m_cuda.lrn_computediff(nThreads, hBottomData, hTopData, hScaleData, hTopDiff, m_nNum, m_nChannels, m_nHeight, m_nWidth, m_nSize, convert(-m_dfBeta), convert(2.0 * m_dfAlpha * m_dfBeta / m_nSize), hBottomDiff);
        }

        private void WithinChannelBackward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
            {
                List<bool> rgbProductPropagateDown = Utility.Create<bool>(2, true);
                m_productLayer.Backward(colTop, rgbProductPropagateDown, m_colProductBottomVec);
                m_powerLayer.Backward(m_colPowerTopVec, rgbPropagateDown, m_colPoolTopVec);
                m_poolLayer.Backward(m_colPoolTopVec, rgbPropagateDown, m_colSquareTopVec);
                m_squareLayer.Backward(m_colSquareTopVec, rgbPropagateDown, m_colSquareBottomVec);
                m_splitLayer.Backward(m_colSplitTopVec, rgbPropagateDown, colBottom);
            }
        }
    }
}
