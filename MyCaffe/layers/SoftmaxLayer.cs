﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The SoftmaxLayer computes the softmax function.
    /// This layer is initialized with the MyCaffe.param.SoftmaxParameter.
    /// </summary>
    /// <remarks>
    /// @see [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580v1) by Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R. Salakhutdinov, 2012.
    /// @see [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144v2) by Wu, et al., 2016.
    /// @see [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538v1) by Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean, 2017.
    /// @see [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410v2) by Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SoftmaxLayer<T> : Layer<T>
    {
        int m_nOuterNum;
        int m_nInnerNum;
        int m_nSoftmaxAxis;
        Blob<T> m_blobScale;
        SOFTMAX_ALGORITHM m_algorithm = SOFTMAX_ALGORITHM.DEFAULT;
        SOFTMAX_MODE m_mode = SOFTMAX_MODE.CHANNEL;
        List<int> m_rgScaleDims = null;

        long m_hCudnn = 0;
        long m_hBottomDesc = 0;
        long m_hTopDesc = 0;

        /// <summary>
        /// The SoftmaxLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type SOFTMAX with parameter softmax_param,
        /// with options:
        ///   - engine. The engine to use, either Engine.CAFFE, or Engine.CUDNN.
        ///   
        ///   - axis (\b optional, default = 1). The axis along which to perform the softmax.
        /// </param>
        public SoftmaxLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SOFTMAX;
            m_blobScale = new Blob<T>(cuda, log);
            m_blobScale.Name = m_param.name + " scale";

            setup_internal_blobs(m_colInternalBlobs);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_hCudnn != 0)
            {
                m_cuda.FreeCuDNN(m_hCudnn);
                m_hCudnn = 0;
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
            
            m_blobScale.Dispose();
            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            if (!m_param.softmax_param.useCudnn())
            {
                col.Add(m_blobScale);
            }
        }

        /// <summary>
        /// Returns the minimum number of bottom blobs (input) Blobs: input.
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of bottom blobs (input) Blobs: input, target (ignored)
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: softmax
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer to run in either Engine.CAFFE or Engine.CUDNN mode.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_algorithm = m_param.softmax_param.algorithm;
            if (m_param.softmax_param.engine == EngineParameter.Engine.CAFFE)
            {
                if (m_algorithm != SOFTMAX_ALGORITHM.ACCURATE)
                    m_log.WriteLine("WARNING: SoftmaxLayer: Caffe mode does not support the ACCURATE algorithm, the default FAST algorithm will be used instead.");
            }

            if (!m_param.softmax_param.useCudnn())
                return;

            // Initialize cuDNN
            m_hCudnn = m_cuda.CreateCuDNN();
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
            m_nSoftmaxAxis = colBottom[0].CanonicalAxisIndex(m_param.softmax_param.axis);

            if (m_phase == Phase.TRAIN && m_param.softmax_param.algorithm_train.HasValue)
                m_algorithm = m_param.softmax_param.algorithm_train.Value;

            colTop[0].ReshapeLike(colBottom[0]);

            m_nOuterNum = colBottom[0].count(0, m_nSoftmaxAxis);
            m_nInnerNum = colBottom[0].count(m_nSoftmaxAxis + 1);

            if (!m_param.softmax_param.useCudnn())
            {
                shareLayerBlob(m_blobScale, colBottom[0].shape());
                m_blobScale.ReshapeLike(colBottom[0]);

                m_rgScaleDims = Utility.Clone<int>(colBottom[0].shape());
                m_rgScaleDims[m_nSoftmaxAxis] = 1;
                
                m_blobScale.Reshape(m_rgScaleDims);

                return;
            }

            int nN = m_nOuterNum;
            int nK = colBottom[0].shape(m_nSoftmaxAxis);
            int nH = m_nInnerNum;
            int nW = 1;

            if (nH == 1 && nW == 1)
                m_mode = SOFTMAX_MODE.INSTANCE;

            m_cuda.SetTensorDesc(m_hBottomDesc, nN, nK, nH, nW);
            m_cuda.SetTensorDesc(m_hTopDesc, nN, nK, nH, nW);
        }

        /// <summary>
        /// Computes the forward calculation using either the Engine.CAFFE or Engine.CUDNN mode.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the computed outputs.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (!m_param.softmax_param.useCudnn())
                forward_cuda(colBottom, colTop);
            else
                forward_cudnn(colBottom, colTop);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs using either the Engine.CAFFE or Engine.CUDNN mode.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     containing error gradients with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!m_param.softmax_param.useCudnn())
                backward_cuda(colTop, rgbPropagateDown, colBottom);
            else
                backward_cudnn(colTop, rgbPropagateDown, colBottom);
        }

        /// <summary>
        /// Computes the forward calculation using the Engine.CAFFE mode.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the computed outputs.
        /// </param>
        protected void forward_cuda(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            long hScaleData = m_blobScale.mutable_gpu_data;
            long hScaleDiff = m_blobScale.mutable_gpu_diff;
            int nCount = colBottom[0].count();
            int nChannels = colTop[0].shape(m_nSoftmaxAxis);

            m_cuda.copy(nCount, hBottomData, hTopData);

            // We need to subtract the max to avoid numerical issues, compute the exp
            // and then normalize.
            // c = x.max(dim=axis)
            m_cuda.channel_max(m_nOuterNum * m_nInnerNum, m_nOuterNum, nChannels, m_nInnerNum, hTopData, hScaleData);

            // Subtract c
            // xm = x - c (along each channel)
            m_cuda.channel_sub(nCount, m_nOuterNum, nChannels, m_nInnerNum, hScaleData, hTopData);

            // exponentiate
            // exp_x = exp(xm)
            m_cuda.exp(nCount, hTopData, hTopData);

            // Sum across each channel after exp
            // exp_sum = exp_x.sum(dim=axis)
            m_cuda.channel_sum(m_nOuterNum * m_nInnerNum, m_nOuterNum, nChannels, m_nInnerNum, hTopData, hScaleDiff);

            if (m_param.softmax_param.algorithm == SOFTMAX_ALGORITHM.LOG)
            {
                // exp_log = exp_sum.log()
                m_cuda.log(m_nOuterNum * m_nInnerNum, hScaleDiff, hScaleDiff);
                // log_z = c + exp_log
                m_cuda.add(m_nOuterNum * m_nInnerNum, hScaleData, hScaleDiff, hScaleData);
                // sm = x - log_z
                m_cuda.copy(nCount, hBottomData, hTopData);
                m_cuda.channel_sub(nCount, m_nOuterNum, nChannels, m_nInnerNum, hScaleData, hTopData);
            }
            else
            {
                // divide            
                m_cuda.channel_div(nCount, m_nOuterNum, nChannels, m_nInnerNum, hScaleDiff, hTopData);
            }
        }


        /// <summary>
        /// Computes the error gradient w.r.t the inputs using either the Engine.CAFFE.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     containing error gradients with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs.
        /// </param>
        protected void backward_cuda(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopDiff = colTop[0].gpu_diff;
            long hTopData = colTop[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            long hScaleData = m_blobScale.mutable_gpu_data;
            int nCount = colTop[0].count();
            int nChannels = colTop[0].shape(m_nSoftmaxAxis);

            if (m_param.softmax_param.algorithm == SOFTMAX_ALGORITHM.LOG)
            {
                // sumgy = sum channel diff
                m_blobScale.Reshape(m_rgScaleDims);
                m_cuda.channel_sum(nCount, m_nOuterNum, nChannels, m_nInnerNum, hTopDiff, m_blobScale.mutable_gpu_diff);

                // expy = exp(y)
                m_cuda.exp(nCount, hTopData, hBottomDiff);

                // Fill the expy values across each channel.
                m_blobScale.ReshapeLike(colBottom[0]);
                m_cuda.channel_fillfrom(nCount, m_nOuterNum, 1, nChannels, m_blobScale.gpu_diff, m_blobScale.mutable_gpu_data, DIR.FWD);

                // expy * sumgy
                m_cuda.mul(nCount, hBottomDiff, m_blobScale.gpu_data, hBottomDiff);
                m_blobScale.Reshape(m_rgScaleDims);

                // grad = gy - (expy * sumgy)
                m_cuda.sub(nCount, hTopDiff, hBottomDiff, hBottomDiff);
            }
            else
            {
                m_cuda.copy(nCount, hTopDiff, hBottomDiff);

                // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
                m_cuda.channel_dot(m_nOuterNum * m_nInnerNum, m_nOuterNum, nChannels, m_nInnerNum, hTopDiff, hTopData, hScaleData);
                m_cuda.channel_sub(nCount, m_nOuterNum, nChannels, m_nInnerNum, hScaleData, hBottomDiff);

                // elementwise multiplication
                m_cuda.mul(nCount, hBottomDiff, hTopData, hBottomDiff);
            }
        }

        /// <summary>
        /// Computes the forward calculation using the Engine.CUDNN mode.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the computed outputs.
        /// </param>
        protected void forward_cudnn(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;

            m_cuda.SoftmaxForward(m_hCudnn, m_algorithm, m_mode, m_tOne, m_hBottomDesc, hBottomData, m_tZero, m_hTopDesc, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs using either the Engine.CUDNN.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     containing error gradients with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs.
        /// </param>
        protected void backward_cudnn(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopData = colTop[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomData = colBottom[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;

            m_cuda.SoftmaxBackward(m_hCudnn, m_algorithm, m_mode, m_tOne, m_hTopDesc, hTopData, m_hTopDesc, hTopDiff, m_tZero, m_hBottomDesc, hBottomDiff);
        }
    }
}
