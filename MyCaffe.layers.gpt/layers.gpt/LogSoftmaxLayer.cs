using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The LogSoftmaxLayer computes the log_softmax function.
    /// This layer is initialized with the MyCaffe.param.LogSoftmaxParameter.
    /// </summary>
    /// <remarks>
    /// @see [Sofmax vs LogSoftmax](https://medium.com/@AbhiramiVS/softmax-vs-logsoftmax-eb94254445a2) by Abhirami V S, Medium, 2021.
    /// @see [Advantage of using LogSoftmax vs Softmax vs Crossentropyloss in PyTorch](https://androidkt.com/advantage-using-logs-softmax-softmax-crossentropyloss-in-pytorch/) by androidkt, 2022.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class LogSoftmaxLayer<T> : Layer<T>
    {
        int m_nOuterNum;
        int m_nInnerNum;
        int m_nSoftmaxAxis;
        Blob<T> m_blobMax;
        Blob<T> m_blobExpX;
        Blob<T> m_blobExpXSum;
        Blob<T> m_blobScale;

        /// <summary>
        /// The SoftmaxLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type LOG_SOFTMAX with parameter log_softmax_param,
        /// with options:
        ///   - axis (\b optional, default = 1). The axis along which to perform the softmax.
        /// </param>
        public LogSoftmaxLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.LOG_SOFTMAX;
            m_blobMax = new Blob<T>(m_cuda, m_log);
            m_blobMax.Name = m_param.name + " max";
            m_blobExpX = new Blob<T>(cuda, log);
            m_blobExpX.Name = m_param.name + " exp_x";
            m_blobExpXSum = new Blob<T>(cuda, log);
            m_blobExpXSum.Name = m_param.name + " exp_x_sum";
            m_blobScale = new Blob<T>(cuda, log);
            m_blobScale.Name = m_param.name + " scale";

            setup_internal_blobs(m_colInternalBlobs);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobMax);
            dispose(ref m_blobExpX);
            dispose(ref m_blobExpXSum);
            dispose(ref m_blobScale);
            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobMax);
            col.Add(m_blobExpX);
            col.Add(m_blobExpXSum);
            col.Add(m_blobScale);
        }

        /// <summary>
        /// Returns the exact number of bottom blobs (input) Blobs: input
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
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
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nSoftmaxAxis = colBottom[0].CanonicalAxisIndex(m_param.log_softmax_param.axis);

            colTop[0].ReshapeLike(colBottom[0]);

            shareLayerBlob(m_blobExpX, colTop[0].shape());
            m_blobExpX.ReshapeLike(colTop[0]);

            m_nOuterNum = colBottom[0].count(0, m_nSoftmaxAxis);
            m_nInnerNum = colBottom[0].count(m_nSoftmaxAxis + 1);

            List<int> rgScaleDims = Utility.Clone<int>(colBottom[0].shape());
            rgScaleDims[m_nSoftmaxAxis] = 1;
            
            shareLayerBlob(m_blobMax, rgScaleDims);
            m_blobMax.Reshape(rgScaleDims);
            shareLayerBlob(m_blobScale, rgScaleDims);
            m_blobScale.Reshape(rgScaleDims);
            shareLayerBlob(m_blobExpXSum, rgScaleDims);
            m_blobExpXSum.Reshape(rgScaleDims);
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the computed outputs.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            long hMaxData = m_blobMax.mutable_gpu_data;
            long hScaleData = m_blobScale.mutable_gpu_data;
            int nCount = colBottom[0].count();
            int nChannels = colTop[0].shape(m_nSoftmaxAxis);

            m_cuda.copy(nCount, hBottomData, hTopData);

            // We need to subtract the max to avoid numerical issues, compute the exp
            // and then normalize.
            // c = channel max along axis.
            m_cuda.channel_max(m_nOuterNum * m_nInnerNum, m_nOuterNum, nChannels, m_nInnerNum, hTopData, hMaxData);

            // xm = x - c (along each channel)
            m_cuda.channel_sub(nCount, m_nOuterNum, nChannels, m_nInnerNum, hMaxData, hTopData);

            // exp_x = exp(xm)
            m_cuda.exp(nCount, hTopData, m_blobExpX.mutable_gpu_data);

            // exp_sum = exp_x.sum(dim=axis)
            m_cuda.channel_sum(m_nOuterNum * m_nInnerNum, m_nOuterNum, nChannels, m_nInnerNum, m_blobExpX.gpu_data, m_blobExpXSum.mutable_gpu_data);

            // exp_log = exp_sum.log()
            m_cuda.log(m_nOuterNum * m_nInnerNum, m_blobExpXSum.gpu_data, hScaleData);
            
            // log_z = c + exp_log
            m_cuda.add(m_nOuterNum * m_nInnerNum, hMaxData, hScaleData, hScaleData);

            // sm = x - log_z
            m_cuda.copy(nCount, hBottomData, hTopData);
            m_cuda.channel_sub(nCount, m_nOuterNum, nChannels, m_nInnerNum, hScaleData, hTopData);
        }


        /// <summary>
        /// Computes the error gradient w.r.t the inputs.
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
            long hTopDiff = colTop[0].gpu_diff;
            long hTopData = colTop[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int nCount = colTop[0].count();
            int nChannels = colTop[0].shape(m_nSoftmaxAxis);

            // exp_log_grad = -1 * sum channel diff
            m_cuda.channel_sum(nCount, m_nOuterNum, nChannels, m_nInnerNum, hTopDiff, m_blobScale.mutable_gpu_diff);

            // expy = exp(y)
            m_cuda.exp(nCount, hTopData, m_blobExpX.mutable_gpu_data);
            
            // Fill the expy values across each channel.
            m_cuda.channel_fillfrom(nCount, m_nOuterNum, 1, nChannels, m_blobScale.gpu_diff, m_blobExpX.mutable_gpu_diff, DIR.FWD);

            // expy * sumgy
            m_cuda.mul(nCount, m_blobExpX.gpu_data, m_blobExpX.gpu_diff, m_blobExpX.mutable_gpu_diff);

            // grad = gy - expy * sumgy
            m_cuda.sub(nCount, hTopDiff, m_blobExpX.gpu_diff, hBottomDiff);
        }
    }
}
