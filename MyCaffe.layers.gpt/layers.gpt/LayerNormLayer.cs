using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The LayerNormalizationLayer performs layer normalization similar to the PyTorch LayerNorm layer.
    /// </summary>
    /// <remarks>
    /// @see [GitHub:CyberZHG](https://github.com/CyberZHG/torch-layer-normalization/blob/master/torch_layer_normalization/layer_normalization.py) by Zhao HG (MIT Liceense).
    /// @see [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) PyTorch
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class LayerNormLayer<T> : Layer<T>
    {
        Blob<T> m_blobWork;
        Blob<T> m_blobMu;
        Blob<T> m_blobXmu;
        Blob<T> m_blobXmuSq;
        Blob<T> m_blobVar;
        Blob<T> m_blobStdev;
        Blob<T> m_blobStdevFull;
        long m_hLayerNorm = 0;
        int m_nCount = 0;
        int m_nOuterNum = 0;
        int m_nChannels = 0;
        int m_nInnerNum = 0;

        /// <summary>
        /// The LayerNormalizationLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type LayerNormalizationLayer with parameter layer_norm_param,
        /// with options:
        ///   - epsilon (\b optional, default 1e-10). The epsilon value used to avoid Nan values.
        /// </param>
        public LayerNormLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.LAYERNORM;

            m_blobWork = new Blob<T>(cuda, log);
            m_blobWork.Name = m_param.name + " work";
            m_blobMu = new Blob<T>(cuda, log);
            m_blobMu.Name = m_param.name + " mu";
            m_blobXmu = new Blob<T>(cuda, log);
            m_blobXmu.Name = m_param.name + " xmu";
            m_blobXmuSq = new Blob<T>(cuda, log);
            m_blobXmuSq.Name = m_param.name + " xmu_sq";
            m_blobVar = new Blob<T>(cuda, log);
            m_blobVar.Name = m_param.name + " var";
            m_blobStdev = new Blob<T>(cuda, log);
            m_blobStdev.Name = m_param.name + " stdev";
            m_blobStdevFull = new Blob<T>(cuda, log);
            m_blobStdevFull.Name = m_param.name + " stdev_full";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobWork);
            dispose(ref m_blobMu);
            dispose(ref m_blobXmu);
            dispose(ref m_blobXmuSq);
            dispose(ref m_blobVar);
            dispose(ref m_blobStdev);
            dispose(ref m_blobStdevFull);

            if (m_hLayerNorm != 0)
            {
                m_cuda.FreeLayerNorm(m_hLayerNorm);
                m_hLayerNorm = 0;
            }
            
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobWork);
                col.Add(m_blobMu);
                col.Add(m_blobXmu);
                col.Add(m_blobXmuSq);
                col.Add(m_blobVar);
                col.Add(m_blobStdev);
                col.Add(m_blobStdevFull);

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: data
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: norm
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
            if (m_param.layer_norm_param.enable_cuda_impl)
            {
                if (colBottom[0].count() != m_nCount || colBottom[0].num != m_nOuterNum || colBottom[0].channels != m_nChannels || colBottom[0].count(2) != m_nInnerNum)
                {
                    if (m_hLayerNorm != 0)
                        m_cuda.FreeLayerNorm(m_hLayerNorm);

                    int nGpuID = m_cuda.GetDeviceID();
                    m_nCount = colBottom[0].count();
                    m_nOuterNum = colBottom[0].num;
                    m_nChannels = colBottom[0].channels;
                    m_nInnerNum = colBottom[0].count(2);

                    m_hLayerNorm = m_cuda.CreateLayerNorm(nGpuID, m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, (float)m_param.layer_norm_param.epsilon);
                    if (m_hLayerNorm == 0)
                        m_log.FAIL("Failed to create CUDA version LayerNorm!");
                }
            }
            else
            {
                if (!shareLayerBlob(m_blobWork, colBottom[0].shape()))
                    m_blobWork.ReshapeLike(colBottom[0]);
                if (!shareLayerBlob(m_blobMu, colBottom[0].shape()))
                    m_blobMu.ReshapeLike(colBottom[0]);
                if (!shareLayerBlob(m_blobXmu, colBottom[0].shape()))
                    m_blobXmu.ReshapeLike(colBottom[0]);
                if (!shareLayerBlob(m_blobXmuSq, colBottom[0].shape()))
                    m_blobXmuSq.ReshapeLike(colBottom[0]);
                if (!shareLayerBlob(m_blobVar, colBottom[0].shape()))
                    m_blobVar.ReshapeLike(colBottom[0]);
                if (!shareLayerBlob(m_blobStdev, colBottom[0].shape()))
                    m_blobStdev.ReshapeLike(colBottom[0]);
                if (!shareLayerBlob(m_blobStdevFull, colBottom[0].shape()))
                    m_blobStdevFull.ReshapeLike(colBottom[0]);
            }
            
            colTop[0].ReshapeLike(colBottom[0]);
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the outputs.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_param.layer_norm_param.enable_cuda_impl)
                m_cuda.LayerNormForward(m_hLayerNorm, colBottom[0].gpu_data, colTop[0].mutable_gpu_data);
            else
                forward_local(colBottom, colTop);
        }

        private void forward_local(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nAxes = colBottom[0].num_axes;
            int nCount = colBottom[0].count();
            int nOuterNum = colBottom[0].num;
            int nChannel = colBottom[0].channels;
            int nInnerNum = colBottom[0].count(2);

            //-----------------------------------
            // Calculate the mean across the last dim.
            // mean = x.mean(dim=-1, keepdim=True)
            m_cuda.channel_sum(nCount, nOuterNum, nChannel, nInnerNum, colBottom[0].gpu_data, m_blobMu.mutable_gpu_data, false);
            m_blobMu.Reshape(m_blobMu.num, m_blobMu.channels, 1, 1);
            m_blobMu.scale_data(1.0 / nInnerNum);

            //-----------------------------------
            // var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            // Copy each mean value per channel across all items in the channel (e.g. 1 -> channel items)
            m_cuda.channel_fillfrom(nCount, nOuterNum, nChannel, nInnerNum, m_blobMu.gpu_data, m_blobXmu.mutable_gpu_data, DIR.FWD);

            // Subtract the mean from the input.
            // xmu = x - mean
            m_cuda.sub(nCount, colBottom[0].gpu_data, m_blobXmu.gpu_data, m_blobXmu.mutable_gpu_data);

            // Square the values
            // xmusq = (xmu) ** 2
            m_cuda.powx(nCount, m_blobXmu.gpu_data, 2.0, m_blobXmuSq.mutable_gpu_data);

            // Calculate the mean across the last dim.
            // var = xmusq.mean(dim=-1, keepdim=True)
            // var shape = (n, c, 1)
            m_blobVar.SetData(0);
            m_cuda.channel_sum(nCount, nOuterNum, nChannel, nInnerNum, m_blobXmuSq.gpu_data, m_blobVar.mutable_gpu_data, false);
            m_blobVar.scale_data(1.0 / nInnerNum);
            m_blobVar.Reshape(m_blobVar.num, m_blobVar.channels, 1, 1);

            //-----------------------------------
            // std = (var + self.epsilon).sqrt()
            // Calculate the stdev across the last dim
            // std = sqrt(var + eps)
            // stdev shape: (n, c, 1)
            m_blobStdev.Reshape(m_blobStdev.num, m_blobStdev.channels, 1, 1);
            m_cuda.add_scalar(nOuterNum * nChannel, m_param.layer_norm_param.epsilon, m_blobVar.mutable_gpu_data);
            m_cuda.sqrt(nOuterNum * nChannel, m_blobVar.gpu_data, m_blobStdev.mutable_gpu_data);

            //-----------------------------------
            // y = (x - mean) / std
            // Normalize the input by centering and dividing by stdev across channels.
            // Copy each stdev value per channel across all items in the channel (e.g. 1 -> channel items)
            m_cuda.channel_fillfrom(nCount, nOuterNum, nChannel, nInnerNum, m_blobStdev.gpu_data, m_blobStdevFull.mutable_gpu_data, DIR.FWD);
            m_cuda.div(nCount, m_blobXmu.gpu_data, m_blobStdevFull.gpu_data, colTop[0].mutable_gpu_data);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
            {
                if (m_param.layer_norm_param.enable_cuda_impl)
                    m_cuda.LayerNormBackward(m_hLayerNorm, colTop[0].gpu_diff, colBottom[0].mutable_gpu_diff);
                else
                    backward_local(colTop, rgbPropagateDown, colBottom);
            }
        }

        private void backward_local(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            int nAxes = colBottom[0].num_axes;
            int nCount = colBottom[0].count();
            int nOuterNum = colBottom[0].num;
            int nChannel = colBottom[0].channels;
            int nInnerNum = colBottom[0].count(2);


            //-----------------------------------
            // y = (x - mean) / std
            // Normalize the input by centering and dividing by stdev across channels.
            // Copy each stdev value per channel across all items in the channel (e.g. 1 -> channel items)
            // xmu' = y' / std 
            m_cuda.div(nCount, colTop[0].gpu_diff, m_blobStdevFull.gpu_data, m_blobXmu.mutable_gpu_diff);

            // std' = y' * -xmu / std^2
            m_cuda.powx(nCount, m_blobStdevFull.gpu_data, 2.0, m_blobStdevFull.mutable_gpu_diff);
            m_cuda.div(nCount, m_blobXmu.gpu_data, m_blobStdevFull.gpu_diff, m_blobWork.mutable_gpu_diff);
            m_cuda.scal(nCount, -1, m_blobWork.mutable_gpu_diff);
            m_cuda.mul(nCount, colTop[0].gpu_diff, m_blobWork.gpu_diff, m_blobStdevFull.mutable_gpu_diff);

            // std' = channel_sum(stdfull')
            m_cuda.channel_sum(nCount, nOuterNum, nChannel, nInnerNum, m_blobStdevFull.gpu_diff, m_blobStdev.mutable_gpu_diff, false);

            //-----------------------------------
            // std = var1.sqrt()
            // var' = std' * 0.5 * std^-1
            m_blobWork.SetDiff(0);
            m_cuda.powx(nOuterNum * nChannel, m_blobStdev.gpu_data, -1.0, m_blobWork.mutable_gpu_diff);
            m_cuda.scal(nOuterNum * nChannel, 0.5, m_blobWork.mutable_gpu_diff);
            m_cuda.mul(nOuterNum * nChannel, m_blobStdev.gpu_diff, m_blobWork.gpu_diff, m_blobVar.mutable_gpu_diff);

            //-----------------------------------
            // var = xmusq.mean(dim=-1, keepdim=True)
            // xmusq' = 1 / n * var'
            m_cuda.channel_fillfrom(nCount, nOuterNum, nChannel, nInnerNum, m_blobVar.gpu_diff, m_blobXmuSq.mutable_gpu_diff, DIR.FWD);
            m_blobXmuSq.scale_diff(1.0 / nInnerNum);

            //-----------------------------------
            // xmusq = (xmu) ** 2
            // xmu' = 2 * xmu * xmusq' + previous xmu' (xmu' = y' / std)
            m_cuda.mul(nCount, m_blobXmu.gpu_data, m_blobXmuSq.gpu_diff, m_blobWork.mutable_gpu_diff);
            m_cuda.scale(nCount, 2.0, m_blobWork.gpu_diff, m_blobWork.mutable_gpu_diff);
            m_cuda.add(nCount, m_blobXmu.gpu_diff, m_blobWork.gpu_diff, m_blobXmu.mutable_gpu_diff);

            //-----------------------------------
            // xmu = x - mean
            // x' = xmu' 
            // mean' = -channel_sum(xmu')
            m_cuda.channel_sum(nCount, nOuterNum, nChannel, nInnerNum, m_blobXmu.gpu_diff, m_blobMu.mutable_gpu_diff, false);
            m_cuda.scal(nCount, -1, m_blobMu.mutable_gpu_diff);

            //-----------------------------------
            // mean = x.mean(dim=-1, keepdim=True)
            // x' = 1 / n * mean' + x'
            m_cuda.channel_fillfrom(nCount, nOuterNum, nChannel, nInnerNum, m_blobMu.gpu_diff, m_blobWork.mutable_gpu_diff, DIR.FWD);
            m_blobWork.scale_diff(1.0 / nInnerNum);
            m_cuda.add(nCount, m_blobXmu.gpu_diff, m_blobWork.gpu_diff, colBottom[0].mutable_gpu_diff);
        }
    }
}
