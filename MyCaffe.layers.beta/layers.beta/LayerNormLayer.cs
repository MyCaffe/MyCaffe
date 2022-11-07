using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

///
/// WORK IN PROGRESS
///
namespace MyCaffe.layers.beta
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
        Blob<T> m_blobMean;
        Blob<T> m_blobMeanDiff;
        Blob<T> m_blobMeanDiffSq;
        Blob<T> m_blobVar;
        Blob<T> m_blobStdev;
        Blob<T> m_blobStdevFull;

        /// <summary>
        /// The LayerNormalizationLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type NORMALIZATION1 with parameter normalization1_param,
        /// with options:
        ///   - norm (\b optional, default L2). The normalization mode to use: L1 or L2.
        /// </param>
        public LayerNormLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.LAYERNORM;

            m_blobMean = new Blob<T>(cuda, log);
            m_blobMeanDiff = new Blob<T>(cuda, log);
            m_blobMeanDiffSq = new Blob<T>(cuda, log);
            m_blobVar = new Blob<T>(cuda, log);
            m_blobStdev = new Blob<T>(cuda, log);
            m_blobStdevFull = new Blob<T>(cuda, log);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobMean);
            dispose(ref m_blobMeanDiff);
            dispose(ref m_blobMeanDiffSq);
            dispose(ref m_blobVar);
            dispose(ref m_blobStdev);
            dispose(ref m_blobStdevFull);

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

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
            m_blobMean.ReshapeLike(colBottom[0]);
            m_blobMeanDiff.ReshapeLike(colBottom[0]);
            m_blobMeanDiffSq.ReshapeLike(colBottom[0]);
            m_blobVar.ReshapeLike(colBottom[0]);
            m_blobStdev.ReshapeLike(colBottom[0]);
            m_blobStdevFull.ReshapeLike(colBottom[0]);
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
            int nAxes = colBottom[0].num_axes;
            int nCount = colBottom[0].count();
            int nOuterNum = colBottom[0].num;
            int nChannel = colBottom[0].channels;
            int nInnerNum = colBottom[0].count(2);
                     
            //-----------------------------------
            // Calculate the mean across the last dim.
            // mean = x.mean(dim=-1, keepdim=True)
            m_cuda.channel_sum(nCount, nOuterNum, nChannel, nInnerNum, colBottom[0].gpu_data, m_blobMean.mutable_gpu_data, false);
            m_blobMean.scale_data(1.0 / nInnerNum);


            //-----------------------------------
            // var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
            // Copy each mean value per channel across all items in the channel (e.g. 1 -> channel items)
            m_cuda.channel_fillfrom(nCount, nOuterNum, nChannel, nInnerNum, m_blobMean.gpu_data, m_blobMeanDiff.mutable_gpu_data, DIR.FWD);

            // Subtract the mean from the input.
            // meandiff = x - mean
            m_cuda.sub(nCount, colBottom[0].gpu_data, m_blobMeanDiff.gpu_data, m_blobMeanDiff.mutable_gpu_data);
            // Square the values
            // meandiffsq = (meandiff) ** 2
            m_cuda.powx(nCount, m_blobMeanDiff.gpu_data, 2.0, m_blobMeanDiffSq.mutable_gpu_data);

            // Calculate the ean across the last dim.
            // var = meandiffsq.mean(dim=-1, keepdim=True)
            // var shape = (n, c, 1)
            m_cuda.channel_sum(nCount, nOuterNum, nChannel, nInnerNum, m_blobMeanDiffSq.gpu_data, m_blobVar.mutable_gpu_data, false);
            m_blobVar.scale_data(1.0 / nInnerNum);

            //-----------------------------------
            // std = (var + self.epsilon).sqrt()
            // Calculate the stdev across the last dim
            // std = sqrt(var + eps)
            // stdev shape: (n, c, 1)
            m_cuda.add_scalar(nOuterNum * nChannel, m_param.layer_norm_param.epsilon, m_blobVar.mutable_gpu_data);
            m_cuda.sqrt(nOuterNum * nChannel, m_blobVar.gpu_data, m_blobStdev.mutable_gpu_data);

            //-----------------------------------
            // y = (x - mean) / std
            // Normalize the input by centering and dividing by stdev across channels.
            // Copy each stdev value per channel across all items in the channel (e.g. 1 -> channel items)
            m_cuda.channel_fillfrom(nCount, nOuterNum, nChannel, nInnerNum, m_blobStdev.gpu_data, m_blobStdevFull.mutable_gpu_data, DIR.FWD);
            m_cuda.div(nCount, m_blobMeanDiff.gpu_data, m_blobStdevFull.gpu_data, colTop[0].mutable_gpu_data);
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
            int nAxes = colBottom[0].num_axes;
            int nCount = colBottom[0].count();
            int nOuterNum = colBottom[0].num;
            int nChannel = colBottom[0].channels;
            int nInnerNum = colBottom[0].count(2);

            // WORK IN PROGRESS
        }
    }
}
