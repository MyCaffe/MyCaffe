using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.param;

namespace MyCaffe.layers.gpt
{
    /// <summary>
    /// The RMSNormLayer performs layer normalization implements the Root Mean Square Normalization which is similar to LayerNorm but faster.
    /// </summary>
    /// <remarks>
    /// @see [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) by Zhang et al., 2019, arXiv:1910.07467
    /// @see [RMSNorm](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) PyTorch
    /// @see [GitHub:karpathy/llama2.c](https://github.com/karpathy/llama2.c) by Karpathy (MIT Liceense).
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class RMSNormLayer<T> : Layer<T>
    {
        int m_nAxis = 0;
        int m_nCount = 0;
        int m_nOuterNum = 0;
        int m_nChannels = 0;
        int m_nInnerNum = 0;
        Blob<T> m_blobRms1;
        Blob<T> m_blobRms;
        Blob<T> m_blobRms1Full;
        Blob<T> m_blobOut;
        List<int> m_rgWeightShape = new List<int>() { 1 };
 
        /// <summary>
        /// The RMSNormLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type RMSNormLayer with parameter layer_norm_param,
        /// with options:
        ///   - epsilon (\b optional, default 1e-10). The epsilon value used to avoid Nan values.
        /// </param>
        public RMSNormLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.RMSNORM;

            m_blobRms1 = createIntraLayerBlob("rms1", true, true);
            m_blobRms1Full = createIntraLayerBlob("rms1full", true, true);
            m_blobRms = createIntraLayerBlob("rms", true, true);
            m_blobOut = createIntraLayerBlob("rmsOut", true, true);

            setup_internal_blobs(m_colInternalBlobs);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobRms1);
            dispose(ref m_blobRms);
            dispose(ref m_blobRms1Full);
            dispose(ref m_blobOut);
            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobRms1);
            col.Add(m_blobRms);
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
            m_nAxis = colBottom[0].CanonicalAxisIndex(m_param.rms_norm_param.axis);

            m_nCount = colBottom[0].count();
            m_nInnerNum = colBottom[0].count(m_nAxis);
            m_nChannels = (m_nAxis == 0) ? 1 : colBottom[0].shape(m_nAxis-1);
            m_nOuterNum = (m_nAxis == 0) ? 1 : colBottom[0].count(0, m_nAxis-1);

            if (m_param.rms_norm_param.enable_weights)
            {
                m_rgWeightShape[0] = m_nInnerNum;

                Blob<T> blobWt = new Blob<T>(m_cuda, m_log, !layer_param.freeze_learning);
                blobWt.Name = m_param.name + ".weights";
                blobWt.blob_type = BLOB_TYPE.WEIGHT;

                if (!shareParameter(blobWt, m_rgWeightShape, true))
                {
                    blobWt.Reshape(m_rgWeightShape);
                    FillerParameter fp = new FillerParameter("constant", 1.0);
                    Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                    filler.Fill(blobWt);
                }
                m_colBlobs.Add(blobWt);
            }
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nCount = colBottom[0].count();
            m_nInnerNum = colBottom[0].count(m_nAxis);
            m_nChannels = colBottom[0].shape(m_nAxis-1);
            m_nOuterNum = colBottom[0].count(0, m_nAxis-1);

            if (m_param.rms_norm_param.enable_weights)
                m_log.CHECK_EQ(m_nInnerNum, m_colBlobs[0].count(), "The weight count must equal the inner num.");

            m_blobRms1.Reshape(m_nOuterNum, m_nChannels, 1, 1);
            m_blobRms.Reshape(m_nOuterNum, m_nChannels, 1, 1);
            m_blobOut.ReshapeLike(colBottom[0]);
            m_blobRms1Full.ReshapeLike(colBottom[0]);

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
            // x1 = x.pow(2)
            m_cuda.powx(m_nCount, colBottom[0].gpu_data, 2.0, colTop[0].mutable_gpu_data); 
            // x2 = x1.mean(-1, keepdim=True)
            m_cuda.channel_mean(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, colTop[0].gpu_data, m_blobRms.mutable_gpu_data); 
            // x4 = (x2 + eps).sqrt()
            m_cuda.sqrt(m_blobRms.count(), m_blobRms.gpu_data, m_blobRms.mutable_gpu_data, m_param.rms_norm_param.epsilon);
            // x5 = 1.0/x4
            m_cuda.invert(m_blobRms1.count(), m_blobRms.gpu_data, m_blobRms1.mutable_gpu_data);
            // x6 = x * x5
            m_cuda.channel_duplicate(m_nCount, m_nOuterNum * m_nChannels, 1, m_nInnerNum, m_blobRms1.gpu_data, m_blobRms1Full.mutable_gpu_data); 
            m_cuda.mul(m_nCount, colBottom[0].gpu_data, m_blobRms1Full.gpu_data, m_blobOut.mutable_gpu_data);

            if (m_param.rms_norm_param.enable_weights)
            {
                m_cuda.channel_copyall(m_nCount, m_nOuterNum * m_nChannels, 1, m_nInnerNum, m_colBlobs[0].gpu_data, m_blobOut.mutable_gpu_diff);
                m_cuda.mul(m_nCount, m_blobOut.gpu_diff, m_blobOut.gpu_data, colTop[0].mutable_gpu_data);
            }
            else
            {
                colTop[0].CopyFrom(m_blobOut);
            }
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
                if (m_param.rms_norm_param.enable_weights)
                {
                    // grad wrt wts
                    m_cuda.mul(m_nCount, colTop[0].gpu_diff, m_blobOut.gpu_data, m_blobOut.mutable_gpu_data);
                    m_cuda.channel_sum(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_blobOut.gpu_data, m_colBlobs[0].mutable_gpu_diff);

                    // grad wrt x
                    m_cuda.mul(m_nCount, colTop[0].gpu_diff, m_blobOut.gpu_diff, m_blobOut.mutable_gpu_diff);
                }

                // x6 = x * x5 -> dx 
                m_cuda.mul(m_nCount, m_blobOut.gpu_diff, m_blobRms1Full.gpu_data, colBottom[0].mutable_gpu_diff);
                // x6 = x * x5 -> dx5
                m_cuda.mul(m_nCount, m_blobOut.gpu_diff, colBottom[0].gpu_data, m_blobRms1Full.mutable_gpu_diff);
                m_cuda.channel_sum(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_blobRms1Full.gpu_diff, m_blobRms1.mutable_gpu_diff, false);

                // x5 = 1.0/x4 -> dx4
                m_cuda.powx(m_blobRms.count(), m_blobRms.gpu_data, 2.0, m_blobRms.mutable_gpu_diff);
                m_cuda.invert(m_blobRms.count(), m_blobRms.gpu_diff, m_blobRms.mutable_gpu_diff, 0, 0, -1);
                m_cuda.mul(m_blobRms1.count(), m_blobRms1.gpu_diff, m_blobRms.gpu_diff, m_blobRms.mutable_gpu_diff);

                // x4 = (x2 + eps).sqrt() -> dx2
                m_cuda.invert(m_blobRms1.count(), m_blobRms.gpu_data, m_blobRms1.mutable_gpu_diff, 0, 0, 1, 2);
                m_cuda.mul(m_blobRms.count(), m_blobRms1.gpu_diff, m_blobRms.gpu_diff, m_blobRms.mutable_gpu_diff);

                // x2 = x1.mean(-1, keepdim=True) -> dx1
                m_cuda.channel_duplicate(m_nCount, m_nOuterNum * m_nChannels, 1, m_nInnerNum, m_blobRms.gpu_diff, m_blobRms1Full.mutable_gpu_diff);
                m_cuda.scal(m_blobRms1Full.count(), 1.0/m_nInnerNum, m_blobRms1Full.mutable_gpu_diff);

                // x1 = x.pow(2) -> dx
                m_cuda.scale(m_nCount, 2.0, colBottom[0].gpu_data, m_blobOut.mutable_gpu_diff);;
                m_cuda.mul(m_nCount, m_blobOut.gpu_diff, m_blobRms1Full.gpu_diff, m_blobOut.mutable_gpu_diff);

                // x = dx1 + dx2
                m_cuda.add(m_nCount, m_blobOut.gpu_diff, colBottom[0].gpu_diff, colBottom[0].mutable_gpu_diff);
            }
        }
    }
}
