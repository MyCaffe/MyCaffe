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
        Blob<T> m_blobWork;
        Blob<T> m_blobX1;
        Blob<T> m_blobX2;
        Blob<T> m_blobX3;
        Blob<T> m_blobX4;
        Blob<T> m_blobX5;
 
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

            m_blobX1 = new Blob<T>(cuda, log);
            m_blobX1.Name = m_param.name + ".x1";
            m_blobX2 = new Blob<T>(cuda, log);
            m_blobX2.Name = m_param.name + ".x2";
            m_blobX3 = new Blob<T>(cuda, log);
            m_blobX3.Name = m_param.name + ".x3";
            m_blobX4 = new Blob<T>(cuda, log);
            m_blobX4.Name = m_param.name + ".x4";
            m_blobX5 = new Blob<T>(cuda, log);
            m_blobX5.Name = m_param.name + ".x5";
            setup_internal_blobs(m_colInternalBlobs);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobX1);
            dispose(ref m_blobX2);
            dispose(ref m_blobX3);
            dispose(ref m_blobX4);
            dispose(ref m_blobX5);
            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobX1);
            col.Add(m_blobX2);
            col.Add(m_blobX3);
            col.Add(m_blobX4);
            col.Add(m_blobX5);
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

            List<int> rgWeightShape = Utility.Create<int>(1, 0);
            rgWeightShape[0] = m_nInnerNum;

            Blob<T> blobWt = new Blob<T>(m_cuda, m_log, !layer_param.freeze_learning);
            blobWt.Name = m_param.name + ".weights";
            blobWt.blob_type = BLOB_TYPE.WEIGHT;

            if (!shareParameter(blobWt, rgWeightShape, true))
            {
                blobWt.Reshape(rgWeightShape);
                FillerParameter fp = new FillerParameter("constant", 1.0);
                Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                filler.Fill(blobWt);
            }
            m_colBlobs.Add(blobWt);
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

            m_log.CHECK_EQ(m_nInnerNum, m_colBlobs[0].count(), "The weight count must equal the inner num.");

            m_blobX1.Reshape(m_nOuterNum, m_nChannels, m_nInnerNum, 1);
            m_blobX2.Reshape(m_nOuterNum, m_nChannels, 1, 1);
            m_blobX3.Reshape(m_nOuterNum, m_nChannels, 1, 1);
            m_blobX4.Reshape(m_nOuterNum, m_nChannels, 1, 1);
            m_blobX5.Reshape(m_nOuterNum, m_nChannels, m_nInnerNum, 1);

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
            m_cuda.powx(m_nCount, colBottom[0].gpu_data, 2.0, m_blobX1.mutable_gpu_data); // x1 = x^2
            m_cuda.channel_mean(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_blobX1.gpu_data, m_blobX2.mutable_gpu_data); // x2 = mean(x^2)
            m_cuda.sqrt(m_blobX3.count(), m_blobX2.gpu_data, m_blobX3.mutable_gpu_data, m_param.rms_norm_param.epsilon); // x3 = sqrt(mean(x^2))
            m_cuda.invert(m_blobX4.count(), m_blobX3.gpu_data, m_blobX4.mutable_gpu_data); // x4 = 1 / sqrt(mean(x^2))
            m_cuda.channel_duplicate(m_nCount, m_nOuterNum * m_nChannels, 1, m_nInnerNum, m_blobX4.gpu_data, m_blobX1.mutable_gpu_data); 
            m_cuda.mul(m_nCount, colBottom[0].gpu_data, m_blobX1.gpu_data, m_blobX5.mutable_gpu_data); // x5 = x * 1 / sqrt(mean(x^2))
            m_cuda.channel_copyall(m_nCount, m_nOuterNum * m_nChannels, 1, m_nInnerNum, m_colBlobs[0].gpu_data, m_blobX1.mutable_gpu_data); // x1 = w.repeat(1, inner_num)
            m_cuda.mul(m_nCount, m_blobX5.gpu_data, m_blobX1.gpu_data, colTop[0].mutable_gpu_data); // y = x * 1 / sqrt(mean(x^2)) * w
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
                if (!m_param.freeze_learning)
                {
                    m_cuda.mul(m_nCount, colTop[0].gpu_diff, m_blobX5.gpu_data, m_blobX5.mutable_gpu_diff); // x5_grad = top_diff * w
                    m_cuda.channel_sum(m_nCount, 1, m_nOuterNum * m_nChannels, m_nInnerNum, m_blobX5.gpu_diff, m_colBlobs[0].mutable_gpu_diff, true); // w_grad = sum(top_diff * x, 0)
                }

                m_cuda.channel_copyall(m_nCount, m_nOuterNum * m_nChannels, 1, m_nInnerNum, m_colBlobs[0].gpu_data, m_blobX1.mutable_gpu_diff); // x1 = w.repeat(1, inner_num)
                m_cuda.mul(m_nCount, colTop[0].gpu_diff, m_blobX1.gpu_diff, m_blobX5.mutable_gpu_diff);     // x5_grad = top_diff * w

                m_cuda.channel_duplicate(m_nCount, m_nOuterNum * m_nChannels, 1, m_nInnerNum, m_blobX4.gpu_data, m_blobX1.mutable_gpu_diff);                    // x1_grad = x4_grad.repeat(1, inner_num)   
                m_cuda.mul(m_nCount, m_blobX5.gpu_diff, m_blobX1.gpu_diff, colBottom[0].mutable_gpu_diff);                                                      // x_grad = x5_grad * x1 (accumulated)

                m_cuda.mul(m_nCount, m_blobX5.gpu_diff, colBottom[0].gpu_data, m_blobX5.mutable_gpu_diff);
                m_cuda.channel_sum(m_nCount, m_nOuterNum, m_nChannels, m_nInnerNum, m_blobX5.gpu_diff, m_blobX4.mutable_gpu_diff, false);                       // x4b_grad = sum(x5_grad * x, -1)

                m_cuda.powx(m_blobX3.count(), m_blobX3.gpu_data, 2.0, m_blobX3.mutable_gpu_diff);                                                               // re-use x3 space for the following calculations                                                                                                                                                                
                m_cuda.invert(m_blobX3.count(), m_blobX3.gpu_diff, m_blobX3.mutable_gpu_diff, 0, 0, -1.0);                                                      // x4_grad = x4b_grad * -1 / x4.pow(2)
                m_cuda.mul(m_blobX3.count(), m_blobX3.gpu_diff, m_blobX4.gpu_diff, m_blobX3.mutable_gpu_diff);

                m_cuda.sqrt(m_blobX2.count(), m_blobX2.gpu_data, m_blobX2.mutable_gpu_diff, m_param.rms_norm_param.epsilon);                                    // x3_grad = x4_grad / (2 * x3.sqrt())
                m_cuda.invert(m_blobX2.count(), m_blobX2.gpu_diff, m_blobX2.mutable_gpu_diff, 0, 0, 1, 2);                                                      //   |
                m_cuda.mul(m_blobX2.count(), m_blobX2.gpu_diff, m_blobX3.gpu_diff, m_blobX2.mutable_gpu_diff);                                                  // x2_grad

                m_cuda.scale(m_nCount, 1.0/m_nInnerNum, m_blobX2.gpu_diff, m_blobX2.mutable_gpu_diff);                                                          // x1_grad = x2_grad / inner_num
                m_cuda.channel_duplicate(m_nCount, m_nOuterNum * m_nChannels, 1, m_nInnerNum, m_blobX2.gpu_diff, m_blobX1.mutable_gpu_diff);                    // x1_grad = x1_grad.repeat(1, inner_num)

                m_cuda.scale(m_nCount, 2.0, colBottom[0].gpu_data, m_blobX5.gpu_diff);                                                                          // re-use x5_grad for x*2 scale
                m_cuda.mul(m_nCount, m_blobX5.gpu_diff, m_blobX1.gpu_diff, m_blobX1.mutable_gpu_diff);
                m_cuda.add(m_nCount, colBottom[0].gpu_diff, m_blobX1.gpu_diff, colBottom[0].mutable_gpu_diff);                                                  // x_grad = x_grad + x1_grad (accumulated)
            }
        }
    }
}
