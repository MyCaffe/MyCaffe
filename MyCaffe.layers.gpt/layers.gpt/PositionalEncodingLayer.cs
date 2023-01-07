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
    /// The PositionalEncodingLayer is a neuron layer that adds positional encoding to the input.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class PositionalEncodingLayer<T> : NeuronLayer<T>
    {
        Blob<T> m_blobPosEnc;
        List<int> m_rgShape = new List<int>() { 1, 1, 1 };
        double m_dfScale;
        int m_nBlockSize;
        int m_nEmbed;

        /// <summary>
        /// The PositionalEncoderLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Mish with parameter Mish_param
        /// </param>
        public PositionalEncodingLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.POSITIONAL_ENCODING;
            m_nBlockSize = (int)p.positional_encoder_param.block_size;
            m_nEmbed = (int)p.positional_encoder_param.embed;
            m_dfScale = Math.Sqrt(m_nEmbed);

            m_blobPosEnc = new Blob<T>(m_cuda, m_log, false);

            setup_internal_blobs(m_colInternalBlobs);
        }

        /// <summary>
        /// Release any resources used.
        /// </summary>
        protected override void dispose()
        {
            dispose(ref m_blobPosEnc);
            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobPosEnc);
        }

        /// <summary>
        /// Reshape the data as needed by the layer.
        /// </summary>
        /// <param name="colBottom"></param>
        /// <param name="colTop"></param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            colTop[0].ReshapeLike(colBottom[0]);

            int nBatch = colBottom[0].num;
            m_rgShape[0] = nBatch;
            m_rgShape[1] = m_nBlockSize;
            m_rgShape[2] = m_nEmbed;

            if (!shareLayerBlob(m_blobPosEnc, m_rgShape))
            {
                if (!m_blobPosEnc.CompareShape(m_rgShape))
                {
                    m_blobPosEnc.Reshape(m_rgShape);

                    T[] rgPosEnc1 = new T[m_nBlockSize * m_nEmbed];
                    for (int pos = 0; pos < m_nBlockSize; pos++)
                    {
                        for (int i = 0; i < m_nEmbed; i++)
                        {
                            int nIdx = pos * m_nEmbed + i;

                            if (i % 2 == 0)
                                rgPosEnc1[nIdx] = Utility.ConvertVal<T>(Math.Sin(pos / Math.Pow(10000.0, (2 * i / (double)m_nEmbed))));
                            else if (i % 2 == 1)
                                rgPosEnc1[nIdx] = Utility.ConvertVal<T>(Math.Cos(pos / Math.Pow(10000.0, (2 * i / (double)m_nEmbed))));
                        }
                    }

                    T[] rgPosEnc = new T[nBatch * m_nBlockSize * m_nEmbed];
                    for (int n = 0; n < nBatch; n++)
                    {
                        Array.Copy(rgPosEnc1, 0, rgPosEnc, n * m_nBlockSize * m_nEmbed, m_nBlockSize * m_nEmbed);
                    }

                    m_blobPosEnc.mutable_cpu_data = rgPosEnc;
                }
            }
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the computed outputs @f$ 
        ///         y  = x * sqrt(m_nEmbed) + pos_enc
        ///     @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();

            m_cuda.add(nCount, m_blobPosEnc.gpu_data, hBottomData, hTopData, m_dfScale);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the Mish value inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     gradients f$ y' = sqrt(m_nEmbed) @f$
        ///     @f$ if propagate_down[0]
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hTopDiff = colTop[0].gpu_diff;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int nCount = colBottom[0].count();

            m_cuda.scale(nCount, m_dfScale, hTopDiff, hBottomDiff);
        }
    }
}
