using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.param;

namespace MyCaffe.layers.ssd
{
    /// <summary>
    /// The Normalization2Layer performs normalization used by the SSD algorithm.
    /// This layer is initialized with the MyCaffe.param.Normalization2Parameter.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class Normalization2Layer<T> : Layer<T>
    {
        Blob<T> m_blobNorm;
        Blob<T> m_blobSumChannelMultiplier;
        Blob<T> m_blobSumSpatialMultiplier;
        Blob<T> m_blobBuffer;
        Blob<T> m_blobBufferChannel;
        Blob<T> m_blobBufferSpatial;
        bool m_bAcrossSpatial;
        bool m_bChannelShared;
        float m_fEps;

        /// <summary>
        /// The Normalization2Layer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type NORMALIZATION2 with parameter normalization1_param,
        /// with options:
        ///   - across_spatial (\b optional, default true). Normalize across spatial dimensions.
        ///   - channel_shared (\b optional, default true). Whether or not to scale parameters are shared across channels.
        ///   - eps (\b optional, default = 1e-10f). The epsilon to avoid dividing by zero while normalizing variance.
        ///   - scale_filler (\b optional, default = 'constant',1.0). The filler for the initial value of scale.
        /// </param>
        public Normalization2Layer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.NORMALIZATION2;
            m_blobNorm = new Blob<T>(cuda, log, false);
            m_blobNorm.Name = "norm";
            m_blobSumChannelMultiplier = new Blob<T>(cuda, log, false);
            m_blobSumChannelMultiplier.Name = "sum_chan_mult";
            m_blobSumSpatialMultiplier = new Blob<T>(cuda, log, false);
            m_blobSumSpatialMultiplier.Name = "sum_spat_mult";
            m_blobBuffer = new Blob<T>(cuda, log, false);
            m_blobBuffer.Name = "buffer";
            m_blobBufferChannel = new Blob<T>(cuda, log, false);
            m_blobBufferChannel.Name = "buffer_chan";
            m_blobBufferSpatial = new Blob<T>(cuda, log, false);
            m_blobBufferSpatial.Name = "buffer_spat";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobNorm.Dispose();
            m_blobSumChannelMultiplier.Dispose();
            m_blobSumSpatialMultiplier.Dispose();
            m_blobBuffer.Dispose();
            m_blobBufferChannel.Dispose();
            m_blobBufferSpatial.Dispose();
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobNorm);
                col.Add(m_blobSumChannelMultiplier);
                col.Add(m_blobSumSpatialMultiplier);
                col.Add(m_blobBuffer);
                col.Add(m_blobBufferChannel);
                col.Add(m_blobBufferSpatial);

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
            m_log.CHECK_GE(colBottom[0].num_axes, 2, "The bottom(0) must have >= 2 axes.");
            m_blobBuffer.Reshape(1, colBottom[0].channels, colBottom[0].height, colBottom[0].width);
            m_blobBufferChannel.Reshape(1, colBottom[0].channels, 1, 1);
            m_blobBufferSpatial.Reshape(1, 1, colBottom[0].height, colBottom[0].width);

            Normalization2Parameter norm_param = layer_param.normalization2_param;
            m_bAcrossSpatial = norm_param.across_spatial;

            if (m_bAcrossSpatial)
                m_blobNorm.Reshape(colBottom[0].num, 1, 1, 1);
            else
                m_blobNorm.Reshape(colBottom[0].num, 1, colBottom[0].height, colBottom[0].width);

            m_fEps = norm_param.eps;

            int nChannels = colBottom[0].channels;
            int nSpatialDim = colBottom[0].width * colBottom[0].height;

            m_blobSumChannelMultiplier.Reshape(1, nChannels, 1, 1);
            m_blobSumChannelMultiplier.SetData(1);

            m_blobSumSpatialMultiplier.Reshape(1, 1, colBottom[0].height, colBottom[0].width);
            m_blobSumSpatialMultiplier.SetData(1);

            m_bChannelShared = norm_param.channel_shared;

            if (m_colBlobs.Count > 0)
            {
                m_log.WriteLine("Skipping parameter initialization.");
            }
            else
            {
                Blob<T> blobScale = new Blob<T>(m_cuda, m_log);
                List<int> rgShape = new List<int>();

                if (!m_bChannelShared)
                    rgShape.Add(nChannels);

                blobScale.Reshape(rgShape);
                Filler<T> filler = Filler<T>.Create(m_cuda, m_log, norm_param.scale_filler);
                filler.Fill(blobScale);

                m_colBlobs.Add(blobScale);
            }

            if (m_bChannelShared)
                m_log.CHECK_EQ(m_colBlobs[0].count(), 1, "Scale size is inconsistent with prototxt config.");
            else
                m_log.CHECK_EQ(m_colBlobs[0].count(), nChannels, "Scale size is inconsistent with prototxt config.");

            m_rgbParamPropagateDown[0] = true;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_GE(colBottom[0].num_axes, 2, "Number of axes of bottom blob must be >= 2.");
            colTop[0].ReshapeLike(colBottom[0]);
            m_blobBuffer.Reshape(1, colBottom[0].channels, colBottom[0].height, colBottom[0].width);

            if (!m_bAcrossSpatial)
                m_blobNorm.Reshape(colBottom[0].num, 1, colBottom[0].height, colBottom[0].width);

            int nSpatialDim = colBottom[0].height * colBottom[0].width;

            if (nSpatialDim != m_blobSumSpatialMultiplier.count())
            {
                m_blobSumSpatialMultiplier.Reshape(1, 1, colBottom[0].height, colBottom[0].width);
                m_blobSumSpatialMultiplier.SetData(1);
                m_blobBufferSpatial.ReshapeLike(m_blobSumSpatialMultiplier);
            }
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
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            long hBufferData = m_blobBuffer.mutable_gpu_data;
            long hScale = 0;
            long hSumChannelMultiplier = m_blobSumChannelMultiplier.gpu_data;
            long hNormData = 0;
            T[] rgNormData = null;
            T[] rgScale = null;

            if (m_bAcrossSpatial)
            {
                // Need to index it.
                rgNormData = m_blobNorm.mutable_cpu_data;
            }
            else
            {
                hNormData = m_blobNorm.mutable_gpu_data;
                // Add eps to avoid overflow.
                m_blobNorm.SetData(m_fEps);
            }

            if (m_bChannelShared)
                rgScale = m_colBlobs[0].mutable_cpu_data;
            else
                hScale = m_colBlobs[0].gpu_data;

            int nNum = colBottom[0].num;
            int nDim = colBottom[0].count() / nNum;
            int nSpatialDim = colBottom[0].height * colBottom[0].width;
            int nChannels = colBottom[0].channels;
            int nNormDataOffset = 0;
            int nBottomOffset = 0;
            int nTopOffset = 0;

            for (int n = 0; n < nNum; n++)
            {
                m_cuda.powx(nDim, hBottomData, 2.0, hBufferData, nBottomOffset, 0);

                if (m_bAcrossSpatial)
                {
                    double dfNormSqr = Utility.ConvertVal<T>(m_blobBuffer.asum_data());
                    dfNormSqr += m_fEps; // Add eps to avoid overflow.
                    double dfNorm = Math.Pow(dfNormSqr, 0.5);
                    m_cuda.scale(nDim, Utility.ConvertVal<T>(1.0 / dfNorm), hBottomData, hTopData, nBottomOffset, nTopOffset);
                    rgNormData[n] = Utility.ConvertVal<T>(dfNorm);
                }
                else
                {
                    // compute norm
                    m_cuda.gemv(true, nChannels, nSpatialDim, m_tOne, hBufferData, hSumChannelMultiplier, m_tOne, hNormData, 0, 0, nNormDataOffset);
                    m_cuda.powx(nSpatialDim, hNormData, 0.5, hNormData, nNormDataOffset, nNormDataOffset);

                    // Scale the layer.
                    m_cuda.divbsx(nDim, hBottomData, nBottomOffset, hNormData, nNormDataOffset, nChannels, nSpatialDim, false, hTopData, nTopOffset);
                    nNormDataOffset += nSpatialDim;
                }

                // Scale the output
                if (m_bChannelShared)
                {
                    m_cuda.scal(nDim, rgScale[0], hTopData, nTopOffset);
                }
                else
                {
                    m_cuda.mulbsx(nDim, hTopData, nTopOffset, hScale, 0, nChannels, nSpatialDim, true, hTopData, nTopOffset);
                }

                nBottomOffset += nDim;
                nTopOffset += nDim;
            }

            if (rgNormData != null)
                m_blobNorm.mutable_cpu_data = rgNormData;
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
            long hTopDiff = colTop[0].gpu_diff;
            long hTopData = colTop[0].gpu_data;
            long hBottomData = colBottom[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            long hBufferData = m_blobBuffer.mutable_gpu_data;
            long hScale = 0;
            long hBufferChannel = m_blobBufferChannel.mutable_gpu_data;
            long hBufferSpatial = m_blobBufferSpatial.mutable_gpu_data;
            long hSumChannelMultiplier = m_blobSumChannelMultiplier.gpu_data;
            long hSumSpatialMultiplier = m_blobSumSpatialMultiplier.gpu_data;
            long hNormData = 0;
            double dfScale = 0;
            double[] rgNormData = null;

            if (m_bAcrossSpatial)
            {
                // Need to index it.
                rgNormData = Utility.ConvertVec<T>(m_blobNorm.mutable_cpu_data);
            }
            else
            {
                hNormData = m_blobNorm.mutable_gpu_data;
            }

            if (m_bChannelShared)
                dfScale = Utility.ConvertVal<T>(m_colBlobs[0].GetData(0));
            else
                hScale = m_colBlobs[0].mutable_gpu_data;

            int nCount = colTop[0].count();
            int nNum = colTop[0].num;
            int nDim = nCount / nNum;
            int nSpatialDim = colTop[0].height * colTop[0].width;
            int nChannels = colTop[0].channels;

            // Propagate to param
            if (m_rgbParamPropagateDown[0])
            {
                if (m_bChannelShared)
                {
                    double dfScaleDiff = Utility.ConvertVal<T>(m_colBlobs[0].GetDiff(0));
                    double dfA = Utility.ConvertVal<T>(m_cuda.dot(nCount, hTopData, hTopDiff));

                    dfScaleDiff += (dfA / dfScale);
                    m_colBlobs[0].SetDiff(dfScaleDiff, 0);
                }
                else
                {
                    long hScaleDiff = m_colBlobs[0].mutable_gpu_diff;

                    for (int n = 0; n < nNum; n++)
                    {
                        // Compute A
                        m_cuda.mul(nDim, hTopData, hTopDiff, hBufferData, n * nDim, n * nDim, 0);
                        m_cuda.gemv(false, nChannels, nSpatialDim, m_tOne, hBufferData, hSumSpatialMultiplier, m_tZero, hBufferChannel);

                        // Store A / scale[i] in bufferdata temporarily.
                        m_cuda.div(nChannels, hBufferChannel, hScale, hBufferChannel);
                        m_cuda.add(nChannels, hBufferChannel, hScaleDiff, hScaleDiff);
                    }
                }
            }

            // Propagate to bottom
            if (rgbPropagateDown[0])
            {
                int nBottomDataOffset = 0;
                int nBottomDiffOffset = 0;
                int nTopDiffOffset = 0;
                int nNormDataOffset = 0;

                for (int n = 0; n < nNum; n++)
                {
                    if (m_bAcrossSpatial)
                    {
                        double dfA = Utility.ConvertVal<T>(m_cuda.dot(nDim, hBottomData, hTopDiff, nBottomDataOffset, nTopDiffOffset));
                        double dfScale1 = dfA / rgNormData[n] / rgNormData[n];
                        m_cuda.scale(nDim, Utility.ConvertVal<T>(dfScale1), hBottomData, hBottomDiff, nBottomDataOffset, nBottomDiffOffset);
                        m_cuda.sub(nDim, hTopDiff, hBottomDiff, hBottomDiff, nTopDiffOffset, nBottomDiffOffset, nBottomDiffOffset);

                        dfScale1 = 1.0 / rgNormData[n];
                        m_cuda.scale(nDim, Utility.ConvertVal<T>(dfScale1), hBottomDiff, hBottomDiff, nBottomDiffOffset, nBottomDiffOffset);
                    }
                    else
                    {
                        // dot product between bottom_data and top_diff
                        m_cuda.mul(nDim, hBottomData, hTopDiff, hBufferData, nBottomDataOffset, nTopDiffOffset, 0);
                        m_cuda.gemv(true, nChannels, nSpatialDim, m_tOne, hBufferData, hSumChannelMultiplier, m_tZero, hBufferSpatial);

                        // scale bottom diff
                        m_cuda.mulbsx(nDim, hBottomData, nBottomDataOffset, hBufferSpatial, 0, nChannels, nSpatialDim, false, hBottomDiff, nBottomDiffOffset);

                        // divide by square of norm
                        m_cuda.powx(nSpatialDim, hNormData, 2.0, hBufferSpatial);
                        m_cuda.divbsx(nDim, hBottomDiff, nBottomDiffOffset, hBufferSpatial, 0, nChannels, nSpatialDim, false, hBottomDiff, nBottomDiffOffset);
                        m_cuda.sub(nDim, hTopDiff, hBottomDiff, hBottomDiff, nTopDiffOffset, nBottomDiffOffset, nBottomDiffOffset);

                        // divide by norm
                        m_cuda.divbsx(nDim, hBottomDiff, nBottomDiffOffset, hNormData, nNormDataOffset, nChannels, nSpatialDim, false, hBottomDiff, nBottomDiffOffset);
                        nNormDataOffset += nSpatialDim;
                    }

                    // Scale the diff
                    if (m_bChannelShared)
                        m_cuda.scal(nDim, dfScale, hBottomDiff, nBottomDiffOffset);
                    else
                        m_cuda.mulbsx(nDim, hBottomDiff, nBottomDiffOffset, hScale, 0, nChannels, nSpatialDim, true, hBottomDiff, nBottomDiffOffset);

                    nBottomDataOffset += nDim;
                    nBottomDiffOffset += nDim;
                    nTopDiffOffset += nDim;
                }
            }
        }
    }
}
