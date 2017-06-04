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
    /// The SPPLayer does spatial pyramid pooling on the input image
    /// by taking the max, average, etc. within regions
    /// so that the result vector of different sized
    /// images are of the same size.
    /// This layer is initialized with the MyCaffe.param.SPPParameter.
    /// </summary>
    /// <remarks>
    /// @see [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, 2014.
    /// @see [Image-based Localization using Hourglass Networks](https://arxiv.org/abs/1703.07971v1) by Iaroslav Melekhov, Juha Ylioinas, Juho Kannala, and Esa Rahtu, 2017.
    /// @see [Relative Camera Pose Estimation Using Convolutional Neural Networks](https://arxiv.org/abs/1702.01381v2) by Iaroslav Melekhov, Juha Ylioinas, Juho Kannala, Esa Rahtu, 2017.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SPPLayer<T> : Layer<T>
    {
        int m_nPyramidHeight;
        int m_nBottomH;
        int m_nBottomW;
        int m_nNum;
        int m_nChannels;
        //int m_nKernelH;
        //int m_nKernelW;
        //int m_nPadH;
        //int m_nPadW;
        bool m_bReshapedFirstTime;
        /// <summary>
        /// The internal Split layer that feeds the pooling layers.
        /// </summary>
        SplitLayer<T> m_split_layer;
        /// <summary>
        /// Top vector holder used in call to the underlying SplitLayer::Forward
        /// </summary>
        BlobCollection<T> m_colBlobSplitTopVec = new BlobCollection<T>();
        /// <summary>
        /// Bottom vector holder used in call to the underlying PoolingLayer::Forward
        /// </summary>
        List<BlobCollection<T>> m_rgPoolingBottomVec = new List<BlobCollection<T>>();
        /// <summary>
        /// The internal Pooling layers of different kernel sizes.
        /// </summary>
        List<PoolingLayer<T>> m_rgPoolingLayers = new List<PoolingLayer<T>>();
        /// <summary>
        /// The vector holders used in call to underlying PoolingLayer::Forward
        /// </summary>
        List<BlobCollection<T>> m_rgPoolingTopVecs = new List<BlobCollection<T>>();
        /// <summary>
        /// Pooling outputs stores the outputs of the PoolingLayers.
        /// </summary>
        BlobCollection<T> m_colBlobPoolingOutputs = new BlobCollection<T>();
        /// <summary>
        /// The internal Flatten layers that the Pooling layers feed into.
        /// </summary>
        List<FlattenLayer<T>> m_rgFlattenLayers = new List<FlattenLayer<T>>();
        /// <summary>
        /// The top vector holders used to call to the underlying FlattenLayer::Forward
        /// </summary>
        List<BlobCollection<T>> m_rgFlattenLayerTopVecs = new List<BlobCollection<T>>();
        /// <summary>
        /// Flatten outputs stores the outputs of the FlattenLayers.
        /// </summary>
        BlobCollection<T> m_colBlobFlattenOutputs = new BlobCollection<T>();
        /// <summary>
        /// Bottom vector holder used in call to the underlying ConcatLayer::Forward
        /// </summary>
        BlobCollection<T> m_colBlobConcatBottomVec = new BlobCollection<T>();
        /// <summary>
        /// The internal Concat layer that the Flatten layers feed into.
        /// </summary>
        ConcatLayer<T> m_concat_layer;


        /// <summary>
        /// The SPPLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type SPP with parameter spp_param,
        /// with options:
        ///   - engine. Specifies whether to use the Engine.CAFFE or Engine.CUDNN for pooling.
        ///   
        ///   - pyramid_height. The pyramid height.
        /// </param>
        public SPPLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SPP;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            foreach (PoolingLayer<T> layer in m_rgPoolingLayers)
            {
                layer.Dispose();
            }

            m_rgPoolingLayers.Clear();

            if (m_colBlobSplitTopVec != null)
            {
                m_colBlobSplitTopVec.Dispose();
                m_colBlobSplitTopVec = null;
            }

            if (m_split_layer != null)
            {
                m_split_layer.Dispose();
                m_split_layer = null;
            }

            if (m_colBlobPoolingOutputs != null)
            {
                m_colBlobPoolingOutputs.Dispose();
                m_colBlobPoolingOutputs = null;
            }

            if (m_colBlobFlattenOutputs != null)
            {
                m_colBlobFlattenOutputs.Dispose();
                m_colBlobFlattenOutputs = null;
            }

            foreach (FlattenLayer<T> layer in m_rgFlattenLayers)
            {
                layer.Dispose();
            }

            m_rgFlattenLayers.Clear();

            m_rgPoolingBottomVec.Clear();
            m_rgPoolingTopVecs.Clear();
            m_rgFlattenLayerTopVecs.Clear();

            base.dispose();
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: spp.
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Calculates the kernel and stride dimensions for the pooling layer,
        /// returns a correctly configured LayerParameter for a PoolingLayer.
        /// </summary>
        /// <param name="nPyramidLevel">Specifies the pyramid level.</param>
        /// <param name="nBottomH">Specifies the bottom height.</param>
        /// <param name="nBottomW">Specifies the bottom width.</param>
        /// <param name="spp_param">Specifies the SPPParameter used.</param>
        /// <returns>The pooling parameter is returned.</returns>
        protected virtual LayerParameter getPoolingParam(int nPyramidLevel, int nBottomH, int nBottomW, SPPParameter spp_param)
        {
            LayerParameter pool_param = new param.LayerParameter(LayerParameter.LayerType.POOLING);
            int nNumBins = (int)Math.Pow(2, nPyramidLevel);

            // find padding and kernel size so that the pooling is
            // performed across the entrie image
            int nKernelH = (int)Math.Ceiling(nBottomH / (double)nNumBins);
            int nKernelW = (int)Math.Ceiling(nBottomW / (double)nNumBins);
            // remainder_H is the min number of pixels that need to be padded before
            // entire image height is pooled over with the chosen kernel simension
            int nRemainderH = nKernelH * nNumBins - nBottomH;
            int nRemainderW = nKernelW * nNumBins - nBottomW;
            // pooling layer pads (2 * pad_h) pixels on the top and bottom of the
            // image.
            int nPadH = (nRemainderH + 1) / 2;
            int nPadW = (nRemainderW + 1) / 2;

            pool_param.pooling_param.pad_h = (uint)nPadH;
            pool_param.pooling_param.pad_w = (uint)nPadW;
            pool_param.pooling_param.kernel_h = (uint)nKernelH;
            pool_param.pooling_param.kernel_w = (uint)nKernelW;
            pool_param.pooling_param.stride_h = (uint)nKernelH;
            pool_param.pooling_param.stride_w = (uint)nKernelW;
            pool_param.pooling_param.pool = spp_param.pool;

            return pool_param;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nNum = colBottom[0].num;
            m_nChannels = colBottom[0].channels;
            m_nBottomH = colBottom[0].height;
            m_nBottomW = colBottom[0].width;
            m_bReshapedFirstTime = false;

            m_log.CHECK_GT(m_nBottomH, 0, "Input dimensions cannot be zero.");
            m_log.CHECK_GT(m_nBottomW, 0, "Input dimensions cannot be zero.");

            m_nPyramidHeight = (int)m_param.spp_param.pyramid_height;

            m_colBlobSplitTopVec = new BlobCollection<T>();
            m_rgPoolingBottomVec = new List<BlobCollection<T>>();
            m_rgPoolingLayers = new List<PoolingLayer<T>>();
            m_rgPoolingTopVecs = new List<BlobCollection<T>>();
            m_colBlobPoolingOutputs = new BlobCollection<T>();
            m_rgFlattenLayers = new List<FlattenLayer<T>>();
            m_rgFlattenLayerTopVecs = new List<BlobCollection<T>>();
            m_colBlobFlattenOutputs = new BlobCollection<T>();
            m_colBlobConcatBottomVec = new BlobCollection<T>();

            if (m_nPyramidHeight == 1)
            {
                // pooling layer setup
                LayerParameter pp = getPoolingParam(0, m_nBottomH, m_nBottomW, m_param.spp_param);
                m_rgPoolingLayers.Add(new PoolingLayer<T>(m_cuda, m_log, pp));
                m_rgPoolingLayers[0].Setup(colBottom, colTop);
                return;
            }

            // split layer output holders setup
            for (int i = 0; i < m_nPyramidHeight; i++)
            {
                m_colBlobSplitTopVec.Add(new Blob<T>(m_cuda, m_log));
            }

            // split layer setup
            LayerParameter split_param = new param.LayerParameter(LayerParameter.LayerType.SPLIT);
            m_split_layer = new SplitLayer<T>(m_cuda, m_log, split_param);
            m_split_layer.Setup(colBottom, m_colBlobSplitTopVec);

            for (int i = 0; i < m_nPyramidHeight; i++)
            {
                // pooling layer input holders setup
                m_rgPoolingBottomVec.Add(new BlobCollection<T>());
                m_rgPoolingBottomVec[i].Add(m_colBlobSplitTopVec[i]);

                // pooling layer output holders setup
                m_colBlobPoolingOutputs.Add(new Blob<T>(m_cuda, m_log));
                m_rgPoolingTopVecs.Add(new BlobCollection<T>());
                m_rgPoolingTopVecs[i].Add(m_colBlobPoolingOutputs[i]);

                // pooling layer setup
                LayerParameter pooling_param = getPoolingParam(i, m_nBottomH, m_nBottomW, m_param.spp_param);
                m_rgPoolingLayers.Add(new PoolingLayer<T>(m_cuda, m_log, pooling_param));
                m_rgPoolingLayers[i].Setup(m_rgPoolingBottomVec[i], m_rgPoolingTopVecs[i]);

                // flatten layer output holders setup
                m_colBlobFlattenOutputs.Add(new Blob<T>(m_cuda, m_log));
                m_rgFlattenLayerTopVecs.Add(new BlobCollection<T>());
                m_rgFlattenLayerTopVecs[i].Add(m_colBlobFlattenOutputs[i]);

                // flatten layer setup
                LayerParameter flatten_param = new LayerParameter(LayerParameter.LayerType.FLATTEN);
                m_rgFlattenLayers.Add(new FlattenLayer<T>(m_cuda, m_log, flatten_param));
                m_rgFlattenLayers[i].Setup(m_rgPoolingTopVecs[i], m_rgFlattenLayerTopVecs[i]);

                // concat layer input holders setup
                m_colBlobConcatBottomVec.Add(m_colBlobFlattenOutputs[i]);
            }

            // concat layer setup
            LayerParameter concat_param = new LayerParameter(LayerParameter.LayerType.CONCAT);
            m_concat_layer = new ConcatLayer<T>(m_cuda, m_log, concat_param);
            m_concat_layer.Setup(m_colBlobConcatBottomVec, colTop);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(4, colBottom[0].num_axes, "Input must have 4 axes, corresponding to (num, channels, height, width)");

            // Do nothing if bottom shape is unchanged since last Reshape.
            if (m_nNum == colBottom[0].num &&
                m_nChannels == colBottom[0].channels &&
                m_nBottomH == colBottom[0].height &&
                m_nBottomW == colBottom[0].width &&
                m_bReshapedFirstTime)
                return;

            m_nNum = colBottom[0].num;
            m_nChannels = colBottom[0].channels;
            m_nBottomH = colBottom[0].height;
            m_nBottomW = colBottom[0].width;
            m_bReshapedFirstTime = true;

            if (m_nPyramidHeight == 1)
            {
                LayerParameter pooling_param = getPoolingParam(0, m_nBottomH, m_nBottomW, m_param.spp_param);

                if (m_rgPoolingLayers[0] != null)
                    m_rgPoolingLayers[0].Dispose();

                m_rgPoolingLayers[0] = new PoolingLayer<T>(m_cuda, m_log, pooling_param);
                m_rgPoolingLayers[0].Setup(colBottom, colTop);
                m_rgPoolingLayers[0].Reshape(colBottom, colTop);
                return;
            }

            m_split_layer.Reshape(colBottom, m_colBlobSplitTopVec);

            for (int i = 0; i < m_nPyramidHeight; i++)
            {
                LayerParameter pooling_param = getPoolingParam(i, m_nBottomH, m_nBottomW, m_param.spp_param);

                if (m_rgPoolingLayers[i] != null)
                    m_rgPoolingLayers[i].Dispose();

                m_rgPoolingLayers[i] = new PoolingLayer<T>(m_cuda, m_log, pooling_param);
                m_rgPoolingLayers[i].Setup(m_rgPoolingBottomVec[i], m_rgPoolingTopVecs[i]);
                m_rgPoolingLayers[i].Reshape(m_rgPoolingBottomVec[i], m_rgPoolingTopVecs[i]);
                m_rgFlattenLayers[i].Reshape(m_rgPoolingTopVecs[i], m_rgFlattenLayerTopVecs[i]);
            }

            m_concat_layer.Reshape(m_colBlobConcatBottomVec, colTop);
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the output.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_nPyramidHeight == 1)
            {
                m_rgPoolingLayers[0].Forward(colBottom, colTop);
                return;
            }

            m_split_layer.Forward(colBottom, m_colBlobSplitTopVec);

            for (int i = 0; i < m_nPyramidHeight; i++)
            {
                m_rgPoolingLayers[i].Forward(m_rgPoolingBottomVec[i], m_rgPoolingTopVecs[i]);
                m_rgFlattenLayers[i].Forward(m_rgPoolingTopVecs[i], m_rgFlattenLayerTopVecs[i]);
            }

            m_concat_layer.Forward(m_colBlobConcatBottomVec, colTop);
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
            if (!rgbPropagateDown[0])
                return;

            if (m_nPyramidHeight == 1)
            {
                m_rgPoolingLayers[0].Backward(colTop, rgbPropagateDown, colBottom);
                return;
            }

            List<bool> rgbConcatPropagateDown = Utility.Create<bool>(m_nPyramidHeight, true);
            m_concat_layer.Backward(colTop, rgbConcatPropagateDown, m_colBlobConcatBottomVec);

            for (int i = 0; i < m_nPyramidHeight; i++)
            {
                m_rgFlattenLayers[i].Backward(m_rgFlattenLayerTopVecs[i], rgbPropagateDown, m_rgPoolingTopVecs[i]);
                m_rgPoolingLayers[i].Backward(m_rgPoolingTopVecs[i], rgbPropagateDown, m_rgPoolingBottomVec[i]);
            }

            m_split_layer.Backward(m_colBlobSplitTopVec, rgbPropagateDown, colBottom);
        }
    }
}
