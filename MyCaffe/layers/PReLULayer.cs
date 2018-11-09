using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;

namespace MyCaffe.layers
{
    /// <summary>
    /// The PReLULayer computes the "Parameterized Rectified Linear Unit" non-linearity.
    /// This layer is initialized with the MyCaffe.param.PReLUParameter.
    /// </summary>
    /// <remarks>
    /// Computation: @f$ y = max(0,x_i) + a_i min(0, x_i) @f$
    /// <br/>
    /// The differences from ReLULayer are 1.) netative slopes are
    /// learnable through backprop and 2) negative slopes can vary across
    /// channels.  The number of axes of input blob should be greater than or
    /// equal to 2.  The 1st axis (0-based) is seen as channels.
    /// <br/>
    /// @see [Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853) by Bing Xu, Naiyan Wang, Tianqi Chen, and Mu Li, 2015.
    /// @see [Revise Saturated Activation Functions](https://arxiv.org/abs/1602.05980?context=cs) by Bing Xu, Ruitong Huang, and Mu Li, 2016.
    /// @see [Understanding Deep Neural Networks with Rectified Linear Units](https://arxiv.org/abs/1611.01491) by Raman Arora, Amitabh Basu, Poorya Mianjy, and Anirbit Mukherjee, 2016.
    /// @see [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852v1) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class PReLULayer<T> : NeuronLayer<T>
    {
        bool m_bChannelShared;
        Blob<T> m_blobMultiplier;      // dot multiplier for backward computation of params.
        Blob<T> m_blobBackwardBuff;    // temporary buffer for backward computation.
        Blob<T> m_blobBottomMemory;    // memory for in-place computation.


        /// <summary> 
        /// The PReLULayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type PRELU with parameter prelu_param,
        /// with options:
        ///     - filler (/b optional, default = "constant", 0.25).  The filler used to fill the learnable parameters.
        ///     
        ///     - channel_shared (\b optional, default false). Whether or not slope parameters are shared across channels. 
        /// </param>
        public PReLULayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.PRELU;
            m_blobMultiplier = new Blob<T>(cuda, log);
            m_blobMultiplier.Name = m_param.name + " mult";
            m_blobBackwardBuff = new Blob<T>(cuda, log);
            m_blobBackwardBuff.Name = m_param.name + " backbuf";
            m_blobBottomMemory = new Blob<T>(cuda, log);
            m_blobBottomMemory.Name = m_param.name + " btmmem";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobBottomMemory.Dispose();
            m_blobBackwardBuff.Dispose();
            m_blobMultiplier.Dispose();

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobMultiplier);
                col.Add(m_blobBackwardBuff);
                col.Add(m_blobBottomMemory);

                return col;
            }
        }

        /// <summary>
        /// Re-initialize the parameters of the layer.
        /// </summary>
        /// <returns>When handled, this method returns <i>true</i>, otherwise <i>false</i>.</returns>
        public override bool ReInitializeParameters()
        {
            base.ReInitializeParameters();

            FillerParameter fp = m_param.prelu_param.filler;
            if (fp == null)
                fp = new FillerParameter("constant", 0.25);

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(m_colBlobs[0]);

            return true;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_GE(colBottom[0].num_axes, 2, "Number of axes of bottom must be >= 2");
            PReLUParameter p = m_param.prelu_param;
            int nChannels = colBottom[0].channels;

            m_bChannelShared = p.channel_shared;

            if (m_colBlobs.Count > 0)
            {
                m_log.WriteLine("Skipping parameter initialization.");
            }
            else
            {
                m_colBlobs = new BlobCollection<T>();

                List<int> rgSlopeShape = new List<int>();
                if (!m_bChannelShared)
                    rgSlopeShape.Add(nChannels);

                Blob<T> blobSlope = new Blob<T>(m_cuda, m_log);
                blobSlope.Name = m_param.name + " slope";

                if (!shareParameter(blobSlope, rgSlopeShape))
                {
                    blobSlope.Reshape(rgSlopeShape);
                    FillerParameter fp = p.filler;

                    if (fp == null)
                        fp = new FillerParameter("constant", 0.25);

                    Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                    filler.Fill(blobSlope);
                }
                m_colBlobs.Add(blobSlope);
            }

            if (m_bChannelShared)
                m_log.CHECK_EQ(m_colBlobs[0].count(), 1, "Negative slope size is inconsistent with prototxt config.");
            else
                m_log.CHECK_EQ(m_colBlobs[0].count(), nChannels, "Negative slope size is inconsistent with prototxt config.");

            // Propagate gradients to the parameters (as directed by backward pass)
            m_rgbParamPropagateDown = new DictionaryMap<bool>(m_colBlobs.Count, true);

            List<int> rgShape = new List<int>() { colBottom[0].count(1) };

            m_blobMultiplier.Reshape(rgShape);
            m_blobBackwardBuff.Reshape(rgShape);
            m_blobMultiplier.SetData(1.0);
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

            if (colBottom[0] == colTop[0])
            {
                // For in-place computation.
                m_blobBottomMemory.ReshapeLike(colBottom[0]);
            }
        }

        /// <summary>
        /// Forward operation
        /// </summary>
        /// <param name="colBottom">
        /// bottom input Blob<T> vector (length 1)
        ///     -# @f$ (N \times C \times ...) @f$
        ///     the inputs @f$ x @f$
        /// </param>
        /// <param name="colTop">
        /// top output Blob<T> vector (length 1)
        ///     -# @f$ (N \times C \times ...) @f$
        ///     the computed outputs for each channel @f$ i @f$
        ///     @f$ y_i = max(0, x_i) + a_i min(0, x_i) @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();
            int nDim = colBottom[0].count(2);
            int nChannels = colBottom[0].channels;
            long hSlopeData = m_colBlobs[0].gpu_data;
            int nDivFactor = m_bChannelShared ? nChannels : 1;

            if (colTop[0] == colBottom[0])
                m_cuda.copy(nCount, hBottomData, m_blobBottomMemory.mutable_gpu_data);

            m_cuda.prelu_fwd(nCount, nChannels, nDim, hBottomData, hTopData, hSlopeData, nDivFactor);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the PReLU inputs.
        /// </summary>
        /// <param name="colTop">
        /// top output Blob<T> vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///     -# @f$ (N \times C \times ...) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$.
        /// </param>
        /// <param name="rgbPropagateDown">
        /// see Layer::backward.
        /// </param>
        /// <param name="colBottom">
        /// bottom input Blob<T> vector (length 1)
        ///     -# @f$ (N \times C \times ...) @f$
        ///     the inputs @f$ x @f$; For each channel @f$ i @f$ backward fills their
        ///     diff with gradients @f$
        ///         \frac{\partial E}{\partial x_i} = \left\{
        ///         \begin{array}{lr}
        ///             a_i \frac{\partial E}{\partial y_i} & \mathrm{if} \; x_i \le 0 \\
        ///             \frac{\partial E}{\partial y_i} & \mathrm{if} \; x_i > 0
        ///         \end{array} \right.
        ///     @f$
        ///     If param_propagate_down[0] == true, it fills the diff with gradients
        ///     @f$
        ///         \frac{\partial E}{\partial a_i} = \left\{
        ///         \begin{array}{lr}
        ///             \sum_{x_i} x_i \frac{\partial E}{\partial y_i} & \mathrm{if} \; x_i \le 0 \\
        ///             0 & \mathrm{if} \; x_i > 0
        ///         \end{array} \right.
        ///     @f$.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            int nCount = colBottom[0].count();
            int nDim = colBottom[0].count(2);
            int nChannels = colBottom[0].channels;

            // for In-place computation
            if (colTop[0] == colBottom[0])
                hBottomData = m_blobBottomMemory.gpu_data;

            // Propagate to param
            // Since to write bottom diff will affect top diff if top and bottom blobs
            // are identical (in-place computation), we first compute param backward to
            // keep top_diff unchanged.
            if (m_rgbParamPropagateDown[0])
            {
                long hSlopeDiff = m_colBlobs[0].mutable_gpu_diff;
                int nCDim = nChannels * nDim;

                m_cuda.prelu_bwd_param(nCDim, colBottom[0].num, colTop[0].offset(1), hTopDiff, hBottomData, m_blobBackwardBuff.mutable_gpu_diff);

                if (m_bChannelShared)
                {
                    T dfSum = m_cuda.dot(nCDim, m_blobBackwardBuff.gpu_diff, m_blobMultiplier.gpu_data);
                    m_cuda.add_scalar(m_colBlobs[0].count(), dfSum, hSlopeDiff);
                }
                else
                {
                    m_cuda.gemv(false, nChannels, nDim, m_tOne, m_blobBackwardBuff.gpu_diff, m_blobMultiplier.gpu_data, m_tOne, hSlopeDiff);
                }
            }

            // Propagate to bottom
            if (rgbPropagateDown[0])
            {
                long hBottomDiff = colBottom[0].mutable_gpu_diff;
                long hSlopeData = m_colBlobs[0].gpu_data;
                int nDivFactor = m_bChannelShared ? nChannels : 1;

                m_cuda.prelu_bwd(nCount, nChannels, nDim, hTopDiff, hBottomData, hBottomDiff, hSlopeData, nDivFactor);
            }
        }
    }
}
