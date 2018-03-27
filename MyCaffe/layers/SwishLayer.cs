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
    /// The SwishLayer provides a novel activation function that tends to work better than ReLU.
    /// This layer is initialized with the MyCaffe.param.SwishParameter.
    /// </summary>
    /// <remarks>
    /// Computes the swish non-linearity @f$ y = x \sigma (\beta x) @f$.
    /// 
    /// @see [Activation Functions](https://arxiv.org/abs/1710.05941v2) by Prajit Ramachandran, Barret Zoph, Quoc V. Le., 2017.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SwishLayer<T> : NeuronLayer<T>
    {
        SigmoidLayer<T> m_sigmoidLayer;
        Blob<T> m_blobSigmoidInput;
        Blob<T> m_blobSigmoidOutput;
        BlobCollection<T> m_colSigmoidBottom = new BlobCollection<T>();
        BlobCollection<T> m_colSigmoidTop = new BlobCollection<T>();

        /// <summary>
        /// The SwishLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Swish with parameter Swish_param,
        /// with options:
        ///     - beta (\b default 1) the value @f$ \beta @f$
        /// </param>
        public SwishLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SWISH;

            m_blobSigmoidInput = new Blob<T>(cuda, log);
            m_blobSigmoidOutput = new Blob<T>(cuda, log);

            LayerParameter sigmoidParam = new LayerParameter(LayerParameter.LayerType.SIGMOID);
            sigmoidParam.sigmoid_param.engine = p.swish_param.engine;

            m_sigmoidLayer = new SigmoidLayer<T>(cuda, log, sigmoidParam);
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        protected override void dispose()
        {
            m_blobSigmoidOutput.Dispose();
            m_blobSigmoidInput.Dispose();

            base.dispose();
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);

            m_colSigmoidBottom.Clear();
            m_colSigmoidBottom.Add(m_blobSigmoidInput);
            m_colSigmoidTop.Clear();
            m_colSigmoidTop.Add(m_blobSigmoidOutput);
            m_sigmoidLayer.LayerSetUp(m_colSigmoidBottom, m_colSigmoidTop);
        }

        /// <summary>
        /// Reshape the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_blobSigmoidInput.ReshapeLike(colBottom[0]);
            m_sigmoidLayer.Reshape(m_colSigmoidBottom, m_colSigmoidTop);
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
        ///         y = x \sigma (\beta x)
        ///     @f$.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hSigmoidInputData = m_blobSigmoidInput.mutable_gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colBottom[0].count();
            double dfBeta = m_param.swish_param.beta;

            m_cuda.copy(nCount, hBottomData, hSigmoidInputData);
            m_cuda.scal(nCount, dfBeta, hSigmoidInputData);
            m_sigmoidLayer.Forward(m_colSigmoidBottom, m_colSigmoidTop);
            m_cuda.mul(nCount, hBottomData, m_blobSigmoidOutput.gpu_data, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the Swish value inputs.
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
        ///     the inputs @f$ x @f$; Backward fills their diff with 
        ///     gradients @f$
        ///         \frac{\partial E}{\partial x}
        ///             = \frac{\partial E}{\partial y}(\beta y + 
        ///               \sigma (\beta x)(1 - \beta y))
        ///     @f$ if propagate_down[0]
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
            {
                long hTopData = colTop[0].gpu_data;
                long hTopDiff = colTop[0].gpu_diff;
                long hSigmoidOutputData = m_blobSigmoidOutput.gpu_data;
                long hBottomDiff = colBottom[0].mutable_gpu_diff;
                int nCount = colBottom[0].count();
                double dfBeta = m_param.swish_param.beta;

                m_cuda.swish_bwd(nCount, hTopDiff, hTopData, hSigmoidOutputData, hBottomDiff, dfBeta);
            }
        }
    }
}
