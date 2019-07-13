using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The Normalization1Layer performs an L2 normalization over the input data.
    /// This layer is initialized with the MyCaffe.param.Normalization1Parameter.
    /// </summary>
    /// <remarks>
    /// Original C++ code added by Binbin Xu; declanxu@gmail.com or declanxu@126.com
    /// @see [Layer Normalization](https://arxiv.org/abs/1607.06450) by Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class Normalization1Layer<T> : Layer<T>
    {
        Blob<T> m_blobSquared;

        /// <summary>
        /// The Normalization1Layer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type NORMALIZATION1 with parameter normalization1_param,
        /// with options:
        ///   - norm (\b optional, default L2). The normalization mode to use: L1 or L2.
        /// </param>
        public Normalization1Layer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.NORMALIZATION1;
            m_blobSquared = new Blob<T>(cuda, log);
            m_blobSquared.Name = "squared";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobSquared.Dispose();
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobSquared);

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
            colTop[0].ReshapeLike(colBottom[0]);
            m_blobSquared.ReshapeLike(colBottom[0]);
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
            long hSquaredData = m_blobSquared.mutable_gpu_data;
            double dfNormSqr;
            int n = colBottom[0].num;
            int d = colBottom[0].count() / n;

            m_cuda.powx(n * d, hBottomData, 2.0, hSquaredData);

            for (int i = 0; i < n; i++)
            {
                dfNormSqr = m_cuda.asum_double(d, hSquaredData, i * d);
                dfNormSqr = Math.Pow(dfNormSqr, -0.5);
                m_cuda.scale(d, convert(dfNormSqr), hBottomData, hTopData, i * d, i * d);
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
            long hTopDiff = colTop[0].gpu_diff;
            long hTopData = colTop[0].gpu_data;
            long hBottomData = colBottom[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int n = colTop[0].num;
            int d = colTop[0].count() / n;
            T a;

            for (int i = 0; i < n; i++)
            {
                a = m_cuda.dot(d, hTopData, hTopDiff, i * d, i * d);
                m_cuda.scale(d, a, hTopData, hBottomDiff, i * d, i * d);
                m_cuda.sub(d, hTopDiff, hBottomDiff, hBottomDiff, i * d, i * d, i * d);
                a = m_cuda.dot(d, hBottomData, hBottomData, i * d, i * d);
                double dfA = Math.Pow(convertD(a), -0.5);
                m_cuda.scale(d, convert(dfA), hBottomDiff, hBottomDiff, i * d, i * d);
            }
        }
    }
}
