using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.ptst
{
    /// <summary>
    /// The RevINLayer performs a Reversible Instance Normalization.  
    /// </summary>
    /// <remarks>
    /// This layer performs the reversible instance normalization that normalizes the inputs by centering the data then brining
    /// it to the unit variance and then applying a learnable affine weight and bias.  This layer performs both normalization and
    /// denormalization functions.  The output of the layer is a (B x T x Ch) in size.
    /// 
    /// @see [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://openreview.net/forum?id=cGDAkQo1C0p) by Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo, 2022, ICLR 2022
    /// @see [Github - ts-kim/RevIN](https://github.com/ts-kim/RevIN) by ts-kim, 2022, GitHub.
    /// @see [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730) by Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam, International conference on machine learning, 2022, arXiv:2211.14730
    /// @see [Github - yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) by yuqinie98, 2022, GitHub.
    /// 
    /// WORK IN PROGRESS.
    /// </remarks>
    public class RevINLayer<T> : Layer<T>
    {
        Blob<T> m_blobMean;
        Blob<T> m_blobStdev;
        Blob<T> m_blobLast;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type Gelu with parameter gelu_param</param>
        public RevINLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.REVIN;

            m_blobMean = new Blob<T>(cuda, log);
            m_blobStdev = new Blob<T>(cuda, log);
            m_blobLast = new Blob<T>(cuda, log);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            dispose(ref m_blobMean);
            dispose(ref m_blobStdev);
            dispose(ref m_blobLast);

            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobMean);
            col.Add(m_blobStdev);
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: x
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: y
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs, where the numeric blobs are ordered first, then the categorical blbos.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_param.revin_param.affine)
            {
                Blob<T> blobAffineWeight = new Blob<T>(m_cuda, m_log);
                Blob<T> blobAffineBias = new Blob<T>(m_cuda, m_log);

                blobAffineWeight.Reshape(m_param.revin_param.num_features, 1, 1, 1);
                blobAffineBias.Reshape(m_param.revin_param.num_features, 1, 1, 1);

                blobAffineWeight.SetData(1.0);
                blobAffineBias.SetData(0.0);

                blobs.Add(blobAffineWeight);
                blobs.Add(blobAffineBias);
            }
        }

        /// <summary>
        /// Reshape the top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nN = colBottom[0].num;
            int nC = colBottom[0].channels;

            m_blobMean.Reshape(nN, nC, 1, 1);
            m_blobStdev.Reshape(nN, nC, 1, 1);

            if (m_param.revin_param.subtract_last)
                m_blobLast.Reshape(nN, 1, colBottom[0].height, colBottom[0].width);

            colTop[0].ReshapeLike(colBottom[0]);    
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 1)
        ///  -# @f$ (N \times T \times H \times W) @f$ 
        ///     the numeric inputs @f$ x @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector)
        ///  -# @f$ (N \times T \times H \times W size) @f$
        ///     the computed outputs @f$ y @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_param.revin_param.mode == param.tft.RevINParameter.MODE.NORMALIZE)
            {
                if (m_param.revin_param.subtract_last)
                {
                }
                

            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the inputs.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient
        /// with respect to outputs
        ///  -# @f$ (N \times T \times H \times W) @f$
        ///     containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///     with respect to computed outputs @f$ y @f$
        /// </param>
        /// <param name="rgbPropagateDown">propagate_down see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times T \times H \times W) @f$
        ///     the inputs @f$ x @f$;  
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
        }
    }
}
