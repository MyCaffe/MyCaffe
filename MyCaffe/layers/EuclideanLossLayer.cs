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
    /// The EuclideanLossLayer computes the Euclidean (L2) loss @f$
    ///     E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
    ///         \right| \right|_2^2 @f$ for real-valued regression tasks.
    /// </summary>
    /// <remarks>
    /// This can be used for least-squares regression tasks.  An InnerProductLayer
    /// input to a EuclideanLossLayer exactly formulates a linear least squares
    /// regression problem.  With non-zero weight decay the problem becomes one of
    /// ridge regression -- see MyCaffe.test/TestGradientBasedSolver.cs for a concrete
    /// example wherein we check that the gradients computed for a Net with exactly
    /// this structure match hand-computed gradient formulas for ridge regression.
    /// 
    /// (Note: Caffe, and SGD in general, is certainly not the best way to solve
    /// linear least squares problems! We use it only as an instructive example.)
    /// 
    /// @see [Linking Image and Text with 2-Way Nets](https://arxiv.org/abs/1608.07973) by Aviv Eisenschtat,  and Lior Wolf, 2016.
    /// @see [Constrained Structured Regression with Convolutional Neural Networks](https://arxiv.org/abs/1511.07497) by Deepak Pathak, Philipp Krähenbühl, Stella X. Yu, and Trevor Darrell, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class EuclideanLossLayer<T> : LossLayer<T>
    {
        Blob<T> m_blobDiff;

        /// <summary>
        /// The EuclideanLossLayer constructor
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type EUCLIDEAN_LOSS.</param>
        public EuclideanLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.EUCLIDEAN_LOSS;
            m_blobDiff = new Blob<T>(cuda, log);
            m_blobDiff.Name = m_param.name + " diff";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobDiff.Dispose();
            base.dispose();
        }

        /// <summary>
        /// Unlike most loss layers, in the EuclideanLossLayer we can backpropagate
        /// to both inputs -- override to return true and always allow force_backward.
        /// </summary>
        /// <param name="nBottomIdx">Specifies the index of the bottom element.</param>
        /// <returns>Returns <i>true</i>.</returns>
        public override bool AllowForceBackward(int nBottomIdx)
        {
            return true;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_log.CHECK_EQ(colBottom[0].count(1), colBottom[1].count(1), "Inputs must have the same dimension.");
            m_blobDiff.ReshapeLike(colBottom[0]);
        }

        /// <summary>
        /// Forward computation
        /// </summary>
        /// <param name="colBottom">inpub Blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ \hat{y} \in [-\infty, +\infty] @f$
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the targets @f$ y \in [-infty, +\infty] @f$
        ///  </param>
        /// <param name="colTop">top output Blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed Euclidean loss: @f$  
        ///     E = \frac{1}{2n} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
        ///         \right| \right|_2^2 @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colBottom[0].count();

            m_cuda.sub(nCount, colBottom[0].gpu_data, colBottom[1].gpu_data, m_blobDiff.mutable_gpu_data);
            T fDot = m_cuda.dot(nCount, m_blobDiff.gpu_data, m_blobDiff.gpu_data);
            double dfLoss = convertD(fDot) / colBottom[0].num / 2.0;

            colTop[0].SetData(dfLoss, 0);
        }

        /// <summary>
        /// Computes the Euclidean error gradient w.r.t. the inputs.
        /// </summary>
        /// <remarks>
        /// Unlike other children of LossLayer, EuclideanLossLayer can compute
        /// gradients with respect to the label inputs bottom[1] (but still only will
        /// if propagate_down[1] is set, due to being produced by learnable parameters
        /// or if force_backward is set). In fact, this layer is 'compmutative' -- the
        /// result is the same regardless of the order of the two bottoms.
        /// </remarks>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     This blob's diff will simply contain the loss_weight * @f$ \lambda @f$,
        ///     as @f$ \lambda @f$ is the coefficient of this layer's output
        ///     @f$\ell_i@f$ in the overall Net loss.
        ///     @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
        ///     @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
        ///     (*Assuming that this top Blob is not used by a bottom (input) by any
        ///     other layer of the Net.)</param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$\hat{y}@f$; Backward fills their diff with
        ///     gradients @f$
        ///      \frac{\partial E}{\partial \hat{y}} = 
        ///        \frac{1}[n} \sum\limits_{n-1}^N (\hat{y}_n - y_n)
        ///     @f$ if propagate_down[0] == true.
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the targets @f$ y @f$; Backward fills their diff with gradients
        ///     @f$ \frac{\partial E}{\partial y} = 
        ///         \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
        ///     @f$ if propagate_down[1] == true.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            for (int i = 0; i < 2; i++)
            {
                if (rgbPropagateDown[i])
                {
                    double dfSign = (i == 0) ? 1 : -1;
                    double dfTopDiff = convertD(colTop[0].GetDiff(0));
                    double dfAlpha = dfSign * dfTopDiff / colBottom[i].num;
                    int nCount = colBottom[i].count();

                    m_cuda.axpby(nCount, convert(dfAlpha), m_blobDiff.gpu_data, m_tZero, colBottom[i].mutable_gpu_diff);
                }
            }
        }
    }
}
