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
    /// The HingeLossLayer computes the hinge loss for a one-of-many classification task.
    /// This layer is initialized with the MyCaffe.param.HingeLossParameter.
    /// </summary>
    /// <remarks>
    /// @see [CNN-based Patch Matching for Optical Flow with Thresholded Hinge Loss](https://arxiv.org/abs/1607.08064) by Christian Bailer, Kiran Varanasi, and Didier Stricker, 2016.
    /// @see [Hinge-Loss Markov Random Fields and Probabilistic Soft Logic](https://arxiv.org/abs/1505.04406) by Stephen H. Bach, Matthias Broecheler, Bert Huang, and Lise Getoor, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class HingeLossLayer<T> : LossLayer<T>
    {
        /// <summary>
        /// The HingeLoss constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides LossParameter loss_param, with options:
        ///  - ignore_label (optional)
        ///    Specify a label value that whould be ignored when computing the loss.
        ///  - normalize (optional, default true)
        ///    If true, the loss is normalized by the number of (nonignored) labels
        ///    present; otherwise the loss is imply summed over spatial locations.
        /// </param>
        public HingeLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.HINGE_LOSS;
        }

        /** @copydoc LossLayer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ t @f$, a Blob with values in
        ///     @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
        ///     the @f$ K = CHW @f$ classes.  In an SVM, @f$ t @f$ is the result of
        ///     taking the inner product @f$ X^T W @f$ of the D-dimensional features
        ///     @f$ X \in \mathcal{R}^{D \times N} @f$ and the learned hyperplane
        ///     parameters @f$ W \in \mathcal{R}^{D \times K} @f$, so a Net with just
        ///     an InnerProductLayer (with num_output = @f$ D @f$) providing predictions to a
        ///     HingeLossLayer and no other learnable parameters or losses is
        ///     equivalent to an SVM.
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels @f$ l @f$, an integer-valued Blob with values
        ///     @f$ l_n \in [0, 1, 2, ..., K-1] @f$
        ///     indicating the correct class label among the @f$ K @f$ classes
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed hinge loss: @f$ E = 
        ///     \frac{1}{N} \sum\limits_{n=1}^N \sum\limits_{k=1}^K
        ///     [\max(0, 1 - \delta\{l_n = k\} t_{nk})] ^ p @f$,
        ///     for the @f$ L^p @f$ norm (defaults to @f$ p=1 @f$, the L1 norm; L2 norm, as in L2-SVM,
        ///     is also available), and @f$
        ///     \delta\{\mathrm{condition}\} = \left\{
        ///       \begin{array}{lr}
        ///         1 & \mbox{if condition} \\
        ///        -1 & \mbox{otherwise}
        ///       \end{array} \right. 
        ///     @f$
        /// </param>
        /// <remarks>
        /// In an SVM, @f$ t\in \mathcal{R}^{N x K} @f$ is the result of taking
        /// the inner product @f$ X^T W @f$ of the features
        /// @f$ X \in \mathcal{R}^{D x N} @f$
        /// and the learned hyperplane parameters
        /// @f$ W \in \mathcal{R}^{D x K} @f$.  So, a Net with just an
        /// InnerProductLayer (with num_output = @f$ k @f$) providing predictions to a
        /// HingeLossLayer is equivalent to an SVM (assuming it has no other learned
        /// outside the InnerProductLayer and no other losses outside the
        /// HingeLossLayer.
        /// </remarks>  
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int nNum = colBottom[0].num;
            int nCount = colBottom[0].count();
            int nDim = nCount / nNum;

            m_cuda.copy(nCount, hBottomData, hBottomDiff);

            T[] rgBottomDiff = colBottom[0].mutable_cpu_diff;
            T[] rgLabel = colBottom[1].update_cpu_data();

            if (typeof(T) == typeof(double))
            {
                double[] rgBottomDiffD = (double[])Convert.ChangeType(rgBottomDiff, typeof(double[]));
                double[] rgLabelD = (double[])Convert.ChangeType(rgLabel, typeof(double[]));

                for (int i = 0; i < nNum; i++)
                {
                    rgBottomDiffD[i * nDim + (int)rgLabelD[i]] *= -1;
                }
                for (int i = 0; i < nNum; i++)
                {
                    for (int j = 0; j < nDim; j++)
                    {
                        int nIdx = i * nDim + j;
                        double dfDiff = rgBottomDiffD[nIdx];
                        rgBottomDiffD[nIdx] = Math.Max(0.0, 1.0 + dfDiff);
                    }
                }
            }
            else
            {
                float[] rgBottomDiffF = (float[])Convert.ChangeType(rgBottomDiff, typeof(float[]));
                float[] rgLabelF = (float[])Convert.ChangeType(rgLabel, typeof(float[]));

                for (int i = 0; i < nNum; i++)
                {
                    rgBottomDiffF[i * nDim + (int)rgLabelF[i]] *= -1;
                }
                for (int i = 0; i < nNum; i++)
                {
                    for (int j = 0; j < nDim; j++)
                    {
                        int nIdx = i * nDim + j;
                        float fDiff = rgBottomDiffF[nIdx];
                        rgBottomDiffF[nIdx] = Math.Max(0.0f, 1.0f + fDiff);
                    }
                }
            }

            colBottom[0].mutable_cpu_diff = rgBottomDiff;

            double dfLoss = 0;
            switch (m_param.hinge_loss_param.norm)
            {
                case HingeLossParameter.Norm.L1:
                    dfLoss = convertD(m_cuda.asum(nCount, hBottomDiff)) / nNum;
                    break;

                case HingeLossParameter.Norm.L2:
                    dfLoss = convertD(m_cuda.dot(nCount, hBottomDiff, hBottomDiff)) / nNum;
                    break;

                default:
                    m_log.FAIL("Unknown norm in HingeLoss!");
                    break;
            }

            colTop[0].SetData(dfLoss, 0);
        }

        /// <summary>
        /// Computes the hinge loss error gradient w.r.t the predictions.
        /// </summary>
        /// <remarks>
        /// Gradients cannot be computed with respect to the label inputs (bottom[1]),
        /// so this method ignores bottom[1] and requires !propagate_down[1], crashing
        /// if propagate_down[1] == true.
        /// </remarks>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///   -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///      This blob's diff will simply contain the loss_weight * @f$ \lambda @f$ as
        ///      @f$ \lambda @f$ is the coefficient of this layer's output
        ///      @f$ \ell_i @f$ in the overall Net loss.
        ///      @f$ E = \lambda_i \ell_i + \mbox{other loss terms} @f$; hence
        ///      @f$ \frac{partial E}{\partial \ell_i} = \lambda_i @f$
        ///        (*Assuming that this top blob is not used as a bottom (input) by any
        ///        other layer of the Net.)
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.  propagate_down[1] must be false as
        /// we can't compute gradients with respect to the labels.</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$; backward computes diff
        ///       @f$ \frac{\partial E}{\partial x} @f$
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels -- ignored as we can't compute their error gradients.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[1])
                m_log.FAIL(type.ToString() + " Layer cannot backpropagate to laabel inputs.");

            if (rgbPropagateDown[0])
            {
                long hBottomDiff = colBottom[0].mutable_gpu_diff;
                int nNum = colBottom[0].num;
                int nCount = colBottom[0].count();
                int nDim = nCount / nNum;

                T[] rgBottomDiff = colBottom[0].mutable_cpu_diff;
                T[] rgLabel = colBottom[1].update_cpu_data();

                if (typeof(T) == typeof(double))
                {
                    double[] rgBottomDiffD = (double[])Convert.ChangeType(rgBottomDiff, typeof(double[]));
                    double[] rgLabelD = (double[])Convert.ChangeType(rgLabel, typeof(double[]));

                    for (int i = 0; i < nNum; i++)
                    {
                        rgBottomDiffD[i * nDim + (int)rgLabelD[i]] *= -1;
                    }
                }
                else
                {
                    float[] rgBottomDiffF = (float[])Convert.ChangeType(rgBottomDiff, typeof(float[]));
                    float[] rgLabelF = (float[])Convert.ChangeType(rgLabel, typeof(float[]));

                    for (int i = 0; i < nNum; i++)
                    {
                        rgBottomDiffF[i * nDim + (int)rgLabelF[i]] *= -1;
                    }
                }

                colBottom[0].mutable_cpu_diff = rgBottomDiff;

                double dfLossWeight = convertD(colTop[0].GetDiff(0));
                switch (m_param.hinge_loss_param.norm)
                {
                    case HingeLossParameter.Norm.L1:
                        m_cuda.sign(nCount, hBottomDiff, hBottomDiff);
                        m_cuda.scal(nCount, dfLossWeight / nNum, hBottomDiff);
                        break;

                    case HingeLossParameter.Norm.L2:
                        m_cuda.scal(nCount, dfLossWeight * 2 / nNum, hBottomDiff);
                        break;

                    default:
                        m_log.FAIL("Unknown norm in HingeLoss!");
                        break;
                }
            }
        }
    }
}