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
    /// The MultinomialLogicistLossLayer computes the multinomial logistc loss for a one-of-many
    /// classification task, directly taking a predicted probability
    /// distribution as input.
    /// </summary>
    /// <remarks>
    /// When predictions are not already a probability distribution, you should
    /// instead use the SoftmaxWithLossLayer, which maps predictions to a
    /// distribution using the SoftmaxLayer, before computing the multinomial
    /// logistic loss.  The SoftmaxWithLossLayer should be preferred over separate
    /// SoftmaxLayer + MultinomialLogisticLossLayer
    /// as its gradient computation is more numerically stable.
    /// 
    /// @see [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797) by Fausto Milletari, Nassir Navab and Seyed-Ahmad Ahmadi, 2016. 
    /// @see [Estimating Depth from Monocular Images as Classification Using Deep Fully Convolutional Residual Networks](https://arxiv.org/abs/1605.02305) by Yuanzhouhan Cao, Zifeng Wu and Chunhua Shen, 2016. 
    /// @see [Deep Multitask Architecture for Integrated 2D and 3D Human Sensing](https://arxiv.org/abs/1701.08985) by Alin-lonut Popa, Mihai Zanfir and Cristian Sminchisescu, 2017.
    /// @see [HD-CNN: Hierarchical Deep Convolutional Neural Network for Large Scale Visual Recognition](https://arxiv.org/abs/1410.0736) by Zhicheng Yan, Hao Zhang, Robinson Piramuthu, Vignesh Jadadeesh, Dennis DeCoste, Wei Di and Yizhou Yu, 2014.
    /// @see [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) by Jonathan Long, Evan Shelhamer and Trevor Darrell, 2014.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class MultinomialLogisticLossLayer<T> : LossLayer<T>
    {
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides LossParameter loss_param, with options:
        ///  - ignore_label (optional)
        ///    Specify a label value that whould be ignored when computing the loss.
        ///  - normalize (optional, default true)
        ///    If true, the loss is normalized by the number of (nonignored) labels
        ///    present; otherwise the loss is imply summed over spatial locations.
        /// </param>
        public MultinomialLogisticLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.MULTINOMIALLOGISTIC_LOSS;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_log.CHECK_EQ(1, colBottom[1].channels, "The bottom[1] should have 1 channel.");
            m_log.CHECK_EQ(1, colBottom[1].height, "The bottom[1] should have height = 1.");
            m_log.CHECK_EQ(1, colBottom[1].width, "The bottom[1] should have width = 1.");
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ \hat{p} @f$, a Blob with values in
        ///     [0,1] indicating the predicted probability of each of the
        ///     K = CHW classes.  Each prediction vector @f$ \hat{p}_n @f$
        ///     should sum to 1 as in a probability distribution:
        ///       @f$ \forall n \sum\limits_{k=1}^K \hat{p}_{nk} = 1 @f$
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels @f$ l @f$, an integer-valued Blob with values
        ///     @f$ l_n \in [0, 1, 2, ..., K-1] @f$
        ///     indicating the correct class label among the @f$ K @f$ classes
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed multinomial logistic loss: @f$ E = 
        ///       \frac{-1}{N} \sum\limits_{n=1}^N \log(\hat{p}_{n,l_n})
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNum = colBottom[0].num;
            int nDim = colBottom[0].count() / nNum;
            double dfLoss = 0;

            if (typeof(T) == typeof(double))
            {
                double[] rgBottomData = (double[])Convert.ChangeType(colBottom[0].update_cpu_data(), typeof(double[]));
                double[] rgBottomLabel = (double[])Convert.ChangeType(colBottom[1].update_cpu_data(), typeof(double[]));

                for (int i = 0; i < nNum; i++)
                {
                    int nLabel = (int)rgBottomLabel[i];

                    double dfProb = Math.Max(rgBottomData[i * nDim + nLabel], kLOG_THRESHOLD);
                    dfLoss -= Math.Log(dfProb);
                }
            }
            else
            {
                float[] rgBottomData = (float[])Convert.ChangeType(colBottom[0].update_cpu_data(), typeof(float[]));
                float[] rgBottomLabel = (float[])Convert.ChangeType(colBottom[1].update_cpu_data(), typeof(float[]));

                for (int i = 0; i < nNum; i++)
                {
                    int nLabel = (int)rgBottomLabel[i];

                    double dfProb = Math.Max(rgBottomData[i * nDim + nLabel], (float)kLOG_THRESHOLD);
                    dfLoss -= Math.Log(dfProb);
                }
            }

            colTop[0].SetData(dfLoss / nNum, 0);
        }

        /// <summary>
        /// Computes the infogain loss error gradient w.r.t the predictions.
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
        ///      @f$ E = \lambda_i \ell_i + \mbox{other loss terms} @f$; hence @f$
        ///        \frac{partial E}{\partial \ell_i} = \lambda_i @f$
        ///        (*Assuming that this top blob is not used as a bottom (input) by any
        ///        other layer of the Net.)
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.  propagate_down[1] must be false as
        /// we can't compute gradients with respect to the labels (similarly for progagate_down[2] and
        /// the infogain matrix, if provided as bottom[2]).</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ \hat{p} @f$; backward computes diff 
        ///     @f$
        ///       \frac{\partial E}{\partial \hat{p}} 
        ///     @f$
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels -- ignored as we can't compute their error gradients.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[1])
                m_log.FAIL(type.ToString() + " Layer cannot backpropagate to label inputs.");

            if (rgbPropagateDown[0])
            {
                int nNum = colBottom[0].num;
                int nDim = colBottom[0].count() / nNum;
                double dfScale = -1 * convertD(colTop[0].GetDiff(0)) / nNum;

                colBottom[0].SetDiff(0);

                if (typeof(T) == typeof(double))
                {
                    double[] rgBottomData = (double[])Convert.ChangeType(colBottom[0].update_cpu_data(), typeof(double[]));
                    double[] rgBottomLabel = (double[])Convert.ChangeType(colBottom[1].update_cpu_data(), typeof(double[]));
                    double[] rgBottomDiff = (double[])Convert.ChangeType(colBottom[0].mutable_cpu_diff, typeof(double[]));

                    for (int i = 0; i < nNum; i++)
                    {
                        int nLabel = (int)rgBottomLabel[i];

                        double dfProb = Math.Max(rgBottomData[i * nDim + nLabel], kLOG_THRESHOLD);
                        rgBottomDiff[i * nDim + nLabel] = dfScale / dfProb;
                    }

                    colBottom[0].mutable_cpu_data = (T[])Convert.ChangeType(rgBottomDiff, typeof(T[]));
                }
                else
                {
                    float[] rgBottomData = (float[])Convert.ChangeType(colBottom[0].update_cpu_data(), typeof(float[]));
                    float[] rgBottomLabel = (float[])Convert.ChangeType(colBottom[1].update_cpu_data(), typeof(float[]));
                    float[] rgBottomDiff = (float[])Convert.ChangeType(colBottom[0].mutable_cpu_diff, typeof(float[]));

                    for (int i = 0; i < nNum; i++)
                    {
                        int nLabel = (int)rgBottomLabel[i];

                        double dfProb = Math.Max(rgBottomData[i * nDim + nLabel], kLOG_THRESHOLD);
                        rgBottomDiff[i * nDim + nLabel] = (float)(dfScale / dfProb);
                    }

                    colBottom[0].mutable_cpu_data = (T[])Convert.ChangeType(rgBottomDiff, typeof(T[]));
                }
            }
        }
    }
}
