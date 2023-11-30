using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.beta;
using static MyCaffe.param.beta.BCEWithLogitsLossParameter;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The BCEWithLogitsLossLayer computes the loss using binary cross entropy with logistics regression using the function below.
    /// This type of loss is often used in image classification tasks by measuring how well a model is able to predict the correct 
    /// class for an input image.
    /// 
    /// @f$ \ell_n = -w_n \left[ t_n \cdot \log \sigma(x_n) + (1 - t_n) \cdot \log (1 - \sigma(x_n)) \right] @f$
    /// 
    /// where @f$ x_n @f$​ is the model output, @f$ t_n @f$​ is the target label, @f$ w_n @f$​ is the weight, and σ is the sigmoid function.
    /// The loss is then reduced by either taking the mean or the sum over all the elements, depending on the reduction argument.
    /// </summary>
    /// <remarks>
    /// @see [BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss) by PyTorch
    /// @see [What does BCEWithLogitsLoss actually do?](https://kamilelukosiute.com/2022/04/14/bce-with-logits-loss/) by Kamile Lukosiute, 2022
    /// @see [How to Use PyTorch's BCEWithLogitsLoss Function](https://reason.town/pytorch-bcewithlogitsloss/) by joseph, 2022
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.</typeparam>
    public class BCEWithLogitsLossLayer<T> : LossLayer<T>
    {
        Blob<T> m_blobLoss;
        Blob<T> m_blobWeights;
        bool m_bMeanReduction = false;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides LossParameter loss_param.
        /// </param>
        public BCEWithLogitsLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.BCE_WITH_LOGITS_LOSS;
            m_blobLoss = new Blob<T>(cuda, log, false);
            m_blobWeights = new Blob<T>(cuda, log, false);
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_blobLoss != null)
            {
                m_blobLoss.Dispose();
                m_blobLoss = null;
            }

            if (m_blobWeights != null)
            {
                m_blobWeights.Dispose();
                m_blobWeights = null;
            }

            base.dispose();
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs as variable.
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
            base.LayerSetUp(colBottom, colTop);

            if (m_param.bce_with_logits_loss_param.weights != null && m_param.bce_with_logits_loss_param.weights.Count > 0)
            {
                if (m_param.bce_with_logits_loss_param.weights.Count != colBottom[0].num * colBottom[0].channels)
                    m_log.FAIL("The weights count must equal the Num x Channels in the bottom(0) blob.");

                m_blobWeights.Reshape(colBottom[0].num, colBottom[0].channels, 1, 1);
                m_blobWeights.SetData(convert(m_param.bce_with_logits_loss_param.weights.ToArray()));
            }

            m_bMeanReduction = (m_param.bce_with_logits_loss_param.reduction == BCEWithLogitsLossParameter.REDUCTION.MEAN);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nOuterNum = colBottom[0].num;
            m_nInnerNum = colBottom[0].count(1);
            m_blobLoss.ReshapeLike(colBottom[0]);
            base.Reshape(colBottom, colTop);
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$, a blob with values in
        ///     @f$ [-\infty, +\infty] @f$ indicating the predicted values.
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the targets @f$ y @f$, a blob with values in
        ///     @f$ [-\infty, +\infty] @f$ indicating the target values.
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///     BSEWithLogitsLoss - the computed binary cross entropy loss:
        ///     @f$ \ell_n = -w_n \left[ t_n \cdot \log \sigma(x_n) + (1 - t_n) \cdot \log (1 - \sigma(x_n)) \right] @f$
        /// 
        ///     where @f$ x_n @f$​ is the model output, @f$ t_n @f$​ is the target label, @f$ w_n @f$​ is the weight, and σ is the sigmoid function.
        ///     The loss is then reduced by either taking the mean or the sum over all the elements, depending on the reduction argument.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hPredicted = colBottom[0].gpu_data;
            long hTarget = colBottom[1].gpu_data;
            long hWeights = 0;
            int nCount = colBottom[0].count();
            int nN = colBottom[0].num;

            if (m_blobWeights.count() > 0)
                hWeights = m_blobWeights.gpu_data;  

            m_log.CHECK_EQ(nCount, colBottom[1].count(), "The bottom(0) predicted and bottom(1) target must have the same shapes!");
            
            m_cuda.bce_with_logits_loss_fwd(nCount, nN, hPredicted, hTarget, hWeights, 0, m_blobLoss.mutable_gpu_data);

            T fVal = m_blobLoss.asum_data();
            double dfLoss = convertD(fVal);

            if (m_param.bce_with_logits_loss_param.reduction == BCEWithLogitsLossParameter.REDUCTION.MEAN)
                dfLoss /= nCount;
                       
            double dfNormalizer = get_normalizer(m_normalization, -1);
            colTop[0].SetData(dfLoss / dfNormalizer, 0);

            // Clear scratch memory to prevent with interfering with backward pass (see #602)
            colBottom[0].SetDiff(0);
        }

        /// <summary>
        /// Computes the BSE with logits loss error gradient w.r.t the predictions.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///   -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///      This blob's diff will simply contain the loss_weight * @f$ \lambda @f$ as
        ///      @f$ \lambda @f$ is the coefficient of this layer's output
        ///      @f$ \ell_i @f$ in the overall Net loss.
        ///      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}; hence
        ///        \frac{partial E}{\partial \ell_i} = \lambda_i
        ///      @f$
        ///        (*Assuming that this top blob is not used as a bottom (input) by any
        ///        other layer of the Net.)
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.  propagate_down[1] must be false as
        /// we can't compute gradients with respect to the labels.</param>
        /// <param name="colBottom">bottom input blob vector (length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the gradients @f$ \hat{x} @f$; backward computes diff @f$
        ///       \frac{\partial E}{\partial x}
        ///     @f$
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            long hPredicted = colBottom[0].gpu_data;
            long hTarget = colBottom[1].gpu_data;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int nCount = colBottom[0].count();
            int nN = colBottom[0].num;
            long hWeights = 0;

            if (m_blobWeights.count() > 0)
                hWeights = m_blobWeights.gpu_data;

            m_cuda.bce_with_logits_loss_bwd(nCount, nN, hPredicted, hTarget, hWeights, 0, m_bMeanReduction, hBottomDiff);

            double dfTopDiff = convertD(colTop[0].GetDiff(0));
            double dfNormalizer = get_normalizer(m_normalization, -1);
            double dfLossWeight = dfTopDiff / dfNormalizer;

            m_cuda.scal(nCount, dfLossWeight, hBottomDiff);
        }
    }
}
