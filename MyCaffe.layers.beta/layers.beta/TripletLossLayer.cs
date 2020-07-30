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
    /// <H3>PRE ALPHA</H3>
    /// 
    /// TripletLoss Layer - this is the triplet loss layer used to calculate the triplet loss and gradients using the
    /// triplet loss method of learning.  The triplet loss method involves image triplets using the following format:
    ///     Anchor (A), Positives (P) and Negatives (N)
    ///     
    /// Use the DataSequenceLayer with k=1 with balanced_matches = false and output_labels = true to provide the required data sequencing noted above.
    ///     
    /// Where Anchors and Positives are from the same class and Negatives are from a different class.  In the basic algorithm,
    /// the distance between AP and AN are determined and the learning occurs by shrinking the distance between AP and increasing
    /// the distance between AN.
    /// 
    /// This layer is initialized with the MyCaffe.param.beta.TripleLossParameter.
    /// </summary>
    /// <remarks>
    /// * Python code for TripletLoss layer by luhaofang
    /// @see https://github.com/luhaofang/tripletloss/blob/master/tripletloss/tripletlosslayer.py
    /// 
    /// * C++ code for TripletLoss layer by eli-oscherovich in 'Triplet loss #3663' pull request on BVLC/caffe github.
    /// @see https://github.com/BVLC/caffe/pull/3663/commits/c6518fb5752344e1922eaa1b1eb686bae5cc3964 - for triplet loss layer implementation
    /// 
    /// For an explanation of the gradient calculations,
    /// @see http://stackoverflow.com/questions/33330779/whats-the-triplet-loss-back-propagation-gradient-formula/33349475#33349475 - for gradient calculations
    ///
    /// @see [Deep Metric Learning Using Triplet Network](https://arxiv.org/pdf/1412.6622.pdf) by Hoffer and Ailon, 2018, for triplet loss function.
    /// @see [One Shot learning, Siamese networks and Triplet Loss with Keras](https://medium.com/@crimy/one-shot-learning-siamese-networks-and-triplet-loss-with-keras-2885ed022352) by Craeymeersch, 2019.
    /// @see [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737v2) by Alexander Hermans, Lucas Beyer, and Bastian Leibe, 2017. 
    /// @see [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) by Florian Schroff, and Dmitry Kalenichenko, and James Philbin, 2015.
    /// @see [Generalisation and Sharing in Triplet Convnets for Sketch based Visual Search](https://arxiv.org/abs/1611.05301v1) by Tu Bui, Leonardo Ribeiro, Moacir Ponti, and John Collomosse, 2016.
    /// </remarks> 
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TripletLossLayer<T> : LossLayer<T>
    {
        Blob<T> m_blobDiffAP;
        Blob<T> m_blobDiffSqAP;
        Blob<T> m_blobDistSqAP;
        Blob<T> m_blobDiffAN;
        Blob<T> m_blobDiffSqAN;
        Blob<T> m_blobDistSqAN;
        Blob<T> m_blobDiffPN;
        Blob<T> m_blobSumVec;
        Blob<T> m_blobLossVec;
        Blob<T> m_blobWork;
        double m_dfAlpha;

        /// <summary>
        /// The TripletLossLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type TRIPLET_LOSS with parameter triplet_loss_param.
        /// </param>
        public TripletLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.TRIPLET_LOSS;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_blobDiffAP != null)
            {
                m_blobDiffAP.Dispose();
                m_blobDiffAP = null;
            }

            if (m_blobDiffSqAP != null)
            {
                m_blobDiffSqAP.Dispose();
                m_blobDiffSqAP = null;
            }

            if (m_blobDistSqAP != null)
            {
                m_blobDistSqAP.Dispose();
                m_blobDistSqAP = null;
            }

            if (m_blobDiffAN != null)
            {
                m_blobDiffAN.Dispose();
                m_blobDiffAN = null;
            }

            if (m_blobDiffSqAN != null)
            {
                m_blobDiffSqAN.Dispose();
                m_blobDiffSqAN = null;
            }

            if (m_blobDistSqAN != null)
            {
                m_blobDistSqAN.Dispose();
                m_blobDistSqAN = null;
            }

            if (m_blobDiffPN != null)
            {
                m_blobDiffPN.Dispose();
                m_blobDiffPN = null;
            }

            if (m_blobSumVec != null)
            {
                m_blobSumVec.Dispose();
                m_blobSumVec = null;
            }

            if (m_blobLossVec != null)
            {
                m_blobLossVec.Dispose();
                m_blobLossVec = null;
            }

            if (m_blobWork != null)
            {
                m_blobWork.Dispose();
                m_blobWork = null;
            }

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobDiffAP);
                col.Add(m_blobDiffSqAP);
                col.Add(m_blobDistSqAP);
                col.Add(m_blobDiffAN);
                col.Add(m_blobDiffSqAN);
                col.Add(m_blobDistSqAN);
                col.Add(m_blobDiffPN);
                col.Add(m_blobSumVec);
                col.Add(m_blobLossVec);
                col.Add(m_blobWork);

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: anchor, pos, neg, label
        /// </summary>
        /// <remarks>
        /// Use DataSequenceLayer with k=1, balanced_matching = false and output_labels = true to receive correct bottoms.
        /// </remarks>
        public override int ExactNumBottomBlobs
        {
            get { return 4; }   // Label is only added to consume it so that it is not treated as an output.
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: loss
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns <i>true</i> for all but the labels, for we want the loss value to be propagated back.
        /// </summary>
        /// <param name="nBottomIdx">Returns the index of the bottom for which the propagation should occur.</param>
        /// <returns><i>true</i> is returned for each bottom index to propagate back.</returns>
        public override bool AllowForceBackward(int nBottomIdx)
        {
            if (nBottomIdx <= 2)
                return true;

            return false;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.LayerSetUp(colBottom, colTop);
            m_dfAlpha = m_param.triplet_loss_param.alpha;

            m_blobDiffAP = new Blob<T>(m_cuda, m_log);
            m_blobDiffAP.Name = "positive delta";

            m_blobDiffSqAP = new Blob<T>(m_cuda, m_log, false);
            m_blobDiffSqAP.Name = "positive delta sq";

            m_blobDistSqAP = new Blob<T>(m_cuda, m_log, false);
            m_blobDistSqAP.Name = "positive dist sq";

            m_blobDiffAN = new Blob<T>(m_cuda, m_log);
            m_blobDiffAN.Name = "negative delta";

            m_blobDiffSqAN = new Blob<T>(m_cuda, m_log, false);
            m_blobDiffSqAN.Name = "negative delta sq";

            m_blobDistSqAN = new Blob<T>(m_cuda, m_log, false);
            m_blobDistSqAN.Name = "negative dist sq";

            m_blobDiffPN = new Blob<T>(m_cuda, m_log);
            m_blobDiffPN.Name = "pos/neg delta";

            m_blobSumVec = new Blob<T>(m_cuda, m_log, false);
            m_blobSumVec.Name = "summer vec";

            m_blobLossVec = new Blob<T>(m_cuda, m_log, false);
            m_blobLossVec.Name = "loss vec";

            m_blobWork = new Blob<T>(m_cuda, m_log, false);
            m_blobWork.Name = "work";
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_log.CHECK(Utility.Compare<int>(colBottom[0].shape(), colBottom[1].shape()), "Inputs must have the same dimension.");
            m_log.CHECK(Utility.Compare<int>(colBottom[0].shape(), colBottom[2].shape()), "Inputs must have the same dimension.");

            m_blobDiffAP.ReshapeLike(colBottom[0]);
            m_blobDiffSqAP.ReshapeLike(colBottom[0]);
            m_blobDiffAN.ReshapeLike(colBottom[0]);
            m_blobDiffSqAN.ReshapeLike(colBottom[0]);
            m_blobDiffPN.ReshapeLike(colBottom[0]);
            m_blobLossVec.ReshapeLike(colBottom[0]);

            int nNum = colBottom[0].num;
            int nDim = colBottom[0].count(1);
            m_blobSumVec.Reshape(nDim, 1, 1, 1);
            m_blobSumVec.SetData(1.0);

            m_blobWork.Reshape(nNum, 1, 1, 1);
            m_blobWork.SetData(0.0);

            m_blobDistSqAP.ReshapeLike(m_blobWork);
            m_blobDistSqAN.ReshapeLike(m_blobWork);

            List<int> rgLossShape = new List<int>();    // Loss layers output a scalar, 0 axes.
            colTop[0].Reshape(rgLossShape);
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 4)
        ///  -# @f$ (N/3 \times C \times H \times W) @f$
        ///     the anchors.
        ///  -# @f$ (N/3 \times C \times H \times W) @f$
        ///     the positives.
        ///  -# @f$ (N/3 \times C \times H \times W) @f$
        ///     the negatives.
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the labels.
        /// </param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$ 
        ///     computed loss. 
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(4, colBottom.Count, "The bottom must have 4 items: anchor, positives, negatives and label.");
            int nCount = colBottom[0].count();
            int nNum = colBottom[0].num;
            int nDim = colBottom[0].count(1);
            long hAnchor = colBottom[0].gpu_data;
            long hPositive = colBottom[1].gpu_data;
            long hNegative = colBottom[2].gpu_data;

            m_cuda.sub(nCount, hAnchor, hPositive, m_blobDiffAP.mutable_gpu_data);   // a_i - p_i
            m_cuda.sub(nCount, hAnchor, hNegative, m_blobDiffAN.mutable_gpu_data);   // a_i - n_i
            m_cuda.sub(nCount, hPositive, hNegative, m_blobDiffPN.mutable_gpu_data); // p_i - n_i

            m_cuda.powx(nCount, m_blobDiffAP.gpu_data, 2.0, m_blobDiffSqAP.mutable_gpu_data); // (a_i - p_i)^2
            m_cuda.gemv(false, nNum, nDim, 1.0, m_blobDiffSqAP.gpu_data, m_blobSumVec.gpu_data, 0.0, m_blobDistSqAP.mutable_gpu_data); // \Sum (a_i - p_i)^2

            m_cuda.powx(nCount, m_blobDiffAN.gpu_data, 2.0, m_blobDiffSqAN.mutable_gpu_data); // (a_i - p_i)^2
            m_cuda.gemv(false, nNum, nDim, 1.0, m_blobDiffSqAN.gpu_data, m_blobSumVec.gpu_data, 0.0, m_blobDistSqAN.mutable_gpu_data); // \Sum (a_i - p_i)^2

            double dfMargin = m_dfAlpha;

            m_cuda.sub(nNum, m_blobDistSqAP.gpu_data, m_blobDistSqAN.gpu_data, m_blobWork.mutable_gpu_data);
            m_cuda.add_scalar(nNum, dfMargin, m_blobWork.mutable_gpu_data);
            m_cuda.set_bounds(nNum, 0, float.MaxValue, m_blobWork.mutable_gpu_data);
            m_cuda.copy_expand(nCount, nNum, nDim, m_blobWork.gpu_data, m_blobLossVec.mutable_gpu_data);
            m_cuda.sign(nCount, m_blobLossVec.gpu_data, m_blobLossVec.mutable_gpu_data);

            double dfLoss = m_cuda.asum_double(nNum, m_blobWork.gpu_data);
            dfLoss /= (nNum * 2.0);

            colTop[0].SetData(dfLoss, 0);
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$ containing the loss.
        /// </param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 3)
        ///  -# @f$ (N/3 \times C \times H \times W) @f$ 
        ///     the anchors.
        ///  -# @f$ (N/3 \times C \times H \times W) @f$ 
        ///     the positives.
        ///  -# @f$ (N/3 \times C \times H \times W) @f$ 
        ///     the negatives.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hAnchor = colBottom[0].gpu_data;
            long hPositive = colBottom[1].gpu_data;
            long hNegative = colBottom[2].gpu_data;
            int nCount = colBottom[0].count();
            int nNum = colBottom[0].num;
            double dfMargin = m_dfAlpha;
            double dfDiff = convertD(colTop[0].GetDiff(0));
            double dfScale = dfDiff / (double)nNum;

            m_cuda.sub(nCount, hNegative, hPositive, m_blobDiffPN.mutable_gpu_diff);
            m_cuda.sub(nCount, hPositive, hAnchor, m_blobDiffAP.mutable_gpu_diff);
            m_cuda.sub(nCount, hAnchor, hNegative, m_blobDiffAN.mutable_gpu_diff);

            BlobCollection<T> colDiff = new BlobCollection<T>();
            colDiff.Add(m_blobDiffPN);
            colDiff.Add(m_blobDiffAP);
            colDiff.Add(m_blobDiffAN);

            m_blobLossVec.scale_data(dfMargin);

            for (int i = 0; i < 3; i++)
            {
                // calculate the gradients.
                m_cuda.scale(nCount, dfScale, colDiff[i].gpu_diff, colBottom[i].mutable_gpu_diff);

                // zero out all diff that have zero loss.
                m_cuda.mul(nCount, colBottom[i].gpu_diff, m_blobLossVec.gpu_data, colBottom[i].mutable_gpu_diff);
            }
        }
    }
}
