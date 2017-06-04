using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers.alpha
{
    /// <summary>
    /// <H3>PRE ALPHA</H3>
    /// 
    /// TripletLoss Layer - this is the triplet loss layer used to calculate the triplet loss and gradients using the
    /// triplet loss method of learning.  The triplet loss method involves image triplets using the following format:
    ///     Anchor (A), Positives (P) and Negatives (N),
    ///     
    /// Where Anchors and Positives are from the same class and Negatives are from a different class.  In the basic algorithm,
    /// the distance between AP and AN are determined and the learning occurs by shrinking the distance between AP and increasing
    /// the distance between AN.
    /// 
    /// This layer is initialized with the MyCaffe.param.TripleLossParameter.
    /// </summary>
    /// <remarks>
    /// * Initial Python code for TripletDataLayer/TripletSelectionLayer/TripletLossLayer by luhaofang/tripletloss on github. 
    /// See https://github.com/luhaofang/tripletloss - for general architecture
    /// 
    /// * Initial C++ code for TripletLoss layer by eli-oscherovich in 'Triplet loss #3663' pull request on BLVC/caffe github.
    /// See https://github.com/BVLC/caffe/pull/3663/commits/c6518fb5752344e1922eaa1b1eb686bae5cc3964 - for triplet loss layer implementation
    /// 
    /// For an explanation of the gradient calculations,
    /// See http://stackoverflow.com/questions/33330779/whats-the-triplet-loss-back-propagation-gradient-formula/33349475#33349475 - for gradient calculations
    /// 
    /// @see [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737v2) by Alexander Hermans, Lucas Beyer, and Bastian Leibe, 2017. 
    /// @see [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) by Florian Schroff, and Dmitry Kalenichenko, and James Philbin, 2015.
    /// @see [Generalisation and Sharing in Triplet Convnets for Sketch based Visual Search](https://arxiv.org/abs/1611.05301v1) by Tu Bui, Leonardo Ribeiro, Moacir Ponti, and John Collomosse, 2016.
    /// </remarks> 
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TripletLossLayer<T> : LossLayer<T>
    {
        Blob<T> m_blobDiffSameClass;
        Blob<T> m_blobDiffDiffClass;
        List<double> m_rgdfLoss = new List<double>();
        int m_nBatchSize;
        int m_nVecDimension;
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
            m_blobDiffSameClass = new Blob<T>(cuda, log, false);
            m_blobDiffSameClass.Name = "diff_pos";
            m_blobDiffDiffClass = new Blob<T>(cuda, log, false);
            m_blobDiffDiffClass.Name = "diff_neg";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobDiffSameClass);
                col.Add(m_blobDiffDiffClass);

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: anchor, pos, neg, label
        /// </summary>
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
            m_blobDiffSameClass.ReshapeLike(colBottom[0]);
            m_blobDiffDiffClass.ReshapeLike(colBottom[0]);

            List<int> rgLossShape = new List<int>();    // Loss layers output a scalar, 0 axes.
            colTop[0].Reshape(rgLossShape);

            m_nBatchSize = colBottom[0].shape(0);
            m_nVecDimension = colBottom[0].count() / m_nBatchSize;
            m_rgdfLoss = Utility.Create<double>(m_nBatchSize, 0);
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
            long hAnchor = colBottom[0].gpu_data;
            long hPositive = colBottom[1].gpu_data;
            long hNegative = colBottom[2].gpu_data;

            m_cuda.sub(nCount, hAnchor, hPositive, m_blobDiffSameClass.mutable_gpu_data);
            m_cuda.sub(nCount, hAnchor, hNegative, m_blobDiffDiffClass.mutable_gpu_data);

            double dfLoss = 0;

            for (int v = 0; v < m_nBatchSize; v++)
            {
                double dfAPdot = convertD(m_cuda.dot(m_nVecDimension, m_blobDiffSameClass.gpu_data, m_blobDiffSameClass.gpu_data, v * m_nVecDimension, v * m_nVecDimension));
                double dfANdot = convertD(m_cuda.dot(m_nVecDimension, m_blobDiffDiffClass.gpu_data, m_blobDiffDiffClass.gpu_data, v * m_nVecDimension, v * m_nVecDimension));
                m_rgdfLoss[v] = m_dfAlpha + dfAPdot - dfANdot;
                dfLoss += m_rgdfLoss[v];
            }

            dfLoss /= m_nBatchSize * 2.0;
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
            long hAnchorDiff = colBottom[0].mutable_gpu_diff;
            long hPositiveDiff = colBottom[1].mutable_gpu_diff;
            long hNegativeDiff = colBottom[2].mutable_gpu_diff;
            double dfScale = convertD(colTop[0].GetDiff(0)) / colBottom[0].num;
            int nCount = colBottom[0].count();

            m_cuda.sub(nCount, m_blobDiffSameClass.gpu_data, m_blobDiffDiffClass.gpu_data, hAnchorDiff);
            m_cuda.scal(nCount, dfScale, hAnchorDiff);

            m_cuda.scale(nCount, -dfScale, m_blobDiffSameClass.gpu_data, hPositiveDiff);
            m_cuda.scale(nCount, dfScale, m_blobDiffDiffClass.gpu_data, hNegativeDiff);

            for (int v = 0; v < m_nBatchSize; v++)
            {
                if (m_rgdfLoss[v] == 0)
                {
                    m_cuda.set(m_nVecDimension, hAnchorDiff, m_tZero, -1, v * m_nVecDimension);
                    m_cuda.set(m_nVecDimension, hPositiveDiff, m_tZero, -1, v * m_nVecDimension);
                    m_cuda.set(m_nVecDimension, hNegativeDiff, m_tZero, -1, v * m_nVecDimension);
                }
            }
        }
    }
}
