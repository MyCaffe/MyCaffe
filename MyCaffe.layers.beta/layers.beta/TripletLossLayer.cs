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
        Blob<T> m_blobPreGenTargetsPos;
        Blob<T> m_blobPreGenTargetsNeg;
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

            m_blobDiffAP = new Blob<T>(m_cuda, m_log);
            m_blobDiffAP.Name = m_param.name + ".positive delta";

            m_blobDiffSqAP = new Blob<T>(m_cuda, m_log, false);
            m_blobDiffSqAP.Name = m_param.name + ".positive delta sq";

            m_blobDistSqAP = new Blob<T>(m_cuda, m_log, false);
            m_blobDistSqAP.Name = m_param.name + ".positive dist sq";

            m_blobDiffAN = new Blob<T>(m_cuda, m_log);
            m_blobDiffAN.Name = m_param.name + ".negative delta";

            m_blobDiffSqAN = new Blob<T>(m_cuda, m_log, false);
            m_blobDiffSqAN.Name = m_param.name + ".negative delta sq";

            m_blobDistSqAN = new Blob<T>(m_cuda, m_log, false);
            m_blobDistSqAN.Name = m_param.name + ".negative dist sq";

            m_blobDiffPN = new Blob<T>(m_cuda, m_log);
            m_blobDiffPN.Name = m_param.name + ".pos/neg delta";

            m_blobSumVec = new Blob<T>(m_cuda, m_log, false);
            m_blobSumVec.Name = m_param.name + ".summer vec";

            m_blobLossVec = new Blob<T>(m_cuda, m_log, false);
            m_blobLossVec.Name = m_param.name + ".loss vec";

            m_blobWork = new Blob<T>(m_cuda, m_log);
            m_blobWork.Name = m_param.name + ".work";
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

            if (m_blobPreGenTargetsPos != null)
            {
                m_blobPreGenTargetsPos.Dispose();
                m_blobPreGenTargetsPos = null;
            }

            if (m_blobPreGenTargetsNeg != null)
            {
                m_blobPreGenTargetsNeg.Dispose();
                m_blobPreGenTargetsNeg = null;
            }

            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

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

            if (m_blobPreGenTargetsPos != null)
                col.Add(m_blobPreGenTargetsPos);

            if (m_blobPreGenTargetsNeg != null)
                col.Add(m_blobPreGenTargetsNeg);
        }

        /// <summary>
        /// Returns the exact number of bottom blobs which are variable so -1 is returned.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the minimum number of bottom blobs: anchor, positive, negative, label
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 4; } // anchor, positive, negative, label
        }

        /// <summary>
        /// Returns the maximum number of bottom blobs: anchor, positive, negative, label, centroids (from decode layer)
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 5; } // anchor, positive, negative, label, cetroids (from decode layer)
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

            // If the fifth bottom exists (the centroids) initialize the pregen targets.
            if (colBottom.Count == 5)
            {
                m_blobPreGenTargetsNeg = new Blob<T>(m_cuda, m_log, false);
                m_blobPreGenTargetsNeg.Name = "pregen neg";
                m_blobPreGenTargetsPos = new Blob<T>(m_cuda, m_log);
                m_blobPreGenTargetsPos.Name = "pregen pos";
            }
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

            if (m_blobPreGenTargetsNeg != null)
                m_blobPreGenTargetsNeg.ReshapeLike(colBottom[0]);

            if (m_blobPreGenTargetsPos != null)
                m_blobPreGenTargetsPos.ReshapeLike(colBottom[0]);
        }

        /// <summary>
        /// Loads the pre-gen targets, only made public for testing.
        /// </summary>
        /// <param name="lbl">Specifies the blob containing the labels.</param>
        /// <param name="tgt">Specifies the blob containing the pre-generated targets.</param>
        /// <param name="tgtNeg">Specifies the blob where the negatively matching targets are copied.</param>
        /// <param name="tgtPos">Specifies the blob where the positively matching targets are copied.</param>
        public void loadPreGenTargets(Blob<T> lbl, Blob<T> tgt, Blob<T> tgtNeg, Blob<T> tgtPos)
        {
            float[] rgLabels = convertF(lbl.update_cpu_data());
            int nLblDim = lbl.count(1);
            int nLblNum = tgt.num;
            int nNum = lbl.num;
            int nDim = tgt.count(1);
            Random rand = new Random();
            List<int> rgLabelVals = new List<int>();
            Dictionary<int, List<int>> rgrgLabelSel = new Dictionary<int, List<int>>();

            for (int i = 0; i < tgt.num; i++)
            {
                rgLabelVals.Add(i + m_param.triplet_loss_param.pregen_label_start);
                rgrgLabelSel.Add(i + m_param.triplet_loss_param.pregen_label_start, new List<int>());
            }

            m_log.CHECK_EQ(nNum, tgtNeg.num, "The neg targets have an incorrect num!");
            m_log.CHECK_EQ(nNum, tgtPos.num, "The pos targets have an incorrect num!");
            m_log.CHECK_EQ(nDim, tgtNeg.count(1), "The neg targets have an incorrect dim!");
            m_log.CHECK_EQ(nDim, tgtPos.count(1), "The pos targets have an incorrect dim!");

            for (int i = 0; i < nNum; i++)
            {
                int nLabel = (int)rgLabels[i * nLblDim];


                // Copy the positive to match the anchor label.
                m_cuda.copy(nDim, tgt.gpu_data, tgtPos.mutable_gpu_data, nLabel * nDim, i * nDim);

                // Copy the negative to NOT match the anchor label.
                if (rgrgLabelSel[nLabel].Count == 0)
                {
                    for (int l = 0; l < rgLabelVals.Count; l++)
                    {
                        if (rgLabelVals[l] != nLabel)
                            rgrgLabelSel[nLabel].Add(rgLabelVals[l]);
                    }
                }

                int nLabelIdx = rand.Next(rgrgLabelSel[nLabel].Count);
                int nLabelX = rgrgLabelSel[nLabel][nLabelIdx];
                rgrgLabelSel[nLabel].Remove(nLabelX);

                m_cuda.copy(nDim, tgt.gpu_data, tgtNeg.mutable_gpu_data, nLabelX * nDim, i * nDim);
            }
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
            m_log.CHECK_GE(colBottom.Count, 4, "The bottom must have at least 4 items: anchor, positives, negatives and label.");
            int nCount = colBottom[0].count();
            int nNum = colBottom[0].num;
            int nDim = colBottom[0].count(1);
            long hAnchor = colBottom[0].gpu_data;
            long hPositive = colBottom[1].gpu_data;
            long hNegative = colBottom[2].gpu_data;


            m_blobWork.Reshape(nNum, 1, 1, 1);

            m_log.CHECK_EQ(colBottom.Count, 4, "Currently, external targts such as centroids are not supported.");
            //if (colBottom.Count == 5)
            //    loadPreGenTargets(colBottom[3], colBottom[4], m_blobPreGenTargetsNeg, m_blobPreGenTargetsPos);

            m_cuda.sub(nCount, hAnchor, hPositive, m_blobDiffAP.mutable_gpu_data);   // a_i - p_i
            m_cuda.sub(nCount, hAnchor, hNegative, m_blobDiffAN.mutable_gpu_data);   // a_i - n_i

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
            int nCount = colBottom[0].count();
            int nNum = colBottom[0].num;
            double dfDiff = convertD(colTop[0].GetDiff(0));
            double dfAlpha = dfDiff / (double)nNum;
            long hAnchor = colBottom[0].gpu_data;
            long hPositive = colBottom[1].gpu_data;
            long hNegative = colBottom[2].gpu_data;

            m_blobLossVec.scale_data(dfAlpha);

            if (rgbPropagateDown[0])
            {
                m_cuda.sub(nCount, hNegative, hPositive, m_blobDiffPN.mutable_gpu_diff);
                m_cuda.mul(nCount, m_blobLossVec.gpu_data, m_blobDiffPN.gpu_diff, colBottom[0].mutable_gpu_diff);
            }

            if (rgbPropagateDown[1])
            {
                m_cuda.sub(nCount, hPositive, hAnchor, m_blobDiffAP.mutable_gpu_diff);
                m_cuda.mul(nCount, m_blobLossVec.gpu_data, m_blobDiffAP.gpu_diff, colBottom[1].mutable_gpu_diff);
            }

            if (rgbPropagateDown[2])
            {
                m_cuda.sub(nCount, hAnchor, hNegative, m_blobDiffAN.mutable_gpu_diff);
                m_cuda.mul(nCount, m_blobLossVec.gpu_data, m_blobDiffAN.gpu_diff, colBottom[2].mutable_gpu_diff);
            }
        }
    }
}
