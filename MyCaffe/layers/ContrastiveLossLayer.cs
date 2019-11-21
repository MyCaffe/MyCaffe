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
    /// The ContrastiveLossLayer computes the contrastive loss @f$
    ///     E = \frac{1}{2N} \sum\limits_{n=1}^N \left(y\right) d^2 +
    ///         \left(1-y\right) \max \left(margin-d, 0\right)^2
    ///         @f$ where @f$
    ///         d = \left| \left| a_n - b_n \right| \right|_2 @f$. 
    /// This layer is initialized with the MyCaffe.param.ContrastiveLossParameter.
    /// </summary>
    /// <remarks>
    /// This can be used to train siamese networks.
    /// 
    /// @see [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549) by Luca Bertinetto, Jack Valmadre, João F. Henriques, Andrea Vedaldi, and Philip H. S. Torr, 2016.
    /// @see [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) by Raia Hadsel, Sumit Chopra, and Yann LeCun, 2006.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ContrastiveLossLayer<T> : LossLayer<T>
    {
        Blob<T> m_blobDiff; // cached for backward pass.
        Blob<T> m_blobDistSq; // cached for backward pass.
        Blob<T> m_blobDiffSq; // cached for backward pass.
        Blob<T> m_blobSummerVec; // tmp storage for gpu forward pass.
        T[] m_rgMatches = null;

        /// <summary>
        /// The ContrastiveLossLayer constructor.
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
        public ContrastiveLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.CONTRASTIVE_LOSS;

            m_blobDiff = new Blob<T>(cuda, log, false);
            m_blobDiff.Name = m_param.name + " diff";
            m_blobDistSq = new Blob<T>(cuda, log, false);
            m_blobDistSq.Name = m_param.name + " distsq";
            m_blobDiffSq = new Blob<T>(cuda, log, false);
            m_blobDiffSq.Name = m_param.name + " diffsq";
            m_blobSummerVec = new Blob<T>(cuda, log, false);
            m_blobSummerVec.Name = m_param.name + " sum";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            if (m_blobDiff != null)
            {
                m_blobDiff.Dispose();
                m_blobDiff = null;
            }

            if (m_blobDistSq != null)
            {
                m_blobDistSq.Dispose();
                m_blobDistSq = null;
            }

            if (m_blobDiffSq != null)
            {
                m_blobDiffSq.Dispose();
                m_blobDiffSq = null;
            }

            if (m_blobSummerVec != null)
            {
                m_blobSummerVec.Dispose();
                m_blobSummerVec = null;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: featA, featB, binSim
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 3; }
        }

        /// <summary>
        /// Returns -1 specifying a variable number of tops.
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Specifies the minimum number of required top (output) Blobs: loss
        /// </summary>
        public override int MinTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Specifies the maximum number of required top (output) Blobs: loss, matches
        /// </summary>
        public override int MaxTopBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Unlike most loss layers, in the ContrastiveLossLayer we can backpropagate
        /// to the first two inputs.
        /// </summary>
        public override bool AllowForceBackward(int nBottomIdx)
        {
            if (nBottomIdx != 2)
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

            m_log.CHECK_EQ(colBottom[0].channels, colBottom[1].channels, "the bottom[0] and bottom[1] should have equal channel values.");
            m_log.CHECK_EQ(1, colBottom[0].height, "The bottom[0] should have height = 1.");
            m_log.CHECK_EQ(1, colBottom[0].width, "The bottom[0] should have width = 1.");
            m_log.CHECK_EQ(1, colBottom[1].height, "The bottom[1] should have height = 1.");
            m_log.CHECK_EQ(1, colBottom[1].width, "The bottom[1] should have width = 1.");
            m_log.CHECK_EQ(1, colBottom[2].channels, "The bottom[2] should have channels = 1.");
            m_log.CHECK_EQ(1, colBottom[2].height, "The bottom[2] should have height = 1.");
            m_log.CHECK_EQ(1, colBottom[2].width, "The bottom[2] should have width = 1.");
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            base.Reshape(colBottom, colTop);

            m_blobDiff.Reshape(colBottom[0].num, colBottom[0].channels, 1, 1);
            m_blobDiffSq.Reshape(colBottom[0].num, colBottom[0].channels, 1, 1);
            m_blobDistSq.Reshape(colBottom[0].num, 1, 1, 1);
            // vector of ones used to sum along channels.
            m_blobSummerVec.Reshape(colBottom[0].channels, 1, 1, 1);
            m_blobSummerVec.SetData(1.0);

            if (colTop.Count > 1 && m_param.contrastive_loss_param.output_matches)
            {
                if (m_rgMatches == null || m_rgMatches.Length != colBottom[0].num)
                    m_rgMatches = new T[colBottom[0].num];

                colTop[1].Reshape(colBottom[0].num, 1, 1, 1);
            }
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 3)
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the features @f$ a \in [-\infty, +\infty]@f$
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the features @f$ b \in [-\infty, +\infty]@f$
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the binary similarity @f$ s \in [0, 1]@f$
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed contrastive loss: @f$ E = 
        ///       \frac{-1}{N} \sum\limits_{n=1}^N \left(y\right) d^2 +
        ///       \left(1-y\right) \max \left(margin-d, 0\right)^2
        ///       @f$ where @f$
        ///       d = \left| \left| a_n - b_n \right| \right|_2 @f$.
        /// </param>
        /// <remarks>
        /// This can be used to train siamese networks.
        /// </remarks>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colBottom[0].count();

            m_cuda.sub(nCount,
                       colBottom[0].gpu_data,           // a
                       colBottom[1].gpu_data,           // b
                       m_blobDiff.mutable_gpu_data);    // a_i - b_i

            m_cuda.powx(nCount,
                       m_blobDiff.mutable_gpu_data,     // a_i - b_i
                       2.0,
                       m_blobDiffSq.mutable_gpu_data);  // (a_i - b_i)^2

            m_cuda.gemv(false,
                       colBottom[0].num,
                       colBottom[0].channels,
                       1.0,
                       m_blobDiffSq.gpu_data,           // (a_i - b_i)^2
                       m_blobSummerVec.gpu_data,
                       0.0,
                       m_blobDistSq.mutable_gpu_data);  // \Sum (a_i - b_i)^2
            
            double dfMargin = m_param.contrastive_loss_param.margin;
            bool bLegacyVersion = m_param.contrastive_loss_param.legacy_version;
            double dfLoss = 0;

            if (typeof(T) == typeof(double))
            {
                double[] rgDistSq = Utility.ConvertVec<T>(m_blobDistSq.update_cpu_data());
                double[] rgSimPairs = Utility.ConvertVec<T>(colBottom[2].update_cpu_data());
                int nSimDim = colBottom[2].count(1);

                for (int i = 0; i < colBottom[0].num; i++)
                {
                    int nIdx = i * nSimDim;
                    double dfDist = (bLegacyVersion) ? dfMargin - rgDistSq[i] : dfMargin - Math.Sqrt(rgDistSq[i]);

                    if (((int)rgSimPairs[nIdx]) != 0)  // similar pairs
                    {
                        if (m_rgMatches != null)
                        {
                            if (dfDist >= 0)
                                m_rgMatches[i] = m_tOne;
                            else
                                m_rgMatches[i] = m_tZero;
                        }

                        dfLoss += rgDistSq[i];
                    }
                    else // dissimilar pairs
                    {
                        if (m_rgMatches != null)
                        {
                            if (dfDist >= 0)
                                m_rgMatches[i] = m_tZero;
                            else
                                m_rgMatches[i] = m_tOne;
                        }

                        dfDist = Math.Max(dfDist, 0);

                        if (bLegacyVersion)
                            dfLoss += dfDist;
                        else
                            dfLoss += dfDist * dfDist;
                    }
                }
            }
            else
            {
                float[] rgDistSq = Utility.ConvertVecF<T>(m_blobDistSq.update_cpu_data());
                float[] rgSimPairs = Utility.ConvertVecF<T>(colBottom[2].update_cpu_data());
                int nSimDim = colBottom[2].count(1);

                for (int i = 0; i < colBottom[0].num; i++)
                {
                    int nIdx = i * nSimDim;
                    double dfDist = (bLegacyVersion) ? dfMargin - rgDistSq[i] : dfMargin - Math.Sqrt(rgDistSq[i]);

                    if (((int)rgSimPairs[nIdx]) != 0)  // similar pairs
                    {
                        if (m_rgMatches != null)
                        {
                            if (dfDist >= 0)
                                m_rgMatches[i] = m_tOne;
                            else
                                m_rgMatches[i] = m_tZero;
                        }

                        dfLoss += rgDistSq[i];
                    }
                    else // dissimilar pairs
                    {
                        if (m_rgMatches != null)
                        {
                            if (dfDist >= 0)
                                m_rgMatches[i] = m_tZero;
                            else
                                m_rgMatches[i] = m_tOne;
                        }

                        dfDist = Math.Max(dfDist, 0);

                        if (bLegacyVersion)
                            dfLoss += dfDist;
                        else
                            dfLoss += dfDist * dfDist;
                    }
                }
            }

            dfLoss = dfLoss / (double)colBottom[0].num / 2.0;
            colTop[0].SetData(dfLoss, 0);

            if (colTop.Count > 1 && m_rgMatches != null)
                colTop[1].mutable_cpu_data = m_rgMatches;
        }

        /// <summary>
        /// Computes the infogain loss error gradient w.r.t the inputs.
        /// </summary>
        /// <remarks>
        /// Computes the gradients with respect to the two input vectors (bottom[0] and
        /// bottom[1]), but not the similarity label (bottom[2]).
        /// </remarks>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///   -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///      This blob's diff will simply contain the loss_weight * @f$ \lambda @f$, as
        ///      @f$ \lambda @f$ is the coefficient of this layer's output
        ///      @f$\ell_i@f$ in the overall Net loss.
        ///      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
        ///          @f$\frac{partial E}{\partial \ell_i} = \lambda_i @f$.
        ///      (*Assuming that this top blob is not used as a bottom (input) by any
        ///        other layer of the Net.)
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.  propagate_down[1] must be false as
        /// we can't compute gradients with respect to the labels (similarly for progagate_down[2] and
        /// the infogain matrix, if provided as bottom[2]).</param>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the features @f$a@f$; Backward fills their diff with 
        ///     gradients if propagate_down[0] == true.
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the features @f$b@f$; Backward fills their diff with gradients if
        ///     propagate_down[1] == true.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0] && !rgbPropagateDown[1])
                return;

            double dfTopDiff = convertD(colTop[0].GetDiff(0)) / colBottom[0].num;

            for (int i = 0; i < 2; i++)
            {
                if (rgbPropagateDown[i])
                {
                    int nCount = colBottom[0].count();
                    int nChannels = colBottom[0].channels;
                    double dfMargin = m_param.contrastive_loss_param.margin;
                    bool bLegacyVersion = m_param.contrastive_loss_param.legacy_version;
                    double dfSign = (i == 0) ? 1 : -1;
                    double dfAlpha = dfSign * dfTopDiff;

                    m_cuda.cll_bwd(nCount,
                                   nChannels,
                                   dfMargin,
                                   bLegacyVersion,
                                   dfAlpha,
                                   colBottom[2].gpu_data,   // pair similarity 0 or 1
                                   m_blobDiff.gpu_data,     // the cached eltwise difference between a and b
                                   m_blobDistSq.gpu_data,   // the cached square distance between a and b
                                   colBottom[i].mutable_gpu_diff);
                }
            }
        }
    }
}
