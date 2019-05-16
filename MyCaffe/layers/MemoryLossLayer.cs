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
    /// The MemoryLossLayer provides a method of performing a custom loss functionality.  Similar to the MemoryDataLayer,
    /// the MemoryLossLayer supports an event used to get the loss value.  This event is called OnGetLoss, which once
    /// retrieved is used for learning on the backward pass.
    /// </summary>
    /// <remarks>
    /// To use this layer, you must implement the OnGetLoss event.
    /// </remarks>
    /// <typeparam name="T"></typeparam>
    public class MemoryLossLayer<T> : LossLayer<T>
    {
        object m_userState = null;
        bool m_bEnableLoss = true;
        bool m_bWarningMade = false;

        /// <summary>
        /// The OnGetLoss event fires during each forward pass.  The value returned is saved,
        /// and applied on the backward pass during training.
        /// </summary>
        public event EventHandler<MemoryLossLayerGetLossArgs<T>> OnGetLoss;

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
        public MemoryLossLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.MEMORY_LOSS;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /// <summary>
        /// Optionally specifies a user-state that is passed to the OnGetLoss event.
        /// </summary>
        public object user_state
        {
            get { return m_userState; }
            set { m_userState = value; }
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();
                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs as variable.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return -1; }
        }

        /// <summary>
        /// Returns the minimum number of required bottom (output) Blobs: input 1.
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of required bottom (output) Blobs: input 1 & 2.
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: loss.
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

            if (!m_param.loss_param.normalization.HasValue)
                m_normalization = (m_param.loss_param.normalize) ? LossParameter.NormalizationMode.VALID : LossParameter.NormalizationMode.BATCH_SIZE;
            else
                m_normalization = m_param.loss_param.normalization.Value;

            m_bWarningMade = false;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            bool bUniformSize = true;
            int nAxis = colBottom[0].CanonicalAxisIndex(1);
            int nCount = colBottom[0].count(nAxis);

            for (int i = 1; i < colBottom.Count; i++)
            {
                int nCount1 = colBottom[i].count(nAxis);
                if (nCount1 != nCount)
                {
                    bUniformSize = false;
                    break;
                }
            }

            if (!bUniformSize)
            {
                if (!m_bWarningMade)
                {
                    m_log.WriteLine("WARNING: The MemoryDataLayer bottoms are not of uniform size, so the normalization will be set to NONE.");
                    m_bWarningMade = true;
                }

                m_normalization = LossParameter.NormalizationMode.NONE;
                m_nOuterNum = 0;
                m_nInnerNum = 0;
            }
            else
            {
                m_nOuterNum = colBottom[0].count(0, nAxis);
                m_nInnerNum = colBottom[0].count(nAxis + 1);
            }

            List<int> rgLossShape = new List<int>(); // Loss layers output a scalar, 0 axes.
            colTop[0].Reshape(rgLossShape);
            colTop[0].type = Blob<T>.BLOB_TYPE.LOSS;
        }

        /// <summary>
        /// The forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$, a blob with values in
        ///     @f$ [-\infty, +\infty] @f$ indicating the predicted score for eachy of
        ///     the K = CHW classes.  This layer maps these scores to a
        ///     probability distribution over classes using the softmax function @f$
        ///     \hat{p}_{nk} = \exp(x_{nk}) /
        ///     \left[\sum_{k'} \exp(x_{nk'})\right] @f$ (see MemoryLayer).
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels l, an integer valued blob with values @f$ l_n \in [0, 1, 2, ..., K-1] @f$
        ///     indicating the correct class label among the K classes.</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///     the computed cross_entropy classification loss: @f$ E = 
        ///     \frac{-1}{N} \sum\limits_{n=1}^N \log(\hat{p}_{n,l_n})
        ///     @f$ for softmax output class probabilities @f$ \hat{p} @f$.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (OnGetLoss == null)
                m_log.FAIL("The OnGetLoss event must be implemented.  Make sure the SolverParameter 'custom_trainer' points to a trainer that connects the OnGetLoss event.");

            double dfNormalizer = get_normalizer(m_normalization, -1);
            MemoryLossLayerGetLossArgs<T> e = new MemoryLossLayerGetLossArgs<T>(colBottom, m_userState, dfNormalizer);
            OnGetLoss(this, e);

            m_bEnableLoss = e.EnableLossUpdate;

            colTop[0].SetData(e.Loss / dfNormalizer, 0);
        }

        /// <summary>
        /// Backpropagates the previously acquired (within the forward pass) loss error gradient w.r.t the predictions.
        /// </summary>
        /// <param name="colTop">top output blob vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///   -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///      This blob's diff will simply contain the loss_weight * the loss stored during the forward pass.
        /// </param>
        /// <param name="rgbPropagateDown">see Layer::Backward.</param>
        /// <param name="colBottom">bottom input blob vector (length 1-2)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the predictions @f$ x @f$; backward propagates loss previously calculated
        ///     in the forward pass.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            // mutliply the loss by the loss weight (in top[0].diff)
            if (m_bEnableLoss)
                m_cuda.mul(colTop[0].count(), colTop[0].gpu_data, colTop[0].gpu_diff, colTop[0].mutable_gpu_diff);

            double dfTopDiff = convertD(colTop[0].GetDiff(0)); // loss weight
            double dfNormalizer = get_normalizer(m_normalization, -1);
            double dfLossWeight = dfTopDiff / dfNormalizer;

            // Apply the loss weight to the bottom diffs.
            if (dfLossWeight != 1.0)
            {
                for (int i = 0; i < colBottom.Count; i++)
                {
                    m_cuda.scal(colBottom[i].count(), convert(dfLossWeight), colBottom[i].mutable_gpu_diff);
                }
            }
        }
    }

    /// <summary>
    /// The MemoryLossLayerGetLossArgs class is passed to the OnGetLoss event.
    /// </summary>
    public class MemoryLossLayerGetLossArgs<T> : EventArgs
    {
        object m_userState = null;
        double m_dfLoss = 0;
        double m_dfNormalizer = 1;
        BlobCollection<T> m_colBottom;
        bool m_bEnableLossUpdate = true;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="colBottom">Specifes the bottom inputs to the forward pass.</param>
        /// <param name="userState">Specifies a user-state.</param>
        /// <param name="dfNormalizer">Specifies the normalizer value.</param>
        public MemoryLossLayerGetLossArgs(BlobCollection<T> colBottom, object userState, double dfNormalizer)
        {
            m_userState = userState;
            m_colBottom = colBottom;
        }

        /// <summary>
        /// Specifies a user-state.
        /// </summary>
        /// <remarks>The user-state is set via the 'user_state' property on the MemoryLossLayer.</remarks>
        public object user_state
        {
            get { return m_userState; }
        }

        /// <summary>
        /// Specifies the bottom passed in during the forward pass.
        /// </summary>
        public BlobCollection<T> Bottom
        {
            get { return m_colBottom; }
        }

        /// <summary>
        /// Specifies the normalizer.
        /// </summary>
        public double Normalizer
        {
            get { return m_dfNormalizer; }
        }

        /// <summary>
        /// Get/set the externally calculated total loss.
        /// </summary>
        public double Loss
        {
            get { return m_dfLoss; }
            set { m_dfLoss = value; }
        }

        /// <summary>
        /// Get/set enabling the loss update within the backpropagation pass.
        /// </summary>
        public bool EnableLossUpdate
        {
            get { return m_bEnableLossUpdate; }
            set { m_bEnableLossUpdate = value; }
        }
    }
}
