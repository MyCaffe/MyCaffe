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
    /// TripletSelect Layer - this layer selects the semi hard samples and places them into
    ///     Top[0] - anchors
    ///     Top[1] - positives
    ///     Top[2] - negatives
    ///     
    /// The bottom[0] num must be a factory of 3 and its contents must contain the
    /// n/3 anchors followed by n/3 positives followed by n/3 negatives.
    /// 
    /// This ordering is filled by the TripletData layer.
    /// </summary>
    /// <remarks>
    /// * Oringinal implementation in Python at TripletDataLayer/TripletSelectionLayer/TripletSelectionLayer by luhaofang/tripletloss on github (https://github.com/luhaofang/tripletloss) 
    /// 
    /// @see [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737v2) by Alexander Hermans, Lucas Beyer, and Bastian Leibe, 2017. 
    /// @see [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) by Florian Schroff, and Dmitry Kalenichenko, and James Philbin, 2015.
    /// @see [Generalisation and Sharing in Triplet Convnets for Sketch based Visual Search](https://arxiv.org/abs/1611.05301v1) by Tu Bui, Leonardo Ribeiro, Moacir Ponti, and John Collomosse, 2016.
    /// </remarks> 
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TripletSelectLayer<T> : Layer<T>
    {
        int m_nTripletBatchSize = 0;
        int m_nVectorDim = 0;
        Blob<T> m_blobAP;
        Blob<T> m_blobAN;
        List<Tuple<int, int, int>> m_rgTripletList = new List<Tuple<int, int, int>>();
        List<int> m_rgNoResidualList = new List<int>();

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides TripletLossParameter triplet_loss_param</param>
        public TripletSelectLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.TRIPLET_SELECT;
            m_blobAP = new common.Blob<T>(m_cuda, m_log);
            m_blobAP.Name = "ap";
            m_blobAN = new common.Blob<T>(m_cuda, m_log);
            m_blobAN.Name = "an";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobAP.Dispose();
            m_blobAN.Dispose();
            base.dispose();
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
        /// Returns the triplet tuples which each contain an anchor, positive and negative.
        /// </summary>
        public List<Tuple<int, int, int>> triplets
        {
            get { return m_rgTripletList; }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: data
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }   // data
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: anchor, pos, neg
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 3; }   // anchors, positives, negatives
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(0, colBottom[0].num % 3, "The batch size must be a factor of 3.");
            m_nTripletBatchSize = colBottom[0].num / 3;
            m_nVectorDim = colBottom[0].count() / colBottom[0].num;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // Setup the TripletSelect layer.
            List<int> rgShape = Utility.Clone<int>(colBottom[0].shape());
            rgShape[0] = m_nTripletBatchSize;

            colTop[0].Reshape(rgShape);
            colTop[1].Reshape(rgShape);
            colTop[2].Reshape(rgShape);
            m_blobAP.Reshape(rgShape);
            m_blobAN.Reshape(rgShape);
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 3)
        ///  -# @f$ (N/3 \times C \times H \times W) @f$ 
        ///     the anchors. 
        ///  -# @f$ (N/3 \times C \times H \times W) @f$ 
        ///     the positives. 
        ///  -# @f$ (N/3 \times C \times H \times W) @f$ 
        ///     the negtives. 
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            List<KeyValuePair<int, double>> rgAps = new List<KeyValuePair<int, double>>();
            List<KeyValuePair<int, double>> rgAns = new List<KeyValuePair<int, double>>();


            // Find AP distances, ordered by largest first.
            m_cuda.sub(m_blobAP.count(), colBottom[0].gpu_data, colBottom[0].gpu_data, m_blobAP.mutable_gpu_data, 0, m_nTripletBatchSize * m_nVectorDim);

            for (int i=0; i<m_nTripletBatchSize; i++)
            {
                double dfDot = convertD(m_cuda.dot(m_nVectorDim, m_blobAP.gpu_data, m_blobAP.gpu_data, i * m_nVectorDim, i * m_nVectorDim));
                rgAps.Add(new KeyValuePair<int, double>(i + m_nTripletBatchSize, dfDot));
            }

            rgAps = rgAps.OrderByDescending(p => p.Value).ToList();


            // Find AN distances, ordered by smallest first.
            m_cuda.sub(m_blobAN.count(), colBottom[0].gpu_data, colBottom[0].gpu_data, m_blobAN.mutable_gpu_data, 0, m_nTripletBatchSize * 2 * m_nVectorDim);

            for (int i = 0; i < m_nTripletBatchSize; i++)
            {
                double dfDot = convertD(m_cuda.dot(m_nVectorDim, m_blobAN.gpu_data, m_blobAN.gpu_data, i * m_nVectorDim, i * m_nVectorDim));
                rgAns.Add(new KeyValuePair<int, double>(i + 2 * m_nTripletBatchSize, dfDot));
            }

            rgAns = rgAns.OrderBy(p => p.Value).ToList();


            // Copy the anchors.
            m_cuda.copy(colTop[0].count(), colBottom[0].gpu_data, colTop[0].mutable_gpu_data);


            // Copy the positive and negatives in their respetive orderings
            int nOffsetSrc = 0;
            int nOffsetDst = 0;

            m_rgTripletList = new List<Tuple<int, int, int>>();
            m_rgNoResidualList = new List<int>();

            for (int i = 0; i < m_nTripletBatchSize; i++)
            {
                nOffsetDst = i * m_nVectorDim;

                // Copy the positives.
                nOffsetSrc = rgAps[i].Key * m_nVectorDim;
                m_cuda.copy(m_nVectorDim, colBottom[0].gpu_data, colTop[1].mutable_gpu_data, nOffsetSrc, nOffsetDst);

                // Copy the negatives.
                nOffsetSrc = rgAns[i].Key * m_nVectorDim;
                m_cuda.copy(m_nVectorDim, colBottom[0].gpu_data, colTop[2].mutable_gpu_data, nOffsetSrc, nOffsetDst);

                if (rgAps[i].Value >= rgAns[i].Value)
                    m_rgNoResidualList.Add(i);

                m_rgTripletList.Add(new Tuple<int, int, int>(i, rgAps[i].Key, rgAns[i].Key));
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs by copying the top diffs to the bottom.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 3), providing the error gradient
        /// with respect to computed outputs.
        ///  -# @f$ (N/3 \times C \times H \times W) @f$ 
        ///     the anchor diffs.
        ///  -# @f$ (N/3 \times C \times H \times W) @f$ 
        ///     the positive diffs.
        ///  -# @f$ (N/3 \times C \times H \times W) @f$ 
        ///     the negative diffs.
        /// </param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the inputs.
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            for (int i = 0; i < m_rgTripletList.Count; i++)
            {
                if (!m_rgNoResidualList.Contains(i))
                {
                    m_cuda.copy(m_nVectorDim, colTop[0].gpu_diff, colBottom[0].mutable_gpu_diff, i * m_nVectorDim, m_rgTripletList[i].Item1 * m_nVectorDim);
                    m_cuda.copy(m_nVectorDim, colTop[1].gpu_diff, colBottom[0].mutable_gpu_diff, i * m_nVectorDim, m_rgTripletList[i].Item2 * m_nVectorDim);
                    m_cuda.copy(m_nVectorDim, colTop[2].gpu_diff, colBottom[0].mutable_gpu_diff, i * m_nVectorDim, m_rgTripletList[i].Item3 * m_nVectorDim);
                }
                else
                {
                    m_cuda.set(m_nVectorDim, colBottom[0].mutable_gpu_data, m_tZero, -1, m_rgTripletList[i].Item1 * m_nVectorDim);
                    m_cuda.set(m_nVectorDim, colBottom[0].mutable_gpu_data, m_tZero, -1, m_rgTripletList[i].Item2 * m_nVectorDim);
                    m_cuda.set(m_nVectorDim, colBottom[0].mutable_gpu_data, m_tZero, -1, m_rgTripletList[i].Item3 * m_nVectorDim);
                }
            }
        }
    }
}
