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
    /// The AccuracyDecodeLayer compares the labels output by the DecodeLayer with the expected labels output
    /// by the DataLayer.
    /// This layer is initialized with the MyCaffe.param.AccuracyParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class AccuracyDecodeLayer<T> : Layer<T>
    {
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="cuda">Cuda engine.</param>
        /// <param name="log">General log.</param>
        /// <param name="p">provides AccuracyParameter accuracy_param,
        /// with AccuracyDecodeLayer. 
        ///  - top_k (optional, default 1)
        ///          Not used.
        /// </param>
        public AccuracyDecodeLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.ACCURACY_DECODE;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /// <summary>
        /// Returns the number of bottom blobs used: predicted, label
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the number of top blobs: accuracy
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
            m_log.CHECK_EQ((int)m_param.accuracy_param.top_k, 1, "Accuracy Encoding Layer only supports a topk = 1.");
            m_log.CHECK_EQ((int)m_param.accuracy_param.axis, 1, "Accuracy Encoding Layer expects axis to = 1.");

            if (m_param.accuracy_param.ignore_label.HasValue)
                m_log.WriteLine("WARNING: The Accuracy Encoding Layer does not use the 'ignore_label' parameter.");
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            List<int> rgTopShape = new List<int>(); // Accuracy is a scalar; 0 axes.
            colTop[0].Reshape(rgTopShape);
            colTop[0].type = BLOB_TYPE.ACCURACY;
        }

        /// <summary>
        /// Forward compuation.
        /// </summary>
        /// <param name="colBottom">bottom input blob (length 2)
        ///  -# @f$ (N \times C \times 1 \times 1) @f$
        ///     the distance predictions @f$ x @f$, a blob with the minimum index a value in
        ///     @f$ l_n \in [0, 1, 2, ..., K-1] @f$
        ///     indicating the predicted class label among the @f$ K @f$ classes.
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the labels l, an integer-valued blob with values
        ///     @f$ l_n \in [0, 1, 2, ..., K-1] @f$
        ///     indicating the correct class label among the @f$ K @f$ classes.
        /// </param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (1 \times 1 \times 1 \times 1) @f$
        ///     the computed accuracy each calculated by finding the label with the minimum
        ///     distance to each encoding.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_EQ(colBottom[0].num, colBottom[1].num, "The bottom[0] and bottom[1] must have the same num.");

            int nNum = colBottom[0].num;
            int nClasses = colBottom[0].channels;
            int nLabelDim = colBottom[1].channels;
            float[] rgDecode = convertF(colBottom[0].update_cpu_data()); // num of items where each item contains a distance for each class.
            float[] rgLabel = convertF(colBottom[1].update_cpu_data());  // num of labels where labels are in pairs of label dim (we only use the first label).
            int nCorrectCount = 0;

            for (int i = 0; i < nNum; i++)
            {
                int nExpectedLabel = (int)rgLabel[i * nLabelDim];
                int nActualLabel = -1;
                float fMin = float.MaxValue;

                for (int j = 0; j < nClasses; j++)
                {
                    float fDist = rgDecode[i * nClasses + j];
                    if (fDist < fMin)
                    {
                        fMin = fDist;
                        nActualLabel = j;
                    }
                }

                if (nActualLabel == nExpectedLabel)
                    nCorrectCount++;
            }

            double dfAccuracy = (double)nCorrectCount / (double)nNum;

            colTop[0].SetData(dfAccuracy, 0);
            colTop[0].Tag = m_param.accuracy_param.top_k;
        }

        /// @brief Not implemented -- AccuracyDecodeLayer cannot be used as a loss.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0])
                throw new NotImplementedException();
        }
    }
}
