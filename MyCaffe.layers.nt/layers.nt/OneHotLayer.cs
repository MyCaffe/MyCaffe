using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;

namespace MyCaffe.layers.nt
{
    /// <summary>
    /// The OneHotLayer is a layer for converting real values into a one-hot vector where a 1 is placed
    /// within the bucketized range for which the input value falls.
    /// </summary>
    /// <remarks>
    /// The min/max define the range spanning the num_outputs which defines the number of buckets.
    /// 
    /// For example, when using a min/max range of -1,1 spread across 8 vector items (num_output), inputs
    /// less than or equal to -1 go in the first bucket, inputs greater than or equal to 1 go in the last
    /// bucket and values in between -1 and 1 go into their repsective buckets (e.g input -0.12 goes into bucket
    /// index 3 and input 0.12 goes into bucket 4)
    /// 
    /// 8 inputs span across -1 to 1 range creates the following buckets:
    /// 
    /// index:        0            1            2            3           4           5           6           7 
    /// bucket: [-1.00,-0.75][-0.75,-0.50][-0.50,-0.25][-0.25, 0.00][0.00, 0.25][0.25, 0.50][0.50, 0.75][0.75, 1.00]
    /// 
    /// input: -0.75 or less set bucket #0 = 1
    /// input:  0.75 or greater set bucket #7 = 1
    /// 
    /// Except for end buckets, inputs are placed in bucket where:  bucket min &lt;= input &lt; bucket max.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class OneHotLayer<T> : Layer<T>
    {
        int m_nAxis;
        BucketCollection m_colBuckets;
        float[] m_rgOneHotVector;
        float[] m_rgTop = null;

        /// <summary>
        /// The OneHotLayer constructor
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides OneHotLayer embed_param,
        /// with OneHotLayer options:
        /// - num_output (/bdefault = 16). The number of outputs for the Layer (which defines the number of buckets in the one-hot vector output).
        /// 
        /// - axis (/bdefault = 2). The axis who's input is to be bucketized.  The count at this axis (and below) should equal 1.  
        /// 
        /// - min (/bdefault = -1.0). The minimum of the input data range to bucketize.
        /// 
        /// - max (/bdefault = 1.0). The maximum of the input data range to bucketize.
        /// </param>
        public OneHotLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.ONEHOT;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /// <summary>
        /// Returns the exact number of required bottom (intput) Blobs: input.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: onehot
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
            m_rgOneHotVector = new float[m_param.onehot_param.num_output];
            m_colBuckets = new BucketCollection(m_param.onehot_param.min, m_param.onehot_param.max, (int)m_param.onehot_param.num_output);
            m_nAxis = colBottom[0].CanonicalAxisIndex(m_param.onehot_param.axis);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colBottom[0].count(m_nAxis);
            m_log.CHECK_EQ(nCount, 1, "The bottom[0] count at axis " + m_nAxis.ToString() + " must equal 1");

            List<int> rgTopShape = Utility.Clone<int>(colBottom[0].shape());
            rgTopShape[m_nAxis] = m_colBuckets.Count;

            while (rgTopShape.Count < m_param.onehot_param.min_axes)
            {
                rgTopShape.Add(1);
            }

            colTop[0].Reshape(rgTopShape);

            int nTopCount = colTop[0].count();
            if (m_rgTop == null || m_rgTop.Length < nTopCount)
                m_rgTop = new float[nTopCount];
        }

        /// <summary>
        /// The Forward computation.
        /// </summary>
        /// <param name="colBottom">input blob vector (length 1)</param>
        /// <param name="colTop">output blob vector (length 1)</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            float[] rgBottom = convertF(colBottom[0].mutable_cpu_data);
            int nCount = colBottom[0].count(0, m_nAxis);

            for (int i = 0; i < nCount; i++)
            {
                int nIdx = m_colBuckets.Add(rgBottom[i]);

                for (int j = 0; j < m_rgOneHotVector.Length; j++)
                {
                    if (j == nIdx)
                        m_rgOneHotVector[j] = 1.0f;
                    else
                        m_rgOneHotVector[j] = 0;
                }

                Array.Copy(m_rgOneHotVector, 0, m_rgTop, i * m_rgOneHotVector.Length, m_rgOneHotVector.Length);
            }

            colTop[0].mutable_cpu_data = convert(m_rgTop);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the input.
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (length 1).</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            int nItemCount = colTop[0].count(m_nAxis);
            m_log.CHECK_EQ(nItemCount, m_colBuckets.Count, "The count at the top[axis] is incorrect!");

            int nCount1 = colTop[0].count(0, m_nAxis);
            int nCount2 = colBottom[0].count(0, m_nAxis);
            m_log.CHECK_EQ(nCount1, nCount2, "The top and bottom have incompatible sizes.");

            // Convert top one-hot vectors to softmax indexes.
            float[] rgBottomDiff = convertF(colBottom[0].mutable_cpu_diff);
            float[] rgTopData = convertF(colTop[0].mutable_cpu_data);
            float[] rgTopDiff = convertF(colTop[0].mutable_cpu_diff);

            for (int i = 0; i < nCount1; i++)
            {
                int nItemIdx = i * nItemCount;
                float fDiff = 0;
                float fDiffSum = 0;

                for (int j = 0; j < nItemCount; j++)
                {
                    fDiff = rgTopDiff[nItemIdx + j];

                    if (rgTopData[nItemIdx + j] == 0)
                        fDiff *= -1;

                    fDiffSum += fDiff;
                }

                rgBottomDiff[i] = fDiffSum / nItemCount;
            }

            colBottom[0].mutable_cpu_diff = convert(rgBottomDiff);
        }
    }
}
