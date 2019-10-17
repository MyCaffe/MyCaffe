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
    /// The FlattenLayer reshapes the input Blob into flat vectors
    /// This layer is initialized with the MyCaffe.param.FlattenParameter.
    /// </summary>
    /// <remarks>
    /// Note: because this layer does not change the input values -- merely the
    /// dimensions -- it can simply copy the input.  The copy happens 'virtually'
    /// (thus taking effectively 0 real time) by setting, in Forward, the data
    /// pointer of the top Blob to that of the bottom Blob (see Blob::ShareData),
    /// and in Backward, the diff pointer to the bottom Blob to that of the top Blob
    /// (see Blob::ShareDiff)
    /// 
    /// @see [Representation Learning and Pairwise Ranking for Implicit and Explicit Feedback in Recommendation Systems](https://arxiv.org/abs/1705.00105v1) by Mikhail Trofimov, Sumit Sidana, Oleh Horodnitskii, Charlotte Laclau, Yury Maximov, and Massih-Reza Amini, 2017. 
    /// @see [Deep Neural Networks to Enable Real-time Multimessenger Astrophysics](https://arxiv.org/abs/1701.00008v2) by Daniel George, and E. A. Huerta, 2016.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class FlattenLayer<T> : Layer<T>
    {
        /// <summary>
        /// The FlattenLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides FlattenParameter flatten_param
        /// </param>
        public FlattenLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.FLATTEN;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: flatten
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
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK(colTop[0] != colBottom[0], "Layer does not allow in-place computation.");

            int nStartAxis = colBottom[0].CanonicalAxisIndex(m_param.flatten_param.axis);
            int nEndAxis = colBottom[0].CanonicalAxisIndex(m_param.flatten_param.end_axis);

            //List<int> rgTopShape = new List<int>();
            //for (int i = 0; i < nStartAxis; i++)
            //{
            //    rgTopShape.Add(colBottom[0].shape(i));
            //}

            //int nFlattenDim = colBottom[0].count(nStartAxis, nEndAxis + 1);
            //rgTopShape.Add(nFlattenDim);

            //for (int i = nEndAxis + 1; i < colBottom[0].num_axes; i++)
            //{
            //    rgTopShape.Add(colBottom[0].shape(i));
            //}

            List<int> rgTopShape = FlattenParameter.Reshape(m_param.flatten_param.axis, m_param.flatten_param.end_axis, colBottom[0].shape(), nStartAxis, nEndAxis);

            colTop[0].Reshape(rgTopShape);
            m_log.CHECK_EQ(colTop[0].count(), colBottom[0].count(), "The top[0] and bottom[0] should have the same count.");
        }

        /// <summary>
        /// Forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 2+)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.
        ///     the inputs.</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (N \times CHW \times 1 \times 1) @f$ the outputs -- i.e., the (virtually) copied, flattened inputs
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_cuda.copy(colTop[0].count(), colBottom[0].gpu_data, colTop[0].mutable_gpu_data);
//            colTop[0].ShareData(colBottom[0]);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the concatenate inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vecotr (length 1), 
        /// providing the error gradient with respect to the outputs.</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">input Blob vecotor (length @f$ k @f$), into which the top error
        /// gradient is (virtually) copied.</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            m_cuda.copy(colBottom[0].count(), colTop[0].gpu_diff, colBottom[0].mutable_gpu_diff);
//            colBottom[0].ShareDiff(colTop[0]);
        }
    }
}
