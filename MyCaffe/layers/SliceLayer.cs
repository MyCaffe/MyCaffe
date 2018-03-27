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
    /// The SliceLayer takes a blob and slices it along either the num or channel dimensions
    /// outputting multiple sliced blob results.
    /// This layer is initialized with the MyCaffe.param.SliceParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SliceLayer<T> : Layer<T>
    {
        int m_nNumSlices;
        int m_nSliceSize;
        int m_nSliceAxis;
        List<uint> m_rgSlicePoints = new List<uint>();

        /// <summary>
        /// The SliceLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type SLICE with parameter slice_param,
        /// with options:
        ///   - axis (\b optional, default = 1). The axis along wich to slice.  By default the channel axis (1) is used.
        ///   
        ///   - slice_point (\b optional). The optional slice points.
        /// </param>
        public SliceLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SLICE;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: slice
        /// </summary>
        public override int MinTopBlobs
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
            m_rgSlicePoints = Utility.Clone<uint>(m_param.slice_param.slice_point);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNumAxes = colBottom[0].num_axes;

            if (m_param.slice_param.slice_dim > 0)
            {
                m_nSliceAxis = (int)m_param.slice_param.slice_dim;
                // Don't allow negative indexing for slice_dim a (uint) -- almost
                // certainly unintended.
                m_log.CHECK_GE(m_nSliceAxis, 0, "casting slice_dim from uint to int produced a negative result; slice_dim must satisfy 0 <= slice_dim < " + Blob<T>.MAX_BLOB_AXES.ToString());
                m_log.CHECK_LT(m_nSliceAxis, nNumAxes, "slice_dim is out of range.");
            }
            else
            {
                m_nSliceAxis = colBottom[0].CanonicalAxisIndex(m_param.slice_param.axis);
            }

            List<int> rgTopShape = Utility.Clone<int>(colBottom[0].shape());
            int bottom_slice_axis = colBottom[0].shape(m_nSliceAxis);

            m_nNumSlices = colBottom[0].count(0, m_nSliceAxis);
            m_nSliceSize = colBottom[0].count(m_nSliceAxis + 1);

            int nCount = 0;

            if (m_rgSlicePoints.Count != 0)
            {
                m_log.CHECK_EQ(m_rgSlicePoints.Count, colTop.Count - 1, "The slice point count is incorrect.");
                m_log.CHECK_LE(colTop.Count, bottom_slice_axis, "slice axis: " + bottom_slice_axis.ToString() + ", bottom[0] shape: '" + colBottom[0].shape_string + "'");

                int nPrev = 0;
                List<int> rgSlices = new List<int>();

                for (int i = 0; i < m_rgSlicePoints.Count; i++)
                {
                    m_log.CHECK_GT((int)m_rgSlicePoints[i], nPrev, "The slice point at " + i.ToString() + " should be greater than the previous slice point of " + nPrev.ToString());
                    rgSlices.Add((int)m_rgSlicePoints[i] - nPrev);
                    nPrev = (int)m_rgSlicePoints[i];
                }

                rgSlices.Add(bottom_slice_axis - nPrev);

                for (int i = 0; i < colTop.Count; i++)
                {
                    rgTopShape[m_nSliceAxis] = rgSlices[i];
                    colTop[i].Reshape(rgTopShape);
                    nCount += colTop[i].count();
                }
            }
            else
            {
                m_log.CHECK_EQ(bottom_slice_axis % colTop.Count, 0, "Number of top blobs (" + colTop.Count.ToString() + ") should evenly divide input slice axis (" + bottom_slice_axis.ToString() + ")");
                rgTopShape[m_nSliceAxis] = bottom_slice_axis / colTop.Count;

                for (int i = 0; i < colTop.Count; i++)
                {
                    colTop[i].Reshape(rgTopShape);
                    nCount += colTop[i].count();
                }
            }

            m_log.CHECK_EQ(nCount, colBottom[0].count(), "The count (" + nCount.ToString() + ") should be the same as the bottom count (" + colBottom[0].count().ToString() + ")");

            if (colTop.Count == 1)
            {
                colTop[0].ShareData(colBottom[0]);
                colTop[0].ShareDiff(colBottom[0]);
            }
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1+)
        ///  -# @f$ (N \times C \times H \times W) @f$ 
        ///     the first slice.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (colTop.Count == 1)
                return;

            int nOffsetSliceAxis = 0;
            long hBottomData = colBottom[0].gpu_data;
            int nBottomSliceAxis = colBottom[0].shape(m_nSliceAxis);

            for (int i = 0; i < colTop.Count; i++)
            {
                long hTopData = colTop[i].mutable_gpu_data;
                int nTopSliceAxis = colTop[i].shape(m_nSliceAxis);
                int nTopSliceSize = nTopSliceAxis * m_nSliceSize;
                int nCount = nTopSliceSize * m_nNumSlices;

                m_cuda.slice_fwd(nCount, hBottomData, m_nNumSlices, m_nSliceSize, nBottomSliceAxis, nTopSliceAxis, nOffsetSliceAxis, hTopData);
                nOffsetSliceAxis += nTopSliceAxis;
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1+), providing the error gradient
        /// with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0] || colTop.Count == 1)
                return;

            int nOffsetSliceAxis = 0;
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int nBottomSliceAxis = colBottom[0].shape(m_nSliceAxis);

            for (int i = 0; i < colTop.Count; i++)
            {
                long hTopDiff = colTop[i].gpu_diff;
                int nTopSliceAxis = colTop[i].shape(m_nSliceAxis);
                int nTopSliceSize = nTopSliceAxis * m_nSliceSize;
                int nCount = nTopSliceSize * m_nNumSlices;

                m_cuda.slice_bwd(nCount, hTopDiff, m_nNumSlices, m_nSliceSize, nBottomSliceAxis, nTopSliceAxis, nOffsetSliceAxis, hBottomDiff);
                nOffsetSliceAxis += nTopSliceAxis;
            }
        }
    }
}
