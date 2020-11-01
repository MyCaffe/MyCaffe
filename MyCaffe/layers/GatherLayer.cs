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
    /// The GatherLayer extracts (gathers) data from specified indices along a given axis
    /// from the input and returns it as the output.  The indices are passed in as the 
    /// second bottom blob.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class GatherLayer<T> : Layer<T>
    {
        int m_nAxis = 0;
        int m_nM = 0;
        int m_nN = 0;
        int m_nDim = 0;
        int m_nDimAtAxis = 0;

        /// <summary>
        /// The GatherLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides FlattenParameter flatten_param
        /// </param>
        public GatherLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.GATHER;
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input.
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
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
            m_nAxis = colBottom[0].CanonicalAxisIndex(m_param.gather_param.axis);

            if (m_nAxis != 0 && m_nAxis != 1)
                m_log.FAIL("Currently only axis = 0 or axis = 1 are supported.");

            m_nDim = colBottom[0].count(m_nAxis + 1);
            m_nDimAtAxis = colBottom[0].shape()[m_nAxis];
            m_nM = colBottom[0].count(0, m_nAxis);
            m_nN = colBottom[1].count();
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            float[] rgIdx = convertF(colBottom[1].mutable_cpu_data);

            m_log.CHECK_EQ(m_nN, rgIdx.Length, "N should equal the number of indices.");
            for (int i=0; i<m_nN; i++)
            {
                int nIdx = (int)rgIdx[i];
                if (nIdx < -m_nDimAtAxis || nIdx > m_nDimAtAxis)
                    m_log.FAIL("The index at idx=" + i.ToString() + " is out of range!  Must be within range [-" + m_nDimAtAxis.ToString() + "," + m_nDimAtAxis.ToString() + "]");
            }

            List<int> rgShape = new List<int>(colBottom[1].shape());
            int nLen = rgShape.Count;

            while (rgShape.Count > 0 && rgShape[rgShape.Count - 1] == 1)
            {
                rgShape.RemoveAt(rgShape.Count - 1);
            }

            if (m_nAxis == 0)
                rgShape.Add(m_nDim);
            else if (m_nAxis == 1)
                rgShape.Insert(0, m_nM);

            for (int i = rgShape.Count; i < nLen; i++)
            {
                rgShape.Add(1);
            }

            colTop[0].Reshape(rgShape);
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
            int nCount = colBottom[0].count();
            long hBottom = colBottom[0].gpu_data;
            long hIdx = colBottom[1].gpu_data;
            long hTop = colTop[0].mutable_gpu_data;
            int nExpectedCount = (m_nAxis == 0) ? (m_nN * m_nDim) : (m_nN * m_nM);

            m_log.CHECK_EQ(colTop[0].count(), nExpectedCount, "The top count should equal " + nExpectedCount.ToString() + "!");

            m_cuda.gather_fwd(nCount, hBottom, hTop, m_nAxis, m_nDim, m_nDimAtAxis, m_nM, m_nN, hIdx);
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
            int nCount = colBottom[0].count();
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            long hIdx = colBottom[1].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;

            m_cuda.gather_bwd(nCount, hTopDiff, hBottomDiff, m_nAxis, m_nDim, m_nDimAtAxis, m_nM, m_nN, hIdx);
        }
    }
}
