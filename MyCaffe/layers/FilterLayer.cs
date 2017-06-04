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
    /// The FilterLayer takes two+ Blobs, interprets last Blob as a selector and
    /// filters remaining Blobs accordingly with selector data (0 means that
    /// the corresponding item has to be filtered, non-zero means that corresponding
    /// item needs to stay).
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class FilterLayer<T> : Layer<T>
    {
        bool m_bFirstShape;
        List<int> m_rgIndicesToForward = new List<int>();

        /// <summary>
        /// The FilterLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides FilterParameter filter_param
        /// </param>
        public FilterLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.FILTER;
        }

        /// <summary>
        /// Returns the minimum number of required bottom (intput) Blobs: input, selector
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the minimum number of required top (output) Blobs: filter
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
            m_log.CHECK_EQ(colTop.Count, colBottom.Count - 1, "The top size must be equal to the bottom size - 1.");
            m_bFirstShape = true;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            // bottom[0...k-1] are the blobs to filter
            // bottom[last] is the 'selector_blob'
            int nSelectorIdx = colBottom.Count - 1;
            for (int i = 1; i < colBottom[nSelectorIdx].num_axes; i++)
            {
                m_log.CHECK_EQ(colBottom[nSelectorIdx].shape(i), 1, "Selector blob dimensions must be singletons (1), except the first");
            }

            for (int i = 0; i < colBottom.Count - 1; i++)
            {
                m_log.CHECK_EQ(colBottom[nSelectorIdx].shape(0), colBottom[i].shape(0), "Each bottom should have the same 0th dimension as the selector blob.");
            }

            T[] rgBottomDataSelector = colBottom[nSelectorIdx].update_cpu_data();
            m_rgIndicesToForward = new List<int>();

            // look for non-zero elements in bottom[0].  Items of each bottom that
            // have the same index as the items in bottom[0] with value == non-zero
            // will be forwarded.
            if (typeof(T) == typeof(double))
            {
                double[] rgBottomDataSelectorD = (double[])Convert.ChangeType(rgBottomDataSelector, typeof(double[]));

                for (int i = 0; i < colBottom[nSelectorIdx].shape(0); i++)
                {
                    if (rgBottomDataSelectorD[i] != 0.0)
                        m_rgIndicesToForward.Add(i);
                }
            }
            else
            {
                float[] rgBottomDataSelectorF = (float[])Convert.ChangeType(rgBottomDataSelector, typeof(float[]));

                for (int i = 0; i < colBottom[nSelectorIdx].shape(0); i++)
                {
                    if (rgBottomDataSelectorF[i] != 0.0)
                        m_rgIndicesToForward.Add(i);
                }
            }

            // only filtered items will be forwarded
            int nNewTopsNum = m_rgIndicesToForward.Count;

            // init
            if (m_bFirstShape)
            {
                nNewTopsNum = colBottom[0].shape(0);
                m_bFirstShape = false;
            }

            for (int t = 0; t < colTop.Count; t++)
            {
                int nNumAxes = colBottom[t].num_axes;
                List<int> rgShape = new List<int>();
                rgShape.Add(nNewTopsNum);

                for (int ts = 1; ts < nNumAxes; ts++)
                {
                    rgShape.Add(colBottom[t].shape(ts));
                }

                colTop[t].Reshape(rgShape);
            }
        }

        /// <summary>
        /// Forward computation.
        /// </summary>
        /// <param name="colBottom">bottom input blob vector (length 2+)
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs to be filtered @f$ x_1 @f$
        ///  -# ...
        ///  -# @f$ (N \times C \times H \times W) @f$
        ///     the inputs to be filtered @f$ x_K @f$
        ///  -# @f$ (N \times 1 \times 1 \times 1) @f$
        ///     the selector blob</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (S \times C \times H \times W) @f$ 
        ///     the filtered output @f$ x_1 @f$ where @f$ S @f$ is the number of
        ///     items that haven't been filtered.
        ///  -# @f$ (S \times C \times H \times W) @f$
        ///     the filtered output @f$ x_K @f$ where @f$ S @f$ is the number of 
        ///     items that haven't been filtered
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nNewTopsNum = m_rgIndicesToForward.Count;

            // forward all filtered items for all bottoms but the Selector (bottom[last])
            for (int t = 0; t < colTop.Count; t++)
            {
                long hBottomData = colBottom[t].gpu_data;
                long hTopData = colTop[t].mutable_gpu_data;
                int nDim = colBottom[t].count() / colBottom[t].shape(0);

                for (int n = 0; n < nNewTopsNum; n++)
                {
                    int nDataOffsetTop = n * nDim;
                    int nDataOffsetBottom = m_rgIndicesToForward[n] * nDim;
                    m_cuda.copy(nDim, hBottomData, hTopData, nDataOffsetBottom, nDataOffsetTop);
                }
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the forwarded inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vecotr (length 1+), 
        /// providing the error gradient with respect to the outputs.</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">input Blob vecotor (length 2+), into which the top error
        /// gradient is copied.</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[colBottom.Count - 1])
                m_log.FAIL("Layer cannot backpropagate to filter index inputs.");

            for (int i = 0; i < colTop.Count; i++)
            {
                // bottom[last] is the selector and never needs backpropagation
                // so we can iterate over top vecotr because top.size() == bottom.size() - 1

                if (rgbPropagateDown[i])
                {
                    int nDim = colTop[i].count() / colTop[i].shape(0);
                    int nNextToBackwardOffset = 0;
                    int nBatchOffset = 0;
                    int nDataOffsetBottom = 0;
                    int nDataOffsetTop = 0;

                    for (int n = 0; n < colBottom[i].shape(0); n++)
                    {
                        if (nNextToBackwardOffset >= m_rgIndicesToForward.Count)
                        {
                            // we already visited all items that have been forwarded, so
                            // just set to zero remaining ones.
                            nDataOffsetBottom = n * nDim;
                            m_cuda.set(nDim, colBottom[i].mutable_gpu_diff, m_tZero, -1, nDataOffsetBottom);
                        }
                        else
                        {
                            nBatchOffset = m_rgIndicesToForward[nNextToBackwardOffset];
                            nDataOffsetBottom = n * nDim;

                            if (n != nBatchOffset) // this data has not been forwarded
                            {
                                m_cuda.set(nDim, colBottom[i].mutable_gpu_diff, m_tZero, -1, nDataOffsetBottom);
                            }
                            else // this data has been forwarded
                            {
                                nDataOffsetTop = nNextToBackwardOffset * nDim;
                                nNextToBackwardOffset++;    // point to next forwarded item index
                                m_cuda.copy(nDim, colTop[i].mutable_gpu_diff, colBottom[i].mutable_gpu_diff, nDataOffsetTop, nDataOffsetBottom);
                            }
                        }
                    }
                }
            }    
        }
    }
}
