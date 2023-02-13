using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.beta;

namespace MyCaffe.layers.beta
{
    /// <summary>
    /// The TransposeLayer performs a permute and transpose operation similar to numpy.transpose.
    /// </summary>
    /// <remarks>
    /// @see [GitHub: senlinuc/caffe_ocr](https://github.com/senlinuc/caffe_ocr)
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class TransposeLayer<T> : Layer<T>
    {
        Blob<T> m_blobBottomCounts;
        Blob<T> m_blobTopCounts;
        Blob<T> m_blobForwardMap;
        Blob<T> m_blobBackwardMap;
        Blob<T> m_blobBuffer;
        bool m_bForceReshape = false;

        /// <summary>
        /// The TransposeLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">provides FlattenParameter flatten_param
        /// </param>
        public TransposeLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.TRANSPOSE;

            m_blobBottomCounts = new Blob<T>(cuda, log, false);
            m_blobBottomCounts.Name = m_param.name + " bottom counts";
            m_blobTopCounts = new Blob<T>(cuda, log, false);
            m_blobTopCounts.Name = m_param.name + " top counts";
            m_blobForwardMap = new Blob<T>(cuda, log, false);
            m_blobForwardMap.Name = m_param.name + " forward map";
            m_blobBackwardMap = new Blob<T>(cuda, log, false);
            m_blobBackwardMap.Name = m_param.name + " backward map";
            m_blobBuffer = new Blob<T>(cuda, log, false);
            m_blobBuffer.Name = m_param.name + " buffer";

            setup_internal_blobs(m_colInternalBlobs);
        }

        /// <summary>
        ///  Release any resources used.
        /// </summary>
        protected override void dispose()
        {
            dispose(ref m_blobBottomCounts);
            dispose(ref m_blobTopCounts);
            dispose(ref m_blobForwardMap);
            dispose(ref m_blobBackwardMap);
            dispose(ref m_blobBuffer);
            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobBottomCounts);
            col.Add(m_blobTopCounts);
            col.Add(m_blobForwardMap);
            col.Add(m_blobBackwardMap);
            col.Add(m_blobBuffer);
        }        

        private List<int> permute(List<int> rg)
        {
            List<int> rgNew = new List<int>();

            m_log.CHECK_EQ(rg.Count, m_param.transpose_param.dim.Count, "The index array must be the same size as the transpose_param.dim array.");

            for (int i = 0; i < rg.Count; i++)
            {
                int nAxis = m_param.transpose_param.dim[i];
                int nDim = rg[nAxis];
                rgNew.Add(nDim);
            }

            return rgNew;
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
            m_log.CHECK(colBottom[0] != colTop[0], "The Transpose layer does not support in-place computation.");
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_log.CHECK_GT(colBottom[0].shape().Count, 0, "The dimension of the transposed blob should be greater than zero.");
            m_log.CHECK_LE(colBottom[0].shape().Count, Blob<T>.MAX_BLOB_AXES, "The dimension of the transposed blob should be less than " + Blob<T>.MAX_BLOB_AXES.ToString() + ".");
            m_log.CHECK_EQ(colBottom[0].shape().Count, m_param.transpose_param.dim.Count, "The dimension of the bottom blob must equal the number of dimensions in the transpose parameter.");

            List<int> rgTopShape = permute(colBottom[0].shape());
            colTop[0].Reshape(rgTopShape);

            int nNumAxes = m_param.transpose_param.dim.Count;
            List<int> rgShape = new List<int>();

            rgShape.Add(nNumAxes);

            shareLayerBlob(m_blobBottomCounts, rgShape);
            m_blobBottomCounts.Reshape(rgShape);
            shareLayerBlob(m_blobTopCounts, rgShape);
            m_blobTopCounts.Reshape(rgShape);

            List<float> rgBottomCounts = new List<float>();
            List<float> rgTopCounts = new List<float>();

            for (int i = 1; i < nNumAxes; i++)
            {
                rgBottomCounts.Add(colBottom[0].count(i));
                rgTopCounts.Add(colTop[0].count(i));
            }

            rgBottomCounts.Add(1);
            rgTopCounts.Add(1);

            m_blobBottomCounts.mutable_cpu_data = convert(rgBottomCounts.ToArray());
            m_blobTopCounts.mutable_cpu_data = convert(rgTopCounts.ToArray());

            shareLayerBlob(m_blobForwardMap, rgShape);
            m_blobForwardMap.Reshape(rgShape);
            shareLayerBlob(m_blobBackwardMap, rgShape);
            m_blobBackwardMap.Reshape(rgShape);

            List<float> rgForwardMap = new List<float>();
            List<float> rgBackwardMap = Utility.Create<float>(nNumAxes, 0);

            for (int i = 0; i < nNumAxes; i++)
            {
                int nDim = m_param.transpose_param.dim[i];
                rgForwardMap.Add(nDim);
                rgBackwardMap[nDim] = i;
            }

            m_blobForwardMap.mutable_cpu_data = convert(rgForwardMap.ToArray());
            m_blobBackwardMap.mutable_cpu_data = convert(rgBackwardMap.ToArray());

            rgShape.Clear();
            rgShape.Add(colBottom[0].count() * nNumAxes);

            shareLayerBlob(m_blobBuffer, rgShape);
            m_blobBuffer.Reshape(rgShape);
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
            m_cuda.transpose(colBottom[0].count(), colBottom[0].gpu_data, colTop[0].mutable_gpu_data, m_blobBottomCounts.gpu_data, m_blobTopCounts.gpu_data, m_blobForwardMap.gpu_data, colBottom[0].shape().Count, m_blobBuffer.mutable_gpu_data);
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
            if (!rgbPropagateDown[0])
                return;

            m_bForceReshape = true;
            Reshape(colBottom, colTop);

            m_cuda.transpose(colBottom[0].count(), colTop[0].gpu_diff, colBottom[0].mutable_gpu_diff, m_blobTopCounts.gpu_data, m_blobBottomCounts.gpu_data, m_blobBackwardMap.gpu_data, colBottom[0].shape().Count, m_blobBuffer.mutable_gpu_data);
        }
    }
}
