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
    /// The BatchReindexLayer provides an index into the input blob along its first axis.
    /// </summary>
    /// <remarks>
    /// This layer can be used to select, reorder, and even replicate examples in a
    /// batch.  The second blob is cast to int and treated as an index into the 
    /// first axis of the first Blob.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class BatchReindexLayer<T> : Layer<T>
    {
        Blob<T> m_blobCounts;
        Blob<T> m_blobBegins;
        Blob<T> m_blobTopIndexes;

        /// <summary>
        /// The BatchReindexLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type BATCHREINDEX.
        /// </param>
        public BatchReindexLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.BATCHREINDEX;
            m_blobCounts = new common.Blob<T>(cuda, log);
            m_blobCounts.Name = "bri_counts";
            m_blobBegins = new common.Blob<T>(cuda, log);
            m_blobBegins.Name = "bri_begins";
            m_blobTopIndexes = new common.Blob<T>(cuda, log);
            m_blobTopIndexes.Name = "bri_topidx";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_blobCounts != null)
            {
                m_blobCounts.Dispose();
                m_blobCounts = null;
            }

            if (m_blobBegins != null)
            {
                m_blobBegins.Dispose();
                m_blobBegins = null;
            }

            if (m_blobTopIndexes != null)
            {
                m_blobTopIndexes.Dispose();
                m_blobTopIndexes = null;
            }

            base.dispose();
        }

        /** @copydoc Layer */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobCounts);
                col.Add(m_blobBegins);
                col.Add(m_blobTopIndexes);

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of bottom (input) Blobs required: input, axis
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of top (output) Blobs required: batchreidx
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
            m_log.CHECK_EQ(1, colBottom[1].num_axes, "The bottom[1] should have num_axes = 1.");
            List<int> rgNewShape = new List<int>();
            rgNewShape.Add(colBottom[1].shape(0));

            for (int i = 1; i < colBottom[0].shape().Count; i++)
            {
                rgNewShape.Add(colBottom[0].shape(i));
            }

            colTop[0].Reshape(rgNewShape);

            List<int> rgShape = new List<int>();
            rgShape.Add(colBottom[1].count());
            m_blobTopIndexes.Reshape(rgShape);

            rgShape[0] = colBottom[0].shape(0);
            m_blobCounts.Reshape(rgShape);
            m_blobBegins.Reshape(rgShape);
        }

        private void check_batch_reindex(int nInitialNum, int nFinalNum, Blob<T> b)
        {
            T[] rgData = b.update_cpu_data();

            if (typeof(T) == typeof(double))
            {
                double[] rgidx_Data = (double[])Convert.ChangeType(rgData, typeof(double[]));
                for (int i = 0; i < nFinalNum; i++)
                {
                    m_log.CHECK_GE(rgidx_Data[i], 0, "Index specified for reindex layer was negative.");
                    m_log.CHECK_LT(rgidx_Data[i], nInitialNum, "Index specified for reindex layer was greater than batch size.");
                }
            }
            else
            {
                float[] rgidx_Data = (float[])Convert.ChangeType(rgData, typeof(float[]));
                for (int i = 0; i < nFinalNum; i++)
                {
                    m_log.CHECK_GE(rgidx_Data[i], 0, "Index specified for reindex layer was negative.");
                    m_log.CHECK_LT(rgidx_Data[i], nInitialNum, "Index specified for reindex layer was greater than batch size.");
                }
            }
        }

        /// <summary>
        /// The Forward computation.
        /// </summary>
        /// <param name="colBottom">input blob vector (length 2+)
        ///  -# @f$ (N \times ...) @f$
        ///     the inputs @f$ x_1 @f$
        ///  -# @f$ (M) @f$
        ///     the inputs @f$ x_2 @f$
        /// </param>
        /// <param name="colTop">output blob vector (length 1)
        ///  -# @f$ (M \times ...) @f$: 
        ///     the reindexed array @f$
        ///         y = x_1[x_2]
        ///     @f$
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            check_batch_reindex(colBottom[0].shape(0), colBottom[1].count(), colBottom[1]);

            int nCount = colTop[0].count();
            if (nCount == 0)
                return;

            m_cuda.batchreidx_fwd(nCount, colBottom[0].count() / colBottom[0].shape(0), colBottom[0].gpu_data, colBottom[1].gpu_data, colTop[0].mutable_gpu_data);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the reordered input.
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1),
        /// providing the error gradient with respect to the outputs
        ///     -# @f$ (M \times ...) @f$:
        ///         containing error gradients @f$ \frac{\partial E}{\partial y} @f$
        ///         with respect to concatenated outputs @f$ y @f$.</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (length 2):
        ///     -# @f$ \frac{\partial E}{\partial y} @f$ is de-indexed (summing where
        ///         required) back to the input @f$ x_1 @f$.
        ///     -# This layer cannot backprop to @f$ x_2 @f$, i.e. propagate{down[1] must be
        ///         false.</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            m_log.CHECK(!rgbPropagateDown[1], "Cannot backprop to index.");

            if (!rgbPropagateDown[0])
                return;

            List<KeyValuePair<int, int>> rgMapping = new List<KeyValuePair<int, int>>();
            T[] rgData = colBottom[1].update_cpu_data();

            if (typeof(T) == typeof(double))
            {
                double[] rgPerm = (double[])Convert.ChangeType(rgData, typeof(double[]));
                for (int i = 0; i < colBottom[1].count(); i++)
                {
                    rgMapping.Add(new KeyValuePair<int, int>((int)rgPerm[i], i));
                }
            }
            else
            {
                float[] rgPerm = (float[])Convert.ChangeType(rgData, typeof(float[]));
                for (int i = 0; i < colBottom[1].count(); i++)
                {
                    rgMapping.Add(new KeyValuePair<int, int>((int)rgPerm[i], i));
                }
            }

            rgMapping.Sort(new Comparison<KeyValuePair<int, int>>(sort));

            // Each element of the bottom diff is potentially the sum of many top diffs.
            // However, we'd like each CUDA thread to handle exactly one output.  Hence,
            // we first pre-compute a list of lists of indices that need to be summed for
            // each output.  'top_indexes' holds the data of this list of lists.  The
            // k'th element of 'begins' points to the location in 'top_indexes' where the
            // list for the k'th example begin, and the kth element of 'counts' is the
            // length of that list.

            m_blobBegins.SetData(-1);
            m_blobCounts.SetData(0);

            T[] rgTopIndexes = m_blobTopIndexes.mutable_cpu_data;
            T[] rgCounts = m_blobCounts.mutable_cpu_data;
            T[] rgBegins = m_blobBegins.mutable_cpu_data;

            if (typeof(T) == typeof(double))
            {
                double[] t_i_data = (double[])Convert.ChangeType(rgTopIndexes, typeof(double[]));
                double[] c_data = (double[])Convert.ChangeType(rgCounts, typeof(double[]));
                double[] b_data = (double[])Convert.ChangeType(rgBegins, typeof(double[]));

                for (int i = 0; i < rgMapping.Count; i++)
                {
                    t_i_data[i] = rgMapping[i].Value;

                    if (b_data[rgMapping[i].Key] == -1)
                        b_data[rgMapping[i].Key] = i;

                    c_data[rgMapping[i].Key] += 1;
                }
            }
            else
            {
                float[] t_i_data = (float[])Convert.ChangeType(rgTopIndexes, typeof(float[]));
                float[] c_data = (float[])Convert.ChangeType(rgCounts, typeof(float[]));
                float[] b_data = (float[])Convert.ChangeType(rgBegins, typeof(float[]));

                for (int i = 0; i < rgMapping.Count; i++)
                {
                    t_i_data[i] = rgMapping[i].Value;

                    if (b_data[rgMapping[i].Key] == -1)
                        b_data[rgMapping[i].Key] = i;

                    c_data[rgMapping[i].Key] += 1;
                }
            }

            m_blobTopIndexes.mutable_cpu_data = rgTopIndexes;
            m_blobCounts.mutable_cpu_data = rgCounts;
            m_blobBegins.mutable_cpu_data = rgBegins;

            int nCount = colBottom[0].count();

            m_cuda.batchreidx_bwd(nCount, colBottom[0].count() / colBottom[0].shape(0), colTop[0].gpu_diff, m_blobTopIndexes.gpu_data, m_blobBegins.gpu_data, m_blobCounts.gpu_data, colBottom[0].mutable_gpu_diff);
        }

        private int sort(KeyValuePair<int, int> a, KeyValuePair<int, int> b)
        {
            if (a.Key < b.Key)
                return -1;

            if (a.Key > b.Key)
                return 1;

            return 0;
        }
    }
}
