using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.fillers;
using MyCaffe.param;
using MyCaffe.param.ssd;

namespace MyCaffe.layers.ssd
{
    /// <summary>
    /// The PermuteLayer performs permutation on the input blob by changing the memory order of the data which is used by the SSD algorithm.
    /// This layer is initialized with the MyCaffe.param.PermuteParameter.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class PermuteLayer<T> : Layer<T>
    {
        Blob<T> m_blobPermuteOrder;
        Blob<T> m_blobOldSteps;
        Blob<T> m_blobNewSteps;
        int m_nNumAxes = 0;
        bool m_bNeedPermute = false;

        /// <summary>
        /// The PermuteLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type PERMUTE with parameter permute_param,
        /// with options:
        ///   - order Specifies the order of the permuations.
        /// </param>
        public PermuteLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.PERMUTE;

            m_blobPermuteOrder = new Blob<T>(cuda, log);
            m_blobPermuteOrder.Name = m_param.name + " order";
            m_blobNewSteps = new Blob<T>(cuda, log);
            m_blobNewSteps.Name = m_param.name + " new-steps";
            m_blobOldSteps = new Blob<T>(cuda, log);
            m_blobOldSteps.Name = m_param.name + " old-steps";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_blobPermuteOrder != null)
            {
                m_blobPermuteOrder.Dispose();
                m_blobPermuteOrder = null;
            }

            if (m_blobNewSteps != null)
            {
                m_blobNewSteps.Dispose();
                m_blobNewSteps = null;
            }

            if (m_blobOldSteps != null)
            {
                m_blobOldSteps.Dispose();
                m_blobOldSteps = null;
            }

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobPermuteOrder);
                col.Add(m_blobNewSteps);
                col.Add(m_blobOldSteps);

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: data
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: permute
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
            m_log.CHECK_EQ(colBottom.Count, 1, "There should only be one botom blob.");
            PermuteParameter permute_param = layer_param.permute_param;

            m_nNumAxes = colBottom[0].num_axes;

            // Push the specified new orders
            List<int> rgOrders = new List<int>();
            foreach (int nOrder in permute_param.order)
            {
                m_log.CHECK_LT(nOrder, m_nNumAxes, "The order should be less than the input dimension '" + m_nNumAxes.ToString() + "'!");

                if (rgOrders.Contains(nOrder))
                    m_log.FAIL("The order '" + nOrder.ToString() + "' is a duplicate order!");

                rgOrders.Add(nOrder);
            }

            // Push the rest of the orders and save the orginal step sizes for each axis.
            for (int i = 0; i < m_nNumAxes; i++)
            {
                if (!rgOrders.Contains(i))
                    rgOrders.Add(i);
            }

            m_log.CHECK_EQ(rgOrders.Count, m_nNumAxes, "The order count should be the same as the input dimension of '" + m_nNumAxes.ToString() + "'!");

            // Check if we need to reorder the data or keep it.
            m_bNeedPermute = false;

            for (int i = 0; i < m_nNumAxes; i++)
            {
                if (rgOrders[i] != i)
                {
                    // As long as there is one order which is different from the natural order
                    // of the data, we need to permute.  Otherwise, we share the data and diff.
                    m_bNeedPermute = true;
                    break;
                }
            }

            List<int> rgTopShape = Utility.Create<int>(m_nNumAxes, 1);

            m_blobPermuteOrder.Reshape(m_nNumAxes, 1, 1, 1);
            m_blobOldSteps.ReshapeLike(m_blobPermuteOrder);
            m_blobNewSteps.ReshapeLike(m_blobPermuteOrder);

            T[] rgOrder1 = new T[m_nNumAxes];
            for (int i = 0; i < m_nNumAxes; i++)
            {
                int nOrder = rgOrders[i];
                rgOrder1[i] = Utility.ConvertVal<T>(nOrder);
                int nShape = colBottom[0].shape(nOrder);
                rgTopShape[i] = nShape;
            }

            m_blobPermuteOrder.mutable_cpu_data = rgOrder1;
            colTop[0].Reshape(rgTopShape);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            T[] rgOldSteps = new T[m_nNumAxes];
            T[] rgNewSteps = new T[m_nNumAxes];
            T[] rgOrder1 = m_blobPermuteOrder.mutable_cpu_data;
            List<int> rgOrder = new List<int>();

            for (int i = 0; i < rgOrder1.Length; i++)
            {
                if (i < m_nNumAxes)
                {
                    if (i == m_nNumAxes - 1)
                        rgOldSteps[i] = m_tOne;
                    else
                        rgOldSteps[i] = Utility.ConvertVal<T>(colBottom[0].count(i + 1));
                }

                rgOrder.Add((int)Utility.ConvertVal<T>(rgOrder1[i]));
            }

            m_blobOldSteps.mutable_cpu_data = rgOldSteps;
            List<int> rgTopShape = PermuteParameter.Reshape(rgOrder, colBottom[0].shape(), colBottom[0].num_axes);
            colTop[0].Reshape(rgTopShape);

            for (int i = 0; i < m_nNumAxes; i++)
            {
                if (i == m_nNumAxes - 1)
                    rgNewSteps[i] = m_tOne;
                else
                    rgNewSteps[i] = Utility.ConvertVal<T>(colTop[0].count(i + 1));
            }

            m_blobNewSteps.mutable_cpu_data = rgNewSteps;
        }

        /// <summary>
        /// Computes the forward calculation.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the inputs.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times C \times H \times W) @f$ the outputs.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_bNeedPermute)
            {
                long hBottomData = colBottom[0].mutable_gpu_data;
                long hTopData = colTop[0].mutable_gpu_data;
                int nCount = colTop[0].count();
                long hPermuteOrder = m_blobPermuteOrder.gpu_data;
                long hNewSteps = m_blobNewSteps.gpu_data;
                long hOldSteps = m_blobOldSteps.gpu_data;
                bool bForward = true;

                m_cuda.permute(nCount, hBottomData, bForward, hPermuteOrder, hOldSteps, hNewSteps, m_nNumAxes, hTopData);
            }
            else
            {
                colTop[0].ShareData(colBottom[0]);
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 1)
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (m_bNeedPermute)
            {
                long hTopDiff = colTop[0].mutable_gpu_diff;
                long hBottomDiff = colBottom[0].mutable_gpu_diff;
                int nCount = colTop[0].count();
                long hPermuteOrder = m_blobPermuteOrder.gpu_data;
                long hNewSteps = m_blobNewSteps.gpu_data;
                long hOldSteps = m_blobOldSteps.gpu_data;
                bool bForward = false;

                m_cuda.permute(nCount, hBottomDiff, bForward, hPermuteOrder, hOldSteps, hNewSteps, m_nNumAxes, hTopDiff);
            }
            else
            {
                colBottom[0].ShareDiff(colTop[0]);
            }
        }
    }
}
