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
    /// The EltwiseLayer computes elementwise oeprations, such as product and sum,
    /// along multiple input blobs.
    /// This layer is initialized with the MyCaffe.param.EltwiseParameter.
    /// </summary>
    /// <remarks>
    /// @see [DeMeshNet: Blind Face Inpainting for Deep MeshFace Verification](https://arxiv.org/abs/1611.05271v1) by Shu Zhang, Ran He, and Tieniu Tan, 2016. 
    /// @see [Mixed context networks for semantic segmentation](https://arxiv.org/abs/1610.05854v1) by Haiming Sun, Di Xie, and Shiliang Pu, 2016. 
    /// @see [Why M Heads are Better than One: Training a Diverse Ensemble of Deep Networks](https://arxiv.org/abs/1511.06314v1) by Stefan Lee, Senthil Purushwalkam, Michael Cogswell, David Crandall, and Dhruv Batra, 2015.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class EltwiseLayer<T> : Layer<T>
    {
        EltwiseParameter.EltwiseOp m_op;
        List<double> m_rgdfCoeffs = new List<double>();
        Blob<T> m_blobIdx;
        Blob<T> m_blobSingleSecondary = null;
        bool m_bStableProdGrad;
        bool m_bCoeffBlob;

        /// <summary>
        /// The EltwiseLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">
        /// Provides EltwiseParameter eltwise_param with options:
        ///  - operation. The eltwise operation (e.g. product, summation, maximum).
        ///  
        ///  - coeff.  A Blob-wise coefficient for summation.
        ///  
        ///  - stable_prod_grad.  Optionally use an asymtotically slower but more stable method for computing the gradient for product operations.
        /// </param>
        public EltwiseLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.ELTWISE;
            m_blobIdx = new Blob<T>(cuda, log);
            m_blobIdx.Name = m_param.name + " idx";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobIdx.Dispose();

            if (m_blobSingleSecondary != null)
                m_blobSingleSecondary.Dispose();

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobIdx);

                return col;
            }
        }

        /// <summary>
        /// Returns the minimum required number of bottom (input) Blobs: input1, input2
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: output (result of eltwise operation in input1 and input2)
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
            m_log.CHECK(m_param.eltwise_param.coeff.Count == 0 ||
                        m_param.eltwise_param.coeff.Count == colBottom.Count, "Eltwise layer takes one coefficient per bottom blob.");

            m_log.CHECK(!((m_param.eltwise_param.operation == EltwiseParameter.EltwiseOp.PROD || m_param.eltwise_param.operation == EltwiseParameter.EltwiseOp.DIV) &&
                          m_param.eltwise_param.coeff.Count > 0), "Eltwise layer only takes coefficients for SUM and SUB operations.");

            m_op = m_param.eltwise_param.operation;
            m_bCoeffBlob = m_param.eltwise_param.coeff_blob;

            if (m_bCoeffBlob)
                m_log.CHECK(m_op == EltwiseParameter.EltwiseOp.SUM || m_op == EltwiseParameter.EltwiseOp.SUB, "coeff_blob option only implemented for the SUM and SUB operation.");

            int nCoeffSize = m_param.eltwise_param.coeff.Count;
            m_log.CHECK(nCoeffSize == 0 || (!m_bCoeffBlob && nCoeffSize == colBottom.Count)
                                        || (m_bCoeffBlob && nCoeffSize == colBottom.Count - 1), "Eltwise Layer takes one coefficient per bottom blob.");
            m_log.CHECK(m_op == EltwiseParameter.EltwiseOp.SUM || m_op == EltwiseParameter.EltwiseOp.SUB ||
                        layer_param.eltwise_param.coeff.Count == 0, "Eltwise layer only takes coefficients for SUM and SUB operations.");

            // Blob-wise coefficients for the elementwise operation.
            m_rgdfCoeffs = Utility.Create<double>(colBottom.Count, 1.0);

            int nCoeffBlobCount = (m_bCoeffBlob) ? 1 : 0;

            for (int i = 0; i < m_param.eltwise_param.coeff.Count - nCoeffBlobCount; i++)
            {
                m_rgdfCoeffs[i] = m_param.eltwise_param.coeff[i];
            }

            m_bStableProdGrad = m_param.eltwise_param.stable_prod_grad;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (m_param.eltwise_param.allow_single_batch_input)
                m_log.CHECK_EQ(colBottom.Count, 2, "Only two inputs allowed when 'allow_single_batch_input' = true.");

            if (colBottom[1].count() == 1)
                m_log.CHECK_EQ(colBottom.Count, 2, "Only two inputs allowed when colBottom[1].count() == 1.");

            for (int i = 1; i < colBottom.Count; i++)
            {
                if (m_bCoeffBlob && i == colBottom.Count - 1)
                {
                    m_log.CHECK_EQ(i, colBottom[i].shape(0), "Dimensions of coeff blob axis 0 must equal the number of bottom blobs (not including the coeff blob itself).");

                    for (int input_axis = 0, coeff_axis = 1; coeff_axis < colBottom[i].num_axes; input_axis++, coeff_axis++)
                    {
                        m_log.CHECK_EQ(colBottom[0].shape(input_axis), colBottom[i].shape(coeff_axis), "Each axis i >= 1 of the coeff blob must match the (i-1)th axis of the input.");
                    }
                }
                else
                {
                    if (colBottom.Count == 2 && colBottom[1].count() == 1)
                    {
                        if (m_blobSingleSecondary == null)
                        {
                            m_blobSingleSecondary = new Blob<T>(m_cuda, m_log);
                            m_blobSingleSecondary.ReshapeLike(colBottom[0]);

                            double dfVal = Utility.ConvertVal<T>(colBottom[i].GetData(0));
                            m_blobSingleSecondary.SetData(dfVal);
                        }
                    }
                    else
                    {
                        if (!m_param.eltwise_param.allow_single_batch_input)
                            m_log.CHECK(Utility.Compare<int>(colBottom[i].shape(), colBottom[0].shape(), false), "The bottoms should all be of the same shape.");
                        else
                        {
                            if (m_blobSingleSecondary == null)
                                m_blobSingleSecondary = new Blob<T>(m_cuda, m_log);
                            m_blobSingleSecondary.ReshapeLike(colBottom[0]);

                            m_log.CHECK_EQ(colBottom[i].num, 1, "The batch for the second input must be 1.");
                            m_log.CHECK_EQ(colBottom[i].count(1), colBottom[0].count(1), "All shapes other than the first shape must match!");
                        }
                    }
                }
            }

            colTop[0].ReshapeLike(colBottom[0]);

            // If max operation, we will initialize the vector index part.
            if ((m_param.eltwise_param.operation == EltwiseParameter.EltwiseOp.MAX || m_param.eltwise_param.operation == EltwiseParameter.EltwiseOp.MIN) && colTop.Count == 1)
                m_blobIdx.Reshape(colBottom[0].shape());
        }

        /// <summary>
        /// The Forward computation.
        /// </summary>
        /// <param name="colBottom">input blob vector (length 1-2)</param>
        /// <param name="colTop">output blob vector (length 1)</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            Blob<T> blob = (m_blobSingleSecondary != null) ? m_blobSingleSecondary : colBottom[1];
            long hMask = 0;
            int nCount = colTop[0].count();
            long hTopData = colTop[0].mutable_gpu_data;
            long hCoeffData = 0;
            int nCoeffCount = 0;

            if (m_param.eltwise_param.allow_single_batch_input)
            {
                // Copy each colBottom[1] to each batch item in blobSingleSecondary.
                m_cuda.channel_copyall(blob.count(),
                                       blob.num,
                                       blob.channels,
                                       blob.count(2),
                                       colBottom[1].gpu_data,
                                       blob.mutable_gpu_data);
            }

            switch (m_op)
            {
                case EltwiseParameter.EltwiseOp.PROD:
                    m_cuda.mul(nCount, colBottom[0].gpu_data, blob.gpu_data, hTopData);

                    for (int i = 2; i < colBottom.Count; i++)
                    {
                        m_cuda.mul(nCount, hTopData, colBottom[i].gpu_data, hTopData);
                    }
                    break;

                case EltwiseParameter.EltwiseOp.DIV:
                    m_cuda.div(nCount, colBottom[0].gpu_data, blob.gpu_data, hTopData);

                    for (int i = 2; i < colBottom.Count; i++)
                    {
                        m_cuda.div(nCount, hTopData, colBottom[i].gpu_data, hTopData);
                    }
                    break;

                case EltwiseParameter.EltwiseOp.SUM:
                    if (m_bCoeffBlob)
                    {
                        int nNum = colTop[0].num;
                        int nDim = nCount / nNum;
                        hCoeffData = colBottom[colBottom.Count - 1].gpu_data;
                        nCoeffCount = 1;

                        for (int i = 0; i < colBottom.Count - nCoeffCount; i++)
                        {
                            long hBottomData = (i == 0 || colBottom.Count > 3) ? colBottom[i].gpu_data : blob.gpu_data;
                            m_cuda.coeff_sum_fwd(nCount, nDim, i * nNum, m_rgdfCoeffs[i], hCoeffData, hBottomData, hTopData);
                        }
                    }
                    else
                    {
                        m_cuda.set(nCount, hTopData, 0);
                        // TODO(shelhamer) does cuBLAS optimize to sum of coeff = 1?
                        for (int i = 0; i < colBottom.Count; i++)
                        {
                            long hBottomData = (i == 0 || colBottom.Count > 2) ? colBottom[i].gpu_data : blob.gpu_data;
                            m_cuda.axpy(nCount, m_rgdfCoeffs[i], hBottomData, hTopData);
                        }
                    }
                    break;

                case EltwiseParameter.EltwiseOp.SUB:
                    if (m_bCoeffBlob)
                    {
                        int nNum = colTop[0].num;
                        int nDim = nCount / nNum;
                        hCoeffData = colBottom[colBottom.Count - 1].gpu_data;
                        nCoeffCount = 1;

                        for (int i = 0; i < colBottom.Count - nCoeffCount; i++)
                        {
                            long hBottomData = (i == 0 || colBottom.Count > 3) ? colBottom[i].gpu_data : blob.gpu_data;
                            m_cuda.coeff_sub_fwd(nCount, nDim, i * nNum, m_rgdfCoeffs[i], hCoeffData, hBottomData, hTopData);
                        }
                    }
                    else
                    {
                        m_cuda.scale(nCount, m_rgdfCoeffs[0], colBottom[0].gpu_data, hTopData);

                        for (int i = 1; i < colBottom.Count; i++)
                        {
                            long hBottomData = (i == 0 || colBottom.Count > 2) ? colBottom[i].gpu_data : blob.gpu_data;
                            m_cuda.axpy(nCount, -1 * m_rgdfCoeffs[i], hBottomData, hTopData);
                        }
                    }
                    break;

                case EltwiseParameter.EltwiseOp.MAX:
                    hMask = m_blobIdx.mutable_gpu_data;
                    m_cuda.max_fwd(nCount, colBottom[0].gpu_data, colBottom[1].gpu_data, 0, hTopData, hMask);

                    for (int i = 2; i < colBottom.Count; i++)
                    {
                        m_cuda.max_fwd(nCount, hTopData, colBottom[i].gpu_data, i-1, hTopData, hMask);
                    }
                    break;

                case EltwiseParameter.EltwiseOp.MIN:
                    hMask = m_blobIdx.mutable_gpu_data;
                    m_cuda.min_fwd(nCount, colBottom[0].gpu_data, colBottom[1].gpu_data, 0, hTopData, hMask);

                    for (int i = 2; i < colBottom.Count; i++)
                    {
                        m_cuda.min_fwd(nCount, hTopData, colBottom[i].gpu_data, i - 1, hTopData, hMask);
                    }
                    break;

                default:
                    m_log.FAIL("Unknown elementwise operation.");
                    break;
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the input.
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (length 1-2).</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            long hMask = 0;
            int nCount = colTop[0].count();
            long hTopData = colTop[0].gpu_data;
            long hTopDiff = colTop[0].gpu_diff;
            long hCoeffData = 0;
            int nNum = colTop[0].num;
            int nDim = nCount / nNum;

            if (m_bCoeffBlob)
                hCoeffData = colBottom[colBottom.Count - 1].gpu_data;

            for (int i = 0; i < colBottom.Count; i++)
            {
                if (rgbPropagateDown[i])
                {
                    long hBottomData = colBottom[i].gpu_data;
                    long hBottomDiff = colBottom[i].mutable_gpu_diff;

                    if (i == 1 && m_blobSingleSecondary != null)
                    {
                        hBottomData = m_blobSingleSecondary.gpu_data;
                        hBottomDiff = m_blobSingleSecondary.mutable_gpu_diff;
                    }

                    switch (m_op)
                    {
                        case EltwiseParameter.EltwiseOp.PROD:
                            if (m_bStableProdGrad)
                            {
                                bool bInitialized = false;
                                for (int j = 0; j < colBottom.Count; j++)
                                {
                                    if (i == j)
                                        continue;

                                    if (!bInitialized)
                                    {
                                        m_cuda.copy(nCount, colBottom[j].gpu_data, hBottomDiff);
                                        bInitialized = true;
                                    }
                                    else
                                    {
                                        m_cuda.mul(nCount, colBottom[j].gpu_data, hBottomDiff, hBottomDiff);
                                    }
                                }
                            }
                            else
                            {
                                m_cuda.div(nCount, hTopData, hBottomData, hBottomDiff);
                            }
                            m_cuda.mul(nCount, hBottomDiff, hTopDiff, hBottomDiff);
                            break;

                        case EltwiseParameter.EltwiseOp.DIV:
                            m_cuda.mul(nCount, hTopData, hBottomData, hBottomDiff);
                            m_cuda.mul(nCount, hBottomDiff, hTopDiff, hBottomDiff);
                            break;

                        case EltwiseParameter.EltwiseOp.SUM:
                            if (m_bCoeffBlob)
                            {
                                m_cuda.coeff_sum_bwd(nCount, nDim, i * nNum, m_rgdfCoeffs[i], hCoeffData, hTopDiff, hBottomDiff);
                            }
                            else
                            {
                                if (m_rgdfCoeffs[i] == 1.0)
                                    m_cuda.copy(nCount, hTopDiff, hBottomDiff);
                                else
                                    m_cuda.scale(nCount, m_rgdfCoeffs[i], hTopDiff, hBottomDiff);
                            }
                            break;

                        case EltwiseParameter.EltwiseOp.SUB:
                            if (m_bCoeffBlob)
                            {
                                m_cuda.coeff_sub_bwd(nCount, nDim, i * nNum, m_rgdfCoeffs[i], hCoeffData, hTopDiff, hBottomDiff);
                            }
                            else
                            {
                                double dfScale = (i == 0) ? 1 : -1;
                                m_cuda.scale(nCount, dfScale * m_rgdfCoeffs[i], hTopDiff, hBottomDiff);
                            }
                            break;

                        case EltwiseParameter.EltwiseOp.MAX:
                            hMask = m_blobIdx.gpu_data;
                            m_cuda.max_bwd(nCount, hTopDiff, i, hMask, hBottomDiff);
                            break;

                        case EltwiseParameter.EltwiseOp.MIN:
                            hMask = m_blobIdx.gpu_data;
                            m_cuda.min_bwd(nCount, hTopDiff, i, hMask, hBottomDiff);
                            break;

                        default:
                            m_log.FAIL("Unknown elementwise operation.");
                            break;
                    }
                }
            }

            // sum the gradients across channels.
            if (m_param.eltwise_param.allow_single_batch_input && colBottom[1].num == 1 && m_blobSingleSecondary != null)
                m_cuda.channel_sum(nCount, 1, nNum, colTop[0].channels * colTop[0].count(2), m_blobSingleSecondary.gpu_diff, colBottom[1].mutable_gpu_diff);
        }
    }
}
