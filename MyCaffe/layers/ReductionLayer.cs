using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;

namespace MyCaffe.layers
{
    /// <summary>
    /// The ReductionLayer computes the 'reductions' -- operations that return a scalar output Blob
    /// for an input Blob of arbitrary size, such as the sum, absolute sum,
    /// and sum of squares.
    /// This layer is initialized with the MyCaffe.param.ReductionParameter.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ReductionLayer<T> : Layer<T>
    {
        /// <summary>
        /// the reduction operation performed by the layer.
        /// </summary>
        ReductionParameter.ReductionOp m_op;
        /// <summary>
        /// a scalar coefficient applied to all outputs.
        /// </summary>
        double m_dfCoeff;
        /// <summary>
        /// the index of the first input axis to reduce.
        /// </summary>
        int m_nAxis;
        /// <summary>
        /// the number of reductions performed.
        /// </summary>
        int m_nNum;
        /// <summary>
        /// the input size of each reduction.
        /// </summary>
        int m_nDim;
        /// <summary>
        /// a helper Blob used for summation (op_ == SUM)
        /// </summary>
        Blob<T> m_blobSumMultiplier;

        /// <summary>
        /// The ReductionLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type REDUCTION with parameter reduction_param, 
        /// with options:
        ///   - operation. The operation (SUM, ASUM, SUMSQ or MEAN) to run.
        /// 
        ///   - axis (\b optional, default = 0). The first axis to reduce to scalar.
        ///   
        ///   - coeff (\b optional, default = 1).  The coefficient used to scale the output.
        /// </param>
        public ReductionLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.REDUCTION;
            m_blobSumMultiplier = new Blob<T>(cuda, log);
            m_blobSumMultiplier.Name = m_param.name + " summult";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();

            if (m_blobSumMultiplier != null)
            {
                m_blobSumMultiplier.Dispose();
                m_blobSumMultiplier = null;
            }
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobSumMultiplier);

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: input
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: reduction
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
            m_op = m_param.reduction_param.operation;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            m_nAxis = colBottom[0].CanonicalAxisIndex(m_param.reduction_param.axis);

            // In the output, we'll keep all axes up to the reduction axis, but
            //  throw away all after than.
            // Note: currently reducing along non-tail axes is not supported; otherwise,
            //  we'd need to also copy any axes followign an 'end_axis'.
            List<int> rgTopShape = Utility.Clone<int>(colBottom[0].shape(), m_nAxis);
            colTop[0].Reshape(rgTopShape);
            m_nNum = colBottom[0].count(0, m_nAxis);
            m_nDim = colBottom[0].count(m_nAxis);
            m_log.CHECK_EQ(m_nNum, colTop[0].count(), "The 'num' should equal the top[0].count!");

            if (m_op == ReductionParameter.ReductionOp.SUM ||
                m_op == ReductionParameter.ReductionOp.MEAN)
            {
                List<int> rgSumMultShape = new List<int>() { m_nDim };
                m_blobSumMultiplier.Reshape(rgSumMultShape);
                m_blobSumMultiplier.SetData(1);
            }

            m_dfCoeff = m_param.reduction_param.coeff;

            if (m_op == ReductionParameter.ReductionOp.MEAN)
                m_dfCoeff /= m_nDim;
        }

        /// <summary>
        /// Forward operation
        /// </summary>
        /// <param name="colBottom">
        /// bottom input Blob<T> vector (length 1)
        ///     -# @f$ (N \times C \times ...) @f$
        ///     the inputs @f$ x @f$
        /// </param>
        /// <param name="colTop">
        /// top output Blob<T> vector (length 1)
        ///     -# (Shape depends on <i>axis</i> parameter setting)
        ///     the computed outputs.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            long hBottomData = colBottom[0].gpu_data;
            long hMultData = 0;

            if (m_blobSumMultiplier.count() > 0)
                hMultData = m_blobSumMultiplier.gpu_data;

            T[] rgTop = colTop[0].mutable_cpu_data;
            int nOffset = 0;

            for (int i = 0; i < m_nNum; i++)
            {
                switch (m_op)
                {
                    case ReductionParameter.ReductionOp.SUM:
                    case ReductionParameter.ReductionOp.MEAN:
                        rgTop[i] = m_cuda.dot(m_nDim, hMultData, hBottomData, 0, nOffset);
                        break;

                    case ReductionParameter.ReductionOp.ASUM:
                        rgTop[i] = m_cuda.asum(m_nDim, hBottomData, nOffset);
                        break;

                    case ReductionParameter.ReductionOp.SUMSQ:
                        rgTop[i] = m_cuda.dot(m_nDim, hBottomData, hBottomData, nOffset, nOffset);
                        break;

                    default:
                        m_log.FAIL("Unknown reduction op: " + m_op.ToString());
                        break;
                }

                nOffset += m_nDim;
            }

            colTop[0].mutable_cpu_data = rgTop;

            if (m_dfCoeff != 1.0)
                m_cuda.scal(m_nNum, m_dfCoeff, colTop[0].mutable_gpu_data);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the Reduction inputs.
        /// </summary>
        /// <param name="colTop">
        /// top output Blob<T> vector (length 1), providing the error gradient with
        /// respect to the outputs.
        ///     -# (Shape depends on <i>axis</i> parameter setting) 
        /// </param>
        /// <param name="rgbPropagateDown">
        /// see Layer::backward.
        /// </param>
        /// <param name="colBottom">
        /// bottom input Blob<T> vector (length 1)
        ///     -# @f$ (N \times C \times ...) @f$
        ///     the inputs. 
        /// </param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (!rgbPropagateDown[0])
                return;

            // Get bottom_data, if needed.
            long hBottomData = 0;

            if (m_op == ReductionParameter.ReductionOp.ASUM ||
                m_op == ReductionParameter.ReductionOp.SUMSQ)
                hBottomData = colBottom[0].gpu_data;

            T[] rgTopDiff = colTop[0].update_cpu_diff();
            long hBottomDiff = colBottom[0].mutable_gpu_diff;
            int nOffset = 0;

            for (int i = 0; i < m_nNum; i++)
            {
                double dfBottomCoeff = convertD(rgTopDiff[i]) * m_dfCoeff;

                switch (m_op)
                {
                    case ReductionParameter.ReductionOp.SUM:
                    case ReductionParameter.ReductionOp.MEAN:
                        m_cuda.set(m_nDim, hBottomDiff, convert(dfBottomCoeff), -1, nOffset);
                        break;

                    case ReductionParameter.ReductionOp.ASUM:
                        m_cuda.sign(m_nDim, hBottomData, hBottomDiff, nOffset, nOffset);
                        m_cuda.scal(m_nDim, dfBottomCoeff, hBottomDiff, nOffset);
                        break;

                    case ReductionParameter.ReductionOp.SUMSQ:
                        m_cuda.scale(m_nDim, convert(2 * dfBottomCoeff), hBottomData, hBottomDiff, nOffset, nOffset);
                        break;

                    default:
                        m_log.FAIL("Unknown reduction op: " + m_op.ToString());
                        break;
                }

                nOffset += m_nDim;
            }
        }
    }
}
