using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.fillers;

namespace MyCaffe.layers
{
    /// <summary>
    /// The BiasLayer computes a sum of two input Blobs, with the shape of the latter Blob 
    /// 'broadcast' to match the shape of the former. Equivalent to tiling 
    /// the latter Blob, then computing the elementwise sum.
    /// This layer is initialized with the MyCaffe.param.BiasParameter.
    /// </summary>
    /// <remarks>
    /// The second input may be omitted, in which case it's learned as a parameter
    /// of the layer.  Note: in case bias and scaling are desired, both operations can
    /// be handled by 'ScaleLayer' configured with 'bias_term: true'.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class BiasLayer<T> : Layer<T>
    {
        Blob<T> m_blobBiasMultiplier;
        int m_nOuterDim;
        int m_nBiasDim;
        int m_nInnerDim;
        int m_nDim;

        /// <summary>
        /// The BiasLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type BIAS, with bias_param.
        /// </param>
        public BiasLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.BIAS;
            m_blobBiasMultiplier = new Blob<T>(cuda, log);
            m_blobBiasMultiplier.Name = "bias_biasmult";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            m_blobBiasMultiplier.Dispose();
            base.dispose();
        }

        /** @copydoc Layer::setup_internal_blobs */
        protected override void setup_internal_blobs(BlobCollection<T> col)
        {
            if (col.Count > 0)
                return;

            col.Add(m_blobBiasMultiplier);
        }

        /// <summary>
        /// Returns the minimum number of required bottom (input) Blobs: input1
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of required bottom (input) Blobs: input1, input2
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: bias
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Re-initialize the parameters of the layer.
        /// </summary>
        /// <param name="target">Specifies the weights to target (e.g. weights, bias or both).</param>
        /// <returns>When handled, this method returns <i>true</i>, otherwise <i>false</i>.</returns>
        public override bool ReInitializeParameters(WEIGHT_TARGET target)
        {
            base.ReInitializeParameters(target);

            if (target == WEIGHT_TARGET.BOTH || target == WEIGHT_TARGET.BIAS)
            {
                FillerParameter fp = m_param.bias_param.filler;
                if (fp == null)
                    fp = new FillerParameter("constant", 0.0);

                Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                filler.Fill(m_colBlobs[0]);
            }

            return true;
        }

        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (colBottom.Count == 1 && m_colBlobs.Count > 0)
            {
                m_log.WriteLine("Skipping parameter initialization.");
            }
            else if (colBottom.Count == 1)
            {
                // bias is a learned parameter; initialize it.
                BiasParameter p = m_param.bias_param;
                int nAxis = colBottom[0].CanonicalAxisIndex(p.axis);
                int nNumAxes = p.num_axes;

                m_log.CHECK_GE(nNumAxes, -1, "num_axes must be non-negative, or -1 to extend to end of bottom[0].");

                if (nNumAxes >= 0)
                    m_log.CHECK_GE(colBottom[0].num_axes, nAxis + nNumAxes, "bias blob's shape extends past bottom[0]'s shape when applied starting with bottom[0] axis = " + nAxis.ToString());

                m_colBlobs = new BlobCollection<T>();

                List<int> rgBiasShape = new List<int>();
                int nStart = nAxis;
                int nEnd = (nNumAxes == -1) ? colBottom[0].shape().Count : nStart + nNumAxes;

                for (int i = nStart; i < nEnd; i++)
                {
                    rgBiasShape.Add(colBottom[0].shape(i));
                }

                Blob<T> blobBias = new Blob<T>(m_cuda, m_log);
                blobBias.Name = m_param.name + " bias";
                blobBias.type = BLOB_TYPE.INTERNAL;
                blobBias.type = BLOB_TYPE.WEIGHT;

                if (!shareParameter(blobBias, rgBiasShape))
                {
                    blobBias.Reshape(rgBiasShape);
                    FillerParameter fp = p.filler;
                    if (fp == null)
                        fp = new FillerParameter("constant", 0.0);

                    Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                    filler.Fill(blobBias);
                }
                m_colBlobs.Add(blobBias);
            }

            m_rgbParamPropagateDown = new DictionaryMap<bool>(m_colBlobs.Count, true);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            BiasParameter p = m_param.bias_param;
            Blob<T> blobBias = (colBottom.Count > 1) ? colBottom[1] : m_colBlobs[0];

            // Always set axis == 0 in special case where bias is a scalar
            // (num_axes == 0.  Mathematically eqiuvalent for any choice of axis, os the
            // actual setting can be safely ignored; and computation is most efficient
            // with axis == 0 and (therefore) outer_dim == 1.
            int nAxis = (blobBias.num_axes == 0) ? 0 : colBottom[0].CanonicalAxisIndex(p.axis);

            m_log.CHECK_GE(colBottom[0].num_axes, nAxis + blobBias.num_axes, "bias blob's shape extends past bottom[0]'s shape when applied starting with bottom[0] axis = " + nAxis.ToString());

            for (int i = 0; i < blobBias.num_axes; i++)
            {
                m_log.CHECK_EQ(colBottom[0].shape(nAxis + i), blobBias.shape(i), "dimension mismatch between bottom[0]->shape(" + (nAxis + i).ToString() + ") and bias->shape(" + i.ToString() + ")");
            }

            m_nOuterDim = colBottom[0].count(0, nAxis);
            m_nBiasDim = blobBias.count();
            m_nInnerDim = colBottom[0].count(nAxis + blobBias.num_axes);
            m_nDim = m_nBiasDim * m_nInnerDim;

            if (colBottom[0] != colTop[0])
                colTop[0].ReshapeLike(colBottom[0]);

            m_blobBiasMultiplier.Reshape(new List<int>() { m_nInnerDim });
            m_blobBiasMultiplier.SetData(1.0);
        }

        /// <summary>
        /// The Forward computation.
        /// </summary>
        /// <param name="colBottom">input blob vector (length 1-2)</param>
        /// <param name="colTop">output blob vector (length 1)</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nCount = colTop[0].count();
            long hBottomData = colBottom[0].gpu_data;
            long hBiasData = ((colBottom.Count > 1) ? colBottom[1].gpu_data : m_colBlobs[0].gpu_data);
            long hTopData = colTop[0].mutable_gpu_data;

            m_cuda.bias_fwd(nCount, hBottomData, hBiasData, m_nBiasDim, m_nInnerDim, hTopData);
        }

        /// <summary>
        /// Computes the error gradient w.r.t. the input.
        /// </summary>
        /// <param name="colTop">top output Blob vector (length 1).</param>
        /// <param name="rgbPropagateDown">see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (length 1-2).</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (rgbPropagateDown[0] && colBottom[0] != colTop[0])
            {
                long hTopDiff = colTop[0].gpu_diff;
                long hBottomDiff = colBottom[0].mutable_gpu_diff;
                int nCount = colBottom[0].count();

                m_cuda.copy(nCount, hTopDiff, hBottomDiff);
            }

            // in-place, we don't need to do anyting with the data diff.
            bool bBiasParam = (colBottom.Count == 1) ? true : false;

            if ((!bBiasParam && rgbPropagateDown[1]) || (bBiasParam && m_rgbParamPropagateDown[0]))
            {
                long hTopDiff = colTop[0].gpu_diff;
                long hBiasDiff = (bBiasParam) ? m_colBlobs[0].mutable_gpu_diff : colBottom[1].mutable_gpu_diff;
                double dfAccum = (bBiasParam) ? 1.0 : 0.0;
                int nTopDiffOffset = 0;

                for (int n = 0; n < m_nOuterDim; n++)
                {
                    m_cuda.gemv(false, m_nBiasDim, m_nInnerDim, m_tOne, hTopDiff, m_blobBiasMultiplier.gpu_data, convert(dfAccum), hBiasDiff, nTopDiffOffset);
                    nTopDiffOffset += m_nDim;
                    dfAccum = 1.0;
                }
            }
        }
    }
}
