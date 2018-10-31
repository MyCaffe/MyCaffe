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
    /// The ScaleLayer computes the elementwise product of two input Blobs, with the sanpe of
    /// the latter Blob 'broadcast' to match the shape of the former.
    /// Equivalent to tiling the later Blob, then computing the elementwise
    /// product.  Note: for efficiency and convienience this layer can
    /// additionally perform a 'broadcast' sum too when 'bias_term: true'
    /// This layer is initialized with the MyCaffe.param.ScaleParameter.
    /// is set.
    /// </summary>
    /// <remarks>
    /// The latter, scale input may be omitted, in which case it's learned as 
    /// parameter of the layer (as in the bias, if it is included).
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ScaleLayer<T> : Layer<T>
    {
        BiasLayer<T> m_biasLayer = null;
        BlobCollection<T> m_colBiasBottomVec = new BlobCollection<T>();
        List<bool> m_rgbBiasPropagateDown = new List<bool>();
        int m_nBiasParamId;
        Blob<T> m_blobSumMultiplier;
        Blob<T> m_blobSumResult;
        Blob<T> m_blobTemp;
        int m_nAxis;
        int m_nOuterDim;
        int m_nScaleDim;
        int m_nInnerDim;

        /// <summary>
        /// The ScaleLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type SCALE with parameter scale_param,
        /// with options:
        ///   - bias_term (/b optional, default = false).  TWhether to also learn a bias (equivalent to a ScalarLayer + BiasLayer, but may be more efficient).
        ///   
        ///   - bias_filler (/b optional, default = "constant", 0.1). The filler used to initialize the bias values when <i>bias_term</i> = <i>true</i>.
        /// </param>
        public ScaleLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.SCALE;
            m_blobSumMultiplier = new Blob<T>(cuda, log);
            m_blobSumMultiplier.Name = "scale_summult";
            m_blobSumResult = new Blob<T>(cuda, log);
            m_blobSumResult.Name = "scale_sumres";
            m_blobTemp = new Blob<T>(cuda, log);
            m_blobTemp.Name = "scale_sumres";
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            if (m_biasLayer != null)
            {
                m_biasLayer.Dispose();
                m_biasLayer = null;
            }

            if (m_blobSumMultiplier != null)
            {
                m_blobSumMultiplier.Dispose();
                m_blobSumMultiplier = null;
            }

            if (m_blobSumResult != null)
            {
                m_blobSumResult.Dispose();
                m_blobSumResult = null;
            }

            if (m_blobTemp != null)
            {
                m_blobTemp.Dispose();
                m_blobTemp = null;
            }

            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                col.Add(m_blobSumMultiplier);
                col.Add(m_blobSumResult);
                col.Add(m_blobTemp);

                return col;
            }
        }

        /// <summary>
        /// Returns the minimum number of required bottom (input) Blobs: firstfactor
        /// </summary>
        public override int MinBottomBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Returns the maximum number of required bottom (input) Blobs: firstfactor, secondfactor
        /// </summary>
        public override int MaxBottomBlobs
        {
            get { return 2; }
        }

        /// <summary>
        /// Returns the exact number of required top (output) Blobs: scale
        /// </summary>
        public override int ExactNumTopBlobs
        {
            get { return 1; }
        }

        /// <summary>
        /// Re-initialize the parameters of the layer.
        /// </summary>
        /// <returns>When handled, this method returns <i>true</i>, otherwise <i>false</i>.</returns>
        public override bool ReInitializeParameters()
        {
            base.ReInitializeParameters();

            FillerParameter fp = m_param.scale_param.filler;
            if (fp == null)
                fp = new FillerParameter("constant", 1.0);

            Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
            filler.Fill(m_colBlobs[0]);

            if (m_param.scale_param.bias_term)
                m_biasLayer.ReInitializeParameters();

            return true;
        }


        /// <summary>
        /// Setup the layer.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void LayerSetUp(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            ScaleParameter p = m_param.scale_param;

            if (colBottom.Count == 1 && blobs.Count > 0)
            {
                m_log.WriteLine("Skipping parameter initialization.");
            }
            else if (colBottom.Count == 1)
            {
                // scale is a learned parameter; initialize it.
                m_nAxis = colBottom[0].CanonicalAxisIndex(p.axis);
                int nNumAxes = p.num_axes;
                m_log.CHECK_GE(nNumAxes, -1, "num_axes must be non-negative, or -1 to extend to the end of bottom[0].");

                if (nNumAxes >= 0)
                    m_log.CHECK_GE(colBottom[0].num_axes, m_nAxis + nNumAxes, "scale blob's shape extends past bottom[0]'s shape when applied starting with bottom[0] axis = " + m_nAxis.ToString());

                m_colBlobs = new BlobCollection<T>();

                List<int> rgShape = new List<int>();
                int nStart = m_nAxis;
                int nEnd = (nNumAxes == -1) ? colBottom[0].shape().Count : nStart + nNumAxes;

                for (int i = nStart; i < nEnd; i++)
                {
                    rgShape.Add(colBottom[0].shape(i));
                }

                Blob<T> blobScale = new Blob<T>(m_cuda, m_log);
                blobScale.Name = "scale";

                if (!shareParameter(blobScale, rgShape))
                {
                    blobScale.Reshape(rgShape);
                    FillerParameter fp = p.filler;

                    // Default to unit (1) filler for identity operation.
                    if (fp == null)
                        fp = new FillerParameter("constant", 1.0);

                    Filler<T> filler = Filler<T>.Create(m_cuda, m_log, fp);
                    filler.Fill(blobScale);
                }
                m_colBlobs.Add(blobScale);
            }

            if (p.bias_term)
            {
                LayerParameter pb = new LayerParameter(LayerParameter.LayerType.BIAS);
                pb.bias_param.axis = p.axis;
                pb.bias_param.num_axes = (colBottom.Count > 1) ? colBottom[1].num_axes : p.num_axes;
                pb.bias_param.filler = p.bias_filler;

                m_colBiasBottomVec = new BlobCollection<T>();
                m_colBiasBottomVec.Add(colBottom[0]);

                m_biasLayer = new BiasLayer<T>(m_cuda, m_log, pb);
                m_biasLayer.Setup(m_colBiasBottomVec, colTop);

                shareLayerBlobs(m_biasLayer);

                m_nBiasParamId = m_colBlobs.Count;
                m_colBlobs.Add(m_biasLayer.blobs[0]);
                m_rgbBiasPropagateDown = Utility.Create<bool>(1, false);
            }

            m_rgbParamPropagateDown = new DictionaryMap<bool>(m_colBlobs.Count(), true);
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            ScaleParameter p = m_param.scale_param;
            Blob<T> blobScale = (colBottom.Count > 1) ? colBottom[1] : m_colBlobs[0];

            // Always set axis == 0 in special case where bias is a scalar
            // (num_axes == 0.  Mathematically eqiuvalent for any choice of axis, so the
            // actual setting can be safely ignored; and computation is most efficient
            // with axis == 0 and (therefore) outer_dim == 1. (Setting m_nAxis to
            // bottom[0].num_axes - 1, giving inner_dim_ == 1, would be equally
            // performant.)
            m_nAxis = (blobScale.num_axes == 0) ? 0 : colBottom[0].CanonicalAxisIndex(p.axis);
            m_log.CHECK_GE(colBottom[0].num_axes, m_nAxis + blobScale.num_axes, "scale blob's shape extends past bottom[0]'s shape when applied starting with bottom[0] axis = " + m_nAxis.ToString());

            for (int i = 0; i < blobScale.num_axes; i++)
            {
                m_log.CHECK_EQ(colBottom[0].shape(m_nAxis + i), blobScale.shape(i), "dimension mismatch between bottom[0]->shape(" + (m_nAxis + i).ToString() + ") and scale->shape(" + i.ToString() + ")");
            }

            m_nOuterDim = colBottom[0].count(0, m_nAxis);
            m_nScaleDim = blobScale.count();
            m_nInnerDim = colBottom[0].count(m_nAxis + blobScale.num_axes);

            if (colBottom[0] == colTop[0])  // in-place computation
                m_blobTemp.ReshapeLike(colBottom[0]);
            else
                colTop[0].ReshapeLike(colBottom[0]);

            m_blobSumResult.Reshape(new List<int>() { m_nOuterDim * m_nScaleDim });
            int nSumMultSize = Math.Max(m_nOuterDim, m_nInnerDim);
            m_blobSumMultiplier.Reshape(new List<int>() { nSumMultSize });
            m_blobSumMultiplier.SetData(1.0);

            if (m_biasLayer != null)
            {
                m_colBiasBottomVec[0] = colTop[0];
                m_biasLayer.Reshape(m_colBiasBottomVec, colTop);
            }
        }

        /// <summary>
        /// Forward computation.
        /// </summary>
        /// <remarks>
        /// In the below shape specifications, i denotes the value of the
        /// 'axis' field given by 'this.layer_param.scale_param.axis', after
        /// canonicalization (i.e., conversion from negative to positive index,
        /// if applicable).
        /// </remarks>
        /// <param name="colBottom">bottom input blob vector (length 2)
        ///  -# @f$ (d_0 \times ... \times 
        ///      d_i \times ... d_j ... d_n) @f$
        ///      the first factor @f$ x @f$.
        ///  -# @f$ (d_i \times d_j) @f$
        ///      the second factor @f$ y @f$.</param>
        /// <param name="colTop">top output blob vector (length 1)
        ///  -# @f$ (d_0 \times ... \times 
        ///      d_i \times ... \times d_j \times ... \times d_n) @f$
        ///      the product @f$ z = x y @f$ computed after 'broadcasting' @f$ y @f$.
        ///      Equivalent to tiling @f$ y @f$ to have the same shape as @f$ x @f$
        ///      then computing the elementwise product.
        /// </param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            if (colBottom[0] == colTop[0])
            {
                // in-place computation; need to store bottom data before overwriting it.
                // Note that this is only necessary for backward; we could skip this if not
                // doing backward, but Caffe currently provides no way of knowing whether
                // we'll need to do backward at the time of the forward call.
                m_blobTemp.CopyFrom(colBottom[0]);
            }

            long hScaleData = (colBottom.Count > 1) ? colBottom[1].gpu_data : m_colBlobs[0].gpu_data;
            long hTopData = colTop[0].mutable_gpu_data;
            int nCount = colTop[0].count();
            long hBottomData = colBottom[0].gpu_data;

            if (m_biasLayer != null)
            {
                long hBiasData = m_colBlobs[m_nBiasParamId].gpu_data;
                m_cuda.scale_fwd(nCount, hBottomData, hScaleData, m_nScaleDim, m_nInnerDim, hTopData, hBiasData);
            }
            else
            {
                m_cuda.scale_fwd(nCount, hBottomData, hScaleData, m_nScaleDim, m_nInnerDim, hTopData);
            }
        }

        /// <summary>
        /// Computes the error gradient w.r.t the inputs.
        /// </summary>
        /// <param name="colTop">top output Blob vector (Length 1), providing the error gradient
        /// with respect to computed outputs.</param>
        /// <param name="rgbPropagateDown">propagate down see Layer::Backward</param>
        /// <param name="colBottom">bottom input Blob vector (Length 2)</param>
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            if (m_biasLayer != null && m_rgbParamPropagateDown[m_rgbParamPropagateDown.Count - 1])
                m_biasLayer.Backward(colTop, m_rgbBiasPropagateDown, m_colBiasBottomVec);

            bool bScaleParam = (colBottom.Count == 1) ? true : false;
            Blob<T> blobScale = (bScaleParam) ? m_colBlobs[0] : colBottom[1];

            if ((!bScaleParam && rgbPropagateDown[1]) || (bScaleParam && m_rgbParamPropagateDown[0]))
            {
                long hTopDiff = colTop[0].gpu_diff;
                bool bInPlace = (colBottom[0] == colTop[0]) ? true : false;
                long hBottomData = (bInPlace) ? m_blobTemp.gpu_data : colBottom[0].gpu_data;

                // Hack: store big eltwise product in bottom[0].diff, except in the special
                // case where this layer itself does the eltwise product, in which case we
                // can store it directly in the scale diff, and we're done.
                // If we're computing in-place (and not doing eltwise computation), this
                // hack doesn't work and we store the product in temp_.
                bool bIsEltwise = (colBottom[0].count() == blobScale.count()) ? true : false;
                long hProduct = (bIsEltwise) ? blobScale.mutable_gpu_diff : ((bInPlace) ? m_blobTemp.mutable_gpu_data : colBottom[0].mutable_gpu_diff);
                long hSumMult = m_blobSumMultiplier.gpu_data;

                m_cuda.mul(colTop[0].count(), hTopDiff, hBottomData, hProduct);

                if (!bIsEltwise)
                {
                    long hSumResult = 0;

                    if (m_nInnerDim == 1)
                    {
                        hSumResult = hProduct;
                    }
                    else if (m_blobSumResult.count() == 1)
                    {
                        double dfScaleDiff = convertD(blobScale.GetDiff(0));

                        if (bScaleParam)
                        {
                            T fDot = m_cuda.dot(m_nInnerDim, hProduct, hSumMult);
                            dfScaleDiff += convertD(fDot);
                            blobScale.SetDiff(dfScaleDiff, 0);
                        }
                        else
                        {
                            T fDot = m_cuda.dot(m_nInnerDim, hProduct, hSumMult);
                            blobScale.SetDiff(convertD(fDot), 0);
                        }
                    }
                    else
                    {
                        hSumResult = (m_nOuterDim == 1) ? blobScale.mutable_gpu_diff : m_blobSumResult.mutable_gpu_data;
                        m_cuda.gemv(false, m_blobSumResult.count(), m_nInnerDim, m_tOne, hProduct, hSumMult, m_tZero, hSumResult);
                    }

                    if (m_nOuterDim != 1)
                    {
                        if (m_nScaleDim == 1)
                        {
                            double dfScaleDiff = convertD(blobScale.GetDiff(0));

                            if (bScaleParam)
                            {
                                T fDot = m_cuda.dot(m_nOuterDim, hSumMult, hSumResult);
                                dfScaleDiff += convertD(fDot);
                                blobScale.SetDiff(dfScaleDiff, 0);
                            }
                            else
                            {
                                T fDot = m_cuda.dot(m_nOuterDim, hSumMult, hSumResult);
                                blobScale.SetDiff(convertD(fDot), 0);
                            }
                        }
                        else
                        {
                            long hScaleDiff = blobScale.mutable_gpu_diff;
                            m_cuda.gemv(true, m_nOuterDim, m_nScaleDim, m_tOne, hSumResult, hSumMult, (bScaleParam) ? m_tOne : m_tZero, hScaleDiff);
                        }
                    }
                }
            }

            if (rgbPropagateDown[0])
            {
                int nCount = colTop[0].count();
                long hTopDiff = colTop[0].gpu_diff;
                long hScaleData = blobScale.gpu_data;
                long hBottomDiff = colBottom[0].mutable_gpu_diff;

                m_cuda.scale_fwd(nCount, hTopDiff, hScaleData, m_nScaleDim, m_nInnerDim, hBottomDiff);
            }
        }
    }
}
