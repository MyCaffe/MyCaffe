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
    /// The PriorBoxLayer generates prior boxes of designated sizes and aspect ratios across all dimensions of @f$ (H \times W) @f$ which is used by the SSD algorithm.
    /// This layer is initialized with the MyCaffe.param.ssd.PriorBoxParameter.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class PriorBoxLayer<T> : Layer<T>
    {
        List<float> m_rgfMinSizes = new List<float>();
        List<float> m_rgfMaxSizes = new List<float>();
        List<float> m_rgfAspectRatios = new List<float>();
        bool m_bFlip;
        int m_nNumPriors;
        bool m_bClip;
        List<float> m_rgfVariance = new List<float>();
        int m_nImgW;
        int m_nImgH;
        float m_fStepW;
        float m_fStepH;
        float m_fOffset;

        /// <summary>
        /// The PriorBoxLayer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="p">Specifies the LayerParameter of type PRIORBOX with parameter prior_box_param,
        /// with options:
        ///   - min_size (\b minimum box size in pixels, can be multiple items - required!).
        ///   - max_size (\b maximum box size in pixels, can be ignored or same as the # of min_size).
        ///   - aspect_ratio (\b optional aspect ratios of the boxes, can be multiple items).
        ///   - flip (\b optional bool, default true)  If set, flip the aspect ratio.
        /// </param>
        public PriorBoxLayer(CudaDnn<T> cuda, Log log, LayerParameter p)
            : base(cuda, log, p)
        {
            m_type = LayerParameter.LayerType.PRIORBOX;
        }

        /** @copydoc Layer::dispose */
        protected override void dispose()
        {
            base.dispose();
        }

        /** @copydoc Layer::internal_blobs */
        public override BlobCollection<T> internal_blobs
        {
            get
            {
                BlobCollection<T> col = new BlobCollection<T>();

                return col;
            }
        }

        /// <summary>
        /// Returns the exact number of required bottom (input) Blobs: data
        /// </summary>
        public override int ExactNumBottomBlobs
        {
            get { return 2; }
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
            PriorBoxParameter p = m_param.prior_box_param;
            m_log.CHECK_GT(p.min_size.Count, 0, "Must provied at least one min_size!");

            for (int i = 0; i < p.min_size.Count; i++)
            {
                float fMin = p.min_size[i];
                m_log.CHECK_GT(fMin, 0, "min_size must be positive greater than zero.");
                m_rgfMinSizes.Add(fMin);
            }

            m_rgfAspectRatios.Clear();
            m_rgfAspectRatios.Add(1.0f);
            m_bFlip = p.flip;

            for (int i = 0; i < p.aspect_ratio.Count; i++)
            {
                float fAr = p.aspect_ratio[i];
                bool bAlreadyExists = false;

                for (int j = 0; j < m_rgfAspectRatios.Count; j++)
                {
                    if (Math.Abs(fAr - m_rgfAspectRatios[j]) < 1e-6f)
                    {
                        bAlreadyExists = true;
                        break;
                    }
                }

                if (!bAlreadyExists)
                {
                    m_rgfAspectRatios.Add(fAr);
                    if (m_bFlip)
                        m_rgfAspectRatios.Add(1.0f / fAr);
                }
            }

            m_nNumPriors = m_rgfAspectRatios.Count * m_rgfMinSizes.Count;

            if (p.max_size.Count > 0)
            {
                m_log.CHECK_EQ(p.min_size.Count, p.max_size.Count, "The max_size count must equal the min_size count!");
                for (int i = 0; i < p.max_size.Count; i++)
                {
                    float fMax = p.max_size[i];
                    m_log.CHECK_GT(fMax, m_rgfMinSizes[i], "The max_size must be greater than the min_size.");
                    m_rgfMaxSizes.Add(fMax);
                    m_nNumPriors++;
                }
            }

            m_bClip = p.clip;

            if (p.variance.Count > 1)
            {
                // Must and only provide 4 variance values.
                m_log.CHECK_EQ(p.variance.Count, 4, "Must only have 4 variance values.");

                for (int i = 0; i < p.variance.Count; i++)
                {
                    float fVar = p.variance[i];
                    m_log.CHECK_GT(fVar, 0, "The variance values must be greater than zero.");
                    m_rgfVariance.Add(fVar);
                }
            }
            else if (p.variance.Count == 1)
            {
                float fVar = p.variance[0];
                m_log.CHECK_GT(fVar, 0, "The variance value must be greater than zero.");
                m_rgfVariance.Add(fVar);
            }
            else
            {
                // Set default to 0.1.
                m_rgfVariance.Add(0.1f);
            }

            if (p.img_h.HasValue || p.img_w.HasValue)
            {
                m_log.CHECK(!p.img_size.HasValue, "Either img_size or img_h/img_w should be specified; but not both.");
                m_nImgH = (int)p.img_h.Value;
                m_log.CHECK_GT(m_nImgH, 0, "The img_h should be greater than 0.");
                m_nImgW = (int)p.img_w.Value;
                m_log.CHECK_GT(m_nImgW, 0, "The img_w should be greater than 0.");
            }
            else if (p.img_size.HasValue)
            {
                int nImgSize = (int)p.img_size.Value;
                m_log.CHECK_GT(nImgSize, 0, "The img_size should be greater than 0.");
                m_nImgH = nImgSize;
                m_nImgW = nImgSize;
            }
            else
            {
                m_nImgH = 0;
                m_nImgW = 0;
            }

            if (p.step_h.HasValue || p.step_w.HasValue)
            {
                m_log.CHECK(!p.step.HasValue, "Either step_size or step_h/step_w should be specified; but not both.");
                m_fStepH = p.step_h.Value;
                m_log.CHECK_GT(m_nImgH, 0, "The step_h should be greater than 0.");
                m_fStepW = p.step_w.Value;
                m_log.CHECK_GT(m_nImgW, 0, "The step_w should be greater than 0.");
            }
            else if (p.step.HasValue)
            {
                float fStep = p.step.Value;
                m_log.CHECK_GT(fStep, 0, "The step should be greater than 0.");
                m_fStepH = fStep;
                m_fStepW = fStep;
            }
            else
            {
                m_fStepH = 0;
                m_fStepW = 0;
            }

            m_fOffset = p.offset;
        }

        /// <summary>
        /// Reshape the bottom (input) and top (output) blobs.
        /// </summary>
        /// <param name="colBottom">Specifies the collection of bottom (input) Blobs.</param>
        /// <param name="colTop">Specifies the collection of top (output) Blobs.</param>
        public override void Reshape(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            List<int> rgTopShape = PriorBoxParameter.Reshape(m_param.prior_box_param, colBottom[0].width, colBottom[0].height, m_nNumPriors);

            // Since all images in a batch have the same height and width, we only need to
            // generate one set of priors which can be shared across all images.
            m_log.CHECK_EQ(rgTopShape[0], 1, "The topshape(0) should be 1.");

            // 2 channels. 
            // First channel stores the mean of each prior coordinate.
            // Second channel stores the variance of each prior coordiante.
            m_log.CHECK_EQ(rgTopShape[1], 2, "The topshape(1) should be 1.");
            m_log.CHECK_GT(rgTopShape[2], 0, "The top shape at index 2 must be greater than zero.");

            colTop[0].Reshape(rgTopShape);
        }

        /// <summary>
        /// Generates prior boxes for a layer with specified parameters.
        /// </summary>
        /// <param name="colBottom">bottom input Blob vector (Length - at least 2)
        ///  -# @f$ (N \times C \times H_i \times W_i) @f$ the input layer @f$ x_i @f$.</param>
        ///  -# @f$ (N \times C \times H_0 \times W_0) @f$ the data layer @f$ x_0 @f$.</param>
        /// <param name="colTop">top otuput Blob vector (Length 1)
        ///  -# @f$ (N \times 2 \times K*4) @f$ where @f$ K @f$ are the prior numbers.  
        ///  By default, a box of aspect ratio 1 and min_size and a box of aspect ratio 1
        ///  and sqrt(min_size * max_size) is created.</param>
        protected override void forward(BlobCollection<T> colBottom, BlobCollection<T> colTop)
        {
            int nLayerW = colBottom[0].width;
            int nLayerH = colBottom[0].height;
            int nImgW;
            int nImgH;

            if (m_nImgW == 0 || m_nImgH == 0)
            {
                nImgW = colBottom[1].width;
                nImgH = colBottom[1].height;
            }
            else
            {
                nImgW = m_nImgW;
                nImgH = m_nImgH;
            }

            float fStepW;
            float fStepH;

            if (m_fStepW == 0 || m_fStepH == 0)
            {
                fStepW = (float)nImgW / (float)nLayerW;
                fStepH = (float)nImgH / (float)nLayerH;
            }
            else
            {
                fStepW = m_fStepW;
                fStepH = m_fStepH;
            }

            float[] rgfTopData = Utility.ConvertVecF<T>(colTop[0].mutable_cpu_data);
            int nDim = nLayerH * nLayerW * m_nNumPriors * 4;
            int nIdx = 0;

            for (int h = 0; h < nLayerH; h++)
            {
                for (int w = 0; w < nLayerW; w++)
                {
                    float fCenterX = (w + m_fOffset) * fStepW;
                    float fCenterY = (h + m_fOffset) * fStepH;
                    float fBoxWidth;
                    float fBoxHeight;

                    for (int s = 0; s < m_rgfMinSizes.Count; s++)
                    {
                        int nMinSize = (int)m_rgfMinSizes[s];

                        // first prior; aspect_ratio = 1, size = min_size
                        fBoxHeight = nMinSize;
                        fBoxWidth = nMinSize;
                        // xmin
                        rgfTopData[nIdx] = (fCenterX - fBoxWidth / 2.0f) / nImgW;
                        nIdx++;
                        // ymin
                        rgfTopData[nIdx] = (fCenterY - fBoxHeight / 2.0f) / nImgH;
                        nIdx++;
                        // xmax
                        rgfTopData[nIdx] = (fCenterX + fBoxWidth / 2.0f) / nImgW;
                        nIdx++;
                        // ymax
                        rgfTopData[nIdx] = (fCenterY + fBoxHeight / 2.0f) / nImgH;
                        nIdx++;

                        if (m_rgfMaxSizes.Count > 0)
                        {
                            m_log.CHECK_EQ(m_rgfMinSizes.Count, m_rgfMaxSizes.Count, "The max_sizes and min_sizes must have the same count.");
                            int nMaxSize = (int)m_rgfMaxSizes[s];

                            // second prior; aspect_ratio = 1, size = sqrt(min_size * max_size)
                            fBoxWidth = (float)Math.Sqrt(nMinSize * nMaxSize);
                            fBoxHeight = fBoxWidth;
                            // xmin
                            rgfTopData[nIdx] = (fCenterX - fBoxWidth / 2.0f) / nImgW;
                            nIdx++;
                            // ymin
                            rgfTopData[nIdx] = (fCenterY - fBoxHeight / 2.0f) / nImgH;
                            nIdx++;
                            // xmax
                            rgfTopData[nIdx] = (fCenterX + fBoxWidth / 2.0f) / nImgW;
                            nIdx++;
                            // ymax
                            rgfTopData[nIdx] = (fCenterY + fBoxHeight / 2.0f) / nImgH;
                            nIdx++;
                        }

                        // rest of priors
                        for (int r = 0; r < m_rgfAspectRatios.Count; r++)
                        {
                            float fAr = m_rgfAspectRatios[r];

                            if (Math.Abs(fAr - 1.0f) < 1e-6f)
                                continue;

                            fBoxWidth = (float)(nMinSize * Math.Sqrt(fAr));
                            fBoxHeight = (float)(nMinSize / Math.Sqrt(fAr));
                            // xmin
                            rgfTopData[nIdx] = (fCenterX - fBoxWidth / 2.0f) / nImgW;
                            nIdx++;
                            // ymin
                            rgfTopData[nIdx] = (fCenterY - fBoxHeight / 2.0f) / nImgH;
                            nIdx++;
                            // xmax
                            rgfTopData[nIdx] = (fCenterX + fBoxWidth / 2.0f) / nImgW;
                            nIdx++;
                            // ymax
                            rgfTopData[nIdx] = (fCenterY + fBoxHeight / 2.0f) / nImgH;
                            nIdx++;
                        }
                    }
                }
            }

            // Clip the prior's coordinate such that it is within [0,1]
            if (m_bClip)
            {
                for (int d = 0; d < nDim; d++)
                {
                    rgfTopData[d] = Math.Min(Math.Max(rgfTopData[d], 0.0f), 1.0f);
                }
            }

            // Set the variance.
            int nTopOffset = colTop[0].offset(0, 1);

            if (m_rgfVariance.Count > 1)
            {
                int nCount = 0;
                for (int h = 0; h < nLayerH; h++)
                {
                    for (int w = 0; w < nLayerW; w++)
                    {
                        for (int i = 0; i < m_nNumPriors; i++)
                        {
                            for (int j = 0; j < 4; j++)
                            {
                                rgfTopData[nTopOffset + nCount] = m_rgfVariance[j];
                                nCount++;
                            }
                        }
                    }
                }
            }

            colTop[0].mutable_cpu_data = Utility.ConvertVec<T>(rgfTopData);

            if (m_rgfVariance.Count == 1)
                colTop[0].SetData(m_rgfVariance[0], nTopOffset, nDim);
        }

        /// @brief Not implemented.
        protected override void backward(BlobCollection<T> colTop, List<bool> rgbPropagateDown, BlobCollection<T> colBottom)
        {
            new NotImplementedException();
        }
    }
}
