using MyCaffe.basecode;
using MyCaffe.fillers;
using MyCaffe.param;
using MyCaffe.param.ssd;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The SsdSampler is used by the SSD algorithm to sample BBoxes.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class SsdSampler<T> : IDisposable
    {
        Log m_log;
        CudaDnn<T> m_cuda;
        BBoxUtility<T> m_util;
        Blob<T> m_blobWork;
        CryptoRandom m_random = new CryptoRandom();

        /// <summary>
        /// The constructor.
        /// </summary>
        public SsdSampler(CudaDnn<T> cuda, Log log)
        {
            m_log = log;
            m_cuda = cuda;
            m_blobWork = new Blob<T>(cuda, log, false);
            m_util = new BBoxUtility<T>(cuda, log);
        }

        /// <summary>
        /// Free all resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_blobWork != null)
            {
                m_blobWork.Dispose();
                m_blobWork = null;
            }
        }

        /// <summary>
        /// Find all annotated NormalizedBBox.
        /// </summary>
        /// <param name="anno_datum">Specifies the annotated datum.</param>
        /// <returns>The grouped object BBoxes are returned.</returns>
        public List<NormalizedBBox> GroupObjectBBoxes(SimpleDatum anno_datum)
        {
            List<NormalizedBBox> rgObjectBboxes = new List<NormalizedBBox>();

            if (anno_datum.annotation_group == null)
                return rgObjectBboxes;

            for (int i = 0; i < anno_datum.annotation_group.Count; i++)
            {
                AnnotationGroup anno_group = anno_datum.annotation_group[i];

                for (int j = 0; j < anno_group.annotations.Count; j++)
                {
                    Annotation annotation = anno_group.annotations[j];
                    rgObjectBboxes.Add(annotation.bbox);
                }
            }

            return rgObjectBboxes;
        }

        /// <summary>
        /// Check if the sampled bbox satisfies the constraints with all object bboxes.
        /// </summary>
        /// <param name="sampledBBox">Specifies the sampled BBox.</param>
        /// <param name="rgObjectBboxes">Specifies the list of object normalized BBoxes.</param>
        /// <param name="sampleConstraint">Specifies the sample constraint.</param>
        /// <returns>Returns whether or not the sample constraints are satisfied.</returns>
        public bool SatisfySampleConstraint(NormalizedBBox sampledBBox, List<NormalizedBBox> rgObjectBboxes, SamplerConstraint sampleConstraint)
        {
            bool bHasJaccardOverlap = sampleConstraint.min_jaccard_overlap.HasValue || sampleConstraint.max_jaccard_overlap.HasValue;
            bool bHasSampleCoverage = sampleConstraint.min_sample_coverage.HasValue || sampleConstraint.max_sample_coverage.HasValue;
            bool bHasObjectCoverage = sampleConstraint.min_object_coverage.HasValue || sampleConstraint.max_object_coverage.HasValue;
            bool bSatisfy = !bHasJaccardOverlap && !bHasSampleCoverage && !bHasObjectCoverage;

            // By default, the sampledBBox is 'positive' if not constraints are defined.
            if (bSatisfy)
                return true;

            // Check constraints.
            bool bFound = false;
            for (int i = 0; i < rgObjectBboxes.Count; i++)
            {
                NormalizedBBox objectBbox = rgObjectBboxes[i];

                // Test jaccard overlap.
                if (bHasJaccardOverlap)
                {
                    float fJaccardOverlap = m_util.JaccardOverlap(sampledBBox, objectBbox);

                    if (sampleConstraint.min_jaccard_overlap.HasValue && fJaccardOverlap < sampleConstraint.min_jaccard_overlap.Value)
                        continue;

                    if (sampleConstraint.max_jaccard_overlap.HasValue && fJaccardOverlap > sampleConstraint.max_jaccard_overlap.Value)
                        continue;

                    bFound = true;
                }

                // Test sample coverage
                if (bHasSampleCoverage)
                {
                    float fSampleCoverage = m_util.Coverage(sampledBBox, objectBbox);

                    if (sampleConstraint.min_sample_coverage.HasValue && fSampleCoverage < sampleConstraint.min_sample_coverage.Value)
                        continue;

                    if (sampleConstraint.max_sample_coverage.HasValue && fSampleCoverage > sampleConstraint.max_sample_coverage.Value)
                        continue;

                    bFound = true;
                }

                // Test object coverage
                if (bHasObjectCoverage)
                {
                    float fObjectOverage = m_util.Coverage(objectBbox, sampledBBox);

                    if (sampleConstraint.min_object_coverage.HasValue && fObjectOverage < sampleConstraint.min_object_coverage.Value)
                        continue;

                    if (sampleConstraint.max_jaccard_overlap.HasValue && fObjectOverage > sampleConstraint.max_jaccard_overlap.Value)
                        continue;

                    bFound = true;
                }

                if (bFound)
                    return true;
            }

            return bFound;
        }

        private float randomUniformValue(float fMin, float fMax)
        {
            fMin = (float)Math.Round(fMin, 5);
            fMax = (float)Math.Round(fMax, 5);

            m_log.CHECK_LE(fMin, fMax, "The min mumst be <= the max!");

            if (fMin == 0 && fMax == 0)
                return 0.0f;
            else if (fMin == 1 && fMax == 1)
                return 1.0f;
            else
            {
                double dfRandom = m_random.NextDouble();
                float fRange = fMax - fMin;

                return (float)(dfRandom * fRange) + fMin;
            }
        }

        private NormalizedBBox sampleBBox(Sampler sampler)
        {
            // Get random scale.
            m_log.CHECK_GE(sampler.max_scale, sampler.min_scale, "The sampler max scale must be >= the min scale.");
            m_log.CHECK_GT(sampler.min_scale, 0, "The sampler min scale must be > 0.");
            m_log.CHECK_LE(sampler.max_scale, 1, "The sampler max scale must be <= 1.");

            float fScale = randomUniformValue(sampler.min_scale, sampler.max_scale);
            float fAspectRatio = randomUniformValue(sampler.min_aspect_ratio, sampler.max_aspect_ratio);
            float fPow2Scale = (float)Math.Pow(fScale, 2.0);

            fAspectRatio = (float)Math.Max(fAspectRatio, fPow2Scale);
            fAspectRatio = (float)Math.Min(fAspectRatio, 1.0 / fPow2Scale);
            float fSqrtAspectRatio = (float)Math.Sqrt(fAspectRatio);

            // Figure out bbox dimension
            float fBboxWidth = fScale * fSqrtAspectRatio;
            float fBboxHeight = fScale / fSqrtAspectRatio;

            // Figure out top left coordinates
            float fWoff = randomUniformValue(0, 1.0f - fBboxWidth);
            float fHoff = randomUniformValue(0, 1.0f - fBboxHeight);

            return new NormalizedBBox(fWoff, fHoff, fWoff + fBboxWidth, fHoff + fBboxHeight);
        }

        /// <summary>
        /// Generate samples from the NormalizedBBox using the BatchSampler.
        /// </summary>
        /// <param name="sourceBBox">Specifies the source BBox.</param>
        /// <param name="rgObjectBboxes">Specifies the object normalized BBoxes.</param>
        /// <param name="batchSampler">Specifies the batch sampler.</param>
        /// <returns>The list of normalized BBoxes generated is returned.</returns>
        public List<NormalizedBBox> GenerateSamples(NormalizedBBox sourceBBox, List<NormalizedBBox> rgObjectBboxes, BatchSampler batchSampler)
        {
            List<NormalizedBBox> rgSampledBBoxes = new List<NormalizedBBox>();
            int nFound = 0;

            for (int i = 0; i < batchSampler.max_trials; i++)
            {
                if (batchSampler.max_sample > 0 && nFound >= batchSampler.max_sample)
                    break;

                // Generate sampleBbox in the normalized space [0,1]
                NormalizedBBox sampledBbox = sampleBBox(batchSampler.sampler);

                // Transform the sampledBbox w.r.t. the source BBox.
                sampledBbox = m_util.Locate(sourceBBox, sampledBbox);

                // Determine if the sampled BBox is positive or negative by the constraint.
                if (SatisfySampleConstraint(sampledBbox, rgObjectBboxes, batchSampler.sample_constraint))
                {
                    nFound++;
                    rgSampledBBoxes.Add(sampledBbox);
                }
            }

            return rgSampledBBoxes;
        }

        /// <summary>
        /// Generate samples from the annotated Datum using the list of BatchSamplers.
        /// </summary>
        /// <param name="anno_datum"></param>
        /// <param name="rgBatchSamplers"></param>
        /// <returns>All samples bboxes that satisfy the constraints defined in the BatchSampler are returned.</returns>
        public List<NormalizedBBox> GenerateBatchSamples(SimpleDatum anno_datum, List<BatchSampler> rgBatchSamplers)
        {
            List<NormalizedBBox> rgSampledBBoxes = new List<NormalizedBBox>();
            List<NormalizedBBox> rgObjectBBoxes = GroupObjectBBoxes(anno_datum);

            for (int i = 0; i < rgBatchSamplers.Count; i++)
            {
                if (rgBatchSamplers[i].use_original_image)
                {
                    NormalizedBBox unitBbox = new NormalizedBBox(0, 0, 1, 1);
                    rgSampledBBoxes.AddRange(GenerateSamples(unitBbox, rgObjectBBoxes, rgBatchSamplers[i]));
                }
            }

            return rgSampledBBoxes;
        }
    }
}
