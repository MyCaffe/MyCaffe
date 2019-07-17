using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The BBox class processes the NormalizedBBox data.
    /// </summary>
    public class BBoxUtility<T>
    {
        CudaDnn<T> m_cuda;
        Log m_log;


        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the connection to Cuda and cuDNN.</param>
        /// <param name="log">Specifies the output log.</param>
        public BBoxUtility(CudaDnn<T> cuda, Log log)
        {
            m_cuda = cuda;
            m_log = log;
        }

        /// <summary>
        /// Clip the BBox to a set range.
        /// </summary>
        /// <param name="bbox">Specifies the input bounding box.</param>
        /// <param name="fHeight">Specifies the clipping height.</param>
        /// <param name="fWidth">Specifies the clipping width.</param>
        /// <returns>A new, clipped NormalizedBBox is returned.</returns>
        public NormalizedBBox Clip(NormalizedBBox bbox, float fHeight = 1.0f, float fWidth = 1.0f)
        {
            NormalizedBBox clipped = bbox.Clone();
            clipped.xmin = Math.Max(Math.Min(bbox.xmin, fWidth), 0.0f);
            clipped.ymin = Math.Max(Math.Min(bbox.ymin, fHeight), 0.0f);
            clipped.xmax = Math.Max(Math.Min(bbox.xmax, fWidth), 0.0f);
            clipped.ymax = Math.Max(Math.Min(bbox.ymax, fHeight), 0.0f);
            clipped.size = Size(clipped);
            return clipped;
        }

        /// <summary>
        /// Scale the BBox to a set range.
        /// </summary>
        /// <param name="bbox">Specifies the input bounding box.</param>
        /// <param name="fHeight">Specifies the scaling height.</param>
        /// <param name="fWidth">Specifies the scaling width.</param>
        /// <returns>A new, scaled NormalizedBBox is returned.</returns>
        public NormalizedBBox Scale(NormalizedBBox bbox, int nHeight, int nWidth)
        {
            NormalizedBBox scaled = bbox.Clone();
            scaled.xmin = bbox.xmin * nWidth;
            scaled.ymin = bbox.ymin * nHeight;
            scaled.xmax = bbox.xmax * nWidth;
            scaled.ymax = bbox.ymax * nHeight;
            bool bNormalized = !(nWidth > 1 || nHeight > 1);
            scaled.size = Size(scaled, bNormalized);
            return scaled;
        }

        /// <summary>
        /// Calculate the size of a BBox.
        /// </summary>
        /// <param name="bbox">Specifies the input bounding box.</param>
        /// <param name="bNormalized">Specifies whether or not the bounding box falls within the range [0,1].</param>
        /// <returns>The size of the bounding box is returned.</returns>
        public float Size(NormalizedBBox bbox, bool bNormalized = true)
        {
            if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin)
            {
                // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0
                return 0f;
            }

            float fWidth = bbox.xmax - bbox.xmin;
            float fHeight = bbox.ymax - bbox.ymin;

            if (bNormalized)
                return fWidth * fHeight;
            else // bbox is not in range [0,1]
                return (fWidth + 1) * (fHeight + 1);
        }
    }
}
