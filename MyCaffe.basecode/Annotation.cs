using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The Annotation class is used by annotations attached to SimpleDatum's and used in SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class Annotation
    {
        int m_nInstanceId = 0;
        NormalizedBBox m_bbox;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bbox">Specifies the bounding box.</param>
        /// <param name="nInstanceId">Specifies the instance ID.</param>
        public Annotation(NormalizedBBox bbox, int nInstanceId = 0)
        {
            m_bbox = bbox;
            m_nInstanceId = nInstanceId;
        }

        /// <summary>
        /// Get/set the instance ID.
        /// </summary>
        public int instance_id
        {
            get { return m_nInstanceId; }
            set { m_nInstanceId = value; }
        }

        /// <summary>
        /// Get/set the bounding box.
        /// </summary>
        public NormalizedBBox bbox
        {
            get { return m_bbox; }
            set { m_bbox = value; }
        }
    }

    /// <summary>
    /// The AnnoationGroup class manages a group of annotations.
    /// </summary>
    public class AnnotationGroup
    {
        int m_nGroupLabel = 0;
        List<Annotation> m_rgAnnotations = new List<Annotation>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="rgAnnotations">Optionally, specifies the list of group annotations.</param>
        /// <param name="nGroupLabel">Specifies the group label.</param>
        public AnnotationGroup(List<Annotation> rgAnnotations = null, int nGroupLabel = 0)
        {
            if (rgAnnotations != null && rgAnnotations.Count > 0)
                m_rgAnnotations.AddRange(rgAnnotations);

            m_nGroupLabel = nGroupLabel;
        }

        /// <summary>
        /// Get/set the group annoations.
        /// </summary>
        public List<Annotation> annotations
        {
            get { return m_rgAnnotations; }
            set { m_rgAnnotations = value; }
        }

        /// <summary>
        /// Get/set the group label.
        /// </summary>
        public int group_label
        {
            get { return m_nGroupLabel; }
            set { m_nGroupLabel = value; }
        }
    }
}
