using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The LabelBBox manages a bounding box used in SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    public class LabelBBox
    {
        Dictionary<int, List<NormalizedBBox>> m_rgItems = new Dictionary<int, List<NormalizedBBox>>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public LabelBBox()
        {
        }

        /// <summary>
        /// Returns the internal dictionary of items as a list.
        /// </summary>
        /// <returns></returns>
        public List<KeyValuePair<int, List<NormalizedBBox>>> ToList()
        {
            return m_rgItems.ToList();
        }

        /// <summary>
        /// Returns the number of items.
        /// </summary>
        public int Count
        {
            get { return m_rgItems.Count; }
        }

        /// <summary>
        /// Returns whether or not the label is contained in the label bounding boxe set.
        /// </summary>
        /// <param name="nLabel">Specifies the label.</param>
        /// <returns>If the label exists, <i>true</i> is returned, otherwise, <i>false</i> is returned.</returns>
        public bool Contains(int nLabel)
        {
            return m_rgItems.ContainsKey(nLabel);
        }

        /// <summary>
        /// Returns the list of NormalizedBBox items at the label specified.
        /// </summary>
        /// <param name="nLabel">Specifies the label.</param>
        /// <returns>The list of NormalizedBBox items at the label is returned.</returns>
        public List<NormalizedBBox> this[int nLabel]
        {
            get
            {
                if (!m_rgItems.ContainsKey(nLabel))
                    m_rgItems.Add(nLabel, new List<NormalizedBBox>());

                return m_rgItems[nLabel];
            }
        }

        /// <summary>
        /// Remove all items from the group.
        /// </summary>
        public void Clear()
        {
            m_rgItems.Clear();
        }
    }
}
