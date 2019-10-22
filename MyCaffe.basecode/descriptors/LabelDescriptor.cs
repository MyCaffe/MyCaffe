using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The LabelDescriptor class describes a single label.
    /// </summary>
    [Serializable]
    public class LabelDescriptor
    {
        int m_nActiveLabel;
        int m_nLabel;
        string m_strName;
        int m_nImageCount;

        /// <summary>
        /// The LabelDescriptor constructor.
        /// </summary>
        /// <param name="nLabel">Specifies the original label.</param>
        /// <param name="nActiveLabel">Specifies the active label (used during training).</param>
        /// <param name="strName">Specifies the label name.</param>
        /// <param name="nImageCount">Specifies the number of images under this label.</param>
        public LabelDescriptor(int nLabel, int nActiveLabel, string strName, int nImageCount)
        {
            m_nLabel = nLabel;
            m_nActiveLabel = nActiveLabel;
            m_strName = strName;
            m_nImageCount = nImageCount;
        }

        /// <summary>
        /// The LabelDescriptor constructor.
        /// </summary>
        /// <param name="l">Specifies another LabelDescriptor used to create this one.</param>
        public LabelDescriptor(LabelDescriptor l)
            : this(l.Label, l.ActiveLabel, l.Name, l.ImageCount)
        {
        }

        /// <summary>
        /// Specifies the original label
        /// </summary>
        public int Label
        {
            get { return m_nLabel; }
        }

        /// <summary>
        /// Specifies the active label (used during training).
        /// </summary>
        public int ActiveLabel
        {
            get { return m_nActiveLabel; }
        }

        /// <summary>
        /// Specifies the label name.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Specifies the number of images under this label.
        /// </summary>
        public int ImageCount
        {
            get { return m_nImageCount; }
            set { m_nActiveLabel = value; }
        }

        /// <summary>
        /// Creates the string representation of the descriptor.
        /// </summary>
        /// <returns>The string representation of the descriptor is returned.</returns>
        public override string ToString()
        {
            return m_nActiveLabel.ToString() + " -> " + m_strName;
        }
    }
}
