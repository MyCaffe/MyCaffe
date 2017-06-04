using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The SourceDescriptor class contains all information describing a data source.
    /// </summary>
    [Serializable]
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class SourceDescriptor : BaseDescriptor
    {
        int m_nImageHt;
        int m_nImageWd;
        int m_nImageCh;
        bool m_bIsRealData;
        int m_nImageCount;
        List<LabelDescriptor> m_rgLabels = new List<LabelDescriptor>();
        string m_strLabelCounts;

        /// <summary>
        /// The SourceDescriptor constructor.
        /// </summary>
        /// <param name="nID">Specifies the database ID of the item.</param>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="nWd">Specifies the width of each data item.</param>
        /// <param name="nHt">Specifies the height of each data item.</param>
        /// <param name="nCh">Specifies the channels of each data item.</param>
        /// <param name="bIsRealData">Specifies whether or not the data items contain real data or <i>byte</i> data.</param>
        /// <param name="strOwner">Optionally, specifies the identifier of the item's owner.</param>
        /// <param name="nCount">Optionallty, specifies the number of items in the data source.</param>
        /// <param name="rgLabels">Optionally, specifies a list of LabelDescriptors that describe the labels used by the data items.</param>
        /// <param name="strLabelCounts">Optionally, specifies a string containing the label counts.</param>
        public SourceDescriptor(int nID, string strName, int nWd, int nHt, int nCh, bool bIsRealData, string strOwner = null, int nCount = 0, List<LabelDescriptor> rgLabels = null, string strLabelCounts = null)
            : base(nID, strName, strOwner)
        {
            m_nImageHt = nHt;
            m_nImageWd = nWd;
            m_nImageCh = nCh;
            m_bIsRealData = bIsRealData;
            m_nImageCount = nCount;
            m_strLabelCounts = strLabelCounts;
            m_rgLabels = rgLabels;
        }

        /// <summary>
        /// The SourceDescriptor constructor.
        /// </summary>
        /// <param name="strName">Specifies the name of the item.</param>
        public SourceDescriptor(string strName)
            : this(0, strName, 0, 0, 0, false)
        {
        }

        /// <summary>
        /// The SourceDescriptor constructor.
        /// </summary>
        /// <param name="s">Specifies another SourceDescriptor used to create this one.</param>
        public SourceDescriptor(SourceDescriptor s)
            : this(s.ID, s.Name, s.ImageWidth, s.ImageHeight, s.ImageChannels, s.IsRealData, s.Owner, s.ImageCount, s.Labels, s.LabelCountsAsText)
        {
        }

        /// <summary>
        /// Get/set the list of LabelDescriptors that describe the labels used by the data items.
        /// </summary>
        [Browsable(false)]
        public List<LabelDescriptor> Labels
        {
            get { return m_rgLabels; }
            set { m_rgLabels = value; }
        }

        /// <summary>
        /// Returns the height of each data item in the data source.
        /// </summary>
        [Description("Specifies the image height in pixels.")]
        public int ImageHeight
        {
            get { return m_nImageHt; }
        }

        /// <summary>
        /// Returns the width of each data item in the data source.
        /// </summary>
        [Description("Specifies the image width in pixels.")]
        public int ImageWidth
        {
            get { return m_nImageWd; }
        }

        /// <summary>
        /// Returns the image colors - 1 channel = black/white, 3 channels = RGB color.
        /// </summary>
        [Description("Specifies the image colors - 1 channel = black/white, 3 channels = RGB color.")]
        public int ImageChannels
        {
            get { return m_nImageCh; }
        }

        /// <summary>
        /// Returns whether or not the each data point represents a real or integer number.  Integer numbers are used for black/white and color images where each data point falls within the range [0 - 255].
        /// </summary>
        [Description("Specifies whether or not the each data point represents a real or integer number.  Integer numbers are used for black/white and color images where each data point falls within the range [0 - 255].")]
        public bool IsRealData
        {
            get { return m_bIsRealData; }
        }

        /// <summary>
        /// Returns the number of images within this data source.
        /// </summary>
        [Description("Specifies the number of images within this data source.")]
        public int ImageCount
        {
            get { return m_nImageCount; }
        }

        /// <summary>
        /// Get/set a string that lists the number of images for each label associated with this data source.
        /// </summary>
        [Description("Lists the number of images for each label associated with this data source.")]
        [ReadOnly(true)]
        public string LabelCountsAsText
        {
            get { return m_strLabelCounts; }
            set { m_strLabelCounts = value; }
        }

        /// <summary>
        /// Return a string representation of thet SourceDescriptor.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            return "Source Description " + Name;
        }
    }
}
