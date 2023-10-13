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
        int m_nHt;
        int m_nWd;
        int m_nCh;
        bool m_bIsRealData;
        int m_nImageCount;
        int m_nInactiveCount;
        bool m_bSaveImagesToFile;
        List<LabelDescriptor> m_rgLabels = new List<LabelDescriptor>();
        ParameterDescriptorCollection m_colParameters = new ParameterDescriptorCollection();
        string m_strLabelCounts;
        int m_nCopyOfSourceID;
        TemporalDescriptor m_temporalDesc = null;

        /// <summary>
        /// The SourceDescriptor constructor.
        /// </summary>
        /// <param name="nID">Specifies the database ID of the item.</param>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="nWd">Specifies the width of each data item.</param>
        /// <param name="nHt">Specifies the height of each data item.</param>
        /// <param name="nCh">Specifies the channels of each data item.</param>
        /// <param name="bIsRealData">Specifies whether or not the data items contain real data or <i>byte</i> data.</param>
        /// <param name="bSaveImagesToFile">Specifies whether the images are saved to the file system (<i>true</i>), or directly to the database (<i>false</i>).</param>
        /// <param name="nCopyOfSourceId">Specifies whether or not this is a copy of another source and if so, this is the ID of the original source.</param>
        /// <param name="strOwner">Optionally, specifies the identifier of the item's owner.</param>
        /// <param name="nCount">Optionallty, specifies the number of items in the data source.</param>
        /// <param name="rgLabels">Optionally, specifies a list of LabelDescriptors that describe the labels used by the data items.</param>
        /// <param name="strLabelCounts">Optionally, specifies a string containing the label counts.</param>
        /// <param name="rgbTemporal">Optionally, specifies the temporal data.</param>
        public SourceDescriptor(int nID, string strName, int nWd, int nHt, int nCh, bool bIsRealData, bool bSaveImagesToFile, int nCopyOfSourceId = 0, string strOwner = null, int nCount = 0, List<LabelDescriptor> rgLabels = null, string strLabelCounts = null, byte[] rgbTemporal = null)
            : base(nID, strName, strOwner)
        {
            m_nHt = nHt;
            m_nWd = nWd;
            m_nCh = nCh;
            m_bIsRealData = bIsRealData;
            m_nImageCount = nCount;
            m_strLabelCounts = strLabelCounts;
            m_rgLabels = rgLabels;
            m_bSaveImagesToFile = bSaveImagesToFile;
            m_nCopyOfSourceID = nCopyOfSourceId;
            m_temporalDesc = TemporalDescriptor.FromBytes(rgbTemporal);
        }

        /// <summary>
        /// The SourceDescriptor constructor.
        /// </summary>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="bSaveImagesToFile">Specifies whether the images are saved to the file system (<i>true</i>), or directly to the database (<i>false</i>).</param>
        public SourceDescriptor(string strName, bool bSaveImagesToFile)
            : this(0, strName, 0, 0, 0, false, bSaveImagesToFile)
        {
        }

        /// <summary>
        /// The SourceDescriptor constructor.
        /// </summary>
        /// <param name="s">Specifies another SourceDescriptor used to create this one.</param>
        public SourceDescriptor(SourceDescriptor s)
            : this(s.ID, s.Name, s.Width, s.Height, s.Channels, s.IsRealData, s.SaveImagesToFile, s.CopyOfSourceID, s.Owner, s.ImageCount, s.Labels, s.LabelCountsAsText)
        {
            m_colParameters = new descriptors.ParameterDescriptorCollection();
            m_nInactiveCount = s.m_nInactiveCount;
            
            foreach (ParameterDescriptor p in s.m_colParameters)
            {
                m_colParameters.Add(new ParameterDescriptor(p));
            }

            if (s.m_temporalDesc != null)
                m_temporalDesc = new TemporalDescriptor(s.m_temporalDesc);
        }

        /// <summary>
        /// Copy another SourceDesciptor into this one.
        /// </summary>
        /// <param name="sd">Specifies the SourceDesciptor to copy.</param>
        public void Copy(SourceDescriptor sd)
        {
            base.Copy(sd);

            m_nCh = sd.m_nCh;
            m_nHt = sd.m_nHt;
            m_nWd = sd.m_nWd;
            m_bIsRealData = sd.m_bIsRealData;
            m_nImageCount = sd.m_nImageCount;
            m_nInactiveCount = sd.m_nInactiveCount;
            m_bSaveImagesToFile = sd.m_bSaveImagesToFile;
            m_nCopyOfSourceID = sd.m_nCopyOfSourceID;

            m_rgLabels = new List<descriptors.LabelDescriptor>();
            foreach (LabelDescriptor ld in sd.m_rgLabels)
            {
                m_rgLabels.Add(new descriptors.LabelDescriptor(ld));
            }

            m_colParameters = new descriptors.ParameterDescriptorCollection();
            foreach (ParameterDescriptor p in sd.m_colParameters)
            {
                m_colParameters.Add(new ParameterDescriptor(p));
            }

            m_strLabelCounts = sd.m_strLabelCounts;

            if (sd.m_temporalDesc != null)
                m_temporalDesc = new TemporalDescriptor(sd.m_temporalDesc);
        }

        /// <summary>
        /// Resize the testing and training data sources.
        /// </summary>
        /// <param name="nChannels">Specifies the new channel size.</param>
        /// <param name="nHeight">Specifies the new height size.</param>
        /// <param name="nWidth">Specifies the new width size.</param>
        public void Resize(int nChannels, int nHeight, int nWidth)
        {
            m_nCh = nChannels;
            m_nHt = nHeight;
            m_nWd = nWidth;
        }

        /// <summary>
        /// Gets whether or not the images are saved to the file system (<i>true</i>), or directly to the database (<i>false</i>).
        /// </summary>
        [Description("Specifies whether the images are saved to the file system or directly to the database.")]
        public bool SaveImagesToFile
        {
            get { return m_bSaveImagesToFile; }
        }

        /// <summary>
        /// Get/set the Source ID from which this source was copied.  If this Source is an original, this property should be 0.
        /// </summary>
        [Description("If this source was copied from another source, this property returns the ID of the original Source, otherwise 0 is returned.")]
        [ReadOnly(true)]
        public int CopyOfSourceID
        {
            get { return m_nCopyOfSourceID; }
            set { m_nCopyOfSourceID = value; }
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
        [Description("Specifies the item height in pixels.")]
        public int Height
        {
            get { return m_nHt; }
        }

        /// <summary>
        /// Returns the width of each data item in the data source.
        /// </summary>
        [Description("Specifies the item width in pixels.")]
        public int Width
        {
            get { return m_nWd; }
        }

        /// <summary>
        /// Returns the item colors - 1 channel = black/white, 3 channels = RGB color.
        /// </summary>
        [Description("Specifies the item colors - 1 channel = black/white, 3 channels = RGB color.")]
        public int Channels
        {
            get { return m_nCh; }
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
        /// Set the image count.
        /// </summary>
        /// <param name="nCount">Specifies the new count.</param>
        public void SetImageCount(int nCount)
        {
            m_nImageCount = nCount;
        }

        /// <summary>
        /// Returns the number of inactive images within this data source.
        /// </summary>
        [Description("Specifies the number of inactive images within this data source.")]
        public int InactiveImageCount
        {
            get { return m_nInactiveCount; }
        }

        /// <summary>
        /// Set the number of inactive images within this data source.
        /// </summary>
        /// <param name="nCount"></param>
        public void SetInactiveImageCount(int nCount)
        {
            m_nInactiveCount = nCount;
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

        /// <summary>
        /// Get/set the source parameters (if any).
        /// </summary>
        [Description("Specifies the parameters of the data source (if any).")]
        public ParameterDescriptorCollection Parameters
        {
            get { return m_colParameters; }
            set { m_colParameters = value; }
        }

        /// <summary>
        /// Get/set the temporal descriptor (if any).
        /// </summary>
        [ReadOnly(true)]
        public TemporalDescriptor TemporalDescriptor
        {
            get { return m_temporalDesc; }
            set { m_temporalDesc = value; }
        }
    }
}
