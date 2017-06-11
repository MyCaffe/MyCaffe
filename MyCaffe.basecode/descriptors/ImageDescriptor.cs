using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The ImageDescriptor class describes a single image in the database.
    /// </summary>
    [Serializable]
    public class ImageDescriptor
    {
        int m_nID;
        int m_nHeight;
        int m_nWidth;
        int m_nChannels;
        bool m_bEncoded;
        int m_nSourceID;
        int m_nIdx;
        int m_nActiveLabel;
        bool m_bActive;
        string m_strDescription;

        /// <summary>
        /// The ImageDescriptor constructor.
        /// </summary>
        /// <param name="nID">Specifies the image ID.</param>
        /// <param name="nH">Specifies the image heights.</param>
        /// <param name="nW">Specifies the image width.</param>
        /// <param name="nC">Specifies teh image channels.</param>
        /// <param name="bEncoded">Specifies whether or not the image has encoded data.</param>
        /// <param name="nSrcID">Specifies the ID of the source holding the image.</param>
        /// <param name="nIdx">Specifies the image index.</param>
        /// <param name="nActiveLabel">Specifies the active label of the image.</param>
        /// <param name="bActive">Specifies whether or not the image is active.</param>
        /// <param name="strDesc">Specifies the description of the image.</param>
        public ImageDescriptor(int nID, int nH, int nW, int nC, bool bEncoded, int nSrcID, int nIdx, int nActiveLabel, bool bActive, string strDesc)
        {
            m_nID = nID;
            m_nHeight = nH;
            m_nWidth = nW;
            m_nChannels = nC;
            m_bEncoded = bEncoded;
            m_nSourceID = nSrcID;
            m_nIdx = nIdx;
            m_nActiveLabel = nActiveLabel;
            m_bActive = bActive;
            m_strDescription = strDesc;
        }

        /// <summary>
        /// The ImageDescriptor constructor.
        /// </summary>
        /// <param name="id">Specifies another ImageDescriptor to copy.</param>
        public ImageDescriptor(ImageDescriptor id)
            : this(id.ID, id.Height, id.Width, id.Channels, id.Encoded, id.SourceID, id.Index, id.ActiveLabel, id.Active, id.Description)
        {
        }

        /// <summary>
        /// Returns the ID of the image.
        /// </summary>
        public int ID
        {
            get { return m_nID; }
        }

        /// <summary>
        /// Returns the height of the image.
        /// </summary>
        public int Height
        {
            get { return m_nHeight; }
        }

        /// <summary>
        /// Returns the width of the image.
        /// </summary>
        public int Width
        {
            get { return m_nWidth; }
        }

        /// <summary>
        /// Returns the channels of the image (i.e. 3 = RGB, 1 = B/W)
        /// </summary>
        public int Channels
        {
            get { return m_nChannels; }
        }

        /// <summary>
        /// Returns whether or not the image is encoded.
        /// </summary>
        public bool Encoded
        {
            get { return m_bEncoded; }
        }

        /// <summary>
        /// Returns the ID of the source associated with the image.
        /// </summary>
        public int SourceID
        {
            get { return m_nSourceID; }
        }

        /// <summary>
        /// Returns the index of the image.
        /// </summary>
        public int Index
        {
            get { return m_nIdx; }
        }

        /// <summary>
        /// Returns the active label of the image.
        /// </summary>
        public int ActiveLabel
        {
            get { return m_nActiveLabel; }
        }

        /// <summary>
        /// Returns whether or not the image is active.
        /// </summary>
        public bool Active
        {
            get { return m_bActive; }
        }

        /// <summary>
        /// Returns the description of the image.
        /// </summary>
        public string Description
        {
            get { return m_strDescription; }
        }
    }
}
