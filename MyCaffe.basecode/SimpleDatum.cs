using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The SimpleDatum class holds a data input within host memory.
    /// </summary>
    [Serializable]
    public class SimpleDatum 
    {
        int m_nIndex = 0;
        int m_nOriginalLabel = 0;
        int m_nLabel = 0;
        int m_nHeight = 0;
        int m_nWidth = 0;
        int m_nChannels = 0;
        byte[] m_rgByteData = null;
        float[] m_rgRealDataF = null;
        double[] m_rgRealDataD = null;
        bool m_bIsRealData = false;
        DateTime m_dt = DateTime.MinValue;
        int m_nOriginalBoost = 0;
        int m_nBoost = 0;
        bool m_bAutoLabeled = false;
        int m_nImageID = 0;
        int m_nVirtualID = 0;
        int m_nGroupID = 0;
        byte[] m_rgDataCriteria = null;
        DATA_FORMAT m_dataCriteriaFormat = DATA_FORMAT.NONE;
        byte[] m_rgDebugData = null;
        DATA_FORMAT m_debugDataFormat = DATA_FORMAT.NONE;
        string m_strDesc = null;
        int m_nSourceID = 0;
        int m_nOriginalSourceID = 0;
        int m_nHitCount = 0;
        ANNOTATION_TYPE m_nAnnotationType = ANNOTATION_TYPE.NONE;
        AnnotationGroupCollection m_rgAnnotationGroup = null;
        /// <summary>
        /// Specifies a user value.
        /// </summary>
        protected object m_tag = null;
        /// <summary>
        /// Specifies the name of the user value.
        /// </summary>
        protected string m_strTagName = null;

        /// <summary>
        /// Specifies the annotation type when using annotations.
        /// </summary>
        public enum ANNOTATION_TYPE
        {
            /// <summary>
            /// Specifies that annotations are not used.
            /// </summary>
            NONE = -1,
            /// <summary>
            /// Specifies to use the bounding box annoation type.
            /// </summary>
            BBOX = 0
        }

        /// <summary>
        /// Defines the data format of the DebugData and DataCriteria when specified.
        /// </summary>
        public enum DATA_FORMAT
        {
            /// <summary>
            /// Specifies that there is no data.
            /// </summary>
            NONE,
            /// <summary>
            /// Specifies that the data contains a black and white square image packed as a set of byte values [0-255] in the order: image wxh.
            /// </summary>
            IMAGE_1CH,
            /// <summary>
            /// Specifies that the data contains a RGB square image packed as a set of byte values [0-255] in the order: wxh of R, wxh of G and wxh of B.
            /// </summary>
            IMAGE_3CH,
            /// <summary>
            /// Specifies that the data contains segmentation data where the height and width are equal.
            /// </summary>
            SEGMENTATION,
            /// <summary>
            /// Specifies that the data contains a dictionary of values.
            /// </summary>
            DICTIONARY,
            /// <summary>
            /// Specifies that the data contains custom data.
            /// </summary>
            CUSTOM,
            /// <summary>
            /// Specifies that the data contains an image converted to a byte array using ImageTools.ImageToByteArray.
            /// </summary>
            /// <remarks>To convert back to an image, use the ImageTools.ByteArrayToImage method.</remarks>
            BITMAP,
            /// <summary>
            /// Specifies that the data contains a list of double values where the first item is an Int32 which is the count followed by that many double values.
            /// </summary>
            LIST_DOUBLE,
            /// <summary>
            /// Specifies that the data contains a list of float values where the first item is an Int32 which is the count followed by that many float values.
            /// </summary>
            LIST_FLOAT,
            /// <summary>
            /// Specifies that the data contains annotation data used with SSD.
            /// </summary>
            ANNOTATION_DATA
        }


        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        public SimpleDatum()
        {
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="bIsReal">Specifies whether or not the data values are <i>double</i> or <i>byte</i>.</param>
        /// <param name="nChannels">Specifies the number of channels in the data (e.g. 3 for color, 1 for black and white images)</param>
        /// <param name="nWidth">Specifies the width of the data (e.g. the number of pixels wide).</param>
        /// <param name="nHeight">Specifies the height of the data (e.g. the number of pixels high).</param>
        /// <param name="nLabel">Optionally, specifies the known label of the data (default = -1).</param>
        /// <param name="dtTime">Optionally, specifies a time-stamp associated with the data (default = null).</param>
        /// <param name="nBoost">Optionally, specifies the boost to use with the data (default = 0, where a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Optionally, specifies whether or not the label was auto-generated (default = false).</param>
        /// <param name="nIdx">Optionally, specifies the index of the data (default = -1).</param>
        /// <param name="nVirtualID">Optionally, specifies a virtual index for the data (default = 0).  When specified, the SimpleDatum is used to reference another.</param>
        /// <param name="nImageID">Optionally, specifies the image ID within the database (default = 0).</param>
        /// <param name="nSourceID">Optionally, specifies the data source ID of the data source that owns this image (default = 0).</param>
        /// <param name="nOriginalSourceID">Optionally, specifies the ogiginal source ID which is set when using a virtual ID - the original source ID is the ID of the source associated with the image with the ID of the virtual ID.</param>
        public SimpleDatum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel = -1, DateTime? dtTime = null, int nBoost = 0, bool bAutoLabeled = false, int nIdx = -1, int nVirtualID = 0, int nImageID = 0, int nSourceID = 0, int nOriginalSourceID = 0)
        {
            m_nChannels = nChannels;
            m_nWidth = nWidth;
            m_nHeight = nHeight;
            m_nOriginalLabel = nLabel;
            m_nLabel = nLabel;
            m_dt = dtTime.GetValueOrDefault(DateTime.MinValue);
            m_nOriginalBoost = nBoost;
            m_nBoost = nBoost;
            m_bAutoLabeled = bAutoLabeled;
            m_nVirtualID = nVirtualID;
            m_bIsRealData = bIsReal;
            m_nIndex = nIdx;
            m_nImageID = nImageID;
            m_nSourceID = nSourceID;
            m_nOriginalSourceID = nOriginalSourceID;
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="bIsReal">Specifies whether or not the data values are <i>double</i> or <i>byte</i>.</param>
        /// <param name="nChannels">Specifies the number of channels in the data (e.g. 3 for color, 1 for black and white images)</param>
        /// <param name="nWidth">Specifies the width of the data (e.g. the number of pixels wide).</param>
        /// <param name="nHeight">Specifies the height of the data (e.g. the number of pixels high).</param>
        /// <param name="nLabel">Specifies the known label of the data.</param>
        /// <param name="dtTime">Specifies a time-stamp associated with the data.</param>
        /// <param name="rgData">Specifies the data as a list of <i>bytes</i> (expects <i>bIsReal</i> = <i>false</i>).</param>
        /// <param name="nBoost">Specifies the boost to use with the data (a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Specifies whether or not the label was auto-generated.</param>
        /// <param name="nIdx">Specifies the index of the data.</param>
        /// <param name="nVirtualID">Specifies a virtual index for the data (default = 0).  When specified, the SimpleDatum is used to reference another.</param>
        /// <param name="nImageID">Specifies the image ID within the database.</param>
        /// <param name="nSourceID">Specifies the data source ID of the data source that owns this image.</param>
        /// <param name="nOriginalSourceID">Optionally, specifies the ogiginal source ID which is set when using a virtual ID - the original source ID is the ID of the source associated with the image with the ID of the virtual ID.</param>
        public SimpleDatum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, List<byte> rgData, int nBoost, bool bAutoLabeled, int nIdx, int nVirtualID = 0, int nImageID = 0, int nSourceID = 0, int nOriginalSourceID = 0)
        {
            m_nChannels = nChannels;
            m_nWidth = nWidth;
            m_nHeight = nHeight;
            m_nOriginalLabel = nLabel;
            m_nLabel = nLabel;
            m_dt = dtTime;
            m_nOriginalBoost = nBoost;
            m_nBoost = nBoost;
            m_bAutoLabeled = bAutoLabeled;
            m_nVirtualID = nVirtualID;
            m_bIsRealData = bIsReal;
            m_nIndex = nIdx;
            m_nImageID = nImageID;
            m_nSourceID = nSourceID;
            m_nOriginalSourceID = nOriginalSourceID;

            if (bIsReal)
                throw new ArgumentException("The data sent is not real, but the bIsReal is set to true!");

            m_rgByteData = rgData.ToArray();
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="bIsReal">Specifies whether or not the data values are <i>double</i> or <i>byte</i>.</param>
        /// <param name="nChannels">Specifies the number of channels in the data (e.g. 3 for color, 1 for black and white images)</param>
        /// <param name="nWidth">Specifies the width of the data (e.g. the number of pixels wide).</param>
        /// <param name="nHeight">Specifies the height of the data (e.g. the number of pixels high).</param>
        /// <param name="nLabel">Specifies the known label of the data.</param>
        /// <param name="dtTime">Specifies a time-stamp associated with the data.</param>
        /// <param name="rgfData">Specifies the data as a list of <i>double</i> (expects <i>bIsReal</i> = <i>true</i>).</param>
        /// <param name="nBoost">Specifies the boost to use with the data (a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Specifies whether or not the label was auto-generated.</param>
        /// <param name="nIdx">Specifies the index of the data.</param>
        /// <param name="nVirtualID">Specifies a virtual index for the data (default = 0).  When specified, the SimpleDatum is used to reference another.</param>
        /// <param name="nImageID">Specifies the image ID within the database.</param>
        /// <param name="nSourceID">Specifies the data source ID of the data source that owns this image.</param>
        /// <param name="nOriginalSourceID">Optionally, specifies the ogiginal source ID which is set when using a virtual ID - the original source ID is the ID of the source associated with the image with the ID of the virtual ID.</param>
        public SimpleDatum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, List<double> rgfData, int nBoost, bool bAutoLabeled, int nIdx, int nVirtualID = 0, int nImageID = 0, int nSourceID = 0, int nOriginalSourceID = 0)
        {
            m_nChannels = nChannels;
            m_nWidth = nWidth;
            m_nHeight = nHeight;
            m_nOriginalLabel = nLabel;
            m_nLabel = nLabel;
            m_dt = dtTime;
            m_nOriginalBoost = nBoost;
            m_nBoost = nBoost;
            m_bAutoLabeled = bAutoLabeled;
            m_nVirtualID = nVirtualID;
            m_bIsRealData = bIsReal;
            m_nIndex = nIdx;
            m_nImageID = nImageID;
            m_nSourceID = nSourceID;
            m_nOriginalSourceID = nOriginalSourceID;

            if (!bIsReal)
                throw new ArgumentException("The data sent is real, but the bIsReal is set to false!");

            m_rgRealDataD = rgfData.ToArray();
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="bIsReal">Specifies whether or not the data values are <i>double</i> or <i>byte</i>.</param>
        /// <param name="nChannels">Specifies the number of channels in the data (e.g. 3 for color, 1 for black and white images)</param>
        /// <param name="nWidth">Specifies the width of the data (e.g. the number of pixels wide).</param>
        /// <param name="nHeight">Specifies the height of the data (e.g. the number of pixels high).</param>
        /// <param name="nLabel">Specifies the known label of the data.</param>
        /// <param name="dtTime">Specifies a time-stamp associated with the data.</param>
        /// <param name="rgfData">Specifies the data as a list of <i>float</i> (expects <i>bIsReal</i> = <i>true</i>).</param>
        /// <param name="nBoost">Specifies the boost to use with the data (a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Specifies whether or not the label was auto-generated.</param>
        /// <param name="nIdx">Specifies the index of the data.</param>
        /// <param name="nVirtualID">Specifies a virtual index for the data (default = 0).  When specified, the SimpleDatum is used to reference another.</param>
        /// <param name="nImageID">Specifies the image ID within the database.</param>
        /// <param name="nSourceID">Specifies the data source ID of the data source that owns this image.</param>
        /// <param name="nOriginalSourceID">Optionally, specifies the ogiginal source ID which is set when using a virtual ID - the original source ID is the ID of the source associated with the image with the ID of the virtual ID.</param>
        public SimpleDatum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, List<float> rgfData, int nBoost, bool bAutoLabeled, int nIdx, int nVirtualID = 0, int nImageID = 0, int nSourceID = 0, int nOriginalSourceID = 0)
        {
            m_nChannels = nChannels;
            m_nWidth = nWidth;
            m_nHeight = nHeight;
            m_nOriginalLabel = nLabel;
            m_nLabel = nLabel;
            m_dt = dtTime;
            m_nOriginalBoost = nBoost;
            m_nBoost = nBoost;
            m_bAutoLabeled = bAutoLabeled;
            m_nVirtualID = nVirtualID;
            m_bIsRealData = bIsReal;
            m_nIndex = nIdx;
            m_nImageID = nImageID;
            m_nSourceID = nSourceID;
            m_nOriginalSourceID = nOriginalSourceID;

            if (!bIsReal)
                throw new ArgumentException("The data sent is real, but the bIsReal is set to false!");

            m_rgRealDataF = rgfData.ToArray();
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="bIsReal">Specifies whether or not the data values are <i>double</i> or <i>byte</i>.</param>
        /// <param name="nChannels">Specifies the number of channels in the data (e.g. 3 for color, 1 for black and white images)</param>
        /// <param name="nWidth">Specifies the width of the data (e.g. the number of pixels wide).</param>
        /// <param name="nHeight">Specifies the height of the data (e.g. the number of pixels high).</param>
        /// <param name="nLabel">Specifies the known label of the data.</param>
        /// <param name="dtTime">Specifies a time-stamp associated with the data.</param>
        /// <param name="rgData">Specifies the data as a list of <i>bytes</i> (expects <i>bIsReal</i> = <i>false</i>).</param>
        /// <param name="nBoost">Specifies the boost to use with the data (a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Specifies whether or not the label was auto-generated.</param>
        /// <param name="nIdx">Specifies the index of the data.</param>
        /// <param name="nVirtualID">Specifies a virtual index for the data (default = 0).  When specified, the SimpleDatum is used to reference another.</param>
        /// <param name="nImageID">Specifies the image ID within the database.</param>
        /// <param name="nSourceID">Specifies the data source ID of the data source that owns this image.</param>
        /// <param name="nOriginalSourceID">Optionally, specifies the ogiginal source ID which is set when using a virtual ID - the original source ID is the ID of the source associated with the image with the ID of the virtual ID.</param>
        public SimpleDatum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, byte[] rgData, int nBoost, bool bAutoLabeled, int nIdx, int nVirtualID = 0, int nImageID = 0, int nSourceID = 0, int nOriginalSourceID = 0)
        {
            m_nChannels = nChannels;
            m_nWidth = nWidth;
            m_nHeight = nHeight;
            m_nOriginalLabel = nLabel;
            m_nLabel = nLabel;
            m_dt = dtTime;
            m_nOriginalBoost = nBoost;
            m_nBoost = nBoost;
            m_bAutoLabeled = bAutoLabeled;
            m_nVirtualID = nVirtualID;
            m_bIsRealData = bIsReal;
            m_nIndex = nIdx;
            m_nImageID = nImageID;
            m_nSourceID = nSourceID;
            m_nOriginalSourceID = nOriginalSourceID;

            if (bIsReal)
                throw new ArgumentException("The data sent is not real, but the bIsReal is set to true!");

            m_rgByteData = rgData;
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="bIsReal">Specifies whether or not the data values are <i>double</i> or <i>byte</i>.</param>
        /// <param name="nChannels">Specifies the number of channels in the data (e.g. 3 for color, 1 for black and white images)</param>
        /// <param name="nWidth">Specifies the width of the data (e.g. the number of pixels wide).</param>
        /// <param name="nHeight">Specifies the height of the data (e.g. the number of pixels high).</param>
        /// <param name="nLabel">Specifies the known label of the data.</param>
        /// <param name="dtTime">Specifies a time-stamp associated with the data.</param>
        /// <param name="rgdfData">Specifies the data as a list of <i>double</i> (expects <i>bIsReal</i> = <i>true</i>).</param>
        /// <param name="nBoost">Specifies the boost to use with the data (a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Specifies whether or not the label was auto-generated.</param>
        /// <param name="nIdx">Specifies the index of the data.</param>
        /// <param name="nVirtualID">Specifies a virtual index for the data (default = 0).  When specified, the SimpleDatum is used to reference another.</param>
        /// <param name="nImageID">Specifies the image ID within the database.</param>
        /// <param name="nSourceID">Specifies the data source ID of the data source that owns this image.</param>
        /// <param name="nOriginalSourceID">Optionally, specifies the ogiginal source ID which is set when using a virtual ID - the original source ID is the ID of the source associated with the image with the ID of the virtual ID.</param>
        public SimpleDatum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, double[] rgdfData, int nBoost, bool bAutoLabeled, int nIdx, int nVirtualID = 0, int nImageID = 0, int nSourceID = 0, int nOriginalSourceID = 0)
        {
            m_nChannels = nChannels;
            m_nWidth = nWidth;
            m_nHeight = nHeight;
            m_nOriginalLabel = nLabel;
            m_nLabel = nLabel;
            m_dt = dtTime;
            m_nOriginalBoost = nBoost;
            m_nBoost = nBoost;
            m_bAutoLabeled = bAutoLabeled;
            m_nVirtualID = nVirtualID;
            m_bIsRealData = bIsReal;
            m_nIndex = nIdx;
            m_nImageID = nImageID;
            m_nSourceID = nSourceID;
            m_nOriginalSourceID = nOriginalSourceID;

            if (!bIsReal)
                throw new ArgumentException("The data sent is real, but the bIsReal is set to false!");

            m_rgRealDataD = rgdfData;
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="bIsReal">Specifies whether or not the data values are <i>double</i> or <i>byte</i>.</param>
        /// <param name="nChannels">Specifies the number of channels in the data (e.g. 3 for color, 1 for black and white images)</param>
        /// <param name="nWidth">Specifies the width of the data (e.g. the number of pixels wide).</param>
        /// <param name="nHeight">Specifies the height of the data (e.g. the number of pixels high).</param>
        /// <param name="nLabel">Specifies the known label of the data.</param>
        /// <param name="dtTime">Specifies a time-stamp associated with the data.</param>
        /// <param name="rgfData">Specifies the data as a list of <i>float</i> (expects <i>bIsReal</i> = <i>true</i>).</param>
        /// <param name="nBoost">Specifies the boost to use with the data (a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Specifies whether or not the label was auto-generated.</param>
        /// <param name="nIdx">Specifies the index of the data.</param>
        /// <param name="nVirtualID">Specifies a virtual index for the data (default = 0).  When specified, the SimpleDatum is used to reference another.</param>
        /// <param name="nImageID">Specifies the image ID within the database.</param>
        /// <param name="nSourceID">Specifies the data source ID of the data source that owns this image.</param>
        /// <param name="nOriginalSourceID">Optionally, specifies the ogiginal source ID which is set when using a virtual ID - the original source ID is the ID of the source associated with the image with the ID of the virtual ID.</param>
        public SimpleDatum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, float[] rgfData, int nBoost, bool bAutoLabeled, int nIdx, int nVirtualID = 0, int nImageID = 0, int nSourceID = 0, int nOriginalSourceID = 0)
        {
            m_nChannels = nChannels;
            m_nWidth = nWidth;
            m_nHeight = nHeight;
            m_nOriginalLabel = nLabel;
            m_nLabel = nLabel;
            m_dt = dtTime;
            m_nOriginalBoost = nBoost;
            m_nBoost = nBoost;
            m_bAutoLabeled = bAutoLabeled;
            m_nVirtualID = nVirtualID;
            m_bIsRealData = bIsReal;
            m_nIndex = nIdx;
            m_nImageID = nImageID;
            m_nSourceID = nSourceID;
            m_nOriginalSourceID = nOriginalSourceID;

            if (!bIsReal)
                throw new ArgumentException("The data sent is real, but the bIsReal is set to false!");

            m_rgRealDataF = rgfData;
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="bIsReal">Specifies whether or not the data values are <i>double</i> or <i>byte</i>.</param>
        /// <param name="nChannels">Specifies the number of channels in the data (e.g. 3 for color, 1 for black and white images)</param>
        /// <param name="nWidth">Specifies the width of the data (e.g. the number of pixels wide).</param>
        /// <param name="nHeight">Specifies the height of the data (e.g. the number of pixels high).</param>
        /// <param name="nLabel">Specifies the known label of the data.</param>
        /// <param name="dtTime">Specifies a time-stamp associated with the data.</param>
        /// <param name="nBoost">Specifies the boost to use with the data (a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Specifies whether or not the label was auto-generated.</param>
        /// <param name="nIdx">Specifies the index of the data.</param>
        /// <param name="nVirtualID">Specifies a virtual index for the data (default = 0).  When specified, the SimpleDatum is used to reference another.</param>
        /// <param name="nImageID">Specifies the image ID within the database.</param>
        /// <param name="nSourceID">Specifies the data source ID of the data source that owns this image.</param>
        /// <param name="nOriginalSourceID">Optionally, specifies the ogiginal source ID which is set when using a virtual ID - the original source ID is the ID of the source associated with the image with the ID of the virtual ID.</param>
        public SimpleDatum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, int nBoost = 0, bool bAutoLabeled = false, int nIdx = -1, int nVirtualID = 0, int nImageID = 0, int nSourceID = 0, int nOriginalSourceID = 0)
        {
            m_nChannels = nChannels;
            m_nWidth = nWidth;
            m_nHeight = nHeight;
            m_nOriginalLabel = nLabel;
            m_nLabel = nLabel;
            m_dt = dtTime;
            m_nOriginalBoost = nBoost;
            m_nBoost = nBoost;
            m_bAutoLabeled = bAutoLabeled;
            m_nVirtualID = nVirtualID;
            m_bIsRealData = bIsReal;
            m_nIndex = nIdx;
            m_nImageID = nImageID;
            m_nSourceID = nSourceID;
            m_nOriginalSourceID = nOriginalSourceID;
            m_rgByteData = null;
            m_rgRealDataD = null;
            m_rgRealDataF = null;
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="nChannels">Specifies the number of channels in the data (e.g. 3 for color, 1 for black and white images)</param>
        /// <param name="nWidth">Specifies the width of the data (e.g. the number of pixels wide).</param>
        /// <param name="nHeight">Specifies the height of the data (e.g. the number of pixels high).</param>
        /// <param name="rgf">Specifies the data to copy.</param>
        /// <param name="nOffset">Specifies the offset into the data where the copying should start.</param>
        /// <param name="nCount">Specifies the number of data items to copy.</param>
        /// <param name="bDataIsReal">Optionally, specifies whether or not the data is real.</param>
        public SimpleDatum(int nChannels, int nWidth, int nHeight, float[] rgf, int nOffset, int nCount, bool bDataIsReal = true)
        {
            m_nChannels = nChannels;
            m_nWidth = nWidth;
            m_nHeight = nHeight;
            m_nOriginalLabel = 0;
            m_nOriginalBoost = 0;
            m_dt = DateTime.MinValue;
            m_bAutoLabeled = false;
            m_nVirtualID = 0;
            m_nImageID = 0;
            m_nImageID = 0;
            m_nSourceID = 0;
            m_nOriginalSourceID = 0;
            m_rgByteData = null;

            int nLen = nChannels * nWidth * nHeight;
            if (nLen != nCount)
                throw new Exception("The channel x width x height should equal the count!");

            if (bDataIsReal)
            {
                m_bIsRealData = true;
                m_rgRealDataF = new float[nCount];
                Array.Copy(rgf, nOffset, m_rgRealDataF, 0, nCount);
            }
            else
            {
                m_bIsRealData = false;
                m_rgByteData = new byte[nCount];

                for (int i = 0; i < nCount; i++)
                {
                    m_rgByteData[i] = Math.Min(Math.Max((byte)rgf[nOffset + i], (byte)0), (byte)255);
                }
            }
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="nChannels">Specifies the number of channels in the data (e.g. 3 for color, 1 for black and white images)</param>
        /// <param name="nWidth">Specifies the width of the data (e.g. the number of pixels wide).</param>
        /// <param name="nHeight">Specifies the height of the data (e.g. the number of pixels high).</param>
        public SimpleDatum(int nChannels, int nWidth, int nHeight)
        {
            m_nChannels = nChannels;
            m_nWidth = nWidth;
            m_nHeight = nHeight;
            m_nOriginalLabel = 0;
            m_nOriginalBoost = 0;
            m_dt = DateTime.MinValue;
            m_bAutoLabeled = false;
            m_nVirtualID = 0;
            m_nImageID = 0;
            m_nImageID = 0;
            m_nSourceID = 0;
            m_nOriginalSourceID = 0;
            m_rgByteData = null;
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="data">Specifies the byte data to fill the SimpleDatum with.</param>
        public SimpleDatum(Bytemap data)
        {
            m_nChannels = data.Channels;
            m_nWidth = data.Width;
            m_nHeight = data.Height;
            m_nOriginalLabel = -1;
            m_nLabel = -1;
            m_bIsRealData = false;

            m_rgByteData = data.Bytes;
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="data">Specifies the valuse data to fill the SimpleDatum with.</param>
        public SimpleDatum(Valuemap data)
        {
            m_nChannels = data.Channels;
            m_nWidth = data.Width;
            m_nHeight = data.Height;
            m_nOriginalLabel = -1;
            m_nLabel = -1;
            m_bIsRealData = true;

            m_rgRealDataD = data.Values;
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="d">Specifies a SimpleDatum used to create this new Datum.</param>
        /// <param name="bCopyData">Specifies whether or not to copy the data, or just share it (default = false, share the data).</param>
        public SimpleDatum(SimpleDatum d, bool bCopyData = false)
        {
            Copy(d, bCopyData);
        }

        /// <summary>
        /// The SimpleDatum constructor.
        /// </summary>
        /// <param name="d">Specifies a SimpleDatum used to create this new Datum.</param>
        /// <param name="nHeight">Specifies a height override.</param>
        /// <param name="nWidth">Specifies a width override.</param>
        public SimpleDatum(SimpleDatum d, int nHeight, int nWidth)
        {
            Copy(d, false, nHeight, nWidth);
        }

        /// <summary>
        /// Constructor that copies an array into a single SimpleDatum by appending each to the other in order.
        /// </summary>
        /// <remarks>
        /// Data is ordered by HxWxC where C is filled with the channels of each input.  So if three inputs
        /// are used the output is HxWx[c1,c2,c3].
        /// </remarks>
        /// <param name="rg">Specifies the array of SimpleDatum to append together.</param>
        /// <param name="bAlignChannels">When true, the data is packed with each channel following the other.  For example,
        /// packing three hxw images together using channel ordering, so three single channel images would then result
        /// in the following ordering:
        /// 
        /// h0,w0,c0,c1,c2  h0,w1,c0,c1,c2 ...
        /// 
        /// When false (default), the channel data from each data item are stacked back to back similar to the way a single
        /// data item is already ordered.
        /// </param>
        public SimpleDatum(List<SimpleDatum> rg, bool bAlignChannels)
        {
            if (rg.Count == 1)
            {
                Copy(rg[0], true);
                return;
            }

            Copy(rg[0], false);

            m_nChannels *= rg.Count;

            if (bAlignChannels)
            {
                if (m_nChannels != rg.Count)
                    throw new Exception("Currently channel alignment is only allowed on single channel data.");

                if (rg.Count >= 1)
                {
                    if (rg[0].IsRealData)
                    {
                        if (rg[0].RealDataD != null)
                        {
                            double[] rgData = new double[m_nChannels * Height * Width];

                            for (int h = 0; h < Height; h++)
                            {
                                for (int w = 0; w < Width; w++)
                                {
                                    int nIdxSrc = (h * Width) + w;
                                    int nIdxDst = nIdxSrc * m_nChannels;

                                    for (int c = 0; c < m_nChannels; c++)
                                    {
                                        rgData[nIdxDst + c] = rg[c].RealDataD[nIdxSrc];
                                    }
                                }
                            }

                            m_rgRealDataD = rgData;
                        }
                        else if (rg[0].RealDataF != null)
                        {
                            float[] rgData = new float[m_nChannels * Height * Width];

                            for (int h = 0; h < Height; h++)
                            {
                                for (int w = 0; w < Width; w++)
                                {
                                    int nIdxSrc = (h * Width) + w;
                                    int nIdxDst = nIdxSrc * m_nChannels;

                                    for (int c = 0; c < m_nChannels; c++)
                                    {
                                        rgData[nIdxDst + c] = rg[c].RealDataF[nIdxSrc];
                                    }
                                }
                            }

                            m_rgRealDataF = rgData;
                        }
                        else
                        {
                            throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                        }
                        m_rgByteData = null;
                    }
                    else
                    {
                        byte[] rgData = new byte[m_nChannels * Height * Width];

                        for (int h = 0; h < Height; h++)
                        {
                            for (int w = 0; w < Width; w++)
                            {
                                int nIdxSrc = (h * Width) + w;
                                int nIdxDst = nIdxSrc * m_nChannels;

                                for (int c = 0; c < m_nChannels; c++)
                                {
                                    rgData[nIdxDst + c] = rg[c].ByteData[nIdxSrc];
                                }
                            }
                        }

                        m_rgByteData = rgData;
                        m_rgRealDataF = null;
                        m_rgRealDataD = null;
                    }
                }
            }
            else
            {
                bool bReal = rg[0].IsRealData;
                int nDstIdx = 0;
                int nItemCount = m_nChannels / rg.Count;

                if (bReal)
                {
                    if (rg[0].RealDataD != null)
                    {
                        m_rgRealDataD = new double[m_nChannels * Height * Width];
                        m_rgByteData = null;

                        for (int i = 0; i < rg.Count; i++)
                        {
                            Array.Copy(rg[i].RealDataD, 0, m_rgRealDataD, nDstIdx, nItemCount);
                            nDstIdx += nItemCount;
                        }
                    }
                    else if (rg[0].RealDataF != null)
                    {
                        m_rgRealDataF = new float[m_nChannels * Height * Width];
                        m_rgByteData = null;

                        for (int i = 0; i < rg.Count; i++)
                        {
                            Array.Copy(rg[i].RealDataF, 0, m_rgRealDataF, nDstIdx, nItemCount);
                            nDstIdx += nItemCount;
                        }
                    }
                    else
                    {
                        throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                    }
                }
                else
                {
                    m_rgByteData = new byte[m_nChannels * Height * Width];
                    m_rgRealDataD = null;
                    m_rgRealDataF = null;

                    for (int i = 0; i < rg.Count; i++)
                    {
                        Array.Copy(rg[i].ByteData, 0, m_rgByteData, nDstIdx, nItemCount);
                        nDstIdx += nItemCount;
                    }
                }
            }
        }

        /// <summary>
        /// Get/set the hit count for the SimpleDatum.
        /// </summary>
        public int HitCount
        {
            get { return m_nHitCount; }
            set { m_nHitCount = value; }
        }

        /// <summary>
        /// Specifies user data associated with the SimpleDatum.
        /// </summary>
        public object Tag
        {
            get { return m_tag; }
            set { m_tag = value; }
        }

        /// <summary>
        /// Returns a user-defined name of the tag associated with the data.
        /// </summary>
        public string TagName
        {
            get { return m_strTagName; }
            set { m_strTagName = value; }
        }

        /// <summary>
        /// Returns the minimum value in the data or double.NaN if there is no data.
        /// </summary>
        public double Min
        {
            get
            {
                if (m_bIsRealData)
                {
                    if (m_rgRealDataD != null)
                        return m_rgRealDataD.Min(p => p);
                    else if (m_rgRealDataF != null)
                        return m_rgRealDataF.Min(p => p);
                    else
                        throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                }
                else
                {
                    if (m_rgByteData != null)
                        return (double)m_rgByteData.Min(p => p);
                }

                return double.NaN;
            }
        }

        /// <summary>
        /// Returns the maximum value in the data or double.NaN if there is no data.
        /// </summary>
        public double Max
        {
            get
            {
                if (m_bIsRealData)
                {
                    if (m_rgRealDataD != null)
                        return m_rgRealDataD.Max(p => p);
                    else if (m_rgRealDataF != null)
                        return m_rgRealDataF.Max(p => p);
                    else
                        throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                }
                else
                {
                    if (m_rgByteData != null)
                        return (double)m_rgByteData.Max(p => p);
                }

                return double.NaN;
            }
        }

        /// <summary>
        /// Clip the data length down to a smaller size and copies the clipped data.
        /// </summary>
        /// <param name="nDataLen">Specifies the new, smaller, size.</param>
        /// <param name="nNewChannel">Specifies the new channel size, or null to ignore.</param>
        /// <param name="nNewHeight">Specifies the new height size, or null to ignore.</param>
        /// <param name="nNewWidth">Specifies the new width size, or null to ignore.</param>
        public void Clip(int nDataLen, int? nNewChannel, int? nNewHeight, int? nNewWidth)
        {           
            if (m_rgByteData != null && m_rgByteData.Length > nDataLen)
            {
                byte[] rgData = new byte[nDataLen];
                Array.Copy(m_rgByteData, rgData, nDataLen);
                m_rgByteData = rgData;
            }
            if (m_rgRealDataD != null && m_rgRealDataD.Length > nDataLen)
            {
                double[] rgData = new double[nDataLen];
                Array.Copy(m_rgRealDataD, rgData, nDataLen);
                m_rgRealDataD = rgData;
            }
            if (m_rgRealDataF != null && m_rgRealDataF.Length > nDataLen)
            {
                float[] rgData = new float[nDataLen];
                Array.Copy(m_rgRealDataF, rgData, nDataLen);
                m_rgRealDataF = rgData;
            }

            m_nChannels = nNewChannel.GetValueOrDefault(m_nChannels);
            m_nHeight = nNewHeight.GetValueOrDefault(m_nHeight);
            m_nWidth = nNewWidth.GetValueOrDefault(m_nWidth);
        }

        /// <summary>
        /// Returns all indexes with non-zero data.
        /// </summary>
        /// <returns>The list of indexes corresponding to non-zero data is returned.</returns>
        public List<int> GetNonZeroIndexes()
        {
            List<int> rgIdx = new List<int>();

            if (m_bIsRealData)
            {
                if (m_rgRealDataD != null)
                {
                    for (int i = 0; i < m_rgRealDataD.Length; i++)
                    {
                        if (m_rgRealDataD[i] != 0)
                            rgIdx.Add(i);
                    }
                }
                else if (m_rgRealDataF != null)
                {
                    for (int i = 0; i < m_rgRealDataF.Length; i++)
                    {
                        if (m_rgRealDataF[i] != 0)
                            rgIdx.Add(i);
                    }
                }
                else
                {
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                }
            }
            else
            {
                for (int i = 0; i < m_rgByteData.Length; i++)
                {
                    if (m_rgByteData[i] != 0)
                        rgIdx.Add(i);
                }
            }

            return rgIdx;
        }

        /// <summary>
        /// Zero out all data in the datum but keep the size and other settings.
        /// </summary>
        public void Zero()
        {
            if (m_rgByteData != null)
                Array.Clear(m_rgByteData, 0, m_rgByteData.Length);

            if (m_rgRealDataD != null)
                Array.Clear(m_rgRealDataD, 0, m_rgRealDataD.Length);

            if (m_rgRealDataF != null)
                Array.Clear(m_rgRealDataF, 0, m_rgRealDataF.Length);
        }

        /// <summary>
        /// Subtract the data of another SimpleDatum from this one, so this = this - sd.
        /// </summary>
        /// <param name="sd">Specifies the other SimpleDatum to subtract.</param>
        /// <returns>If both data values are different <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Sub(SimpleDatum sd)
        {
            bool bDifferent = false;

            if (sd.ItemCount != ItemCount)
                throw new Exception("Both simple datums must have the same number of elements!");

            if (m_rgByteData != null)
            {
                if (sd.m_rgByteData == null)
                    throw new Exception("Both simple datums must have the same type of data!");

                for (int i = 0; i < m_rgByteData.Length; i++)
                {
                    m_rgByteData[i] -= sd.m_rgByteData[i];
                    if (m_rgByteData[i] != 0)
                        bDifferent = true;
                }
            }

            if (m_rgRealDataD != null)
            {
                if (sd.m_rgRealDataD == null)
                    throw new Exception("Both simple datums must have the same type of data!");

                for (int i = 0; i < m_rgRealDataD.Length; i++)
                {
                    m_rgRealDataD[i] -= sd.m_rgRealDataD[i];
                    if (m_rgRealDataD[i] != 0)
                        bDifferent = true;
                }
            }

            if (m_rgRealDataF != null)
            {
                if (sd.m_rgRealDataF == null)
                    throw new Exception("Both simple datums must have the same type of data!");

                for (int i = 0; i < m_rgRealDataF.Length; i++)
                {
                    m_rgRealDataF[i] -= sd.m_rgRealDataF[i];
                    if (m_rgRealDataF[i] != 0)
                        bDifferent = true;
                }
            }

            return bDifferent;
        }

        /// <summary>
        /// Subtract the data of another SimpleDatum from this one, and take the absolute value, so this = Math.Abs(this - sd).
        /// </summary>
        /// <param name="sd">Specifies the other SimpleDatum to subtract.</param>
        /// <returns>If both data values are different <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool SubAbs(SimpleDatum sd)
        {
            bool bDifferent = false;

            if (sd.ItemCount != ItemCount)
                throw new Exception("Both simple datums must have the same number of elements!");

            if (m_rgByteData != null)
            {
                if (sd.m_rgByteData == null)
                    throw new Exception("Both simple datums must have the same type of data!");

                for (int i = 0; i < m_rgByteData.Length; i++)
                {
                    m_rgByteData[i] = (byte)Math.Abs(m_rgByteData[i] - sd.m_rgByteData[i]);
                    if (m_rgByteData[i] != 0)
                        bDifferent = true;
                }
            }

            if (m_rgRealDataD != null)
            {
                if (sd.m_rgRealDataD == null)
                    throw new Exception("Both simple datums must have the same type of data!");

                for (int i = 0; i < m_rgRealDataD.Length; i++)
                {
                    m_rgRealDataD[i] = Math.Abs(m_rgRealDataD[i] - sd.m_rgRealDataD[i]);
                    if (m_rgRealDataD[i] != 0)
                        bDifferent = true;
                }
            }

            if (m_rgRealDataF != null)
            {
                if (sd.m_rgRealDataF == null)
                    throw new Exception("Both simple datums must have the same type of data!");

                for (int i = 0; i < m_rgRealDataF.Length; i++)
                {
                    m_rgRealDataF[i] = Math.Abs(m_rgRealDataF[i] - sd.m_rgRealDataF[i]);
                    if (m_rgRealDataF[i] != 0)
                        bDifferent = true;
                }
            }

            return bDifferent;
        }

        /// <summary>
        /// Copy another SimpleDatum into this one.
        /// </summary>
        /// <param name="d">Specifies the SimpleDatum to copy.</param>
        /// <param name="bCopyData">Specifies whether or not to copy the data.</param>
        /// <param name="nHeight">Optionally, specifies a height override.</param>
        /// <param name="nWidth">Optionally, specifies a width override.</param>
        public void Copy(SimpleDatum d, bool bCopyData, int? nHeight = null, int? nWidth = null)
        {
            m_bIsRealData = d.m_bIsRealData;
            m_nLabel = d.m_nLabel;
            m_nOriginalLabel = d.m_nOriginalLabel;
            m_nChannels = d.m_nChannels;
            m_nHeight = nHeight.GetValueOrDefault(d.m_nHeight);
            m_nWidth = nWidth.GetValueOrDefault(d.m_nWidth);

            if (bCopyData)
            {
                m_rgRealDataD = Utility.Clone<double>(d.m_rgRealDataD);
                m_rgRealDataF = Utility.Clone<float>(d.m_rgRealDataF);
                m_rgByteData = Utility.Clone<byte>(d.m_rgByteData);
            }
            else
            {
                m_rgRealDataD = d.m_rgRealDataD;
                m_rgRealDataF = d.m_rgRealDataF;
                m_rgByteData = d.m_rgByteData;
            }

            m_dt = d.m_dt;
            m_nOriginalBoost = d.m_nOriginalBoost;
            m_nBoost = d.m_nBoost;
            m_bAutoLabeled = d.m_bAutoLabeled;
            m_nImageID = d.m_nImageID;
            m_nVirtualID = d.m_nVirtualID;
            m_nGroupID = d.m_nGroupID;
            m_nIndex = d.m_nIndex;
            m_strDesc = d.m_strDesc;
            m_rgDataCriteria = d.m_rgDataCriteria;
            m_dataCriteriaFormat = d.m_dataCriteriaFormat;
            m_rgDebugData = d.m_rgDebugData;
            m_debugDataFormat = d.m_debugDataFormat;
            m_nSourceID = d.m_nSourceID;
            m_nOriginalSourceID = d.m_nOriginalSourceID;
            m_tag = d.m_tag;
            m_strTagName = d.m_strTagName;

            m_nAnnotationType = d.m_nAnnotationType;
            m_rgAnnotationGroup = null;

            if (d.m_rgAnnotationGroup != null)
            {
                m_rgAnnotationGroup = new AnnotationGroupCollection();

                foreach (AnnotationGroup g in d.m_rgAnnotationGroup)
                {
                    m_rgAnnotationGroup.Add(g.Clone());
                }
            }
        }

        /// <summary>
        /// Copy just the data from another SimpleDatum, making sure to update the C x H x W dimensions and IsReal settings to fit the new data.
        /// </summary>
        /// <param name="d">Specifies the SimpleDatum whos data is to be copied.</param>
        public void CopyData(SimpleDatum d)
        {
            m_nHeight = d.m_nHeight;
            m_nWidth = d.m_nWidth;
            m_nChannels = d.m_nChannels;
            m_rgByteData = d.m_rgByteData;
            m_bIsRealData = d.m_bIsRealData;
            m_rgRealDataD = d.m_rgRealDataD;
            m_rgRealDataF = d.m_rgRealDataF;
        }

        /// <summary>
        /// Set the data of the current SimpleDatum by copying the data of another.
        /// </summary>
        /// <param name="d">Specifies the SimpleDatum to copy.</param>
        public void SetData(SimpleDatum d)
        {
            if (d.IsRealData)
            {
                if (d.RealDataD != null)
                    SetData(d.RealDataD.ToList(), d.Label);
                else if (d.RealDataF != null)
                    SetData(d.RealDataF.ToList(), d.Label);
                else
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
            }
            else
                SetData(d.ByteData.ToList(), d.Label);
        }

        /// <summary>
        /// Returns <i>true</i> if the ByteData or RealDataD or RealDataF are not null, <i>false</i> otherwise.
        /// </summary>
        /// <param name="bByType">Specifies to only test if real (RealDataD and RealDataF) or if not real (ByteData).  When false, all data types are tested.</param>
        public bool GetDataValid(bool bByType = true)
        {
            if (bByType)
            {
                if (m_bIsRealData && m_rgRealDataD == null && m_rgRealDataF == null)
                    return false;
                else if (!m_bIsRealData && m_rgByteData == null)
                    return false;

                return true;
            }

            if (m_rgRealDataD == null && m_rgRealDataF == null && m_rgByteData == null)
                return false;

            return true;
        }

        /// <summary>
        /// DEPRECIATED: Clips the SimpleDatum to the last <i>nLastColumns</i> and returns the data.
        /// </summary>
        /// <typeparam name="T">Specifies base the type of data returned, either <i>double</i> or <i>float</i>.</typeparam>
        /// <param name="nLastColumns">Specifies the number of last columns of data to keep.</param>
        /// <returns>The last columns of data are returned.</returns>
        public List<T> ClipToLastColumnsX<T>(int nLastColumns = 10)
        {
            int nC = m_nChannels;
            int nW = m_nWidth;
            int nH = m_nHeight;
            int nXStart = nW - nLastColumns;
            List<T> rg = new List<T>();

            if (nXStart < 0)
                nXStart = 0;

            for (int c = 0; c < nC; c++)
            {
                for (int y = 0; y < nH; y++)
                {
                    for (int x = nXStart; x < nW; x++)
                    {
                        int nIdxImg = (c * (nH * nW)) + (y * nW) + x;
                        rg.Add((T)Convert.ChangeType(ByteData[nIdxImg], typeof(T)));
                    }
                }
            }

            return rg;
        }

        /// <summary>
        /// DEPRECIATED: Masks out all data except for the last columns of data.
        /// </summary>
        /// <param name="nLastColumsToRetain">Specifies the number of last columns to retain.</param>
        /// <param name="nMaskingValue">Specifies the value to use for the masked columns.</param>
        public void MaskOutAllButLastColumnsX(int nLastColumsToRetain, int nMaskingValue)
        {
            if (nLastColumsToRetain <= 0 || nLastColumsToRetain >= m_nWidth)
                return;

            int nC = m_nChannels;
            int nW = m_nWidth;
            int nH = m_nHeight;
            int nMaskWid = nW - nLastColumsToRetain;

            for (int c = 0; c < nC; c++)
            {
                for (int y = 0; y < nH; y++)
                {
                    for (int x = 0; x < nMaskWid; x++)
                    {
                        int nIdxImg = (c * (nH * nW)) + (y * nW) + x;
                        ByteData[nIdxImg] = (byte)nMaskingValue;
                    }
                }
            }
        }

        /// <summary>
        /// Sets the <i>byte</i> data of the SimpleDatum and its Label.
        /// </summary>
        /// <param name="rgByteData">Specifies the <i>byte</i> data.</param>
        /// <param name="bAllowVirtualOverride">Optionally, allow virtual ID override.  When <i>true</i> the data can be set on a virtual SimpleDatum, otherwise it cannot.</param>
        /// <param name="nLabel">Specifies the label.</param>
        public void SetData(List<byte> rgByteData, int nLabel, bool bAllowVirtualOverride = false)
        {
            if (!bAllowVirtualOverride && m_nVirtualID != 0)
                throw new Exception("Cannot set the data of a virtual item!");

            m_nVirtualID = 0;
            m_bIsRealData = false;
            m_rgByteData = rgByteData.ToArray();
            m_rgRealDataD = null;
            m_rgRealDataF = null;
            m_nLabel = nLabel;
        }

        /// <summary>
        /// Sets the <i>double</i> data of the SimpleDatum and its Label.
        /// </summary>
        /// <param name="rgRealData">Specifies the <i>double</i> data.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="bAllowVirtualOverride">Optionally, allow virtual ID override.  When <i>true</i> the data can be set on a virtual SimpleDatum, otherwise it cannot.</param>
        public void SetData(List<double> rgRealData, int nLabel, bool bAllowVirtualOverride = false)
        {
            if (!bAllowVirtualOverride && m_nVirtualID != 0)
                throw new Exception("Cannot set the data of a virtual item!");

            m_nVirtualID = 0;
            m_bIsRealData = true;
            m_rgByteData = null;
            m_rgRealDataF = null;
            m_rgRealDataD = rgRealData.ToArray();
            m_nLabel = nLabel;
        }

        /// <summary>
        /// Sets the <i>float</i> data of the SimpleDatum and its Label.
        /// </summary>
        /// <param name="rgRealData">Specifies the <i>float</i> data.</param>
        /// <param name="nLabel">Specifies the label.</param>
        /// <param name="bAllowVirtualOverride">Optionally, allow virtual ID override.  When <i>true</i> the data can be set on a virtual SimpleDatum, otherwise it cannot.</param>
        public void SetData(List<float> rgRealData, int nLabel, bool bAllowVirtualOverride = false)
        {
            if (!bAllowVirtualOverride && m_nVirtualID != 0)
                throw new Exception("Cannot set the data of a virtual item!");

            m_nVirtualID = 0;
            m_bIsRealData = true;
            m_rgByteData = null;
            m_rgRealDataD = null;
            m_rgRealDataF = rgRealData.ToArray();
            m_nLabel = nLabel;
        }

        /// <summary>
        /// Set the data to the <i>byte</i> array specified.
        /// </summary>
        /// <param name="rgb">Specifies the data to set.</param>
        /// <param name="nLabel">Specifies the label to set.</param>
        /// <remarks>
        /// The data of the array is cast to either (double) for real data, or (byte) for the byte data.
        /// </remarks>
        public void SetData(byte[] rgb, int nLabel)
        {
            m_nLabel = nLabel;
            m_rgByteData = new byte[rgb.Length];
            Array.Copy(rgb, m_rgByteData, rgb.Length);
            m_bIsRealData = false;
            m_rgRealDataD = null;
            m_rgRealDataF = null;
        }

        /// <summary>
        /// Set the data to the <i>double</i> array specified.
        /// </summary>
        /// <param name="rgdf">Specifies the data to set.</param>
        /// <param name="nLabel">Specifies the label to set.</param>
        /// <remarks>
        /// The data of the array is cast to either (double) for real data, or (byte) for the byte data.
        /// </remarks>
        public void SetData(double[] rgdf, int nLabel)
        {
            m_nLabel = nLabel;
            m_rgRealDataF = null;
            m_rgRealDataD = new double[rgdf.Length];
            Array.Copy(rgdf, m_rgRealDataD, rgdf.Length);
            m_rgByteData = null;
            m_bIsRealData = true;
        }

        /// <summary>
        /// Set the data to the <i>float</i> array specified.
        /// </summary>
        /// <param name="rgf">Specifies the data to set.</param>
        /// <param name="nLabel">Specifies the label to set.</param>
        /// <remarks>
        /// The data of the array is cast to either (double) for real data, or (byte) for the byte data.
        /// </remarks>
        public void SetData(float[] rgf, int nLabel)
        {
            m_nLabel = nLabel;
            m_rgRealDataD = null;
            m_rgRealDataF = new float[rgf.Length];
            Array.Copy(rgf, m_rgRealDataF, rgf.Length);
            m_rgByteData = null;
            m_bIsRealData = true;
        }

        /// <summary>
        /// Sets the label.
        /// </summary>
        /// <param name="nLabel">Specifies the label.</param>
        public void SetLabel(int nLabel)
        {
            m_nLabel = nLabel;
        }

        /// <summary>
        /// Resets the label to the original label used when creating the SimpleDatum.
        /// </summary>
        public void ResetLabel()
        {
            m_nLabel = m_nOriginalLabel;
        }

        /// <summary>
        /// Returns the number of data items.
        /// </summary>
        public int ItemCount
        {
            get
            {
                if (IsRealData)
                {
                    if (m_rgRealDataD != null)
                        return m_rgRealDataD.Length;
                    else if (m_rgRealDataF != null)
                        return m_rgRealDataF.Length;
                    else
                        throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                }
                else
                    return m_rgByteData.Length;
            }
        }

        /// <summary>
        /// Returns <i>true</i> if either the RealDataD or RealDataF are non <i>null</i> and have length > 0.
        /// </summary>
        public bool HasRealData
        {
            get
            {
                if (m_rgRealDataD != null && m_rgRealDataD.Length > 0)
                    return true;

                if (m_rgRealDataF != null && m_rgRealDataF.Length > 0)
                    return true;

                return false;
            }
        }

        /// <summary>
        /// Returns the item at a specified index in the type specified.
        /// </summary>
        /// <typeparam name="T">Specifies the output type.</typeparam>
        /// <param name="nIdx">Specifies the index of the data to retrieve.</param>
        /// <returns>The converted value is returned.</returns>
        public T GetDataAt<T>(int nIdx)
        {
            if (IsRealData)
            {
                if (m_rgRealDataD != null)
                    return (T)Convert.ChangeType(m_rgRealDataD[nIdx], typeof(T));
                else if (m_rgRealDataF != null)
                    return (T)Convert.ChangeType(m_rgRealDataF[nIdx], typeof(T));
                else
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
            }
            else
            {
                return (T)Convert.ChangeType(m_rgByteData[nIdx], typeof(T));
            }
        }

        /// <summary>
        /// Returns the item at a specified index in the <i>double</i> type.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the data to retrieve.</param>
        /// <returns>The value is returned as a <i>double</i>.</returns>
        public double GetDataAtD(int nIdx)
        {
            if (IsRealData)
            {
                if (m_rgRealDataD != null)
                    return m_rgRealDataD[nIdx];
                else if (m_rgRealDataF != null)
                    return m_rgRealDataF[nIdx];
                else
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
            }
            else
            {
                return m_rgByteData[nIdx];
            }
        }

        /// <summary>
        /// Returns the item at a specified index in the <i>float</i> type.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the data to retrieve.</param>
        /// <returns>The value is returned as a <i>float</i>.</returns>
        public float GetDataAtF(int nIdx)
        {
            if (IsRealData)
            {
                if (m_rgRealDataD != null)
                    return (float)m_rgRealDataD[nIdx];
                else if (m_rgRealDataF != null)
                    return m_rgRealDataF[nIdx];
                else
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
            }
            else
            {
                return m_rgByteData[nIdx];
            }
        }

        /// <summary>
        /// Returns the item at a specified index in the <i>byte</i> type.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the data to retrieve.</param>
        /// <returns>The value is returned as a <i>byte</i>.</returns>
        public byte GetDataAtByte(int nIdx)
        {
            if (IsRealData)
            {
                double dfVal = 0;

                if (m_rgRealDataD != null)
                    dfVal = m_rgRealDataD[nIdx];
                else if (m_rgRealDataF != null)
                    dfVal = m_rgRealDataF[nIdx];
                else
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");

                return (byte)Math.Min(Math.Max(0, dfVal), 255);
            }
            else
            {
                return m_rgByteData[nIdx];
            }
        }

        /// <summary>
        /// Returns the data as a generic array and optionally pads the data.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="nImagePadX">Optionally, specifies the amount to pad the data width.</param>
        /// <param name="nImagePadY">Optionally, specifies the amount to pad the data height.</param>
        /// <returns>The data is returned as a generic array.</returns>
        public T[] GetData<T>(int nImagePadX = 0, int nImagePadY = 0)
        {
            if (IsRealData)
            {
                if (m_rgRealDataD != null)
                {
                    if (m_rgRealDataD.Length == 0)
                        return null;

                    if (typeof(T) == typeof(double))
                        return (T[])Convert.ChangeType(m_rgRealDataD.ToArray(), typeof(T[]));

                    T[] rg = new T[m_rgRealDataD.Length];
                    for (int i = 0; i < rg.Length; i++)
                    {
                        rg[i] = (T)Convert.ChangeType(m_rgRealDataD[i], typeof(T));
                    }

                    return rg;
                }
                else if (m_rgRealDataF != null)
                {
                    if (m_rgRealDataF.Length == 0)
                        return null;

                    if (typeof(T) == typeof(float))
                        return (T[])Convert.ChangeType(m_rgRealDataF.ToArray(), typeof(T[]));

                    T[] rg = new T[m_rgRealDataF.Length];
                    for (int i = 0; i < rg.Length; i++)
                    {
                        rg[i] = (T)Convert.ChangeType(m_rgRealDataF[i], typeof(T));
                    }

                    return rg;
                }
                else
                {
                    return null;
                }
            }
            else
            {
                if (m_rgByteData == null || m_rgByteData.Length == 0)
                    return null;

                T[] rg = new T[m_rgByteData.Length];

                for (int i = 0; i < rg.Length; i++)
                {
                    rg[i] = (T)Convert.ChangeType(m_rgByteData[i], typeof(T));
                }

                return rg;
            }
        }

        /// <summary>
        /// Return the non-real data as a <i>byte</i> array after padding the data.
        /// </summary>
        /// <param name="rgData">Specifies the data.</param>
        /// <param name="nImagePadX">Specifies the amount to pad the data width.</param>
        /// <param name="nImagePadY">Specifies the amount to pad the data height.</param>
        /// <param name="nHeight">Specifies the height of the original data.</param>
        /// <param name="nWidth">Specifies the width of the original data.</param>
        /// <param name="nChannels">Specifies the number of channels in the original data.</param>
        /// <returns>The data is returned as a <i>byte</i> array.</returns>
        public static byte[] GetByteData(byte[] rgData, int nImagePadX, int nImagePadY, int nHeight, int nWidth, int nChannels)
        {
            if (nImagePadX == 0 && nImagePadY == 0)
                return rgData;

            return PadData<byte>(new List<byte>(rgData), nImagePadX, nImagePadY, nHeight, nWidth, nChannels);
        }

        /// <summary>
        /// Return the real data as a <i>double</i> or <i>float</i> array (depending on the original encoding data type) after padding the data.
        /// </summary>
        /// <param name="rgData">Specifies the data.</param>
        /// <param name="nImagePadX">Specifies the amount to pad the data width.</param>
        /// <param name="nImagePadY">Specifies the amount to pad the data height.</param>
        /// <param name="nHeight">Specifies the height of the original data.</param>
        /// <param name="nWidth">Specifies the width of the original data.</param>
        /// <param name="nChannels">Specifies the number of channels in the original data.</param>
        /// <returns>The data is returned as a <i>double</i> or <i>float</i> array depending on the original encoding type.</returns>
        public static Tuple<double[], float[]> GetRealData(byte[] rgData, int nImagePadX, int nImagePadY, int nHeight, int nWidth, int nChannels)
        {
            Tuple<double[], float[]> rgRealData = GetRealData(rgData);

            double[] rgRealDataD = rgRealData.Item1;
            float[] rgRealDataF = rgRealData.Item2;

            if (rgRealDataD != null)
                rgRealDataD = PadData<double>(rgRealDataD.ToList(), nImagePadX, nImagePadY, nHeight, nWidth, nChannels);
            else
                rgRealDataF = PadData<float>(rgRealDataF.ToList(), nImagePadX, nImagePadY, nHeight, nWidth, nChannels);

            return new Tuple<double[], float[]>(rgRealDataD, rgRealDataF);
        }

        /// <summary>
        /// Padd the data.
        /// </summary>
        /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
        /// <param name="rgData">Specifies the data to pad.</param>
        /// <param name="nImagePadX">Optionally, specifies the amount to pad the data width.</param>
        /// <param name="nImagePadY">Optionally, specifies the amount to pad the data height.</param>
        /// <param name="nHeight">Specifies the height of the original data.</param>
        /// <param name="nWidth">Specifies the width of the original data.</param>
        /// <param name="nChannels">Specifies the number of channels in the original data.</param>
        /// <returns>The padded data is returned as a generic array.</returns>
        public static T[] PadData<T>(List<T> rgData, int nImagePadX, int nImagePadY, int nHeight, int nWidth, int nChannels)
        {
            int nDstIdx = 0;
            int nCount = nChannels * (nHeight * (nImagePadX + nWidth) + (nImagePadY * nImagePadX));
            T[] rgDataNew = new T[nCount];

            for (int c = 0; c < nChannels; c++)
            {
                for (int y = 0; y < nHeight; y++)
                {
                    for (int i = 0; i < nImagePadX; i++)
                    {
                        rgDataNew[nDstIdx] = (T)Convert.ChangeType(0.0, typeof(T));
                        nDstIdx++;
                    }

                    for (int x = 0; x < nWidth; x++)
                    {
                        int nIdx = (c * nWidth * nHeight) + (y * nWidth) + x;

                        rgDataNew[nDstIdx] = rgData[nIdx];
                        nDstIdx++;
                    }
                }

                for (int i = 0; i < nImagePadY; i++)
                {
                    for (int x = 0; x < nWidth + nImagePadX; x++)
                    {
                        rgDataNew[nDstIdx] = (T)Convert.ChangeType(0.0, typeof(T));
                        nDstIdx++;
                    }
                }
            }

            return rgDataNew;
        }

        /// <summary>
        /// Returns the data as a <i>byte</i> array regardless of how it is stored.
        /// </summary>
        /// <param name="bEncoded">Returns whether or not the original data is real (<i>true</i>) or not (<i>false</i>).</param>
        /// <returns>A <i>byte</i> array of the data is returned.</returns>
        public byte[] GetByteData(out bool bEncoded)
        {
            if (!IsRealData)
            {
                bEncoded = false;
                return ByteData;
            }

            bEncoded = true;

            if (m_rgRealDataD != null)
                return GetByteData(new List<double>(m_rgRealDataD));
            else if (m_rgRealDataF != null)
                return GetByteData(new List<float>(m_rgRealDataF));
            else
                throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
        }

        /// <summary>
        /// Encodes a list of <i>double</i> values to an encoded <i>byte</i> array.
        /// </summary>
        /// <remarks>
        /// Each double in the stored data is converted using a BitConverter.
        /// </remarks>
        /// <param name="rgData">Specifies the data as list of <i>double</i> values.</param>
        /// <returns>The encoded doubles are returned as an array of <i>byte</i> values.</returns>
        public static byte[] GetByteData(List<double> rgData)
        {
            int nCount = rgData.Count;
            int nSize = sizeof(double);

            int nOffset = 0;
            int nDataCount = sizeof(int) + sizeof(int) + (nSize * rgData.Count);
            byte[] rgByte = new byte[nDataCount];

            byte[] rg = BitConverter.GetBytes(nCount);
            Array.Copy(rg, 0, rgByte, nOffset, rg.Length);
            nOffset += rg.Length;

            rg = BitConverter.GetBytes(nSize);
            Array.Copy(rg, 0, rgByte, nOffset, rg.Length);
            nOffset += rg.Length;

            foreach (double df in rgData)
            {
                rg = BitConverter.GetBytes(df);
                Array.Copy(rg, 0, rgByte, nOffset, rg.Length);
                nOffset += rg.Length;
            }

            return rgByte;
        }

        /// <summary>
        /// Encodes a list of <i>float</i> values to an encoded <i>byte</i> array.
        /// </summary>
        /// <remarks>
        /// Each double in the stored data is converted using a BitConverter.
        /// </remarks>
        /// <param name="rgData">Specifies the data as list of <i>float</i> values.</param>
        /// <returns>The encoded doubles are returned as an array of <i>byte</i> values.</returns>
        public static byte[] GetByteData(List<float> rgData)
        {
            int nCount = rgData.Count;
            int nSize = sizeof(float);

            int nOffset = 0;
            int nDataCount = sizeof(int) + sizeof(int) + (nSize * rgData.Count);
            byte[] rgByte = new byte[nDataCount];

            byte[] rg = BitConverter.GetBytes(nCount);
            Array.Copy(rg, 0, rgByte, nOffset, rg.Length);
            nOffset += rg.Length;

            rg = BitConverter.GetBytes(nSize);
            Array.Copy(rg, 0, rgByte, nOffset, rg.Length);
            nOffset += rg.Length;

            foreach (float df in rgData)
            {
                rg = BitConverter.GetBytes(df);
                Array.Copy(rg, 0, rgByte, nOffset, rg.Length);
                nOffset += rg.Length;
            }

            return rgByte;
        }

        /// <summary>
        /// Decodes an array of <i>byte</i> values into a array of either <i>double</i> or <i>float</i> values depending on how the original encoding was made.
        /// </summary>
        /// <param name="rgData">Specifies the array of <i>byte</i> values containing the encoded <i>double</i> or <i>float</i> values.</param>
        /// <returns>The array of decoded <i>double</i> or <i>float</i> values is returned in a Tuple where only one item is set depending on the encoding data type used.</returns>
        public static Tuple<double[], float[]> GetRealData(byte[] rgData)
        {
            double[] rgDataD = null;
            float[] rgDataF = null;
            int nIdx = 0;

            int nCount = BitConverter.ToInt32(rgData, nIdx);
            nIdx += 4;
            int nSize = BitConverter.ToInt32(rgData, nIdx);
            nIdx += 4;

            // If the size is invalid, revert back to the legacy double only data.
            if (nSize != sizeof(float) && nSize != sizeof(double))
            {
                nIdx = 0;
                nSize = sizeof(double);
            }

            if (nSize == sizeof(double))
                rgDataD = getRealDataD(rgData, nIdx);
            else
                rgDataF = getRealDataF(rgData, nIdx);

            return new Tuple<double[], float[]>(rgDataD, rgDataF);
        }

        /// <summary>
        /// Decodes an array of <i>byte</i> values into a array of <i>double</i> values.
        /// </summary>
        /// <param name="rgData">Specifies the array of <i>byte</i> values containing the encoded <i>double</i> values.</param>
        /// <param name="nIdx">Specifies the offset where reading is to start.</param>
        /// <returns>The array of decoded <i>double</i> values is returned.</returns>
        protected static double[] getRealDataD(byte[] rgData, int nIdx)
        {
            List<double> rgData0 = new List<double>();

            while (nIdx < rgData.Length)
            {
                rgData0.Add(BitConverter.ToDouble(rgData, nIdx));
                nIdx += 8;
            }

            return rgData0.ToArray();
        }

        /// <summary>
        /// Decodes an array of <i>byte</i> values into a array of <i>float</i> values.
        /// </summary>
        /// <param name="rgData">Specifies the array of <i>byte</i> values containing the encoded <i>float</i> values.</param>
        /// <param name="nIdx">Specifies the offset where reading is to start.</param>
        /// <returns>The array of decoded <i>float</i> values is returned.</returns>
        protected static float[] getRealDataF(byte[] rgData, int nIdx)
        {
            List<float> rgData0 = new List<float>();

            while (nIdx < rgData.Length)
            {
                rgData0.Add(BitConverter.ToSingle(rgData, nIdx));
                nIdx += 8;
            }

            return rgData0.ToArray();
        }

        /// <summary>
        /// Creates a new SimpleDatum and adds another SimpleDatum to it.
        /// </summary>
        /// <param name="d">Specifies the other SimpleDatum.</param>
        /// <returns>The new SimpleDatum is returned.</returns>
        public SimpleDatum Add(SimpleDatum d)
        {
            if (m_nChannels != d.Channels ||
                m_nHeight != d.Height ||
                m_nWidth != d.Width)
                throw new Exception("Datum dimmensions do not match!");

            if (ItemCount != d.ItemCount)
                throw new Exception("Datum counts do not match!");

            SimpleDatum d1 = new SimpleDatum(d, false);

            d1.m_rgRealDataD = null;
            d1.m_rgRealDataF = null;
            d1.m_rgByteData = null;

            if (m_bIsRealData)
            {
                if (m_rgRealDataD != null)
                {
                    d1.m_rgRealDataD = new double[m_rgRealDataD.Length];

                    for (int i = 0; i < m_rgRealDataD.Length; i++)
                    {
                        d1.m_rgRealDataD[i] = m_rgRealDataD[i] + d.GetDataAtD(i);
                    }
                }
                else if (m_rgRealDataF != null)
                {
                    d1.m_rgRealDataF = new float[m_rgRealDataF.Length];

                    for (int i = 0; i < m_rgRealDataF.Length; i++)
                    {
                        d1.m_rgRealDataF[i] = m_rgRealDataF[i] + d.GetDataAtF(i);
                    }
                }
                else
                {
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                }

                d1.m_bIsRealData = true;
            }
            else
            {
                d1.m_rgByteData = new byte[m_rgByteData.Length];
                d1.m_bIsRealData = false;

                for (int i = 0; i < m_rgByteData.Length; i++)
                {
                    int nVal = m_rgByteData[i] + d.GetDataAtByte(i);
                    d1.m_rgByteData[i] = (byte)Math.Min(Math.Max(0, nVal), 255);
                }
            }

            return d1;
        }

        /// <summary>
        /// Divides all elements of the SimpleDatum by a value and returns the result as a new SimpleDatum.
        /// </summary>
        /// <param name="dfVal">Specifies the non-zero divisor.</param>
        /// <param name="bConvertToByte">If the SimpleDatum contains real numbers, specifies whether or not to convert the data to <i>byte</i> data.</param>
        /// <returns>The new SimpleDatum is returned.</returns>
        public SimpleDatum Div(double dfVal, bool bConvertToByte)
        {
            if (dfVal == 0)
                throw new ArgumentOutOfRangeException("dfVal", 0, "Cannot divide the simple datums by zero!");

            SimpleDatum d1 = new SimpleDatum(this, false);

            d1.m_rgRealDataD = null;
            d1.m_rgRealDataF = null;
            d1.m_rgByteData = null;

            int nCount = ItemCount;

            if (m_bIsRealData && !bConvertToByte)
            {
                if (m_rgRealDataD != null)
                {
                    d1.m_rgRealDataD = new double[nCount];
                    d1.m_bIsRealData = true;

                    for (int i = 0; i < nCount; i++)
                    {
                        d1.m_rgRealDataD[i] = m_rgRealDataD[i] / dfVal;
                    }
                }
                else if (m_rgRealDataF != null)
                {
                    d1.m_rgRealDataF = new float[nCount];
                    d1.m_bIsRealData = true;

                    for (int i = 0; i < nCount; i++)
                    {
                        d1.m_rgRealDataF[i] = (float)(m_rgRealDataF[i] / dfVal);
                    }
                }
                else
                {
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                }
            }
            else if (m_bIsRealData && bConvertToByte)
            {
                d1.m_rgByteData = new byte[nCount];
                d1.m_bIsRealData = false;

                if (m_rgRealDataD != null)
                {
                    for (int i = 0; i < nCount; i++)
                    {
                        double dfVal1 = m_rgRealDataD[i] / dfVal;
                        m_rgByteData[i] = (byte)Math.Min(Math.Max(dfVal1, 0), 255);
                    }
                }
                else if (m_rgRealDataF != null)
                {
                    for (int i = 0; i < nCount; i++)
                    {
                        double dfVal1 = m_rgRealDataF[i] / dfVal;
                        m_rgByteData[i] = (byte)Math.Min(Math.Max(dfVal1, 0), 255);
                    }
                }
                else
                {
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                }
            }
            else
            {
                d1.m_rgByteData = new byte[nCount];
                d1.m_bIsRealData = false;

                for (int i = 0; i < nCount; i++)
                {
                    double dfVal1 = (double)m_rgByteData[i] / dfVal;
                    m_rgByteData[i] = (byte)Math.Min(Math.Max(dfVal1, 0), 255);
                }
            }

            return d1;
        }

        /// <summary>
        /// Returns the ID of the data source that owns this image.
        /// </summary>
        public int SourceID
        {
            get { return m_nSourceID; }
        }

        /// <summary>
        /// Returns the original source ID which is set when using a virtual ID.
        /// </summary>
        public int OriginalSourceID
        {
            get { return m_nOriginalSourceID; }
        }

        /// <summary>
        /// Returns the ID of the image in the database.
        /// </summary>
        public int ImageID
        {
            get { return m_nImageID; }
        }

        /// <summary>
        /// Returns the virtual ID of the SimpleDatum.
        /// </summary>
        public int VirtualID
        {
            get { return m_nVirtualID; }
        }

        /// <summary>
        /// Get/set the group ID of the SimpleDatum.
        /// </summary>
        public int GroupID
        {
            get { return m_nGroupID; }
            set { m_nGroupID = value; }
        }

        /// <summary>
        /// Returns the index of the SimpleDatum.
        /// </summary>
        public int Index
        {
            get { return m_nIndex; }
            set { m_nIndex = value; }
        }

        /// <summary>
        /// Get/set the Timestamp.
        /// </summary>
        public DateTime TimeStamp
        {
            get { return m_dt; }
            set { m_dt = value; }
        }

        /// <summary>
        /// Get/set whether or not the label was auto generated.
        /// </summary>
        public bool AutoLabeled
        {
            get { return m_bAutoLabeled; }
            set { m_bAutoLabeled = value; }
        }

        /// <summary>
        /// Returns whether or not the data contains real numbers or byte data.
        /// </summary>
        public bool IsRealData
        {
            get { return m_bIsRealData; }
        }

        /// <summary>
        /// Return the height of the data.
        /// </summary>
        public int Height
        {
            get { return m_nHeight; }
        }

        /// <summary>
        /// Return the width of the data.
        /// </summary>
        public int Width
        {
            get { return m_nWidth; }
        }

        /// <summary>
        /// Return the number of channels of the data.
        /// </summary>
        public int Channels
        {
            get { return m_nChannels; }
        }

        /// <summary>
        /// Return the known label of the data.
        /// </summary>
        public int Label
        {
            get { return m_nLabel; }
        }

        /// <summary>
        /// Get/set the original known label of the data.
        /// </summary>
        public int OriginalLabel
        {
            get { return m_nOriginalLabel; }
            set { m_nOriginalLabel = value; }
        }

        /// <summary>
        /// Return the <i>byte</i> data.  This field is valid when <i>IsRealData</i> = <i>false</i>.
        /// </summary>
        public byte[] ByteData
        {
            get { return m_rgByteData; }
        }

        /// <summary>
        /// Return the <i>double</i> data.  This field is valid when <i>IsRealData</i> = <i>true</i>.
        /// </summary>
        public double[] RealDataD
        {
            get { return m_rgRealDataD; }
        }

        /// <summary>
        /// Return the <i>float</i> data.  This field is valid when <i>IsRealData</i> = <i>true</i>.
        /// </summary>
        public float[] RealDataF
        {
            get { return m_rgRealDataF; }
        }

        /// <summary>
        /// Get/set the boost for this data.
        /// </summary>
        public int Boost
        {
            get { return m_nBoost; }
            set { m_nBoost = value; }
        }

        /// <summary>
        /// Reset the boost to the original boost.
        /// </summary>
        public void ResetBoost()
        {
            m_nBoost = m_nOriginalBoost;
        }

        /// <summary>
        /// Get/set the data format of the data criteria.
        /// </summary>
        public DATA_FORMAT DataCriteriaFormat
        {
            get { return m_dataCriteriaFormat; }
            set { m_dataCriteriaFormat = value; }
        }

        /// <summary>
        /// Get/set data criteria associated with the data.
        /// </summary>
        public byte[] DataCriteria
        {
            get { return m_rgDataCriteria; }
            set { m_rgDataCriteria = value; }
        }

        /// <summary>
        /// Get/set the data format of the debug data.
        /// </summary>
        public DATA_FORMAT DebugDataFormat
        {
            get { return m_debugDataFormat; }
            set { m_debugDataFormat = value; }
        }

        /// <summary>
        /// Get/set debug data associated with the data.
        /// </summary>
        public byte[] DebugData
        {
            get { return m_rgDebugData; }
            set { m_rgDebugData = value; }
        }

        /// <summary>
        /// Get/set a description of the data.
        /// </summary>
        public string Description
        {
            get { return m_strDesc; }
            set { m_strDesc = value; }
        }

        /// <summary>
        /// When using annotations, the annotation type specifies the type of annotation.  Currently, only
        /// the BBOX annotation type is supported.
        /// </summary>
        public ANNOTATION_TYPE annotation_type
        {
            get { return m_nAnnotationType; }
            set { m_nAnnotationType = value; }
        }

        /// <summary>
        /// When using annoations, each annotation group contains an annotation for a particular class used with SSD.
        /// </summary>
        public AnnotationGroupCollection annotation_group
        {
            get { return m_rgAnnotationGroup; }
            set { m_rgAnnotationGroup = value; }
        }

        /// <summary>
        /// Resize the data and return it as a new SimpleDatum.
        /// </summary>
        /// <param name="nH">Specifies the new height.</param>
        /// <param name="nW">Specifies the new width.</param>
        /// <returns>A new resized SimpleDatum is returned.</returns>
        public SimpleDatum Resize(int nH, int nW)
        {
            Image bmp = ImageData.GetImage(new Datum(this));
            Bitmap bmpNew = ImageTools.ResizeImage(bmp, nH, nW);
            Datum d = ImageData.GetImageData(bmpNew, this);

            bmp.Dispose();
            bmpNew.Dispose();

            return d;
        }

        /// <summary>
        /// Return a string representation of the SimpleDatum.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            string strOut = "Idx = " + m_nIndex.ToString("N0");

            strOut += "; Label = " + m_nLabel.ToString();
            strOut += "; Boost = " + m_nBoost.ToString();

            if (m_dt != DateTime.MinValue)
                strOut += "; Time = " + m_dt.ToString();

            if (!string.IsNullOrEmpty(m_strDesc))
                strOut += "; Desc = " + m_strDesc;

            return strOut;
        }

        /// <summary>
        /// Returns a string containing the items of the SimpleDatum.
        /// </summary>
        /// <param name="nMaxItems">Specifies the maximum number of items to output.</param>
        /// <returns>Returns a string containing the data items.</returns>
        public string ToArrayAsString(int nMaxItems)
        {
            string str = "";

            if (m_bIsRealData)
            {
                if (m_rgRealDataD != null)
                {
                    for (int i = 0; i < m_rgRealDataD.Length && i < nMaxItems; i++)
                    {
                        str += m_rgRealDataD[i].ToString() + ",";
                    }
                }
                else if (m_rgRealDataF != null)
                {
                    for (int i = 0; i < m_rgRealDataF.Length && i < nMaxItems; i++)
                    {
                        str += m_rgRealDataF[i].ToString() + ",";
                    }
                }
                else
                {
                    str = "No Real Data Found!";
                }
            }
            else
            {
                for (int i = 0; i < m_rgByteData.Length && i < nMaxItems; i++)
                {
                    str += m_rgByteData[i].ToString() + ",";
                }
            }

            return str.TrimEnd(',');
        }


        /// <summary>
        /// Return the SimpleData data as a Bytemap.
        /// </summary>
        /// <remarks>
        /// This function is only suported on byte based SimpleDatum's.
        /// </remarks>
        /// <returns>The Bytemap data is returned.</returns>
        public Bytemap ToBytemap()
        {
            if (m_rgByteData == null)
                throw new Exception("Bytemaps are only supported with byte based data.");

            return new Bytemap(m_nChannels, m_nHeight, m_nWidth, m_rgByteData);
        }

        /// <summary>
        /// Accumulate a portion of a SimpleDatum to calculate the mean value.
        /// </summary>
        /// <param name="rgdfMean">Specifies the accumulated mean value.</param>
        /// <param name="sd">Specifies the SimpleDatum to add to the mean.</param>
        /// <param name="nTotal">Specifies the overall total used to calculate the portion of the sd to add to the mean value.</param>
        /// <returns>After successfully adding to the total used to calculate the mean, <i>true</i> is returned, otherwise if the SimpleDatum is a virtual datum <i>false</i> is returned.</returns>
        public static bool AccumulateMean(ref double[] rgdfMean, SimpleDatum sd, int nTotal)
        {
            if (rgdfMean == null)
                rgdfMean = new double[sd.ItemCount];

            if (sd.IsRealData)
            {
                if (sd.RealDataD != null)
                {
                    for (int i = 0; i < sd.ItemCount; i++)
                    {
                        rgdfMean[i] += sd.RealDataD[i] / nTotal;
                    }
                }
                else if (sd.RealDataF != null)
                {
                    for (int i = 0; i < sd.ItemCount; i++)
                    {
                        rgdfMean[i] += sd.RealDataF[i] / nTotal;
                    }
                }
                else
                {
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                }
            }
            else
            {
                for (int i = 0; i < sd.ItemCount; i++)
                {
                    rgdfMean[i] += (double)sd.ByteData[i] / (double)nTotal;
                }
            }

            return true;
        }

        /// <summary>
        /// Calculate the mean of an array of SimpleDatum and return the mean as a new SimpleDatum.
        /// </summary>
        /// <param name="log">Specifies the Log used for output.</param>
        /// <param name="rgImg">Specifies the input SimpleDatum.</param>
        /// <param name="rgAbort">Specifies a set of wait handles used to abort the process.</param>
        /// <returns>A new SimpleDatum containing the mean is returned.</returns>
        public static SimpleDatum CalculateMean(Log log, SimpleDatum[] rgImg, WaitHandle[] rgAbort)
        {
            if (rgImg.Length < 2)
                throw new Exception("There must be at least 2 images in the simple datum array.");

            float[] rgSums;

            if (rgImg[0].ByteData != null)
                rgSums = new float[rgImg[0].ByteData.Length];
            else if (rgImg[0].RealDataD != null)
                rgSums = new float[rgImg[0].RealDataD.Length];
            else if (rgImg[0].RealDataF != null)
                rgSums = new float[rgImg[0].RealDataF.Length];
            else
                throw new Exception("No data in rgImg[0]!");

            Stopwatch sw = new Stopwatch();

            try
            {
                sw.Start();

                for (int i = 0; i < rgImg.Length; i++)
                {
                    if (rgImg[i] != null)
                    {
                        if (rgImg[i].ByteData != null)
                        {
                            for (int n = 0; n < rgSums.Length; n++)
                            {
                                rgSums[n] += rgImg[i].ByteData[n];
                            }
                        }
                        else if (rgImg[i].RealDataD != null)
                        {
                            for (int n = 0; n < rgSums.Length; n++)
                            {
                                rgSums[n] += (float)rgImg[i].RealDataD[n];
                            }
                        }
                        else if (rgImg[i].RealDataF != null)
                        {
                            for (int n = 0; n < rgSums.Length; n++)
                            {
                                rgSums[n] += (float)rgImg[i].RealDataF[n];
                            }
                        }
                        else
                        {
                            throw new Exception("No data in rgImg[" + i.ToString() + "]!");
                        }

                        if (sw.Elapsed.TotalMilliseconds > 2000)
                        {
                            double dfPct = (double)i / (double)rgImg.Length;
                            log.WriteLine("processing mean (" + dfPct.ToString("P") + ")");
                            sw.Restart();

                            if (rgAbort != null)
                            {
                                if (EventWaitHandle.WaitAny(rgAbort, 0) != EventWaitHandle.WaitTimeout)
                                    return null;
                            }
                        }
                    }
                }

                if (rgImg[0].ByteData != null)
                {
                    byte[] rgbMean = new byte[rgSums.Length];

                    for (int n = 0; n < rgSums.Length; n++)
                    {
                        rgbMean[n] = (byte)(rgSums[n] / (float)rgImg.Length);
                    }

                    SimpleDatum d = new SimpleDatum(rgImg[0], false);
                    d.SetData(new List<byte>(rgbMean), -1, true);
                    return d;
                }
                else if (rgImg[0].RealDataD != null)
                {
                    double[] rgdfMean = new double[rgSums.Length];

                    for (int n = 0; n < rgSums.Length; n++)
                    {
                        rgdfMean[n] = (double)rgSums[n] / rgImg.Length;
                    }

                    SimpleDatum d = new SimpleDatum(rgImg[0], false);
                    d.SetData(new List<double>(rgdfMean), -1, true);
                    return d;
                }
                else if (rgImg[0].RealDataF != null)
                {
                    float[] rgfMean = new float[rgSums.Length];

                    for (int n = 0; n < rgSums.Length; n++)
                    {
                        rgfMean[n] = (float)rgSums[n] / rgImg.Length;
                    }

                    SimpleDatum d = new SimpleDatum(rgImg[0], false);
                    d.SetData(new List<float>(rgfMean), -1, true);
                    return d;
                }
                else
                {
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                }
            }
            finally
            {
            }
        }

        /// <summary>
        /// Save the annotation data to a byte array.
        /// </summary>
        /// <param name="type">Specifies the annotation type.</param>
        /// <param name="annotations">Specifies the annotations to save.</param>
        /// <returns>The byte array containing the annotations is returned.</returns>
        public static byte[] SaveAnnotationDataToDataCriteriaByteArray(ANNOTATION_TYPE type, AnnotationGroupCollection annotations)
        {
            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                bw.Write((int)type);

                int nCount = 0;
                if (annotations != null)
                    nCount = annotations.Count;

                bw.Write(nCount);

                if (annotations != null)
                {
                    for (int i = 0; i < annotations.Count; i++)
                    {
                        annotations[i].Save(bw);
                    }
                }

                bw.Flush();
                return ms.ToArray();
            }
        }

        /// <summary>
        /// Save the annotation data and type to the data criteria.
        /// </summary>
        public void SaveAnnotationDataToDataCriteria()
        {
            DataCriteria = SaveAnnotationDataToDataCriteriaByteArray(m_nAnnotationType, m_rgAnnotationGroup);
            DataCriteriaFormat = DATA_FORMAT.ANNOTATION_DATA;
        }

        /// <summary>
        /// Load the AnnotationGroups from the byte array.
        /// </summary>
        /// <param name="rg">Specifies the byte array containing the annotation data.</param>
        /// <param name="fmt">Specifies the annotation data format (expected to be ANNOATION_DATA).</param>
        /// <param name="type">Returns the annoation data type.</param>
        /// <returns>The AnnotationGroupCollection loaded is returned.</returns>
        public static AnnotationGroupCollection LoadAnnotationDataFromDataCriteria(byte[] rg, DATA_FORMAT fmt, out ANNOTATION_TYPE type)
        {
            type = ANNOTATION_TYPE.NONE;

            if (rg == null || fmt != DATA_FORMAT.ANNOTATION_DATA)
                return null;

            using (MemoryStream ms = new MemoryStream(rg))
            using (BinaryReader br = new BinaryReader(ms))
            {
                type = (ANNOTATION_TYPE)br.ReadInt32();
                AnnotationGroupCollection rgGroups = new AnnotationGroupCollection();

                int nCount = br.ReadInt32();
                if (nCount > 0)
                {
                    for (int i = 0; i < nCount; i++)
                    {
                        rgGroups.Add(AnnotationGroup.Load(br));
                    }
                }

                return rgGroups;
            }
        }

        /// <summary>
        /// Load the annotation data and type from the data criteria.
        /// </summary>
        /// <returns>When successfully loaded <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool LoadAnnotationDataFromDataCriteria()
        {
            m_rgAnnotationGroup = LoadAnnotationDataFromDataCriteria(DataCriteria, DataCriteriaFormat, out m_nAnnotationType);

            if (m_rgAnnotationGroup == null)
                return false;

            return true;
        }

        /// <summary>
        /// Save the SimpleDatum information to a text file.
        /// </summary>
        /// <remarks>
        /// This function is typically used for debugging.
        /// </remarks>
        /// <param name="strFile">Specifies the name of the target file.</param>
        public void SaveInfo(string strFile)
        {
            using (StreamWriter sw = new StreamWriter(strFile + ".txt"))
            {
                sw.WriteLine("Index = " + Index.ToString());
                sw.WriteLine("ImageID = " + ImageID.ToString());
                sw.WriteLine("VirtualID = " + VirtualID.ToString());
                sw.WriteLine("Label = " + Label.ToString());
                sw.WriteLine("AutoLabeled = " + AutoLabeled.ToString());
                sw.WriteLine("Boost = " + Boost.ToString());
                sw.WriteLine("SourceID = " + SourceID.ToString());
                sw.WriteLine("OriginalSourceID = " + OriginalSourceID.ToString());
                sw.WriteLine("Description = " + Description);
                sw.WriteLine("Time = " + TimeStamp.ToString());
                sw.WriteLine("Size = {1," + Channels.ToString() + "," + Height.ToString() + "," + Width.ToString() + "}");
            }
        }

        /// <summary>
        /// Load a SimpleData from text file information previously saved with SaveInfo.
        /// </summary>
        /// <remarks>
        /// Note, the SimpleDatum only contains the information but no data for this is used for debugging.
        /// </remarks>
        /// <param name="strFile">Specifies the file to load the SimpleDatum from.</param>
        /// <returns>The SimpleDatum is returned.</returns>
        public static SimpleDatum LoadInfo(string strFile)
        {
            using (StreamReader sr = new StreamReader(strFile))
            {
                string strVal = parseValue(sr.ReadLine(), '=');
                int nIndex = int.Parse(strVal);

                strVal = parseValue(sr.ReadLine(), '=');
                int nImageID = int.Parse(strVal);

                strVal = parseValue(sr.ReadLine(), '=');
                int nVirtualID = int.Parse(strVal);

                strVal = parseValue(sr.ReadLine(), '=');
                int nLabel = int.Parse(strVal);

                strVal = parseValue(sr.ReadLine(), '=');
                bool bAutoLabeled = bool.Parse(strVal);

                strVal = parseValue(sr.ReadLine(), '=');
                int nBoost = int.Parse(strVal);

                strVal = parseValue(sr.ReadLine(), '=');
                int nSrcId = int.Parse(strVal);

                strVal = parseValue(sr.ReadLine(), '=');
                int nOriginalSrcId = int.Parse(strVal);

                strVal = parseValue(sr.ReadLine(), '=');
                string strDesc = strVal;

                strVal = parseValue(sr.ReadLine(), '=');
                DateTime dt = DateTime.Parse(strVal);

                strVal = parseValue(sr.ReadLine(), '=');
                strVal = strVal.Trim('{', '}');
                string[] rgstr = strVal.Split(',');

                int nChannels = int.Parse(rgstr[1]);
                int nHeight = int.Parse(rgstr[2]);
                int nWidth = int.Parse(rgstr[3]);

                return new SimpleDatum(false, nChannels, nWidth, nHeight, nLabel, dt, nBoost, bAutoLabeled, nIndex, nVirtualID, nImageID, nSrcId, nOriginalSrcId);
            }
        }

        private static string parseValue(string str, char chDelim)
        {
            int nPos = str.LastIndexOf(chDelim);
            if (nPos < 0)
                return str;

            return str.Substring(nPos + 1).Trim();
        }

        /// <summary>
        /// Load all SimpleDatums from a directory of files previously stored with SaveInfo.
        /// </summary>
        /// <param name="strPath">Specifies the path to the files to load.</param>
        /// <returns>The list of SimpleDatum are returned.</returns>
        public static List<SimpleDatum> LoadFromPath(string strPath)
        {
            string[] rgstrFiles = Directory.GetFiles(strPath, "*.txt");
            List<SimpleDatum> rgData = new List<SimpleDatum>();

            foreach (string strFile in rgstrFiles)
            {
                rgData.Add(SimpleDatum.LoadInfo(strFile));
            }

            return rgData;
        }
    }

    /// <summary>
    /// The SimpleDatumCollection holds a named array of SimpleDatums
    /// </summary>
    public class SimpleDatumCollection : IEnumerable<SimpleDatum>
    {
        string m_strName;
        SimpleDatum[] m_rgItems = null;
        List<int> m_rgShape;
        object m_tag = null;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nCount">Specifies the number of items in the array.</param>
        /// <param name="strName">Optionally, specifies the name of the array (default = "").</param>
        /// <param name="rgShape">Optionally, specifies the shape of the items within the array.</param>
        public SimpleDatumCollection(int nCount, string strName = "", List<int> rgShape = null)
        {
            m_strName = strName;
            m_rgItems = new SimpleDatum[nCount];

            m_rgShape = new List<int>() { nCount };

            if (rgShape != null)
                m_rgShape.AddRange(rgShape);
        }

        /// <summary>
        /// Get/set a user defined value.
        /// </summary>
        public object Tag
        {
            get { return m_tag; }
            set { m_tag = value; }
        }

        /// <summary>
        /// Returns the shape of the items within the array (including the array count as the first element).
        /// </summary>
        public List<int> Shape
        {
            get { return m_rgShape; }
        }

        /// <summary>
        /// Get/set the name of the array.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
            set { m_strName = value; }
        }

        /// <summary>
        /// Get the number of items in the array.
        /// </summary>
        public int Count
        {
            get { return m_rgItems.Count(); }
        }

        /// <summary>
        /// Get/set an item in the array at a specified index.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item to get or set.</param>
        /// <returns>The item at the index is returned.</returns>
        public SimpleDatum this[int nIdx]
        {
            get { return m_rgItems[nIdx]; }
            set { m_rgItems[nIdx] = value; }
        }

        /// <summary>
        /// Get the enumerator for the collection.
        /// </summary>
        /// <returns>The enumerator for the collection is returned.</returns>
        public IEnumerator<SimpleDatum> GetEnumerator()
        {
            return (IEnumerator<SimpleDatum>)m_rgItems.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }
    }
}
