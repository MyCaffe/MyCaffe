using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
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
        byte[] m_rgByteData;
        double[] m_rgRealData;
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
        object m_tag = null;
        ANNOTATION_TYPE m_nAnnoationType = ANNOTATION_TYPE.NONE;
        List<AnnotationGroup> m_rgAnnotationGroup = null;

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
            LIST_FLOAT
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
        /// <param name="nLabel">Specifies the known label of the data.</param>
        /// <param name="dtTime">Specifies a time-stamp associated with the data.</param>
        /// <param name="rgData">Specifies the data as a list of <i>bytes</i> (expects <i>bIsReal</i> = <i>false</i>).</param>
        /// <param name="rgfData">Specifies the data as a list of <i>double</i> (expects <i>bIsReal</i> = <i>true</i>).</param>
        /// <param name="nBoost">Specifies the boost to use with the data (a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Specifies whether or not the label was auto-generated.</param>
        /// <param name="nIdx">Specifies the index of the data.</param>
        /// <param name="nVirtualID">Specifies a virtual index for the data (default = 0).  When specified, the SimpleDatum is used to reference another.</param>
        /// <param name="nImageID">Specifies the image ID within the database.</param>
        /// <param name="nSourceID">Specifies the data source ID of the data source that owns this image.</param>
        public SimpleDatum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, List<byte> rgData, List<double> rgfData, int nBoost, bool bAutoLabeled, int nIdx, int nVirtualID = 0, int nImageID = 0, int nSourceID = 0)
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

            if (rgData != null)
            {
                if (bIsReal)
                    throw new ArgumentException("The data sent is not real, but the bIsReal is set to true!");

                m_rgByteData = rgData.ToArray();
            }
            else if (rgfData != null)
            {
                if (!bIsReal)
                    throw new ArgumentException("The data sent is real, but the bIsReal is set to false!");

                m_rgRealData = rgfData.ToArray();
            }
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
        /// <param name="rgfData">Specifies the data as a list of <i>double</i> (expects <i>bIsReal</i> = <i>true</i>).</param>
        /// <param name="nBoost">Specifies the boost to use with the data (a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Specifies whether or not the label was auto-generated.</param>
        /// <param name="nIdx">Specifies the index of the data.</param>
        /// <param name="nVirtualID">Specifies a virtual index for the data (default = 0).  When specified, the SimpleDatum is used to reference another.</param>
        /// <param name="nImageID">Specifies the image ID within the database.</param>
        /// <param name="nSourceID">Specifies the data source ID of the data source that owns this image.</param>
        public SimpleDatum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, byte[] rgData, double[] rgfData, int nBoost, bool bAutoLabeled, int nIdx, int nVirtualID = 0, int nImageID = 0, int nSourceID = 0)
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

            if (rgData != null)
            {
                if (bIsReal)
                    throw new ArgumentException("The data sent is not real, but the bIsReal is set to true!");

                m_rgByteData = rgData;
            }
            else if (rgfData != null)
            {
                if (!bIsReal)
                    throw new ArgumentException("The data sent is real, but the bIsReal is set to false!");

                m_rgRealData = rgfData;
            }
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
        public SimpleDatum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, int nBoost = 0, bool bAutoLabeled = false, int nIdx = -1, int nVirtualID = 0, int nImageID = 0, int nSourceID = 0)
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
            m_rgByteData = null;
            m_rgRealData = null;
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

            m_rgRealData = data.Values;
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
        /// <param name="bCopyData">Specifies whether or not to copy the data, or just share it (default = false, share the data).</param>
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
                        double[] rgData = new double[m_nChannels * Height * Width];

                        for (int h = 0; h < Height; h++)
                        {
                            for (int w = 0; w < Width; w++)
                            {
                                int nIdxSrc = (h * Width) + w;
                                int nIdxDst = nIdxSrc * m_nChannels;

                                for (int c = 0; c < m_nChannels; c++)
                                {
                                    rgData[nIdxDst + c] = rg[c].RealData[nIdxSrc];
                                }
                            }
                        }

                        m_rgRealData = rgData;
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
                        m_rgRealData = null;
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
                    m_rgRealData = new double[m_nChannels * Height * Width];
                    m_rgByteData = null;

                    for (int i = 0; i < rg.Count; i++)
                    {
                        Array.Copy(rg[i].RealData, 0, m_rgRealData, nDstIdx, nItemCount);
                        nDstIdx += nItemCount;
                    }
                }
                else
                {
                    m_rgByteData = new byte[m_nChannels * Height * Width];
                    m_rgRealData = null;

                    for (int i = 0; i < rg.Count; i++)
                    {
                        Array.Copy(rg[i].ByteData, 0, m_rgByteData, nDstIdx, nItemCount);
                        nDstIdx += nItemCount;
                    }
                }
            }
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
        /// Returns the minimum value in the data or double.NaN if there is no data.
        /// </summary>
        public double Min
        {
            get
            {
                if (m_bIsRealData)
                {
                    if (m_rgRealData != null)
                        return m_rgRealData.Min(p => p);
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
                    if (m_rgRealData != null)
                        return m_rgRealData.Max(p => p);
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
            if (m_rgRealData != null && m_rgRealData.Length > nDataLen)
            {
                double[] rgData = new double[nDataLen];
                Array.Copy(m_rgRealData, rgData, nDataLen);
                m_rgRealData = rgData;
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
                for (int i = 0; i < m_rgRealData.Length; i++)
                {
                    if (m_rgRealData[i] != 0)
                        rgIdx.Add(i);
                }
            }
            else
            {
                for (int i = 0; i < m_rgByteData.Length; i++)
                {
                    if (m_rgRealData[i] != 0)
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

            if (m_rgRealData != null)
                Array.Clear(m_rgRealData, 0, m_rgRealData.Length);
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

            if (m_rgRealData != null)
            {
                if (sd.m_rgRealData == null)
                    throw new Exception("Both simple datums must have the same type of data!");

                for (int i = 0; i < m_rgRealData.Length; i++)
                {
                    m_rgRealData[i] -= sd.m_rgRealData[i];
                    if (m_rgRealData[i] != 0)
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
                m_rgRealData = Utility.Clone<double>(d.m_rgRealData);
                m_rgByteData = Utility.Clone<byte>(d.m_rgByteData);
            }
            else
            {
                m_rgRealData = d.m_rgRealData;
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
            m_tag = d.m_tag;
        }

        /// <summary>
        /// Clips the SimpleDatum to the last <i>nLastColumns</i> and returns the data.
        /// </summary>
        /// <typeparam name="T">Specifies base the type of data returned, either <i>double</i> or <i>float</i>.</typeparam>
        /// <param name="nLastColumns">Specifies the number of last columns of data to keep.</param>
        /// <returns>The last columns of data are returned.</returns>
        public List<T> ClipToLastColumns<T>(int nLastColumns = 10)
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
        /// Masks out all data except for the last columns of data.
        /// </summary>
        /// <param name="nLastColumsToRetain">Specifies the number of last columns to retain.</param>
        /// <param name="nMaskingValue">Specifies the value to use for the masked columns.</param>
        public void MaskOutAllButLastColumns(int nLastColumsToRetain, int nMaskingValue)
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
            m_rgRealData = null;
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
            m_rgRealData = rgRealData.ToArray();
            m_nLabel = nLabel;
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
                    return m_rgRealData.Length;
                else
                    return m_rgByteData.Length;
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
                if (typeof(T) == typeof(double))
                    return (T[])Convert.ChangeType(m_rgRealData.ToArray(), typeof(T[]));

                T[] rg = new T[m_rgRealData.Length];
                for (int i = 0; i<rg.Length; i++)
                {
                    rg[i] = (T)Convert.ChangeType(m_rgRealData[i], typeof(T));
                }

                return rg;
            }
            else
            {
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
        /// Return the real data as a <i>double</i> array after padding the data.
        /// </summary>
        /// <param name="rgData">Specifies the data.</param>
        /// <param name="nImagePadX">Specifies the amount to pad the data width.</param>
        /// <param name="nImagePadY">Specifies the amount to pad the data height.</param>
        /// <param name="nHeight">Specifies the height of the original data.</param>
        /// <param name="nWidth">Specifies the width of the original data.</param>
        /// <param name="nChannels">Specifies the number of channels in the original data.</param>
        /// <returns>The data is returned as a <i>double</i> array.</returns>
        public static double[] GetRealData(byte[] rgData, int nImagePadX, int nImagePadY, int nHeight, int nWidth, int nChannels)
        {
            List<double> rgReal = new List<double>();
            int nIdx = 0;

            while (nIdx < rgData.Length)
            {
                rgReal.Add(BitConverter.ToDouble(rgData, nIdx));
                nIdx += 8;
            }

            if (nImagePadX == 0 && nImagePadY == 0)
                return rgReal.ToArray();

            return PadData<double>(rgReal, nImagePadX, nImagePadY, nHeight, nWidth, nChannels);
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
            List<T> rgDataNew = new List<T>();

            for (int c = 0; c < nChannels; c++)
            {
                for (int y = 0; y < nHeight; y++)
                {
                    for (int i = 0; i < nImagePadX; i++)
                    {
                        rgDataNew.Add((T)Convert.ChangeType(0.0, typeof(T)));
                    }

                    for (int x = 0; x < nWidth; x++)
                    {
                        int nIdx = (c * nWidth * nHeight) + (y * nWidth) + x;

                        rgDataNew.Add(rgData[nIdx]);
                    }
                }

                for (int i = 0; i < nImagePadY; i++)
                {
                    for (int x = 0; x < nWidth + nImagePadX; x++)
                    {
                        rgDataNew.Add((T)Convert.ChangeType(0.0, typeof(T)));
                    }
                }
            }

            return rgDataNew.ToArray();
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
            return GetByteData(new List<double>(RealData));
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
            List<byte> rgByte = new List<byte>();

            foreach (double df in rgData)
            {
                rgByte.AddRange(BitConverter.GetBytes(df));
            }

            return rgByte.ToArray();
        }

        /// <summary>
        /// Decodes an array of <i>byte</i> values into a array of <i>double</i> values.
        /// </summary>
        /// <param name="rgData">Specifies the array of <i>byte</i> values containing the encoded <i>double</i> values.</param>
        /// <returns>The array of decoded <i>double</i> values is returned.</returns>
        public static double[] GetRealData(byte[] rgData)
        {
            List<double> rgData0 = new List<double>();
            int nIdx = 0;

            while (nIdx < rgData.Length)
            {
                rgData0.Add(BitConverter.ToDouble(rgData, nIdx));
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

            int nCount0 = (m_rgByteData != null) ? m_rgByteData.Length : m_rgRealData.Length;
            int nCount1 = (d.m_rgByteData != null) ? d.m_rgByteData.Length : d.m_rgRealData.Length;

            if (nCount0 != nCount1)
                throw new Exception("Datum counts do not match!");

            SimpleDatum d1 = new SimpleDatum(d, false);

            d1.m_rgRealData = null;
            d1.m_rgByteData = null;

            List<double> rgData = new List<double>();

            if (IsRealData && d.IsRealData)
            {
                for (int i = 0; i < d.RealData.Length; i++)
                {
                    rgData.Add((double)m_rgRealData[i] + (double)d.RealData[i]);
                }
            }
            else if (IsRealData)
            {
                for (int i = 0; i < m_rgRealData.Length; i++)
                {
                    rgData.Add((double)m_rgRealData[i] + (double)d.ByteData[i]);
                }
            }
            else if (d.IsRealData)
            {
                for (int i = 0; i < d.RealData.Length; i++)
                {
                    rgData.Add((double)m_rgByteData[i] + (double)d.RealData[i]);
                }
            }
            else
            {
                for (int i = 0; i < d.ByteData.Length; i++)
                {
                    rgData.Add((double)m_rgByteData[i] + (double)d.ByteData[i]);
                }
            }

            d1.m_rgRealData = rgData.ToArray();
            d1.m_bIsRealData = true;

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

            d1.m_rgRealData = null;
            d1.m_rgByteData = null;

            double[] rgRealData = null;
            byte[] rgByteData = null;

            if (m_bIsRealData && !bConvertToByte)
            {
                rgRealData = new double[m_rgRealData.Length];

                for (int i = 0; i < m_rgRealData.Length; i++)
                {
                    double dfDivVal = m_rgRealData[i] / dfVal;
                    rgRealData[i] = dfDivVal;
                }

                d1.m_bIsRealData = true;
            }
            else if (m_bIsRealData && bConvertToByte)
            {
                rgByteData = new byte[m_rgRealData.Length];

                for (int i = 0; i < m_rgRealData.Length; i++)
                {
                    double dfDivVal = m_rgRealData[i] / dfVal;
                    rgByteData[i] = (byte)dfDivVal;
                }

                d1.m_bIsRealData = false;
            }
            else
            {
                rgByteData = new byte[m_rgByteData.Length];

                for (int i = 0; i < m_rgByteData.Length; i++)
                {
                    double dfDivVal = (double)m_rgByteData[i] / dfVal;
                    rgByteData[i] = (byte)dfDivVal;
                }

                d1.m_bIsRealData = false;
            }

            d1.m_rgRealData = rgRealData;
            d1.m_rgByteData = rgByteData;

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
        public double[] RealData
        {
            get { return m_rgRealData; }
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
        public ANNOTATION_TYPE annoation_type
        {
            get { return m_nAnnoationType; }
            set { m_nAnnoationType = value; }
        }

        /// <summary>
        /// When using annoations, each annotation group contains an annotation for a particular class used with SSD.
        /// </summary>
        public List<AnnotationGroup> annotation_group
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
            Datum d = ImageData.GetImageData(bmpNew, Channels, IsRealData, Label);

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
            return "Idx = " + m_nIndex.ToString("N0");
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
                for (int i=0; i<m_rgRealData.Length && i < nMaxItems; i++)
                {
                    str += m_rgRealData[i].ToString() + ",";
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
            else
                rgSums = new float[rgImg[0].RealData.Length];

            Stopwatch sw = new Stopwatch();

            try
            {
                sw.Start();

                for (int i = 0; i < rgImg.Length; i++)
                {
                    if (rgImg[i].ByteData != null)
                    {
                        for (int n = 0; n < rgSums.Length; n++)
                        {
                            rgSums[n] += rgImg[i].ByteData[n];
                        }
                    }
                    else
                    {
                        for (int n = 0; n < rgSums.Length; n++)
                        {
                            rgSums[n] += (float)rgImg[i].RealData[n];
                        }
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

                byte[] rgbData = new byte[rgSums.Length];
                double[] rgdfData = new double[rgSums.Length];

                for (int n = 0; n < rgSums.Length; n++)
                {
                    float fMean = rgSums[n] / (float)rgImg.Length;
                    byte bMean = (byte)fMean;
                    double dfMean = fMean;

                    rgbData[n] = bMean;
                    rgdfData[n] = dfMean;
                }

                SimpleDatum d = new SimpleDatum(rgImg[0], false);

                if (rgImg[0].ByteData != null)
                    d.SetData(new List<byte>(rgbData), -1, true);
                else
                    d.SetData(new List<double>(rgdfData), -1, true);

                return d;
            }
            finally
            {
            }
        }
    }
}
