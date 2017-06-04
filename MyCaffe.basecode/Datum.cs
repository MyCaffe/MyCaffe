using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The Datum class is a simple wrapper to the SimpleDatum class to ensure compatibility with the original C++ %Caffe code.
    /// </summary>
    public class Datum : SimpleDatum 
    {
        object m_tag;
        string m_strTagName = null;

        /// <summary>
        /// The Datum constructor.
        /// </summary>
        public Datum() : base()
        {
        }

        /// <summary>
        /// The Datum constructor.
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
        public Datum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, List<byte> rgData, List<double> rgfData, int nBoost, bool bAutoLabeled, int nIdx)
            : base(bIsReal, nChannels, nWidth, nHeight, nLabel, dtTime, rgData, rgfData, nBoost, bAutoLabeled, nIdx)
        {
        }

        /// <summary>
        /// The Datum constructor.
        /// </summary>
        /// <param name="d">Specifies a SimpleDatum used to create this new Datum.</param>
        public Datum(SimpleDatum d)
            : base(d)
        {
        }

        /// <summary>
        /// The Datum constructor.
        /// </summary>
        /// <param name="d">Specifies another Datum to copy when creating this Datum.</param>
        public Datum(Datum d)
            : base(d)
        {
        }

        /// <summary>
        /// Returns the number of channels in the data.
        /// </summary>
        public int channels
        {
            get { return Channels; }
        }

        /// <summary>
        /// Specifies the height of the data.
        /// </summary>
        public int height
        {
            get { return Height; }
        }

        /// <summary>
        /// Specifies the width of the data.
        /// </summary>
        public int width
        {
            get { return Width; }
        }

        /// <summary>
        /// Specifies the index of the data.
        /// </summary>
        public int index
        {
            get { return Index; }
        }

        /// <summary>
        /// Returns the data as an array of <i>doubles</i>.  The datum must be initialized with <i>bIsReal</i> = <i>true</i>.
        /// </summary>
        public double[] float_data
        {
            get { return RealData; }
        }

        /// <summary>
        /// Returns the non-real data as an array of <i>bytes</i>.  The datum must be initialized with <i>bIsReal</i> = <i>false</i>.
        /// </summary>
        public byte[] data
        {
            get { return ByteData; }
        }

        /// <summary>
        /// Returns the known label of the data.
        /// </summary>
        public int label
        {
            get { return Label; }
        }

        /// <summary>
        /// Returns a user-defined tag associated with the data.
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
        /// Copies another Datum into this one.
        /// </summary>
        /// <param name="d">Specifies the other Datum to copy.</param>
        public void Copy(Datum d)
        {
            base.Copy(d);
            m_tag = d.Tag;
            m_strTagName = d.m_strTagName;
        }
    }
}
