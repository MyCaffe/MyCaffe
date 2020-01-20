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
        /// <param name="nLabel">Optionally, specifies the known label of the data (default = -1).</param>
        /// <param name="dtTime">Optionally, specifies a time-stamp associated with the data (default = null).</param>
        /// <param name="nBoost">Optionally, specifies the boost to use with the data (default = 0, where a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Optionally, specifies whether or not the label was auto-generated (default = false).</param>
        /// <param name="nIdx">Optionally, specifies the index of the data (default = -1).</param>
        public Datum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel = -1, DateTime? dtTime = null, int nBoost = 0, bool bAutoLabeled = false, int nIdx = -1)
            : base(bIsReal, nChannels, nWidth, nHeight, nLabel, dtTime, nBoost, bAutoLabeled, nIdx)
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
        /// <param name="nBoost">Specifies the boost to use with the data (a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Specifies whether or not the label was auto-generated.</param>
        /// <param name="nIdx">Specifies the index of the data.</param>
        public Datum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, List<byte> rgData, int nBoost, bool bAutoLabeled, int nIdx)
            : base(bIsReal, nChannels, nWidth, nHeight, nLabel, dtTime, rgData, nBoost, bAutoLabeled, nIdx)
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
        /// <param name="rgfData">Specifies the data as a list of <i>double</i> (expects <i>bIsReal</i> = <i>true</i>).</param>
        /// <param name="nBoost">Specifies the boost to use with the data (a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Specifies whether or not the label was auto-generated.</param>
        /// <param name="nIdx">Specifies the index of the data.</param>
        public Datum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, List<double> rgfData, int nBoost, bool bAutoLabeled, int nIdx)
            : base(bIsReal, nChannels, nWidth, nHeight, nLabel, dtTime, rgfData, nBoost, bAutoLabeled, nIdx)
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
        /// <param name="rgfData">Specifies the data as a list of <i>float</i> (expects <i>bIsReal</i> = <i>true</i>).</param>
        /// <param name="nBoost">Specifies the boost to use with the data (a value of 0 indicates no boost).</param>
        /// <param name="bAutoLabeled">Specifies whether or not the label was auto-generated.</param>
        /// <param name="nIdx">Specifies the index of the data.</param>
        public Datum(bool bIsReal, int nChannels, int nWidth, int nHeight, int nLabel, DateTime dtTime, List<float> rgfData, int nBoost, bool bAutoLabeled, int nIdx)
            : base(bIsReal, nChannels, nWidth, nHeight, nLabel, dtTime, rgfData, nBoost, bAutoLabeled, nIdx)
        {
        }

        /// <summary>
        /// The Datum constructor.
        /// </summary>
        /// <param name="d">Specifies a SimpleDatum used to create this new Datum.</param>
        /// <param name="bCopyData">Specifies whether or not to copy the data, or just share it.</param>
        public Datum(SimpleDatum d, bool bCopyData = false)
            : base(d, bCopyData)
        {
        }

        /// <summary>
        /// The Datum constructor.
        /// </summary>
        /// <param name="d">Specifies another Datum to copy when creating this Datum.</param>
        /// <param name="bCopyData">Specifies whether or not to copy the data, or just share it.</param>
        public Datum(Datum d, bool bCopyData = false)
            : base(d, bCopyData)
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
        /// Returns the data as an array of <i>float</i>.  The datum must be initialized with <i>bIsReal</i> = <i>true</i> with <i>float</i> data.
        /// </summary>
        public float[] float_data
        {
            get { return RealDataF; }
        }

        /// <summary>
        /// Returns the data as an array of <i>double</i>.  The datum must be initialized with <i>bIsReal</i> = <i>true</i> with <i>double</i> data.
        /// </summary>
        public double[] double_data
        {
            get { return RealDataD; }
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
    }
}
