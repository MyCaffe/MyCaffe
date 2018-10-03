using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The Realmap operates similar to a bitmap but is actually just an array of doubles.
    /// </summary>
    [Serializable]
    public class Valuemap
    {
        int m_nChannels;
        int m_nHeight;
        int m_nWidth;
        int m_nChannelOffset;
        double[] m_rgData;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nChannels">Specifies the number of channels in the map.</param>
        /// <param name="nHeight">Specifies the height of the map.</param>
        /// <param name="nWidth">Specifies the width of the map.</param>
        /// <param name="rgData">Optionally, specifies the data to use.</param>
        public Valuemap(int nChannels, int nHeight, int nWidth, double[] rgData = null)
        {
            int nSize = nChannels * nHeight * nWidth;
            m_rgData = (rgData != null) ? rgData : new double[nSize];
            m_nChannels = nChannels;
            m_nHeight = nHeight;
            m_nWidth = nWidth;
            m_nChannelOffset = nHeight * nWidth;
        }

        /// <summary>
        /// The constructorl
        /// </summary>
        /// <param name="data">Specifies another Realmap to copy.</param>
        public Valuemap(Valuemap data)
        {
            int nChannels = data.Channels;
            int nHeight = data.Height;
            int nWidth = data.Width;

            int nSize = nChannels * nHeight * nWidth;
            m_rgData = new double[nSize];
            m_nChannels = nChannels;
            m_nHeight = nHeight;
            m_nWidth = nWidth;
            m_nChannelOffset = nHeight * nWidth;

            Array.Copy(data.Values, m_rgData, nSize);
        }

        /// <summary>
        /// Set a given pixel to a given color.
        /// </summary>
        /// <param name="nX">Specifies the x location of the pixel.</param>
        /// <param name="nY">Specifies the y location of the pixel.</param>
        /// <param name="clr">Specifies the value to set the pixel.</param>
        public void SetPixel(int nX, int nY, double clr)
        {
            int nIdx = nY * m_nWidth + nX;
            m_rgData[nIdx] = clr;

            if (m_nChannels == 3)
            {
                m_rgData[nIdx + m_nChannelOffset] = clr;
                m_rgData[nIdx + m_nChannelOffset * 2] = clr;
            }
        }

        /// <summary>
        /// Get the value of a pixel in the map.
        /// </summary>
        /// <param name="nX">Specifies the x location of the pixel.</param>
        /// <param name="nY">Specifies the y location of the pixel.</param>
        /// <returns>The color of the pixel is returned.</returns>
        public double GetPixel(int nX, int nY)
        {
            int nIdx = nY * m_nWidth + nX;
            return m_rgData[nIdx];
        }

        /// <summary>
        /// Reset all values to zero.
        /// </summary>
        public void Clear()
        {
            Array.Clear(m_rgData, 0, m_rgData.Length);
        }

        /// <summary>
        /// Specifies the data itself.
        /// </summary>
        public double[] Values
        {
            get { return m_rgData; }
        }

        /// <summary>
        /// Specifies the channels of the data.
        /// </summary>
        public int Channels
        {
            get { return m_nChannels; }
        }

        /// <summary>
        /// Specifies the height of the data.
        /// </summary>
        public int Height
        {
            get { return m_nHeight; }
        }

        /// <summary>
        /// Specifies the width of the data.
        /// </summary>
        public int Width
        {
            get { return m_nWidth; }
        }
    }
}
