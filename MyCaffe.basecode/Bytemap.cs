using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The Bytemap operates similar to a bitmap but is actually just an array of bytes.
    /// </summary>
    [Serializable]
    public class Bytemap
    {
        int m_nChannels;
        int m_nHeight;
        int m_nWidth;
        int m_nChannelOffset;
        byte[] m_rgData;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nChannels">Specifies the number of channels in the map.</param>
        /// <param name="nHeight">Specifies the height of the map.</param>
        /// <param name="nWidth">Specifies the width of the map.</param>
        /// <param name="rgData">Optionally, specifies the data to use.</param>
        public Bytemap(int nChannels, int nHeight, int nWidth, byte[] rgData = null)
        {
            int nSize = nChannels * nHeight * nWidth;
            m_rgData = (rgData != null) ? rgData : new byte[nSize];
            m_nChannels = nChannels;
            m_nHeight = nHeight;
            m_nWidth = nWidth;
            m_nChannelOffset = nHeight * nWidth;
        }

        /// <summary>
        /// The constructorl
        /// </summary>
        /// <param name="data">Specifies another Bytemap to copy.</param>
        public Bytemap(Bytemap data)
        {
            int nChannels = data.Channels;
            int nHeight = data.Height;
            int nWidth = data.Width;

            int nSize = nChannels * nHeight * nWidth;
            m_rgData = new byte[nSize];
            m_nChannels = nChannels;
            m_nHeight = nHeight;
            m_nWidth = nWidth;
            m_nChannelOffset = nHeight * nWidth;

            Array.Copy(data.Bytes, m_rgData, nSize);
        }

        /// <summary>
        /// Set a given pixel to a given color.
        /// </summary>
        /// <param name="nX">Specifies the x location of the pixel.</param>
        /// <param name="nY">Specifies the y location of the pixel.</param>
        /// <param name="clr">Specifies the color to set the pixel.</param>
        public void SetPixel(int nX, int nY, byte clr)
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
        /// Set a given pixel to a given color.
        /// </summary>
        /// <param name="nX">Specifies the x location of the pixel.</param>
        /// <param name="nY">Specifies the y location of the pixel.</param>
        /// <param name="clr">Specifies the color to set the pixel.</param>
        public void SetPixel(int nX, int nY, Color clr)
        {
            byte bR = (byte)clr.R;
            byte bG = (byte)clr.G;
            byte bB = (byte)clr.B;

            int nIdx = nY * m_nWidth + nX;
            m_rgData[nIdx] = bR;

            if (m_nChannels == 3)
            {
                m_rgData[nIdx + m_nChannelOffset] = bG;
                m_rgData[nIdx + m_nChannelOffset * 2] = bB;
            }
        }

        /// <summary>
        /// Get the color of a pixel in the map.
        /// </summary>
        /// <param name="nX">Specifies the x location of the pixel.</param>
        /// <param name="nY">Specifies the y location of the pixel.</param>
        /// <returns>The color of the pixel is returned.</returns>
        public Color GetPixel(int nX, int nY)
        {
            int nIdx = nY * m_nWidth + nX;
            byte bR = m_rgData[nIdx];
            byte bG = bR;
            byte bB = bR;

            if (m_nChannels == 3)
            {
                bG = m_rgData[nIdx + m_nChannelOffset];
                bB = m_rgData[nIdx + m_nChannelOffset * 2];
            }

            return Color.FromArgb(bR, bG, bB);
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
        public byte[] Bytes
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

        /// <summary>
        /// Converts the Bytemap into a Bitmap.
        /// </summary>
        /// <returns>A new bitmap is returned.</returns>
        public Bitmap ToImage()
        {
            Bitmap bmp = new Bitmap(m_nWidth, m_nHeight);
            LockBitmap bmpA = new LockBitmap(bmp);

            bmpA.LockBits();

            for (int y = 0; y < m_nHeight; y++)
            {
                for (int x = 0; x < m_nWidth; x++)
                {
                    Color clr = GetPixel(x, y);
                    bmpA.SetPixel(x, y, clr);
                }
            }

            bmpA.UnlockBits();

            return bmp;
        }

        /// <summary>
        /// Converts a bitmap into a new Bytemap.
        /// </summary>
        /// <param name="bmp">Specifies the bitmap.</param>
        /// <returns>A new Bytemap is returned.</returns>
        public static Bytemap FromImage(Bitmap bmp)
        {
            Bytemap data = new Bytemap(3, bmp.Height, bmp.Width);
            LockBitmap bmpA = new LockBitmap(bmp);

            bmpA.LockBits();

            for (int y = 0; y < bmp.Height; y++)
            {
                for (int x = 0; x < bmp.Width; x++)
                {
                    Color clr = bmpA.GetPixel(x, y);
                    data.SetPixel(x, y, clr);
                }
            }

            bmpA.UnlockBits();

            return data;
        }
    }
}
