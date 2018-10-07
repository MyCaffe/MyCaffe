using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The ImageData class is a helper class used to convert between Datum, other raw data, and Images such as a Bitmap.
    /// </summary>
    public class ImageData
    {
        /// <summary>
        /// The GetImageData function converts a Bitmap into a Datum.
        /// </summary>
        /// <param name="bmp">Specifies the Bitmap containing the image.</param>
        /// <param name="nChannels">Specifies the number of channels contained in the Bitmap (e.g. 3 = color, 1 = black and white).</param>
        /// <param name="bDataIsReal">Specifies whether or not to add each color to the List of <i>double</i> or to the list of <i>byte</i>.  Using the <i>byte</i> array is more common for it already separates a 3 color Bitmap into 3 channels of data.</param>
        /// <param name="nLabel">Specifies the known label.</param>
        /// <returns>The Datum representing the image is returned.</returns>
        public static Datum GetImageData(Bitmap bmp, int nChannels, bool bDataIsReal, int nLabel)
        {
            if (nChannels != 1 && nChannels != 3)
                throw new Exception("Images only support either 1 or 3 channels.");

            List<byte>[] rgrgByteData = new List<byte>[nChannels];
            List<double>[] rgrgRealData = new List<double>[nChannels];

            for (int i = 0; i < nChannels; i++)
            {
                rgrgByteData[i] = new List<byte>();
                rgrgRealData[i] = new List<double>();
            }

            LockBitmap bmp1 = new LockBitmap(bmp);

            try
            {
                bmp1.LockBits();
                for (int y = 0; y < bmp1.Height; y++)
                {
                    for (int x = 0; x < bmp1.Width; x++)
                    {
                        Color clr = bmp1.GetPixel(x, y);

                        if (nChannels == 1)
                        {
                            if (bDataIsReal)
                                rgrgRealData[0].Add(clr.ToArgb());
                            else
                                rgrgByteData[0].Add((byte)((clr.R * 0.3) + (clr.G * 0.59) + (clr.B * 0.11)));
                        }
                        else
                        {
                            if (bDataIsReal)
                            {
                                rgrgRealData[0].Add(clr.R);
                                rgrgRealData[1].Add(clr.G);
                                rgrgRealData[2].Add(clr.B);
                            }
                            else
                            {
                                rgrgByteData[0].Add(clr.R);
                                rgrgByteData[1].Add(clr.G);
                                rgrgByteData[2].Add(clr.B);
                            }
                        }
                    }
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                bmp1.UnlockBits();
            }

            List<byte> rgByteData = new List<byte>();
            List<double> rgRealData = new List<double>();

            for (int i = 0; i < nChannels; i++)
            {
                rgByteData.AddRange(rgrgByteData[i]);
                rgRealData.AddRange(rgrgRealData[i]);
            }

            if (bDataIsReal)
                return new Datum(true, nChannels, bmp.Width, bmp.Height, nLabel, DateTime.MinValue, null, new List<double>(rgRealData), 0, false, -1);
            else
                return new Datum(false, nChannels, bmp.Width, bmp.Height, nLabel, DateTime.MinValue, new List<byte>(rgByteData), null, 0, false, -1);
        }

        /// <summary>
        /// The GetImageData function converts an array of type 'T' into a Datum.
        /// </summary>
        /// <param name="rgData">Specifies the array of type 'T'.</param>
        /// <param name="nChannels">Specifies the number of channels contained in the Bitmap (e.g. 3 = color, 1 = black and white).</param>
        /// <param name="nHeight">Specifies the height of the data.</param>
        /// <param name="nWidth">Specifies the width of the data.</param>
        /// <param name="bDataIsReal">Specifies whether or not to add each color to the List of <i>double</i> or to the list of <i>byte</i>.  Using the <i>byte</i> array is more common for it already separates a 3 color Bitmap into 3 channels of data.</param>
        /// <param name="nStartIdx">Specifies where to start the conversion within the data.</param>
        /// <param name="nCount">Specifies the number of items within the data to convert.</param>
        /// <returns>The Datum representing the image is returned.</returns>
        public static Datum GetImageData<T>(T[] rgData, int nChannels, int nHeight, int nWidth, bool bDataIsReal, int nStartIdx = 0, int nCount = -1)
        {
            if (nCount == -1)
                nCount = rgData.Length;

            if (nChannels != 1 && nChannels != 3)
                throw new Exception("Images only support either 1 or 3 channels.");

            if (bDataIsReal)
            {
                List<double> rgRealData = new List<double>();

                for (int i = nStartIdx; i < nStartIdx + nCount; i++)
                {
                    rgRealData.Add((double)Convert.ChangeType(rgData[i], typeof(double)));
                }

                return new Datum(true, nChannels, nWidth, nHeight, 0, DateTime.MinValue, null, new List<double>(rgRealData), 0, false, -1);
            }
            else
            {
                List<byte> rgByteData = new List<byte>();
                float[] rgDataF = Utility.ConvertVecF<T>(rgData);

                for (int i = nStartIdx; i < nStartIdx + nCount; i++)
                {
                    float fVal = rgDataF[i];

                    if (float.IsInfinity(fVal) || float.IsNaN(fVal))
                        fVal = 0;

                    rgByteData.Add((byte)fVal);
                }

                return new Datum(false, nChannels, nWidth, nHeight, 0, DateTime.MinValue, new List<byte>(rgByteData), null, 0, false, -1);
            }
        }

        /// <summary>
        /// Converts a SimplDatum (or Datum) into an image, optionally using a ColorMapper.
        /// </summary>
        /// <param name="d">Specifies the Datum to use.</param>
        /// <param name="clrMap">Optionally, specifies a color mapper to use when converting each value into a color (default = null, not used).</param>
        /// <param name="rgClrOrder">Optionally, specifies the color ordering. Note, this list must have the same number of elements as there are channels.</param>
        /// <returns>The Image of the data is returned.</returns>
        public static Bitmap GetImage(SimpleDatum d, ColorMapper clrMap = null, List<int> rgClrOrder = null)
        {
            if (d.Channels != 1 && d.Channels != 3)
                throw new Exception("Standard images only support either 1 or 3 channels.");

            Bitmap bmp = new Bitmap(d.Width, d.Height);
            List<byte>[] rgrgByteData = new List<byte>[d.Channels];
            List<double>[] rgrgRealData = new List<double>[d.Channels];
            int nOffset = 0;
            int nCount = d.Height * d.Width;
            bool bDataIsReal = (d.RealData != null && d.RealData.Length > 0) ? true : false;
            double dfMin = 1;
            double dfMax = 0;


            for (int i = 0; i < d.Channels; i++)
            {
                List<byte> rgByteData = new List<byte>();
                List<double> rgRealData = new List<double>();
                int nChIdx = i;

                if (rgClrOrder != null)
                    nChIdx = rgClrOrder[i];

                if (bDataIsReal)
                {
                    for (int j = 0; j < nCount; j++)
                    {
                        double dfVal = d.RealData[nOffset + j];
                        dfMin = Math.Min(dfMin, dfVal);
                        dfMax = Math.Max(dfMax, dfVal);
                        rgRealData.Add(dfVal);
                    }

                    rgrgRealData[nChIdx] = rgRealData;
                }
                else
                {
                    for (int j = 0; j < nCount; j++)
                    {
                        rgByteData.Add(d.ByteData[nOffset + j]);
                    }

                    rgrgByteData[nChIdx] = rgByteData;
                }

                nOffset += nCount;
            }

            LockBitmap bmp1 = new LockBitmap(bmp);

            try
            {
                bmp1.LockBits();

                for (int y = 0; y < bmp1.Height; y++)
                {
                    for (int x = 0; x < bmp1.Width; x++)
                    {
                        Color clr;
                        int nIdx = (y * bmp1.Width) + x;

                        if (d.Channels == 1)
                        {
                            if (bDataIsReal)
                            {
                                if (dfMin >= 0 && dfMax <= 1.0)
                                {
                                    int nG = (int)(rgrgRealData[0][nIdx] * 255.0);
                                    if (nG < 0)
                                        nG = 0;
                                    if (nG > 255)
                                        nG = 255;

                                    clr = Color.FromArgb(nG, nG, nG);
                                }
                                else
                                {
                                    clr = Color.FromArgb((int)rgrgRealData[0][nIdx]);
                                }

                                if (clrMap != null)
                                    clr = clrMap.GetColor(clr.ToArgb());
                            }
                            else
                            {
                                clr = Color.FromArgb((int)rgrgByteData[0][nIdx], (int)rgrgByteData[0][nIdx], (int)rgrgByteData[0][nIdx]);
                            }
                        }
                        else
                        {
                            if (bDataIsReal)
                            {
                                clr = Color.FromArgb((int)rgrgRealData[0][nIdx], (int)rgrgRealData[1][nIdx], (int)rgrgRealData[2][nIdx]);

                                if (clrMap != null)
                                    clr = clrMap.GetColor(clr.ToArgb());
                            }
                            else
                                clr = Color.FromArgb((int)rgrgByteData[0][nIdx], (int)rgrgByteData[1][nIdx], (int)rgrgByteData[2][nIdx]);
                        }

                        bmp1.SetPixel(x, y, clr);
                    }
                }
            }
            catch (Exception excpt)
            {
                throw excpt;
            }
            finally
            {
                bmp1.UnlockBits();
            }

            return bmp;
        }

        /// <summary>
        /// Converts a list of KeyValuePairs into an image using a ColorMapper.
        /// </summary>
        /// <param name="rg">Specifies a KeyValuePair where the Value is converted to a color.</param>
        /// <param name="sz">Specifies the size of the image.</param>
        /// <param name="clrMap">Specifies a color mapper to use when converting each value into a color.</param>
        /// <returns>The Image of the data is returned.</returns>
        public static Bitmap GetImage(List<KeyValuePair<int, double>> rg, Size sz, ColorMapper clrMap)
        {
            Bitmap bmp = new Bitmap(sz.Width, sz.Height);
            int nSize = (int)Math.Ceiling(Math.Sqrt(rg.Count));
            float fX = 0;
            float fY = 0;
            float fIncX = (float)sz.Width / (float)nSize;
            float fIncY = (float)sz.Height / (float)nSize;

            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.FillRectangle(Brushes.Black, new RectangleF(0, 0, bmp.Width, bmp.Height));

                for (int i=0; i<rg.Count; i++)
                {
                    Brush br = new SolidBrush(clrMap.GetColor(rg[i].Value));
                    g.FillRectangle(br, fX, fY, fIncX, fIncY);
                    br.Dispose();
                     
                    fX += fIncX;

                    if (fX >= bmp.Width)
                    {
                        fX = 0;
                        fY += fIncY;
                    }
                }
            }

            return bmp;
        }

        private static double getColorAverage(Bitmap bmp, int nRow)
        {
            double dfColorTotal = 0;

            for (int x = 0; x < bmp.Width; x++)
            {
                Color clr = bmp.GetPixel(x, nRow);

                dfColorTotal += clr.R;
                dfColorTotal += clr.B;
                dfColorTotal += clr.G;
            }

            return dfColorTotal / bmp.Width;
        }

        private static void offsetInward(Bitmap bmp, ref int nYBottom, ref int nYTop, int nSegmentHeight)
        {
            double dfAve1;
            double dfAve2;

            while (nYBottom - nYTop > nSegmentHeight)
            {
                dfAve1 = getColorAverage(bmp, nYBottom - 1);

                if (dfAve1 == 0)
                    nYBottom--;

                if (nYBottom - nYTop > nSegmentHeight)
                {
                    dfAve2 = getColorAverage(bmp, nYTop);

                    if (dfAve2 == 0 || dfAve2 < dfAve1)
                        nYTop++;
                    else
                        nYBottom--;

                    if (dfAve1 > 0 && dfAve2 > 0)
                        break;
                }
            }
        }

        private static double[] getImageData(Bitmap bmp, int nYBottom, int nYTop, int nSegmentHeight)
        {
            double[] rgrgData = new double[nSegmentHeight * bmp.Width];

            offsetInward(bmp, ref nYBottom, ref nYTop, nSegmentHeight);

            for (int y = 0; y < nSegmentHeight; y++)
            {
                for (int x = 0; x < bmp.Width; x++)
                {
                    Color clr = bmp.GetPixel(x, y + nYTop);

                    int nDataIdx = (y * bmp.Width) + x;
                    int nR = clr.R;
                    int nG = clr.G;
                    int nB = clr.B;
                    int nColor = (nR << 16) + (nG << 8) + nB;
                    double dfClr = (double)nColor / (double)0xFFFFFF;

                    rgrgData[nDataIdx] = dfClr;
                }
            }

            return rgrgData;
        }
    }
}
