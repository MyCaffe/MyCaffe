using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The ImageTools class is a helper class used to manipulate image data.
    /// </summary>
    public class ImageTools
    {
        /// <summary>
        /// Defines the odering for which the AdjustContrast applies brightness, contrast and gamma adjustments.
        /// </summary>
        public enum ADJUSTCONTRAST_ORDERING
        {
            /// <summary>
            /// Applies brightness, then contrast, then adjust gamma.
            /// </summary>
            BRIGHTNESS_CONTRAST_GAMMA,
            /// <summary>
            /// Applies brightness, then adjusts gamma, then adjusts contrast.
            /// </summary>
            BRIGHTNESS_GAMMA_CONTRAST
        }

        /// <summary>
        /// Resize the image to the specified width and height.
        /// </summary>
        /// <param name="image">The image to resize.</param>
        /// <param name="width">The width to resize to.</param>
        /// <param name="height">The height to resize to.</param>
        /// <returns>The resized image.</returns>
        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            if (image.Width == width && image.Height == height)
                return new Bitmap(image);

            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        /// <summary>
        /// The AdjustContrast function adjusts the brightness, contrast and gamma of the image and returns the newly adjusted image.
        /// </summary>
        /// <param name="bmp">Specifies the image to adjust.</param>
        /// <param name="fBrightness">Specifies the brightness to apply.</param>
        /// <param name="fContrast">Specifies the contrast to apply.</param>
        /// <param name="fGamma">Specifies the gamma to apply.</param>
        /// <returns>The updated image is returned.</returns>
        public static Bitmap AdjustContrast(Image bmp, float fBrightness = 0.0f, float fContrast = 1.0f, float fGamma = 1.0f)
        {
            float fAdjBrightNess = fBrightness - 1.0f;
            float[][] ptsArray =
            {
                new float[] { fContrast, 0, 0, 0, 0 },  // scale red.
                new float[] { 0, fContrast, 0, 0, 0 },  // scale green.
                new float[] { 0, 0, fContrast, 0, 0 },  // scale blue.
                new float[] { 0, 0, 0, 1.0f, 0 }, // don't scale alpha.
                new float[] { fAdjBrightNess, fAdjBrightNess, fAdjBrightNess, 0, 1 }
            };

            ImageAttributes imageAttributes = new ImageAttributes();
            imageAttributes.ClearColorMatrix();
            imageAttributes.SetColorMatrix(new ColorMatrix(ptsArray), ColorMatrixFlag.Default, ColorAdjustType.Bitmap);
            imageAttributes.SetGamma(fGamma, ColorAdjustType.Bitmap);

            Bitmap bmpNew = new Bitmap(bmp.Width, bmp.Height);

            using (Graphics g = Graphics.FromImage(bmpNew))
            {
                g.DrawImage(bmp, new Rectangle(0, 0, bmpNew.Width, bmpNew.Height), 0, 0, bmp.Width, bmp.Height, GraphicsUnit.Pixel, imageAttributes);
            }

            return bmpNew;
        }

        private static int truncate(int n)
        {
            if (n < 0)
                return 0;
            else if (n > 255)
                return 255;       
            else
                return n;
        }

        /// <summary>
        /// Apply the brightness to the pixel.
        /// </summary>
        /// <remarks>
        /// @see [Image Processing Algorithms Part 4: Brightness Adjustment](https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-4-brightness-adjustment/)
        /// </remarks>
        /// <param name="nR">Specifies the red color of the pixel.</param>
        /// <param name="nG">Specifies the green color of the pixel.</param>
        /// <param name="nB">Specifies the blue color of the pixel.</param>
        /// <param name="fBrightness">Specifies the brightness adjustment in the range [1,255] (note 255 will cause complete saturation).</param>
        private static void applyBrightness(ref int nR, ref int nG, ref int nB, float fBrightness)
        {
            nR = truncate(nR + (int)fBrightness);
            nG = truncate(nG + (int)fBrightness);
            nB = truncate(nB + (int)fBrightness);
        }

        /// <summary>
        /// Apply the contrast to the pixel.
        /// </summary>
        /// <remarks>
        /// @see [Image Processing Algorithms Part 5: Contrast Adjustment](https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/)
        /// </remarks>
        /// <param name="nR">Specifies the red color of the pixel.</param>
        /// <param name="nG">Specifies the green color of the pixel.</param>
        /// <param name="nB">Specifies the blue color of the pixel.</param>
        /// <param name="fContrast">Specifies the brightness adjustment in the range [0.0, 2.0].</param>
        private static void applyContrast(ref int nR, ref int nG, ref int nB, float fContrast)
        {
            double dfC = (fContrast - 1.0f) * 255.0;
            double dfFactor = (259.0f * (dfC + 255.0)) / (255.0 * (259 - dfC));

            nR = truncate((int)(dfFactor * (nR - 128) + 128));
            nG = truncate((int)(dfFactor * (nG - 128) + 128));
            nB = truncate((int)(dfFactor * (nB - 128) + 128));
        }

        /// <summary>
        /// Apply the gamma correction to the pixel.
        /// </summary>
        /// <remarks>
        /// @see [Image Processing Algorithms Part 6: Gamma Correction](https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-6-gamma-correction/)
        /// </remarks>
        /// <param name="nR">Specifies the red color of the pixel.</param>
        /// <param name="nG">Specifies the green color of the pixel.</param>
        /// <param name="nB">Specifies the blue color of the pixel.</param>
        /// <param name="fGamma">Specifies the brightness adjustment in the range [0.01, 7.99].</param>
        private static void applyGamma(ref int nR, ref int nG, ref int nB, float fGamma)
        {
            double dfG = 1.0 / fGamma;

            nR = (int)(255.0 * Math.Pow((nR / 255.0), dfG));
            nG = (int)(255.0 * Math.Pow((nG / 255.0), dfG));
            nB = (int)(255.0 * Math.Pow((nB / 255.0), dfG));
        }

        /// <summary>
        /// The AdjustContrast function adjusts the brightness, contrast and gamma of the image and returns the newly adjusted image.
        /// </summary>
        /// <param name="sd">Specifies the SimpleDatum to adjust.</param>
        /// <param name="fBrightness">Specifies the brightness to apply.</param>
        /// <param name="fContrast">Specifies the contrast to apply.</param>
        /// <param name="fGamma">Specifies the gamma to apply.</param>
        /// <param name="ordering">Specifies the ordering for which the brightness, contrast and gamma are applied.</param>
        /// <returns>The updated image is returned.</returns>
        public static void AdjustContrast(SimpleDatum sd, float fBrightness = 0.0f, float fContrast = 1.0f, float fGamma = 1.0f, ADJUSTCONTRAST_ORDERING ordering = ADJUSTCONTRAST_ORDERING.BRIGHTNESS_CONTRAST_GAMMA)
        {
            if (fBrightness == 0 && fContrast == 1 && fGamma == 1)
                return;

            if (sd.IsRealData)
                throw new Exception("AdjustContrast only valid on ByteData!");

            int nC = sd.Channels;
            int nH = sd.Height;
            int nW = sd.Width;

            if (nC != 3)
                throw new Exception("AdjustContrast requires 3 channels!");

            int nSpatialDim = nH * nW;
            for (int h = 0; h < nH; h++)
            {
                for (int w = 0; w < nW; w++)
                {
                    int nIdxR = (nSpatialDim * 0) + ((h * nW) + w);
                    int nIdxG = (nSpatialDim * 1) + ((h * nW) + w);
                    int nIdxB = (nSpatialDim * 2) + ((h * nW) + w);
                    int nR = sd.ByteData[nIdxR];
                    int nG = sd.ByteData[nIdxG];
                    int nB = sd.ByteData[nIdxB];
                    
                    if (fBrightness != 0)
                        applyBrightness(ref nR, ref nG, ref nB, fBrightness);

                    if (ordering == ADJUSTCONTRAST_ORDERING.BRIGHTNESS_CONTRAST_GAMMA)
                    {
                        if (fContrast != 1.0f)
                            applyContrast(ref nR, ref nG, ref nB, fContrast);

                        if (fGamma != 1.0f)
                            applyGamma(ref nR, ref nG, ref nB, fGamma);
                    }
                    else
                    {
                        if (fGamma != 1.0f)
                            applyGamma(ref nR, ref nG, ref nB, fGamma);

                        if (fContrast != 1.0f)
                            applyContrast(ref nR, ref nG, ref nB, fContrast);
                    }

                    sd.ByteData[nIdxR] = (byte)nR;
                    sd.ByteData[nIdxG] = (byte)nG;
                    sd.ByteData[nIdxB] = (byte)nB;
                }
            }
        }

        /// <summary>
        /// Converts an Image into an array of <i>byte</i>.
        /// </summary>
        /// <param name="imageIn">Specifies the Image.</param>
        /// <returns>The array of <i>byte</i> is returned.</returns>
        public static byte[] ImageToByteArray(Image imageIn)
        {
            MemoryStream ms = new MemoryStream();
            imageIn.Save(ms, System.Drawing.Imaging.ImageFormat.Png);

            return ms.ToArray();
        }

        /// <summary>
        /// Converts an array of <i>byte</i> into an Image.
        /// </summary>
        /// <param name="byteArrayIn">Specifies the array of <i>byte</i>.</param>
        /// <returns>The Image is returned.</returns>
        public static Image ByteArrayToImage(byte[] byteArrayIn)
        {
            MemoryStream ms = new MemoryStream(byteArrayIn);
            Image returnImage = Image.FromStream(ms);

            return returnImage;
        }

        /// <summary>
        /// Find the first and last colored rows of an image and centers the colored portion of the image vertically.
        /// </summary>
        /// <param name="bmp">Specifies the image to center.</param>
        /// <param name="clrBackground">Specifies the back-ground color to use for the non-colored portions.</param>
        /// <returns>The centered Image is returned.</returns>
        public static Image Center(Bitmap bmp, Color clrBackground)
        {
            int nTopYColorRow = getFirstColorRow(bmp, clrBackground);
            int nBottomYColorRow = getLastColorRow(bmp, clrBackground);

            if (nTopYColorRow <= 0 && nBottomYColorRow >= bmp.Height - 1)
                return bmp;

            int nVerticalShift = (((bmp.Height - 1) - nBottomYColorRow) - nTopYColorRow) / 2;

            if (nVerticalShift == 0)
                return bmp;

            Bitmap bmpNew = new Bitmap(bmp.Width, bmp.Height);
            Brush br = new SolidBrush(clrBackground);

            using (Graphics g = Graphics.FromImage(bmpNew))
            {
                g.FillRectangle(br, 0, 0, bmp.Width, bmp.Height);
                g.DrawImage(bmp, 0, nVerticalShift);
            }

            br.Dispose();

            return bmpNew;
        }

        private static int getFirstColorRow(Bitmap bmp, Color clrTarget)
        {
            for (int y = 0; y < bmp.Height; y++)
            {
                for (int x = 0; x < bmp.Width; x++)
                {
                    Color clr = bmp.GetPixel(x, y);

                    if (clr.R != clrTarget.R || clr.G != clrTarget.G || clr.B != clrTarget.B || clr.A != clrTarget.A)
                        return y;
                }
            }

            return bmp.Height;
        }

        private static int getLastColorRow(Bitmap bmp, Color clrTarget)
        {
            for (int y = bmp.Height - 1; y >= 0; y-- )
            {
                for (int x = 0; x < bmp.Width; x++)
                {
                    Color clr = bmp.GetPixel(x, y);

                    if (clr.R != clrTarget.R || clr.G != clrTarget.G || clr.B != clrTarget.B || clr.A != clrTarget.A)
                        return y;
                }
            }

            return -1;
        }
    }
}
