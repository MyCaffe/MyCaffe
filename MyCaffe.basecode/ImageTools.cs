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
        public static Bitmap AdjustContrast(Image bmp, float fBrightness = 1.0f, float fContrast = 1.0f, float fGamma = 1.0f)
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

        /// <summary>
        /// Converts an Image into an array of <i>byte</i>.
        /// </summary>
        /// <param name="imageIn">Specifies the Image.</param>
        /// <returns>The array of <i>byte</i> is returned.</returns>
        public static byte[] ImageToByteArray(Image imageIn)
        {
            MemoryStream ms = new MemoryStream();
            imageIn.Save(ms, System.Drawing.Imaging.ImageFormat.Gif);

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
