using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The DirectBitmap class provides an efficient bitmap creating class.
    /// </summary>
    /// <remarks>
    /// @see [C# - Faster Alternatives to SetPixel and GetPixel for Bitmaps for Windows Forms App](https://stackoverflow.com/questions/24701703/c-sharp-faster-alternatives-to-setpixel-and-getpixel-for-bitmaps-for-windows-f), Stackoverflow.com.
    /// </remarks>
    public class DirectBitmap : IDisposable
    {
        /// <summary>
        /// Returns the Bitmap itself.
        /// </summary>
        public Bitmap Bitmap { get; private set; }
        /// <summary>
        /// Returns an array containing the raw bitmap data.
        /// </summary>
        public Int32[] Bits { get; private set; }
        /// <summary>
        /// Returns <i>true</i> when disposed.
        /// </summary>
        public bool Disposed { get; private set; }
        /// <summary>
        /// Returns the bitmap height.
        /// </summary>
        public int Height { get; private set; }
        /// <summary>
        /// Returns the bitmap width.
        /// </summary>
        public int Width { get; private set; }
        /// <summary>
        /// Returns the bitmap memory handle.
        /// </summary>
        protected GCHandle BitsHandle { get; private set; }

        /// <summary>
        /// The constructro.
        /// </summary>
        /// <param name="width">Specifies the bitmap width.</param>
        /// <param name="height">Specifies the bitmap height.</param>
        public DirectBitmap(int width, int height)
        {
            Width = width;
            Height = height;
            Bits = new Int32[width * height];
            BitsHandle = GCHandle.Alloc(Bits, GCHandleType.Pinned);
            Bitmap = new Bitmap(width, height, width * 4, PixelFormat.Format32bppPArgb, BitsHandle.AddrOfPinnedObject());
        }

        /// <summary>
        /// Sets a pixel within the bitmap with the specified color.
        /// </summary>
        /// <param name="x">Specifies the x position of the pixel.</param>
        /// <param name="y">Specifies the y position of the pixel.</param>
        /// <param name="colour">Specifies the color to set the pixel.</param>
        public void SetPixel(int x, int y, Color colour)
        {
            int index = x + (y * Width);
            int col = colour.ToArgb();

            Bits[index] = col;
        }

        /// <summary>
        /// Set an entire row to the same color.
        /// </summary>
        /// <param name="y">Specifies the row.</param>
        /// <param name="colour">Specifies the color.</param>
        public void SetRow(int y, Color colour)
        {
            int col = colour.ToArgb();
            int index = y * Width;

            for (int x = 0; x < Width; x++)
            {
                Bits[index + x] = col;
            }
        }

        /// <summary>
        /// Returns the color of a pixel in the bitmap.
        /// </summary>
        /// <param name="x">Specifies the x position of the pixel.</param>
        /// <param name="y">Specifies the y position of the pixel.</param>
        /// <returns>The color of the pixel is returned.</returns>
        public Color GetPixel(int x, int y)
        {
            int index = x + (y * Width);
            int col = Bits[index];
            Color result = Color.FromArgb(col);

            return result;
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            if (Disposed) return;
            Disposed = true;

            if (Bitmap != null)
                Bitmap.Dispose();

            BitsHandle.Free();
        }
    }
}
