using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.param;
using MyCaffe.param.ssd;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.data
{
    /// <summary>
    /// The ImageTransforms class provides several useful image transformation function used with SSD.
    /// </summary>
    /// <remarks>
    /// @see [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, 2016.
    /// @see [GitHub: SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd), by weiliu89/caffe, 2016
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class ImageTransforms<T>
    {
        CryptoRandom m_random;
        CudaDnn<T> m_cuda;
        Log m_log;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance used to communidate with Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="random">Specifies the random number generator.</param>
        public ImageTransforms(CudaDnn<T> cuda, Log log, CryptoRandom random)
        {
            m_random = random;
            m_cuda = cuda;
            m_log = log;
        }

        /// <summary>
        /// Update the BBox size based on the Resize policy.
        /// </summary>
        /// <param name="p">Specifies the ResizeParameter with the resize policy.</param>
        /// <param name="nOldWidth">Specifies the old width.</param>
        /// <param name="nOldHeight">Specifies the old height.</param>
        /// <param name="bbox1">Specifies the BBox to update.</param>
        /// <returns>The update NormalizedBBox is returned.</returns>
        public NormalizedBBox UpdateBBoxByResizePolicy(ResizeParameter p, int nOldWidth, int nOldHeight, NormalizedBBox bbox1)
        {
            NormalizedBBox bbox = bbox1.Clone();
            float fNewHeight = p.height;
            float fNewWidth = p.width;
            float fOrigAspect = (float)nOldWidth / (float)nOldHeight;
            float fNewAspect = fNewWidth / fNewHeight;

            float fxmin = bbox.xmin * nOldWidth;
            float fymin = bbox.ymin * nOldHeight;
            float fxmax = bbox.xmax * nOldWidth;
            float fymax = bbox.ymax * nOldHeight;
            float fPadding;

            switch (p.resize_mode)
            {
                case ResizeParameter.ResizeMode.WARP:
                    fxmin = (float)Math.Max(0.0, fxmin * fNewWidth / nOldWidth);
                    fxmax = (float)Math.Min(fNewWidth, fxmax * fNewWidth / nOldWidth);
                    fymin = (float)Math.Max(0.0, fymin * fNewHeight / nOldHeight);
                    fymax = (float)Math.Min(fNewHeight, fymax * fNewHeight / nOldHeight);
                    break;

                case ResizeParameter.ResizeMode.FIT_LARGE_SIZE_AND_PAD:
                    if (fOrigAspect > fNewAspect)
                    {
                        fPadding = (fNewHeight - fNewWidth / fOrigAspect) / 2;
                        fxmin = (float)Math.Max(0.0, fxmin * fNewWidth / nOldWidth);
                        fxmax = (float)Math.Min(fNewWidth, fxmax * fNewWidth / nOldWidth);
                        fymin = fymin * (fNewHeight - 2 * fPadding) / nOldHeight;
                        fymin = fPadding + Math.Max(0, fymin);
                        fymax = fymax * (fNewHeight - 2 * fPadding) / nOldHeight;
                        fymax = fPadding + Math.Min(fNewHeight, fymax);
                    }
                    else
                    {
                        fPadding = (fNewWidth - fOrigAspect * fNewHeight) / 2;
                        fxmin = fxmin * (fNewWidth - 2 * fPadding) / nOldWidth;
                        fxmin = fPadding + Math.Max(0, fxmin);
                        fxmax = fxmax * (fNewWidth - 2 * fPadding) / nOldWidth;
                        fxmax = fPadding + Math.Min(fNewWidth, fxmax);
                        fymin = (float)Math.Max(0.0, fymin * fNewHeight / nOldHeight);
                        fymax = (float)Math.Min(fNewHeight, fymax * fNewHeight / nOldHeight);
                    }
                    break;

                case ResizeParameter.ResizeMode.FIT_SMALL_SIZE:
                    if (fOrigAspect < fNewAspect)
                        fNewHeight = fNewWidth / fOrigAspect;
                    else
                        fNewWidth = fOrigAspect * fNewHeight;

                    fxmin = (float)Math.Max(0.0, fxmin * fNewWidth / nOldWidth);
                    fxmax = (float)Math.Min(fNewWidth, fxmax * fNewWidth / nOldWidth);
                    fymin = (float)Math.Max(0.0, fymin * fNewHeight / nOldHeight);
                    fymax = (float)Math.Min(fNewHeight, fymax * fNewHeight / nOldHeight);
                    break;

                default:
                    m_log.FAIL("Unknown resize mode '" + p.resize_mode.ToString() + "'!");
                    break;
            }

            bbox.xmin = fxmin / fNewWidth;
            bbox.ymin = fymin / fNewHeight;
            bbox.xmax = fxmax / fNewWidth;
            bbox.ymax = fymax / fNewHeight;

            return bbox;
        }

        /// <summary>
        /// Infer the new shape based on the resize policy.
        /// </summary>
        /// <param name="p">Specifies the ResizeParameter with the resize policy.</param>
        /// <param name="nOldWidth">Specifies the old width.</param>
        /// <param name="nOldHeight">Specifies the old height.</param>
        /// <param name="nNewWidth">Specifies the new 'inferred' width.</param>
        /// <param name="nNewHeight">Specifies the new 'inferred' width.</param>
        public void InferNewSize(ResizeParameter p, int nOldWidth, int nOldHeight, out int nNewWidth, out int nNewHeight)
        {
            int height = (int)p.height;
            int width = (int)p.width;
            float fOrigAspect = (float)nOldWidth / (float)nOldHeight;
            float fAspect = (float)width / (float)height;

            switch (p.resize_mode)
            {
                case ResizeParameter.ResizeMode.WARP:
                    break;

                case ResizeParameter.ResizeMode.FIT_LARGE_SIZE_AND_PAD:
                    break;

                case ResizeParameter.ResizeMode.FIT_SMALL_SIZE:
                    if (fOrigAspect < fAspect)
                        height = (int)(width / fOrigAspect);
                    else
                        width = (int)(fOrigAspect * height);
                    break;

                default:
                    m_log.FAIL("Unknown resize mode '" + p.resize_mode.ToString() + "'!");
                    break;
            }

            nNewHeight = height;
            nNewWidth = width;
        }

        private float getRandom(float fMin, float fMax, float fTotal = 256)
        {
            float fRange = fMax - fMin;
            double dfVal = m_random.NextDouble();
            double dfPct = ((fMin + (float)(fRange * dfVal)) / fTotal);

            return 1.0f + (float)dfPct;
        }

        private Bitmap randomBrightness(Bitmap bmp, float fProb, float fDelta)
        {
            if (m_random.NextDouble() < fProb)
            {
                float fBrightness = getRandom(-fDelta, fDelta);
                Bitmap bmpNew = ImageTools.AdjustContrast(bmp, fBrightness);
                bmp.Dispose();
                return bmpNew;
            }

            return bmp;
        }

        private Bitmap randomContrast(Bitmap bmp, float fProb, float fLower, float fUpper)
        {
            if (m_random.NextDouble() < fProb)
            {
                float fContrast = getRandom(fLower, fUpper);
                Bitmap bmpNew = ImageTools.AdjustContrast(bmp, 1, fContrast);
                bmp.Dispose();
                return bmpNew;
            }

            return bmp;
        }

        private Bitmap randomSaturation(Bitmap bmp, float fProb, float fLower, float fUpper)
        {
            if (m_random.NextDouble() < fProb)
            {
                float fSaturation = getRandom(fLower, fUpper);
                Bitmap bmpNew = ImageTools.AdjustContrast(bmp, 1, 1, fSaturation);
                bmp.Dispose();
                return bmpNew;
            }

            return bmp;
        }

        private Bitmap randomBrightnessContrastSaturation(Bitmap bmp, float fProbBrightness, float fBrightnessDelta, float fProbContrast, float fLowerContrast, float fUpperContrast, float fProbSaturation, float fLowerSaturation, float fUpperSaturation)
        {
            float fBrightness = 1;
            float fContrast = 1;
            float fSaturation = 1;

            if (m_random.NextDouble() < fProbBrightness)
                fBrightness = getRandom(-fBrightnessDelta, fBrightnessDelta);

            if (m_random.NextDouble() < fProbContrast)
                fContrast = getRandom(fLowerContrast, fUpperContrast);

            if (m_random.NextDouble() < fProbSaturation)
                fSaturation = getRandom(fLowerSaturation, fUpperSaturation);

            if (fBrightness == 0 && fContrast == 1 && fSaturation == 1)
                return bmp;

            Bitmap bmpNew = ImageTools.AdjustContrast(bmp, fBrightness, fContrast, fSaturation);
            bmp.Dispose();

            return bmpNew;
        }

        private Bitmap randomBrightnessSaturationContrast(Bitmap bmp, float fProbBrightness, float fBrightnessDelta, float fProbContrast, float fLowerContrast, float fUpperContrast, float fProbSaturation, float fLowerSaturation, float fUpperSaturation)
        {
            float fBrightness = 1;
            float fContrast = 1;
            float fSaturation = 1;

            if (m_random.NextDouble() < fProbBrightness)
                fBrightness = getRandom(-fBrightnessDelta, fBrightnessDelta);

            if (m_random.NextDouble() < fProbContrast)
                fContrast = getRandom(fLowerContrast, fUpperContrast);

            if (m_random.NextDouble() < fProbSaturation)
                fSaturation = getRandom(fLowerSaturation, fUpperSaturation);

            if (fBrightness == 0 && fContrast == 1 && fSaturation == 1)
                return bmp;

            Bitmap bmpNew = ImageTools.AdjustContrast(bmp, fBrightness, 1, fSaturation);
            bmp.Dispose();
            bmp = bmpNew;

            if (fContrast != 1)
            {
                bmpNew = ImageTools.AdjustContrast(bmp, 1, fContrast, 1);
                bmp.Dispose();
                bmp = bmpNew;
            }

            return bmp;
        }

        private void randomBrightnessContrastSaturation(SimpleDatum sd, float fProbBrightness, float fBrightnessDelta, float fProbContrast, float fLowerContrast, float fUpperContrast, float fProbSaturation, float fLowerSaturation, float fUpperSaturation, ImageTools.ADJUSTCONTRAST_ORDERING ordering)
        {
            float fBrightness = 1;
            float fContrast = 1;
            float fSaturation = 1;

            if (m_random.NextDouble() < fProbBrightness)
                fBrightness = getRandom(-fBrightnessDelta, fBrightnessDelta);

            if (m_random.NextDouble() < fProbContrast)
                fContrast = getRandom(fLowerContrast, fUpperContrast);

            if (m_random.NextDouble() < fProbSaturation)
                fSaturation = getRandom(fLowerSaturation, fUpperSaturation);

            if (fBrightness == 0 && fContrast == 1 && fSaturation == 1)
                return;

            ImageTools.AdjustContrast(sd, fBrightness, fContrast, fSaturation, ordering);
        }

        private SimpleDatum randomChannelOrder(SimpleDatum sd, float fProb)
        {
            if (m_random.NextDouble() < fProb)
            {
                if (sd.IsRealData)
                {
                    if (sd.RealDataD != null)
                    {
                        int nOffset = 0;
                        int nCount = sd.ItemCount / sd.Channels;
                        List<double[]> rgrgData = new List<double[]>();
                        List<int> rgIdx = new List<int>();

                        for (int i = 0; i < sd.Channels; i++)
                        {
                            double[] rgData = new double[nCount];
                            Array.Copy(sd.RealDataD, nOffset, rgData, 0, nCount);
                            rgrgData.Add(rgData);
                            rgIdx.Add(i);
                            nOffset += nCount;
                        }

                        List<int> rgOrder = new List<int>();
                        while (rgOrder.Count < sd.Channels)
                        {
                            int nIdx = m_random.Next(sd.Channels);
                            if (!rgOrder.Contains(nIdx))
                                rgOrder.Add(nIdx);
                        }

                        nOffset = 0;
                        for (int i = 0; i < sd.Channels; i++)
                        {
                            int nIdx = rgOrder[i];
                            Array.Copy(rgrgData[nIdx], 0, sd.RealDataD, nOffset, nCount);
                            nOffset += nCount;
                        }
                    }
                    else if (sd.RealDataF != null)
                    {
                        int nOffset = 0;
                        int nCount = sd.ItemCount / sd.Channels;
                        List<float[]> rgrgData = new List<float[]>();
                        List<int> rgIdx = new List<int>();

                        for (int i = 0; i < sd.Channels; i++)
                        {
                            float[] rgData = new float[nCount];
                            Array.Copy(sd.RealDataF, nOffset, rgData, 0, nCount);
                            rgrgData.Add(rgData);
                            rgIdx.Add(i);
                            nOffset += nCount;
                        }

                        List<int> rgOrder = new List<int>();
                        while (rgOrder.Count < sd.Channels)
                        {
                            int nIdx = m_random.Next(sd.Channels);
                            if (!rgOrder.Contains(nIdx))
                                rgOrder.Add(nIdx);
                        }

                        nOffset = 0;
                        for (int i = 0; i < sd.Channels; i++)
                        {
                            int nIdx = rgOrder[i];
                            Array.Copy(rgrgData[nIdx], 0, sd.RealDataF, nOffset, nCount);
                            nOffset += nCount;
                        }
                    }
                    else
                    {
                        throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                    }
                }
                else
                {
                    int nOffset = 0;
                    int nCount = sd.ItemCount / sd.Channels;
                    List<byte[]> rgrgData = new List<byte[]>();
                    List<int> rgIdx = new List<int>();

                    for (int i = 0; i < sd.Channels; i++)
                    {
                        byte[] rgData = new byte[nCount];
                        Array.Copy(sd.ByteData, nOffset, rgData, 0, nCount);
                        rgrgData.Add(rgData);
                        rgIdx.Add(i);
                        nOffset += nCount;
                    }

                    List<int> rgOrder = new List<int>();
                    while (rgOrder.Count < sd.Channels)
                    {
                        int nIdx = m_random.Next(sd.Channels);
                        if (!rgOrder.Contains(nIdx))
                            rgOrder.Add(nIdx);
                    }

                    nOffset = 0;
                    for (int i = 0; i < sd.Channels; i++)
                    {
                        int nIdx = rgOrder[i];
                        Array.Copy(rgrgData[nIdx], 0, sd.ByteData, nOffset, nCount);
                        nOffset += nCount;
                    }
                }
            }

            return sd;
        }

        /// <summary>
        /// The ApplyDistort method applies the distortion policy to the simple datum.
        /// </summary>
        /// <param name="sd">Specifies the SimpleDatum to distort.</param>
        /// <param name="p">Specifies the distortion parameters that define the distortion policy.</param>
        /// <returns>The distorted SimpleDatum is returned.</returns>
        public SimpleDatum ApplyDistort(SimpleDatum sd, DistortionParameter p)
        {
            double dfProb = m_random.NextDouble();
            Bitmap bmp = ImageData.GetImage(sd);

            if (dfProb > 0.5)
                bmp = randomBrightnessContrastSaturation(bmp, p.brightness_prob, p.brightness_delta, p.contrast_prob, p.contrast_lower, p.contrast_upper, p.saturation_prob, p.saturation_lower, p.saturation_upper);
            else
                bmp = randomBrightnessSaturationContrast(bmp, p.brightness_prob, p.brightness_delta, p.contrast_prob, p.contrast_lower, p.contrast_upper, p.saturation_prob, p.saturation_lower, p.saturation_upper);

            SimpleDatum sd1 = ImageData.GetImageData(bmp, sd);
            sd.SetData(sd1);

            bmp.Dispose();

            return randomChannelOrder(sd, p.random_order_prob);
        }

        /// <summary>
        /// The ApplyDistortEx method applies the distortion policy to the simple datum.
        /// </summary>
        /// <param name="sd">Specifies the SimpleDatum to distort.</param>
        /// <param name="p">Specifies the distortion parameters that define the distortion policy.</param>
        /// <returns>The distorted SimpleDatum is returned.</returns>
        public SimpleDatum ApplyDistortEx(SimpleDatum sd, DistortionParameter p)
        {
            double dfProb = m_random.NextDouble();

            if (dfProb > 0.5)
                randomBrightnessContrastSaturation(sd, p.brightness_prob, p.brightness_delta, p.contrast_prob, p.contrast_lower, p.contrast_upper, p.saturation_prob, p.saturation_lower, p.saturation_upper, ImageTools.ADJUSTCONTRAST_ORDERING.BRIGHTNESS_CONTRAST_GAMMA);
            else
                randomBrightnessContrastSaturation(sd, p.brightness_prob, p.brightness_delta, p.contrast_prob, p.contrast_lower, p.contrast_upper, p.saturation_prob, p.saturation_lower, p.saturation_upper, ImageTools.ADJUSTCONTRAST_ORDERING.BRIGHTNESS_GAMMA_CONTRAST);

            return randomChannelOrder(sd, p.random_order_prob);
        }

        /// <summary>
        /// The ApplyNoise method applies the noise policy to the SimpleDatum.
        /// </summary>
        /// <param name="sd">Specifies the SimpleDatum for which noise is to be added.</param>
        /// <param name="p">Specifies the NoiseParameter that defines the noise policy.</param>
        /// <remarks>
        /// NOTE: This method is not yet complete.
        /// </remarks>
        public SimpleDatum ApplyNoise(SimpleDatum sd, NoiseParameter p)
        {
            m_log.WriteLine("WARNING: noise application not yet implemented.");
            return sd;
        }

        /// <summary>
        /// The ApplyResize method resizes the SimpleDatum containing an image to a newly resized image as specified by the resize parameter.
        /// </summary>
        /// <param name="sd">Specifies the SimpleDatum to resize - must contain a 3 channel image.</param>
        /// <param name="p">Specifies the resize parameter to apply.</param>
        /// <returns>The newly resized SimpleDatum is returned.</returns>
        public SimpleDatum ApplyResize(SimpleDatum sd, ResizeParameter p)
        {
            if (p.width == sd.Width && p.height == sd.Height)
                return sd;

            Bitmap bmp = ImageData.GetImage(sd);
            Bitmap bmpNew = ImageTools.ResizeImage(bmp, (int)p.width, (int)p.height);
            SimpleDatum sdResize = ImageData.GetImageData(bmpNew, sd, false);
            SimpleDatum sdNew = new SimpleDatum(sd);

            sdNew.CopyData(sdResize);

            bmp.Dispose();
            bmpNew.Dispose();

            return sdNew;
        }
    }
}
