using MyCaffe.basecode;
using MyCaffe.data;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The DeepDraw class implements both deep drawing and deep dream as originally
    /// introduced by Google.
    /// </summary>
    /// <remarks>
    /// @see [Diving Deeper into Deep Dreams](http://www.kpkaiser.com/machine-learning/diving-deeper-into-deep-dreams/) by Kirk Kaiser, 2015.
    /// @see [auduno/deepdraw](https://github.com/auduno/deepdraw/blob/master/deepdraw.ipynb) on GitHub.
    /// @see [kylemcdonald/deepdream](https://github.com/kylemcdonald/deepdream/blob/master/dream.ipynb) on GitHub
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DeepDraw<T> : IDisposable, IEnumerable<Octaves>
    {
        OctavesCollection m_rgOctaves = new OctavesCollection();
        CancelEvent m_evtCancel;
        Net<T> m_net;
        DataTransformer<T> m_transformer;
        Log m_log;
        CudaDnn<T> m_cuda;
        int m_nWid = 0;
        int m_nHt = 0;
        Blob<T> m_blobBase;
        Blob<T> m_blobDetail;
        Blob<T> m_blobBlur;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="evtCancel">Specifies the cancel event used to cancel the drawing operation.</param>
        /// <param name="net">Specifies the Net to use for the drawing operation.</param>
        /// <param name="transformer">Specifies the DataTransformer to use when preprocessing input images.</param>
        /// <param name="strSrcBlobName">Specifies the the name of the Blob containing the data source.</param>
        public DeepDraw(CancelEvent evtCancel, Net<T> net, DataTransformer<T> transformer, string strSrcBlobName = "data")
        {
            m_evtCancel = evtCancel;
            m_net = net;
            m_transformer = transformer;

            Blob<T> blobSrc = m_net.blob_by_name(strSrcBlobName);
            m_cuda = blobSrc.Cuda;
            m_log = blobSrc.Log;

            m_nWid = blobSrc.width;
            m_nHt = blobSrc.height;

            m_blobBase = new common.Blob<T>(m_cuda, m_log, false);
            m_blobDetail = new common.Blob<T>(m_cuda, m_log, false);
            m_blobBlur = new common.Blob<T>(m_cuda, m_log, false);

            m_blobBase.ReshapeLike(blobSrc);
            m_blobDetail.ReshapeLike(blobSrc);
            m_blobBlur.ReshapeLike(blobSrc);
        }

        /// <summary>
        /// Releases all resources used by DeepDraw.
        /// </summary>
        public void Dispose()
        {
            if (m_blobBase != null)
            {
                m_blobBase.Dispose();
                m_blobBase = null;
            }

            if (m_blobDetail != null)
            {
                m_blobDetail.Dispose();
                m_blobDetail = null;
            }

            if (m_blobBlur != null)
            {
                m_blobBlur.Dispose();
                m_blobBlur = null;
            }
        }

        /// <summary>
        /// Returns the numberof Octaves installed.
        /// </summary>
        public int Count
        {
            get { return m_rgOctaves.Count; }
        }

        /// <summary>
        /// Adds a new Octave to run the deep drawing over.
        /// </summary>
        /// <param name="strLayer">Specifies the name of the target 'end' layer in the network.</param>
        /// <param name="nIterations">Specifies the number of iterations to run over the Octave.</param>
        /// <param name="dfStartSigma">Specifies the starting sigma used when performing the gaussian bluring, on each iteration this value moves toward the end sigma.</param>
        /// <param name="dfEndSigma">Specifies the ending sigma to use whenp performing the gaussian bluring.</param>
        /// <param name="dfStartStep">Specifis the starting step, this value moves towards the end step on each iteration.</param>
        /// <param name="dfEndStep">Specifies the ending step.</param>
        /// <param name="bSaveFile">Specifies whether or not to save the final image for the octave to disk.</param>
        /// <param name="dfPctDetailsToApply">Specifies the percentage of the details from the previous octave run to apply to the source for this Octave - this value must be 1.0 when only using one Octave.</param>
        public void Add(string strLayer, int nIterations, double dfStartSigma, double dfEndSigma, double dfStartStep, double dfEndStep, bool bSaveFile = false, double dfPctDetailsToApply = 0.25)
        {
            m_rgOctaves.Add(new common.Octaves(strLayer, nIterations, dfStartSigma, dfEndSigma, dfStartStep, dfEndStep, bSaveFile, dfPctDetailsToApply));
        } 

        /// <summary>
        /// Returns the Octave enumerator.
        /// </summary>
        /// <returns>The Octave enumerator is returned.</returns>
        public IEnumerator<Octaves> GetEnumerator()
        {
            return m_rgOctaves.GetEnumerator();
        }

        /// <summary>
        /// Returns the Octave enumerator.
        /// </summary>
        /// <returns>The Octave enumerator is returned.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgOctaves.GetEnumerator();
        }

        /// <summary>
        /// Creates an image with random noise.
        /// </summary>
        /// <param name="clrBack">Specifies the base back color.</param>
        /// <param name="nW">Optionally, specifies the width.  When not specified, the width of the source Blob is used instead (recommended).</param>
        /// <param name="nH">Optionally, specifies the height.  When not specified, the height of the source Blob is used instead (recommended).</param>
        /// <param name="dfScale">Optionally, specifies the spread of pixels randomized around the base color.  For example the default of 16 randomly picks a color value +8 and -8 from the base color.</param>
        /// <returns></returns>
        public Bitmap CreateRandomImage(Color clrBack, int nW = -1, int nH = -1, double dfScale = 16.0)
        {
            if (nW <= 0)
                nW = m_nWid;

            if (nH <= 0)
                nH = m_nWid;

            Bitmap bmp = new Bitmap(nW, nH);
            Random rand = new Random();
            double dfVal;

            for (int y = 0; y < nH; y++)
            {
                for (int x = 0; x < nW; x++)
                {
                    dfVal = rand.NextDouble() * dfScale;
                    int nR = clrBack.R - (int)((dfScale / 2) + dfVal);

                    dfVal = rand.NextDouble() * dfScale;
                    int nG = clrBack.G - (int)((dfScale / 2) + dfVal);

                    dfVal = rand.NextDouble() * dfScale;
                    int nB = clrBack.B - (int)((dfScale / 2) + dfVal);

                    bmp.SetPixel(x, y, Color.FromArgb(nR, nG, nB));
                }
            }

            return bmp;
        }

        /// <summary>
        /// Renders the deep draw image(s) depending on the Octave's installed.
        /// </summary>
        /// <param name="bmpInput">Specifies the input image.</param>
        /// <param name="nFocusLabel">Specifies a label to focus on (use this when running on classifying layers).</param>
        /// <param name="dfDetailPercentageToOutput">Optionally, specifies the amount of detail to apply to the original image when producing the final image (Default = 0.25 for 25%).</param>
        /// <param name="strOutputDir">Optionally, specifies the output directory wheren images are to be output.  When <i>null</i>, no images are output, but are instead set in each Octave.</param>
        /// <param name="bVisualizeEachStep">Optionally, specifies to create an image at each step of the process which can be useful when making a video of the evolution (default = <i>false</i>).</param>
        /// <returns></returns>
        public bool Render(Bitmap bmpInput, int nFocusLabel = -1, double dfDetailPercentageToOutput = 0.25, string strOutputDir = null, bool bVisualizeEachStep = false)
        {
            // get the input dimensions from net
            Blob<T> blobSrc = m_net.blob_by_name("data");

            int nW = blobSrc.width;
            int nH = blobSrc.height;

            m_log.WriteLine("Starting drawing...");
            blobSrc.Reshape(1, 3, nH, nW);    // resize the networks input.

            // Set the base data.
            if (strOutputDir != null)
                bmpInput.Save(strOutputDir + "\\inputimage.png");
            Datum d = ImageData.GetImageData(bmpInput, 3, false, -1);
            m_blobBase.mutable_cpu_data = m_transformer.Transform(d);

            m_blobDetail.SetData(0.0);
            m_blobBlur.SetData(0);

            foreach (Octaves o in m_rgOctaves)
            {
                // Select layer.
                string strLayer = o.LayerName;

                // Add changed details to the image.
                if (nFocusLabel == -1)
                    m_cuda.add(blobSrc.count(), m_blobBase.gpu_data, m_blobDetail.gpu_data, blobSrc.mutable_gpu_data, o.PercentageOfPreviousOctaveDetailsToApply);

                for (int i = 0; i < o.IterationN; i++)
                {
                    if (m_evtCancel.WaitOne(0))
                        return false;

                    if (nFocusLabel >= 0)
                        blobSrc.CopyFrom(m_blobBase);

                    double dfSigma = o.StartSigma + ((o.EndSigma - o.StartSigma) * i) / o.IterationN;
                    double dfStepSize = o.StartStepSize + ((o.EndStepSize - o.StartStepSize) * i) / o.IterationN;

                    make_step(m_blobBlur, strLayer, dfSigma, dfStepSize, nFocusLabel);

                    if (strOutputDir != null && (bVisualizeEachStep || (i == o.IterationN - 1 && o.Save)))
                    {
                        if (dfDetailPercentageToOutput < 1.0)
                        {
                            // Get the detail.
                            m_cuda.sub(m_blobDetail.count(), blobSrc.gpu_data, m_blobBase.gpu_data, m_blobDetail.mutable_gpu_data);
                            // reuse blob blur memory.
                            m_cuda.add(m_blobBlur.count(), m_blobBase.gpu_data, m_blobDetail.gpu_data, m_blobBlur.mutable_gpu_data, dfDetailPercentageToOutput);
                        }
                        else
                        {
                            m_blobBlur.CopyFrom(blobSrc);
                        }

                        Image bmp = getImage(m_blobBlur);
                        string strFile = strOutputDir + "\\" + o.UniqueName + "_" + i.ToString();
                        if (nFocusLabel >= 0)
                            strFile += "_class_" + nFocusLabel.ToString();

                        bmp.Save(strFile + ".png");

                        Bitmap bmp1 = adjustContrast(bmp, 1.0f, 1.1f, 1.1f);
                        bmp1.Save(strFile + "_bright.png");
                        bmp1.Dispose();

                        if (i == o.IterationN - 1)
                            o.Images.Add(nFocusLabel, bmp);
                        else
                            bmp.Dispose();
                    }

                    m_log.Progress = (double)i / (double)o.IterationN;
                    m_log.WriteLine("Octave: '" + o.LayerName + "' - " + i.ToString() + " of " + o.IterationN.ToString() + " " + m_log.Progress.ToString("P"));

                    if (nFocusLabel >= 0)
                        m_blobBase.CopyFrom(blobSrc);
                }

                // Extract details produced on the current octave.
                if (nFocusLabel == -1)
                    m_cuda.sub(m_blobDetail.count(), blobSrc.gpu_data, m_blobBase.gpu_data, m_blobDetail.mutable_gpu_data);
            }

            m_log.WriteLine("Rendering completed!");
            return true;
        }

        private void make_step(Blob<T> blobBlur, string strLayer, double dfSigma, double dfStepSize = 1.5, int nFocusLabel = -1)
        {
            Blob<T> blobSrc = m_net.blob_by_name("data"); // input image is stored in Net's 'data' blob
            Blob<T> blobDst = m_net.blob_by_name(strLayer);
            int nDstIdx = m_net.layer_index_by_name(strLayer);
            double dfLoss;

            m_net.Forward(out dfLoss);

            if (nFocusLabel < 0 || nFocusLabel >= blobDst.count())
            {
                m_cuda.copy(blobDst.count(), blobDst.gpu_data, blobDst.mutable_gpu_diff);
            }
            else
            {
                blobDst.SetDiff(0);
                blobDst.SetDiff(1.0, nFocusLabel);
            }

            m_net.Backward(nDstIdx);

            // Apply normalized ascent step to the input image.
            double dfAsum = Utility.ConvertVal<T>(blobSrc.asum_diff());
            double dfMean = dfAsum / blobSrc.count();
            double dfStep = dfStepSize / dfMean;
            m_cuda.scal(blobSrc.count(), dfStep, blobSrc.mutable_gpu_diff);
            m_cuda.add(blobSrc.count(), blobSrc.gpu_diff, blobSrc.gpu_data, blobSrc.mutable_gpu_data);

            if (dfSigma != 0)
            {
                m_cuda.gaussian_blur(blobSrc.count(), blobSrc.channels, blobSrc.height, blobSrc.width, dfSigma, blobSrc.gpu_data, blobBlur.mutable_gpu_data);
                blobSrc.CopyFrom(blobBlur);
            }

            // reset objective for nest step.
            blobDst.SetDiff(0);
        }

        private Image getImage(Blob<T> blobSrc)
        {
            scale(blobSrc, 0.0, 255.0, true);
            return ImageData.GetImage(blobSrc.ToDatum());
        }

        private void scale(Blob<T> b, double dfMin1, double dfMax1, bool bByChannel)
        {
            int nDim = b.height * b.width;
            double[] rgdf = Utility.ConvertVec<T>(b.mutable_cpu_data);
            double dfMinR = double.MaxValue;
            double dfMinG = double.MaxValue;
            double dfMinB = double.MaxValue;
            double dfMaxR = -double.MaxValue;
            double dfMaxG = -double.MaxValue;
            double dfMaxB = -double.MaxValue;

            if (bByChannel)
            {
                for (int i = 0; i < nDim; i++)
                {
                    double dfR = rgdf[i];
                    double dfG = rgdf[i + nDim];
                    double dfB = rgdf[i + nDim * 2];

                    dfMinR = Math.Min(dfR, dfMinR);
                    dfMaxR = Math.Max(dfR, dfMaxR);
                    dfMinG = Math.Min(dfG, dfMinG);
                    dfMaxG = Math.Max(dfG, dfMaxG);
                    dfMinB = Math.Min(dfB, dfMinB);
                    dfMaxB = Math.Max(dfB, dfMaxB);
                }
            }
            else
            {
                dfMinR = b.min_data;
                dfMaxR = b.max_data;
                dfMinG = dfMinR;
                dfMaxG = dfMaxR;
                dfMinB = dfMinR;
                dfMaxB = dfMaxR;
            }

            double dfRs = (dfMax1 - dfMin1) / (dfMaxR - dfMinR);
            double dfGs = (dfMax1 - dfMin1) / (dfMaxG - dfMinG);
            double dfBs = (dfMax1 - dfMin1) / (dfMaxB - dfMinB);

            for (int i = 0; i < nDim; i++)
            {
                double dfR = rgdf[i];
                double dfG = rgdf[i + nDim];
                double dfB = rgdf[i + nDim * 2];

                dfR = (dfR - dfMinR) * dfRs + dfMin1;
                dfG = (dfG - dfMinG) * dfGs + dfMin1;
                dfB = (dfB - dfMinB) * dfBs + dfMin1;

                rgdf[i] = dfR;
                rgdf[i + nDim] = dfG;
                rgdf[i + nDim * 2] = dfB;
            }

            b.mutable_cpu_data = Utility.ConvertVec<T>(rgdf);
        }

        private Bitmap adjustContrast(Image bmp, float fBrightNess = 1.0f, float fContrast = 1.0f, float fGamma = 1.0f)
        {
            float fAdjBrightNess = fBrightNess - 1.0f;
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
    }

    /// <summary>
    /// The Octave class defines the setting sused when images are generated.
    /// </summary>
    public class Octaves : IDisposable
    {
        string m_strLayerName;
        int m_nIterN;
        double m_dfStartSigma;
        double m_dfEndSigma;
        double m_dfStartStepSize;
        double m_dfEndStepSize;
        bool m_bSave = false;
        double m_dfDetailPctToApply = 0.25;
        Dictionary<int, Image> m_rgImages = new Dictionary<int, Image>();

        /// <summary>
        /// Specifies the constructor.
        /// </summary>
        /// <param name="strLayerName">Specifies the target 'end' layer in the network.</param>
        /// <param name="nIterN">Specifies the number of iterations to run.</param>
        /// <param name="dfStartSigma">Specifies the starting sigma to use during gaussian blurring.</param>
        /// <param name="dfEndSigma">Specifies the ending sigma to use during gaussian blurring.</param>
        /// <param name="dfStartStepSize">Specifies the starting step size.</param>
        /// <param name="dfEndStepSize">Specifies the ending step size.</param>
        /// <param name="bSaveFile">Specifies whether or not to save the ending image for the Octave to disk.</param>
        /// <param name="dfPctDetailsToApply">Specifies the percentage of the previous octave detail to apply to the source for this octave run.</param>
        public Octaves(string strLayerName, int nIterN, double dfStartSigma, double dfEndSigma, double dfStartStepSize, double dfEndStepSize, bool bSaveFile = false, double dfPctDetailsToApply = 0.25)
        {
            m_strLayerName = strLayerName;
            m_nIterN = nIterN;
            m_dfStartSigma = dfStartSigma;
            m_dfEndSigma = dfEndSigma;
            m_dfStartStepSize = dfStartStepSize;
            m_dfEndStepSize = dfEndStepSize;
            m_dfDetailPctToApply = 0.25;
            m_bSave = bSaveFile;
        }

        /// <summary>
        /// Releases all resources used.
        /// </summary>
        public void Dispose()
        {
            foreach (KeyValuePair<int, Image> kv in m_rgImages)
            {
                if (kv.Value != null)
                    kv.Value.Dispose();
            }

            m_rgImages.Clear();
        }

        /// <summary>
        /// Returns the 'end' target network layer.
        /// </summary>
        public string LayerName
        {
            get { return m_strLayerName; }
        }

        /// <summary>
        /// Returns the number of iterations to run.
        /// </summary>
        public int IterationN
        {
            get { return m_nIterN; }
        }

        /// <summary>
        /// Returns the starting sigma used during gaussian blurring.
        /// </summary>
        public double StartSigma
        {
            get { return m_dfStartSigma; }
        }

        /// <summary>
        /// Returns the ending sigma used during gaussian blurring.
        /// </summary>
        public double EndSigma
        {
            get { return m_dfEndSigma; }
        }

        /// <summary>
        /// Returns the starting step.
        /// </summary>
        public double StartStepSize
        {
            get { return m_dfStartStepSize; }
        }

        /// <summary>
        /// Returns the ending step.
        /// </summary>
        public double EndStepSize
        {
            get { return m_dfEndStepSize; }
        }

        /// <summary>
        /// Returns a unique name of the Octave.
        /// </summary>
        public string UniqueName
        {
            get { return m_nIterN.ToString() + "_" + getCleanName(m_strLayerName); }
        }

        /// <summary>
        /// Returns the images generated for the Octave, ordered by lable, where -1 = all.
        /// </summary>
        public Dictionary<int, Image> Images
        {
            get { return m_rgImages; }
            set { m_rgImages = value; }
        }

        /// <summary>
        /// Returns whether or not to save the Octave final image to disk.
        /// </summary>
        public bool Save
        {
            get { return m_bSave; }
        }

        /// <summary>
        /// Returns the percentage of detail from the previous Octave to apply to the source for this Octave.
        /// </summary>
        public double PercentageOfPreviousOctaveDetailsToApply
        {
            get { return m_dfDetailPctToApply; }
        }

        private string getCleanName(string str)
        {
            string strOut = "";

            foreach (char ch in str)
            {
                if (!char.IsSymbol(ch) && ch != '\\' && ch != '/')
                    strOut += ch;
            }

            return strOut;
        }
    }

    class OctavesCollection : IEnumerable<Octaves> /** @private */
    {
        List<Octaves> m_rgOctaves = new List<Octaves>();

        public OctavesCollection()
        {
        }

        public int Count
        {
            get { return m_rgOctaves.Count; }
        }

        public Octaves this[int nIdx]
        {
            get { return m_rgOctaves[nIdx]; }
        }

        public void Add(Octaves o)
        {
            m_rgOctaves.Add(o);
        }

        public IEnumerator<Octaves> GetEnumerator()
        {
            return m_rgOctaves.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgOctaves.GetEnumerator();
        }
    }
}
