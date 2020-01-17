using MyCaffe.basecode;
using MyCaffe.common;
using MyCaffe.data;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// The MyCaffe.extras namespace contains classes that use the MyCaffe and other namespaces to add enhanced features.  
/// </summary>
namespace MyCaffe.extras
{
    /// <summary>
    /// The DeepDraw class implements both deep drawing and deep dream as originally
    /// introduced by Google.
    /// </summary>
    /// <remarks>
    /// @see [Deep Dreams (with Caffe)](https://github.com/google/deepdream/blob/master/dream.ipynb) by the Google Deep Dream Team, 2015.
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
            m_rgOctaves.Add(new Octaves(strLayer, nIterations, dfStartSigma, dfEndSigma, dfStartStep, dfEndStep, bSaveFile, dfPctDetailsToApply));
        }

        /// <summary>
        /// Add a new Octaves to the collection of Octaves to run.
        /// </summary>
        /// <param name="octaves">Specifies the Octaves to add.</param>
        public void Add(Octaves octaves)
        {
            m_rgOctaves.Add(octaves);
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

            return CreateRandomImageEx(clrBack, nW, nH, dfScale);
        }

        /// <summary>
        /// Creates an image with random noise.
        /// </summary>
        /// <param name="clrBack">Specifies the base back color.</param>
        /// <param name="nW">Specifies the width.</param>
        /// <param name="nH">Specifies the height.</param>
        /// <param name="dfScale">Optionally, specifies the spread of pixels randomized around the base color.  For example the default of 16 randomly picks a color value +8 and -8 from the base color.</param>
        /// <returns></returns>
        public static Bitmap CreateRandomImageEx(Color clrBack, int nW, int nH, double dfScale = 16.0)
        {
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
        /// <param name="rgDirectInputs">Optionally, specifies the direct inputs used to set each output.  When not <i>null</i> the direct inputs are used instead of the <i>nFocusLabel</i> whereby the 
        /// network outputs are set to the direct input values and the <i>nFocusLabel</i> is used to index the image and should therefore be unique for each set of direct inputs.  
        /// By default, this value is set to <i>null</i>.
        /// </param>
        /// <returns>Upon completing the render, this method returns <i>true</i>, otherwise if cancelled it returns <i>false</i>.</returns>
        public bool Render(Bitmap bmpInput, int nFocusLabel = -1, double dfDetailPercentageToOutput = 0.25, string strOutputDir = null, bool bVisualizeEachStep = false, float[] rgDirectInputs = null)
        {
            if (rgDirectInputs != null && nFocusLabel < 0)
                throw new Exception("The focus label must be set to a unique value >= 0 that corresponds to this specific direct input set.");

            // get the input dimensions from net
            Blob<T> blobSrc = m_net.blob_by_name("data");

            int nW = blobSrc.width;
            int nH = blobSrc.height;

            m_log.WriteLine("Starting drawing...");
            blobSrc.Reshape(1, 3, nH, nW);    // resize the networks input.

            // Set the base data.
            if (strOutputDir != null)
                bmpInput.Save(strOutputDir + "\\input_image.png");

            Datum d = ImageData.GetImageDataD(bmpInput, 3, false, -1);
            m_blobBase.mutable_cpu_data = m_transformer.Transform(d);

            m_blobDetail.SetData(0.0);
            m_blobBlur.SetData(0);

            for (int i=0; i<m_rgOctaves.Count; i++)
            {
                Octaves o = m_rgOctaves[i];
                // Select layer.
                string strLayer = o.LayerName;

                // Add changed details to the image.
                if (nFocusLabel < 0)
                    m_cuda.add(blobSrc.count(), m_blobBase.gpu_data, m_blobDetail.gpu_data, blobSrc.mutable_gpu_data, o.PercentageOfPreviousOctaveDetailsToApply);

                for (int j = 0; j < o.IterationN; j++)
                {
                    if (m_evtCancel.WaitOne(0))
                        return false;

                    if (nFocusLabel >= 0)
                        blobSrc.CopyFrom(m_blobBase);

                    double dfSigma = o.StartSigma + ((o.EndSigma - o.StartSigma) * j) / o.IterationN;
                    double dfStepSize = o.StartStepSize + ((o.EndStepSize - o.StartStepSize) * j) / o.IterationN;

                    make_step(strLayer, dfSigma, dfStepSize, nFocusLabel, rgDirectInputs);

                    if ((bVisualizeEachStep || (j == o.IterationN - 1 && o.Save)))
                    {
                        // Get the detail.
                        m_cuda.sub(m_blobDetail.count(), blobSrc.gpu_data, m_blobBase.gpu_data, m_blobDetail.mutable_gpu_data);

                        if (dfDetailPercentageToOutput < 1.0)
                        {
                            // reuse blob blur memory.
                            m_cuda.add(m_blobBlur.count(), m_blobBase.gpu_data, m_blobDetail.gpu_data, m_blobBlur.mutable_gpu_data, dfDetailPercentageToOutput);
                        }
                        else
                        {
                            m_blobBlur.CopyFrom(blobSrc);
                        }

                        Image bmp = getImage(m_blobBlur);

                        if (nFocusLabel < 0)
                        {
                            Bitmap bmp1 = AdjustContrast(bmp, 0.9f, 1.6f, 1.2f);
                            bmp.Dispose();
                            bmp = bmp1;
                        }

                        if (strOutputDir != null)
                        {
                            string strFile = strOutputDir + "\\" + o.UniqueName + "_" + j.ToString();
                            if (nFocusLabel >= 0)
                            {
                                if (rgDirectInputs != null)
                                    strFile += "_idx_" + nFocusLabel.ToString();
                                else
                                   strFile += "_class_" + nFocusLabel.ToString();
                            }

                            bmp.Save(strFile + ".png");
                        }

                        if (j == o.IterationN - 1)
                            o.Images.Add(nFocusLabel, bmp);
                        else
                            bmp.Dispose();
                    }

                    m_log.Progress = (double)j / (double)o.IterationN;
                    m_log.WriteLine("Focus Label: " + nFocusLabel.ToString() + "  Octave: '" + o.LayerName + "' - " + j.ToString() + " of " + o.IterationN.ToString() + " " + m_log.Progress.ToString("P"));

                    if (nFocusLabel >= 0)
                        m_blobBase.CopyFrom(blobSrc);
                }

                // Extract details produced on the current octave.
                if (nFocusLabel < 0)
                    m_cuda.sub(m_blobDetail.count(), blobSrc.gpu_data, m_blobBase.gpu_data, m_blobDetail.mutable_gpu_data);
            }

            m_log.WriteLine("Rendering completed!");
            return true;
        }

        private void make_step(string strLayer, double dfSigma, double dfStepSize = 1.5, int nFocusLabel = -1, float[] rgDirectInputs = null)
        {
            Blob<T> blobSrc = m_net.blob_by_name("data"); // input image is stored in Net's 'data' blob
            Blob<T> blobDst = m_net.blob_by_name(strLayer);
            int nDstIdx = m_net.layer_index_by_name(strLayer);
            double dfLoss;

            m_net.Forward(out dfLoss);

            if (nFocusLabel < 0 && rgDirectInputs == null)
            {
                m_cuda.copy(blobDst.count(), blobDst.gpu_data, blobDst.mutable_gpu_diff);
            }
            else if (rgDirectInputs != null)
            {
                blobDst.SetDiff(0);

                for (int i = 0; i < rgDirectInputs.Length && i < blobDst.count(); i++)
                {
                    if (rgDirectInputs[i] != 0)
                        blobDst.SetDiff(rgDirectInputs[i], i);
                }
            }
            else
            {
                if (nFocusLabel >= blobDst.count())
                    throw new Exception("The focus label '" + nFocusLabel + "' is greater than the number of outputs for blob '" + blobDst.Name + "' -- it only has '" + blobDst.count().ToString() + "' outputs!");

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
                m_cuda.gaussian_blur(blobSrc.count(), blobSrc.channels, blobSrc.height, blobSrc.width, dfSigma, blobSrc.gpu_data, m_blobBlur.mutable_gpu_data);
                blobSrc.CopyFrom(m_blobBlur);
            }
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

            double dfMeanR = (m_transformer.param.mean_value.Count > 0) ? m_transformer.param.mean_value[0] : 0;
            double dfMeanG = (m_transformer.param.mean_value.Count > 1) ? m_transformer.param.mean_value[1] : 0;
            double dfMeanB = (m_transformer.param.mean_value.Count > 2) ? m_transformer.param.mean_value[2] : 0;

            if (bByChannel)
            {
                for (int i = 0; i < nDim; i++)
                {
                    double dfR = rgdf[i] + dfMeanR;
                    double dfG = rgdf[i + nDim] + dfMeanG;
                    double dfB = rgdf[i + nDim * 2] + dfMeanB;

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
                dfMaxR = b.max_data + Math.Max(dfMeanR, Math.Max(dfMeanG, dfMeanB));
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
                double dfR = rgdf[i] + dfMeanR;
                double dfG = rgdf[i + nDim] + dfMeanG;
                double dfB = rgdf[i + nDim * 2] + dfMeanB;

                dfR = (dfR - dfMinR) * dfRs + dfMin1;
                dfG = (dfG - dfMinG) * dfGs + dfMin1;
                dfB = (dfB - dfMinB) * dfBs + dfMin1;

                rgdf[i] = dfR;
                rgdf[i + nDim] = dfG;
                rgdf[i + nDim * 2] = dfB;
            }

            b.mutable_cpu_data = Utility.ConvertVec<T>(rgdf);
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
            return ImageTools.AdjustContrast(bmp, fBrightness, fContrast, fGamma);
        }

        /// <summary>
        /// The CreateConfigurationString function packs all deep draw settings into a configuration string.
        /// </summary>
        /// <param name="nWd">Specifies the input width.</param>
        /// <param name="nHt">Specifies the input height.</param>
        /// <param name="dfOutputDetailPct">Specifies the percentage of detail to apply to the final output.</param>
        /// <param name="colOctaves">Specifies the collection of Octaves to run.</param>
        /// <param name="strSrcBlobName">Specifies the name of the source blob.</param>
        /// <param name="dfRandomImageScale">Specifies the random image scale to use, a number in the range [0,50] used to create varying degrees of gray in the random input image.  
        /// A value of 0 removes the variation and uses a consistent image.</param>
        /// <returns>The configuration string is returned.</returns>
        public static string CreateConfigurationString(int nWd, int nHt, double dfOutputDetailPct, OctavesCollection colOctaves, string strSrcBlobName, double dfRandomImageScale)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("input_height", nHt.ToString(), RawProto.TYPE.STRING);
            rgChildren.Add("input_width", nWd.ToString(), RawProto.TYPE.STRING);
            rgChildren.Add("output_detail_pct", dfOutputDetailPct.ToString(), RawProto.TYPE.STRING);
            rgChildren.Add("src_blob_name", strSrcBlobName, RawProto.TYPE.STRING);
            rgChildren.Add("random_image_scale", dfRandomImageScale.ToString(), RawProto.TYPE.STRING);

            foreach (Octaves octave in colOctaves)
            {
                rgChildren.Add(octave.ToProto("octave"));
            }

            RawProto proto = new RawProto("root", "", rgChildren);

            return proto.ToString();
        }

        /// <summary>
        /// The ParseConfigurationString method parses a deep draw configuration string into the actual settings.
        /// </summary>
        /// <param name="strConfig">Specifies the configuration string to parse.</param>
        /// <param name="nWd">Returns the input width.</param>
        /// <param name="nHt">Returns the input height.</param>
        /// <param name="dfOutputDetailPct">Returns the percentage of detail to apply to the final image.</param>
        /// <param name="strSrcBlobName">Returns the source blob name.</param>
        /// <param name="dfRandomImageScale">Returns the random image scale to use, a number in the range [0,50] used to create varying degrees of gray in the random input image.  
        /// A value of 0 removes the variation and uses a consistent image.  The default value is 16.</param>
        /// <returns>Returns the collection of Octaves to run.</returns>
        public static OctavesCollection ParseConfigurationString(string strConfig, out int nWd, out int nHt, out double dfOutputDetailPct, out string strSrcBlobName, out double dfRandomImageScale)
        {
            RawProto proto = RawProto.Parse(strConfig);
            string strVal;

            nHt = -1;
            if ((strVal = proto.FindValue("input_height")) != null)
                nHt = int.Parse(strVal);

            nWd = -1;
            if ((strVal = proto.FindValue("input_width")) != null)
                nWd = int.Parse(strVal);

            dfOutputDetailPct = 0.25;
            if ((strVal = proto.FindValue("output_detail_pct")) != null)
                dfOutputDetailPct = double.Parse(strVal);

            strSrcBlobName = "data";
            if ((strVal = proto.FindValue("src_blob_name")) != null)
                strSrcBlobName = strVal;

            dfRandomImageScale = 16;
            if ((strVal = proto.FindValue("random_image_scale")) != null)
                dfRandomImageScale = double.Parse(strVal);

            OctavesCollection col = new OctavesCollection();
            RawProtoCollection rpcol = proto.FindChildren("octave");
            foreach (RawProto protoChild in rpcol)
            {
                col.Add(Octaves.FromProto(protoChild));
            }

            return col;
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
            m_dfDetailPctToApply = dfPctDetailsToApply;
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
        /// Get/set the percentage of detail from the previous Octave to apply to the source for this Octave.
        /// </summary>
        public double PercentageOfPreviousOctaveDetailsToApply
        {
            get { return m_dfDetailPctToApply; }
            set { m_dfDetailPctToApply = value; }
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

        /// <summary>
        /// The ToProto function converts the Octaves settings into a RawProto.
        /// </summary>
        /// <param name="strName">Specifies the name of the RawProto.</param>
        /// <returns>The RawProto is returned.</returns>
        public RawProto ToProto(string strName)
        {
            RawProtoCollection rgChildren = new RawProtoCollection();

            rgChildren.Add("layer", m_strLayerName, RawProto.TYPE.STRING);
            rgChildren.Add("iterations", m_nIterN.ToString(), RawProto.TYPE.STRING);
            rgChildren.Add("sigma_start", m_dfStartSigma.ToString(), RawProto.TYPE.STRING);
            rgChildren.Add("sigma_end", m_dfEndSigma.ToString(), RawProto.TYPE.STRING);
            rgChildren.Add("step_start", m_dfStartStepSize.ToString(), RawProto.TYPE.STRING);
            rgChildren.Add("step_end", m_dfEndStepSize.ToString(), RawProto.TYPE.STRING);
            rgChildren.Add("save", m_bSave.ToString(), RawProto.TYPE.STRING);
            rgChildren.Add("pct_of_prev_detail", m_dfDetailPctToApply.ToString(), RawProto.TYPE.STRING);

            return new RawProto(strName, "", rgChildren);
        }

        /// <summary>
        /// The FromProto function parses a RawProto into a new Octaves.
        /// </summary>
        /// <param name="rp">Specifies the RawProto to parse.</param>
        /// <returns>The new Octaves is returned.</returns>
        public static Octaves FromProto(RawProto rp)
        {
            string strVal;

            string strLayer = "";
            if ((strVal = rp.FindValue("layer")) != null)
                strLayer = strVal;

            int nIterations = 10;
            if ((strVal = rp.FindValue("iterations")) != null)
                nIterations = int.Parse(strVal);

            double dfSigmaStart = 0;
            if ((strVal = rp.FindValue("sigma_start")) != null)
                dfSigmaStart = double.Parse(strVal);

            double dfSigmaEnd = 0;
            if ((strVal = rp.FindValue("sigma_end")) != null)
                dfSigmaEnd = double.Parse(strVal);

            double dfStepStart = 1.5;
            if ((strVal = rp.FindValue("step_start")) != null)
                dfStepStart = double.Parse(strVal);

            double dfStepEnd = 1.5;
            if ((strVal = rp.FindValue("step_end")) != null)
                dfStepEnd = double.Parse(strVal);

            bool bSave = false;
            if ((strVal = rp.FindValue("save")) != null)
                bSave = bool.Parse(strVal);

            double dfPctOfDetail = .25;
            if ((strVal = rp.FindValue("pct_of_prev_detail")) != null)
                dfPctOfDetail = double.Parse(strVal);

            return new Octaves(strLayer, nIterations, dfSigmaStart, dfSigmaEnd, dfStepStart, dfStepEnd, bSave, dfPctOfDetail);
        }
    }

    /// <summary>
    /// The OctavesCollection manages a list of Octaves.
    /// </summary>
    public class OctavesCollection : IEnumerable<Octaves>
    {
        List<Octaves> m_rgOctaves = new List<Octaves>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public OctavesCollection()
        {
        }

        /// <summary>
        /// The number of Octaves in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgOctaves.Count; }
        }

        /// <summary>
        /// Returns the Octaves at an index within the collection.
        /// </summary>
        /// <param name="nIdx"></param>
        /// <returns></returns>
        public Octaves this[int nIdx]
        {
            get { return m_rgOctaves[nIdx]; }
        }

        /// <summary>
        /// Adds a new Octaves to the collection.
        /// </summary>
        /// <param name="o"></param>
        public void Add(Octaves o)
        {
            m_rgOctaves.Add(o);
        }

        /// <summary>
        /// Removes an Octaves from the collection.
        /// </summary>
        /// <param name="o">Specifies the Octaves to remove.</param>
        /// <returns>If the Octaves is found and removed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Remove(Octaves o)
        {
            return m_rgOctaves.Remove(o);
        }

        /// <summary>
        /// Removes an Octaves at a given index in the collection.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        public void RemoveAt(int nIdx)
        {
            m_rgOctaves.RemoveAt(nIdx);
        }

        /// <summary>
        /// Removes all Octaves from the collection.
        /// </summary>
        public void Clear()
        {
            m_rgOctaves.Clear();
        }

        /// <summary>
        /// Returns the enumerator for the collection.
        /// </summary>
        /// <returns>The Octaves enumerator is returned.</returns>
        public IEnumerator<Octaves> GetEnumerator()
        {
            return m_rgOctaves.GetEnumerator();
        }

        /// <summary>
        /// Returns the enumerator for the collection.
        /// </summary>
        /// <returns>The Octaves enumerator is returned.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgOctaves.GetEnumerator();
        }
    }
}
