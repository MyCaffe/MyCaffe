using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.Diagnostics;
using System.IO;
using MyCaffe.param;
using MyCaffe.common;

/// <summary>
/// The MyCaffe.data namespace contains data related classes.
/// </summary>
namespace MyCaffe.data
{
    /// <summary>
    /// Applies common transformations to the input data, such as
    /// scaling, mirroring, subtracting the image mean...
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class DataTransformer<T>
    {
        Log m_log;
        List<double> m_rgMeanValues = new List<double>();
        TransformationParameter m_param;
        SimpleDatum m_imgMean = null;
        double[] m_rgMeanData = null;
        Phase m_phase;
        CryptoRandom m_random;
        T[] m_rgTransformedData = null;
        float[] m_rgfTransformedData = null;
        double m_dfLastMin = 0;
        double m_dfLastMax = 0;
        BlobProto m_protoMean = null;
        BBoxUtility<T> m_bbox = null;
        ImageTransforms<T> m_imgTransforms = null;

        /// <summary>
        /// The DataTransformer constructor.
        /// </summary>
        /// <param name="cuda">Specifies the connection to the CudaDnn dll which is only needed when using the bbox or image transformation functionality.</param>
        /// <param name="log">Specifies the Log used for output.</param>
        /// <param name="p">Specifies the TransformationParameter used to create the DataTransformer.</param>
        /// <param name="phase">Specifies the Phase under which the DataTransformer is run.</param>
        /// <param name="nC">Specifies the channels.</param>
        /// <param name="nH">Specifies the height.</param>
        /// <param name="nW">Specifies the width.</param>
        /// <param name="imgMean">Optionally, specifies the image mean to use.</param>
        public DataTransformer(CudaDnn<T> cuda, Log log, TransformationParameter p, Phase phase, int nC, int nH, int nW, SimpleDatum imgMean = null)
        {
            m_log = log;

            int nDataSize = nC * nH * nW;

            m_param = p;
            InitRand();

            m_phase = phase;
            m_bbox = new BBoxUtility<T>(cuda, log);
            m_imgTransforms = new ImageTransforms<T>(cuda, log, m_random);

            Update(nDataSize, imgMean);
        }

        /// <summary>
        /// Resync the transformer with changes in its parameter.
        /// </summary>
        public void Update(int nDataSize = 0, SimpleDatum imgMean = null)
        {
            TransformationParameter p = m_param;

            if (imgMean != null)
                nDataSize = imgMean.Channels * imgMean.Height * imgMean.Width;

            if (nDataSize > 0 || (m_rgfTransformedData != null &&  nDataSize != m_rgfTransformedData.Length))
                m_rgTransformedData = new T[nDataSize];

            if (p.mean_file != null)
                m_protoMean = loadProtoMean(p.mean_file);

            if (p.use_imagedb_mean)
            {
                if (m_protoMean == null)
                {
                    m_imgMean = imgMean;

                    if (m_imgMean != null)
                        m_rgMeanData = m_imgMean.GetData<double>();
                }
                else
                {
                    if (m_protoMean.data.Count > 0)
                    {
                        m_rgMeanData = new double[m_protoMean.data.Count];
                        Array.Copy(m_protoMean.data.ToArray(), m_rgMeanData, m_rgMeanData.Length);
                    }
                    else
                    {
                        m_rgMeanData = m_protoMean.double_data.ToArray();
                    }
                }
            }

            if (p.mean_value.Count > 0)
            {
                m_log.CHECK(p.use_imagedb_mean == false, "Cannot specify use_image_mean and mean_value at the same time.");

                for (int c = 0; c < p.mean_value.Count; c++)
                {
                    m_rgMeanValues.Add(p.mean_value[c]);
                }
            }
        }

        /// <summary>
        /// Returns the TransformationParameter used.
        /// </summary>
        public TransformationParameter param
        {
            get { return m_param; }
        }

        /// <summary>
        /// Get/set the image mean.
        /// </summary>
        public SimpleDatum ImageMean
        {
            get { return m_imgMean; }
            set
            {
                m_imgMean = value;
                m_rgMeanData = m_imgMean.GetData<double>();
            }
        }

        private BlobProto loadProtoMean(string strFile)
        {
            try
            {
                if (!File.Exists(strFile))
                    throw new Exception("Cannot find the file '" + strFile + "'!");

                byte[] rgBytes;
                using (FileStream fs = new FileStream(strFile, FileMode.Open, FileAccess.Read))
                {
                    using (BinaryReader br = new BinaryReader(fs))
                    {
                        rgBytes = br.ReadBytes((int)fs.Length);
                    }
                }

                PersistCaffe<T> persist = new PersistCaffe<T>(m_log, true);
                return persist.LoadBlobProto(rgBytes, 1);
            }
            catch (Exception excpt)
            {
                m_log.FAIL("Loading Proto Image Mean: " + excpt.Message);
                return null;
            }
        }


        /// <summary>
        /// Infers the shape the transformed blob will have when 
        /// the transformation is applied to the data.
        /// </summary>
        /// <param name="d">Data containing the data to be transformed.</param>
        /// <returns>The inferred shape.</returns>
        public List<int> InferBlobShape(Datum d)
        {
            int nCropSize = (int)m_param.crop_size;
            int nDatumChannels = d.Channels;
            int nDatumHeight = d.Height;
            int nDatumWidth = d.Width;

            // Check dimensions
            m_log.CHECK_GT(nDatumChannels, 0, "There must be 1 or more data channels in the datum.");
            m_log.CHECK_GE(nDatumHeight, nCropSize, "The datum height must be >= the crop size of " + nCropSize.ToString() + ". To fix this change the 'crop_size' DataLayer property.");
            m_log.CHECK_GE(nDatumWidth, nCropSize, "The datum width must be >= the crop size of " + nCropSize.ToString() + ". To fix this change the 'crop_size' DataLayer property.");

            // Build BlobShape.
            List<int> rgShape = new List<int>();
            rgShape.Add(1);
            rgShape.Add(nDatumChannels);
            rgShape.Add((nCropSize > 0) ? nCropSize : nDatumHeight);
            rgShape.Add((nCropSize > 0) ? nCropSize : nDatumWidth);

            return rgShape;
        }

        /// <summary>
        /// Infers the shape the transformed blob will have when 
        /// the transformation is applied to the data.
        /// </summary>
        /// <param name="rgD">A list of data containing the data to be transformed.</param>
        /// <returns>The inferred shape.</returns>
        public List<int> InferBlobShape(List<Datum> rgD)
        {
            int nNum = rgD.Count();
            m_log.CHECK_GT(nNum, 0, "There are no datum in the input vector.");

            /// Use the first datum in the vector to InferBlobShape.
            List<int> rgShape = InferBlobShape(rgD[0]);
            // Adjust num to the size of the vector.
            rgShape[0] = nNum;

            return rgShape;
        }

        /// <summary>
        /// Initialize the underlying random number generator.
        /// </summary>
        public virtual void InitRand()
        {
            if (m_param.random_seed.HasValue)
                m_random = new CryptoRandom(false, m_param.random_seed.Value);
            else
                m_random = new CryptoRandom(false);
        }

        /// <summary>
        /// Generates a random integer from Uniform({0, 1, ..., n-1}).
        /// </summary>
        /// <param name="n">The upper bound (exclusive) value of the random number.</param>
        /// <returns>A uniformly random integer value from ({0, 1, ..., n-1}).</returns>
        protected virtual int Rand(int n)
        {
            return m_random.Next(n);
        }

        /// <summary>
        /// Returns the last min/max observed.
        /// </summary>
        public Tuple<double, double> LastRange
        {
            get { return new Tuple<double, double>(m_dfLastMin, m_dfLastMax); }
        }

        /// <summary>
        /// Transforms a list of Datum and places the transformed data into a Blob.
        /// </summary>
        /// <param name="rgDatum">Specifies a List of Datum to be transformed.</param>
        /// <param name="blobTransformed">Specifies the Blob where all transformed data is placed.</param>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies a Log for all output.</param>
        public void Transform(List<Datum> rgDatum, Blob<T> blobTransformed, CudaDnn<T> cuda, Log log)
        {
            int nDatumNum = rgDatum.Count;
            int nNum = blobTransformed.num;
            int nChannels = blobTransformed.channels;
            int nHeight = blobTransformed.height;
            int nWidth = blobTransformed.width;

            m_log.CHECK_GT(nDatumNum, 0, "There are no datum to add.");
            m_log.CHECK_LE(nDatumNum, nNum, "The size of the rgDatum must be no greater than the transformed blob num.");

            Blob<T> blobUni = new Blob<T>(cuda, log, 1, nChannels, nHeight, nWidth, false);

            for (int i = 0; i < nDatumNum; i++)
            {
                int nOffset = blobTransformed.offset(i);

                if (rgDatum[i] != null)
                    Transform(rgDatum[i], blobUni);
                else
                    blobUni.SetData(0);

                cuda.copy(blobUni.count(), blobUni.gpu_data, blobTransformed.mutable_gpu_data, 0, nOffset);
            }

            blobUni.Dispose();
        }


        /// <summary>
        /// Transforms a list of Datum and places the transformed data into a Blob.
        /// </summary>
        /// <param name="rgDatum">Specifies a List of SimpleDatum to be transformed.</param>
        /// <param name="blobTransformed">Specifies the Blob where all transformed data is placed.</param>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies a Log for all output.</param>
        /// <param name="bJustFill">Optionally, specifies to just fill the data blob with the data without actually transforming it.</param>
        public void Transform(List<SimpleDatum> rgDatum, Blob<T> blobTransformed, CudaDnn<T> cuda, Log log, bool bJustFill = false)
        {
            Transform(rgDatum.ToArray(), blobTransformed, cuda, log, bJustFill);
        }

        /// <summary>
        /// Transforms a list of Datum and places the transformed data into a Blob.
        /// </summary>
        /// <param name="rgDatum">Specifies a Array of SimpleDatum to be transformed.</param>
        /// <param name="blobTransformed">Specifies the Blob where all transformed data is placed.</param>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies a Log for all output.</param>
        /// <param name="bJustFill">Optionally, specifies to just fill the data blob with the data without actually transforming it.</param>
        public void Transform(SimpleDatum[] rgDatum, Blob<T> blobTransformed, CudaDnn<T> cuda, Log log, bool bJustFill = false)
        {
            int nDatumNum = rgDatum.Length;
            int nNum = blobTransformed.num;
            int nChannels = blobTransformed.channels;
            int nHeight = blobTransformed.height;
            int nWidth = blobTransformed.width;

            m_log.CHECK_GT(nDatumNum, 0, "There are no datum to add.");
            m_log.CHECK_LE(nDatumNum, nNum, "The size of the rgDatum must be no greater than the transformed blob num.");

            Blob<T> blobUni = new Blob<T>(cuda, log, 1, nChannels, nHeight, nWidth, false);

            for (int i = 0; i < nDatumNum; i++)
            {
                int nOffset = blobTransformed.offset(i);

                if (rgDatum[i] != null)
                {
                    if (bJustFill)
                        blobUni.mutable_cpu_data = Utility.ConvertVec<T>(rgDatum[i].RealData);
                    else
                        Transform(rgDatum[i], blobUni);
                }
                else
                {
                    blobUni.SetData(0);
                }

                cuda.copy(blobUni.count(), blobUni.gpu_data, blobTransformed.mutable_gpu_data, 0, nOffset);
            }

            blobUni.Dispose();
        }

        /// <summary>
        /// Transforms a Datum and places the dat ainto a Blob.
        /// </summary>
        /// <param name="d">Specifies the Datum to transform.</param>
        /// <param name="blob">Specifies the Blob where the transformed data is placed.</param>
        public void Transform(SimpleDatum d, Blob<T> blob)
        {
            int nCropSize = (int)m_param.crop_size;
            int nDatumChannels = d.Channels;
            int nDatumHeight = d.Height;
            int nDatumWidth = d.Width;

            // Check dimensions
            int nChannels = blob.channels;
            int nHeight = blob.height;
            int nWidth = blob.width;
            int nNum = blob.num;

            m_log.CHECK_EQ(nChannels, nDatumChannels, "The datum and blob must have equal channels.");
            m_log.CHECK_LE(nHeight, nDatumHeight, "The datum and blob must have equal height.");
            m_log.CHECK_LE(nWidth, nDatumWidth, "The datum and blob must have equal width.");
            m_log.CHECK_GE(nNum, 1, "The blob must have at least 1 item.");

            if (nCropSize > 0)
            {
                m_log.CHECK_EQ(nCropSize, nHeight, "The blob height must equal the crop size.");
                m_log.CHECK_EQ(nCropSize, nWidth, "The blob width must equal the crop size.");
            }
            else
            {
                m_log.CHECK_EQ(nDatumHeight, nHeight, "The blob height must equal the datum height.");
                m_log.CHECK_EQ(nDatumWidth, nWidth, "The blob width must equal the datum width.");
            }

            blob.mutable_cpu_data = Transform(d);
        }

        /// <summary>
        /// Transform the data into an array of transformed values.
        /// </summary>
        /// <param name="d">Data to transform.</param>
        /// <returns>Transformed data.</returns>
        public T[] Transform(SimpleDatum d)
        {
            m_dfLastMax = -double.MaxValue;
            m_dfLastMin = double.MaxValue;

            int nDatumChannels = d.Channels;
            int nDatumHeight = d.Height;
            int nDatumWidth = d.Width;
            int nCropSize = (int)m_param.crop_size;
            int nH = ((nCropSize != 0 && nCropSize < nDatumHeight) ? nCropSize : nDatumHeight);
            int nW = ((nCropSize != 0 && nCropSize < nDatumWidth) ? nCropSize : nDatumWidth);
            int nItemCount = nH * nW * nDatumChannels;

            if (m_rgTransformedData.Length != nItemCount)
                m_rgTransformedData = new T[nItemCount];

            double dfScale = m_param.scale;
            bool bDoMirror = m_param.mirror && (Rand(2) == 1) ? true : false;
            bool bUseMeanImage = m_param.use_imagedb_mean;
            List<double> rgMeanValues = null;
            double[] rgMean = null;
            bool bUseReal = d.IsRealData;

            m_log.CHECK_GT(nDatumChannels, 0, "The datum must have at least 1 channel.");
            m_log.CHECK_GE(nDatumHeight, nCropSize, "The datum height must be at least as great as the crop size " + nCropSize.ToString());
            m_log.CHECK_GE(nDatumWidth, nCropSize, "The datum width must be at least as great as the crop size " + nCropSize.ToString());

            if (bUseMeanImage)
            {
                if (m_rgMeanData == null)
                    m_log.FAIL("You must specify an imgMean parameter when using IMAGE mean subtraction.");

                rgMean = m_rgMeanData;

                int nExpected = nDatumChannels * nDatumHeight * nDatumWidth;
                m_log.CHECK_EQ(rgMean.Length, nExpected, "The size of the 'mean' image is incorrect!  Expected '" + nExpected.ToString() + "' elements, yet loaded '" + rgMean.Length + "' elements.");
            }

            if (m_rgMeanValues.Count > 0)
            {
                m_log.CHECK(m_rgMeanValues.Count == 1 || m_rgMeanValues.Count == nDatumChannels, "Specify either 1 mean value or as many as channels: " + nDatumChannels.ToString());
                rgMeanValues = new List<double>();

                for (int c = 0; c < nDatumChannels; c++)
                {
                    // Replicate the mean value for simplicity.
                    if (c == 0 || m_rgMeanValues.Count == 1)    
                        rgMeanValues.Add(m_rgMeanValues[0]);
                    else if (c > 0)
                        rgMeanValues.Add(m_rgMeanValues[c]);
                }
            }

            int nHeight = nDatumHeight;
            int nWidth = nDatumWidth;

            int h_off = 0;
            int w_off = 0;

            if (nCropSize > 0)
            {
                nHeight = nCropSize;
                nWidth = nCropSize;

                // We only do random crop when we do training
                if (m_phase == Phase.TRAIN)
                {
                    h_off = Rand(nDatumHeight - nCropSize + 1);
                    w_off = Rand(nDatumWidth - nCropSize + 1);
                }
                else
                {
                    h_off = (nDatumHeight - nCropSize) / 2;
                    w_off = (nDatumWidth - nCropSize) / 2;
                }
            }

            double dfDataElement;
            double dfTransformedElement;
            int nTopIdx;
            int nDataIdx;
            int nLen = nDatumChannels * nHeight * nWidth;
            double[] rgRealData = d.RealData;
            byte[] rgByteData = d.ByteData;
            int[] rgChannelSwap = null;

            if (m_rgfTransformedData == null || m_rgfTransformedData.Length < nLen)
                m_rgfTransformedData = new float[nLen];

            if (nDatumChannels == 3 && param.color_order == TransformationParameter.COLOR_ORDER.BGR)
                rgChannelSwap = new int[] { 2, 1, 0 };

            for (int c1 = 0; c1 < nDatumChannels; c1++)
            {
                int c = (rgChannelSwap == null) ? c1 : rgChannelSwap[c1];

                for (int h = 0; h < nHeight; h++)
                {
                    for (int w = 0; w < nWidth; w++)
                    {
                        nDataIdx = (c * nDatumHeight + h_off + h) * nDatumWidth + w_off + w;

                        if (bDoMirror)
                            nTopIdx = (c * nHeight + h) * nWidth + (nWidth - 1 - w);
                        else
                            nTopIdx = (c * nHeight + h) * nWidth + w;

                        if (bUseReal)
                            dfDataElement = (double)rgRealData[nDataIdx];
                        else
                            dfDataElement = (double)rgByteData[nDataIdx];

                        if (bUseMeanImage)
                        {
                            dfTransformedElement = (dfDataElement - rgMean[nDataIdx]) * dfScale;
                        }
                        else if (rgMeanValues != null && rgMeanValues.Count > 0)
                        {
                            dfTransformedElement = (dfDataElement - rgMeanValues[c]) * dfScale;
                        }
                        else
                        {
                            dfTransformedElement = dfDataElement * dfScale;
                        }

                        if (m_dfLastMax < dfTransformedElement)
                            m_dfLastMax = dfTransformedElement;

                        if (m_dfLastMin > dfTransformedElement)
                            m_dfLastMin = dfTransformedElement;

                        m_rgfTransformedData[nTopIdx] = (float)dfTransformedElement;
                    }
                }
            }

            Array.Copy(m_rgfTransformedData, m_rgTransformedData, nLen);

            if (m_rgTransformedData.Length == nItemCount)
                return m_rgTransformedData;

            List<T> rg = new List<T>(m_rgTransformedData);
            return rg.GetRange(0, nItemCount).ToArray();
        }

        /// <summary>
        /// Scales the data of a Blob to fit in a given  range based on the DataTransformers parameters.
        /// </summary>
        /// <param name="b"></param>
        public void SetRange(Blob<T> b)
        {
            if (m_param.forced_positive_range_max == 0)
                return;

            double dfMin = b.min_data;
            double dfMax = b.max_data;
            double dfNewMin = 0;
            double dfNewMax = m_param.forced_positive_range_max;
            double dfScale = (dfNewMax - dfNewMin) / (dfMax - dfMin);

            b.add_scalar(-dfMin);
            b.scale_data(dfScale);
        }

        /// <summary>
        /// Crop the SimpleDatum according to the bbox.
        /// </summary>
        /// <param name="d">Specifies the SimpleDatum to crop.</param>
        /// <param name="bbox">Specifies the bounding box.</param>
        /// <returns>The cropped SimpleDatum is returned.</returns>
        public SimpleDatum CropImage(SimpleDatum d, NormalizedBBox bbox)
        {
            int nDatumChannels = d.Channels;
            int nDatumHeight = d.Height;
            int nDatumWidth = d.Width;

            // Get the bbox dimension.
            NormalizedBBox clipped_bbox = m_bbox.Clip(bbox);
            NormalizedBBox scaled_bbox = m_bbox.Scale(bbox, nDatumHeight, nDatumWidth);
            int w_off = (int)scaled_bbox.xmin;
            int h_off = (int)scaled_bbox.ymin;
            int width = (int)(scaled_bbox.xmax - scaled_bbox.xmin);
            int height = (int)(scaled_bbox.ymax - scaled_bbox.ymin);

            // Crop the image using bbox.
            SimpleDatum crop_datum = new SimpleDatum(d, height, width);
            int nCropDatumSize = nDatumChannels * height * width;

            if (d.IsRealData)
            {
                double[] rgData = new double[nCropDatumSize];

                for (int h = h_off; h < h_off + height; h++)
                {
                    for (int w = w_off; w < w_off + width; w++)
                    {
                        for (int c = 0; c < nDatumChannels; c++)
                        {
                            int nDatumIdx = (c * nDatumHeight + h) * nDatumWidth + w;
                            int nCropDatumIdx = (c * height + h - h_off) * width + w - w_off;
                            rgData[nCropDatumIdx] = d.RealData[nDatumIdx];
                        }
                    }
                }

                crop_datum.SetData(rgData.ToList(), d.Label);
            }
            else
            {
                byte[] rgData = new byte[nCropDatumSize];

                for (int h = h_off; h < h_off + height; h++)
                {
                    for (int w = w_off; w < w_off + width; w++)
                    {
                        for (int c = 0; c < nDatumChannels; c++)
                        {
                            int nDatumIdx = (c * nDatumHeight + h) * nDatumWidth + w;
                            int nCropDatumIdx = (c * height + h - h_off) * width + w - w_off;
                            rgData[nCropDatumIdx] = d.ByteData[nDatumIdx];
                        }
                    }
                }

                crop_datum.SetData(rgData.ToList(), d.Label);
            }

            return crop_datum;
        }

        /// <summary>
        /// Expand the SimpleDatum according to the bbox.
        /// </summary>
        /// <param name="d">Specifies the SimpleDatum to expand.</param>
        /// <param name="expand_bbox">Specifies the bounding box.</param>
        /// <param name="fExpandRatio">Specifies the expansion ratio.</param>
        /// <returns>The expanded SimpleDatum is returned.</returns>
        public SimpleDatum ExpandImage(SimpleDatum d, NormalizedBBox expand_bbox, float fExpandRatio)
        {
            int nDatumChannels = d.Channels;
            int nDatumHeight = d.Height;
            int nDatumWidth = d.Width;

            // Get the bbox dimension.
            int width = (int)(nDatumHeight * fExpandRatio);
            int height = (int)(nDatumWidth * fExpandRatio);
            float h_off = (float)m_random.NextDouble();
            float w_off = (float)m_random.NextDouble();

            h_off = (float)Math.Floor(h_off);
            w_off = (float)Math.Floor(w_off);

            expand_bbox.xmin = -w_off / nDatumWidth;
            expand_bbox.ymin = -h_off / nDatumHeight;
            expand_bbox.xmax = (width - w_off) / nDatumWidth;
            expand_bbox.ymax = (height - h_off) / nDatumHeight;

            // Crop the image using bbox.
            SimpleDatum expand_datum = new SimpleDatum(d, height, width);
            int nExpandDatumSize = nDatumChannels * height * width;

            if (d.IsRealData)
            {
                double[] rgData = new double[nExpandDatumSize];

                for (int h = (int)h_off; h < (int)h_off + nDatumHeight; h++)
                {
                    for (int w = (int)w_off; w < (int)w_off + nDatumWidth; w++)
                    {
                        for (int c = 0; c < nDatumChannels; c++)
                        {
                            int nDatumIdx = (int)((c * nDatumHeight + h - h_off) * nDatumWidth + w - w_off);
                            int nExpandIdx = (c * height + h) * width + w;
                            rgData[nExpandIdx] = d.RealData[nDatumIdx];
                        }
                    }
                }

                expand_datum.SetData(rgData.ToList(), d.Label);
            }
            else
            {
                byte[] rgData = new byte[nExpandDatumSize];

                for (int h = (int)h_off; h < (int)h_off + nDatumHeight; h++)
                {
                    for (int w = (int)w_off; w < (int)w_off + nDatumWidth; w++)
                    {
                        for (int c = 0; c < nDatumChannels; c++)
                        {
                            int nDatumIdx = (int)((c * nDatumHeight + h - h_off) * nDatumWidth + w - w_off);
                            int nExpandIdx = (c * height + h) * width + w;
                            rgData[nExpandIdx] = d.ByteData[nDatumIdx];
                        }
                    }
                }

                expand_datum.SetData(rgData.ToList(), d.Label);
            }

            return expand_datum;
        }

        /// <summary>
        /// Distort the SimpleDatum.
        /// </summary>
        /// <param name="d">Specifies the SimpleDatum to distort.</param>
        /// <returns>The distorted SimpleDatum is returned.</returns>
        public SimpleDatum DistortImage(SimpleDatum d)
        {
            if (m_param.distortion_param == null)
                return d;

            if (m_param.distortion_param.brightness_prob == 0 &&
                m_param.distortion_param.contrast_prob == 0 &&
                m_param.distortion_param.saturation_prob == 0)
                return d;

            return m_imgTransforms.ApplyDistort(d, m_param.distortion_param);
        }

        /// <summary>
        /// Reverse the transformation made when calling Transform.
        /// </summary>
        /// <param name="blob">Specifies the input blob.</param>
        /// <param name="bIncludeMean">Specifies whether or not to add the mean back.</param>
        /// <returns>The de-processed output Datum is returned.</returns>
        public Datum UnTransform(Blob<T> blob, bool bIncludeMean = true)
        {
            double[] rgData = Utility.ConvertVec<T>(blob.update_cpu_data());
            byte[] rgOutput = new byte[rgData.Length];
            int nC = blob.channels;
            int nH = blob.height;
            int nW = blob.width;
            int[] rgChannelSwap = null;
            bool bUseMeanImage = m_param.use_imagedb_mean;
            List<double> rgMeanValues = null;
            double[] rgMean = null;
            double dfScale = m_param.scale;

            if (bUseMeanImage)
            {
                if (m_rgMeanData == null)
                    m_log.FAIL("You must specify an imgMean parameter when using IMAGE mean subtraction.");

                rgMean = m_rgMeanData;

                int nExpected = nC * nH * nW;
                m_log.CHECK_EQ(rgMean.Length, nExpected, "The size of the 'mean' image is incorrect!  Expected '" + nExpected.ToString() + "' elements, yet loaded '" + rgMean.Length + "' elements.");
            }

            if (m_rgMeanValues.Count > 0)
            {
                m_log.CHECK(m_rgMeanValues.Count == 1 || m_rgMeanValues.Count == nC, "Specify either 1 mean value or as many as channels: " + nC.ToString());
                rgMeanValues = new List<double>();

                for (int c = 0; c < nC; c++)
                {
                    // Replicate the mean value for simplicity.
                    if (c == 0 || m_rgMeanValues.Count == 1)
                        rgMeanValues.Add(m_rgMeanValues[0]);
                    else if (c > 0)
                        rgMeanValues.Add(m_rgMeanValues[c]);
                }

                rgMean = rgMeanValues.ToArray();
            }

            if (m_param.color_order == TransformationParameter.COLOR_ORDER.BGR)
                rgChannelSwap = new int[] { 2, 1, 0 };

            for (int c1 = 0; c1 < nC; c1++)
            {
                int c = (rgChannelSwap == null) ? c1 : rgChannelSwap[c1];

                for (int h = 0; h < nH; h++)
                {
                    for (int w = 0; w < nW; w++)
                    {
                        int nDataIdx = (c * nH + h) * nW + w;
                        double dfVal = (rgData[nDataIdx] / dfScale);

                        if (bIncludeMean)
                        {
                            if (bUseMeanImage)
                                dfVal += rgMean[nDataIdx];
                            else if (rgMean != null && rgMean.Length == nC)
                                dfVal += rgMean[c];
                        }

                        if (dfVal < 0)
                            dfVal = 0;
                        if (dfVal > 255)
                            dfVal = 255;

                        int nOutIdx = (c1 * nH + h) * nW + w;
                        rgOutput[nOutIdx] = (byte)dfVal;
                    }
                }
            }

            return new Datum(false, nC, nW, nH, -1, DateTime.MinValue, rgOutput.ToList(), null, 0, false, -1);
        }
    }
}
