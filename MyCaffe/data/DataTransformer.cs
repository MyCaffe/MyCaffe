using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;
using System.Diagnostics;
using System.IO;
using MyCaffe.param;
using MyCaffe.common;
using MyCaffe.param.ssd;

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
    public class DataTransformer<T> : IDisposable
    {
        Log m_log;
        CudaDnn<T> m_cuda = null;
        List<double> m_rgMeanValues = new List<double>();
        TransformationParameter m_param;
        SimpleDatum m_imgMean = null;
        float[] m_rgfMeanData = null;
        double[] m_rgdfMeanData = null;
        Phase m_phase;
        CryptoRandom m_random;
        float[] m_rgfTransformedData = null;
        double[] m_rgdfTransformedData = null;
        double m_dfLastMin = 0;
        double m_dfLastMax = 0;
        BlobProto m_protoMean = null;
        BBoxUtility<T> m_bbox = null;
        ImageTransforms<T> m_imgTransforms = null;
        long m_hImageOp = 0;

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
            m_cuda = cuda;

            if (p.resize_param != null && p.resize_param.Active)
            {
                m_log.CHECK_GT(p.resize_param.height, 0, "The resize height must be > 0.");
                m_log.CHECK_GT(p.resize_param.width, 0, "The resize width must be > 0.");
                nH = (int)p.resize_param.height;
                nW = (int)p.resize_param.width;
            }

            int nDataSize = nC * nH * nW;

            m_param = p;
            InitRand();

            m_phase = phase;
            m_bbox = new BBoxUtility<T>(cuda, log);
            m_imgTransforms = new ImageTransforms<T>(cuda, log, m_random);

            Update(nDataSize, imgMean);
        }

        /// <summary>
        /// Cleanup all resources used.
        /// </summary>
        public void Dispose()
        {
            if (m_hImageOp != 0)
            {
                m_cuda.FreeImageOp(m_hImageOp);
                m_hImageOp = 0;
            }
        }

        /// <summary>
        /// Resync the transformer with changes in its parameter.
        /// </summary>
        public void Update(int nDataSize = 0, SimpleDatum imgMean = null)
        {
            TransformationParameter p = m_param;

            if (imgMean != null)
                nDataSize = imgMean.Channels * imgMean.Height * imgMean.Width;

            if (p.mean_file != null)
                m_protoMean = loadProtoMean(p.mean_file);

            if (p.use_imagedb_mean)
            {
                if (m_protoMean == null)
                {
                    m_imgMean = imgMean;

                    if (m_imgMean != null)
                    {
                        if (typeof(T) == typeof(double))
                            m_rgdfMeanData = m_imgMean.GetData<double>();
                        else
                            m_rgfMeanData = m_imgMean.GetData<float>();
                    }
                }
                else
                {
                    if (m_protoMean.data.Count > 0)
                    {
                        if (typeof(T) == typeof(double))
                        {
                            m_rgdfMeanData = new double[m_protoMean.data.Count];
                            Array.Copy(m_protoMean.data.ToArray(), m_rgdfMeanData, m_rgdfMeanData.Length);
                        }
                        else
                        {
                            m_rgfMeanData = new float[m_protoMean.data.Count];
                            Array.Copy(m_protoMean.data.ToArray(), m_rgfMeanData, m_rgfMeanData.Length);
                        }
                    }
                    else
                    {
                        if (typeof(T) == typeof(double))
                            m_rgdfMeanData = m_protoMean.double_data.ToArray();
                        else
                            m_rgfMeanData = m_protoMean.double_data.Select(p1 => (float)p1).ToArray();
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

            if (m_param.resize_param != null && m_param.resize_param.Active)
            {
                m_log.CHECK_GT(m_param.resize_param.height, 0, "The resize height must be > 0.");
                m_log.CHECK_GT(m_param.resize_param.width, 0, "The resize width must be > 0.");
            }

            if (m_param.expansion_param != null && m_param.expansion_param.Active)
            {
                m_log.CHECK_GT(m_param.expansion_param.max_expand_ratio, 1.0, "The expansion ratio must be > 1.0.");
            }

            if (m_param.mask_param != null && m_param.mask_param.Active)
            {
                m_log.CHECK_GT(m_param.mask_param.boundary_right, m_param.mask_param.boundary_left, "The mask right must be > than the left.");
                m_log.CHECK_GT(m_param.mask_param.boundary_bottom, m_param.mask_param.boundary_top, "The mask bottom must be > than the top.");
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
                if (typeof(T) == typeof(double))
                    m_rgdfMeanData = m_imgMean.GetData<double>();
                else
                    m_rgfMeanData = m_imgMean.GetData<float>();
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
        public List<int> InferBlobShape(SimpleDatum d)
        {
            int[] rgShape = null;
            rgShape = InferBlobShape(d, rgShape);
            return new List<int>(rgShape);
        }

        /// <summary>
        /// Infers the shape the transformed blob will have when 
        /// the transformation is applied to the data.
        /// </summary>
        /// <param name="d">Data containing the data to be transformed.</param>
        /// <param name="rgShape">Specifies the shape vector to fill.</param>
        /// <returns>The inferred shape.</returns>
        public int[] InferBlobShape(SimpleDatum d, int[] rgShape)
        {
            int nCropSize = (int)m_param.crop_size;
            int nDatumChannels = d.Channels;
            int nDatumHeight = d.Height;
            int nDatumWidth = d.Width;

            // Check dimensions
            m_log.CHECK_GT(nDatumChannels, 0, "There must be 1 or more data channels in the datum.");

            // If exists and active, resize based on resize parameter.
            if (m_param.resize_param != null && m_param.resize_param.Active)
                m_imgTransforms.InferNewSize(m_param.resize_param, nDatumWidth, nDatumHeight, out nDatumWidth, out nDatumHeight);

            m_log.CHECK_GE(nDatumHeight, nCropSize, "The datum height must be >= the crop size of " + nCropSize.ToString() + ". To fix this change the 'crop_size' DataLayer property.");
            m_log.CHECK_GE(nDatumWidth, nCropSize, "The datum width must be >= the crop size of " + nCropSize.ToString() + ". To fix this change the 'crop_size' DataLayer property.");

            // Build BlobShape.
            if (rgShape == null || rgShape.Length != 4)
                rgShape = new int[4];

            rgShape[0] = 1;
            rgShape[1] = nDatumChannels;
            rgShape[2] = (nCropSize > 0) ? nCropSize : nDatumHeight;
            rgShape[3] = (nCropSize > 0) ? nCropSize : nDatumWidth;

            return rgShape;
        }

        /// <summary>
        /// Infers the shape the transformed blob will have when 
        /// the transformation is applied to the data.
        /// </summary>
        /// <param name="rgD">A list of data containing the data to be transformed.</param>
        /// <param name="rgShape">Specifies the shape vector.</param>
        /// <returns>The inferred shape.</returns>
        public int[] InferBlobShape(List<Datum> rgD, int[] rgShape)
        {
            int nNum = rgD.Count();
            m_log.CHECK_GT(nNum, 0, "There are no datum in the input vector.");

            /// Use the first datum in the vector to InferBlobShape.
            rgShape = InferBlobShape(rgD[0], rgShape);
            // Adjust num to the size of the vector.
            rgShape[0] = nNum;

            return rgShape;
        }

        /// <summary>
        /// Infers the shape of the transformed blow will have with the given channel, width and height.
        /// </summary>
        /// <param name="nChannels">Specifies the channels.</param>
        /// <param name="nWidth">Specifies the width.</param>
        /// <param name="nHeight">Specifies the height.</param>
        /// <returns>The inferred blob shape is returned.</returns>
        public List<int> InferBlobShape(int nChannels, int nWidth, int nHeight)
        {
            int nCropSize = (int)m_param.crop_size;

            // Check dimensions
            m_log.CHECK_GT(nChannels, 0, "There must be 1 or more data channels in the datum.");

            // If exists and active, resize based on resize parameter.
            if (m_param.resize_param != null && m_param.resize_param.Active)
                m_imgTransforms.InferNewSize(m_param.resize_param, nWidth, nWidth, out nWidth, out nHeight);

            m_log.CHECK_GE(nHeight, nCropSize, "The height must be >= the crop size of " + nCropSize.ToString() + ". To fix this change the 'crop_size' DataLayer property.");
            m_log.CHECK_GE(nWidth, nCropSize, "The width must be >= the crop size of " + nCropSize.ToString() + ". To fix this change the 'crop_size' DataLayer property.");

            // Build BlobShape.
            List<int> rgShape = new List<int>();
            rgShape.Add(1);
            rgShape.Add(nChannels);
            rgShape.Add((nCropSize > 0) ? nCropSize : nHeight);
            rgShape.Add((nCropSize > 0) ? nCropSize : nWidth);

            return rgShape;
        }

        /// <summary>
        /// Initialize the underlying random number generator.
        /// </summary>
        public virtual void InitRand()
        {
            if (m_param.random_seed.HasValue)
                m_random = new CryptoRandom(CryptoRandom.METHOD.DEFAULT, m_param.random_seed.Value);
            else
                m_random = new CryptoRandom(CryptoRandom.METHOD.DEFAULT);
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
        /// When active (label_mapping.Active = true), transforms the label if mapped using the label and boost.  Otherwise
        /// if not active or not mapped, no label changes are made.
        /// </summary>
        /// <param name="sd">Specifies the SimpleDatum whos label is to be transformed.</param>
        /// <returns>The new label is returned - this value is also set as the new label for the SimpleDatum.</returns>
        public int TransformLabel(SimpleDatum sd)
        {
            if (m_param.label_mapping == null || !m_param.label_mapping.Active)
                return sd.Label;

            int nNewLabel = m_param.label_mapping.MapLabel(sd.Label, sd.Boost);
            sd.SetLabel(nNewLabel);

            return nNewLabel;
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
                        blobUni.mutable_cpu_data = rgDatum[i].GetData<T>();
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
        /// <returns>When a datum contains annotations, the tranformed annotation groups are returned, otherwise <i>null</i> is returned.</returns>
        public List<AnnotationGroup> Transform(SimpleDatum d, Blob<T> blob)
        {
            bool bDoMirror;
            return Transform(d, blob, out bDoMirror);
        }

        /// <summary>
        /// Transforms a Datum and places the dat ainto a Blob.
        /// </summary>
        /// <param name="d">Specifies the Datum to transform.</param>
        /// <param name="blob">Specifies the Blob where the transformed data is placed.</param>
        /// <param name="bDoMirror">Returns whether or not a mirror took place.</param>
        /// <returns>When a datum contains annotations, the tranformed annotation groups are returned, otherwise <i>null</i> is returned.</returns>
        public List<AnnotationGroup> Transform(SimpleDatum d, Blob<T> blob, out bool bDoMirror)
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

            if (nCropSize == 0)
            {
                m_log.CHECK_EQ(nDatumHeight, nHeight, "The blob height must equal the datum height.");
                m_log.CHECK_EQ(nDatumWidth, nWidth, "The blob width must equal the datum width.");
            }

            NormalizedBBox crop_bbox = (d.annotation_type != SimpleDatum.ANNOTATION_TYPE.NONE) ? new NormalizedBBox(0, 0, 0, 0) : null;

            blob.mutable_cpu_data = Transform(d, out bDoMirror, crop_bbox);

            if (d.annotation_type != SimpleDatum.ANNOTATION_TYPE.NONE)
                return TransformAnnotation(d, crop_bbox, bDoMirror, true);

            return null;
        }

        /// <summary>
        /// Transform the data into an array of transformed values.
        /// </summary>
        /// <param name="d">Data to transform.</param>
        /// <param name="bMirror">Returns whether or not a mirror occurred.</param>
        /// <param name="crop_bbox">Optionally, specifies a crop bbox to fill out.</param>
        /// <returns>Transformed data.</returns>
        public T[] Transform(SimpleDatum d, out bool bMirror, NormalizedBBox crop_bbox = null)
        {
            if (typeof(T) == typeof(double))
                return (T[])Convert.ChangeType(transformD(d, out bMirror, crop_bbox), typeof(T[]));
            else
                return (T[])Convert.ChangeType(transformF(d, out bMirror, crop_bbox), typeof(T[]));
        }

        private float[] transformF(SimpleDatum d, out bool bMirror, NormalizedBBox crop_bbox = null)
        {
            if (!d.GetDataValid(true))
                throw new Exception("There is no " + ((d.IsRealData) ? "REAL" : "BYTE") + " data in the SimpleDatum!");

            m_dfLastMax = -double.MaxValue;
            m_dfLastMin = double.MaxValue;

            if (m_param.resize_param != null && m_param.resize_param.Active)
                d = m_imgTransforms.ApplyResize(d, m_param.resize_param);

            if (m_param.noise_param != null && m_param.noise_param.Active)
                d = m_imgTransforms.ApplyNoise(d, m_param.noise_param);

            int nDatumChannels = d.Channels;
            int nDatumHeight = d.Height;
            int nDatumWidth = d.Width;
            int nCropSize = (int)m_param.crop_size;
            int nHeight = ((nCropSize != 0 && nCropSize < nDatumHeight) ? nCropSize : nDatumHeight);
            int nWidth = ((nCropSize != 0 && nCropSize < nDatumWidth) ? nCropSize : nDatumWidth);

            float fScale = (float)m_param.scale;
            bool bDoMirror = m_param.mirror && (Rand(2) == 1) ? true : false;
            bool bUseMeanImage = m_param.use_imagedb_mean;
            List<float> rgMeanValues = null;
            float[] rgMean = null;
            bool bUseReal = d.IsRealData;

            bMirror = bDoMirror;

            m_log.CHECK_GT(nDatumChannels, 0, "The datum must have at least 1 channel.");
            m_log.CHECK_GE(nDatumHeight, nCropSize, "The datum height must be at least as great as the crop size " + nCropSize.ToString());
            m_log.CHECK_GE(nDatumWidth, nCropSize, "The datum width must be at least as great as the crop size " + nCropSize.ToString());

            if (bUseMeanImage)
            {
                if (m_rgfMeanData == null)
                    m_log.FAIL("You must specify an imgMean parameter when using IMAGE mean subtraction.");

                rgMean = m_rgfMeanData;

                int nExpected = nDatumChannels * nDatumHeight * nDatumWidth;
                m_log.CHECK_EQ(rgMean.Length, nExpected, "The size of the 'mean' image is incorrect!  Expected '" + nExpected.ToString() + "' elements, yet loaded '" + rgMean.Length + "' elements.");
            }

            if (m_rgMeanValues.Count > 0)
            {
                m_log.CHECK(m_rgMeanValues.Count == 1 || m_rgMeanValues.Count == nDatumChannels, "Specify either 1 mean value or as many as channels: " + nDatumChannels.ToString());
                rgMeanValues = new List<float>();

                for (int c = 0; c < nDatumChannels; c++)
                {
                    // Replicate the mean value for simplicity.
                    if (c == 0 || m_rgMeanValues.Count == 1)
                        rgMeanValues.Add((float)m_rgMeanValues[0]);
                    else if (c > 0)
                        rgMeanValues.Add((float)m_rgMeanValues[c]);
                }
            }

            int h_off = 0;
            int w_off = 0;

            if (nCropSize > 0)
            {
                // We only do random crop when we do training
                if (m_phase == Phase.TRAIN)
                {
                    h_off = Rand(nDatumHeight - nHeight + 1);
                    w_off = Rand(nDatumWidth - nWidth + 1);
                }
                else
                {
                    h_off = (nDatumHeight - nHeight) / 2;
                    w_off = (nDatumWidth - nWidth) / 2;
                }
            }

            // Return the normalized crop bbox if specified
            if (crop_bbox != null)
                crop_bbox.Set((float)w_off / nDatumWidth, (float)h_off / nDatumHeight, (float)(w_off + nWidth) / nDatumWidth, (float)(h_off + nHeight) / nDatumHeight);

            double[] rgdfData = d.RealDataD;
            float[] rgfData = d.RealDataF;
            byte[] rgbData = d.ByteData;
            float fDataElement;
            float fTransformedElement;
            int nTopIdx;
            int nDataIdx;
            int nItemCount = nDatumChannels * nHeight * nWidth;
            int[] rgChannelSwap = null;

            if (m_rgfTransformedData == null || m_rgfTransformedData.Length < nItemCount)
                m_rgfTransformedData = new float[nItemCount];

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

                        fDataElement = (d.IsRealData) ? ((rgfData != null) ? rgfData[nDataIdx] : (float)rgdfData[nDataIdx]) : rgbData[nDataIdx];

                        if (bUseMeanImage)
                        {
                            fTransformedElement = (fDataElement - rgMean[nDataIdx]) * fScale;
                        }
                        else if (rgMeanValues != null && rgMeanValues.Count > 0)
                        {
                            fTransformedElement = (fDataElement - rgMeanValues[c]) * fScale;
                        }
                        else
                        {
                            fTransformedElement = fDataElement * fScale;
                        }

                        if (m_dfLastMax < fTransformedElement)
                            m_dfLastMax = fTransformedElement;

                        if (m_dfLastMin > fTransformedElement)
                            m_dfLastMin = fTransformedElement;

                        m_rgfTransformedData[nTopIdx] = fTransformedElement;
                    }
                }
            }

            return m_rgfTransformedData;
        }

        private double[] transformD(SimpleDatum d, out bool bMirror, NormalizedBBox crop_bbox = null)
        {
            if (!d.GetDataValid(true))
                throw new Exception("There is no " + ((d.IsRealData) ? "REAL" : "BYTE") + " data in the SimpleDatum!");

            m_dfLastMax = -double.MaxValue;
            m_dfLastMin = double.MaxValue;

            if (m_param.resize_param != null && m_param.resize_param.Active)
                d = m_imgTransforms.ApplyResize(d, m_param.resize_param);

            if (m_param.noise_param != null && m_param.noise_param.Active)
                d = m_imgTransforms.ApplyNoise(d, m_param.noise_param);

            int nDatumChannels = d.Channels;
            int nDatumHeight = d.Height;
            int nDatumWidth = d.Width;
            int nCropSize = (int)m_param.crop_size;
            int nHeight = ((nCropSize != 0 && nCropSize < nDatumHeight) ? nCropSize : nDatumHeight);
            int nWidth = ((nCropSize != 0 && nCropSize < nDatumWidth) ? nCropSize : nDatumWidth);

            double dfScale = m_param.scale;
            bool bDoMirror = m_param.mirror && (Rand(2) == 1) ? true : false;
            bool bUseMeanImage = m_param.use_imagedb_mean;
            List<double> rgMeanValues = null;
            double[] rgMean = null;
            bool bUseReal = d.IsRealData;

            bMirror = bDoMirror;

            m_log.CHECK_GT(nDatumChannels, 0, "The datum must have at least 1 channel.");
            m_log.CHECK_GE(nDatumHeight, nCropSize, "The datum height must be at least as great as the crop size " + nCropSize.ToString());
            m_log.CHECK_GE(nDatumWidth, nCropSize, "The datum width must be at least as great as the crop size " + nCropSize.ToString());

            if (bUseMeanImage)
            {
                if (m_rgdfMeanData == null)
                    m_log.FAIL("You must specify an imgMean parameter when using IMAGE mean subtraction.");

                rgMean = m_rgdfMeanData;

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

            int h_off = 0;
            int w_off = 0;

            if (nCropSize > 0)
            {
                // We only do random crop when we do training
                if (m_phase == Phase.TRAIN)
                {
                    h_off = Rand(nDatumHeight - nHeight + 1);
                    w_off = Rand(nDatumWidth - nWidth + 1);
                }
                else
                {
                    h_off = (nDatumHeight - nHeight) / 2;
                    w_off = (nDatumWidth - nWidth) / 2;
                }
            }

            // Return the normalized crop bbox if specified
            if (crop_bbox != null)
                crop_bbox.Set((float)w_off / nDatumWidth, (float)h_off / nDatumHeight, (float)(w_off + nWidth) / nDatumWidth, (float)(h_off + nHeight) / nDatumHeight);

            double[] rgdfData = d.RealDataD;
            float[] rgfData = d.RealDataF;
            byte[] rgbData = d.ByteData;
            double dfDataElement;
            double dfTransformedElement;
            int nTopIdx;
            int nDataIdx;
            int nItemCount = nDatumChannels * nHeight * nWidth;
            int[] rgChannelSwap = null;

            if (m_rgdfTransformedData == null || m_rgdfTransformedData.Length < nItemCount)
                m_rgdfTransformedData = new double[nItemCount];

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

                        dfDataElement = (d.IsRealData) ? ((rgdfData != null) ? rgdfData[nDataIdx] : rgfData[nDataIdx]) : rgbData[nDataIdx];

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

                        m_rgdfTransformedData[nTopIdx] = dfTransformedElement;
                    }
                }
            }

            return m_rgdfTransformedData;
        }

        /// <summary>
        /// Transform the data into an array of transformed values.
        /// </summary>
        /// <param name="d">Data to transform.</param>
        /// <returns>Transformed data.</returns>
        public T[] Transform(SimpleDatum d)
        {
            bool bMirror;
            return Transform(d, out bMirror, null);
        }

        /// <summary>
        /// Transform the data into an array of transformed values.
        /// </summary>
        /// <param name="d">Data to transform.</param>
        /// <param name="rgTransformedAnnoVec">Returns the list of transfomed annoations.</param>
        /// <param name="bMirror">Returns whether or not a mirror occurred.</param>
        /// <param name="bResize">Specifies to resize the data.</param>
        /// <returns>Transformed data.</returns>
        public T[] Transform(SimpleDatum d, out List<AnnotationGroup> rgTransformedAnnoVec, out bool bMirror, bool bResize = true)
        {
            // Transform the datum.
            NormalizedBBox crop_bbox = new NormalizedBBox(0, 0, 0, 0);
            T[] rgTrans = Transform(d, out bMirror, crop_bbox);

            // Transform annoation.
            rgTransformedAnnoVec = TransformAnnotation(d, crop_bbox, bMirror, bResize);

            return rgTrans;
        }

        /// <summary>
        /// Transform the annotation data.
        /// </summary>
        /// <param name="d">Data to transform.</param>
        /// <param name="crop_bbox">Specifies the crop_bbox defined for the data.</param>
        /// <param name="bMirror">Specifies to mirror the data.</param>
        /// <param name="bResize">Specifies to resize the data.</param>
        /// <returns></returns>
        public List<AnnotationGroup> TransformAnnotation(SimpleDatum d, NormalizedBBox crop_bbox, bool bMirror, bool bResize)
        {
            int nImgHt = d.Height;
            int nImgWd = d.Width;
            List<AnnotationGroup> rgTransformedAnnotationGroup = new List<AnnotationGroup>();

            if (d.annotation_type == SimpleDatum.ANNOTATION_TYPE.BBOX)
            {
                // Go through each AnnotationGroup.
                for (int g = 0; g < d.annotation_group.Count; g++)
                {
                    // Go through each Annotation.
                    bool bHasValidAnnotation = false;
                    AnnotationGroup anno_group = d.annotation_group[g];
                    AnnotationGroup transformed_anno_group = new AnnotationGroup();

                    for (int a = 0; a < anno_group.annotations.Count; a++)
                    {
                        Annotation anno = anno_group.annotations[a];
                        NormalizedBBox bbox = anno.bbox;

                        // Adjust bounding box annotation.
                        NormalizedBBox resize_bbox = bbox;
                        if (bResize && m_param.resize_param != null && m_param.resize_param.Active)
                        {
                            m_log.CHECK_GT(nImgHt, 0, "The image height must be > 0!");
                            m_log.CHECK_GT(nImgWd, 0, "The image width must be > 0!");
                            resize_bbox = m_imgTransforms.UpdateBBoxByResizePolicy(m_param.resize_param, nImgWd, nImgHt, resize_bbox);
                        }

                        if (m_param.emit_constraint != null && m_param.emit_constraint.Active && !m_bbox.MeetEmitConstraint(crop_bbox, resize_bbox, m_param.emit_constraint))
                            continue;

                        NormalizedBBox proj_bbox;
                        if (m_bbox.Project(crop_bbox, resize_bbox, out proj_bbox))
                        {
                            bHasValidAnnotation = true;
                            Annotation transformed_anno = new Annotation(proj_bbox.Clone(), anno.instance_id);
                            NormalizedBBox transformed_bbox = transformed_anno.bbox;

                            if (bMirror)
                            {
                                float fTemp = transformed_bbox.xmin;
                                transformed_bbox.xmin = 1 - transformed_bbox.xmax;
                                transformed_bbox.xmax = 1 - fTemp;
                            }
                            else if (bResize && m_param.resize_param != null && m_param.resize_param.Active)
                            {
                                m_bbox.Extrapolate(m_param.resize_param, nImgHt, nImgWd, crop_bbox, transformed_bbox);
                            }

                            transformed_anno_group.annotations.Add(transformed_anno);
                        }
                    }

                    // Save for output.
                    if (bHasValidAnnotation)
                    {
                        transformed_anno_group.group_label = anno_group.group_label;
                        rgTransformedAnnotationGroup.Add(transformed_anno_group);
                    }
                }
            }
            else
            {
                m_log.FAIL("Unknown annotation type.");
            }

            return rgTransformedAnnotationGroup;
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
            NormalizedBBox scaled_bbox = m_bbox.Scale(clipped_bbox, nDatumHeight, nDatumWidth);
            int w_off = (int)scaled_bbox.xmin;
            int h_off = (int)scaled_bbox.ymin;
            int width = (int)(scaled_bbox.xmax - scaled_bbox.xmin);
            int height = (int)(scaled_bbox.ymax - scaled_bbox.ymin);

            // Crop the image using bbox.
            SimpleDatum crop_datum = new SimpleDatum(d, height, width);
            int nCropDatumSize = nDatumChannels * height * width;

            if (d.IsRealData)
            {
                if (d.RealDataD != null)
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
                                rgData[nCropDatumIdx] = d.RealDataD[nDatumIdx];
                            }
                        }
                    }

                    crop_datum.SetData(rgData.ToList(), d.Label);
                }
                else if (d.RealDataF != null)
                {
                    float[] rgData = new float[nCropDatumSize];

                    for (int h = h_off; h < h_off + height; h++)
                    {
                        for (int w = w_off; w < w_off + width; w++)
                        {
                            for (int c = 0; c < nDatumChannels; c++)
                            {
                                int nDatumIdx = (c * nDatumHeight + h) * nDatumWidth + w;
                                int nCropDatumIdx = (c * height + h - h_off) * width + w - w_off;
                                rgData[nCropDatumIdx] = d.RealDataF[nDatumIdx];
                            }
                        }
                    }

                    crop_datum.SetData(rgData.ToList(), d.Label);
                }
                else
                {
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                }
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
            int width = (int)(nDatumWidth * fExpandRatio);
            int height = (int)(nDatumHeight * fExpandRatio);
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
                if (d.RealDataD != null)
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
                                rgData[nExpandIdx] = d.RealDataD[nDatumIdx];
                            }
                        }
                    }

                    expand_datum.SetData(rgData.ToList(), d.Label);
                }
                else if (d.RealDataF != null)
                {
                    float[] rgData = new float[nExpandDatumSize];

                    for (int h = (int)h_off; h < (int)h_off + nDatumHeight; h++)
                    {
                        for (int w = (int)w_off; w < (int)w_off + nDatumWidth; w++)
                        {
                            for (int c = 0; c < nDatumChannels; c++)
                            {
                                int nDatumIdx = (int)((c * nDatumHeight + h - h_off) * nDatumWidth + w - w_off);
                                int nExpandIdx = (c * height + h) * width + w;
                                rgData[nExpandIdx] = d.RealDataF[nDatumIdx];
                            }
                        }
                    }

                    expand_datum.SetData(rgData.ToList(), d.Label);
                }
                else
                {
                    throw new Exception("SimpleDatum: Both the RealDataD and RealDataF are null!");
                }
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

        private float randomValue(float fMin, float fMax)
        {
            float fVal = (float)m_random.NextDouble();
            return (fVal * (fMax - fMin)) + fMin;
        }

        /// <summary>
        /// Expand the datum and adjust the AnnotationGroup.
        /// </summary>
        /// <param name="d">Specifies the datum to expand.</param>
        /// <returns>The newly expanded datum is returned.</returns>
        public SimpleDatum ExpandImage(SimpleDatum d)
        {
            if (m_param.expansion_param == null || !m_param.expansion_param.Active)
                return new SimpleDatum(d, true);

            float fExpandProb = m_param.expansion_param.prob;
            float fProb = (float)m_random.NextDouble();

            if (fProb > fExpandProb)
                return new SimpleDatum(d, true);

            float fMaxExpandRatio = m_param.expansion_param.max_expand_ratio;
            if (Math.Abs(fMaxExpandRatio - 1.0f) < 1e-2)
                return new SimpleDatum(d, true);

            float fExpandRatio = randomValue(1.0f, fMaxExpandRatio);

            // Expand the datum.
            NormalizedBBox expand_bbox = new NormalizedBBox(0, 0, 0, 0);
            SimpleDatum expanded_datum = ExpandImage(d, expand_bbox, fExpandRatio);
            expanded_datum.annotation_type = d.annotation_type;

            // Transform the annotation according to the crop_bbox.
            bool bMirror = false;
            bool bResize = false;
            expanded_datum.annotation_group = TransformAnnotation(d, expand_bbox, bMirror, bResize);

            return expanded_datum;
        }

        /// <summary>
        /// Distort the SimpleDatum.
        /// </summary>
        /// <param name="d">Specifies the SimpleDatum to distort.</param>
        /// <remarks>Note this function only applies when the distortion parameter 'use_gpu' = false, otherwise the
        /// distoration is applied after the data is transferred to the GPU.</remarks>
        /// <returns>The distorted SimpleDatum is returned.</returns>
        public SimpleDatum DistortImage(SimpleDatum d)
        {
            if (m_param.distortion_param == null || !m_param.distortion_param.Active)
                return d;

            if (m_param.distortion_param.use_gpu)
                return d;

            if (m_param.distortion_param.brightness_prob == 0 &&
                m_param.distortion_param.contrast_prob == 0 &&
                m_param.distortion_param.saturation_prob == 0)
                return d;

            return m_imgTransforms.ApplyDistortEx(d, m_param.distortion_param);
        }

        /// <summary>
        /// Distort the images within a Blob.
        /// </summary>
        /// <param name="b">Specifies the Blob to distort.</param>
        public void DistortImage(Blob<T> b)
        {
            if (m_param.distortion_param == null || !m_param.distortion_param.Active)
                return;

            if (!m_param.distortion_param.use_gpu)
                return;

            if (m_param.distortion_param.brightness_prob == 0 &&
                m_param.distortion_param.contrast_prob == 0 &&
                m_param.distortion_param.saturation_prob == 0)
                return;

            if (m_hImageOp == 0)
            {
                m_hImageOp = m_cuda.CreateImageOp(b.num,
                               m_param.distortion_param.brightness_prob,
                               m_param.distortion_param.brightness_delta,
                               m_param.distortion_param.contrast_prob,
                               m_param.distortion_param.contrast_lower,
                               m_param.distortion_param.contrast_upper,
                               m_param.distortion_param.saturation_prob,
                               m_param.distortion_param.saturation_lower,
                               m_param.distortion_param.saturation_upper,
                               m_param.distortion_param.random_seed);
            }

            m_cuda.DistortImage(m_hImageOp, b.count(), b.num, b.count(1), b.gpu_data, b.mutable_gpu_data);
        }

        /// <summary>
        /// Maks out portions of the SimpleDatum.
        /// </summary>
        /// <param name="d">Specifies the SimpleDatum to mask.</param>
        /// <returns>The masked SimpleDatum is returned.</returns>
        public SimpleDatum MaskImage(SimpleDatum d)
        {
            if (m_param.mask_param == null || !m_param.mask_param.Active)
                return d;

            int nL = m_param.mask_param.boundary_left;
            int nR = m_param.mask_param.boundary_right;
            int nT = m_param.mask_param.boundary_top;
            int nB = m_param.mask_param.boundary_bottom;
            int nDim = d.Height * d.Width;

            for (int c = 0; c < d.Channels; c++)
            {
                for (int y = 0; y < d.Height; y++)
                {
                    for (int x = 0; x < d.Width; x++)
                    {
                        int nIdx = c * nDim + y * d.Width + x;

                        if (y >= nT && y <= nB && x >= nL && x <= nR)
                        {
                            if (d.IsRealData)
                            {
                                if (d.RealDataD != null)
                                    d.RealDataD[nIdx] = 0;
                                else if (d.RealDataF != null)
                                    d.RealDataF[nIdx] = 0;
                            }
                            else
                            {
                                d.ByteData[nIdx] = 0;
                            }
                        }
                    }
                }
            }

            return d;
        }

        /// <summary>
        /// Mask out the data based on the shape of the specified SimpleDatum.
        /// </summary>
        /// <param name="rgShape">Specifies the shape of the data.</param>
        /// <param name="rgData">Specifies the data.</param>
        /// <returns>The newly masked data is returned.</returns>
        public float[] MaskData(int[] rgShape, float[] rgData)
        {
            if (m_param.mask_param == null || !m_param.mask_param.Active)
                return rgData;

            int nL = m_param.mask_param.boundary_left;
            int nR = m_param.mask_param.boundary_right;
            int nT = m_param.mask_param.boundary_top;
            int nB = m_param.mask_param.boundary_bottom;
            int nC = rgShape[1];
            int nH = rgShape[2];
            int nW = rgShape[3];
            int nDim = nH * nW;

            for (int c = 0; c < nC; c++)
            {
                for (int y = 0; y < nH; y++)
                {
                    for (int x = 0; x < nW; x++)
                    {
                        int nIdx = c * nDim + y * nW + x;

                        if (y >= nT && y <= nB && x >= nL && x <= nR)
                        {
                            rgData[nIdx] = 0f;
                        }
                    }
                }
            }

            return rgData;
        }

        /// <summary>
        /// Reverse the transformation made when calling Transform.
        /// </summary>
        /// <param name="blob">Specifies the input blob.</param>
        /// <param name="bIncludeMean">Specifies whether or not to add the mean back.</param>
        /// <returns>The de-processed output Datum is returned.</returns>
        public Datum UnTransform(Blob<T> blob, bool bIncludeMean = true)
        {
            if (typeof(T) == typeof(double))
                return unTransformD(blob, bIncludeMean);
            else
                return unTransformF(blob, bIncludeMean);
        }

        private Datum unTransformF(Blob<T> blob, bool bIncludeMean = true)
        {
            float[] rgData = Utility.ConvertVecF<T>(blob.update_cpu_data());
            byte[] rgOutput = new byte[rgData.Length];
            int nC = blob.channels;
            int nH = blob.height;
            int nW = blob.width;
            int[] rgChannelSwap = null;
            bool bUseMeanImage = m_param.use_imagedb_mean;
            List<float> rgMeanValues = null;
            float[] rgMean = null;
            float dfScale = (float)m_param.scale;

            if (bUseMeanImage)
            {
                if (m_rgfMeanData == null)
                    m_log.FAIL("You must specify an imgMean parameter when using IMAGE mean subtraction.");

                rgMean = m_rgfMeanData;

                int nExpected = nC * nH * nW;
                m_log.CHECK_EQ(rgMean.Length, nExpected, "The size of the 'mean' image is incorrect!  Expected '" + nExpected.ToString() + "' elements, yet loaded '" + rgMean.Length + "' elements.");
            }

            if (m_rgMeanValues.Count > 0)
            {
                m_log.CHECK(m_rgMeanValues.Count == 1 || m_rgMeanValues.Count == nC, "Specify either 1 mean value or as many as channels: " + nC.ToString());
                rgMeanValues = new List<float>();

                for (int c = 0; c < nC; c++)
                {
                    // Replicate the mean value for simplicity.
                    if (c == 0 || m_rgMeanValues.Count == 1)
                        rgMeanValues.Add((float)m_rgMeanValues[0]);
                    else if (c > 0)
                        rgMeanValues.Add((float)m_rgMeanValues[c]);
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
                        float fVal = (rgData[nDataIdx] / dfScale);

                        if (bIncludeMean)
                        {
                            if (bUseMeanImage)
                                fVal += rgMean[nDataIdx];
                            else if (rgMean != null && rgMean.Length == nC)
                                fVal += rgMean[c];
                        }

                        if (fVal < 0)
                            fVal = 0;
                        if (fVal > 255)
                            fVal = 255;

                        int nOutIdx = (c1 * nH + h) * nW + w;
                        rgOutput[nOutIdx] = (byte)fVal;
                    }
                }
            }

            return new Datum(false, nC, nW, nH, -1, DateTime.MinValue, rgOutput.ToList(), 0, false, -1);
        }

        private Datum unTransformD(Blob<T> blob, bool bIncludeMean = true)
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
                if (m_rgdfMeanData == null)
                    m_log.FAIL("You must specify an imgMean parameter when using IMAGE mean subtraction.");

                rgMean = m_rgdfMeanData;

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

            return new Datum(false, nC, nW, nH, -1, DateTime.MinValue, rgOutput.ToList(), 0, false, -1);
        }
    }
}
