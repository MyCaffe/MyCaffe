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
        double m_dfLastMin = 0;
        double m_dfLastMax = 0;
        BlobProto m_protoMean = null;

        /// <summary>
        /// The DataTransformer constructor.
        /// </summary>
        /// <param name="log">Specifies the Log used for output.</param>
        /// <param name="p">Specifies the TransformationParameter used to create the DataTransformer.</param>
        /// <param name="phase">Specifies the Phase under which the DataTransformer is run.</param>
        /// <param name="imgMean">Optionally, specifies the image mean to use.</param>
        public DataTransformer(Log log, TransformationParameter p, Phase phase, SimpleDatum imgMean = null)
        {
            m_log = log;

            if (p.mean_file != null)
                m_protoMean = loadProtoMean(p.mean_file);

            int nDataSize = 56 * 56 * 3;

            if (imgMean != null)
                nDataSize = imgMean.Channels * imgMean.Height * imgMean.Width;

            m_rgTransformedData = new T[nDataSize];
            m_param = p;
            m_phase = phase;

            InitRand();

            if (p.use_image_mean)
            {
                if (m_protoMean == null)
                {
                    m_imgMean = imgMean;
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
                m_log.CHECK(p.use_image_mean == false, "Cannot specify use_image_mean and mean_value at the same time.");

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

        private BlobProto loadProtoMean(string strFile)
        {
            try
            {
                if (!File.Exists(strFile))
                    throw new Exception("Cannot find the file '" + strFile + "'!");

                byte[] rgBytes;
                using (FileStream fs = new FileStream(strFile, FileMode.Open))
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
            m_log.CHECK_GE(nDatumHeight, nCropSize, "The datum height must be >= the crop size of " + nCropSize.ToString());
            m_log.CHECK_GE(nDatumWidth, nCropSize, "The datum width must be >= the crop size of " + nCropSize.ToString());

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
                Transform(rgDatum[i], blobUni);
                cuda.copy(blobUni.count(), blobUni.gpu_data, blobTransformed.mutable_gpu_data, 0, nOffset);
            }

            blobUni.Dispose();
        }

        /// <summary>
        /// Transforms a Datum and places the dat ainto a Blob.
        /// </summary>
        /// <param name="d">Specifies the Datum to transform.</param>
        /// <param name="blob">Specifies the Blob where the transformed data is placed.</param>
        public void Transform(Datum d, Blob<T> blob)
        {
            int nCropSize = (int)m_param.crop_size;
            int nDatumChannels = d.channels;
            int nDatumHeight = d.height;
            int nDatumWidth = d.width;

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
        public T[] Transform(Datum d)
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
            bool bUseMeanImage = m_param.use_image_mean;
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
            float[] rgTransformedData = new float[nLen];
            int[] rgChannelSwap = null;

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

                        rgTransformedData[nTopIdx] = (float)dfTransformedElement;
                    }
                }
            }

            Array.Copy(rgTransformedData, m_rgTransformedData, nLen);

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
    }
}
