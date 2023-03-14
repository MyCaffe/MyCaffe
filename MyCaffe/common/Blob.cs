using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading;
using System.Diagnostics;
using System.Drawing;
using MyCaffe.param;
using MyCaffe.basecode;
using System.Drawing.Text;

namespace MyCaffe.common
{
    /// <summary>
    /// The Blob is the main holder of data that moves through the Layers of the Net.
    /// </summary>
    /// <remarks>
    /// Each blob holds Data and optionally Diff where the data is passed through the Net
    /// Layers on each forward pass, and the Diff contains the errors passed backward
    /// throgh the Net Layers on the backward pass.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class Blob<T> : IDisposable
    {
        T m_tZero;
        T m_tOne;
        T m_tMinusOne;
        string m_strName = "";
        CudaDnn<T> m_cuda;
        Log m_log;
        bool m_bIncludeDiff = true;
        bool m_bOwnData = true;
        SyncedMemory<T> m_data = null;
        bool m_bOwnDiff = true;
        SyncedMemory<T> m_diff = null;
        bool m_bOwnShape = true;
        SyncedMemory<T> m_shape = null;
        int m_nCount = 0;
        int m_nCapacity = 0;
        List<int> m_rgShape = new List<int>();
        int m_nIdx = -1;
        BLOB_TYPE m_type = BLOB_TYPE.DATA;
        object m_tag = null;
        bool m_bFreezeLearning = false;
        bool m_bCpuDataReadyForPush = false;
        bool m_bReshapeWhenSharing = false;
        bool m_bSnapshotRequested = false;
        bool m_bPadded = false;
        Dictionary<string, double> m_rgParam = new Dictionary<string, double>();

        /// <summary>
        /// Defines the maximum number of Axes supported by the Blob.
        /// </summary>
        public const int MAX_BLOB_AXES = 32;

        /// <summary>
        /// The Blob constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance used to communidate with Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="bIncludeDiff">Optionally, specifies whether or not to include (and allocate) the Diff data.</param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half size (FP16) for both data and diff.  This option is only available when using the <i>float</i> base type 'T'.</param>
        public Blob(CudaDnn<T> cuda, Log log, bool bIncludeDiff = true, bool bUseHalfSize = false)
        {
            if (bUseHalfSize && typeof(T) != typeof(float))
            {
                bUseHalfSize = false;

                if (log != null)
                    log.WriteLine("WARNING: Half sizes currently only supported with the 'float' base type - changing back to full size.");
            }

            m_tZero = Zero;
            m_tMinusOne = MinusOne;
            m_tOne = One;
            m_bIncludeDiff = bIncludeDiff;
            m_cuda = cuda;
            m_log = log;
            m_shape = new SyncedMemory<T>(m_cuda, m_log);
            m_data = new SyncedMemory<T>(m_cuda, m_log, 0, null, bUseHalfSize);

            if (m_bIncludeDiff)
                m_diff = new SyncedMemory<T>(m_cuda, m_log, 0, null, bUseHalfSize);
        }


        /// <summary>
        /// <b>DEPRECIATED</b>; use <code>Blob(...,rgShape)</code> instead.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance used to communidate with Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="nNum">Specifies the number of inputs.</param>
        /// <param name="nChannels">Specifies the number of channels per input.</param>
        /// <param name="nHeight">Specifies the height of each input.</param>
        /// <param name="nWidth">Specifies the width of each input.</param>
        /// <param name="bIncludeDiff">Optionally, specifies whether or not to include (and allocate) the Diff data.</param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half size (FP16) for both data and diff.  This option is only available when using the <i>float</i> base type 'T'.</param>
        public Blob(CudaDnn<T> cuda, Log log, int nNum, int nChannels, int nHeight, int nWidth, bool bIncludeDiff = true, bool bUseHalfSize = false)
            : this(cuda, log, bIncludeDiff, bUseHalfSize)
        {
            // Capacity must be initialized before calling Reshape.
            m_nCapacity = 0;
            Reshape(nNum, nChannels, nHeight, nWidth);
        }


        /// <summary>
        /// The Blob constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance used to communidate with Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="rgShape">Specifies the shape of each axis of the Blob.</param>
        /// <param name="bIncludeDiff">Optionally, specifies whether or not to include (and allocate) the Diff data.</param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half size (FP16) for both data and diff.  This option is only available when using the <i>float</i> base type 'T'.</param>
        public Blob(CudaDnn<T> cuda, Log log, List<int> rgShape, bool bIncludeDiff = true, bool bUseHalfSize = false)
            : this(cuda, log, bIncludeDiff, bUseHalfSize)
        {
            // Capacity must be initialized before calling Reshape.
            m_nCapacity = 0;
            Reshape(rgShape);
        }

        /// <summary>
        /// The Blob constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance used to communidate with Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="rgShape">Specifies the shape of each axis of the Blob.</param>
        /// <param name="bIncludeDiff">Optionally, specifies whether or not to include (and allocate) the Diff data.</param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half size (FP16) for both data and diff.  This option is only available when using the <i>float</i> base type 'T'.</param>
        public Blob(CudaDnn<T> cuda, Log log, int[] rgShape, bool bIncludeDiff = true, bool bUseHalfSize = false)
            : this(cuda, log, bIncludeDiff, bUseHalfSize)
        {
            // Capacity must be initialized before calling Reshape.
            m_nCapacity = 0;
            Reshape(rgShape);
        }

        /// <summary>
        /// The Blob constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance used to communidate with Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="b">Create this blob to be like another Blob (e.g. same shape).</param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half size (FP16) for both data and diff.  This option is only available when using the <i>float</i> base type 'T'.</param>
        public Blob(CudaDnn<T> cuda, Log log, Blob<T> b, bool bUseHalfSize = false)
            : this(cuda, log, (b.m_diff != null) ? true : false, bUseHalfSize)
        {
            // Capacity must be initialized before calling Reshape.
            m_nCapacity = 0;
            ReshapeLike(b);
        }

        /// <summary>
        /// The Blob constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance used to communidate with Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="d">Specifies the datum for which the Blob is shaped to match.</param>
        /// <param name="bCopyData">Optionally, specifies whether or not to actually copy the data.  When <i>false</i>, the shape is set, but no data is copied.</param>
        /// <param name="bIncludeDiff">Optionally, specifies whether or not to include (and allocate) the Diff data.</param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half size (FP16) for both data and diff.  This option is only available when using the <i>float</i> base type 'T'.</param>
        public Blob(CudaDnn<T> cuda, Log log, SimpleDatum d, bool bCopyData = false, bool bIncludeDiff = true, bool bUseHalfSize = false)
            : this(cuda, log, bIncludeDiff, bUseHalfSize)
        {
            SetData(d, true, bCopyData);
        }

        /// <summary>
        /// The Blob constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance used to communidate with Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="bp">Specifies the BlobProto used to load the Blob.</param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half size (FP16) for both data and diff.  This option is only available when using the <i>float</i> base type 'T'.</param>
        public Blob(CudaDnn<T> cuda, Log log, BlobProto bp, bool bUseHalfSize = false)
            : this(cuda, log, true, bUseHalfSize)
        {
            FromProto(bp);
        }

        /// <summary>
        /// The Blob constructor used to copy another blob by creating memory pointers to its data thus sharing the same GPU memory.
        /// </summary>
        /// <param name="blob">Specifies the blob whos data is to be shared.</param>
        /// <param name="lCount">Specifies the number of items to share.</param>
        /// <param name="lOffset">Specifies the offset into the blob where the shareing starts.</param>
        public Blob(Blob<T> blob, long lCount, long lOffset)
            : this(blob.Cuda, blob.Log, blob.m_bIncludeDiff)
        {
            m_data.set_gpu_data(blob.gpu_data, lCount, lOffset);

            if (m_diff != null)
                m_diff.set_gpu_data(blob.gpu_diff, lCount, lOffset);
        }

        /// <summary>
        /// Get a blob parameter.
        /// </summary>
        /// <param name="strName">Specifies the name of the blob parameter.</param>
        /// <returns>If the parameter exists it is returned, otherwise null is returned.</returns>
        public double? GetParameter(string strName)
        {
            if (m_rgParam.ContainsKey(strName))
                return m_rgParam[strName];

            return null;
        }
        
        /// <summary>
        /// Set a blob parameter.
        /// </summary>
        /// <param name="strName">Specifies the name of the blob parameter.</param>
        /// <param name="dfVal">Specifies the value of the blob parameter.</param>
        public void SetParameter(string strName, double dfVal)
        {
            if (!m_rgParam.ContainsKey(strName))
                m_rgParam.Add(strName, dfVal);
            else
                m_rgParam[strName] = dfVal;
        }

        /// <summary>
        /// Returns Zero (0) in type T.
        /// </summary>
        public static T Zero
        {
            get { return (T)Convert.ChangeType(0, typeof(T)); }
        }

        /// <summary>
        /// Returns One (1) in type T.
        /// </summary>
        public static T One
        {
            get { return (T)Convert.ChangeType(1, typeof(T)); }
        }

        /// <summary>
        /// Returns MinusOne (-1) in type T.
        /// </summary>
        public static T MinusOne
        {
            get { return (T)Convert.ChangeType(-1, typeof(T)); }
        }

        /// <summary>
        /// Get/set the padding state of the blob.
        /// </summary>
        public bool Padded
        {
            get { return m_bPadded; }
            set { m_bPadded = value; }
        }

        /// <summary>
        /// Returns the amount of memory (in bytes) required to convert from base to half and back.
        /// </summary>
        /// <param name="bUseHalfSize">Specifies whether or not we are converting to half size or not.</param>
        public ulong GetConversionWorkSize(bool bUseHalfSize)
        {
            // (count (for data) + count (for diff)) * base type size
            return (ulong)count() * 2 * CudaDnn<T>.basetype_size(bUseHalfSize);
        }

        /// <summary>
        /// Converts this blob from its base type to the half type.
        /// </summary>
        /// <param name="hWorkMem">Specifies the work memory.</param>
        /// <param name="lWorkSize">Specifies the work size.</param>
        /// <param name="bData">Specifies to convert the data.</param>
        /// <param name="bDiff">Specifies to convert the diff</param>
        public void ConvertToHalf(long hWorkMem, ulong lWorkSize, bool bData, bool bDiff)
        {
            int nCount = count();
            ulong lSize = (ulong)nCount * 2 * CudaDnn<T>.basetype_size(true);

            if ((long)lSize < 0)
                throw new Exception("Memory out of range!");

            if (lWorkSize < lSize)
                throw new Exception("Work memory is not large enough!");

            if (bData)
                m_cuda.copy(nCount, gpu_data, hWorkMem, 0, 0, -1, null, true);

            if (bDiff)
                m_cuda.copy(nCount, gpu_diff, hWorkMem, 0, nCount, -1, null, true);

            Reshape(shape(), true);

            if (bData)
                m_cuda.copy(nCount, hWorkMem, mutable_gpu_data, 0, 0, -1, true, null);

            if (bDiff)
                m_cuda.copy(nCount, hWorkMem, mutable_gpu_diff, nCount, 0, -1, true, null);
        }

        /// <summary>
        /// Converts this blob from the half type to the base type.
        /// </summary>
        /// <param name="hWorkMem">Specifies the work memory.</param>
        /// <param name="lWorkSize">Specifies the work size.</param>
        /// <param name="bData">Specifies to convert the data.</param>
        /// <param name="bDiff">Specifies to convert the diff</param>
        public void ConvertToBase(long hWorkMem, ulong lWorkSize, bool bData, bool bDiff)
        {
            int nCount = count();
            ulong lSize = (ulong)nCount * 2 * CudaDnn<T>.basetype_size(false);

            if ((long)lSize < 0)
                throw new Exception("Memory out of range!");

            if (lWorkSize < lSize)
                throw new Exception("Work memory is not large enough!");

            if (bData)
                m_cuda.copy(nCount, gpu_data, hWorkMem, 0, 0, -1, null, false);

            if (bDiff)
                m_cuda.copy(nCount, gpu_diff, hWorkMem, 0, nCount, -1, null, false);

            Reshape(shape(), false);

            if (bData)
                m_cuda.copy(nCount, hWorkMem, mutable_gpu_data, 0, 0, -1, false, null);

            if (bDiff)
                m_cuda.copy(nCount, hWorkMem, mutable_gpu_diff, nCount, 0, -1, false, null);
        }

        /// <summary>
        /// Returns whether or not this blob is using half sizes.
        /// </summary>
        public bool HalfSize
        {
            get { return m_data.HalfSize; }
        }

        /// <summary>
        /// Specifies whether or not the diff is applied to the data during Update.  When freeze learning = <i>true</i>, the update is skipped.
        /// </summary>
        public bool freeze_learning
        {
            get { return m_bFreezeLearning; }
            set { m_bFreezeLearning = value; }
        }

        /// <summary>
        /// Returns the CudaDnn object that manages the Blob's memory."/>
        /// </summary>
        public CudaDnn<T> Cuda
        {
            get { return m_cuda; }
        }

        /// <summary>
        /// Returns the Log associated with the blob.
        /// </summary>
        public Log Log
        {
            get { return m_log; }
        }

        /// <summary>
        /// Releases all resources used by the Blob (including both GPU and Host).
        /// </summary>
        /// <param name="bDisposing">Set to <i>true</i> when disposing the object.</param>
        protected virtual void Dispose(bool bDisposing)
        {
            if (m_diff != null)
            {
                if (m_bOwnDiff)
                    m_diff.Dispose();
                m_diff = null;
            }

            if (m_data != null)
            {
                if (m_bOwnData) 
                    m_data.Dispose();
                m_data = null;
            }

            if (m_shape != null)
            {
                if (m_bOwnShape)
                    m_shape.Dispose();
                m_shape = null;
            }
        }

        /// <summary>
        /// Releases all resources used by the Blob (including both GPU and Host).
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }

        /// <summary>
        /// <b>DEPRECIATED</b>; use <code>Reshape(rgShape)</code> instead.
        /// </summary>
        /// <param name="nNum"></param>
        /// <param name="nChannels"></param>
        /// <param name="nHeight"></param>
        /// <param name="nWidth"></param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half sized memory.</param>
        public void Reshape(int nNum, int nChannels, int nHeight, int nWidth, bool? bUseHalfSize = null)
        {
            Reshape(new List<int>() { nNum, nChannels, nHeight, nWidth }, bUseHalfSize);
        }

        private string toString(List<int> rgShape)
        {
            return toString(rgShape.ToArray());
        }

        private string toString(int[] rgShape)
        {
            string str = "{";

            for (int i = 0; i < rgShape.Length; i++)
            {
                str += rgShape[i].ToString();
                str += ", ";
            }

            str = str.TrimEnd(' ', ',');
            str += "}";

            return str;
        }

        private void reshapeShape(List<int> rgShape)
        {
            reshapeShape(rgShape.ToArray());
        }

        private void reshapeShape(int[] rgShape)
        {
            m_log.CHECK_LE(rgShape.Length, MAX_BLOB_AXES, "The number of axes cannot exceed " + MAX_BLOB_AXES.ToString());
            m_nCount = 1;

            m_rgShape = new List<int>();

            if (m_shape == null)
                m_shape = new SyncedMemory<T>(m_cuda, m_log, rgShape.Length);
            else if (m_shape.Capacity < rgShape.Length)
                m_shape.Allocate(rgShape.Length);
            else if (m_shape.Count != rgShape.Length)
            {
                m_shape.Count = rgShape.Length;
                m_shape.ZeroAll();
            }

            if (rgShape.Length > 0)
            {
                T[] rgShapeData = m_shape.cpu_data;

                if (rgShapeData == null || rgShapeData.Length != rgShape.Length)
                {
                    rgShapeData = m_shape.update_cpu_data();

                    if (rgShapeData == null || rgShapeData.Length != rgShape.Length)
                        rgShapeData = new T[rgShape.Length];
                }

                bool bDirty = false;

                for (int i = 0; i < rgShape.Length; i++)
                {
//                    m_log.CHECK_GE(rgShape[i], 0, "The shape value at " + i.ToString() + " must be >= 0.");

                    if (rgShape[i] < 0)
                    {
                        string strBlobName = (!string.IsNullOrEmpty(m_strName)) ? "Blob '" + m_strName + "': " : "";
                        m_log.FAIL(strBlobName + "The shape value at " + i.ToString() + " of shape " + toString(rgShape) + " must be >= 0.");
                    }

                    if (m_nCount != 0)
                    {
//                        m_log.CHECK_LE(rgShape[i], int.MaxValue / m_nCount, "The blob size exceeds int.MaxValue!");

                        if (rgShape[i] > int.MaxValue / m_nCount)
                        {
                            string strBlobName = (!string.IsNullOrEmpty(m_strName)) ? "Blob '" + m_strName + "': " : "";
                            m_log.FAIL(strBlobName + "The blob size at item " + i.ToString() + " of shape " + toString(rgShape) + " exceeds the maximum of " + (int.MaxValue / m_nCount).ToString() + "!");
                        }
                    }

                    m_nCount *= rgShape[i];
                    m_rgShape.Add(rgShape[i]);

                    int nShape = (int)(double)Convert.ChangeType(rgShapeData[i], typeof(double));

                    if (nShape != rgShape[i])
                    {
                        rgShapeData[i] = (T)Convert.ChangeType(rgShape[i], typeof(T));
                        bDirty = true;
                    }
                }

                if (bDirty)
                {
                    m_shape.mutable_cpu_data = rgShapeData;
                    m_shape.update_cpu_data();
                }
            }
        }

        /// <summary>
        /// Change the dimensions of the blob, allocating new memory if necessary.
        /// </summary>
        /// <remarks>
        /// This function can be called both to create an initial allocation
        /// of memory, and to adjust the dimensions of a top blob during Layer::Reshape
        /// or Layer::Forward.  When changing the size of blob, memory will only be
        /// reallocated if sufficient memory does not already exist, and excess memory
        /// will not be freed until Dispose is called.
        /// 
        /// Note that reshaping an input blob and immediately calling Net::Backward is
        /// an error;  either Net::Forward or Net::Reshape need to be called to 
        /// propagate the new input shape to higher layers.
        /// </remarks>
        /// <param name="rgShape">Specifies the new shape.</param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half sized memory.</param>
        public void Reshape(List<int> rgShape, bool? bUseHalfSize = null)
        {
            Reshape(rgShape.ToArray(), bUseHalfSize);
        }

        /// <summary>
        /// Change the dimensions of the blob, allocating new memory if necessary.
        /// </summary>
        /// <remarks>
        /// This function can be called both to create an initial allocation
        /// of memory, and to adjust the dimensions of a top blob during Layer::Reshape
        /// or Layer::Forward.  When changing the size of blob, memory will only be
        /// reallocated if sufficient memory does not already exist, and excess memory
        /// will not be freed until Dispose is called.
        /// 
        /// Note that reshaping an input blob and immediately calling Net::Backward is
        /// an error;  either Net::Forward or Net::Reshape need to be called to 
        /// propagate the new input shape to higher layers.
        /// </remarks>
        /// <param name="rgShape">Specifies the new shape.</param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half sized memory.</param>
        public void Reshape(int[] rgShape, bool? bUseHalfSize = null)
        {
            reshapeShape(rgShape);

            if (m_nCount > m_nCapacity || (m_data != null && m_nCount > m_data.Capacity) || (m_diff != null && m_nCount > m_diff.Capacity) || (m_data != null && bUseHalfSize.HasValue && m_data.HalfSize != bUseHalfSize.Value))
            {
                if (m_data != null)
                    m_data.Dispose();

                if (m_diff != null)
                    m_diff.Dispose();

                m_nCapacity = m_nCount;

                if (m_data == null)
                    m_data = new SyncedMemory<T>(m_cuda, m_log, m_nCapacity, null, bUseHalfSize.GetValueOrDefault(false));
                else
                    m_data.Allocate(m_nCapacity, bUseHalfSize.GetValueOrDefault(m_data.HalfSize));

                if (m_bIncludeDiff)
                {
                    if (m_diff == null)
                        m_diff = new SyncedMemory<T>(m_cuda, m_log, m_nCapacity, null, bUseHalfSize.GetValueOrDefault(false));
                    else
                        m_diff.Allocate(m_nCapacity, bUseHalfSize.GetValueOrDefault(m_data.HalfSize));
                }
            }

            if (m_data.Count != m_nCount)
                m_data.Count = m_nCount;

            if (m_bIncludeDiff)
            {
                if (m_diff.Count != m_nCount)
                    m_diff.Count = m_nCount;
            }
        }

        /// <summary>
        /// Change the dimensions of the blob, allocating new memory if necessary.
        /// </summary>
        /// <remarks>
        /// This function can be called both to create an initial allocation
        /// of memory, and to adjust the dimensions of a top blob during Layer::Reshape
        /// or Layer::Forward.  When changing the size of blob, memory will only be
        /// reallocated if sufficient memory does not already exist, and excess memory
        /// will not be freed until Dispose is called.
        /// 
        /// Note that reshaping an input blob and immediately calling Net::Backward is
        /// an error;  either Net::Forward or Net::Reshape need to be called to 
        /// propagate the new input shape to higher layers.
        /// </remarks>
        /// <param name="shape">Specifies the new shape of the Blob.</param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half sized memory.</param>
        public void Reshape(BlobShape shape, bool? bUseHalfSize = null)
        {
            m_log.CHECK_LE(shape.dim.Count, MAX_BLOB_AXES, "The shape dimension must be less than " + MAX_BLOB_AXES.ToString());
            Reshape(shape.dim, bUseHalfSize);
        }

        /// <summary>
        /// Reshape this Blob to have the same shape as another Blob.
        /// </summary>
        /// <param name="b">Specifies the other Blob.</param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half sized memory.</param>
        public void ReshapeLike(Blob<T> b, bool? bUseHalfSize = null)
        {
            Reshape(b.shape(), bUseHalfSize);
        }

        /// <summary>
        /// Returns a string describing the Blob's shape.
        /// </summary>
        public string shape_string
        {
            get
            {
                string strOut = "";

                for (int i = 0; i < m_rgShape.Count; i++)
                {
                    strOut += m_rgShape[i].ToString() + " ";
                }

                strOut += "(" + m_rgShape.Count.ToString() + ")";
                return strOut;
            }
        }

        /// <summary>
        /// Returns whether or not the Diff portion exists.
        /// </summary>
        public bool DiffExists
        {
            get { return m_bIncludeDiff; }
        }

        /// <summary>
        /// Returns an array where each element contains the shape of an axis of the Blob.
        /// </summary>
        /// <returns>The shape array is returned.</returns>
        public List<int> shape()
        {
            return m_rgShape;
        }

        /// <summary>
        /// Returns the dimension of the nIdx'th axis (or the negative nIdx'th
        /// axis from teh end, if nIdx is negative.
        /// </summary>
        /// <param name="nIdx">The axis index, which may be negative as it will be
        /// 'canonicalized' using CanonicalAxisIndex.</param>
        /// <returns></returns>
        public int shape(int nIdx)
        {
            return m_rgShape[CanonicalAxisIndex(nIdx)];
        }

        /// <summary>
        /// Returns the number of axes in the Blob.
        /// </summary>
        public int num_axes
        {
            get
            {
                return m_rgShape.Count;
            }
        }

        /// <summary>
        /// Returns the number of true axes, ignoring the trailing ones.
        /// </summary>
        public int num_true_axes
        {
            get
            {
                int nCount = 0;
                bool bCount = false;

                for (int i = m_rgShape.Count - 1; i >= 0; i--)
                {
                    if (bCount || m_rgShape[i] != 1)
                    {
                        nCount++;
                        bCount = true;
                    }
                }

                return nCount;
            }
        }

        /// <summary>
        /// Returns the total number of items in the Blob.
        /// </summary>
        /// <returns></returns>
        public int count()
        {
            return m_nCount;
        }

        /// <summary>
        /// Compute the volume of a slice; i.e., the product of dimensions
        /// among a range of axes.
        /// </summary>
        /// <param name="nStartIdx">The first axis to include in the slice.</param>
        /// <param name="nEndIdx">The first axis to exclude from the slice.</param>
        /// <returns></returns>
        public int count(int nStartIdx, int nEndIdx)
        {
            m_log.CHECK_LE(nStartIdx, nEndIdx, "The start idx must be <= the end idx.");
            m_log.CHECK_GE(nStartIdx, 0, "The start idx must be >= 0.");
            m_log.CHECK_GE(nEndIdx, 0, "The end idx must be >= 0.");
            m_log.CHECK_LE(nStartIdx, num_axes, "The start axis must be <= the number of axes.");
            m_log.CHECK_LE(nEndIdx, num_axes, "The end axis must be <= the number of axes.");

            return Utility.Count(shape(), nStartIdx, nEndIdx);
        }

        /// <summary>
        /// Compute the volume of a slice spanning from a particular first axis to the final
        /// axis.
        /// </summary>
        /// <param name="nStartIdx">The first axis to include in the slice.</param>
        /// <returns></returns>
        public int count(int nStartIdx)
        {
            return count(nStartIdx, num_axes);
        }

        /// <summary>
        /// Returns the 'canonical' version of a (usually) user-specified axis,
        /// allowing for negative indexing (e.g., -1 for the last axis).
        /// </summary>
        /// <param name="nIdx">The axis index. 
        /// </param>
        /// <returns>The zero based index is returned.</returns>
        public int CanonicalAxisIndex(int nIdx)
        {
            m_log.CHECK_GE(nIdx, -num_axes, "The axis " + nIdx.ToString() + " out of range for " + num_axes.ToString() + " -D Blob with shape " + shape_string);
            m_log.CHECK_LT(nIdx, num_axes, "The axis " + nIdx.ToString() + " out of range for " + num_axes.ToString() + " -D Blob with shape " + shape_string);

            return Utility.CanonicalAxisIndex(nIdx, num_axes);
        }

        /// <summary>
        /// <b>DEPRECIATED</b>; legacy shape accessor num: use shape(0) instead.
        /// </summary>
        public int num
        {
            get { return LegacyShape(0); }
        }

        /// <summary>
        /// <b>DEPRECIATED</b>; legacy shape accessor channels: use shape(1) instead.
        /// </summary>
        public int channels
        {
            get { return LegacyShape(1); }
        }

        /// <summary>
        /// <b>DEPRECIATED</b>; legacy shape accessor height: use shape(2) instead.
        /// </summary>
        public int height
        {
            get { return LegacyShape(2); }
        }

        /// <summary>
        /// <b>DEPRECIATED</b>; legacy shape accessor width: use shape(3) instead.
        /// </summary>
        public int width
        {
            get { return LegacyShape(3); }
        }

        /// <summary>
        /// Returns the legacy shape at a given axis.
        /// </summary>
        /// <param name="nIdx">Specifies the axis.</param>
        /// <returns>The shape at a given axis is returned.</returns>
        public int LegacyShape(int nIdx)
        {
            m_log.CHECK_LE(num_axes, 4, "Cannot use legacy accessors on Blobs with > 4 axes.");
            m_log.CHECK_LT(nIdx, 4, "The index must be less than 4.");
            m_log.CHECK_GE(nIdx, -4, "The index must be greater than or equal to -4.");

            if (nIdx >= num_axes || nIdx < -num_axes)
            {
                // Axis is out of range, but still in [0,3] or [-4,-1] for reverse
                // indexing) -- this special case simulates the one-padding used to fill
                // extraneous axes of legacy blobs.
                return 1;
            }

            return shape(nIdx);
        }

        /// <summary>
        /// Returns the flat offset given the number, channel, height and width.
        /// </summary>
        /// <param name="n">Specifies the num.</param>
        /// <param name="c">Specifies the channel.</param>
        /// <param name="h">Specifies the height.</param>
        /// <param name="w">Specifies the width.</param>
        /// <returns>The flat offset is returned.</returns>
        public int offset(int n, int c = 0, int h = 0, int w = 0)
        {
            int c1 = channels;
            int h1 = height;
            int w1 = width;

            m_log.CHECK_GE(n, 0, "n must be >= 0.");
            m_log.CHECK_LE(n, num, "n must be <= num.");
            m_log.CHECK_GE(c1, 0, "channels must be >= 0.");
            m_log.CHECK_LE(c, c1, "c must be <= channels.");
            m_log.CHECK_GE(h1, 0, "height must be >= 0.");
            m_log.CHECK_LE(h, h1, "w must be <= height.");
            m_log.CHECK_GE(w1, 0, "width must be >= 0.");
            m_log.CHECK_LE(w, w1, "w must be <= width.");

            return ((n * c1 + c) * h1 + h) * w1 + w;
        }

        /// <summary>
        /// Returns the flat offset given the array of axes values.
        /// </summary>
        /// <param name="rgIdx">Specifies the array of axes values.</param>
        /// <returns>The flat offset is returned.</returns>
        public int offset(List<int> rgIdx)
        {
            m_log.CHECK_LE(rgIdx.Count, num_axes, "The index array must have an item count <= num_axes.");

            int nOffset = 0;

            for (int i = 0; i < num_axes; i++)
            {
                nOffset *= shape(i);

                if (rgIdx.Count > i)
                {
                    m_log.CHECK_GE(rgIdx[i], 0, "The index at " + i.ToString() + " must be >= 0.");
                    m_log.CHECK_LT(rgIdx[i], shape(i), "The index at " + i.ToString() + " must be <= the shape at " + i.ToString());
                    nOffset += rgIdx[i];
                }
            }

            return nOffset;
        }

        /// <summary>
        /// Copy from a source Blob.
        /// </summary>
        /// <param name="src">The Blob to copy from.</param>
        /// <param name="nSrcOffset">The offset into the source data to copy.</param>
        /// <param name="nDstOffset">The offset into the destination data to copy.</param>
        /// <param name="nCount">The number of items to copy.</param>
        /// <param name="bCopyData">Copy the data.</param>
        /// <param name="bCopyDiff">Copy the diff.</param>
        public void CopyFrom(Blob<T> src, int nSrcOffset, int nDstOffset, int nCount, bool bCopyData, bool bCopyDiff)
        {
            m_log.CHECK_GE(count(), nDstOffset + nCount, "The data to be copied is larger that the destination blob.");
            m_log.CHECK_GE(src.count(), nSrcOffset + nCount, "The data to be copied is larger than the source blob.");

            if (bCopyData)
                m_cuda.copy(nCount, src.gpu_data, mutable_gpu_data, nSrcOffset, nDstOffset);

            if (bCopyDiff)
                m_cuda.copy(nCount, src.gpu_diff, mutable_gpu_diff, nSrcOffset, nDstOffset);
        }

        /// <summary>
        /// Copy from a source Blob.
        /// </summary>
        /// <param name="src">The Blob to copy from.</param>
        /// <param name="bCopyDiff">If false, copy the data; if true, copy the diff.</param>
        /// <param name="bReshape">If false, require this Blob to be pre-shaped to the shape
        /// of other (and die otherwise); If true, Reshape this Blob to other's shape if
        /// necessary.</param>
        /// <param name="hDstHostBuffer">Optionally, specifies the host buffer of the destination.</param>
        /// <param name="bIgnoreShape">Optionally, specifies to ignore the shape and just make sure the count is the same before copying (default = false).</param>
        /// <returns>
        /// When used, the host buffer handle is returned.
        /// </returns>
        public long CopyFrom(Blob<T> src, bool bCopyDiff = false, bool bReshape = false, long hDstHostBuffer = 0, bool bIgnoreShape = false)
        {
            // Ignore copy if the data points to the same handle.
            if (bCopyDiff)
            {
                if (src.gpu_diff == gpu_diff)
                    return 0;
            }
            else
            {
                if (src.gpu_data == gpu_data)
                    return 0;
            }

            if (src.count() != m_nCount || (!bIgnoreShape && !CompareShape(src.m_rgShape)))
            {
                if (bReshape)
                    ReshapeLike(src);
                else
                    m_log.FAIL("Trying to copy blobs of different sizes!");
            }

            if (bCopyDiff)
            {
                if (m_diff == null)
                    return hDstHostBuffer;

                return m_diff.Copy(src.diff, hDstHostBuffer);
            }
            else
            {
                if (m_data == null)
                    return hDstHostBuffer;

                return m_data.Copy(src.data, hDstHostBuffer);
            }
        }

        /// <summary>
        /// Copy the source data to this Blob, and if this blob is larger than the source,
        /// pad this blob with 'dfPad' values to the right.
        /// </summary>
        /// <param name="src">Specifies the source Blob to copy.</param>
        /// <param name="dfPad">Specifies the pad value (default = 0).</param>
        /// <param name="bCopyDiff">Specifies to copy the diff values (default = false).</param>
        public void CopyFromAndPad(Blob<T> src, double dfPad = 0, bool bCopyDiff = false)
        {
            if (count() == src.count())
            {
                CopyFrom(src, bCopyDiff);
                return;
            }

            if (count() < src.count())
                m_log.FAIL("The destination blob must be larger than the source blob.");

            if (bCopyDiff)
            {
                SetDiff(dfPad);
                m_cuda.copy(src.count(), src.gpu_diff, mutable_gpu_diff);
            }
            else
            {
                SetData(dfPad);
                m_cuda.copy(src.count(), src.gpu_data, mutable_gpu_data);
            }
        }

        /// <summary>
        /// Copy from a source Blob and transpose the height and width of the copy.
        /// </summary>
        /// <param name="blobSrc">Specifies the Blob to copy from.</param>
        /// <param name="bCopyDiff">Optionally, specifies to copy and transform the diff instead of the data (default = false).</param>
        /// <param name="bUseCuda">Optionally, specifies to use CUDA function for transformations (default = true)</param>
        public void CopyFromAndTransposeHeightWidth(Blob<T> blobSrc, bool bCopyDiff = false, bool bUseCuda = true)
        {
            m_log.CHECK_EQ(blobSrc.num_axes, 4, "Currently, Blobs only support transposing 4 axis tensors.");

            Reshape(blobSrc.num, blobSrc.channels, blobSrc.width, blobSrc.height);

            SyncedMemory<T> dst = (bCopyDiff) ? m_diff : m_data;
            SyncedMemory<T> src = (bCopyDiff) ? blobSrc.m_diff : blobSrc.m_data;

            int nN = num;
            int nC = channels;
            int nH = height;
            int nW = width;

            if (bUseCuda)
            {
                m_cuda.transposeHW(nN, nC, nH, nW, src.gpu_data, dst.mutable_gpu_data);
            }
            else
            {
                T[] rgSrc = src.update_cpu_data();
                T[] rgDst = dst.mutable_cpu_data;

                for (int n = 0; n < nN; n++)
                {
                    for (int c = 0; c < nC; c++)
                    {
                        int nOffset = (n * nC * nH * nW) + (c * nH * nW);

                        for (int h = 0; h < nH; h++)
                        {
                            for (int w = 0; w < nW; w++)
                            {
                                int nSrcIdx = nOffset + (h * nW) + w;
                                int nDstIdx = nOffset + (w + nH) + h;
                                rgDst[nDstIdx] = rgSrc[nSrcIdx];
                            }
                        }
                    }
                }

                dst.mutable_cpu_data = rgDst;
            }
        }

        /// <summary>
        /// Copy all data along a given channel from the source.
        /// </summary>
        /// <param name="blobSrc">Specifies the source blob to copy from.</param>
        /// <param name="nChannelFrom">Specifies the channel within the source to copy from.</param>
        /// <param name="nChannelTo">Specifies the channel in the destination (this blob) to copy into.</param>
        /// <param name="bCopyDiff">Specifies whether or not to copy the data or diff.</param>
        public void CopyFrom(Blob<T> blobSrc, int nChannelFrom, int nChannelTo, bool bCopyDiff = false)
        {
            m_log.CHECK_EQ(blobSrc.num, num, "The source and destination blobs must have the same num.");
            m_log.CHECK_EQ(blobSrc.height, height, "The source and destination blobs must have the same height.");
            m_log.CHECK_EQ(blobSrc.width, width, "The source and destination blobs must have the same width.");
            m_log.CHECK_LT(nChannelFrom, blobSrc.channels, "The channel form parameter is out of range!");
            m_log.CHECK_LT(nChannelTo, channels, "The channel to parameter is out of range!");

            SyncedMemory<T> dst = (bCopyDiff) ? m_diff : m_data;
            SyncedMemory<T> src = (bCopyDiff) ? blobSrc.m_diff : blobSrc.m_data;

            int nCsrc = blobSrc.channels;
            int nCdst = channels;
            int nDim = height * width;
            int nSrcOffset = nChannelFrom * nDim;
            int nDstOffset = nChannelTo * nDim;
            int nSrcStep = (nCsrc * nDim);
            int nDstStep = (nCdst * nDim);

            for (int n = 0; n < num; n++)
            {
                m_cuda.copy(nDim, src.gpu_data, dst.mutable_gpu_data, nSrcOffset, nDstOffset);
                nSrcOffset += nSrcStep;
                nDstOffset += nDstStep;
            }
        }

        /// <summary>
        /// Compare the data (or diff) of one blob to another and return true if all items fall within the specified tolerance or not.
        /// </summary>
        /// <param name="other">Specifies the other blob to compare.</param>
        /// <param name="work">Specifies a temporary work blob.</param>
        /// <param name="bDiff">Specifies to compare the diff.</param>
        /// <param name="dfTol">Specifies the accepted tolerance.</param>
        /// <param name="bZeroCheck">Optionally, specifies to check for all zeros (default = false).</param>
        /// <returns>If all data (or diff) values fall within the tolerance, true is returned, otherwise false.</returns>
        public bool Compare(Blob<T> other, Blob<T> work, bool bDiff = false, double dfTol = 1e-8, bool bZeroCheck = false)
        {
            int nCount = count();
            if (nCount != other.count())
                return false;

            if (Cuda.KernelHandle != other.Cuda.KernelHandle)
                throw new Exception("The compare blob has a different Cuda Kernel Handles!");

            if (Cuda.KernelHandle != work.Cuda.KernelHandle)
                throw new Exception("The work blob has a different Cuda Kernel Handle!");
            
            work.ReshapeLike(this);

            long h1 = (bDiff) ? gpu_diff : gpu_data;
            long h2 = (bDiff) ? other.gpu_diff : other.gpu_data;
            long lPos;

            m_cuda.sub(nCount, h1, h2, work.mutable_gpu_data);
            double dfMin = m_cuda.min(nCount, work.gpu_data, out lPos, 0, work.mutable_gpu_diff);
            if (Math.Abs(dfMin) > dfTol)
                return false;
            
            double dfMax = m_cuda.max(nCount, work.gpu_data, out lPos, 0, work.mutable_gpu_diff);
            if (dfMax > dfTol)
                return false;

            if (bZeroCheck)
            {
                double dfZero = m_cuda.asum_double(nCount, h2);
                if (dfZero == 0)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Compare the data (or diff) of one blob to another and return true if all items fall within the specified tolerance or not.
        /// </summary>
        /// <param name="cuda">Specifies a double precision CudaDnn.</param>
        /// <param name="other">Specifies the other blob to compare.</param>
        /// <param name="work">Specifies a temporary work blob.</param>
        /// <param name="bDiff">Specifies to compare the diff.</param>
        /// <param name="dfTol">Specifies the accepted tolerance.</param>
        /// <returns>If all data (or diff) values fall within the tolerance, true is returned, otherwise false.</returns>
        public bool Compare(CudaDnn<double> cuda, Blob<T> other, Blob<double> work, bool bDiff = false, double dfTol = 1e-8)
        {
            if (count() != other.count())
                return false;

            work.Reshape(num, channels, height, width);
            work.mutable_cpu_data = (bDiff) ? Utility.ConvertVec<T>(mutable_cpu_diff) : Utility.ConvertVec<T>(mutable_cpu_data);
            work.mutable_cpu_diff = (bDiff) ? Utility.ConvertVec<T>(other.mutable_cpu_diff) : Utility.ConvertVec<T>(other.mutable_cpu_data);

            cuda.sub(count(), work.gpu_data, work.gpu_diff, work.mutable_gpu_data);
            double dfMin = work.min_data;
            if (Math.Abs(dfMin) > dfTol)
                return false;

            double dfMax = work.max_data;
            if (dfMax > dfTol)
                return false;

            return true;
        }

#pragma warning disable 1591

        public void KeepBestResultsByChannel(int nNumberToKeep) /** @private */
        {
            m_log.CHECK_EQ(num_axes, 4, "Currently KeepBestResutls only works on 4-axis blobs.");

            T[] rgData = mutable_cpu_data;
            int nN = num;
            int nC = channels;
            int nH = height;
            int nW = width;
            List<List<KeyValuePair<int, float>>> rgrgKeyValues = new List<List<KeyValuePair<int, float>>>();
            List<List<List<List<float>>>> rgrgrgrgData = new List<List<List<List<float>>>>();

            for (int n = 0; n < nN; n++)
            {
                List<KeyValuePair<int, float>> rgKeyValues = new List<KeyValuePair<int, float>>();
                List<List<List<float>>> rgrgrgData = new List<List<List<float>>>();

                for (int c = 0; c < nC; c++)
                {
                    float fSum = 0;
                    List<List<float>> rgrgData = new List<List<float>>();

                    for (int h = 0; h < nH; h++)
                    {
                        List<float> rgVal = new List<float>();

                        for (int w = 0; w < nW; w++)
                        {
                            int nIdx = (n * nC * nH * nW) + (c * nH * nW) + (h * nW) + w;
                            float fVal = (float)Convert.ChangeType(rgData[nIdx], typeof(float));
                            rgVal.Add(fVal);
                            fSum += fVal;
                        }

                        rgrgData.Add(rgVal);
                    }

                    rgKeyValues.Add(new KeyValuePair<int, float>(c, fSum));
                    rgrgrgData.Add(rgrgData);
                }

                rgKeyValues.Sort(new Comparison<KeyValuePair<int, float>>(sort));

                rgrgKeyValues.Add(rgKeyValues);
                rgrgrgrgData.Add(rgrgrgData);
            }

            SetData(0);
            rgData = mutable_cpu_data;

            for (int n = 0; n < nN; n++)
            {
                List<KeyValuePair<int, float>> rgKeyValues = rgrgKeyValues[n];
                List<List<List<float>>> rgrgrgData = rgrgrgrgData[n];
                List<int> rgIdx = new List<int>();

                for (int i = 0; i < nNumberToKeep && i < rgKeyValues.Count; i++)
                {
                    rgIdx.Add(rgKeyValues[i].Key);
                }

                foreach (int c in rgIdx)
                {
                    List<List<float>> rgrgData = rgrgrgData[c];

                    for (int h = 0; h < nH; h++)
                    {
                        List<float> rgVal = rgrgData[h];

                        for (int w = 0; w < nW; w++)
                        {
                            int nIdx = (n * nC * nH * nW) + (c * nH * nW) + (h * nW) + w;
                            rgData[nIdx] = (T)Convert.ChangeType(rgVal[w], typeof(T));
                        }
                    }
                }
            }

            mutable_cpu_data = rgData;
        }

        public void KeepBestResultsByWeight(int nNumberToKeep) /** @private */
        {
            m_log.CHECK_EQ(num_axes, 4, "Currently KeepBestResutls only works on 4-axis blobs.");

            T[] rgData = mutable_cpu_data;
            int nN = num;
            int nC = channels;
            int nH = height;
            int nW = width;
            List<List<KeyValuePair<int, float>>> rgrgKeyValues = new List<List<KeyValuePair<int, float>>>();

            for (int n = 0; n < nN; n++)
            {
                List<KeyValuePair<int, float>> rgKeyValues = new List<KeyValuePair<int, float>>();

                for (int c = 0; c < nC; c++)
                {
                    for (int h = 0; h < nH; h++)
                    {
                        for (int w = 0; w < nW; w++)
                        {
                            int nIdx = (n * nC * nH * nW) + (c * nH * nW) + (h * nW) + w;
                            rgKeyValues.Add(new KeyValuePair<int, float>(nIdx, (float)Convert.ChangeType(rgData[nIdx], typeof(float))));
                        }
                    }
                }

                rgKeyValues.Sort(new Comparison<KeyValuePair<int, float>>(sort));
                rgrgKeyValues.Add(rgKeyValues);
            }

            SetData(0);
            rgData = mutable_cpu_data;

            for (int n = 0; n < nN; n++)
            {
                List<KeyValuePair<int, float>> rgKeyValues = rgrgKeyValues[n];

                for (int i = 0; i < nNumberToKeep && i < rgKeyValues.Count; i++)
                {
                    KeyValuePair<int, float> kv = rgKeyValues[i];
                    rgData[kv.Key] = (T)Convert.ChangeType(kv.Value, typeof(T));
                }
            }

            mutable_cpu_data = rgData;
        }

#pragma warning restore 1591


        private int sort(KeyValuePair<int, float> a, KeyValuePair<int, float> b)
        {
            if (a.Value < b.Value)
                return 1;

            if (a.Value > b.Value)
                return -1;

            return 0;
        }

        /// <summary>
        /// Returns the data at a given location in the Blob.
        /// </summary>
        /// <param name="n">Specifies the num.</param>
        /// <param name="c">Specifies the channel.</param>
        /// <param name="h">Specifies the height.</param>
        /// <param name="w">Specifies the width.</param>
        /// <returns>The data item at the location is returned.</returns>
        public T data_at(int n, int c, int h, int w)
        {
            return m_data.GetAt(offset(n, c, h, w));
        }

        /// <summary>
        /// Returns the diff at a given location in the Blob.
        /// </summary>
        /// <param name="n">Specifies the num.</param>
        /// <param name="c">Specifies the channel.</param>
        /// <param name="h">Specifies the height.</param>
        /// <param name="w">Specifies the width.</param>
        /// <returns>The diff item at the location is returned.</returns>
        public T diff_at(int n, int c, int h, int w)
        {
            return m_diff.GetAt(offset(n, c, h, w));
        }

        /// <summary>
        /// Returns the data at a given location in the Blob.
        /// </summary>
        /// <param name="rgIdx">Specifies an array of axes.</param>
        /// <returns>The data item at the location is returned.</returns>
        public T data_at(List<int> rgIdx)
        {
            return m_data.GetAt(offset(rgIdx));
        }

        /// <summary>
        /// Returns the diff at a given location in the Blob.
        /// </summary>
        /// <param name="rgIdx">Specifies an array of axes.</param>
        /// <returns>The diff item at the location is returned.</returns>
        public T diff_at(List<int> rgIdx)
        {
            return m_diff.GetAt(offset(rgIdx));
        }

        /// <summary>
        /// Returns the SyncedMemory that stores the data.
        /// </summary>
        public SyncedMemory<T> data
        {
            get { return m_data; }
        }

        /// <summary>
        /// Returns the SyncedMemory that stores the diff.
        /// </summary>
        public SyncedMemory<T> diff
        {
            get { return m_diff; }
        }

        /// <summary>
        /// Returns the last host data retrieved from the GPU.
        /// </summary>
        public T[] cpu_data
        {
            get { return m_data.cpu_data; }
        }

        /// <summary>
        /// Get data from the GPU and bring it over to the host, or
        /// Set data from the Host and send it over to the GPU.
        /// </summary>
        public T[] mutable_cpu_data
        {
            get { return m_data.mutable_cpu_data; }
            set { m_data.mutable_cpu_data = value; }
        }

        /// <summary>
        /// Update the CPU data by transferring the GPU data over to the Host.
        /// </summary>
        /// <returns>The Host data is returned.</returns>
        public T[] update_cpu_data()
        {
            return m_data.update_cpu_data(m_nCount);
        }

        /// <summary>
        /// Returns the data GPU handle used by the CudaDnn connection.
        /// </summary>
        public long gpu_data
        {
            get { return m_data.gpu_data; }
        }

        /// <summary>
        /// Returns the data GPU handle used by the CudaDnn connection.
        /// </summary>
        public long mutable_gpu_data
        {
            get { return m_data.mutable_gpu_data; }
//            set { m_data.mutable_gpu_data = value; }
        }

        /// <summary>
        /// Returns the last host diff retrieved from the GPU.
        /// </summary>
        public T[] cpu_diff
        {
            get 
            {
                if (m_diff == null)
                    return null;

                return m_diff.cpu_data; 
            }
        }

        /// <summary>
        /// Get diff from the GPU and bring it over to the host, or
        /// Set diff from the Host and send it over to the GPU.
        /// </summary>
        public T[] mutable_cpu_diff
        {
            get 
            {
                if (m_diff == null)
                    return null;

                return m_diff.mutable_cpu_data; 
            }
            set 
            { 
                m_diff.mutable_cpu_data = value; 
            }
        }

        /// <summary>
        /// Update the CPU diff by transferring the GPU diff over to the Host.
        /// </summary>
        /// <returns>The Host diff is returned.</returns>
        public T[] update_cpu_diff()
        {
            if (m_diff == null)
                return null;

            return m_diff.update_cpu_data(m_nCount);
        }

        /// <summary>
        /// Returns the diff GPU handle used by the CudaDnn connection.
        /// </summary>
        public long gpu_diff
        {
            get 
            {
                if (m_diff == null)
                    return 0;

                return m_diff.gpu_data; 
            }
        }

        /// <summary>
        /// Returns the diff GPU handle used by the CudaDnn connection.
        /// </summary>
        public long mutable_gpu_diff
        {
            get { return m_diff.mutable_gpu_data; }
//            set { m_diff.mutable_gpu_data = value; }
        }

        /// <summary>
        /// Returns the shape GPU handle used by the CudaDnn connection.  The shape data contains 
        /// the shape information of the Blob for use in GPU operations.
        /// </summary>
        public long gpu_shape
        {
            get { return m_shape.gpu_data; }
        }

        /// <summary>
        /// The 'update' method is used for parameter blobs in a Net.
        /// </summary>
        /// <remarks>
        /// Update is called to apply the diff errors to the data.  When !bIncludeDiff or freeze_learning = <i>true</i>, no diff is applied.
        /// </remarks>
        public void Update()
        {
            if (!m_bIncludeDiff || m_bFreezeLearning)
                return;

            // The GPU is assumed to be the owner of the data.
            m_cuda.axpy(m_nCount, m_tMinusOne, m_diff.gpu_data, m_data.mutable_gpu_data);
        }

        /// <summary>
        /// Create a new Blob from a given BlobProto.
        /// </summary>
        /// <param name="bp">Specifies the BlobProto to use.</param>
        /// <param name="bReshape">Specifies whether or not to reshape the Blob with the BlobProto shape.</param>
        public void FromProto(BlobProto bp, bool bReshape = true)
        {
            if (bReshape)
            {
                List<int> rgShape = new List<int>();

                if (bp.num.HasValue || bp.channels.HasValue || bp.height.HasValue || bp.width.HasValue)
                {
                    // Using depreciated 4D Blob dimensions -- 
                    // shape is (num, channels, height, width).
                    if (bp.num.HasValue)
                        rgShape.Add(bp.num.Value);

                    if (bp.channels.HasValue)
                        rgShape.Add(bp.channels.Value);

                    if (bp.height.HasValue)
                        rgShape.Add(bp.height.Value);

                    if (bp.width.HasValue)
                        rgShape.Add(bp.width.Value);
                }
                else
                {
                    rgShape = Utility.Clone<int>(bp.shape.dim);
                }

                Reshape(rgShape);
            }
            else
            {
                m_log.CHECK(ShapeEquals(bp), "Shape mismatch (reshape not set)!");
            }

            // Copy the data.
            T[] rgData = null;

            if (bp.double_data.Count > 0)
            {
                m_log.CHECK_EQ(m_nCount, bp.double_data.Count, "The double data count is not the same as the blob data count!");
                rgData = Utility.ConvertVec<T>(bp.double_data.ToArray());
            }
            else if (bp.data.Count > 0)
            {
                m_log.CHECK_EQ(m_nCount, bp.data.Count, "The double data count is not the same as the blob data count!");
                rgData = Utility.ConvertVec<T>(bp.data.ToArray());
            }

            if (rgData != null)
                mutable_cpu_data = rgData;

            // Copy the diff.
            T[] rgDiff = null;

            if (bp.double_diff.Count > 0)
            {
                m_log.CHECK_EQ(m_nCount, bp.double_diff.Count, "The double diff count is not the same as the blob data count!");
                rgDiff = Utility.ConvertVec<T>(bp.double_diff.ToArray());
            }
            else if (bp.diff.Count > 0)
            {
                m_log.CHECK_EQ(m_nCount, bp.diff.Count, "The double data count is not the same as the blob data count!");
                rgDiff = Utility.ConvertVec<T>(bp.diff.ToArray());
            }

            if (rgDiff != null)
                mutable_cpu_diff = rgDiff;
        }

        /// <summary>
        /// Writes the Blob to a new BlobProto.
        /// </summary>
        /// <param name="bWriteDiff">When <i>true</i>, the diff is written to the BlobProto, otherwise the data is written.</param>
        /// <returns>The new BlobProto is returned.</returns>
        public BlobProto ToProto(bool bWriteDiff = false)
        {
            BlobProto bp = new BlobProto(m_rgShape);

            T[] rgData = (bWriteDiff) ? null : update_cpu_data();
            T[] rgDiff = (bWriteDiff) ? update_cpu_diff() : null;

            if (typeof(T) == typeof(double))
            {
                if (rgData != null)
                {
                    double[] rgDataD = Utility.ConvertVec<T>(rgData);
                    bp.double_data = new List<double>(rgDataD);
                }

                if (rgDiff != null)
                {
                    double[] rgDiffD = Utility.ConvertVec<T>(rgData);
                    bp.double_diff = new List<double>(rgDiffD);
                }
            }
            else
            {
                if (rgData != null)
                {
                    float[] rgDataF = Utility.ConvertVecF<T>(rgData);
                    bp.data = new List<float>(rgDataF);
                }

                if (rgDiff != null)
                {
                    float[] rgDiffF = Utility.ConvertVecF<T>(rgDiff);
                    bp.diff = new List<float>(rgDiffF);
                }
            }

            return bp;
        }

        /// <summary>
        /// Compute the sum of absolute values (L1 norm) of the data.
        /// </summary>
        /// <returns></returns>
        public T asum_data()
        {
            if (m_nCount == 0 || gpu_data == 0)
                return m_tZero;

            return m_cuda.asum(m_nCount, gpu_data);
        }

        /// <summary>
        /// Compute the sum of absolute values (L1 norm) of the diff.
        /// </summary>
        /// <returns></returns>
        public T asum_diff()
        {
            if (m_nCount == 0 || gpu_diff == 0)
                return m_tZero;

            return m_cuda.asum(m_nCount, gpu_diff);
        }

        /// <summary>
        /// Calcualte the sum of squares (L2 norm squared) of the data.
        /// </summary>
        /// <returns></returns>
        public T sumsq_data()
        {
            if (m_nCount == 0 || gpu_data == 0)
                return m_tZero;

            return m_cuda.dot(m_nCount, gpu_data, gpu_data);
        }

        /// <summary>
        /// Calcualte the sum of squares (L2 norm squared) of the diff.
        /// </summary>
        /// <returns></returns>
        public T sumsq_diff()
        {
            if (m_nCount == 0 || gpu_diff == 0)
                return m_tZero;

            return m_cuda.dot(m_nCount, gpu_diff, gpu_diff);
        }

        /// <summary>
        /// Scale the data by a scaling factor.
        /// </summary>
        /// <param name="df">Specifies the scaling factor.</param>
        public void scale_data(double df)
        {
            scale_data((T)Convert.ChangeType(df, typeof(T)));
        }

        /// <summary>
        /// Scale the diff by a scaling factor.
        /// </summary>
        /// <param name="df">Specifies the scaling factor.</param>
        public void scale_diff(double df)
        {
            scale_diff((T)Convert.ChangeType(df, typeof(T)));
        }

        /// <summary>
        /// Scale the data in the blob to the range [dfMin,dfMax].
        /// </summary>
        /// <param name="dfMin">Specifies the minimum of the range.</param>
        /// <param name="dfMax">Specifies the maximum of the range.</param>
        public void scale_to_range(double dfMin, double dfMax)
        {
            m_cuda.scale_to_range(m_nCount, gpu_data, mutable_gpu_data, dfMin, dfMax);
        }

        /// <summary>
        /// Scale the blob data by a constant factor.
        /// </summary>
        /// <param name="fScaleFactor">Specifies the scaling factor.</param>
        public void scale_data(T fScaleFactor)
        {
            m_cuda.scal(m_nCount, fScaleFactor, mutable_gpu_data);
        }

        /// <summary>
        /// Scale the blob diff by a constant factor.
        /// </summary>
        /// <param name="fScaleFactor">Specifies the scaling factor.</param>
        public void scale_diff(T fScaleFactor)
        {
            m_cuda.scal(m_nCount, fScaleFactor, mutable_gpu_diff);
        }

        /// <summary>
        /// When true, this Blob is reshaped to the source when sharing the source data (default = false).
        /// </summary>
        /// <remarks>
        /// This setting is used by the Net when sharing trained weights.
        /// </remarks>
        public bool reshape_when_sharing
        {
            get { return m_bReshapeWhenSharing; }
            set { m_bReshapeWhenSharing = value; }
        }

        /// <summary>
        /// Set the data to point to the data of the other blob -- useful in Layers which
        /// simply perform a copy in their forward pass.
        /// </summary>
        /// <param name="b"></param>
        public void ShareData(Blob<T> b)
        {
            if (!m_bReshapeWhenSharing)
                m_log.CHECK_EQ(m_nCount, b.count(), "The blob counts are not the same!");
            else 
                reshapeShape(b.shape());

            if (m_bOwnData && m_data != null)
                m_data.Dispose();

            m_data = b.m_data;
            m_bOwnData = false;
        }

        /// <summary>
        /// Set the diff to point to the diff of the other blob -- useful in Layers which
        /// simply perform a copy in their forward pass.
        /// </summary>
        /// <param name="b"></param>
        public void ShareDiff(Blob<T> b)
        {
            if (!m_bReshapeWhenSharing)
                m_log.CHECK_EQ(m_nCount, b.count(), "The blob counts are not the same!");
            else
                reshapeShape(b.shape());

            if (m_bOwnDiff && m_diff != null)
                m_diff.Dispose();

            m_diff = b.m_diff;
            m_bOwnDiff = false;
        }

        /// <summary>
        /// Share another Blob with this one, by setting the data and diff to the same data and diff of the other Blob.
        /// </summary>
        /// <param name="b">Specifies the other Blob to share.</param>
        public void Share(Blob<T> b)
        {
            if (m_bOwnData && m_data != null)
                m_data.Dispose();

            m_data = b.m_data;
            m_bOwnData = false;

            if (m_bOwnDiff && m_diff != null)
                m_diff.Dispose();

            m_diff = b.m_diff;
            m_bOwnDiff = false;

            if (m_bOwnShape && m_shape != null)
                m_shape.Dispose();

            m_shape = b.m_shape;
            m_bOwnShape = false;

            m_nCount = b.m_nCount;
            m_nCapacity = b.m_nCapacity;
            m_rgShape = b.m_rgShape;
            m_nIdx = b.m_nIdx;
        }

        /// <summary>
        /// Get/set the snapshot request.
        /// </summary>
        /// <remarks>
        /// This setting is used by learnable parameters that have requested a snapshot.
        /// </remarks>
        public bool snapshot_requested
        {
            get { return m_bSnapshotRequested; }
            set { m_bSnapshotRequested = value; }
        }

        /// <summary>
        /// Returns the data at a given flat index within the Blob.
        /// </summary>
        /// <param name="nIdx">Specifies the flat index in the range [0,count()-1].</param>
        /// <returns>The data item at the index is returned.</returns>
        public T GetData(int nIdx)
        {
            T[] rg = m_cuda.get(count(), gpu_data, nIdx);
            if (rg.Length == 0)
                throw new Exception("No data at index = " + nIdx.ToString());
            
            return rg[0];
        }
       
        /// <summary>
        /// Returns the diff at a given flat index within the Blob.
        /// </summary>
        /// <param name="nIdx">Specifies the flat index in the range [0,count()-1].</param>
        /// <returns>The diff item at the index is returned.</returns>
        public T GetDiff(int nIdx)
        {
            T[] rg = m_cuda.get(count(), gpu_diff, nIdx);
            if (rg.Length == 0)
                throw new Exception("No data at index = " + nIdx.ToString());

            return rg[0];
        }

        /// <summary>
        /// Sets a number of items within the Blob's data.
        /// </summary>
        /// <param name="rgData">Specifies the data to set.</param>
        /// <param name="nCount">Optionally, specifies a subset count of items to set.</param>
        /// <param name="bSetCount">Optionally, specifies whether or not to set the count.  The count is always set when re-allocating the buffer.</param>
        public void SetData(T[] rgData, int nCount = -1, bool bSetCount = true)
        {
            m_data.SetData(rgData, nCount, bSetCount);
        }

        /// <summary>
        /// Either sets all of the data items in the Blob to a given value, or alternatively only sets a single
        /// indexed item to a given value.
        /// </summary>
        /// <param name="fVal">Specifies the value to set.</param>
        /// <param name="nIdx">Optionally, specifies the index of the item to set.</param>
        public void SetData(T fVal, int nIdx = -1)
        {
            if (mutable_gpu_data == 0)
                return;

            m_cuda.set(count(), mutable_gpu_data, fVal, nIdx);
        }

        /// <summary>
        /// Either sets all of the data items in the Blob to a given value, or alternatively only sets a single
        /// indexed item to a given value.
        /// </summary>
        /// <param name="dfVal">Specifies the value to set.</param>
        /// <param name="nIdx">Optionally, specifies the index of the item to set.</param>
        public void SetData(double dfVal, int nIdx = -1)
        {
            if (mutable_gpu_data == 0)
                return;

            m_cuda.set(count(), mutable_gpu_data, dfVal, nIdx);
        }

        /// <summary>
        /// Set a data range with a given value.
        /// </summary>
        /// <param name="dfVal">Specifies the value to set.</param>
        /// <param name="nStartIdx">Specifies the start index.</param>
        /// <param name="nCount">Specifies the number of items to set.</param>
        public void SetData(double dfVal, int nStartIdx, int nCount)
        {
            T tVal = (T)Convert.ChangeType(dfVal, typeof(T));
            T[] rg = mutable_cpu_data;

            for (int i = 0; i < nCount; i++)
            {
                if (nStartIdx + i < rg.Length)
                    rg[nStartIdx + i] = tVal;
            }

            mutable_cpu_data = rg;
        }

        /// <summary>
        /// Either sets all of the diff items in the Blob to a given value, or alternatively only sets a single
        /// indexed item to a given value.
        /// </summary>
        /// <param name="dfVal">Specifies the value to set.</param>
        /// <param name="nIdx">Optionally, specifies the index of the item to set.</param>
        public void SetDiff(double dfVal, int nIdx = -1)
        {
            if (m_bIncludeDiff)
            {
                if (mutable_gpu_diff == 0)
                    return;

                m_cuda.set(count(), mutable_gpu_diff, dfVal, nIdx);
            }
        }

        /// <summary>
        /// Set a diff range with a given value.
        /// </summary>
        /// <param name="dfVal">Specifies the value to set.</param>
        /// <param name="nStartIdx">Specifies the start index.</param>
        /// <param name="nCount">Specifies the number of items to set.</param>
        public void SetDiff(double dfVal, int nStartIdx, int nCount)
        {
            T tVal = (T)Convert.ChangeType(dfVal, typeof(T));
            T[] rg = mutable_cpu_diff;

            for (int i = 0; i < nCount; i++)
            {
                if (nStartIdx + i < rg.Length)
                    rg[nStartIdx + i] = tVal;
            }

            mutable_cpu_diff = rg;
        }

        /// <summary>
        /// Sets a number of items within the Blob's diff.
        /// </summary>
        /// <param name="rgDiff">Specifies the diff to set.</param>
        /// <param name="nCount">Optionally, specifies a subset count of items to set.</param>
        /// <param name="bSetCount">Optionally, specifies whether or not to set the count.  The count is always set when re-allocating the buffer.</param>
        public void SetDiff(T[] rgDiff, int nCount = -1, bool bSetCount = true)
        {
            m_diff.SetData(rgDiff, nCount, bSetCount);
        }

        /// <summary>
        /// Sets the Blob values to the data contained within a SimpleDatum.
        /// </summary>
        /// <param name="d">Specifies the SimpleDatum.</param>
        /// <param name="bReshape">Specifies whether or not to reshape the Blob to match the SimpleDatum.</param>
        /// <param name="bCopyData">Optionally, specifies whether or not to transfer the data.</param>
        public void SetData(SimpleDatum d, bool bReshape, bool bCopyData = true)
        {
            if (bReshape)
            {
                m_nCapacity = 0;
                Reshape(1, d.Channels, d.Height, d.Width);
            }

            T[] rgData = d.GetData<T>();

            m_log.CHECK_EQ(rgData.Length, count(), "The datum data length of " + rgData.Length.ToString() + " should be equal to the Blob count() of " + count().ToString());

            if (bCopyData)
            {
                mutable_cpu_data = rgData;
                m_nIdx = d.Index;
            }
        }

        /// <summary>
        /// Sets just the CPU data to the data specified.
        /// </summary>
        /// <param name="rg">Specifies the CPU data to set.</param>
        public void SetCPUData(T[] rg)
        {
            m_bCpuDataReadyForPush = true;
            m_data.set_cpu_data_locally(rg);
        }

        /// <summary>
        /// Asynchronously pushes the CPU data, previously set with SetCPUData, to the GPU.
        /// </summary>
        /// <param name="hStream">Specifies a handle to the Cuda Stream to use for synchronization.</param>
        public void AsyncGpuPush(long hStream)
        {
            if (!m_bCpuDataReadyForPush)
                return;

            if (m_data.cpu_data == null)
                throw new Exception("There is no CPU data to push to the GPU!");

            m_data.async_gpu_push(hStream, m_data.cpu_data);
            m_bCpuDataReadyForPush = false;
        }

        /// <summary>
        /// Compares the shape of this blob to the shape within a BlobProto.
        /// </summary>
        /// <param name="bp">Specifies the BlobProto to compare.</param>
        /// <returns>If the shapes are the same this method returns <i>true</i>, otherwise <i>false</i>.</returns>
        public bool ShapeEquals(BlobProto bp)
        {
            if (bp.num.HasValue || bp.channels.HasValue || bp.height.HasValue || bp.width.HasValue)
            {
                // Using drepreciated 4D blob dimensions --
                // shape is (num, channels, height, width).
                // Note: we do not use the normal Blob::num, Blob::channels() etc.
                // methods as these index from the beginning of the blob shape, where legacy
                // parameter blobs were indexed from the end of the blob shape (e.g., bias
                // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N).
                if (m_rgShape.Count <= 4 &&
                    LegacyShape(-4) == bp.num.GetValueOrDefault(1) &&
                    LegacyShape(-3) == bp.channels.GetValueOrDefault(1) &&
                    LegacyShape(-2) == bp.height.GetValueOrDefault(1) &&
                    LegacyShape(-1) == bp.width.GetValueOrDefault(1))
                    return true;
                else
                    return false;
            }

            return Utility.Compare<int>(m_rgShape, bp.shape.dim);
        }

        /// <summary>
        /// Compares the shape of this blob to another shape.
        /// </summary>
        /// <param name="rgShape">Specifies the shape to compare with.</param>
        /// <param name="bCompareCpuDataLen">Optionally, compare the CPU data length.</param>
        /// <returns>If the shapes are the same, return <i>true</i>, othewise <i>false</i>.</returns>
        public bool CompareShape(List<int> rgShape, bool bCompareCpuDataLen = false)
        {
            while (rgShape.Count < num_axes)
            {
                rgShape.Add(1);
            }

            List<int> rgShape1 = new List<int>(shape());
            while (rgShape1.Count < rgShape.Count)
            {
                rgShape1.Add(1);
            }

            if (bCompareCpuDataLen)
            {
                int nCount = 1;
                for (int i = 0; i < rgShape.Count; i++)
                {
                    nCount *= rgShape[i];
                }

                if (cpu_data == null || cpu_data.Length != nCount)
                    return false;
            }

            return Utility.Compare<int>(rgShape1, rgShape);
        }

        /// <summary>
        /// Returns a string describing the 4D shape of the Blob.
        /// </summary>
        /// <returns>A shape string is returned.</returns>
        public string ToSizeString()
        {
            return num.ToString() + "," + channels.ToString() + "," + height.ToString() + "," + width.ToString();
        }

        /// <summary>
        /// Returns a new Datum that contains the shape and data of the Blob.
        /// </summary>
        /// <returns>A Datum is returned.</returns>
        public Datum ToDatum()
        {
            if (typeof(T) == typeof(double))
            {
                double[] rgData = m_cuda.GetMemoryDouble(gpu_data, count());
                return new Datum(true, channels, width, height, -1, DateTime.MinValue, new List<double>(rgData), 0, false, -1);
            }
            else
            {
                float[] rgData = m_cuda.GetMemoryFloat(gpu_data, count());
                return new Datum(true, channels, width, height, -1, DateTime.MinValue, new List<float>(rgData), 0, false, -1);
            }
        }

        /// <summary>
        /// Get/set the name of the Blob.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
            set
            {
                m_strName = value;

                if (m_data != null)
                    m_data.Tag = m_strName + " data";

                if (m_diff != null)
                    m_diff.Tag = m_strName + " diff";
            }
        }

        /// <summary>
        /// Copies the Blob, including its data and diff.
        /// </summary>
        /// <returns>A new copy of the Blob is returned.</returns>
        public Blob<T> Clone()
        {
            Blob<T> b = new Blob<T>(m_cuda, m_log);

            b.ReshapeLike(this, HalfSize);

            if (m_diff != null)
                b.m_diff.Copy(m_diff);

            b.m_data.Copy(m_data);
            b.Name = Name;
            b.m_bReshapeWhenSharing = reshape_when_sharing;

            return b;
        }

        /// <summary>
        /// Clones the input Blob, scales the clone and then adds the data from this Blob to it.
        /// </summary>
        /// <remarks>
        /// Calculation: Y = Clone(blobA) * dfScale + this
        /// </remarks>
        /// <param name="blobA">Specifies the Blob to clone.</param>
        /// <param name="fScale">Specifies the scaling factor to apply to the clone.</param>
        /// <returns>The Blob copy is returned.</returns>
        public Blob<T> MathAdd(Blob<T> blobA, T fScale)
        {
            Blob<T> bOut = blobA.Clone();

            if ((double)Convert.ChangeType(fScale, typeof(double)) != 1.0)
                bOut.scale_data(fScale);

            m_cuda.add(bOut.count(), bOut.gpu_data, gpu_data, bOut.mutable_gpu_data);

            return bOut;
        }

        /// <summary>
        /// Clones the input Blob and subtracts the data from this blob from it.
        /// </summary>
        /// <remarks>
        /// Calculation: Y = Clone(blobA) + this
        /// </remarks>
        /// <param name="blobA">Specifies the Blob to clone.</param>
        /// <returns>The Blob copy is returned.</returns>
        public Blob<T> MathSub(Blob<T> blobA)
        {
            Blob<T> bOut = blobA.Clone();

            m_cuda.sub(bOut.count(), bOut.gpu_data, gpu_data, bOut.mutable_gpu_data);

            return bOut;
        }

        /// <summary>
        /// Clones the input Blob and divides a scalar from all of the clones data items.
        /// </summary>
        /// <remarks>
        /// Calculation: Y = Clone(this) * fScale
        /// </remarks>
        /// <param name="fScale">Specifies scaling factor.</param>
        /// <returns>The Blob copy is returned.</returns>
        public Blob<T> MathDiv(T fScale)
        {
            Blob<T> bOut = Clone();

            double dfVal = 1.0/(double)Convert.ChangeType(fScale, typeof(double));

            m_cuda.mul_scalar(bOut.count(), (T)Convert.ChangeType(dfVal, typeof(T)), bOut.mutable_gpu_data);

            return bOut;
        }

        /// <summary>
        /// Saves this Blob to a binary stream.
        /// </summary>
        /// <param name="bw">Specifies the binary writer.</param>
        /// <param name="bData">Specifies whether or not to save the data.</param>
        /// <param name="bDiff">Specifies whether or not to save the diff.</param>
        public void Save(BinaryWriter bw, bool bData, bool bDiff)
        {
            bw.Write(m_strName);
            bw.Write(m_rgShape.Count);

            for (int i = 0; i < m_rgShape.Count; i++)
            {
                bw.Write(m_rgShape[i]);
            }

            bw.Write(m_nCount);

            if (bData)
            {
                bw.Write(m_nCount);

                double[] rgdfData = Utility.ConvertVec<T>(update_cpu_data());

                foreach (double dfVal in rgdfData)
                {
                    bw.Write(dfVal);
                }
            }
            else
            {
                bw.Write((int)0);
            }

            if (bDiff)
            {
                bw.Write(m_nCount);

                double[] rgdfDiff = Utility.ConvertVec<T>(update_cpu_diff());

                foreach (double dfVal in rgdfDiff)
                {
                    bw.Write(dfVal);
                }
            }
            else
            {
                bw.Write((int)0);
            }
        }

        /// <summary>
        /// Lods a new Blob from a binary reader.
        /// </summary>
        /// <param name="cuda">Specifies the instance of the CudaDnn connection to use.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="br">Specifies the binary reader.</param>
        /// <param name="bData">Specifies whether or not to read in the data.</param>
        /// <param name="bDiff">Specifies whether or not to read in the diff.</param>
        /// <returns></returns>
        public static Blob<T> Load(CudaDnn<T> cuda, Log log, BinaryReader br, bool bData, bool bDiff)
        {
            Blob<T> b = new Blob<T>(cuda, log);

            b.Name = br.ReadString();

            List<int> rgShape = new List<int>();
            int nCount = br.ReadInt32();

            for (int i = 0; i < nCount; i++)
            {
                rgShape.Add(br.ReadInt32());
            }

            b.Reshape(rgShape);

            int nItemCount = br.ReadInt32();
            int nDataCount = br.ReadInt32();

            if (nDataCount > nItemCount)
                throw new Exception("Invalid data count read!");

            List<double> rgData = new List<double>();

            for (int i = 0; i < nDataCount; i++)
            {
                rgData.Add(br.ReadDouble());
            }

            int nDiffCount = br.ReadInt32();

            if (nDiffCount > nItemCount)
                throw new Exception("Invalid diff count read!");

            List<double> rgDiff = new List<double>();

            for (int i = 0; i < nDiffCount; i++)
            {
                rgDiff.Add(br.ReadDouble());
            }

            if (nDataCount > 0 && nDiffCount > 0 && nDataCount != nDiffCount)
                throw new Exception("Invalid diff and data counts read - they should be equal!");

            if (bData && rgData.Count > 0)
                b.mutable_cpu_data = Utility.ConvertVec<T>(rgData.ToArray());

            if (bDiff && rgDiff.Count > 0)
                b.mutable_cpu_diff = Utility.ConvertVec<T>(rgDiff.ToArray());

            return b;
        }

        /// <summary>
        /// Returns a string representation of the Blob.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string strSize = (HalfSize) ? "HALF " : "FULL ";
            return strSize + m_strName + " (" + shape_string + ")";
        }

        /// <summary>
        /// Get the string representation containing up to the first 'nMax' items.
        /// </summary>
        /// <param name="nMax">Specifies the maximum number of data items to return.</param>
        /// <param name="bDiff">Specifies to returive the diff when <i>true</i>.</param>
        /// <returns>The string representation is returned.</returns>
        public string ToString(int nMax, bool bDiff = false)
        {
            double[] rg = Utility.ConvertVec<T>((bDiff) ? update_cpu_diff() : update_cpu_data());
            string str = "{";

            for (int i = 0; i < rg.Length && i < nMax; i++)
            {
                str += rg[i].ToString("N4");
                str += ",";
            }

            str = str.TrimEnd(',');
            str += "}";

            return str;
        }

        /// <summary>
        /// Returns the minimum value in the data of the Blob.
        /// </summary>
        public double min_data
        {
            get
            {
                long lPos;
                return GetMinData(out lPos);
            }
        }

        /// <summary>
        /// Returns the minimum data and the position where the minimum is located in the data.
        /// </summary>
        /// <param name="lPos">Returns the location of the minimum.</param>
        /// <returns>Returns the minimum value.</returns>
        public double GetMinData(out long lPos)
        {
            lPos = -1;
            if (count() == 0 || gpu_data == 0)
                return 0;

            return m_cuda.min(count(), gpu_data, out lPos);
        }

        /// <summary>
        /// Returns the maximum value in the data of the Blob.
        /// </summary>
        public double max_data
        {
            get
            {
                long lPos;
                return GetMaxData(out lPos);
            }
        }

        /// <summary>
        /// Returns the maximum data and the position where the maximum is located in the data.
        /// </summary>
        /// <param name="lPos">Returns the location of the maximum.</param>
        /// <returns>Returns the maximum value.</returns>
        public double GetMaxData(out long lPos)
        {
            lPos = -1;
            if (count() == 0 || gpu_data == 0)
                return 0;

            return m_cuda.max(count(), gpu_data, out lPos);
        }

        /// <summary>
        /// Returns the minimum value in the diff of the Blob.
        /// </summary>
        public double min_diff
        {
            get
            {
                long lPos;
                return GetMinDiff(out lPos);
            }
        }

        /// <summary>
        /// Returns the minimum diff and the position where the minimum is located in the diff.
        /// </summary>
        /// <param name="lPos">Returns the location of the minimum.</param>
        /// <returns>Returns the minimum value.</returns>
        public double GetMinDiff(out long lPos)
        {
            lPos = -1;
            if (count() == 0 || gpu_diff == 0)
                return 0;

            return m_cuda.min(count(), gpu_diff, out lPos);
        }

        /// <summary>
        /// Returns the maximum value in the diff of the Blob.
        /// </summary>
        public double max_diff
        {
            get
            {
                long lPos;
                return GetMaxDiff(out lPos);
            }
        }

        /// <summary>
        /// Returns the maximum diff and the position where the maximum is located in the diff.
        /// </summary>
        /// <param name="lPos">Returns the location of the maximum.</param>
        /// <returns>Returns the maximum value.</returns>
        public double GetMaxDiff(out long lPos)
        {
            lPos = -1;
            if (count() == 0 || gpu_diff == 0)
                return 0;

            return m_cuda.max(count(), gpu_diff, out lPos);
        }

        private int get_index_up_to(List<int> rgShape, int nMax = 12800)
        {
            int nIndexUpTo = m_rgShape.Count - 1;
            int nNum = m_rgShape[nIndexUpTo];

            while (nIndexUpTo > 0 && nNum < nMax)
            {
                nNum *= m_rgShape[nIndexUpTo-1];
                if (nNum < nMax)
                    nIndexUpTo--;
            }

            if (nNum > 1 && nIndexUpTo == 0 && m_rgShape.Count > 1)
                nIndexUpTo++;

            return nIndexUpTo;
        }

        /// <summary>
        /// Returns the minimum and maximum values in the data of the Blob.
        /// </summary>
        /// <param name="work">Specifies a workspace used to optimize the query.</param>
        /// <param name="bDetectNans">Optionally, specifies whether or not to detect Nan's and Infinity values.</param>
        /// <param name="bUseChunks">Optionally, specifies whether or not to use the min/max on all of the memory (default) or run in chunks (true).</param>
        /// <returns>A tuple containing the 'min', 'max' and optionally 'number of nans' and 'number of infinity' is returned for the data.</returns>
        public Tuple<double, double, double, double> minmax_data(Blob<T> work, bool bDetectNans = false, bool bUseChunks = false)
        {
            int nCount = count();

            if (nCount == 0 || gpu_data == 0)
                return new Tuple<double, double, double, double>(0, 0, 0, 0);

            if (nCount == 1)
            {
                double[] rgdf = Utility.ConvertVec<T>(mutable_cpu_data);
                return new Tuple<double, double, double, double>(rgdf[0], rgdf[0], double.IsNaN(rgdf[0]) ? 1 : 0, double.IsInfinity(rgdf[0]) ? 1 : 0);
            }

            work.Reshape(nCount + 64, 1, 1, 1);
            work.ReshapeLike(this);

            int nIndexUpTo = get_index_up_to(m_rgShape);
            if (nIndexUpTo == 0 || nIndexUpTo >= m_rgShape.Count || !bUseChunks)
                return m_cuda.minmax(nCount, gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff, bDetectNans);

            List<double> rgdfMax = new List<double>();
            List<double> rgdfMin = new List<double>();
            List<double> rgdfItem3 = new List<double>();
            List<double> rgdfItem4 = new List<double>();
            List<int> rgShape = new List<int>();
            int nNum = 1;

            for (int i = 0; i < m_rgShape.Count; i++)
            {
                if (i < nIndexUpTo)
                {
                    nNum *= m_rgShape[i];
                    rgShape.Add(1);
                }
                else
                {
                    rgShape.Add(m_rgShape[i]);
                }
            }

            Blob<T> item = new Blob<T>(m_cuda, m_log, rgShape, false);        

            for (int i = 0; i < nNum; i++)
            {
                nCount = item.count();
                m_cuda.copy(nCount, gpu_data, item.mutable_gpu_data, i * nCount, 0);
                Tuple<double, double, double, double> minmax = m_cuda.minmax(nCount, item.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff, bDetectNans);
                rgdfMin.Add(minmax.Item1);
                rgdfMax.Add(minmax.Item2);
                rgdfItem3.Add(minmax.Item3);
                rgdfItem4.Add(minmax.Item4);
            }

            item.Dispose();

            double dfMin = rgdfMin.Min(p => p);
            double dfMax = rgdfMax.Max(p => p);
            double dfItem3 = rgdfItem3.Sum(p => p);
            double dfItem4 = rgdfItem4.Sum(p => p);

            return new Tuple<double, double, double, double>(dfMin, dfMax, dfItem3, dfItem4);
        }

        /// <summary>
        /// Returns the minimum and maximum values in the diff of the Blob.
        /// </summary>
        /// <param name="work">Specifies a workspace used to optimize the query.</param>
        /// <param name="bDetectNans">Optionally, specifies whether or not to detect Nan's and Infinity values.</param>
        /// <param name="bUseChunks">Optionally, specifies whether or not to use the min/max on all of the memory (default) or run in chunks (true).</param>
        /// <returns>A tuple containing the 'min', 'max' and optionally 'number of nans' and 'number of infinity' is returned for the data.</returns>
        public Tuple<double, double, double, double> minmax_diff(Blob<T> work, bool bDetectNans = false, bool bUseChunks = false)
        {
            int nCount = count();

            if (nCount == 0 || gpu_diff == 0)
                return new Tuple<double, double, double, double>(0, 0, 0, 0);

            if (nCount == 1)
            {
                double[] rgdf = Utility.ConvertVec<T>(mutable_cpu_diff);
                return new Tuple<double, double, double, double>(rgdf[0], rgdf[0], double.IsNaN(rgdf[0]) ? 1 : 0, double.IsInfinity(rgdf[0]) ? 1 : 0);
            }

            work.Reshape(nCount + 64, 1, 1, 1);
            work.ReshapeLike(this);

            int nIndexUpTo = get_index_up_to(m_rgShape);
            if (nIndexUpTo == 0 || nIndexUpTo >= m_rgShape.Count || !bUseChunks)
                return m_cuda.minmax(nCount, gpu_diff, work.mutable_gpu_data, work.mutable_gpu_diff, bDetectNans);

            List<double> rgdfMax = new List<double>();
            List<double> rgdfMin = new List<double>();
            List<double> rgdfItem3 = new List<double>();
            List<double> rgdfItem4 = new List<double>();
            List<int> rgShape = new List<int>();
            int nNum = 1;

            for (int i = 0; i < m_rgShape.Count; i++)
            {
                if (i < nIndexUpTo)
                {
                    nNum *= m_rgShape[i];
                    rgShape.Add(1);
                }
                else
                {
                    rgShape.Add(m_rgShape[i]);
                }
            }

            Blob<T> item = new Blob<T>(m_cuda, m_log, rgShape, false);

            for (int i = 0; i < nNum; i++)
            {
                nCount = item.count();
                m_cuda.copy(nCount, gpu_diff, item.mutable_gpu_data, i * nCount, 0);
                Tuple<double, double, double, double> minmax = m_cuda.minmax(nCount, item.gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff, bDetectNans);
                rgdfMin.Add(minmax.Item1);
                rgdfMax.Add(minmax.Item2);
                rgdfItem3.Add(minmax.Item3);
                rgdfItem4.Add(minmax.Item4);
            }

            item.Dispose();

            double dfMin = rgdfMin.Min(p => p);
            double dfMax = rgdfMax.Max(p => p);
            double dfItem3 = rgdfItem3.Sum(p => p);
            double dfItem4 = rgdfItem4.Sum(p => p);

            return new Tuple<double, double, double, double>(dfMin, dfMax, dfItem3, dfItem4);
        }

        /// <summary>
        /// Returns the BLOB_TYPE of the Blob.
        /// </summary>
        public BLOB_TYPE type
        {
            get { return m_type; }
            set { m_type = value; }
        }

        /// <summary>
        /// Returns a user defined object associated with the Blob.
        /// </summary>
        public object Tag
        {
            get { return m_tag; }
            set { m_tag = value; }
        }

        /// <summary>
        /// Adds a scalar value to the Blob.
        /// </summary>
        /// <param name="dfVal">Specifies the scalar value.</param>
        public void add_scalar(double dfVal)
        {
            m_cuda.add_scalar(count(), dfVal, mutable_gpu_data);
        }

#pragma warning disable 1591

        public void rollaxis() /** @private */
        {
            long hDataT = m_cuda.AllocMemory(mutable_cpu_data);
            m_cuda.matrix_transpose(width * height, channels, hDataT, mutable_gpu_data);
            m_cuda.FreeMemory(hDataT);
        }

        public void save_images(string strPath, Blob<T> blobLabels, double dfScale) /** @private */
        {
            if (!Directory.Exists(strPath))
                Directory.CreateDirectory(strPath);

            T[] rgLabels = blobLabels.update_cpu_data();
            T[] rgData = update_cpu_data();

            if (dfScale != 1.0)
            {
                for (int i = 0; i < rgData.Length; i++)
                {
                    double dfVal = Utility.ConvertVal<T>(rgData[i]);

                    if (dfVal != 0)
                    {
                        dfVal /= dfScale;

                        if (dfVal < 0)
                            dfVal = 0;

                        if (dfVal > 255)
                            dfVal = 255;

                        rgData[i] = Utility.ConvertVal<T>(dfVal);
                    }
                }
            }

            strPath = strPath.TrimEnd('\\');

            int nCount = channels * height * width;

            for (int n = 0; n < num; n++)
            {
                int nIdx = n * nCount;
                Datum d = ImageData.GetImageData<T>(rgData, channels, height, width, false, nIdx, nCount);
                Image img = ImageData.GetImage(d);

                int nLabel = (int)(float)Convert.ChangeType(rgLabels[n], typeof(float));

                string strName = strPath + "\\img_" + n.ToString() + "_label_" + nLabel.ToString() + ".png";
                img.Save(strName, System.Drawing.Imaging.ImageFormat.Png);

                img.Dispose();
            }
        }

#pragma warning restore 1591

        /// <summary>
        /// The Resize method resizes the 3rd and 4th axes of the blob.
        /// </summary>
        /// <remarks>
        /// Currently, the Resize method only works on 4 axis blobs.  Resize is different from Reshape in that Resize
        /// actually averages the data when resizing the blob.
        /// </remarks>
        /// <param name="rgShape">Specifies the new shape to resize to.</param>
        /// <returns>A newly re-sized Blob is returned.</returns>
        public Blob<T> Resize(List<int> rgShape)
        {
            m_log.CHECK_EQ(num_axes, rgShape.Count, "When resizing, the new shape must have the same number of axes as the blob to be resized.");
            m_log.CHECK_EQ(num_axes, 4, "Resizing only allowed on 4 axis blobs.");
            m_log.CHECK_EQ(num, rgShape[0], "Resizing only allowed on the last two axes.");
            m_log.CHECK_EQ(channels, rgShape[1], "Resizing only allowed on the last two axes.");

            T[] rgData = mutable_cpu_data;
            float[] rgDataF = Utility.ConvertVecF<T>(rgData);

            Blob<T> newBlob = Clone();
            newBlob.Reshape(rgShape);

            T[] rgDataNew = newBlob.mutable_cpu_data;
            float[] rgDataNewF = Utility.ConvertVecF<T>(rgDataNew);

            Bitmap bmp = new Bitmap(width, height);

            for (int n = 0; n < num; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float fMin = float.MaxValue;
                    float fMax = -float.MaxValue;
                    int nH = height;
                    int nW = width;
                    int nSize = nH * nW;

                    for (int y = 0; y < nH; y++)
                    {
                        for (int x = 0; x < nW; x++)
                        {
                            int nIdx = n * (channels * nSize) + c * nSize + y * nW + x;
                            float fVal = rgDataF[nIdx];

                            if (fVal < fMin)
                                fMin = fVal;

                            if (fVal > fMax)
                                fMax = fVal;
                        }
                    }

                    for (int y = 0; y < nH; y++)
                    {
                        for (int x = 0; x < nW; x++)
                        {
                            int nIdx = n * (channels * nSize) + c * nSize + y * nW + x;
                            float fVal = rgDataF[nIdx];

                            fVal = ((fVal - fMin) / (fMax - fMin)) * 255.0f;    // move into range 0,255

                            bmp.SetPixel(x, y, Color.FromArgb((int)fVal, (int)fVal, (int)fVal));
                        }
                    }

                    Bitmap bmpNew = ImageTools.ResizeImage(bmp, newBlob.width, newBlob.height);

                    nH = newBlob.height;
                    nW = newBlob.width;
                    nSize = nH * nW;

                    for (int y = 0; y < nH; y++)
                    {
                        for (int x = 0; x < nW; x++)
                        {
                            int nIdx = n * (channels * nSize) + c * nSize + y * nW + x;

                            Color clr = bmpNew.GetPixel(x, y);
                            float fVal = ((clr.R / 255.0f) * (fMax - fMin)) + fMin; // move back to original range.

                            rgDataNewF[nIdx] = fVal;
                        }
                    }

                    bmpNew.Dispose();
                }
            }

            bmp.Dispose();
            newBlob.mutable_cpu_data = Utility.ConvertVec<T>(rgDataNewF);

            return newBlob;
        }

        /// <summary>
        /// Normalize the blob data by subtracting the mean and dividing by the standard deviation.
        /// </summary>
        /// <param name="dfMean">Optionally, specifies a mean value to use (default = <i>null</i>).</param>
        /// <param name="dfStd">Optionally, specifies a std value to use (default = <i>null</i>).</param>
        public void NormalizeData(double? dfMean = null, double? dfStd = null)
        {
            if (!dfMean.HasValue || !dfStd.HasValue)
            {
                float[] rgF = Utility.ConvertVecF<T>(update_cpu_data());

                dfMean = mean(rgF);
                dfStd = std(dfMean.Value, rgF);
            }

            if (dfMean.Value != 0)
                m_cuda.add_scalar(count(), -1 * dfMean.Value, mutable_gpu_data);

            if (dfStd.Value != 0 && dfStd.Value != 1.0)
                m_cuda.mul_scalar(count(), 1.0 / dfStd.Value, mutable_gpu_data);
        }

        /// <summary>
        /// Calculate the mean of the blob data.
        /// </summary>
        /// <param name="rgDf">Optionally, specifies the CPU data to use (default = <i>null</i>).</param>
        /// <param name="bDiff">Optionally, specifies to use the diff instead of the data.</param>
        /// <returns>The mean is returned.</returns>
        public double mean(float[] rgDf = null, bool bDiff = false)
        {
            double dfSum = 0;

            if (rgDf == null)
                rgDf = Utility.ConvertVecF<T>((bDiff) ? update_cpu_diff() : update_cpu_data());

            for (int i = 0; i < rgDf.Length; i++)
            {
                dfSum += rgDf[i];
            }

            return dfSum / rgDf.Length;
        }

        /// <summary>
        /// Calculate the standard deviation of the blob data.
        /// </summary>
        /// <param name="dfMean">Optionally, specifies the mean to use (default = <i>null</i>).</param>
        /// <param name="rgDf">Optionally, specifies the CPU data to calculate the std on (default = <i>null</i>).</param>
        /// <returns>The standard deviation of the bob data is returned.</returns>
        public double std(double? dfMean = null, float[] rgDf = null)
        {
            double dfSum = 0;

            if (rgDf == null)
                rgDf = Utility.ConvertVecF<T>(update_cpu_data());

            if (!dfMean.HasValue)
                dfMean = mean(rgDf);

            for (int i = 0; i < rgDf.Length; i++)
            {
                dfSum += Math.Pow(rgDf[i] - dfMean.Value, 2);
            }

            return Math.Sqrt(dfSum / rgDf.Length);
        }

        /// <summary>
        /// Set the values of a 3 channel image embedded within the blob in the order RGB or BGR where the full hxw is stored for each color in order.
        /// </summary>
        /// <param name="nX">Specifies the x pixel within the image.</param>
        /// <param name="nY">Specifies the y pxiel within the image.</param>
        /// <param name="R">Specifies the color for the Red color channel.</param>
        /// <param name="G">Specifies the color for the Green color channel.</param>
        /// <param name="B">Specifies the color for the Blue color channel.</param>
        /// <param name="order">Optionally, specifies the color ordering RGB or BGR (default = RGB).</param>
        /// <returns>The original pixel is returned as a three item tuple.</returns>
        public Tuple<T, T, T> SetPixel(int nX, int nY, byte R, byte G, byte B, TransformationParameter.COLOR_ORDER order = TransformationParameter.COLOR_ORDER.RGB)
        {
            int nDim = width * height;
            int nIdxR = nY * width + nX;
            int nIdxG = nDim + nIdxR;
            int nIdxB = nDim + nDim + nIdxR;

            if (order == TransformationParameter.COLOR_ORDER.BGR)
            {
                int nTemp = nIdxB;
                nIdxB = nIdxR;
                nIdxR = nTemp;
            }

            T fR = Utility.ConvertVal<T>((double)R);
            T fG = Utility.ConvertVal<T>((double)G);
            T fB = Utility.ConvertVal<T>((double)B);

            T[] rg = m_cuda.SetPixel(mutable_gpu_data, count(), true, 0, new Tuple<int, T>(nIdxR, fR), new Tuple<int, T>(nIdxG, fG), new Tuple<int, T>(nIdxB, fB));

            return new Tuple<T, T, T>(rg[0], rg[1], rg[2]);
        }

        /// <summary>
        /// Sets a pixel to the values within a three item tuple where the first item is assigned RED, second GREEN and third BLUE.
        /// </summary>
        /// <param name="nX">Specifies the x pixel within the image.</param>
        /// <param name="nY">Specifies the y pxiel within the image.</param>
        /// <param name="pixel">Specifies the color values for RED, GREEN and BLUE.</param>
        /// <param name="bReturnOriginal">Optionally, specifies to return the original pixel.</param>
        /// <param name="order">Optionally, specifies the color ordering RGB or BGR (default = RGB).</param>
        /// <param name="nOffset">Optionally, specifies the offset to start at (default = 0).</param>
        /// <returns>The original pixel is returned as a three item tuple.</returns>
        public Tuple<T, T, T> SetPixel(int nX, int nY, Tuple<T, T, T> pixel, bool bReturnOriginal = false, TransformationParameter.COLOR_ORDER order = TransformationParameter.COLOR_ORDER.RGB, int nOffset = 0)
        {
            int nDim = width * height;
            int nIdxR = nY * width + nX;
            int nIdxG = nDim + nIdxR;
            int nIdxB = nDim + nDim + nIdxR;

            if (order == TransformationParameter.COLOR_ORDER.BGR)
            {
                int nTemp = nIdxB;
                nIdxB = nIdxR;
                nIdxR = nTemp;
            }

            T fR = pixel.Item1;
            T fG = pixel.Item2;
            T fB = pixel.Item3;

            if (bReturnOriginal)
            {
                T[] rg = m_cuda.SetPixel(mutable_gpu_data, count(), true, nOffset, new Tuple<int, T>(nIdxR, fR), new Tuple<int, T>(nIdxG, fG), new Tuple<int, T>(nIdxB, fB));
                return new Tuple<T, T, T>(rg[0], rg[1], rg[2]);
            }
            else
            {
                m_cuda.SetPixel(mutable_gpu_data, count(), true, nOffset, new Tuple<int, T>(nIdxR, fR), new Tuple<int, T>(nIdxG, fG), new Tuple<int, T>(nIdxB, fB));
                return null;
            }
        }

        /// <summary>
        /// Save the Blob to an image where values less than 0 are colored red, and values greater than 0 are colored green.  Values = 0 are colored black.
        /// </summary>
        /// <param name="strFile">Specifies the file where the image is saved.</param>
        /// <param name="bNonZeroExistOnly">Optionally, specifies to color whether data exists as nonZero only (default = true).</param>
        /// <param name="bSaveDiff">Optionally, specifies to save the diff instead of the data (default = false).</param>
        /// <param name="rgSpecialValues">Optionally, specifies special values with associated colors.</param>
        public void SaveToImage(string strFile, bool bNonZeroExistOnly = true, bool bSaveDiff = false, Dictionary<float, Color> rgSpecialValues = null)
        {
            Blob<T> blobWork = new Blob<T>(m_cuda, m_log);
            float[] rgData;
            double dfMin = 1;
            double dfMax = 1;
            int nWid = width;
            int nHt = height;
            int nNumY = num;
            int nNumX = channels;

            try
            {
                blobWork.ReshapeLike(this);

                if (bSaveDiff)
                {
                    rgData = Utility.ConvertVecF<T>(mutable_cpu_diff);

                    if (!bNonZeroExistOnly)
                    {
                        Tuple<double, double, double, double> minmax = minmax_diff(blobWork);
                        dfMin = minmax.Item1;
                        dfMax = minmax.Item2;
                    }
                }
                else
                {
                    rgData = Utility.ConvertVecF<T>(mutable_cpu_data);

                    if (!bNonZeroExistOnly)
                    {
                        Tuple<double, double, double, double> minmax = minmax_data(blobWork);
                        dfMin = minmax.Item1;
                        dfMax = minmax.Item2;
                    }
                }

                if (nWid == 1 && nHt == 1)
                {
                    nNumX = 1;
                    nNumY = 1;
                    nHt = num;
                    nWid = channels;
                }
                else if (nWid == 1)
                {
                    nNumY = num;
                    nNumX = 1;
                    nWid = height;
                    nHt = channels;
                }

                Bitmap bmp = new Bitmap(nNumX * nWid + (nNumX - 1), nNumY * nHt + (nNumY - 1));
                using (Graphics g = Graphics.FromImage(bmp))
                {
                    g.FillRectangle(Brushes.Black, 0, 0, bmp.Width, bmp.Height);
                }

                LockBitmap bmp1 = new LockBitmap(bmp);
                bmp1.LockBits();

                int nX = 0;
                int nY = 0;
                int nX1 = 0;
                int nY1 = 0;

                for (int y = 0; y < nNumY; y++)
                {
                    for (int x = 0; x < nNumX; x++)
                    {
                        for (int h = 0; h < nHt; h++)
                        {
                            for (int w = 0; w < nWid; w++)
                            {
                                int nIdx = y *  nNumX * nHt * nWid + x * nHt * nWid + h * nWid + w;
                                float fVal = rgData[nIdx];
                                Color clr = Color.Empty;

                                if (rgSpecialValues != null && rgSpecialValues.ContainsKey(fVal))
                                {
                                    clr = rgSpecialValues[fVal];
                                }
                                else if (fVal < 0)
                                {
                                    if (bNonZeroExistOnly)
                                    {
                                        clr = Color.White;
                                    }
                                    else
                                    {
                                        int nClr = (int)(255 * (rgData[nIdx] - dfMin) / (0 - dfMin));
                                        clr = Color.FromArgb(nClr, 0, 0);
                                    }
                                }
                                else if (fVal > 0)
                                {
                                    if (bNonZeroExistOnly)
                                    {
                                        clr = Color.White;
                                    }
                                    else
                                    {
                                        int nClr = (int)(255 * (rgData[nIdx] - 0) / (dfMax - 0));
                                        clr = Color.FromArgb(0, nClr, 0);
                                    }
                                }

                                if (!clr.IsEmpty)
                                    bmp1.SetPixel(nX1, nY1, clr);

                                nX1++;
                            }

                            nX1 = nX;
                            nY1++;                            
                        }

                        nX += nWid + 1;
                        nX1 = nX;
                        nY1 = nY;
                    }

                    nX = 0;
                    nY += nHt + 1;
                    nY1 = nY;
                }

                bmp1.UnlockBits();
                bmp.Save(strFile);
            }
            finally
            {
                blobWork.Dispose();
            }
        }

        /// <summary>
        /// Used to debug the mask and masked data.
        /// </summary>
        /// <param name="b">Specifies the blob to debug.</param>
        /// <param name="strFile">Specifies the bitmap file where the image is saved.</param>
        private void drawImage(Blob<T> b, string strFile)
        {
        }

        /// <summary>
        /// Save a blob with data to a Numpy .npy file.
        /// </summary>
        /// <param name="strFile">Specifies the .npy file name where the data is saved.</param>
        /// <param name="bSaveDiff">Specifies to save the diff, when false, the data is saved.</param>
        /// <remarks>
        /// @see[A Simple File Format for NumPy Arrays](https://numpy.org/doc/1.13/neps/npy-format.html)
        /// </remarks>
        public void SaveToNumpy(string strFile, bool bSaveDiff = false)
        {
            using (FileStream fs = File.Open(strFile, FileMode.Create))
            using (BinaryWriter bw = new BinaryWriter(fs))
            {
                bw.Write((byte)0x93);
                bw.Write((byte)0x4E); // N
                bw.Write((byte)0x55); // U
                bw.Write((byte)0x4D); // M
                bw.Write((byte)0x50); // P
                bw.Write((byte)0x59); // Y
                bw.Write((byte)0x01);
                bw.Write((byte)0x00);

                string strHeader = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
                for (int i = 0; i < shape().Count; i++)
                {
                    strHeader += shape(i).ToString() + ",";
                }

                strHeader = strHeader.TrimEnd(',');
                strHeader += ")";
                strHeader = strHeader.PadRight(117, ' ');
                strHeader += "\n";

                byte bLen = (byte)strHeader.Length;
                bw.Write(bLen);
                bw.Write((byte)0x00);

                foreach (char ch in strHeader)
                {
                    bw.Write((byte)ch);
                }

                float[] rgData;
                if (bSaveDiff)
                    rgData = Utility.ConvertVecF<T>(mutable_cpu_diff);
                else
                    rgData = Utility.ConvertVecF<T>(mutable_cpu_data);
                
                for (int i = 0; i < rgData.Length; i++)
                {
                    bw.Write(rgData[i]);
                }
            }
        }

        /// <summary>
        /// Load a blob with data from a Numpy array .npy file.
        /// </summary>
        /// <param name="strFile">Specifies the .npy file name.</param>
        /// <param name="bLoadDiff">Specifies to load the diff, when false, the data is loaded.</param>
        /// <param name="bLoadDataOnly">Specifies to load the data and return it as an array but do not load the gpu memory.</param>
        /// <param name="log">Optionally, specifies the output log.</param>
        /// <param name="nMax">Optionally, specifies the maximum number of items to load.</param>
        /// <exception cref="Exception">An exception is thrown when an invalid or unsupported feature is located.</exception>
        /// <remarks>
        /// @see[A Simple File Format for NumPy Arrays](https://numpy.org/doc/1.13/neps/npy-format.html)
        /// </remarks>
        /// <returns>A tuple containing the float[] data and int[] shape is returned.</returns>
        public Tuple<float[], int[]> LoadFromNumpy(string strFile, bool bLoadDiff = false, bool bLoadDataOnly = false, Log log = null, int nMax = int.MaxValue)
        {
            using (FileStream fs = File.OpenRead(strFile))
            using (BinaryReader br = new BinaryReader(fs))
            {
                byte[] rgMagic = new byte[6];
                for (int i = 0; i < rgMagic.Length; i++)
                {
                    rgMagic[i] = br.ReadByte();
                }

                if (rgMagic[0] != 0x93 || rgMagic[1] != 0x4E || rgMagic[2] != 0x55 || rgMagic[3] != 0x4D || rgMagic[4] != 0x50 || rgMagic[5] != 0x59)
                    throw new Exception("The file is not a valid Numpy file!");

                byte bMajor = br.ReadByte();
                byte bMinor = br.ReadByte();

                if (bMajor != 1 || bMinor != 0)
                    throw new Exception("The file is not a valid Numpy file!");

                byte bHeaderLen1 = br.ReadByte();
                byte bHeaderLen2 = br.ReadByte();
                int nHeaderLen = bHeaderLen2 << 8 | bHeaderLen1;

                byte[] rgHeader = new byte[nHeaderLen];
                for (int i = 0; i < rgHeader.Length; i++)
                {
                    rgHeader[i] = br.ReadByte();
                }
                string strHeader = Encoding.ASCII.GetString(rgHeader);

                bool bFortranOrder;
                int[] rgShape;
                Type dataType;
                int nCount = parseHeader(strHeader, out bFortranOrder, out rgShape, out dataType, nMax);
                if (nCount < 0)
                    throw new Exception("The file size is too large for a flat array.");

                if (bFortranOrder)
                    throw new Exception("Currently the fortran ordering is not supported");

                Stopwatch sw = null;
                if (log != null)
                {
                    sw = new Stopwatch();
                    sw.Start();
                }

                float[] rgData = new float[nCount];
                for (int i = 0; i < rgData.Length; i++)
                {
                    if (dataType == typeof(float))
                        rgData[i] = br.ReadSingle();
                    else if (dataType == typeof(double))
                        rgData[i] = (float)br.ReadDouble();
                    else if (dataType == typeof(int))
                        rgData[i] = (float)br.ReadInt32();
                    else if (dataType == typeof(long))
                        rgData[i] = (float)br.ReadInt64();
                    else if (dataType == typeof(bool))
                        rgData[i] = (br.ReadBoolean()) ? 1 : 0;
                    else
                        throw new Exception("Unsupported data type!");

                    if (log != null)
                    {
                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            double dfPct = (double)i / (double)rgData.Length;
                            string strOut = "Loading '" + strFile + "' at " + dfPct.ToString("P5") + "...";
                            log.WriteLine(strOut, true);
                            sw.Restart();
                        }
                    }
                }

                if (!bLoadDataOnly)
                {
                    Reshape(rgShape);

                    if (bLoadDiff)
                        mutable_cpu_diff = Utility.ConvertVec<T>(rgData);
                    else
                        mutable_cpu_data = Utility.ConvertVec<T>(rgData);
                }

                return new Tuple<float[], int[]>(rgData, rgShape);
            }            
        }

        /// <summary>
        /// Load a the data from a very large Numpy array .npy file.
        /// </summary>
        /// <param name="strFile">Specifies the .npy file name.</param>
        /// <param name="log">Optionally, specifies the output log.</param>
        /// <param name="nMax">Optionally, specifies the maximum number of items to load.</param>
        /// <exception cref="Exception">An exception is thrown when an invalid or unsupported feature is located.</exception>
        /// <remarks>
        /// @see[A Simple File Format for NumPy Arrays](https://numpy.org/doc/1.13/neps/npy-format.html)
        /// </remarks>
        /// <returns>A tuple containing the float[,] data and int[] shape is returned.</returns>
        public static Tuple<List<float[]>, int[]> LoadFromNumpyEx(string strFile, Log log = null, int nMax = int.MaxValue)
        {
            using (FileStream fs = File.OpenRead(strFile))
            using (BinaryReader br = new BinaryReader(fs))
            {
                byte[] rgMagic = new byte[6];
                for (int i = 0; i < rgMagic.Length; i++)
                {
                    rgMagic[i] = br.ReadByte();
                }

                if (rgMagic[0] != 0x93 || rgMagic[1] != 0x4E || rgMagic[2] != 0x55 || rgMagic[3] != 0x4D || rgMagic[4] != 0x50 || rgMagic[5] != 0x59)
                    throw new Exception("The file is not a valid Numpy file!");

                byte bMajor = br.ReadByte();
                byte bMinor = br.ReadByte();

                if (bMajor != 1 || bMinor != 0)
                    throw new Exception("The file is not a valid Numpy file!");

                byte bHeaderLen1 = br.ReadByte();
                byte bHeaderLen2 = br.ReadByte();
                int nHeaderLen = bHeaderLen2 << 8 | bHeaderLen1;

                byte[] rgHeader = new byte[nHeaderLen];
                for (int i = 0; i < rgHeader.Length; i++)
                {
                    rgHeader[i] = br.ReadByte();
                }
                string strHeader = Encoding.ASCII.GetString(rgHeader);

                bool bFortranOrder;
                int[] rgShape;
                Type dataType;
                Tuple<int,int> count = parseHeaderEx(strHeader, out bFortranOrder, out rgShape, out dataType, nMax);

                if (bFortranOrder)
                    throw new Exception("Currently the fortran ordering is not supported");

                Stopwatch sw = null;
                if (log != null)
                {
                    sw = new Stopwatch();
                    sw.Start();
                }

                ulong ulIdx = 0;
                ulong ulTotal = (ulong)count.Item1 * (ulong)count.Item2;
                List<float[]> rgData = new List<float[]>(count.Item1);
                for (int i = 0; i < count.Item1; i++)
                {
                    float[] rgItem = new float[count.Item2];
                    for (int j = 0; j < count.Item2; j++)
                    {
                        if (dataType == typeof(float))
                            rgItem[j] = br.ReadSingle();
                        else if (dataType == typeof(double))
                            rgItem[j] = (float)br.ReadDouble();
                        else if (dataType == typeof(int))
                            rgItem[j] = (float)br.ReadInt32();
                        else if (dataType == typeof(long))
                            rgItem[j] = (float)br.ReadInt64();
                        else if (dataType == typeof(bool))
                            rgItem[j] = (br.ReadBoolean()) ? 1 : 0;
                        else
                            throw new Exception("Unsupported data type!");
                        
                        if (log != null)
                        {
                            if (sw.Elapsed.TotalMilliseconds > 1000)
                            {
                                double dfPct = (double)ulIdx / (double)ulTotal;
                                string strOut = "Loading '" + strFile + "' at " + dfPct.ToString("P5") + "...";
                                log.WriteLine(strOut, true);
                                sw.Restart();
                            }
                        }
                        ulIdx++;
                    }

                    rgData.Add(rgItem);
                }

                return new Tuple<List<float[]>, int[]>(rgData, rgShape);
            }
        }

        private int parseHeader(string str, out bool bFortranOrder, out int[] rgShape, out Type dataType, int nMax = int.MaxValue)
        {
            int nCount = 1;
            List<int> rgShape1 = new List<int>();
            str = str.Trim('{', '}', ' ', '\n', ',');

            dataType = typeof(object);

            string strShape = null;
            string strTarget = "'shape':";
            int nPos = str.IndexOf(strTarget);
            if (nPos > 0)
            {
                strShape = str.Substring(nPos + strTarget.Length);
                str = str.Substring(0, nPos);

                nPos = strShape.IndexOf(')');
                str += strShape.Substring(nPos + 1);
                str = str.Trim(',', ' ');

                strShape = strShape.Substring(0, nPos);
                strShape = strShape.Trim(' ', '(', ')');
                string[] rgShapeStr = strShape.Split(',');

                for (int i=0; i<rgShapeStr.Length; i++)
                {
                    string strShape1 = rgShapeStr[i];
                    if (!string.IsNullOrEmpty(strShape1))
                    {
                        int nShape = int.Parse(strShape1);

                        if (i == 0 && nShape > nMax)
                            nShape = nMax;

                        rgShape1.Add(nShape);
                        nCount *= rgShape1[rgShape1.Count - 1];
                    }
                }
            }

            rgShape = rgShape1.ToArray();
            bFortranOrder = false;

            string[] rgstr = str.Split(',');
            foreach (string str1 in rgstr)
            {
                string[] rgstrKeyVal = str1.Split(':');
                if (rgstrKeyVal.Length != 2)
                    throw new Exception("Invalid header key value, '" + str1 + "'!");

                string strKey = rgstrKeyVal[0].Trim('\'', ' ');
                string strVal = rgstrKeyVal[1].Trim('\'', ' ');

                switch (strKey)
                {
                    case "descr":
                        if (strVal == "<f4")
                            dataType = typeof(float);
                        else if (strVal == "<f8")
                            dataType = typeof(double);
                        else if (strVal == "<i4")
                            dataType = typeof(int);
                        else if (strVal == "<i8")
                            dataType = typeof(long);
                        else if (strVal == "|b1")
                            dataType = typeof(bool);
                        else
                            throw new Exception("Unsupported data type '" + strVal + "', currenly only support '<f4'");
                        break;

                    case "fortran_order":
                        bFortranOrder = bool.Parse(strVal);
                        break;
                }
            }

            return nCount;
        }

        private static Tuple<int, int> parseHeaderEx(string str, out bool bFortranOrder, out int[] rgShape, out Type dataType, int nMax = int.MaxValue)
        {
            int nNum = 1;
            int nCount = 1;
            List<int> rgShape1 = new List<int>();
            str = str.Trim('{', '}', ' ', '\n', ',');

            dataType = typeof(object);

            string strShape = null;
            string strTarget = "'shape':";
            int nPos = str.IndexOf(strTarget);
            if (nPos > 0)
            {
                strShape = str.Substring(nPos + strTarget.Length);
                str = str.Substring(0, nPos);

                nPos = strShape.IndexOf(')');
                str += strShape.Substring(nPos + 1);
                str = str.Trim(',', ' ');

                strShape = strShape.Substring(0, nPos);
                strShape = strShape.Trim(' ', '(', ')');
                string[] rgShapeStr = strShape.Split(',');

                for (int i=0; i<rgShapeStr.Count(); i++)
                {
                    string strShape1 = rgShapeStr[i];
                    if (!string.IsNullOrEmpty(strShape1))
                    {
                        int nShape = int.Parse(strShape1);

                        if (i == 0 && nShape > nMax)
                            nShape = nMax;
                        
                        rgShape1.Add(nShape);

                        if (i == 0)
                            nNum = rgShape1[rgShape1.Count - 1];
                        else
                            nCount *= rgShape1[rgShape1.Count - 1];
                    }
                }
            }

            rgShape = rgShape1.ToArray();
            bFortranOrder = false;

            string[] rgstr = str.Split(',');
            foreach (string str1 in rgstr)
            {
                string[] rgstrKeyVal = str1.Split(':');
                if (rgstrKeyVal.Length != 2)
                    throw new Exception("Invalid header key value, '" + str1 + "'!");

                string strKey = rgstrKeyVal[0].Trim('\'', ' ');
                string strVal = rgstrKeyVal[1].Trim('\'', ' ');

                switch (strKey)
                {
                    case "descr":
                        if (strVal == "<f4")
                            dataType = typeof(float);
                        else if (strVal == "<f8")
                            dataType = typeof(double);
                        else if (strVal == "<i4")
                            dataType = typeof(int);
                        else if (strVal == "<i8")
                            dataType = typeof(long);
                        else if (strVal == "|b1")
                            dataType = typeof(bool);
                        else
                            throw new Exception("Unsupported data type '" + strVal + "', currenly only support '<f4'");
                        break;

                    case "fortran_order":
                        bFortranOrder = bool.Parse(strVal);
                        break;
                }
            }

            return new Tuple<int, int>(nNum, nCount);
        }

        /// <summary>
        /// MatMul blobA with blobB and place the result in this blob (e.g. this = matmul(A, B)).  All blobs are in row-major format.
        /// </summary>
        /// <param name="blobA">Specifies the first input with last 2 axes size of MxK row-major matrix (first axes must match blobB's)</param>
        /// <param name="blobB">Specifies the second input with last 2 axes size of KxN row-major matrix (first axes must match blobA's)</param>
        /// <param name="dfScale">Specifies the scale applied to blobB.</param>
        /// <param name="bReshape">Specifies to reshape this blob the the expected shape (default = false).</param>
        /// <param name="bTransA">Specifies to transpose A first.</param>
        /// <param name="bTransB">Specifies to transpose B first.</param>
        /// <param name="bADiff">Specifies to use the diff values in blobA, otherwise the data values are used (default = false).</param>
        /// <param name="bBDiff">Specifies to use the diff values in blobB, otherwise the data values are used (default = false).</param>
        /// <param name="bCDiff">Specifies to use the diff values in blobC, otherwise the data values are used (default = false).</param>
        public void MatMul(Blob<T> blobA, Blob<T> blobB, bool bReshape = false, bool bTransA = false, bool bTransB = false, double dfScale = 1.0, bool bADiff = false, bool bBDiff = false, bool bCDiff = false)
        {
            m_log.CHECK_EQ(blobA.num_axes, 4, "The blobA must have 4 axes!");
            m_log.CHECK_EQ(blobB.num_axes, 4, "The blobB must have 4 axes!");

            if (bADiff && blobA.gpu_diff == 0)
                m_log.FAIL("Blob A does not have a diff value!");
            if (bBDiff && blobB.gpu_diff == 0)
                m_log.FAIL("Blob B does not have a diff value!");

            for (int i = 0; i < blobA.num_axes - 2; i++)
            {
                m_log.CHECK_EQ(blobA.shape(i), blobB.shape(i), "Blob A and B must have the same shape at axis '" + i.ToString() + "'!");
            }

            if (bCDiff && gpu_diff == 0)
                m_log.FAIL("This blob does not have a diff value!");

            int nAxis = 2;
            uint nOuterCount = (uint)blobA.count(0, nAxis);
            int m = blobA.shape(2);
            int n = blobB.shape(3);
            int k = blobA.shape(3);

            // Reshape the resulting blob to shape (B,C,M,N)
            List<int> rgShape = Utility.Clone<int>(blobA.shape());
            rgShape[rgShape.Count - 1] = n;
            rgShape[rgShape.Count - 2] = m;
            
            if (bReshape)
                Reshape(rgShape);
            else
                m_log.CHECK(CompareShape(rgShape), "This (resulting) blob does not have the correct shape!  Expected shape = " + Utility.ToString<int>(rgShape));

            long hA = (bADiff) ? blobA.gpu_diff : blobA.gpu_data;
            long hB = (bBDiff) ? blobB.gpu_diff : blobB.gpu_data;
            long hC = (bCDiff) ? mutable_gpu_diff : mutable_gpu_data;

            m_cuda.matmul(nOuterCount, m, n, k, hA, hB, hC, dfScale, bTransA, bTransB);
        }

        /// <summary>
        /// Calculates and propagates the gradient for blobA and BlobB given the input gradient in this blob's diff values.  After
        /// this call, the blobA and blobB diff values are filled with their respective gradients.
        /// </summary>
        /// <param name="blobA">Specifies the blobA where blobA gradients are placed (in diff values).</param>
        /// <param name="blobB">Specifies the blobB where blobB gradients are placed (in diff values).</param>
        /// <param name="blobWork">Specifies a work blob.</param>
        /// <param name="dfScale">Specifies a scale to be applied to the diffs in this blob before the MatMul (default = 1.0).</param>
        /// <remarks>
        /// @see [PyGrad:functions.py](https://github.com/jaketae/pygrad/blob/master/pygrad/functions.py) by Jake Tae, 2020, GitHub:jaketae/pygrad
        /// </remarks>
        public void MatMulGrad(Blob<T> blobA, Blob<T> blobB, Blob<T> blobWork, double dfScale = 1.0)
        {
            if (dfScale != 1.0)
                scale_diff(dfScale);
            blobWork.CopyFromAndTransposeHeightWidth(blobB, false);
            blobA.MatMul(this, blobWork, false, false, false, 1, true, false, true);
            blobWork.CopyFromAndTransposeHeightWidth(blobA, false);
            blobB.MatMul(blobWork, this, false, false, false, 1, false, true, true);
        }
    }
}
