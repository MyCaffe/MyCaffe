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

        /// <summary>
        /// Defines the maximum number of Axes supported by the Blob.
        /// </summary>
        public const int MAX_BLOB_AXES = 32;

        /// <summary>
        /// Defines the tpe of data held by a given Blob.
        /// </summary>
        public enum BLOB_TYPE
        {
            /// <summary>
            /// The Blob holds Data.
            /// </summary>
            DATA,
            /// <summary>
            /// The Blob holds an inner product weight.
            /// </summary>
            IP_WEIGHT,
            /// <summary>
            /// The Blob holds Loss Data.
            /// </summary>
            LOSS,
            /// <summary>
            /// The Blob holds Accuracy Data.
            /// </summary>
            ACCURACY
        }

        /// <summary>
        /// The Blob constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance used to communidate with Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="bIncludeDiff">Optionally, specifies whether or not to include (and allocate) the Diff data.</param>
        public Blob(CudaDnn<T> cuda, Log log, bool bIncludeDiff = true)
        {
            m_tZero = (T)Convert.ChangeType(0, typeof(T));
            m_tMinusOne = (T)Convert.ChangeType(-1, typeof(T));
            m_bIncludeDiff = bIncludeDiff;
            m_cuda = cuda;
            m_log = log;
            m_shape = new SyncedMemory<T>(m_cuda, m_log);
            m_data = new SyncedMemory<T>(m_cuda, m_log);

            if (m_bIncludeDiff)
                m_diff = new SyncedMemory<T>(m_cuda, m_log);
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
        public Blob(CudaDnn<T> cuda, Log log, int nNum, int nChannels, int nHeight, int nWidth, bool bIncludeDiff = true)
            : this(cuda, log, bIncludeDiff)
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
        public Blob(CudaDnn<T> cuda, Log log, List<int> rgShape, bool bIncludeDiff = true)
            : this(cuda, log, bIncludeDiff)
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
        public Blob(CudaDnn<T> cuda, Log log, Blob<T> b)
            : this(cuda, log, (b.m_diff != null) ? true : false)
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
        public Blob(CudaDnn<T> cuda, Log log, SimpleDatum d, bool bCopyData = false, bool bIncludeDiff = true)
            : this(cuda, log, bIncludeDiff)
        {
            SetData(d, true, bCopyData);
        }

        /// <summary>
        /// The Blob constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn instance used to communidate with Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="bp">Specifies the BlobProto used to load the Blob.</param>
        public Blob(CudaDnn<T> cuda, Log log, BlobProto bp)
            : this(cuda, log)
        {
            FromProto(bp);
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
        public void Reshape(int nNum, int nChannels, int nHeight, int nWidth)
        {
            Reshape(new List<int>() { nNum, nChannels, nHeight, nWidth });
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
        public void Reshape(List<int> rgShape)
        {
            m_log.CHECK_LE(rgShape.Count, MAX_BLOB_AXES, "The number of axes cannot exceed " + MAX_BLOB_AXES.ToString());
            m_nCount = 1;

            m_rgShape = new List<int>();

            if (m_shape == null)
                m_shape = new SyncedMemory<T>(m_cuda, m_log, rgShape.Count);
            else if (m_shape.Count < rgShape.Count)
                m_shape.Allocate(rgShape.Count);
            else if (m_shape.Count != rgShape.Count)
                m_shape.ZeroAll();

            if (rgShape.Count > 0)
            {
                T[] rgShapeData = m_shape.cpu_data;

                if (rgShapeData == null || rgShapeData.Length != rgShape.Count)
                {
                    rgShapeData = m_shape.update_cpu_data();

                    if (rgShapeData == null || rgShapeData.Length != rgShape.Count)
                        rgShapeData = new T[rgShape.Count];
                }

                bool bDirty = false;

                for (int i = 0; i < rgShape.Count; i++)
                {
                    m_log.CHECK_GE(rgShape[i], 0, "The shape value at " + i.ToString() + " must be <= 0.");

                    if (m_nCount != 0)
                        m_log.CHECK_LE(rgShape[i], int.MaxValue / m_nCount, "The blob size exceeds int.MaxValue!");

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

            if (m_nCount > m_nCapacity)
            {
                if (m_data != null)
                    m_data.Dispose();

                if (m_diff != null)
                    m_diff.Dispose();

                m_nCapacity = m_nCount;

                if (m_data == null)
                    m_data = new SyncedMemory<T>(m_cuda, m_log, m_nCapacity);
                else
                    m_data.Allocate(m_nCapacity);

                if (m_bIncludeDiff)
                {
                    if (m_diff == null)
                        m_diff = new SyncedMemory<T>(m_cuda, m_log, m_nCapacity);
                    else
                        m_diff.Allocate(m_nCapacity);
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
        public void Reshape(BlobShape shape)
        {
            m_log.CHECK_LE(shape.dim.Count, MAX_BLOB_AXES, "The shape dimension must be less than " + MAX_BLOB_AXES.ToString());
            Reshape(shape.dim);
        }

        /// <summary>
        /// Reshape this Blob to have the same shape as another Blob.
        /// </summary>
        /// <param name="b">Specifies the other Blob.</param>
        public void ReshapeLike(Blob<T> b)
        {
            Reshape(b.shape());
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

            int nCount = 1;

            for (int i = nStartIdx; i < nEndIdx; i++)
            {
                nCount *= shape(i);
            }

            return nCount;
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
        /// <returns></returns>
        public int CanonicalAxisIndex(int nIdx)
        {
            m_log.CHECK_GE(nIdx, -num_axes, "The axis " + nIdx.ToString() + " out of range for " + num_axes.ToString() + " -D Blob with shape " + shape_string);
            m_log.CHECK_LT(nIdx, num_axes, "The axis " + nIdx.ToString() + " out of range for " + num_axes.ToString() + " -D Blob with shape " + shape_string);

            if (nIdx < 0)
                return nIdx + num_axes;
            else
                return nIdx;
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
            m_log.CHECK_GE(n, 0, "n must be >= 0.");
            m_log.CHECK_LE(n, num, "n must be <= num.");
            m_log.CHECK_GE(channels, 0, "channels must be >= 0.");
            m_log.CHECK_LE(c, channels, "c must be <= channels.");
            m_log.CHECK_GE(height, 0, "height must be >= 0.");
            m_log.CHECK_LE(h, height, "w must be <= height.");
            m_log.CHECK_GE(width, 0, "width must be >= 0.");
            m_log.CHECK_LE(w, width, "w must be <= width.");
            return ((n * channels + c) * height + h) * width + w;
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
        /// <param name="bCopyDiff">If false, copy the data; if true, copy the diff.</param>
        /// <param name="bReshape">If false, require this Blob to be pre-shaped to the shape
        /// of other (and die otherwise); If true, Reshape this Blob to other's shape if
        /// necessary.</param>
        /// <param name="hDstHostBuffer">Optionally, specifies the host buffer of the destination.</param>
        /// <returns>
        /// When used, the host buffer handle is returned.
        /// </returns>
        public long CopyFrom(Blob<T> src, bool bCopyDiff = false, bool bReshape = false, long hDstHostBuffer = 0)
        {
            if (src.count() != m_nCount || !Utility.Compare<int>(src.m_rgShape, m_rgShape))
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
        /// Copy from a source Blob and transpose the height and width of the copy.
        /// </summary>
        /// <param name="blobSrc">The Blob to copy from.</param>
        /// <param name="bCopyDiff">If false, copy the data; if true, copy the diff.</param>
        public void CopyFromAndTransposeHeightWidth(Blob<T> blobSrc, bool bCopyDiff = false)
        {
            m_log.CHECK_EQ(blobSrc.num_axes, 4, "Currently, Blobs only support transposing 4 axis tensors.");

            ReshapeLike(blobSrc);

            SyncedMemory<T> dst = (bCopyDiff) ? m_diff : m_data;
            SyncedMemory<T> src = (bCopyDiff) ? blobSrc.m_diff : blobSrc.m_data;

            int nN = num;
            int nC = channels;
            int nH = height;
            int nW = width;

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
            return m_data.update_cpu_data();
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

            return m_diff.update_cpu_data();
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
        /// Update is called to apply the diff errors to the data.  WHen !bIncludeDiff, no diff is applied.
        /// </remarks>
        public void Update()
        {
            if (!m_bIncludeDiff)
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
        /// Set the data to point to the data of the other blob -- useful in Layers which
        /// simply perform a copy in their forward pass.
        /// </summary>
        /// <param name="b"></param>
        public void ShareData(Blob<T> b)
        {
            m_log.CHECK_EQ(m_nCount, b.count(), "The blob counts are not the same!");

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
            m_log.CHECK_EQ(m_nCount, b.count(), "The blob counts are not the same!");

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
        /// <param name="dfVal">Specifies the value to set.</param>
        /// <param name="nIdx">Optionally, specifies the index of the item to set.</param>
        public void SetData(double dfVal, int nIdx = -1)
        {
            if (mutable_gpu_data == 0)
                return;

            m_cuda.set(count(), mutable_gpu_data, dfVal, nIdx);
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
            m_data.set_cpu_data_locally(rg);
        }

        /// <summary>
        /// Asynchronously pushes the CPU data, previously set with SetCPUData, to the GPU.
        /// </summary>
        /// <param name="hStream">Specifies a handle to the Cuda Stream to use for synchronization.</param>
        public void AsyncGpuPush(long hStream)
        {
            if (m_data.cpu_data == null)
                throw new Exception("There is no CPU data to push to the GPU!");

            m_data.async_gpu_push(hStream, m_data.cpu_data);
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
            double[] rgData = m_cuda.GetMemoryDouble(gpu_data, count());
            return new Datum(true, channels, width, height, -1, DateTime.MinValue, null, new List<double>(rgData), 0, false, -1);
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
            Blob<T> b = new Blob<T>(m_cuda, m_log, this);

            if (m_diff != null)
                b.m_diff.Copy(m_diff);

            b.m_data.Copy(m_data);
            b.Name = Name;

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
            return m_strName + " (" + shape_string + ")";
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
                if (count() == 0 || gpu_data == 0)
                    return 0;

                return m_cuda.min(count(), gpu_data);
            }
        }

        /// <summary>
        /// Returns the maximum value in the data of the Blob.
        /// </summary>
        public double max_data
        {
            get
            {
                if (count() == 0 || gpu_data == 0)
                    return 0;

                return m_cuda.max(count(), gpu_data);
            }
        }

        /// <summary>
        /// Returns the minimum value in the diff of the Blob.
        /// </summary>
        public double min_diff
        {
            get
            {
                if (count() == 0 || gpu_diff == 0)
                    return 0;

                return m_cuda.min(count(), gpu_diff);
            }
        }

        /// <summary>
        /// Returns the maximum value in the diff of the Blob.
        /// </summary>
        public double max_diff
        {
            get
            {
                if (count() == 0 || gpu_diff == 0)
                    return 0;

                return m_cuda.max(count(), gpu_diff);
            }
        }

        /// <summary>
        /// Returns the minimum and maximum values in the data of the Blob.
        /// </summary>
        /// <param name="work">Specifies a workspace used to optimize the query.</param>
        /// <param name="bDetectNans">Optionally, specifies whether or not to detect Nan's and Infinity values.</param>
        /// <returns>A tuple containing the 'min', 'max' and optionally 'number of nans' and 'number of infinity' is returned for the data.</returns>
        public Tuple<double, double, double, double> minmax_data(Blob<T> work, bool bDetectNans = false)
        {
            if (count() == 0 || gpu_data == 0)
                return new Tuple<double, double, double, double>(0, 0, 0, 0);

            work.ReshapeLike(this);

            return m_cuda.minmax(count(), gpu_data, work.mutable_gpu_data, work.mutable_gpu_diff, bDetectNans);
        }

        /// <summary>
        /// Returns the minimum and maximum values in the diff of the Blob.
        /// </summary>
        /// <param name="work">Specifies a workspace used to optimize the query.</param>
        /// <param name="bDetectNans">Optionally, specifies whether or not to detect Nan's and Infinity values.</param>
        /// <returns>A tuple containing the 'min', 'max' and optionally 'number of nans' and 'number of infinity' is returned for the data.</returns>
        public Tuple<double, double, double, double> minmax_diff(Blob<T> work, bool bDetectNans = false)
        {
            if (count() == 0 || gpu_diff == 0)
                return new Tuple<double, double, double, double>(0, 0, 0, 0);

            work.ReshapeLike(this);

            return m_cuda.minmax(count(), gpu_diff, work.mutable_gpu_data, work.mutable_gpu_diff, bDetectNans);
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
        /// <returns>The mean is returned.</returns>
        public double mean(float[] rgDf = null)
        {
            double dfSum = 0;

            if (rgDf == null)
                rgDf = Utility.ConvertVecF<T>(update_cpu_data());

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
    }
}
