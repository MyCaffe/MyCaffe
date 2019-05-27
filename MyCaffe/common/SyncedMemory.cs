using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyCaffe.basecode;

namespace MyCaffe.common
{
    /// <summary>
    /// The SyncedMemory manages the low-level connection between the GPU and host memory.
    /// </summary>
    /// <remarks>
    /// The GPU memory is represented by a handle into the memory look-up table managed by the low-level CudaDnn DLL.  The host memory
    /// is copied and stored in a local array of type 'T'.
    /// </remarks>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class SyncedMemory<T> : IDisposable 
    {
        Log m_log;
        CudaDnn<T> m_cuda;
        int m_nDeviceID = -1;
        long m_lCapacity = 0;
        long m_lCount = 0;
        long m_hGpuData = 0;
        T[] m_rgCpuData = null;
        bool m_bOwnData = true;
        bool m_bHalfSize = false;
        object m_tag = null;

        /// <summary>
        /// The SyncedMemory constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn connection to Cuda.</param>
        /// <param name="log">Specifies the Log for output.</param>
        /// <param name="lCapacity">Optionally, specifies the capacity of the SyncedMemory (in items).</param>
        /// <param name="tag">Optionally, specifies a tag used for debugging (the default = <i>null</i>).</param>
        /// <param name="bUseHalfSize">Optionally, specifies to use half size (FP16) for both data and diff.  This option is only available when using the <i>float</i> base type 'T'.</param>
        public SyncedMemory(CudaDnn<T> cuda, Log log, long lCapacity = 0, object tag = null, bool bUseHalfSize = false)
        {
            if (bUseHalfSize && typeof(T) != typeof(float))
            {
                bUseHalfSize = false;

                if (log != null)
                    log.WriteLine("Half size disabled for non 'float' basetypes!");
            }

            m_bHalfSize = bUseHalfSize;
            m_cuda = cuda;
            m_log = log;
            m_tag = tag;

            if (lCapacity > 0)
            {
                m_nDeviceID = m_cuda.GetDeviceID();
                m_hGpuData = m_cuda.AllocMemory(lCapacity);
                m_lCapacity = lCapacity;
                m_lCount = lCapacity;
            }
        }

        private void free()
        {
            if (m_hGpuData != 0)
            {
                check_device();
                if (m_bOwnData)
                    m_cuda.FreeMemory(m_hGpuData);
                else
                    m_cuda.FreeMemoryPointer(m_hGpuData);
            }
        }

        /// <summary>
        /// Releases all GPU and host resources used.
        /// </summary>
        public void Dispose()
        {
            free();
            m_hGpuData = 0;
            m_nDeviceID = -1;
            m_lCapacity = 0;
            m_lCount = 0;
            m_rgCpuData = null;
        }

        /// <summary>
        /// Allocate a number of items in GPU memory and save the handle.
        /// </summary>
        /// <param name="lCount">Specifies the number of items.</param>
        /// <param name="bUseHalfSize">Optionally, specifis to use half sized memory (default = false).  This only applies to the 'float' base type.</param>
        public void Allocate(long lCount, bool bUseHalfSize = false)
        {
            if (bUseHalfSize && typeof(T) != typeof(float))
            {
                bUseHalfSize = false;

                if (m_log != null)
                    m_log.WriteLine("Half size disabled for non 'float' basetypes!");
            }

            free();
            m_nDeviceID = m_cuda.GetDeviceID();
            m_bHalfSize = bUseHalfSize;
            m_hGpuData = m_cuda.AllocMemory(lCount, m_bHalfSize);
            m_lCapacity = lCount;
            m_lCount = 0;
            m_bOwnData = true;
            return;
        }

        /// <summary>
        /// Allocate a number of items and copy the given array into the memory on the GPU.
        /// </summary>
        /// <param name="rg">Specifies the array of items to copy.</param>
        /// <param name="bUseHalfSize">Optionally, specifis to use half sized memory (default = false).  This only applies to the 'float' base type.</param>
        public void Allocate(T[] rg, bool bUseHalfSize = false)
        {
            if (bUseHalfSize && typeof(T) != typeof(float))
            {
                bUseHalfSize = false;

                if (m_log != null)
                    m_log.WriteLine("Half size disabled for non 'float' basetypes!");
            }

            free();
            m_nDeviceID = m_cuda.GetDeviceID();
            m_bHalfSize = bUseHalfSize;
            m_hGpuData = m_cuda.AllocMemory(rg, 0, m_bHalfSize);
            m_lCapacity = rg.Length;
            m_lCount = rg.Length;
            m_bOwnData = true;
            check_device();

            return;
        }

        /// <summary>
        /// Set all items in the GPU memory up to the Count, to zero.
        /// </summary>
        public void Zero()
        {
            if (m_lCount > 0)
            {
                check_device();
                m_cuda.set((int)m_lCount, m_hGpuData, 0.0);
            }
        }

        /// <summary>
        /// Set all items in the GPU memory up to the Capacity, to zero.
        /// </summary>
        public void ZeroAll()
        {
            if (m_lCapacity > 0)
            {
                check_device();
                m_cuda.set((int)m_lCapacity, m_hGpuData, 0.0);
            }
        }

        /// <summary>
        /// Set all items up to Count to a given value.
        /// </summary>
        /// <param name="dfVal">Specifies the value.</param>
        public void Set(double dfVal)
        {
            if (m_lCount > 0)
            {
                check_device();
                m_cuda.set((int)m_lCount, m_hGpuData, dfVal);
            }
        }

        /// <summary>
        /// Set a specific item at a given index to a value.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <param name="fVal">Specifies the value.</param>
        public void SetAt(int nIdx, T fVal)
        {
            check_device();
            m_cuda.set((int)m_lCount, m_hGpuData, fVal, nIdx);
        }

        /// <summary>
        /// Return a value at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <returns>The value at the index is returned.</returns>
        public T GetAt(int nIdx)
        {
            check_device();
            T[] rg = m_cuda.get((int)m_lCount, m_hGpuData, nIdx);
            return rg[0];
        }

        /// <summary>
        /// Copy another SyncedMemory into this one.
        /// </summary>
        /// <param name="src">Specifies the SyncedMemory to copy.</param>
        /// <param name="hDstHostBuffer">Optionally, specifies a host buffer used to copy between kernels (default = 0, not used).</param>
        /// <returns>When used the dst host buffer handle is returned.</returns>
        public long Copy(SyncedMemory<T> src, long hDstHostBuffer = 0)
        {
            if (src == null)
            {
                m_lCount = 0;
                return hDstHostBuffer;
            }

            if (m_lCapacity < src.m_lCount)
                Allocate(src.m_lCount);

            m_lCount = src.m_lCount;

            if (m_lCount > 0)
            {
                if (m_cuda.KernelHandle == src.m_cuda.KernelHandle)
                {
                    check_device();
                    m_cuda.copy((int)m_lCount, src.m_hGpuData, m_hGpuData);
                }
                else
                {
                    if (hDstHostBuffer == 0)
                    {
                        hDstHostBuffer = m_cuda.AllocHostBuffer(m_lCount);
                    }
                    else
                    {
                        long lCount = m_cuda.GetHostBufferCapacity(hDstHostBuffer);
                        if (lCount < m_lCount)
                        {
                            m_cuda.FreeHostBuffer(hDstHostBuffer);
                            hDstHostBuffer = m_cuda.AllocHostBuffer(m_lCount);
                        }
                    }

                    src.m_cuda.KernelCopy((int)m_lCount, src.m_hGpuData, 0, m_cuda.KernelHandle, m_hGpuData, 0, hDstHostBuffer, m_cuda.KernelHandle);
                }
            }

            return hDstHostBuffer;
        }

        /// <summary>
        /// Copy this SyncedMemory.
        /// </summary>
        /// <returns>A new SynedMemory that is a copy of this one, is returned.</returns>
        public SyncedMemory<T> Clone()
        {
            SyncedMemory<T> dst = new SyncedMemory<T>(m_cuda, m_log, m_lCapacity, null, m_bHalfSize);

            if (m_lCount > 0)
                dst.Copy(this);

            return dst;
        }

        public bool HalfSize
        {
            get { return m_bHalfSize; }
        }

        /// <summary>
        /// Get/set data associated with the synced memory.
        /// </summary>
        public object Tag
        {
            get { return m_tag; }
            set { m_tag = value; }
        }

        /// <summary>
        /// Returns the Device ID on which the GPU memory of this SyncedMemory was allocated.
        /// </summary>
        public int DeviceID
        {
            get { return m_nDeviceID; }
        }

        /// <summary>
        /// Returns the total amount of GPU memory held by this SyncedMemory.
        /// </summary>
        public long Capacity
        {
            get { return m_lCapacity; }
        }

        /// <summary>
        /// Returns the current count of items in this SyncedMemory.  Note, the Count may be less than the Capacity.
        /// </summary>
        public long Count
        {
            get { return m_lCount; }
            set { m_lCount= value; }
        }

        /// <summary>
        /// Returns the handle to the GPU memory.
        /// </summary>
        public long gpu_data
        {
            get { return m_hGpuData; }
        }

        /// <summary>
        /// Copies a new Memory Pointer within the low-level CudaDnnDLL where a Memory Pointer 
        /// uses another already allocated block of GPU memory and just indexes into it.
        /// </summary>
        /// <param name="hData">Specifies a handle to the already allocated GPU memory that the new Memory Pointer will index into.</param>
        /// <param name="lCount">Specifies the number of items in this 'virtual memory'</param>
        /// <param name="lOffset">Specifies the offset into the GPU data where the Memory Pointer should start.</param>
        public void set_gpu_data(long hData, long lCount, long lOffset)
        {
            free();
            m_hGpuData = m_cuda.CreateMemoryPointer(hData, lOffset, lCount);
            m_lCapacity = lCount;
            m_lCount = lCount;
            m_bOwnData = false;
            check_device();
        }

        /// <summary>
        /// Returns the mutable handle to GPU data.
        /// </summary>
        /// <remarks>
        /// Note: This is the same as gpu_data, but is provided for compatibility and readability with the original C++ %Caffe code.
        /// </remarks>
        public long mutable_gpu_data
        {
            get
            {
                check_device();
                return m_hGpuData;
            }
//          set { m_hGpuData = value; }
        }

        /// <summary>
        /// Returns the data on the CPU that has already been transferred from GPU to CPU.
        /// </summary>
        public T[] cpu_data
        {
            get { return m_rgCpuData; }
        }

        /// <summary>
        /// Sets the array of host data on the GPU and re-allocates the GPU memory if needed.
        /// </summary>
        /// <param name="rgData">Specifies the host data to set.</param>
        /// <param name="nCount">Specifies the number of items in the host data to set, which may be less than the host data array length.</param>
        /// <param name="bSetCount">Optionally, specifies whether or not to set the count.  The count is always set when re-allocating the buffer.</param>
        public void SetData(T[] rgData, int nCount, bool bSetCount = true)
        {
            if (nCount == -1)
                nCount = rgData.Length;

            if (nCount > m_lCapacity || m_hGpuData == 0)
            {
                bSetCount = true;
                Allocate(nCount);
            }

            m_cuda.SetMemory(m_hGpuData, rgData, 0, nCount);

            if (bSetCount)
                m_lCount = nCount;

            check_device();
        }

        /// <summary>
        /// Get/set the mutable host data.
        /// </summary>
        /// <remarks>
        /// When setting the mutable host data, the data is copied to the GPU.  When get'ing the host data, the data is
        /// transferred from the GPU first and then returned.
        /// </remarks>
        public T[] mutable_cpu_data
        {
            get { return update_cpu_data(); }
            set
            {
                check_device();
                if (value.Length > m_lCapacity || m_hGpuData == 0)
                {
                    Allocate(value);
                }
                else
                {
                    m_cuda.SetMemory(m_hGpuData, value);
                    m_lCount = value.Length;
                    check_device();
                }
            }
        }

        /// <summary>
        /// Updates the host data by copying the GPU data to the host data.
        /// </summary>
        /// <param name="lCount">Optionally, specifies a count (less than Count) to transfer.</param>
        /// <returns>An array of the host data is returned.</returns>
        public T[] update_cpu_data(long lCount = -1)
        {
            if (lCount >= 0)
            {
                if (lCount > m_lCapacity)
                    throw new ArgumentOutOfRangeException();

                m_lCount = lCount;
            }

            check_device();

            if (m_lCount == 0)
                m_rgCpuData = new List<T>().ToArray();
            else
                m_rgCpuData = m_cuda.GetMemory(m_hGpuData, m_lCount);

            return m_rgCpuData;
        }

        /// <summary>
        /// This does not place the data on the GPU - call async_gpu_push() to move it to the GPU.
        /// </summary>
        /// <param name="rg">Specifies an array of host data.</param>
        public void set_cpu_data_locally(T[] rg)
        {
            m_rgCpuData = rg;
        }

        /// <summary>
        /// Pushes the host data, previously set with set_cpu_data_locally(), to the GPU.
        /// </summary>
        /// <remarks>
        /// Note, if necessary, this function re-allocates the GPU memory.
        /// </remarks>
        /// <param name="hStream"></param>
        /// <param name="rg"></param>
        public void async_gpu_push(long hStream, T[] rg)
        {
            check_device();
            if (m_hGpuData == 0)
            {
                m_hGpuData = m_cuda.AllocMemory(rg, hStream);
                m_lCapacity = rg.Length;
            }
            else
            {
                m_cuda.SetMemory(m_hGpuData, rg, hStream);
            }

            m_lCount = rg.Length;
            check_device();
        }

        private void check_device()
        {
#if DEBUG
            if (m_lCount > 0)
            {
                int nDeviceId = m_cuda.GetDeviceID();
                m_log.CHECK_EQ(nDeviceId, m_nDeviceID, "The current device DOES'NT match the device for which the memory was allocated!");
            }
#endif
        }
    }
}
