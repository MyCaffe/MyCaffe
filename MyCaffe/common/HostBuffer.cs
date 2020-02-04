using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The HostBuffer helps manage host memory, often used when implementing CPU versions of a function or layer.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class HostBuffer<T> : IDisposable
    {
        CudaDnn<T> m_cuda;
        long m_hBuffer;
        long m_lCapacity;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="cuda">Specifies the CudaDnn link to all low-level functionality.</param>
        public HostBuffer(CudaDnn<T> cuda)
        {
            m_cuda = cuda;
            m_hBuffer = 0;
            m_lCapacity = 0;
        }

        /// <summary>
        /// Release all resources used.
        /// </summary>
        public void Dispose()
        {
            Free();
        }

        /// <summary>
        /// Returns a handle to the host buffer.
        /// </summary>
        public long Handle
        {
            get { return m_hBuffer; }
        }

        /// <summary>
        /// Returns the capacity of the host buffer.
        /// </summary>
        public long Capacity
        {
            get { return m_lCapacity; }
        }

        /// <summary>
        /// Free the host buffer.
        /// </summary>
        public void Free()
        {
            if (m_hBuffer != 0)
            {
                m_cuda.FreeHostBuffer(m_hBuffer);
                m_hBuffer = 0;
                m_lCapacity = 0;
            }
        }

        /// <summary>
        /// Copy the gpu data from the blob to the host buffer.
        /// </summary>
        /// <param name="b">Specifies the blob to copy from.</param>
        /// <param name="bFromDiff">Optionally, specifies to topy from the diff (default = false, which copies from the data).</param>
        public void CopyFrom(Blob<T> b, bool bFromDiff = false)
        {
            CopyFromGpu(b.count(), (bFromDiff) ? b.gpu_diff : b.gpu_data);
        }

        /// <summary>
        /// Copy from the host buffer to the gpu data of the blob.
        /// </summary>
        /// <param name="b">Specifies the blob to copy to.</param>
        /// <param name="bToDiff">Optionally, specifies to copy to the diff (default = false, which copies to the data).</param>
        public void CopyTo(Blob<T> b, bool bToDiff = false)
        {
            CopyToGpu(b.count(), (bToDiff) ? b.mutable_gpu_diff : b.mutable_gpu_data);
        }

        /// <summary>
        /// Copy data from the GPU into the host buffer making sure to grow the host buffer capacity if needed.
        /// </summary>
        /// <param name="nCount">Specifies the number of items to copy.</param>
        /// <param name="hGpu">Specifies the source GPU data to copy.</param>
        public void CopyFromGpu(int nCount, long hGpu)
        {
            if (nCount > m_lCapacity)
            {
                Free();
                m_hBuffer = m_cuda.AllocHostBuffer(nCount);
                m_lCapacity = nCount;
            }

            m_cuda.CopyDeviceToHost(nCount, hGpu, m_hBuffer);
        }

        /// <summary>
        /// Copy data from the host buffer into the GPU memory.
        /// </summary>
        /// <param name="nCount">Specifies the number of items to copy.</param>
        /// <param name="hGpu">Specifies the destination GPU memory where the data is to be copied.</param>
        public void CopyToGpu(int nCount, long hGpu)
        {
            m_cuda.CopyHostToDevice(nCount, m_hBuffer, hGpu);
        }

        /// <summary>
        /// Returns the host buffer data as an array of the base type.
        /// </summary>
        /// <returns>The data is returned as an array of the base type.</returns>
        public T[] GetHostData()
        {
            return m_cuda.GetHostMemory(m_hBuffer);
        }

        /// <summary>
        /// Returns the host buffer data as an array of doubles.
        /// </summary>
        /// <returns>The data is returned as an array of doubles.</returns>
        public double[] GetHostDataAsDouble()
        {
            return m_cuda.GetHostMemoryDouble(m_hBuffer);
        }

        /// <summary>
        /// Returns the host buffer data as an array of floats.
        /// </summary>
        /// <returns>The data is returned as an array of floats.</returns>
        public float[] GetHostDataAsFloat()
        {
            return m_cuda.GetHostMemoryFloat(m_hBuffer);
        }

        /// <summary>
        /// Set the host buffer data, making to expand the capcity if needed.
        /// </summary>
        /// <param name="rgSrc">Specifies the source data to set in the host buffer.</param>
        public void SetHostData(T[] rgSrc)
        {
            if (m_lCapacity < rgSrc.Length)
            {
                Free();
                m_hBuffer = m_cuda.AllocHostBuffer(rgSrc.Length);
            }

            m_cuda.SetHostMemory(m_hBuffer, rgSrc);
        }
    }
}
