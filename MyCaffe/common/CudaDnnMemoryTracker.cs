﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace MyCaffe.common
{
    /// <summary>
    /// The CudaDnnMemoryTracker is used for diagnostics in that it helps estimate the
    /// amount of memory that a Net will use.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class CudaDnnMemoryTracker<T>
    {
        Dictionary<int, MemoryInfo> m_rgItems = new Dictionary<int, MemoryInfo>();
        bool m_bEnableMemoryTrace = false;

        /// <summary>
        /// The CudaDnnMemoryTracker constructor.
        /// </summary>
        /// <param name="bEnableMemoryTrace">Optionally, specifies to enable the memory trace (only supported in debug builds).</param>
        public CudaDnnMemoryTracker(bool bEnableMemoryTrace = false)
        {
            m_bEnableMemoryTrace = bEnableMemoryTrace;
        }

        /// <summary>
        /// Simulate a memory allocation.
        /// </summary>
        /// <param name="hKernel">Specifies the CudaDnn kernel that holds the allocated memory.</param>
        /// <param name="nDeviceID">Specifies the CudaDnn device ID on which the memory was allocated.</param>
        /// <param name="hMemory">Specifies the CudaDnn handle to the memory.</param>
        /// <param name="lSize">Specifies the size of the memory (in items).</param>
        /// <param name="bHalf">Specifies whether or not half memory is used.</param>
        /// <returns></returns>
        public long AllocMemory(long hKernel, int nDeviceID, long hMemory, ulong lSize, bool bHalf)
        {
            MemoryInfo mi = new MemoryInfo(hKernel, nDeviceID, hMemory, lSize, bHalf);
            string strKey = mi.ToKey();
            int nKeyHash = strKey.GetHashCode();

            if (m_rgItems.ContainsKey(nKeyHash))
                throw new Exception("Memory item '" + strKey + "' already exists!");

            m_rgItems.Add(nKeyHash, mi);

#if DEBUG
            if (m_bEnableMemoryTrace)   
                Trace.WriteLine("Memory Used: " + TotalMemoryUsedText);   
#endif

            return hMemory;
        }

        /// <summary>
        /// Simulate a memory free.
        /// </summary>
        /// <param name="hKernel">Specifies the CudaDnn kernel that holds the allocated memory.</param>
        /// <param name="nDeviceID">Specifies the CudaDnn device ID on which the memory was allocated.</param>
        /// <param name="hMemory">Specifies the CudaDnn handle to the memory.</param>
        public void FreeMemory(long hKernel, int nDeviceID, long hMemory)
        {
            string strKey = MemoryInfo.ToKey(nDeviceID, hKernel, hMemory);
            int nKeyHash = strKey.GetHashCode();

            if (!m_rgItems.ContainsKey(nKeyHash))
                Trace.WriteLine("WARNING: Memory item '" + strKey + "' does not exist!");
            else
                m_rgItems.Remove(nKeyHash);

#if DEBUG
            if (m_bEnableMemoryTrace)
                Trace.WriteLine("Memory Used: " + TotalMemoryUsedText);
#endif
        }

        /// <summary>
        /// Enable/disable the memory trace - this feature is only available in debug builds.
        /// </summary>
        public bool EnableMemoryTrace
        {
            get { return m_bEnableMemoryTrace; }
            set { m_bEnableMemoryTrace = value; }
        }

        /// <summary>
        /// Returns the total number of items allocated.
        /// </summary>
        public ulong TotalItemsAllocated
        {
            get
            {
                ulong lMem = 0;

                foreach (KeyValuePair<int, MemoryInfo> kv in m_rgItems)
                {
                    ulong ulBase = (ulong)((typeof(T) == typeof(float)) ? 4 : 8);
                    if (kv.Value.Half)
                        ulBase = 2;

                    lMem += kv.Value.Size * ulBase;
                }

                return lMem;
            }
        }

        /// <summary>
        /// Returns the total amount of memory used (in bytes).
        /// </summary>
        public ulong TotalMemoryUsed
        {
            get 
            {
                return TotalItemsAllocated;
            }
        }

        /// <summary>
        /// Returns a text string describing the total amount of memory used (in bytes).
        /// </summary>
        public string TotalMemoryUsedText
        {
            get 
            { 
                return (TotalMemoryUsed / 1000000).ToString("N0") + " MB"; 
            }
        }
    }

    class MemoryInfo /** @private */
    {
        long m_hKernel;
        int m_nDeviceID;
        long m_hMemory;
        ulong m_lSize;
        bool m_bHalf;

        public MemoryInfo(long hKernel, int nDeviceID, long hMemory, ulong lSize, bool bHalf)
        {
            m_hKernel = hKernel;
            m_nDeviceID = nDeviceID;
            m_hMemory = hMemory;
            m_lSize = lSize;
            m_bHalf = bHalf;
        }

        public long Kernel
        {
            get { return m_hKernel; }
        }

        public int DeviceID
        {
            get { return m_nDeviceID; }
        }

        public long Memory
        {
            get { return m_hMemory; }
        }

        public ulong Size
        {
            get { return m_lSize; }
        }

        public bool Half
        {
            get { return m_bHalf; }
        }

        public string ToKey()
        {
            return ToKey(m_nDeviceID, m_hKernel, m_hMemory);
        }

        public static string ToKey(int nDeviceID, long hKernel, long hMemory)
        {
            return nDeviceID.ToString() + "_" + hKernel.ToString() + "_" + hMemory.ToString();
        }

        public override string ToString()
        {
            return "ID:" + m_nDeviceID.ToString() + " K:" + m_hKernel.ToString() + " Mem:" + m_hMemory.ToString() + " Size: " + m_lSize.ToString("N0");
        }
    }
}
