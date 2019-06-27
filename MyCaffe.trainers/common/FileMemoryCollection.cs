using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.trainers.common
{
    /// <summary>
    /// The FileMemoryCollection is used during debugging to load from and save to file.
    /// </summary>
    public class FileMemoryCollection : IMemoryCollection
    {
        MemoryCollection m_mem;
        int m_nSampleIdx = 0;
        string m_strFile;
        bool m_bPreLoaded = false;
        bool m_bSaveOnCleanup = false;
        double[] m_rgWts = null;
        int[] m_rgIdx = null;


        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="nMax">Specifies the maximum number of items.</param>
        /// <param name="bPreLoad">Pre-load the data from file.</param>
        /// <param name="bSaveOnCleanup">Save the memory items on cleanup.</param>
        /// <param name="strFile">Specifies the name of the file to load from or save to.</param>
        public FileMemoryCollection(int nMax, bool bPreLoad, bool bSaveOnCleanup, string strFile)
        {
            m_mem = new MemoryCollection(nMax);
            m_strFile = strFile;
            m_bPreLoaded = bPreLoad;
            m_bSaveOnCleanup = bSaveOnCleanup;

            if (bPreLoad)
                m_mem.Load(m_strFile);
        }

        /// <summary>
        /// Complete any final processing.
        /// </summary>
        public void CleanUp()
        {
            if (m_bSaveOnCleanup)
                m_mem.Save(m_strFile);
        }

        /// <summary>
        /// Returns the number of items in the collection.
        /// </summary>
        public int Count
        {
            get { return m_mem.Count; }
        }

        /// <summary>
        /// Add a new item to the collection.
        /// </summary>
        /// <param name="mi">Specifies the item to add.</param>
        public void Add(MemoryItem mi)
        {
            if (!m_bPreLoaded)
                m_mem.Add(mi);
        }

        /// <summary>
        /// Return a batch of items.
        /// </summary>
        /// <param name="random">Specifies the random number generator.</param>
        /// <param name="nCount">Specifies the number of items to sample.</param>
        /// <param name="dfBeta">Not used.</param>
        /// <returns>The array of items is returned.</returns>
        public MemoryCollection GetSamples(CryptoRandom random, int nCount, double dfBeta)
        {
            if (m_rgWts == null || m_rgWts.Length != nCount)
            {
                m_rgWts = new double[nCount];
                for (int i = 0; i < m_rgWts.Length; i++)
                {
                    m_rgWts[i] = 1.0;
                }
            }

            if (m_rgIdx == null || m_rgIdx.Length != nCount)
            {
                m_rgIdx = new int[nCount];
            }

            MemoryCollection mem = new MemoryCollection(nCount);

            if (m_bPreLoaded)
            {
                for (int i = 0; i < nCount; i++)
                {
                    mem.Add(m_mem[m_nSampleIdx]);
                    m_nSampleIdx++;

                    if (m_nSampleIdx == m_mem.Count)
                        m_nSampleIdx = 0;
                }
            }
            else
            {
                mem = m_mem.GetRandomSamples(random, nCount);
            }

            mem.Indexes = m_rgIdx;
            mem.Priorities = m_rgWts;

            return mem;
        }

        /// <summary>
        /// Update - does nothing.
        /// </summary>
        /// <param name="rgSamples">Specifies the list of samples.</param>
        public void Update(MemoryCollection rgSamples)
        {
        }
    }
}
