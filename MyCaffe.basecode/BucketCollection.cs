using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The Bucket class contains the information describing a single range of values within a BucketCollection.
    /// </summary>
    public class Bucket
    {
        double m_fMin;
        double m_fMax;
        double m_fSum;
        int m_nCount;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fMin">The minimum value in the range.</param>
        /// <param name="fMax">The maximum value in the range.</param>
        public Bucket(double fMin, double fMax)
        {
            m_fMin = fMin;
            m_fMax = fMax;
        }

        /// <summary>
        /// Tests to see if the Bucket range contains the value.
        /// </summary>
        /// <param name="fVal">Specifies the value to test.</param>
        /// <returns>Returns 0 if the Bucket contains the value, -1 if the value is less than the Bucket range and 1 if the value is greater.</returns>
        public int Contains(double fVal)
        {
            if (fVal < m_fMin)
                return -1;

            if (fVal >= m_fMax)
                return 1;

            return 0;
        }

        /// <summary>
        /// Attempts to add a new value to the Bucket.
        /// </summary>
        /// <param name="fVal">Specifies the value to add.</param>
        /// <param name="bForce">Optionally, forces adding the value to the Bucket.</param>
        /// <returns>Returns 0 if the value falls within the Buckets range and is added, -1 if the value is less than the Bucket range and 1 if the value is greater.</returns>
        public int Add(double fVal, bool bForce = false)
        {
            if (!bForce)
            {
                int nVal = Contains(fVal);
                if (nVal != 0)
                    return nVal;
            }

            m_nCount++;
            m_fSum += fVal;

            return 0;
        }

        /// <summary>
        /// Returns the number of items added to the Bucket.
        /// </summary>
        public int Count
        {
            get { return m_nCount; }
        }

        /// <summary>
        /// Returns the average value of all values added to the Bucket.
        /// </summary>
        public double Average
        {
            get { return m_fSum / m_nCount; }
        }

        /// <summary>
        /// Returns the bucket minimum value.
        /// </summary>
        public double Minimum
        {
            get { return m_fMin; }
        }

        /// <summary>
        /// Returns the bucket maximum value.
        /// </summary>
        public double Maximum
        {
            get { return m_fMax; }
        }

        /// <summary>
        /// Returns the bucket midpoint.
        /// </summary>
        public double MidPoint
        {
            get { return m_fMin + (m_fMax - m_fMin) / 2.0; }
        }

        /// <summary>
        /// Returns a string representation of the Bucket.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            return "[" + m_fMin.ToString() + "," + m_fMax.ToString() + "]-> " + m_nCount.ToString("N0");
        }

        /// <summary>
        /// Save the Bucket to a BinaryWriter.
        /// </summary>
        /// <param name="bw">Specifies the BinaryWriter.</param>
        public void Save(BinaryWriter bw)
        {
            bw.Write(m_nCount);
            bw.Write(m_fSum);
            bw.Write(m_fMin);
            bw.Write(m_fMax);
        }

        /// <summary>
        /// Load a Bucket from a BinaryReader.
        /// </summary>
        /// <param name="br">Specifies the BinaryReader.</param>
        /// <returns>The newly loaded Bucket is returned.</returns>
        public static Bucket Load(BinaryReader br)
        {
            int nCount = br.ReadInt32();
            double dfSum = br.ReadDouble();
            double dfMin = br.ReadDouble();
            double dfMax = br.ReadDouble();

            Bucket b = new Bucket(dfMin, dfMax);
            b.m_nCount = nCount;
            b.m_fSum = dfSum;

            return b;
        }
    }

    /// <summary>
    /// The BucketCollection contains a set of Buckets.
    /// </summary>
    public class BucketCollection : IEnumerable<Bucket>
    {
        List<Bucket> m_rgBuckets = new List<Bucket>();
        bool m_bIsDataReal = false;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fMin">Specifies the overall minimum of all Buckets.</param>
        /// <param name="fMax">Specifies the overall maximum of all Buckets.</param>
        /// <param name="nCount">Specifies the number of Buckets to use.</param>
        public BucketCollection(double fMin, double fMax, int nCount)
        {
            double fRange = fMax - fMin;
            double fStep = fRange / (double)nCount;
            double fVal = fMin;

            for (int i = 0; i < nCount; i++)
            {
                double dfMax = (i == nCount - 1) ? fMax : Math.Round(fVal + fStep, 9);

                m_rgBuckets.Add(new Bucket(fVal, dfMax));
                fVal = dfMax;
            }

            m_bIsDataReal = true;
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="rgVocab">Specifies an array of vocabulary values to add into Buckets.</param>
        public BucketCollection(List<int> rgVocab)
        {
            rgVocab.Sort();

            for (int i = 0; i < rgVocab.Count; i++)
            {
                int nVal = rgVocab[i];
                Bucket b = new Bucket(nVal, nVal + 1);
                b.Add(nVal);
                m_rgBuckets.Add(b);
            }

            m_bIsDataReal = false;
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="bIsReal">Specifies that the Buckets are to hold Real values.</param>
        public BucketCollection(bool bIsReal)
        {
            m_bIsDataReal = bIsReal;
        }

        /// <summary>
        /// Returns whehter or not the Buckets hold Real values or not.
        /// </summary>
        public bool IsDataReal
        {
            get { return m_bIsDataReal; }
        }

        /// <summary>
        /// Returns the number of Buckets.
        /// </summary>
        public int Count
        {
            get { return m_rgBuckets.Count; }
        }

        /// <summary>
        /// Returns the bucket at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the bucket to retrieve.</param>
        /// <returns>The bucket at the index is returned.</returns>
        public Bucket this[int nIdx]
        {
            get { return m_rgBuckets[nIdx]; }
        }

        /// <summary>
        /// Returns the bucket with the highest count.
        /// </summary>
        /// <returns>The bucket with the highest count is returned.</returns>
        public Bucket GetBucketWithMaxCount()
        {
            int nMax = 0;
            int nMaxIdx = -1;

            for (int i = 0; i < m_rgBuckets.Count; i++)
            {
                if (nMax < m_rgBuckets[i].Count)
                {
                    nMax = m_rgBuckets[i].Count;
                    nMaxIdx = i;
                }
            }

            if (nMaxIdx == -1)
                return null;

            return m_rgBuckets[nMaxIdx];
        }

        /// <summary>
        /// Finds the correct Bucket and adds the value to it.
        /// </summary>
        /// <param name="fVal">Specifies the value to add.</param>
        /// <returns>The index of the bucket for which the value was added is returned.</returns>
        public int Add(double fVal)
        {
            for (int i = 0; i < m_rgBuckets.Count; i++)
            {
                int nVal = m_rgBuckets[i].Add(fVal);
                if (nVal == 0)
                    return i;

                if (nVal < 0 && i == 0)
                {
                    m_rgBuckets[i].Add(fVal, true);
                    return i;
                }

                if (nVal == 1 && i == m_rgBuckets.Count - 1)
                {
                    m_rgBuckets[i].Add(fVal, true);
                    return i;
                }
            }

            throw new Exception("Failed to find a bucket!");
        }

        /// <summary>
        /// Finds the Bucket associated with the value and returns the Bucket's average value.
        /// </summary>
        /// <param name="fVal">Specifies the value to find.</param>
        /// <returns>The Bucket average is returned.</returns>
        public double Translate(double fVal)
        {
            for (int i = 0; i < m_rgBuckets.Count; i++)
            {
                if (m_rgBuckets[i].Contains(fVal) == 0)
                    return m_rgBuckets[i].Average;
            }

            return m_rgBuckets[m_rgBuckets.Count - 1].Average;
        }

        /// <summary>
        /// Finds the index of the Bucket containing the value.
        /// </summary>
        /// <param name="dfVal">Specifies the value to look for.</param>
        /// <returns>The Bucket index is returned.</returns>
        public int FindIndex(double dfVal)
        {
            for (int i = 0; i < m_rgBuckets.Count; i++)
            {
                if (m_rgBuckets[i].Contains(dfVal) == 0)
                    return i;
            }

            return m_rgBuckets.Count - 1;
        }

        /// <summary>
        /// Returns the average of the Bucket at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <param name="bUseMidPoint">Optionally specifies to use the bucket midpoint instead of the average.</param>
        /// <returns>The Bucket average is returned.</returns>
        public double GetValueAt(int nIdx, bool bUseMidPoint = false)
        {
            if (bUseMidPoint)
                return m_rgBuckets[nIdx].MidPoint;
            else
                return m_rgBuckets[nIdx].Average;
        }

        /// <summary>
        /// The Bucketize method adds all values within a SimpleDatum to a new BucketCollection.
        /// </summary>
        /// <param name="strName">Specifies the name to use when writing status information.</param>
        /// <param name="nBucketCount">Specifies the number of Buckets to use.</param>
        /// <param name="sd">Specifies the SimpleDatum containing the data to add.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel the bucketizing process.</param>
        /// <param name="dfMin">Optionally, specifies a overall minimum to use in the BucketCollection, when missing this is calculated from the SimpleDatum.</param>
        /// <param name="dfMax">Optionally, specifies a overall maximum to use in the BucketCollection, when missing this is calculated from the SimpleDatum.</param>
        /// <returns></returns>
        public static BucketCollection Bucketize(string strName, int nBucketCount, SimpleDatum sd, Log log, CancelEvent evtCancel, double? dfMin = null, double? dfMax = null)
        {
            int nIdx = 0;
            int nChannels = sd.Channels;
            int nCount = sd.ItemCount / nChannels;
            int nItemCount = sd.ItemCount;
            int nOffset = 0;
            Stopwatch sw = new Stopwatch();

            sw.Start();

            // Calculate the min/max values if not already specified.
            if (!dfMin.HasValue || !dfMax.HasValue)
            {
                dfMin = double.MaxValue;
                dfMax = -double.MaxValue;

                for (int i = 0; i < nChannels; i++)
                {
                    for (int j = 0; j < nCount; j++)
                    {
                        double dfVal = sd.RealData[nOffset + j];
                        dfMin = Math.Min(dfMin.Value, dfVal);
                        dfMax = Math.Max(dfMax.Value, dfVal);
                        nIdx++;

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            if (evtCancel != null && evtCancel.WaitOne(0))
                                return null;

                            double dfPct = (double)nIdx / (double)nItemCount;
                            log.WriteLine("Calculating min/max at " + dfPct.ToString("P") + "...");
                            sw.Restart();
                        }
                    }

                    nOffset += nCount;
                }
            }

            BucketCollection col = new BucketCollection(dfMin.Value, dfMax.Value, nBucketCount);

            nIdx = 0;
            nOffset = 0;
            for (int i = 0; i < nChannels; i++)
            {
                for (int j = 0; j < nCount; j++)
                {
                    double dfVal = sd.RealData[nOffset + j];
                    col.Add(dfVal);
                    nIdx++;

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        if (evtCancel != null && evtCancel.WaitOne(0))
                            return null;

                        double dfPct = (double)nIdx / (double)nItemCount;
                        log.WriteLine(strName + " at " + dfPct.ToString("P") + "...");
                        sw.Restart();
                    }
                }

                nOffset += nCount;
            }

            return col;
        }

        /// <summary>
        /// The UnBucketize method converts all Data received into their respective Bucket average values.
        /// </summary>
        /// <param name="strName">Specifies the name to use when writing status information.</param>
        /// <param name="rgrgData">Specifies the data to unbucketize.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the CancelEvent used to cancel the unbucketizing process.</param>
        /// <returns>On success, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool UnBucketize(string strName, List<double[]> rgrgData, Log log, CancelEvent evtCancel)
        {
            int nIdx = 0;
            int nItemCount = rgrgData.Count * rgrgData[0].Length;
            Stopwatch sw = new Stopwatch();

            sw.Start();

            for (int i = 0; i < rgrgData.Count; i++)
            {
                for (int j = 0; j < rgrgData[i].Length; j++)
                {
                    double dfVal = rgrgData[i][j];
                    double dfNewVal = Translate(dfVal);
                    rgrgData[i][j] = dfNewVal;

                    if (evtCancel != null && evtCancel.WaitOne(0))
                        return false;

                    if (sw.Elapsed.TotalMilliseconds > 1000)
                    {
                        double dfPct = (double)nIdx / (double)nItemCount;
                        log.WriteLine(strName + " at " + dfPct.ToString("P") + "...");
                        sw.Restart();
                    }

                    nIdx++;
                }
            }

            return true;
        }

        /// <summary>
        /// Converts the BucketCollection into a byte stream.
        /// </summary>
        /// <returns>The byte stream is returned.</returns>
        public byte[] ToByteStream()
        {
            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                bw.Write(m_bIsDataReal);
                bw.Write(m_rgBuckets.Count);

                for (int i = 0; i < m_rgBuckets.Count; i++)
                {
                    m_rgBuckets[i].Save(bw);
                }

                return ms.ToArray();
            }
        }

        /// <summary>
        /// Converts a byte stream into a BucketCollection.
        /// </summary>
        /// <param name="rg">Specifies the byte stream.</param>
        /// <returns>The new BucketCollection is returned.</returns>
        public static BucketCollection FromByteStream(byte[] rg)
        {
            using (MemoryStream ms = new MemoryStream())
            using (BinaryReader br = new BinaryReader(ms))
            {
                bool bIsReal = br.ReadBoolean();
                BucketCollection col = new BucketCollection(bIsReal);
                int nCount = br.ReadInt32();

                for (int i = 0; i < nCount; i++)
                {
                    Bucket b = Bucket.Load(br);
                    col.m_rgBuckets.Add(b);
                }

                return col;
            }
        }

        /// <summary>
        /// Returns the total count across all buckets.
        /// </summary>
        public int TotalCount
        {
            get
            {
                int nCount = 0;

                foreach (Bucket b in m_rgBuckets)
                {
                    nCount += b.Count;
                }

                return nCount;
            }
        }

        /// <summary>
        /// Returns the distribution of buckets as a percentage for each time a bucket was hit.
        /// </summary>
        /// <returns>The distribution string is returned.</returns>
        public string ToDistributionString()
        {
            double dfTotalCount = TotalCount;
            string str = "{";

            foreach (Bucket b in m_rgBuckets)
            {
                double dfPct = (double)b.Count / dfTotalCount;
                str += dfPct.ToString("P");
                str += ",";
            }

            str = str.TrimEnd(',');
            str += "}";

            return str;
        }

        /// <summary>
        /// Returns the enumerator used in foreach loops.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        public IEnumerator<Bucket> GetEnumerator()
        {
            return m_rgBuckets.GetEnumerator();
        }

        /// <summary>
        /// Returns the enumerator used in foreach loops.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgBuckets.GetEnumerator();
        }
    }
}
