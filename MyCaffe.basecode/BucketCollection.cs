using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
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
        object m_tag = null;
        List<List<float>> m_rgReturns = new List<List<float>>();

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
        /// Returns the returns if any exist.
        /// </summary>
        public List<List<float>> Returns
        {
            get { return m_rgReturns; }
        }

        /// <summary>
        /// Get the average returns at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the return index to average.</param>
        /// <returns>The average value or null is returned.</returns>
        public double? AveReturns(int nIdx)
        {
            if (m_rgReturns.Count == 0)
                return null;

            if (m_rgReturns[0].Count <= nIdx)
                return null;

            double fSum = 0;
            int nCount = 0;
            foreach (List<float> rg in m_rgReturns)
            {
                if (nIdx < rg.Count)
                {
                    fSum += rg[nIdx];
                    nCount++;
                }
            }

            if (nCount == 0)
                return null;

            return fSum / nCount;
        }

        /// <summary>
        /// Create a copy of the bucket.
        /// </summary>
        /// <returns>The copy is returned.</returns>
        public Bucket Clone()
        {
            Bucket b = new Bucket(m_fMin, m_fMax);
            b.m_fSum = m_fSum;
            b.m_nCount = m_nCount;
            b.m_tag = m_tag;
            return b;
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
        /// <param name="bPeekOnly">Optinoally, only peek to test if the value fits in the bucket.</param>
        /// <param name="rgReturns">Optionally, specifies an array of returns.</param>
        /// <returns>Returns 0 if the value falls within the Buckets range and is added, -1 if the value is less than the Bucket range and 1 if the value is greater.</returns>
        public int Add(double fVal, bool bForce = false, bool bPeekOnly = false, List<float> rgReturns = null)
        {
            if (!bForce)
            {
                int nVal = Contains(fVal);
                if (nVal != 0)
                    return nVal;
            }

            if (!bPeekOnly)
            {
                m_nCount++;
                m_fSum += fVal;

                if (rgReturns != null && rgReturns.Count > 0)
                    m_rgReturns.Add(rgReturns);
            }

            return 0;
        }

        /// <summary>
        /// Updates the Bucket with a new minimum and maximum value.
        /// </summary>
        /// <param name="dfMin">Specifies the new min.</param>
        /// <param name="dfMax">Specifies the new max.</param>
        public void Update(double dfMin, double dfMax)
        {
            m_fMin = dfMin;
            m_fMax = dfMax;
        }

        /// <summary>
        /// Combine the dst bucket with this one.
        /// </summary>
        public void Combine(Bucket b)
        {
            m_nCount += b.Count;
            m_fSum += b.m_fSum;
            m_fMin = Math.Min(m_fMin, b.Minimum);
            m_fMax = Math.Max(m_fMax, b.Maximum);
        }

        /// <summary>
        /// Returns the number of items added to the Bucket.
        /// </summary>
        public int Count
        {
            get { return m_nCount; }
            set { m_nCount = value; }
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
        /// Get/set a user specified tag.
        /// </summary>
        public object Tag
        {
            get { return m_tag; }
            set { m_tag = value; }
        }

        /// <summary>
        /// Returns a string representation of the Bucket.
        /// </summary>
        /// <returns>The string representation is returned.</returns>
        public override string ToString()
        {
            return "[" + m_fMin.ToString("N10") + "," + m_fMax.ToString("N10") + "]-> " + m_nCount.ToString("N0");
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
        /// Specifies the output format used when creating a distribution string.
        /// </summary>
        public enum OUTPUT_FMT
        {
            /// <summary>
            /// No special output formatting used.
            /// </summary>
            NONE,
            /// <summary>
            /// Format for a text file.
            /// </summary>
            TXT,
            /// <summary>
            /// Format for a CSV file.
            /// </summary>
            CSV
        }

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
        /// The constructor used to build a bucket collection from a configuration string.
        /// </summary>
        /// <param name="strConfigString">Specifies the configuration string created when calling ToConfigString()</param>
        public BucketCollection(string strConfigString)
        {
            List<Tuple<double, double>> rgVal = Parse(strConfigString);
            foreach (Tuple<double, double> val in rgVal)
            {
                Bucket b = new Bucket(val.Item1, val.Item2);
                m_rgBuckets.Add(b);
            }

            m_bIsDataReal = true;
        }

        /// <summary>
        /// The constructor used to build a bucket collection from a list of tuples.
        /// </summary>
        /// <param name="rgVal">Specifies the list of min/max values.</param>
        public BucketCollection(List<Tuple<double, double>> rgVal)
        {
            foreach (Tuple<double, double> val in rgVal)
            {
                Bucket b = new Bucket(val.Item1, val.Item2);
                m_rgBuckets.Add(b);
            }

            m_bIsDataReal = true;
        }

        /// <summary>
        /// Return the list of buckets.
        /// </summary>
        public List<Bucket> Buckets
        {
            get { return m_rgBuckets; }
        }

        /// <summary>
        /// Create a new normalized bucket collection.
        /// </summary>
        /// <param name="nMaxHt">Specifies the max count to use.</param>
        /// <returns>The new normalized bucket collection is returned.</returns>
        public BucketCollection Normalize(int nMaxHt)
        {
            BucketCollection col = new BucketCollection(true);

            int nMax = m_rgBuckets.Max(p => p.Count);   

            foreach (Bucket b in m_rgBuckets)
            {
                double dfPct = (double)b.Count / (double)nMax;
                int nCount = (int)(dfPct * nMaxHt);
                Bucket bNew = new Bucket(b.Minimum, b.Maximum);
                bNew.Count = nCount;
                col.m_rgBuckets.Add(bNew);
            }

            return col;
        }

        /// <summary>
        /// Combine the bucket at the source index with the bucket at the destination index.
        /// </summary>
        /// <param name="nIdxSrc">Specifies the source bucket index.</param>
        /// <param name="nIdxDst">Specifies the destination bucket index.</param>
        /// <returns>true is returned on successful combining and removing of the src bucket.</returns>
        public bool Combine(int nIdxSrc, int nIdxDst)
        {
            if (nIdxSrc < 0 || nIdxSrc >= m_rgBuckets.Count)
                return false;

            if (nIdxDst < 0 || nIdxDst >= m_rgBuckets.Count)
                return false;

            if (nIdxSrc == nIdxDst)
                return false;

            Bucket bSrc = m_rgBuckets[nIdxSrc];
            Bucket bDst = m_rgBuckets[nIdxDst];
            bDst.Combine(bSrc);

            m_rgBuckets.RemoveAt(nIdxSrc);

            return true;
        }

        /// <summary>
        /// Parse a configuration string and return the list of tuples containing the min,max values.
        /// </summary>
        /// <param name="strCfg"></param>
        /// <returns></returns>
        public static List<Tuple<double, double>> Parse(string strCfg)
        {
            List<Tuple<double, double>> rgVal = new List<Tuple<double, double>>();

            string[] rgstr = strCfg.Split(';');
            int nCount = 0;
            foreach (string str in rgstr)
            {
                string strVal = str.Trim();
                if (strVal.Length == 0)
                    continue;
                if (strVal.StartsWith("Count="))
                {
                    string strCount = strVal.Substring(6);
                    nCount = int.Parse(strCount);
                }
                else
                {
                    string strMin = strVal.Substring(1, strVal.IndexOf(',') - 1);
                    string strMax = strVal.Substring(strVal.IndexOf(',') + 1, strVal.Length - strVal.IndexOf(',') - 2);

                    rgVal.Add(new Tuple<double, double>(double.Parse(strMin), double.Parse(strMax)));
                }
            }

            return rgVal;
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
        /// Return the configuration string that can be used to recreate the BucketCollection.
        /// </summary>
        /// <returns>Returns the configuration string.</returns>
        public string ToConfigString()
        {
            StringBuilder sb = new StringBuilder();

            sb.Append("Count=" + m_rgBuckets.Count.ToString() + ";");
            for (int i = 0; i < m_rgBuckets.Count; i++)
            {
                sb.Append("[" + m_rgBuckets[i].Minimum.ToString("N10") + "," + m_rgBuckets[i].Maximum.ToString("N10") + "];");
            }

            return sb.ToString();
        }

        /// <summary>
        /// Remove a given bucket at a certain index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        public void RemoveAt(int nIdx)
        {
            m_rgBuckets.RemoveAt(nIdx);
        }

        /// <summary>
        /// Get/set whether or not the Buckets hold Real values.
        /// </summary>
        public bool IsDataReal
        {
            get { return m_bIsDataReal; }
            set { m_bIsDataReal = value; }
        }

        /// <summary>
        /// Returns the number of Buckets.
        /// </summary>
        public int Count
        {
            get { return m_rgBuckets.Count; }
        }

        /// <summary>
        /// Returns the sum of all bucket counts.
        /// </summary>
        public int BucketCountSum
        {
            get { return m_rgBuckets.Sum(p => p.Count); }
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
        /// Returns the numeric range that all buckets fall into.
        /// </summary>
        /// <returns>A tuple containing the min,max is returned.</returns>
        public Tuple<double,double> GetRange()
        {
            if (m_rgBuckets.Count == 0)
                return new Tuple<double, double>(0, 0);

            return new Tuple<double, double>(m_rgBuckets[0].Minimum, m_rgBuckets[m_rgBuckets.Count - 1].Maximum);
        }

        /// <summary>
        /// Return the bucket at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <returns>The bucket is returned.</returns>
        public Bucket GetBucketAt(int nIdx)
        {
            return m_rgBuckets[nIdx];
        }

        /// <summary>
        /// Finds the correct Bucket and adds the value to it.
        /// </summary>
        /// <param name="fVal">Specifies the value to add.</param>
        /// <param name="bPeekOnly">When true, only return the bucket index but dont add the value (default = false).</param>
        /// <param name="rgReturns">Optionally, specifies an array of returns.</param>
        /// <returns>The index of the bucket for which the value was added is returned.</returns>
        public int Add(double fVal, bool bPeekOnly = false, List<float> rgReturns = null)
        {
            for (int i = 0; i < m_rgBuckets.Count; i++)
            {
                int nVal = m_rgBuckets[i].Add(fVal, false, bPeekOnly, rgReturns);
                if (nVal == 0)
                    return i;

                if (nVal < 0 && i == 0)
                {
                    if (!bPeekOnly)
                        m_rgBuckets[i].Add(fVal, true, false, rgReturns);
                    return i;
                }

                if (nVal == 1 && i == m_rgBuckets.Count - 1)
                {
                    if (!bPeekOnly)
                        m_rgBuckets[i].Add(fVal, true, false, rgReturns);
                    return i;
                }
            }

            throw new Exception("Failed to find a bucket!");
        }

        /// <summary>
        /// Reduces the buckets to only include those that have a count that are within 1.0 - dfPct of the maximum bucket count.
        /// </summary>
        /// <param name="dfPct">Specifies the threshold percent where all buckets with a count >= Max Count * (1.0 - dfPct) are kept.</param>
        public void Reduce(double dfPct)
        {
            Bucket b = GetBucketWithMaxCount();
            int nThreshold = (int)(b.Count * (1.0 - dfPct));

            List<Bucket> rgBuckets = new List<Bucket>();
            foreach (Bucket b1 in m_rgBuckets)
            {
                if (b1.Count > nThreshold)
                    rgBuckets.Add(b1);
            }

            m_rgBuckets = rgBuckets;
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
                        double dfVal = sd.GetDataAtD(nOffset + j);
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
                    double dfVal = sd.GetDataAtD(nOffset + j);
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
            using (MemoryStream ms = new MemoryStream(rg))
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
        /// <param name="fmt">Optionally, specifies the output format of the string (default = NONE).</param>
        /// <param name="nMaxDots">Optionally, specifies the maximum number of dots used when 'bFull' = true, ignored when 'bFull' = false (default = 30).</param>
        /// <param name="strFmt">Optionally, specifies the format string for each range (default = '0.00000').</param>
        /// <param name="bIncludePercents">Optionally, specifies to include the percentages.</param>
        /// <returns>The distribution string is returned.</returns>
        public string ToDistributionString(OUTPUT_FMT fmt = OUTPUT_FMT.NONE, int nMaxDots = 30, string strFmt = "0.00000", bool bIncludePercents = false)
        {
            double dfTotalCount = TotalCount;
            string str = "";

            if (fmt == OUTPUT_FMT.NONE)
                str += "{";
            else if (fmt == OUTPUT_FMT.CSV)
                str += "MINIMUM, MAXIMUM, COUNT" + Environment.NewLine;

            foreach (Bucket b in m_rgBuckets)
            {
                double dfPct = (dfTotalCount == 0) ? 0 : (double)b.Count / dfTotalCount;

                if (fmt == OUTPUT_FMT.TXT)
                {
                    string strDots = "";
                    strDots = strDots.PadRight((int)(nMaxDots * dfPct), '*');
                    str += "[" + b.Minimum.ToString(strFmt) + ", " + b.Maximum.ToString(strFmt) + "] " + strDots + " (" + b.Count.ToString("N0") + ")";

                    if (bIncludePercents)
                        str += " " + (dfPct * 100).ToString("N4") + "%";

                    str += Environment.NewLine;
                }
                else if (fmt == OUTPUT_FMT.CSV)
                {
                    str += b.Minimum.ToString() + "," + b.Maximum.ToString() + "," + b.Count.ToString() + Environment.NewLine;
                }
                else
                {
                    str += dfPct.ToString("P");
                    str += ",";
                }
            }

            if (fmt == OUTPUT_FMT.NONE)
            {
                str = str.TrimEnd(',');
                str += "}";
            }

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

    /// <summary>
    /// Defines a rolling bucket collection that maintains the bucket statistics for a set of up to max buckets.  Once the max is reached, the oldest bucket collection is dropped.
    /// </summary>
    public class RollingBucketCollection
    {
        double m_dfMin;
        double m_dfMax;
        int m_nCount;
        int m_nMax;
        List<BucketCollection> m_rgBucketColletions = new List<BucketCollection>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fMin">Specifies the minimum of the range managed by all bucket collections.</param>
        /// <param name="fMax">Specifies the maximum of the range managed by all bucket collections.</param>
        /// <param name="nCount">Specifies the number of buckets in each bucket collection.</param>
        /// <param name="nMax">Specifies the maximum number of bucket collections (default = 1000).</param>
        public RollingBucketCollection(double fMin, double fMax, int nCount, int nMax = 1000)
        {
            m_nMax = nMax;
            m_dfMin = fMin;
            m_dfMax = fMax;
            m_nCount = nCount;

            m_rgBucketColletions.Add(new BucketCollection(fMin, fMax, nCount));
        }

        /// <summary>
        /// Adds a new value to all bucket collections.
        /// </summary>
        /// <param name="dfVal">Specifies the value to add.</param>
        /// <param name="bPeekOnly">Specifies to peek only and dont add the value, but return the index where the value would fall.</param>
        /// <param name="rgReturns">Optionally, specifies a list of returns to associate with the bucket collection for which the value falls.</param>
        /// <returns>The index of the bucket accpeting the value is returned.</returns>
        public int Add(double dfVal, bool bPeekOnly = false, List<float> rgReturns = null)
        {
            if (bPeekOnly)
                return m_rgBucketColletions[0].Add(dfVal, true);

            int? nIdx = null;

            for (int i = 0; i < m_rgBucketColletions.Count; i++)
            {
                int nIdx1 = m_rgBucketColletions[i].Add(dfVal, false, rgReturns);
                if (!nIdx.HasValue)
                    nIdx = nIdx1;
            }

            return nIdx.Value;
        }

        /// <summary>
        /// Add a new bucket collection to the list.
        /// </summary>
        public void AddCollection()
        {
            m_rgBucketColletions.Add(new BucketCollection(m_dfMin, m_dfMax, m_nCount));
            if (m_rgBucketColletions.Count > m_nMax)
                m_rgBucketColletions.RemoveAt(0);
        }

        /// <summary>
        /// Return the oldest bucket collection.
        /// </summary>
        /// <returns>The bucket collection is returned.</returns>
        public BucketCollection Current
        {
            get { return m_rgBucketColletions[0]; }
        }
    }
}
