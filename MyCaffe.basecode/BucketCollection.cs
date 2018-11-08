using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    public class Bucket
    {
        double m_fMin;
        double m_fMax;
        double m_fSum;
        int m_nCount;

        public Bucket(double fMin, double fMax)
        {
            m_fMin = fMin;
            m_fMax = fMax;
        }

        public int Contains(double fVal)
        {
            if (fVal < m_fMin)
                return -1;

            if (fVal >= m_fMax)
                return 1;

            return 0;
        }

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

        public int Count
        {
            get { return m_nCount; }
        }

        public double Average
        {
            get { return m_fSum / m_nCount; }
        }

        public override string ToString()
        {
            return "[" + m_fMin.ToString() + "," + m_fMax.ToString() + "]-> " + m_nCount.ToString("N0");
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(m_nCount);
            bw.Write(m_fSum);
            bw.Write(m_fMin);
            bw.Write(m_fMax);
        }

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

    public class BucketCollection
    {
        List<Bucket> m_rgBuckets = new List<Bucket>();
        bool m_bIsDataReal = false;

        public BucketCollection(double fMin, double fMax, int nCount)
        {
            double fRange = fMax - fMin;
            double fStep = fRange / nCount;
            double fVal = fMin;

            for (int i = 0; i < nCount; i++)
            {
                m_rgBuckets.Add(new Bucket(fVal, fVal + fStep));
                fVal += fStep;
            }

            m_bIsDataReal = true;
        }

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

        public BucketCollection(bool bIsReal)
        {
            m_bIsDataReal = bIsReal;
        }

        public bool IsDataReal
        {
            get { return m_bIsDataReal; }
        }

        public int Count
        {
            get { return m_rgBuckets.Count; }
        }

        public void Add(double fVal)
        {
            for (int i = 0; i < m_rgBuckets.Count; i++)
            {
                int nVal = m_rgBuckets[i].Add(fVal);
                if (nVal == 0)
                    break;

                if (nVal == 1 && i == m_rgBuckets.Count - 1)
                    m_rgBuckets[i].Add(fVal, true);
            }
        }

        public double Translate(double fVal)
        {
            for (int i = 0; i < m_rgBuckets.Count; i++)
            {
                if (m_rgBuckets[i].Contains(fVal) == 0)
                    return m_rgBuckets[i].Average;
            }

            return m_rgBuckets[m_rgBuckets.Count - 1].Average;
        }

        public int FindIndex(double dfVal)
        {
            for (int i = 0; i < m_rgBuckets.Count; i++)
            {
                if (m_rgBuckets[i].Contains(dfVal) == 0)
                    return i;
            }

            return m_rgBuckets.Count - 1;
        }

        public double GetValueAt(int nIdx)
        {
            return m_rgBuckets[nIdx].Average;
        }

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
    }
}
