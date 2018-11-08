using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.stream.stdqueries.wav
{
    public class WAVProcessor
    {
        WaveFormat m_fmt;
        List<double[]> m_rgrgSamples;
        BucketCollection m_rgBucketCol;
        Log m_log;
        CancelEvent m_evtCancel;

        public WAVProcessor(WaveFormat fmt, List<double[]> rgrgSamples, Log log, CancelEvent evtCancel)
        {
            m_fmt = fmt;
            m_rgrgSamples = rgrgSamples;
            m_log = log;
            m_evtCancel = evtCancel;
        }

        public WaveFormat Format
        {
            get { return m_fmt; }
        }

        public List<double[]> Samples
        {
            get { return m_rgrgSamples; }
        }

        public BucketCollection BucketCollection
        {
            get { return m_rgBucketCol; }
        }

        public WAVProcessor DownSample(int nNewSamplesPerSecond)
        {
            if (m_fmt.nSamplesPerSec < nNewSamplesPerSecond)
                throw new Exception("The new sample rate (" + nNewSamplesPerSecond.ToString() + ") must be less than the old sample rate (" + m_fmt.nSamplesPerSec.ToString() + ").");

            if (m_fmt.nSamplesPerSec % nNewSamplesPerSecond != 0)
                throw new Exception("The new sample rate (" + nNewSamplesPerSecond.ToString() + ") must be a factor of the old sample rate (" + m_fmt.nSamplesPerSec.ToString() + ").");

            int nStep = (int)m_fmt.nSamplesPerSec / nNewSamplesPerSecond;
            List<double[]> rgrgNewSamples = new List<double[]>();
            List<double> rgSamples = new List<double>();
            int nIdx = 0;
            int nTotal = m_rgrgSamples.Count * m_rgrgSamples[0].Length;
            Stopwatch sw = new Stopwatch();

            sw.Start();

            for (int i = 0; i < m_rgrgSamples.Count; i++)
            {
                for (int j = 0; j < m_rgrgSamples[i].Length; j++)
                {
                    if (j % nStep == 0)
                    {
                        double fVal = m_rgrgSamples[i][j];
                        rgSamples.Add(fVal);

                        if (m_evtCancel.WaitOne(0))
                            return null;

                        if (sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            double dfPct = (double)nIdx / (double)nTotal;
                            m_log.WriteLine("Downsampling at " + dfPct.ToString("P") + "...");
                            sw.Restart();
                        }
                    }

                    nIdx++;
                }

                rgrgNewSamples.Add(rgSamples.ToArray());
                rgSamples = new List<double>();
            }

            WaveFormat fmt = m_fmt;
            fmt.nSamplesPerSec = (uint)nNewSamplesPerSecond;
            fmt.nAvgBytesPerSec = (uint)(nNewSamplesPerSecond * fmt.wBitsPerSample);

            return new WAVProcessor(fmt, rgrgNewSamples, m_log, m_evtCancel);
        }
    }
}
