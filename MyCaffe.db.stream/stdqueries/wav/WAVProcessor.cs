using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    /// <summary>
    /// The WAVProcessor is used to process WAV files and perform tasks such as downsampling.
    /// </summary>
    public class WAVProcessor
    {
        WaveFormat m_fmt;
        List<double[]> m_rgrgSamples;
        Log m_log;
        CancelEvent m_evtCancel;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="fmt">Specifies the WaveFormat.</param>
        /// <param name="rgrgSamples">Specifies the WAV file frequency samples.</param>
        /// <param name="log">Specifies the output log.</param>
        /// <param name="evtCancel">Specifies the event used to cancel processes.</param>
        public WAVProcessor(WaveFormat fmt, List<double[]> rgrgSamples, Log log, CancelEvent evtCancel)
        {
            m_fmt = fmt;
            m_rgrgSamples = rgrgSamples;
            m_log = log;
            m_evtCancel = evtCancel;
        }

        /// <summary>
        /// Returns the WaveFormat.
        /// </summary>
        public WaveFormat Format
        {
            get { return m_fmt; }
        }

        /// <summary>
        /// Returns the WAV frequency samples.
        /// </summary>
        public List<double[]> Samples
        {
            get { return m_rgrgSamples; }
        }

        /// <summary>
        /// The DownSample method reduces the number of samples per second in the resulting sample set.
        /// </summary>
        /// <param name="nNewSamplesPerSecond">Specifies the new (lower) samples per second - must be a factor of the original samples per second.</param>
        /// <returns>A new WAVProcessor with the new downsamples samples is returned.</returns>
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
