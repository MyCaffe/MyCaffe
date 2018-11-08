using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// Modified from https://www.codeproject.com/Articles/806042/Spectrogram-generation-in-SampleTagger
/// License: https://www.codeproject.com/info/cpol10.aspx
/// </summary>
namespace MyCaffe.db.stream
{
    /// <summary>
    /// The WAVWriter is a special BinaryWriter used to write WAV files.
    /// </summary>
    public class WAVWriter : BinaryWriter
    {
        Stream m_stream;
        WaveFormat m_format = new WaveFormat();
        Dictionary<string, List<string>> m_rgInfo = new Dictionary<string, List<string>>();
        long m_lSizePos = 0;
        List<double[]> m_rgrgSamples;
        int m_nSampleStep = 1;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="stream">Specifies the output stream.</param>
        public WAVWriter(Stream stream) : base(stream)
        {
            m_stream = stream;
        }

        /// <summary>
        /// Get/set the WaveFormat.
        /// </summary>
        public WaveFormat Format
        {
            get { return m_format; }
            set { m_format = value; }
        }

        /// <summary>
        /// Get/set the frequency samples.
        /// </summary>
        public List<double[]> Samples
        {
            get { return m_rgrgSamples; }
            set { m_rgrgSamples = value; }
        }

        /// <summary>
        /// The WriteAll method writes all WAV file data to the file.
        /// </summary>
        /// <param name="nNewSampleRate">Optionally, specifies a new sample rate to use, if any (default = 0, ignoring this parameter).</param>
        /// <returns>On success, <i>true</i> is returned, otherwise <i>false</i>.</returns>
        public bool WriteAll(int nNewSampleRate = 0)
        {
            if (m_rgrgSamples == null)
                throw new Exception("You must set the audio data first!");

            if (m_format.nChannels == 0)
                throw new Exception("You must set the format first!");

            if (nNewSampleRate > 0)
            {
                m_nSampleStep = (int)m_format.nSamplesPerSec / nNewSampleRate;
                m_format.nSamplesPerSec = (uint)nNewSampleRate;
                m_format.nAvgBytesPerSec = (uint)(nNewSampleRate * m_format.wBitsPerSample / 8);
            }

            if (!writeContent())
                return false;

            if (!writeAudioContent())
                return false;

            int nSize = (int)(m_stream.Position - m_lSizePos) - 4;
            m_stream.Seek(m_lSizePos, SeekOrigin.Begin);
            Write(nSize);

            return true;
        }

        private bool writeAudioContent()
        {
            int nSamples = m_rgrgSamples[0].Length;
            int nChannels = m_rgrgSamples.Count;

            for (int s = 0; s < nSamples; s++)
            {
                for (int ch = 0; ch < nChannels; ch++)
                {
                    if (s % m_nSampleStep == 0)
                    {
                        double dfSample = m_rgrgSamples[ch][s];

                        switch (m_format.wBitsPerSample)
                        {
                            case 32:
                                {
                                    int v = (int)(dfSample * (double)0x80000000);
                                    byte b1 = (byte)(0x000000FF & v);
                                    byte b2 = (byte)((0x0000FF00 & v) >> 8);
                                    byte b3 = (byte)((0x00FF0000 & v) >> 16);
                                    byte b4 = (byte)((0xFF000000 & v) >> 24);
                                    Write(b1);
                                    Write(b2);
                                    Write(b3);
                                    Write(b4);
                                }
                                break;

                            default:
                                throw new NotImplementedException("The bits per sample of " + m_format.wBitsPerSample.ToString() + " is not supported.");
                        }
                    }
                }
            }

            return true;
        }

        private bool writeContent()
        {
            return writeRiff();
        }

        private bool writeRiff()
        {
            if (!writeID("RIFF"))
                return false;

            m_lSizePos = m_stream.Position;
            int nSize = 0;
            Write(nSize);

            if (!writeID("WAVE"))
                return false;

            if (!writeChunk("fmt "))
                return false;

            if (!writeChunk("data"))
                return false;

            return true;
        }

        private bool writeID(string strID)
        {
            if (strID.Length != 4)
                throw new Exception("The ID must have a length of 4.");

            byte b1 = (byte)strID[0];
            byte b2 = (byte)strID[1];
            byte b3 = (byte)strID[2];
            byte b4 = (byte)strID[3];

            Write(b1);
            Write(b2);
            Write(b3);
            Write(b4);

            return true;
        }

        private bool writeChunk(string strType)
        {
            if (!writeID(strType))
                return false;

            switch (strType)
            {
                case "fmt ":
                    return writeFmt();

                case "data":
                    return writeData();

                default:
                    throw new NotImplementedException("The format '" + strType + "' is not supported.");
            }
        }

        private bool writeFmt()
        {
            int nStructSize = Marshal.SizeOf(m_format);
            byte[] rgData = StructureToByteArray<WaveFormat>(m_format);

            if (nStructSize != rgData.Length)
                throw new Exception("Invalid byte sizing.");

            Write(nStructSize);
            Write(rgData);
            return true;
        }

        private bool writeData()
        {
            int nSize = 0;
            int nByteCount = m_format.wBitsPerSample / 8;

            for (int i = 0; i < m_rgrgSamples.Count; i++)
            {
                nSize += nByteCount * m_rgrgSamples[i].Length;
            }

            Write(nSize);

            return true;
        }

        /// <summary>
        /// Converts a byte array into a structure.
        /// </summary>
        /// <typeparam name="T">Specifies the structure type.</typeparam>
        /// <param name="bytes">Specifies the byte array.</param>
        /// <returns>The structure is returned.</returns>
        public static T ByteArrayToStructure<T>(byte[] bytes) where T : struct
        {
            GCHandle handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            T stuff = (T)Marshal.PtrToStructure(handle.AddrOfPinnedObject(),
                typeof(T));
            handle.Free();
            return stuff;
        }

        /// <summary>
        /// Converts a structure into a byte array.
        /// </summary>
        /// <typeparam name="T">Specifies the structure type.</typeparam>
        /// <param name="val">Specifies the structure.</param>
        /// <returns>The byte array is returned.</returns>
        public static byte[] StructureToByteArray<T>(T val)
        {
            int size = Marshal.SizeOf(val);
            byte[] arr = new byte[size];

            IntPtr ptr = Marshal.AllocHGlobal(size);
            Marshal.StructureToPtr(val, ptr, true);
            Marshal.Copy(ptr, arr, 0, size);
            Marshal.FreeHGlobal(ptr);
            return arr;
        }
    }
}
