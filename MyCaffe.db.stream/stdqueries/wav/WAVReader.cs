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
    /// The WaveFormatExtensible structure describes the extended set of information of a WAV file.
    /// </summary>
    /// <remarks>
    /// @see [WAVE PCM soundfile format](http://soundfile.sapp.org/doc/WaveFormat/)
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct WaveFormatExtensible 
    {
        /// <summary>
        /// Specifies the AudioFormat where PCM = 1 for Linear quantization.
        /// </summary>
        public ushort wFormatTag;
        /// <summary>
        /// Specifies the number of channels in the data where Mono = 1 and Stero = 2.
        /// </summary>
        public ushort nChannels;
        /// <summary>
        /// Specifies the sample rate (e.g. 8000, 44100, etc.)
        /// </summary>
        public uint nSamplesPerSec;
        /// <summary>
        /// Specifies the average byte rate per second (nSamplesPerSec * Channels * BitsPerSample / 8)
        /// </summary>
        public uint nAvgBytesPerSec;
        /// <summary>
        /// Specifies the block alignment (Channels * BitsPerSample / 8)
        /// </summary>
        public ushort nBlockAlign;
        /// <summary>
        /// Specifies the number of bits per sample (8, 16, 32, etc.)
        /// </summary>
        public ushort wBitsPerSample;
        /// <summary>
        /// Specifies the extera parameter size.
        /// </summary>
        public ushort cbSize;
        /// <summary>
        /// Specifies the valid bits per sample.
        /// </summary>
        public ushort wValidBitsPerSample;
        /// <summary>
        /// Specifies the channel mask.
        /// </summary>
        public uint dwChannelMask;
        /// <summary>
        /// Specifies the sub format GUID.
        /// </summary>
        public Guid SubFormat;
    }

    /// <summary>
    /// The WaveFormat structure describes the header information of a WAV file.
    /// </summary>
    /// <remarks>
    /// @see [WAVE PCM soundfile format](http://soundfile.sapp.org/doc/WaveFormat/)
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct WaveFormat
    {
        /// <summary>
        /// Specifies the AudioFormat where PCM = 1 for Linear quantization.
        /// </summary>
        public ushort wFormatTag;
        /// <summary>
        /// Specifies the number of channels in the data where Mono = 1 and Stero = 2.
        /// </summary>
        public ushort nChannels;
        /// <summary>
        /// Specifies the sample rate (e.g. 8000, 44100, etc.)
        /// </summary>
        public uint nSamplesPerSec;
        /// <summary>
        /// Specifies the average byte rate per second (nSamplesPerSec * Channels * BitsPerSample / 8)
        /// </summary>
        public uint nAvgBytesPerSec;
        /// <summary>
        /// Specifies the block alignment (Channels * BitsPerSample / 8)
        /// </summary>
        public ushort nBlockAlign;
        /// <summary>
        /// Specifies the number of bits per sample (8, 16, 32, etc.)
        /// </summary>
        public ushort wBitsPerSample;
    }

    /// <summary>
    /// The WAVReader is an extension of the BinaryReader and is used to read WAV files.
    /// </summary>
    public class WAVReader : BinaryReader
    {
        Stream m_stream;
        WaveFormat m_format = new WaveFormat();
        Dictionary<string, List<string>> m_rgInfo = new Dictionary<string, List<string>>();
        long m_lDataPos;
        int m_nDataSize;
        List<double[]> m_rgrgSamples;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="stream">Specifies the data stream to read.</param>
        public WAVReader(Stream stream) : base(stream)
        {
            m_stream = stream;
        }

        /// <summary>
        /// Returns the WAV file header information.
        /// </summary>
        public WaveFormat Format
        {
            get { return m_format; }
        }

        /// <summary>
        /// Returns the frequency samples of the WAV file.
        /// </summary>
        public List<double[]> Samples
        {
            get { return m_rgrgSamples; }
        }

        /// <summary>
        /// Returns the WAV file header information in a key=value format.
        /// </summary>
        public Dictionary<string, List<string>> ExtraInformation
        {
            get { return m_rgInfo; }
        }

        /// <summary>
        /// Reads the WAV file data.
        /// </summary>
        /// <param name="bReadHeaderOnly">Optionally, specifies to only read the header information.</param>
        /// <returns>If read successfully this function returns <i>true</i>, otherwise <i>false</i>.</returns>
        public bool ReadToEnd(bool bReadHeaderOnly = false)
        {
            if (!readContent())
                return false;

            if (bReadHeaderOnly)
                return true;

            if (!readAudioContent())
                return false;

            return true;
        }

        /// <summary>
        /// Returns the number of samples.
        /// </summary>
        public int SampleCount
        {
            get { return m_nDataSize / m_format.nBlockAlign; }
        }

        private bool readAudioContent()
        {
            m_stream.Seek(m_lDataPos, SeekOrigin.Begin);

            int nSamples = m_nDataSize / m_format.nBlockAlign;

            m_rgrgSamples = new List<double[]>();
            for (int i = 0; i < m_format.nChannels; i++)
            {
                m_rgrgSamples.Add(new double[nSamples]);
            }

            for (int s = 0; s < nSamples; s++)
            {
                for (int ch = 0; ch < m_format.nChannels; ch++)
                {
                    double dfSample;

                    switch (m_format.wBitsPerSample)
                    {
                        // 8-bit unsigned
                        case 8:
                            {
                                long v = ReadByte();
                                v = v - 0x80;
                                dfSample = (v / (double)0x80);
                            }
                            break;

                        case 16:
                            {
                                int b1 = ReadByte();
                                int b2 = ReadByte();
                                int v = ((0xFFFF * (b2 >> 7)) << 16) | (b2 << 8) | b1;
                                dfSample = (v / (double)0x8000);
                            }
                            break;

                        case 24:
                            {
                                int b1 = ReadByte();
                                int b2 = ReadByte();
                                int b3 = ReadByte();
                                int v = ((0xFF * (b3 >> 7)) << 24) | (b3 << 16) | (b2 << 8) | b1;
                                dfSample = (v / (double)0x800000);
                            }
                            break;

                        case 32:
                            {
                                int b1 = ReadByte();
                                int b2 = ReadByte();
                                int b3 = ReadByte();
                                int b4 = ReadByte();
                                int v = (b4 << 24) | (b3 << 16) | (b2 << 8) | b1;
                                dfSample = (v / (double)0x80000000);
                            }
                            break;

                        default:
                            throw new NotImplementedException("The bits per sample of " + m_format.wBitsPerSample.ToString() + " is not supported.");
                    }

                    m_rgrgSamples[ch][s] = (float)dfSample;
                }
            }

            return true;
        }

        private bool readContent()
        {
            return readRiff();
        }

        private bool readRiff()
        {
            string strRiff = readID();
            int nSize = ReadInt32();
            string strType = readID();

            if (strRiff != "RIFF" || strType != "WAVE")
                return false;

            bool bEof = readChunk();
            while (!bEof)
            {
                bEof = readChunk();
            }

            return true;
        }

        private string readID()
        {
            int c1 = 0;

            while (c1 < (int)' ' || c1 > 127)
            {
                if (m_stream.Position == m_stream.Length)
                    return null;

                c1 = (int)m_stream.ReadByte();
                if (c1 == -1)
                    return null;
            }

            string str = "";
            str += (char)c1;
            str += (char)m_stream.ReadByte();
            str += (char)m_stream.ReadByte();
            str += (char)m_stream.ReadByte();

            return str;
        }

        private bool readChunk()
        {
            if (m_stream.Position == m_stream.Length)
                return true;

            if (m_stream.Length < m_stream.Position)
                return true;

            string strID = readID();
            if (strID == null)
                return true;

            int nSize = ReadInt32();
            long lPos = m_stream.Position;

            if (lPos + nSize > m_stream.Length)
                nSize = (int)(m_stream.Length - lPos);

            switch (strID)
            {
                case "fmt ":
                    readFmt(nSize);
                    break;

                case "LIST":
                    string strType = readID();
                    if (strType == "INFO")
                        readListInfo(nSize);
                    break;

                case "data":
                    m_lDataPos = lPos;
                    m_nDataSize = nSize;
                    return true;

                default:
                    m_stream.Seek(lPos + nSize, SeekOrigin.Begin);
                    break;
            }

            return false;
        }

        private void readFmt(int nSize)
        {
            int nStructSize = Marshal.SizeOf(m_format);

            if (nSize >= nStructSize)
            {
                byte[] rgData = ReadBytes(nStructSize);
                m_format = ByteArrayToStructure<WaveFormat>(rgData);
            }
        }

        private void readListInfo(int nSize)
        {
            long lPos = m_stream.Position;

            while (m_stream.Position - lPos < nSize - 4)
            {
                string strField = readID();
                if (strField == null)
                    return;

                int nFieldSize = ReadInt32();
                if (nFieldSize > 0)
                {
                    byte[] rgData = ReadBytes(nFieldSize);
                    string strVal = Encoding.UTF8.GetString(rgData).Trim();
                    int nIdx = strVal.IndexOf((char)0);
                    if (nIdx != -1)
                        strVal = strVal.Substring(0, nIdx);

                    if (!m_rgInfo.ContainsKey(strField))
                        m_rgInfo.Add(strField, new List<string>());

                    m_rgInfo[strField].Add(strVal);
                }
            }
        }

        /// <summary>
        /// Converts a byte array to a structure.
        /// </summary>
        /// <typeparam name="T">Specifies the structure type.</typeparam>
        /// <param name="bytes">Specifies the byte array.</param>
        /// <returns>The structure is returned.</returns>
        protected static T ByteArrayToStructure<T>(byte[] bytes) where T : struct
        {
            GCHandle handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
            T stuff = (T)Marshal.PtrToStructure(handle.AddrOfPinnedObject(),
                typeof(T));
            handle.Free();
            return stuff;
        }
    }
}
