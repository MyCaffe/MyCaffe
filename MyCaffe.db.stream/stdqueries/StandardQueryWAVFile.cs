﻿using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    /// <summary>
    /// The StandardQueryWAVFile provides queries that read sound frequencies from (*.WAV) files residing in a given directory.
    /// </summary>
    public class StandardQueryWAVFile : IXCustomQuery
    {
        string m_strPath;
        string[] m_rgstrFiles;
        int m_nFileIdx = 0;
        Dictionary<string, float> m_rgInfo = new Dictionary<string, float>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strParam">Specifies the parameters which shold contains the 'FilePath'=path key=value pair.</param>
        public StandardQueryWAVFile(string strParam = null)
        {
            if (strParam != null)
            {
                strParam = ParamPacker.UnPack(strParam);
                PropertySet ps = new PropertySet(strParam);
                m_strPath = ps.GetProperty("FilePath");
            }
        }

        /// <summary>
        /// Returns the QUERY_TYPE of REAL_FLOAT.
        /// </summary>
        public CUSTOM_QUERY_TYPE QueryType
        {
            get { return CUSTOM_QUERY_TYPE.REAL_DOUBLE; }
        }

        /// <summary>
        /// Returns the custom query name 'StdWAVFileQuery'.
        /// </summary>
        public string Name
        {
            get { return "StdWAVFileQuery"; }
        }

        /// <summary>
        /// Returns the field count of 1.
        /// </summary>
        public int FieldCount
        {
            get { return 1; }  // data
        }

        /// <summary>
        /// Clone the custom query returning a new copy.
        /// </summary>
        /// <param name="strParam">Optionally, initialize the new copy with these parameters.</param>
        /// <returns>The new copy is returned.</returns>
        public IXCustomQuery Clone(string strParam)
        {
            return new StandardQueryWAVFile(strParam);
        }

        /// <summary>
        /// Close the custom query.
        /// </summary>
        public void Close()
        {
            m_rgstrFiles = null;
            m_nFileIdx = 0;
        }

        /// <summary>
        /// Open the custom query.  The query must be opened before calling QueryBytes.
        /// </summary>
        public void Open()
        {
            string[] rgstrFiles = Directory.GetFiles(m_strPath);
            m_nFileIdx = 0;

            List<string> rgstr = new List<string>();

            for (int i = 0; i < rgstrFiles.Length; i++)
            {
                FileInfo fi = new FileInfo(rgstrFiles[i]);

                if (fi.Extension.ToLower() == ".wav")
                    rgstr.Add(rgstrFiles[i]);
            }

            m_rgstrFiles = rgstr.ToArray();

            if (m_rgstrFiles.Length == 0)
                throw new Exception("The CustomTextQuery could not find any audio WAV (*.wav) files to load.");
        }

        /// <summary>
        /// The QueryByTime method is not implemented.
        /// </summary>
        /// <param name="dt">not used.</param>
        /// <param name="ts">non used.</param>
        /// <param name="nCount">not used.</param>
        /// <returns>not used.</returns>
        public double[] QueryByTime(DateTime dt, TimeSpan ts, int nCount)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// The QueryBytes method is not implemented.
        /// </summary>
        /// <returns>not used.</returns>
        public byte[] QueryBytes()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// The QueryRealD method returns all samples of the next file in the directory.
        /// </summary>
        /// <returns>All samples along with the format are returned where the format is placed in the last array.</returns>
        public List<double[]> QueryRealD()
        {
            if (m_nFileIdx == m_rgstrFiles.Length)
                return null;

            using (FileStream fs = File.OpenRead(m_rgstrFiles[m_nFileIdx]))
            using (WAVReader wav = new WAVReader(fs))
            {
                m_nFileIdx++;
                wav.ReadToEnd();

                m_rgInfo = new Dictionary<string, float>();
                m_rgInfo.Add("AveBytesPerSec", wav.Format.nAvgBytesPerSec);
                m_rgInfo.Add("BlockAlign", wav.Format.nBlockAlign);
                m_rgInfo.Add("Channels", wav.Format.nChannels);
                m_rgInfo.Add("SamplesPerSec", wav.Format.nSamplesPerSec);
                m_rgInfo.Add("BitsPerSample", wav.Format.wBitsPerSample);
                m_rgInfo.Add("FormatTag", wav.Format.wFormatTag);

                return wav.Samples;
            }
        }

        /// <summary>
        /// The QueryRealF method returns all samples of the next file in the directory.
        /// </summary>
        /// <returns>All samples along with the format are returned where the format is placed in the last array.</returns>
        public List<float[]> QueryRealF()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// The Query information returns information about the data queried such as header information.
        /// </summary>
        /// <returns>The information about the data is returned.</returns>
        public Dictionary<string, float> QueryInfo()
        {
            if (m_rgInfo != null && m_rgInfo.Count > 0)
                return m_rgInfo;

            if (m_nFileIdx == m_rgstrFiles.Length)
                throw new Exception("Invalid field index.");

            using (FileStream fs = File.OpenRead(m_rgstrFiles[m_nFileIdx]))
            using (WAVReader wav = new WAVReader(fs))
            {
                wav.ReadToEnd(true);

                m_rgInfo = new Dictionary<string, float>();
                m_rgInfo.Add("AveBytesPerSec", wav.Format.nAvgBytesPerSec);
                m_rgInfo.Add("BlockAlign", wav.Format.nBlockAlign);
                m_rgInfo.Add("Channels", wav.Format.nChannels);
                m_rgInfo.Add("SamplesPerSec", wav.Format.nSamplesPerSec);
                m_rgInfo.Add("BitsPerSample", wav.Format.wBitsPerSample);
                m_rgInfo.Add("FormatTag", wav.Format.wFormatTag);

                return m_rgInfo;
            }
        }

        /// <summary>
        /// The PackBytes method packs the wav file information into a byte stream.
        /// </summary>
        /// <param name="fmt">Specifies the wav file format.</param>
        /// <param name="rgSamples">Specifies the wav file samples.</param>
        /// <returns>The byte stream is returned.</returns>
        public static byte[] PackBytes(WaveFormat fmt, List<float[]> rgSamples)
        {
            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                byte[] rgHeader = WAVWriter.StructureToByteArray<WaveFormat>(fmt);
                List<float[]> rgData = rgSamples;

                bw.Write(rgHeader.Length);
                bw.Write(rgHeader);
                bw.Write(rgData.Count);
                bw.Write(rgData[0].Length);

                for (int i = 0; i < rgData[0].Length; i++)
                {
                    for (int j = 0; j < rgData.Count; j++)
                    {
                        bw.Write(rgData[j][i]);
                    }
                }

                return ms.ToArray();
            }
        }

        /// <summary>
        /// The UnPackBytes method is used to unpack a byte stream into the Wav information.
        /// </summary>
        /// <param name="rg">Specifies the byte stream.</param>
        /// <param name="fmt">Returns the WAV file format.</param>
        /// <returns>Returns the WAV file samples.</returns>
        public static List<double[]> UnPackBytes(byte[] rg, out WaveFormat fmt)
        {
            using (MemoryStream ms = new MemoryStream(rg))
            using (BinaryReader br = new BinaryReader(ms))
            {
                int nLen = br.ReadInt32();
                byte[] rgFmt = br.ReadBytes(nLen);
                fmt = WAVWriter.ByteArrayToStructure<WaveFormat>(rgFmt);
                int nCh = br.ReadInt32();
                int nS = br.ReadInt32();

                List<double[]> rgData = new List<double[]>();
                for (int i = 0; i < nCh; i++)
                {
                    rgData.Add(new double[nS]);
                }

                for (int i = 0; i < nS; i++)
                {
                    for (int j = 0; j < nCh; j++)
                    {
                        float fVal = br.ReadSingle();
                        rgData[j][i] = fVal;
                    }
                }

                return rgData;
            }
        }

        /// <summary>
        /// The GetQuerySize method returns the size of the query as {1,1,filesize}.
        /// </summary>
        /// <returns>The query size is returned.</returns>
        public List<int> GetQuerySize()
        {
            List<int> rgSize = new List<int>();

            using (FileStream fs = File.OpenRead(m_rgstrFiles[m_nFileIdx]))
            using (WAVReader wav = new WAVReader(fs))
            {
                wav.ReadToEnd(true);
                rgSize.Add(1);
                rgSize.Add((int)wav.Format.nChannels);
                rgSize.Add(wav.SampleCount);

                return rgSize;
            }
        }

        /// <summary>
        /// Reset the file index to the first file.
        /// </summary>
        public void Reset()
        {
            m_nFileIdx = 0;
        }

        /// <summary>
        /// Converts the output values into the native type used by the CustomQuery.
        /// </summary>
        /// <param name="rg">Specifies the raw output data.</param>
        /// <param name="strType">Returns the output type.</param>
        /// <returns>The converted output data is returned as a byte stream.</returns>
        public byte[] ConvertOutput(float[] rg, out string strType)
        {
            strType = "WAV";
            Dictionary<string, float> rgInfo = QueryInfo();
            WaveFormat fmt = new WaveFormat();

            fmt.nAvgBytesPerSec = (uint)rgInfo["AveBytesPerSec"];
            fmt.nBlockAlign = (ushort)rgInfo["BlockAlign"];
            fmt.nChannels = (ushort)rgInfo["Channels"];
            fmt.nSamplesPerSec = (uint)rgInfo["SamplesPerSec"];
            fmt.wBitsPerSample = (ushort)rgInfo["BitsPerSample"];
            fmt.wFormatTag = (ushort)rgInfo["FormatTag"];

            List<float[]> rgrgSamples = new List<float[]>();
            for (int i = 0; i < fmt.nChannels; i++)
            {
                rgrgSamples.Add(rg);
            }

            return PackBytes(fmt, rgrgSamples);
        }
    }
}
