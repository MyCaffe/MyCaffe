using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MyCaffe.basecode.SimpleDatum;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The BinaryData class is used to pack and unpack DataCriteria binary data, optionally stored within each SimpleDatum.
    /// </summary>
    public class BinaryData
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        public BinaryData()
        {
        }

        /// <summary>
        /// Unpack the type packed into the byte array (if any).
        /// </summary>
        /// <param name="rg">Specifies the byte array.</param>
        /// <returns>The DATA_FORMAT of the byte array is returned.</returns>
        public static DATA_FORMAT UnPackType(byte[] rg)
        {
            using (MemoryStream ms = new MemoryStream(rg))
            using (BinaryReader br = new BinaryReader(ms))
            {
                int nFmt = br.ReadInt32();

                if (nFmt != (int)DATA_FORMAT.LIST_DOUBLE &&
                    nFmt != (int)DATA_FORMAT.LIST_FLOAT)
                    return DATA_FORMAT.NONE;

                return (DATA_FORMAT)nFmt;
            }
        }

        /// <summary>
        /// Pack a list of <i>double</i> into a byte array.
        /// </summary>
        /// <param name="rg">Specifies the list of <i>double</i> values to pack.</param>
        /// <param name="fmt">Returns the format LIST_DOUBLE</param>
        /// <returns>The byte array is returned.</returns>
        public static byte[] Pack(List<double> rg, out DATA_FORMAT fmt)
        {
            fmt = DATA_FORMAT.LIST_DOUBLE;

            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                bw.Write((int)fmt);
                bw.Write(rg.Count);

                for (int i = 0; i < rg.Count; i++)
                {
                    bw.Write(rg[i]);
                }

                bw.Flush();
                return ms.ToArray();
            }
        }

        /// <summary>
        /// Unpack the byte array into a list of <i>double</i> values.
        /// </summary>
        /// <param name="rg">Specifies the byte array containing the list.</param>
        /// <param name="fmtExpected">Specifies the expected format.</param>
        /// <returns>The list of <i>double</i> values is returned.</returns>
        public static List<double> UnPackDoubleList(byte[] rg, DATA_FORMAT fmtExpected)
        {
            using (MemoryStream ms = new MemoryStream(rg))
            using (BinaryReader br = new BinaryReader(ms))
            {
                DATA_FORMAT fmt1 = (DATA_FORMAT)br.ReadInt32();

                if (fmtExpected != DATA_FORMAT.LIST_DOUBLE)
                    throw new Exception("The format expected should be DATA_FORMAT.LIST_DOUBLE, but instead it is '" + fmtExpected.ToString() + "'.");

                if (fmt1 != fmtExpected)
                    throw new Exception("Invalid data format, expected '" + fmtExpected.ToString() + "' but found '" + fmt1.ToString() + "'.");

                int nCount = br.ReadInt32();
                List<double> rgData = new List<double>();

                for (int i = 0; i < nCount; i++)
                {
                    rgData.Add(br.ReadDouble());
                }

                return rgData;
            }
        }

        /// <summary>
        /// Pack a list of <i>float</i> into a byte array.
        /// </summary>
        /// <param name="rg">Specifies the list of <i>float</i> values to pack.</param>
        /// <param name="fmt">Returns the format LIST_FLOAT</param>
        /// <returns>The byte array is returned.</returns>
        public static byte[] Pack(List<float> rg, out DATA_FORMAT fmt)
        {
            fmt = DATA_FORMAT.LIST_FLOAT;

            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                bw.Write((int)fmt);
                bw.Write(rg.Count);

                for (int i = 0; i < rg.Count; i++)
                {
                    bw.Write(rg[i]);
                }

                bw.Flush();
                return ms.ToArray();
            }
        }

        /// <summary>
        /// Unpack the byte array into a list of <i>float</i> values.
        /// </summary>
        /// <param name="rg">Specifies the byte array containing the list.</param>
        /// <param name="fmtExpected">Specifies the expected format.</param>
        /// <returns>The list of <i>float</i> values is returned.</returns>
        public static List<float> UnPackFloatList(byte[] rg, DATA_FORMAT fmtExpected)
        {
            using (MemoryStream ms = new MemoryStream(rg))
            using (BinaryReader br = new BinaryReader(ms))
            {
                DATA_FORMAT fmt1 = (DATA_FORMAT)br.ReadInt32();

                if (fmtExpected != DATA_FORMAT.LIST_FLOAT)
                    throw new Exception("The format expected should be DATA_FORMAT.LIST_FLOAT, but instead it is '" + fmtExpected.ToString() + "'.");

                if (fmt1 != fmtExpected)
                    throw new Exception("Invalid data format, expected '" + fmtExpected.ToString() + "' but found '" + fmt1.ToString() + "'.");

                int nCount = br.ReadInt32();
                List<float> rgData = new List<float>();

                for (int i = 0; i < nCount; i++)
                {
                    rgData.Add(br.ReadSingle());
                }

                return rgData;
            }
        }
    }
}
