using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static MyCaffe.basecode.SimpleDatum;

namespace MyCaffe.basecode
{
    public class BinaryData
    {
        public BinaryData()
        {
        }

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

        public static List<double> UnPackDoubleList(byte[] rg, DATA_FORMAT fmt)
        {
            using (MemoryStream ms = new MemoryStream(rg))
            using (BinaryReader br = new BinaryReader(ms))
            {
                DATA_FORMAT fmt1 = (DATA_FORMAT)br.ReadInt32();

                if (fmt != fmt1)
                    throw new Exception("Invalid data format, expected DATA_FORMAT '" + fmt1.ToString() + "'.");

                int nCount = br.ReadInt32();
                List<double> rgData = new List<double>();

                for (int i = 0; i < nCount; i++)
                {
                    rgData.Add(br.ReadDouble());
                }

                return rgData;
            }
        }

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

        public static List<float> UnPackFloatList(byte[] rg, DATA_FORMAT fmt)
        {
            using (MemoryStream ms = new MemoryStream(rg))
            using (BinaryReader br = new BinaryReader(ms))
            {
                DATA_FORMAT fmt1 = (DATA_FORMAT)br.ReadInt32();

                if (fmt != fmt1)
                    throw new Exception("Invalid data format, expected DATA_FORMAT '" + fmt1.ToString() + "'.");

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
