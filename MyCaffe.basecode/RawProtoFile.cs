using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The RawProtoFile class writes and reads prototxt to and from a file.
    /// </summary>
    public class RawProtoFile
    {
        /// <summary>
        /// The RawProtoFile constructor.
        /// </summary>
        public RawProtoFile()
        {
        }

        /// <summary>
        /// Saves the RawProto to a file.
        /// </summary>
        /// <param name="p">Specifies the RawProto.</param>
        /// <param name="strFileName">Specifies the file name where the RawProto is to be saved.</param>
        public static void SaveToFile(RawProto p, string strFileName)
        {
            string strBody = "";

            if (p.Name != "root")
            {
                strBody = p.ToString();
            }
            else
            {
                foreach (RawProto child in p.Children)
                {
                    strBody += child.ToString();
                    strBody += Environment.NewLine;
                }
            }

            using (StreamWriter sw = new StreamWriter(strFileName))
            {
                sw.Write(strBody);
            }
        }

        /// <summary>
        /// Loads a RawProto from a prototxt file.
        /// </summary>
        /// <param name="strFileName">Specifies the file name.</param>
        /// <returns>The RawProto read in is returned.</returns>
        public static RawProto LoadFromFile(string strFileName)
        {
            string strBody = "";

            using (StreamReader sr = new StreamReader(strFileName))
            {
                strBody = sr.ReadToEnd();
            }

            return RawProto.Parse(strBody);
        }
    }
}
