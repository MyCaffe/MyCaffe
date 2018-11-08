using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.stream
{
    /// <summary>
    /// The StandardQueryTextFile provides queries that read text (*.txt) files residing in a given directory.
    /// </summary>
    public class StandardQueryTextFile : IXCustomQuery
    {
        string m_strPath;
        string[] m_rgstrFiles;
        int m_nFileIdx = 0;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strParam">Specifies the parameters which shold contains the 'FilePath'=path key=value pair.</param>
        public StandardQueryTextFile(string strParam = null)
        {
            if (strParam != null)
            {
                strParam = ParamPacker.UnPack(strParam);
                PropertySet ps = new PropertySet(strParam);
                m_strPath = ps.GetProperty("FilePath");
            }
        }

        /// <summary>
        /// Returns the QUERY_TYPE of BYTE.
        /// </summary>
        public CUSTOM_QUERY_TYPE QueryType
        {
            get { return CUSTOM_QUERY_TYPE.BYTE; }
        }

        /// <summary>
        /// Returns the custom query name 'StdTextFileQuery'.
        /// </summary>
        public string Name
        {
            get { return "StdTextFileQuery"; }
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
            return new StandardQueryTextFile(strParam);
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

                if (fi.Extension.ToLower() == ".txt")
                    rgstr.Add(rgstrFiles[i]);
            }

            m_rgstrFiles = rgstr.ToArray();

            if (m_rgstrFiles.Length == 0)
                throw new Exception("The CustomTextQuery could not find any text (*.txt) files to load.");
        }

        /// <summary>
        /// The QueryByTime method is not implemented for this custom query.
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
        /// The QueryBytes method returns the bytes of the next file in the directory.
        /// </summary>
        /// <returns>All bytes of the file are returned.</returns>
        public byte[] QueryBytes()
        {
            if (m_nFileIdx == m_rgstrFiles.Length)
                return null;

            using (StreamReader sr = new StreamReader(m_rgstrFiles[m_nFileIdx]))
            {
                m_nFileIdx++;
                return Encoding.ASCII.GetBytes(sr.ReadToEnd());
            }
        }

        /// <summary>
        /// The QueryReal method is not implemented for this custom query.
        /// </summary>
        /// <returns>not used.</returns>
        public List<double[]> QueryReal()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// The Query information returns information about the data queried such as header information.
        /// </summary>
        /// <returns>The information about the data is returned.</returns>
        public Dictionary<string, float> QueryInfo()
        {
            return null;
        }

        /// <summary>
        /// The GetQuerySize method returns the size of the query as {1,1,filesize}.
        /// </summary>
        /// <returns>The query size is returned.</returns>
        public List<int> GetQuerySize()
        {
            if (m_nFileIdx == m_rgstrFiles.Length)
                return null;

            List<int> rgSize = new List<int>();

            rgSize.Add(1);
            rgSize.Add(1);

            FileInfo fi = new FileInfo(m_rgstrFiles[m_nFileIdx]);
            rgSize.Add((int)fi.Length);

            return rgSize;
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
            using (MemoryStream ms = new MemoryStream())
            {
                strType = "String";

                for (int i = 0; i < rg.Length; i++)
                {
                    int nVal = (int)Convert.ChangeType(rg[i], typeof(int));
                    char ch = (char)nVal;
                    ms.WriteByte((byte)ch);
                }

                ms.WriteByte(0);

                return ms.ToArray();
            }
        }
    }
}
