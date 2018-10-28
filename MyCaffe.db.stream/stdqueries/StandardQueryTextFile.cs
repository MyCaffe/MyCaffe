using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.db.stream.stdqueries
{
    class StandardQueryTextFile : IXCustomQuery
    {
        string m_strPath;
        string[] m_rgstrFiles;
        int m_nFileIdx = 0;

        public StandardQueryTextFile(string strParam = null)
        {
            if (strParam != null)
            {
                strParam = ParamPacker.UnPack(strParam);
                PropertySet ps = new PropertySet(strParam);
                m_strPath = ps.GetProperty("FilePath");
            }
        }

        public CUSTOM_QUERY_TYPE QueryType
        {
            get { return CUSTOM_QUERY_TYPE.BYTE; }
        }

        public string Name
        {
            get { return "StdTextFileQuery"; }
        }

        public int FieldCount
        {
            get { return 1; }  // data
        }

        public IXCustomQuery Clone(string strParam)
        {
            return new StandardQueryTextFile(strParam);
        }

        public void Close()
        {
            m_rgstrFiles = null;
            m_nFileIdx = 0;
        }

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

        public double[] QueryByTime(DateTime dt, TimeSpan ts, int nCount)
        {
            throw new NotImplementedException();
        }

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

        public int GetQuerySize()
        {
            if (m_nFileIdx == m_rgstrFiles.Length)
                return 0;

            FileInfo fi = new FileInfo(m_rgstrFiles[m_nFileIdx]);
            return (int)fi.Length;
        }

        public void Reset()
        {
            m_nFileIdx = 0;
        }
    }
}
