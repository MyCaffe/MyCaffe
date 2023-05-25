using MyCaffe.basecode;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The NumpyFile reads data from a numpy file in the base type specified.
    /// </summary>
    /// <typeparam name="T">Specifies the base type (e.g, long, int, float, double).</typeparam>
    public class NumpyFile<T> : IDisposable
    {
        FileStream m_fs = null;
        BinaryReader m_br = null;
        Type m_dataType;
        int m_nDataTypeSize;
        int[] m_rgShape;
        long m_nHeaderSize;
        int m_nCount = 0;
        int m_nRows = 0;
        int m_nColumns = 0;
        int m_nFieldCount = 1;
        Tuple<int, int> m_count;
        Stopwatch m_sw = new Stopwatch();
        Log m_log;
        string m_strFile;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="log">Specifies the output log (or null to ignore).</param>
        public NumpyFile(Log log)
        {
            m_log = log;
        }

        /// <summary>
        /// Dispose all resources and close the file.
        /// </summary>
        public void Dispose()
        {
            Close();
        }

        /// <summary>
        /// Close the file if open.
        /// </summary>
        public void Close()
        {
            if (m_br != null)
            {
                m_br.Close();
                m_br.Dispose();
                m_br = null;
            }

            if (m_fs != null)
            {
                m_fs.Close();
                m_fs.Dispose();
                m_fs = null;
            }            
        }

        /// <summary>
        /// Return the base data type in the Numpy file.
        /// </summary>
        public Type DataType
        {
            get { return m_dataType; }
        }

        /// <summary>
        /// Return the data shape of the data in the Numpy file.
        /// </summary>
        public int[] Shape
        {
            get { return m_rgShape; }
        }

        /// <summary>
        /// Returns the number of rows.
        /// </summary>
        public int Rows
        {
            get { return m_nRows; }
        }

        /// <summary>
        /// Returns the number of items per row.
        /// </summary>
        public int Columns
        {
            get { return m_nColumns; }
        }

        /// <summary>
        /// Returns the number of fields per column item.
        /// </summary>
        public int Fields
        {
            get { return m_nFieldCount; }
        }

        /// <summary>
        /// Open the numpy file for reading, and read in the header information.
        /// </summary>
        /// <param name="strFile">Specifies the numpy file to read.</param>
        /// <exception cref="Exception">An exception is thrown when trying to read a numpy file in fortran order.</exception>
        public void OpenRead(string strFile)
        {
            m_strFile = strFile;
            m_fs = File.OpenRead(strFile);
            m_br = new BinaryReader(m_fs);

            BinaryReader br = m_br;

            byte[] rgMagic = new byte[6];
            for (int i = 0; i < rgMagic.Length; i++)
            {
                rgMagic[i] = br.ReadByte();
            }

            if (rgMagic[0] != 0x93 || rgMagic[1] != 0x4E || rgMagic[2] != 0x55 || rgMagic[3] != 0x4D || rgMagic[4] != 0x50 || rgMagic[5] != 0x59)
                throw new Exception("The file is not a valid Numpy file!");

            byte bMajor = br.ReadByte();
            byte bMinor = br.ReadByte();

            if (bMajor != 1 || bMinor != 0)
                throw new Exception("The file is not a valid Numpy file!");

            byte bHeaderLen1 = br.ReadByte();
            byte bHeaderLen2 = br.ReadByte();
            int nHeaderLen = bHeaderLen2 << 8 | bHeaderLen1;

            byte[] rgHeader = new byte[nHeaderLen];
            for (int i = 0; i < rgHeader.Length; i++)
            {
                rgHeader[i] = br.ReadByte();
            }
            string strHeader = Encoding.ASCII.GetString(rgHeader);

            bool bFortranOrder;
            m_count = parseHeaderEx(strHeader, out bFortranOrder, out m_rgShape, out m_dataType, out m_nDataTypeSize);

            if (bFortranOrder)
                throw new Exception("Currently the fortran ordering is not supported");

            m_nCount = 1;

            m_nRows = m_rgShape[0];
            m_nColumns = (m_rgShape.Length == 1) ? 1 : m_rgShape[1];

            for (int i=0; i<m_rgShape.Length; i++)
            {
                m_nCount *= m_rgShape[i];

                if (i > 1)
                    m_nFieldCount *= m_rgShape[i];
            }

            m_nHeaderSize = m_fs.Position;
        }

        private static Tuple<int, int> parseHeaderEx(string str, out bool bFortranOrder, out int[] rgShape, out Type dataType, out int nDataTypeSize, int nMax = int.MaxValue)
        {
            int nNum = 1;
            int nCount = 1;
            List<int> rgShape1 = new List<int>();
            str = str.Trim('{', '}', ' ', '\n', ',');

            dataType = typeof(object);
            nDataTypeSize = 1;

            string strShape = null;
            string strTarget = "'shape':";
            int nPos = str.IndexOf(strTarget);
            if (nPos > 0)
            {
                strShape = str.Substring(nPos + strTarget.Length);
                str = str.Substring(0, nPos);

                nPos = strShape.IndexOf(')');
                str += strShape.Substring(nPos + 1);
                str = str.Trim(',', ' ');

                strShape = strShape.Substring(0, nPos);
                strShape = strShape.Trim(' ', '(', ')');
                string[] rgShapeStr = strShape.Split(',');

                for (int i = 0; i < rgShapeStr.Count(); i++)
                {
                    string strShape1 = rgShapeStr[i];
                    if (!string.IsNullOrEmpty(strShape1))
                    {
                        int nShape = int.Parse(strShape1);

                        if (i == 0 && nShape > nMax)
                            nShape = nMax;

                        rgShape1.Add(nShape);

                        if (i == 0)
                            nNum = rgShape1[rgShape1.Count - 1];
                        else
                            nCount *= rgShape1[rgShape1.Count - 1];
                    }
                }
            }

            rgShape = rgShape1.ToArray();
            bFortranOrder = false;

            string[] rgstr = str.Split(',');
            foreach (string str1 in rgstr)
            {
                string[] rgstrKeyVal = str1.Split(':');
                if (rgstrKeyVal.Length != 2)
                    throw new Exception("Invalid header key value, '" + str1 + "'!");

                string strKey = rgstrKeyVal[0].Trim('\'', ' ');
                string strVal = rgstrKeyVal[1].Trim('\'', ' ');

                switch (strKey)
                {
                    case "descr":
                        if (strVal == "<f4")
                            dataType = typeof(float);
                        else if (strVal == "<f8")
                            dataType = typeof(double);
                        else if (strVal == "<i4")
                            dataType = typeof(int);
                        else if (strVal == "<i8")
                            dataType = typeof(long);
                        else if (strVal == "|b1")
                            dataType = typeof(bool);
                        else if (strVal.StartsWith("<U"))
                        {
                            strVal = strVal.Substring(2);
                            nDataTypeSize = int.Parse(strVal);
                            dataType = typeof(string);
                        }
                        else
                            throw new Exception("Unsupported data type '" + strVal + "', currenly only support '<f4'");
                        break;

                    case "fortran_order":
                        bFortranOrder = bool.Parse(strVal);
                        break;
                }
            }

            nDataTypeSize = Marshal.SizeOf(dataType);

            return new Tuple<int, int>(nNum, nCount);
        }

        /// <summary>
        /// Load a single row (or portion of a row) from the numpy file.
        /// </summary>
        /// <param name="rgVal">Specifies the array where data is copied (this value is also returned).</param>
        /// <param name="nRowIdx">Specifies the row index.</param>
        /// <param name="nStartIdx">Specifies the start index into the row (default = 0).</param>
        /// <param name="nColumnCount">Specifies the number of items to read from the start index (default = int.MaxValue to read entire row).</param>
        /// <returns>The data read in is returned in the template type specified.</returns>
        /// <exception cref="Exception">An exception is thrown on error.</exception>
        public T[] LoadRow(T[] rgVal, int nRowIdx, int nStartIdx = 0, int nColumnCount = int.MaxValue)
        {
            if (m_br == null)
                throw new Exception("The file is not open!");

            if (nRowIdx >= m_nRows)
                throw new Exception("The row index '" + nRowIdx.ToString() + "' is out of range!");

            if (nStartIdx >= m_nColumns)
                throw new Exception("The start index '" + nStartIdx.ToString() + "' is out of range!");

            if (nColumnCount == int.MaxValue)
                nColumnCount = m_nColumns - nStartIdx;
            else if (nStartIdx + nColumnCount > m_nColumns)
                return null;

            int nSize = nColumnCount * m_nFieldCount * m_nDataTypeSize;

            if (nStartIdx > 0)
            {
                long nOffset = m_nHeaderSize + (nRowIdx * m_nColumns + nStartIdx) * m_nFieldCount * m_nDataTypeSize;
                m_fs.Seek(nOffset, SeekOrigin.Begin);
            }

            byte[] rgData = m_br.ReadBytes(nSize);
            int nItemCount = nColumnCount * m_nFieldCount;
            
            if (rgVal == null || rgVal.Length != nItemCount)
                rgVal = new T[nItemCount];

            Buffer.BlockCopy(rgData, 0, rgVal, 0, rgData.Length);

            return rgVal;
        }

        /// <summary>
        /// Load the data from the numpy file, optionally specifying the starting row index and number of rows to read, each row is read in its entirety.
        /// </summary>
        /// <param name="nStartIdx"></param>
        /// <param name="nCount"></param>
        /// <returns>The full set of data read is returned.  For example when reading the entire numpy file the returned data is of size row_count x col_size.</returns>
        /// <exception cref="Exception">An exception is thrown on error.</exception>
        public List<T[]> Load(int nStartIdx = 0, int nCount = int.MaxValue)
        {
            if (m_br == null)
                throw new Exception("The file is not open!");

            if (nStartIdx >= m_rgShape[0])
                throw new Exception("The start index '" + nStartIdx.ToString() + "' is out of range!");

            if (m_dataType == typeof(string))
                throw new Exception("String data types not supported.");

            if (nStartIdx + nCount > m_rgShape[0])
                nCount = m_rgShape[1] - nStartIdx;

            List<T[]> rgVal = new List<T[]>();

            if (m_nCount > 0)
            {
                // Skip ahead to start index (if greater than zero).
                if (nStartIdx > 0)
                {
                    long nItems = 1;

                    for (int i = 1; i < m_rgShape.Length; i++)
                    {
                        nItems *= m_rgShape[i];
                    }

                    long lSeekPos = m_nHeaderSize + nStartIdx * nItems * m_nDataTypeSize;
                    m_fs.Seek(lSeekPos, SeekOrigin.Begin);
                }

                for (int i = nStartIdx; i < nStartIdx + nCount; i++)
                {
                    T[] rgItemT = new T[m_count.Item2 * m_nDataTypeSize];
                    byte[] rgItem = m_br.ReadBytes(m_count.Item2 * m_nDataTypeSize);
                    Buffer.BlockCopy(rgItem, 0, rgItemT, 0, rgItem.Length);

                    if (m_log != null)
                    {
                        if (m_sw.Elapsed.TotalMilliseconds > 1000)
                        {
                            double dfPct = (double)i / (nCount - nStartIdx);
                            string strOut = "Loading '" + m_strFile + "' at " + dfPct.ToString("P5") + "...";
                            m_log.WriteLine(strOut, true);
                            m_sw.Restart();
                        }
                    }

                    rgVal.Add(rgItemT);
                }
            }

            return rgVal;
        }
    }
}
