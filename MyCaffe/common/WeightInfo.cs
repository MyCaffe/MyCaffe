using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The WeightInfo class describes the weights of a given weight set including
    /// the blob names and sizes of the weights.
    /// </summary>
    /// <typeparam name="T">Specifies the base type <i>float</i> or <i>double</i>.  Using <i>float</i> is recommended to conserve GPU memory.</typeparam>
    public class WeightInfo<T>
    {
        Dictionary<string, Tuple<List<int>, BLOB_TYPE>> m_rgBlobInfo = new Dictionary<string, Tuple<List<int>, BLOB_TYPE>>();
        BlobName m_names = new BlobName();

        /// <summary>
        /// The constructor.
        /// </summary>
        public WeightInfo()
        {
        }

        /// <summary>
        /// Add a blob name and shape to the WeightInfo.
        /// </summary>
        /// <param name="strName">Specifies the Blob name.</param>
        /// <param name="rgShape">Specifies the Blob shape.</param>
        /// <param name="type">Specifies the Blob type.</param>
        public void AddBlob(string strName, List<int> rgShape, BLOB_TYPE type)
        {
            strName = m_names.GetName(strName);
            m_rgBlobInfo.Add(strName, new Tuple<List<int>, BLOB_TYPE>(rgShape, type));
        }

        /// <summary>
        /// Add a blob name and shape to the WeightInfo.
        /// </summary>
        /// <param name="b">Specifies the Blob who's name and shape are to be added.</param>
        public void AddBlob(Blob<T> b)
        {
            string strName = m_names.GetName(b.Name);
            m_rgBlobInfo.Add(strName, new Tuple<List<int>, BLOB_TYPE>(b.shape(), b.type));
        }

        /// <summary>
        /// Returns the list of blob information describing the weights.  Each entry within the Dictionary returned contains
        /// the Blob's name and the Blob's dimensions (e.g. {num, channels, height, width}) as a List of integers.
        /// </summary>
        public Dictionary<string, Tuple<List<int>, BLOB_TYPE>> Blobs
        {
            get { return m_rgBlobInfo; }
        }
    }

    /// <summary>
    /// The BlobName class is used to build unique blob names.
    /// </summary>
    public class BlobName
    {
        Dictionary<string, int> m_rgNames = new Dictionary<string, int>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public BlobName()
        {
        }

        /// <summary>
        /// Returns a unique blob name.
        /// </summary>
        /// <param name="strName">Specifies the original name of the blob.</param>
        /// <returns>A unique blob name is returned.</returns>
        public string GetName(string strName)
        {
            if (string.IsNullOrEmpty(strName))
                strName = "b";

            if (!m_rgNames.ContainsKey(strName))
            {
                m_rgNames.Add(strName, 1);
            }
            else
            {
                m_rgNames[strName]++;
                strName = strName + m_rgNames[strName].ToString();
            }

            return strName;
        }
    }
}
