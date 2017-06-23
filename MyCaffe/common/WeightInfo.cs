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
        Dictionary<string, List<int>> m_rgBlobInfo = new Dictionary<string, List<int>>();

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
        public void AddBlob(string strName, List<int> rgShape)
        {
            if (m_rgBlobInfo.ContainsKey(strName))
                throw new Exception("The blob name is already used.");

            m_rgBlobInfo.Add(strName, rgShape);
        }

        /// <summary>
        /// Add a blob name and shape to the WeightInfo.
        /// </summary>
        /// <param name="b">Specifies the Blob who's name and shape are to be added.</param>
        public void AddBlob(Blob<T> b)
        {
            m_rgBlobInfo.Add(b.Name, b.shape());
        }

        /// <summary>
        /// Returns the list of blob information describing the weights.
        /// </summary>
        public Dictionary<string, List<int>> Blobs
        {
            get { return m_rgBlobInfo; }
        }
    }
}
