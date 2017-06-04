using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyCaffe.param
{
    /// <summary>
    /// Specifies a collection of BlobProtos.
    /// </summary>
    public class BlobProtoCollection : IEnumerable<BlobProto>
    {
        List<BlobProto> m_rgProtos = new List<BlobProto>();

        /// <summary>
        /// The BlobProtoCollection Constructor.
        /// </summary>
        public BlobProtoCollection()
        {
        }

        /// <summary>
        /// Specifies the number of items in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgProtos.Count; }
        }

        /// <summary>
        /// Get/set a given element in the collection.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the element.</param>
        /// <returns>The element at the given <i>nIdx</i> is returned.</returns>
        public BlobProto this[int nIdx]
        {
            get { return m_rgProtos[nIdx]; }
            set { m_rgProtos[nIdx] = value; }
        }

        /// <summary>
        /// Add a new BlobProto to the collection.
        /// </summary>
        /// <param name="bp">Specifies the BlobProto.</param>
        public void Add(BlobProto bp)
        {
            m_rgProtos.Add(bp);
        }

        /// <summary>
        /// Remove a BlobProto at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        public void RemoveAt(int nIdx)
        {
            m_rgProtos.RemoveAt(nIdx);
        }

        /// <summary>
        /// Remove a given BlobProto if it exists in the collection.
        /// </summary>
        /// <param name="bp">Specifies the BlobProto to remove.</param>
        /// <returns>If the BlobProto is found in the collection and removed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Remove(BlobProto bp)
        {
            return m_rgProtos.Remove(bp);
        }

        /// <summary>
        /// Remove all elements from the collection.
        /// </summary>
        public void Clear()
        {
            m_rgProtos.Clear();
        }

        /// <summary>
        /// Retrive the enumerator for the collection.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        public IEnumerator<BlobProto> GetEnumerator()
        {
            return m_rgProtos.GetEnumerator();
        }

        /// <summary>
        /// Retrieve the enumerator for the collection.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return m_rgProtos.GetEnumerator();
        }
    }
}
