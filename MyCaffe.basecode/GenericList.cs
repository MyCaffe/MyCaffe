using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The GenericList provides a base used to implement a generic list by only implementing the minimum amount of the list functionality.
    /// </summary>
    /// <typeparam name="T">The base type of the list.</typeparam>
    public class GenericList<T> : IEnumerable<T>
    {
        /// <summary>
        /// The actual list of items.
        /// </summary>
        protected List<T> m_rgItems = new List<T>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public GenericList()
        {
        }

        /// <summary>
        /// Returns the number of items in the list.
        /// </summary>
        public int Count
        {
            get { return m_rgItems.Count; }
        }

        /// <summary>
        /// Add a new item to the list.
        /// </summary>
        /// <param name="item">Specifies the item to add.</param>
        public virtual void Add(T item)
        {
            m_rgItems.Add(item);
        }

        /// <summary>
        /// Remove an item from the list.
        /// </summary>
        /// <param name="item">Specifies the item to remove.</param>
        /// <returns>If the item is found and removed <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Remove(T item)
        {
            return m_rgItems.Remove(item);
        }

        /// <summary>
        /// Remove the item at a given index in the list.
        /// </summary>
        /// <param name="nIdx">Specifies the index at which to remove the item.</param>
        public void RemoveAt(int nIdx)
        {
            m_rgItems.RemoveAt(nIdx);
        }

        /// <summary>
        /// Remove all items from the list.
        /// </summary>
        public void Clear()
        {
            m_rgItems.Clear();
        }

        /// <summary>
        /// Get the list enumerator.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        public IEnumerator<T> GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }

        /// <summary>
        /// Get the list enumerator.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }

        /// <summary>
        /// Get/set the item at an index in the list.
        /// </summary>
        /// <param name="nIdx">Specifies the index of the item to access.</param>
        /// <returns>The item is returned at the given index.</returns>
        public T this[int nIdx]
        {
            get { return m_rgItems[nIdx]; }
            set { m_rgItems[nIdx] = value; }
        }
    }
}
