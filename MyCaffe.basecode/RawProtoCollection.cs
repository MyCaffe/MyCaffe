using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The RawProtoCollection class is a list of RawProto objects.
    /// </summary>
    public class RawProtoCollection : IEnumerable<RawProto>
    {
        List<RawProto> m_rgItems = new List<RawProto>();

        /// <summary>
        /// The RawProtoCollection constructor.
        /// </summary>
        public RawProtoCollection()
        {
        }

        /// <summary>
        /// Returns the number of items in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgItems.Count; }
        }

        /// <summary>
        /// Get/set an item at a given index within the collection.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <returns>Returns the item at the index in the collection.</returns>
        public RawProto this[int nIdx]
        {
            get { return m_rgItems[nIdx]; }
            set { m_rgItems[nIdx] = value; }
        }

        /// <summary>
        /// Inserts a new RawProto into the collection at a given index.
        /// </summary>
        /// <param name="nIdx">Specifies the index where the insert should occur.</param>
        /// <param name="p">Specifies the RawProto to insert.</param>
        public void Insert(int nIdx, RawProto p)
        {
            m_rgItems.Insert(nIdx, p);
        }

        /// <summary>
        /// Adds a RawProto to the collection.
        /// </summary>
        /// <param name="p">Specifies the RawProto to add.</param>
        public void Add(RawProto p)
        {
            m_rgItems.Add(p);
        }

        /// <summary>
        /// Adds all elements of a RawProtoCollection to this collection.
        /// </summary>
        /// <param name="col">Specifies the collection to add.</param>
        public void Add(RawProtoCollection col)
        {
            foreach (RawProto rp in col)
            {
                m_rgItems.Add(rp);
            }
        }

        /// <summary>
        /// Creates a new RawProto and adds it to this collection.
        /// </summary>
        /// <param name="strName">Specifies the name of the new RawProto.</param>
        /// <param name="strVal">Specifies the value of the new RawProto.</param>
        /// <param name="type">Specifies the type of the new RawProto.</param>
        public void Add(string strName, string strVal, RawProto.TYPE type)
        {
            m_rgItems.Add(new RawProto(strName, strVal, null, type));
        }

        /// <summary>
        /// Creates a new RawProto and adds it to this collection.
        /// </summary>
        /// <param name="strName">Specifies the name of the new RawProto.</param>
        /// <param name="strVal">Specifies the value of the new RawProto.</param>
        public void Add(string strName, string strVal)
        {
            m_rgItems.Add(new RawProto(strName, strVal));
        }

        /// <summary>
        /// Creates a new RawProto and adds it to this collection.
        /// </summary>
        /// <param name="strName">Specifies the name of the new RawProto.</param>
        /// <param name="val">Specifies a value of the new RawProto, which is converted to a string before creating the new RawProto.</param>
        public void Add(string strName, object val)
        {
            if (val == null)
                return;

            m_rgItems.Add(new RawProto(strName, val.ToString()));
        }

        /// <summary>
        /// Creates a new RawProto for each element in <i>rg</i> and gives each the name <i>strName</i>, and add all to this collection.
        /// </summary>
        /// <typeparam name="T">Specifies the type of the items in the List <i>rg</i>.</typeparam>
        /// <param name="strName">Specifies the name to give to each new RawProto created.</param>
        /// <param name="rg">Specifies a List of values to add.</param>
        public void Add<T>(string strName, List<T> rg)
        {
            if (rg == null || rg.Count == 0)
                return;

            foreach (T t in rg)
            {
                string strVal = t.ToString();

                if (typeof(T) == typeof(string))
                    strVal = "\"" + strVal + "\"";

                m_rgItems.Add(new RawProto(strName, strVal));
            }
        }

        /// <summary>
        /// Removes a RawProto from the collection.
        /// </summary>
        /// <param name="p">Specifies the RawProto to remove.</param>
        /// <returns>If found and removed, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool Remove(RawProto p)
        {
            return m_rgItems.Remove(p);
        }

        /// <summary>
        /// Removes the RawProto at a given index in the collection.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        public void RemoveAt(int nIdx)
        {
            m_rgItems.RemoveAt(nIdx);
        }

        /// <summary>
        /// Removes all items from the collection.
        /// </summary>
        public void Clear()
        {
            m_rgItems.Clear();
        }

        /// <summary>
        /// Returns the enumerator for the collection.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        public IEnumerator<RawProto> GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }

        /// <summary>
        /// Returns the enumerator for the collection.
        /// </summary>
        /// <returns>The enumerator is returned.</returns>
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return m_rgItems.GetEnumerator();
        }
    }
}
