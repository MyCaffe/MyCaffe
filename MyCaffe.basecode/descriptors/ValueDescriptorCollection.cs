using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The ValueDescriptorCollection class contains a list of ValueDescriptor's.
    /// </summary>
    [Serializable]
    public class ValueDescriptorCollection : IEnumerable<ValueDescriptor>
    {
        List<ValueDescriptor> m_rgValues = new List<ValueDescriptor>();

        /// <summary>
        /// The ValueDescriptorCollection constructor.
        /// </summary>
        public ValueDescriptorCollection()
        {
        }

        /// <summary>
        /// The ValueDescriptorCollection constructor.
        /// </summary>
        /// <param name="rg">Specifies another collection used to create this one.</param>
        public ValueDescriptorCollection(ValueDescriptorCollection rg)
        {
            foreach (ValueDescriptor v in rg)
            {
                m_rgValues.Add(new descriptors.ValueDescriptor(v));
            }
        }

        /// <summary>
        /// Returns the count of items in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgValues.Count; }
        }

        /// <summary>
        /// Returns the item at a given index within the collection.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <returns>Returns the item as the index.</returns>
        public ValueDescriptor this[int nIdx]
        {
            get { return m_rgValues[nIdx]; }
        }

        /// <summary>
        /// Adds a ValueDescriptor to the collection.
        /// </summary>
        /// <param name="p">Specifies the item to add.</param>
        public void Add(ValueDescriptor p)
        {
            m_rgValues.Add(p);
        }

        /// <summary>
        /// Searches for a parameter by name in the collection.
        /// </summary>
        /// <param name="strName">Specifies the name to look for.</param>
        /// <returns>If found, the item is returned, otherwise <i>null</i> is returned.</returns>
        public ValueDescriptor Find(string strName)
        {
            foreach (ValueDescriptor v in m_rgValues)
            {
                if (v.Name == strName)
                    return v;
            }

            return null;
        }

        /// <summary>
        /// Returns the enumerator of the collection.
        /// </summary>
        /// <returns>The collection's enumerator is returned.</returns>
        public IEnumerator<ValueDescriptor> GetEnumerator()
        {
            return m_rgValues.GetEnumerator();
        }

        /// <summary>
        /// Returns the enumerator of the collection.
        /// </summary>
        /// <returns>The collection's enumerator is returned.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgValues.GetEnumerator();
        }
    }
}
