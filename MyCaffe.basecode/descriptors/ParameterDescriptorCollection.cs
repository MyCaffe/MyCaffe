using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The ParameterDescriptorCollection class contains a list of ParameterDescriptor's.
    /// </summary>
    [Serializable]
    public class ParameterDescriptorCollection : IEnumerable<ParameterDescriptor>
    {
        List<ParameterDescriptor> m_rgParams = new List<ParameterDescriptor>();

        /// <summary>
        /// The ParameterDescriptorCollection constructor.
        /// </summary>
        public ParameterDescriptorCollection()
        {
        }

        /// <summary>
        /// The ParameterDescriptorCollection constructor.
        /// </summary>
        /// <param name="rg">Specifies another collection used to create this one.</param>
        public ParameterDescriptorCollection(ParameterDescriptorCollection rg)
        {
            foreach (ParameterDescriptor p in rg)
            {
                m_rgParams.Add(new ParameterDescriptor(p));
            }
        }

        /// <summary>
        /// Returns the count of items in the collection.
        /// </summary>
        public int Count
        {
            get { return m_rgParams.Count; }
        }

        /// <summary>
        /// Returns the item at a given index within the collection.
        /// </summary>
        /// <param name="nIdx">Specifies the index.</param>
        /// <returns>Returns the item as the index.</returns>
        public ParameterDescriptor this[int nIdx]
        {
            get { return m_rgParams[nIdx]; }
        }

        /// <summary>
        /// Adds a ParameterDescriptor to the collection.
        /// </summary>
        /// <param name="p">Specifies the item to add.</param>
        public void Add(ParameterDescriptor p)
        {
            m_rgParams.Add(p);
        }

        /// <summary>
        /// Searches for a parameter by name in the collection.
        /// </summary>
        /// <param name="strName">Specifies the name to look for.</param>
        /// <returns>If found, the item is returned, otherwise <i>null</i> is returned.</returns>
        public ParameterDescriptor Find(string strName)
        {
            foreach (ParameterDescriptor p in m_rgParams)
            {
                if (p.Name == strName)
                    return p;
            }

            return null;
        }

        /// <summary>
        /// Searches for a parameter by name and returns its string value if found, or a default string value if not.
        /// </summary>
        /// <param name="strName">Specifies the name to look for.</param>
        /// <param name="strDefault">Specifies the default value to return if not found.</param>
        /// <returns>The string value of the named parameter is returned if found, otherwise the default string value is returned.</returns>
        public string Find(string strName, string strDefault)
        {
            ParameterDescriptor p = Find(strName);
            if (p == null)
                return strDefault;

            return p.Value;
        }

        /// <summary>
        /// Searches for a parameter by name and returns its value as a <i>double</i> if found, or a default <i>double</i> value if not.
        /// </summary>
        /// <param name="strName">Specifies the name to look for.</param>
        /// <param name="dfDefault">Specifies the default value to return if not found.</param>
        /// <returns>The <i>double</i> value of the named parameter is returned if found, otherwise the default <i>double</i> value is returned.</returns>
        public double Find(string strName, double dfDefault)
        {
            string strVal = Find(strName, null);
            if (strVal == null)
                return dfDefault;

            return double.Parse(strVal);
        }

        /// <summary>
        /// Searches for a parameter by name and returns its value as a <i>bool</i> if found, or a default <i>bool</i> value if not.
        /// </summary>
        /// <param name="strName">Specifies the name to look for.</param>
        /// <param name="bDefault">Specifies the default value to return if not found.</param>
        /// <returns>The <i>bool</i> value of the named parameter is returned if found, otherwise the default <i>bool</i> value is returned.</returns>
        public bool Find(string strName, bool bDefault)
        {
            string strVal = Find(strName, null);
            if (strVal == null)
                return bDefault;

            return bool.Parse(strVal); 
        }

        /// <summary>
        /// Returns the enumerator of the collection.
        /// </summary>
        /// <returns>The collection's enumerator is returned.</returns>
        public IEnumerator<ParameterDescriptor> GetEnumerator()
        {
            return m_rgParams.GetEnumerator();
        }

        /// <summary>
        /// Returns the enumerator of the collection.
        /// </summary>
        /// <returns>The collection's enumerator is returned.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgParams.GetEnumerator();
        }
    }
}
