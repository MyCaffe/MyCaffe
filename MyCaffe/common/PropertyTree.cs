using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.common
{
    /// <summary>
    /// The PropertyTree class implements a simple property tree similar to the ptree in Boost.
    /// </summary>
    public class PropertyTree
    {
        Dictionary<string, List<Property>> m_rgValues = new Dictionary<string, List<Property>>();
        Dictionary<string, List<PropertyTree>> m_rgChildren = new Dictionary<string, List<PropertyTree>>();

        /// <summary>
        /// The constructor.
        /// </summary>
        public PropertyTree()
        {
        }

        /// <summary>
        /// Add a new property string value.
        /// </summary>
        /// <param name="str">Specifies the key.</param>
        /// <param name="strVal">Specifies the string value.</param>
        public void Put(string str, string strVal)
        {
            if (!m_rgValues.ContainsKey(str))
                m_rgValues.Add(str, new List<Property>());

            m_rgValues[str].Add(new Property(strVal));
        }

        /// <summary>
        /// Add a new property numeric value.
        /// </summary>
        /// <param name="str">Specifies the key.</param>
        /// <param name="dfVal">Specifies the numeric value.</param>
        public void Put(string str, double dfVal)
        {
            if (!m_rgValues.ContainsKey(str))
                m_rgValues.Add(str, new List<Property>());

            m_rgValues[str].Add(new Property(dfVal));
        }

        /// <summary>
        /// Add a new child to the Property tree.
        /// </summary>
        /// <param name="str">Specifies the key name of the child.</param>
        /// <param name="pt">Specifies the property child tree.</param>
        public void AddChild(string str, PropertyTree pt)
        {
            if (!m_rgChildren.ContainsKey(str))
                m_rgChildren.Add(str, new List<PropertyTree>());

            m_rgChildren[str].Add(pt);
        }

        /// <summary>
        /// Retrieves a property at the current level of the tree.
        /// </summary>
        /// <param name="strName">Specifies the name of the property.</param>
        /// <returns>The property is returned.</returns>
        public Property Get(string strName)
        {
            if (!m_rgValues.ContainsKey(strName))
                throw new Exception("No value key with name '" + strName + "' found!");

            return m_rgValues[strName][0];
        }

        /// <summary>
        /// Retrieves all properties with the given key at the current level of the tree.
        /// </summary>
        /// <param name="strName">Specifies the name of the children.</param>
        /// <returns>The list of properties for the children are returned.</returns>
        public List<Property> GetChildren(string strName)
        {
            if (!m_rgValues.ContainsKey(strName))
                throw new Exception("No value key with name '" + strName + "' found!");

            return m_rgValues[strName];
        }

        /// <summary>
        /// Clear all nodes and values from the tree.
        /// </summary>
        public void Clear()
        {
            m_rgValues = new Dictionary<string, List<Property>>();
            m_rgChildren = new Dictionary<string, List<PropertyTree>>();
        }

        /// <summary>
        /// Returns a list of all child property trees within the tree.
        /// </summary>
        public List<PropertyTree> Children
        {
            get
            {
                List<PropertyTree> rgChildren = new List<PropertyTree>();

                foreach (KeyValuePair<string, List<PropertyTree>> kv in m_rgChildren)
                {
                    rgChildren.AddRange(kv.Value);
                }

                return rgChildren;
            }
        }

        /// <summary>
        /// Converts the property tree to a Json representation.
        /// </summary>
        /// <remarks>
        /// THIS METHOD IS NOT COMPLETE YET.
        /// </remarks>
        /// <returns>The Json string representing the tree is returned.</returns>
        public string ToJson()
        {
#warning PropertyTree.ToJson NOT completed.
            return "";
        }
    }

    /// <summary>
    /// The Property class stores both a numeric and text value.
    /// </summary>
    public class Property
    {
        string m_strVal;
        double? m_dfVal;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="strVal">Specifies the text value.</param>
        /// <param name="dfVal">Optionally, specifies the numeric value.</param>
        public Property(string strVal, double? dfVal = null)
        {
            m_strVal = strVal;
            m_dfVal = dfVal;
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="dfVal">Specifies the numeric value.</param>
        /// <param name="strVal">Optionally, specifies the text value.</param>
        public Property(double dfVal, string strVal = null)
        {
            m_dfVal = dfVal;
            m_strVal = strVal;
        }

        /// <summary>
        /// Returns the text value.
        /// </summary>
        public string Value
        {
            get { return m_strVal; }
        }

        /// <summary>
        /// Returns the numeric value.
        /// </summary>
        public double? Numeric
        {
            get { return m_dfVal; }
        }
    }
}
