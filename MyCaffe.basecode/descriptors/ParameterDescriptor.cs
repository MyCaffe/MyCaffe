using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode.descriptors
{
    /// <summary>
    /// The ParameterDescriptor class describes a parameter in the database.
    /// </summary>
    [Serializable]
    public class ParameterDescriptor
    {
        int m_nID;
        string m_strName;
        string m_strValue;

        /// <summary>
        /// The ParameterDescriptor constructor.
        /// </summary>
        /// <param name="nID">Specifies the database ID of the item.</param>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="strValue">Specifies the value of the item.</param>
        public ParameterDescriptor(int nID, string strName, string strValue)
        {
            m_nID = nID;
            m_strName = strName;
            m_strValue = strValue;
        }

        /// <summary>
        /// The ParameterDescriptor constructor.
        /// </summary>
        /// <param name="p">Specifies another ParameterDescriptor used to create this one.</param>
        public ParameterDescriptor(ParameterDescriptor p)
            : this(p.ID, p.Name, p.Value)
        {
        }

        /// <summary>
        /// Creates a copy of the parameter descriptor.
        /// </summary>
        /// <returns>The copy of the descriptor is returned.</returns>
        public virtual ParameterDescriptor Clone()
        {
            return new ParameterDescriptor(ID, Name, Value);
        }

        /// <summary>
        /// Return the database ID of the item.
        /// </summary>
        public int ID
        {
            get { return m_nID; }
        }

        /// <summary>
        /// Return the name of the item.
        /// </summary>
        public string Name
        {
            get { return m_strName; }
        }

        /// <summary>
        /// Get/set the value of the item.
        /// </summary>
        public string Value
        {
            get { return m_strValue; }
            set { m_strValue = value; }
        }

        /// <summary>
        /// Creates the string representation of the descriptor.
        /// </summary>
        /// <returns>The string representation of the descriptor is returned.</returns>
        public override string ToString()
        {
            return m_strName + " -> " + m_strValue;
        }
    }
}
