using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// Specifies a key-value pair of properties.
    /// </summary>
    public class PropertySet
    {
        Dictionary<string, string> m_rgProperties;

        /// <summary>
        /// The constructor, initialized with a dictionary of properties.
        /// </summary>
        /// <param name="rgProp">Specifies the properties to initialize the property set with.</param>
        public PropertySet(Dictionary<string, string> rgProp)
        {
            m_rgProperties = rgProp;
        }

        /// <summary>
        /// The constructor, initialized with a string containing a set of ';' separated key-value pairs.
        /// </summary>
        /// <param name="strProp">Specifies the set of key-value pairs separated by ';'.  Each key-value pair is in the format: 'key'='value'.</param>
        public PropertySet(string strProp)
        {
            string[] rgstr = strProp.Split(';');

            m_rgProperties = new Dictionary<string, string>();

            foreach (string strP in rgstr)
            {
                if (strP.Length > 0)
                {
                    string[] rgstrItem = strP.Split('=');
                    if (rgstrItem.Length == 2)
                    {
                        if (!m_rgProperties.ContainsKey(rgstrItem[0]))
                            m_rgProperties.Add(rgstrItem[0], rgstrItem[1]);
                    }
                }
            }
        }

        /// <summary>
        /// Returns a property as a string value.
        /// </summary>
        /// <param name="strName">Specifies the name of the property to retrieve.</param>
        /// <param name="bThrowExceptions">When <i>true</i> (the default), an exception is thrown if the property is not found, otherwise a value of <i>null</i> is returned when not found.</param>
        /// <returns>The property value is returned.</returns>
        public string GetProperty(string strName, bool bThrowExceptions = true)
        {
            if (!m_rgProperties.ContainsKey(strName))
            {
                if (bThrowExceptions)
                    throw new Exception("The property '" + strName + "' was not found!");

                return null;
            }

            return m_rgProperties[strName];
        }

        /// <summary>
        /// Returns a property as a DateTime value.
        /// </summary>
        /// <param name="strName">Specifies the name of the property.</param>
        /// <returns>The property value is returned.</returns>
        public DateTime GetPropertyAsDateTime(string strName)
        {
            string strVal = GetProperty(strName);
            DateTime dt;

            if (!DateTime.TryParse(strVal, out dt))
                throw new Exception("Failed to parse '" + strName + "' as a DateTime.  The value = '" + strVal + "'");

            return dt;
        }

        /// <summary>
        /// Returns a property as a boolean value.
        /// </summary>
        /// <param name="strName">Specifies the name of the property.</param>
        /// <param name="bDefault">Specifies the default value returned when the property is not found.</param>
        /// <returns>The property value is returned.</returns>
        public bool GetPropertyAsBool(string strName, bool bDefault = false)
        {
            string strVal = GetProperty(strName, false);
            if (strVal == null)
                return bDefault;

            bool bVal;

            if (!bool.TryParse(strVal, out bVal))
                throw new Exception("Failed to parse '" + strName + "' as an Boolean.  The value = '" + strVal + "'");

            return bVal;
        }

        /// <summary>
        /// Returns a property as an integer value.
        /// </summary>
        /// <param name="strName">Specifies the name of the property.</param>
        /// <param name="nDefault">Specifies the default value returned when the property is not found.</param>
        /// <returns>The property value is returned.</returns>
        public int GetPropertyAsInt(string strName, int nDefault = 0)
        {
            string strVal = GetProperty(strName, false);
            if (strVal == null)
                return nDefault;

            int nVal;

            if (!int.TryParse(strVal, out nVal))
                throw new Exception("Failed to parse '" + strName + "' as an Integer.  The value = '" + strVal + "'");

            return nVal;
        }

        /// <summary>
        /// Returns a property as an double value.
        /// </summary>
        /// <param name="strName">Specifies the name of the property.</param>
        /// <param name="dfDefault">Specifies the default value returned when the property is not found.</param>
        /// <returns>The property value is returned.</returns>
        public double GetPropertyAsDouble(string strName, double dfDefault = 0)
        {
            string strVal = GetProperty(strName, false);
            if (strVal == null)
                return dfDefault;

            double dfVal;

            if (!double.TryParse(strVal, out dfVal))
                throw new Exception("Failed to parse '" + strName + "' as a Double.  The value = '" + strVal + "'");

            return dfVal;
        }

        /// <summary>
        /// Returns the string representation of the properties.
        /// </summary>
        /// <returns>The string representation of the properties is returned.</returns>
        public override string ToString()
        {
            string str = "";

            foreach (KeyValuePair<string, string> kv in m_rgProperties)
            {
                str += kv.Key + "=" + kv.Value;
                str += ";";
            }

            return str;
        }
    }
}
