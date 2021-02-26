using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// Specifies a key-value pair of properties.
    /// </summary>
    public class PropertySet : IEnumerable<KeyValuePair<string, string>>
    {
        Dictionary<string, string> m_rgProperties = new Dictionary<string, string>();
        Dictionary<string, byte[]> m_rgBlobs = new Dictionary<string, byte[]>();

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
                    int nPos = strP.IndexOf('=');
                    if (nPos > 0)
                    {
                        string strKey = strP.Substring(0, nPos);
                        string strVal = strP.Substring(nPos + 1);

                        if (!m_rgProperties.ContainsKey(strKey))
                            m_rgProperties.Add(strKey, strVal);
                    }
                }
            }
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        public PropertySet()
        {
        }

        /// <summary>
        /// Move the property from the list of properties to the list of blobs, storing the blob as a 0 terminated string of bytes.
        /// </summary>
        /// <param name="strName">Specifies the name of the property to move.</param>
        /// <returns>If the property does not exist, <i>false</i> is returned, otherwise <i>true</i> is returned.</returns>
        public bool MovePropertyToBlob(string strName)
        {
            if (!m_rgProperties.ContainsKey(strName))
                return false;

            string strVal = m_rgProperties[strName];
            m_rgProperties.Remove(strName);

            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter bw = new BinaryWriter(ms))
            {
                foreach (char ch in strVal)
                {
                    bw.Write((byte)ch);
                }

                bw.Write((byte)0);

                m_rgBlobs.Add(strName, ms.ToArray());
            }

            return true;
        }

        /// <summary>
        /// Merge a given property set into this one.
        /// </summary>
        /// <param name="prop">Specifies the property set to merge into this one.</param>
        public void Merge(PropertySet prop)
        {
            foreach (KeyValuePair<string, string> kv in prop.m_rgProperties)
            {
                if (!m_rgProperties.ContainsKey(kv.Key))
                    m_rgProperties.Add(kv.Key, kv.Value);
                else
                    m_rgProperties[kv.Key] = kv.Value;
            }

            foreach (KeyValuePair<string, byte[]> kv in prop.m_rgBlobs)
            {
                if (!m_rgBlobs.ContainsKey(kv.Key))
                    m_rgBlobs.Add(kv.Key, kv.Value);
                else
                    m_rgBlobs[kv.Key] = kv.Value;
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
        /// Returns a property blob as a byte array value.
        /// </summary>
        /// <param name="strName">Specifies the name of the property blob to retrieve.</param>
        /// <param name="bThrowExceptions">When <i>true</i> (the default), an exception is thrown if the property blob is not found, otherwise a value of <i>null</i> is returned when not found.</param>
        /// <returns>The property blob value is returned.</returns>
        public byte[] GetPropertyBlob(string strName, bool bThrowExceptions = true)
        {
            if (!m_rgBlobs.ContainsKey(strName))
            {
                if (bThrowExceptions)
                    throw new Exception("The property '" + strName + "' was not found!");

                return null;
            }

            return m_rgBlobs[strName];
        }

        /// <summary>
        /// Sets a property in the property set to a value if it exists, otherwise it adds the new property.
        /// </summary>
        /// <param name="strName">Specifies the property name.</param>
        /// <param name="strVal">Specifies the new property value.</param>
        public void SetProperty(string strName, string strVal)
        {
            if (!m_rgProperties.ContainsKey(strName))
                m_rgProperties.Add(strName, strVal);
            else
                m_rgProperties[strName] = strVal;
        }

        /// <summary>
        /// Sets a property in the blob set to a byte array if it exists, otherwise it adds the new blob.
        /// </summary>
        /// <param name="strName">Specifies the name of the blob.</param>
        /// <param name="rg">Specifies the blob data.</param>
        public void SetPropertyBlob(string strName, byte[] rg)
        {
            if (!m_rgBlobs.ContainsKey(strName))
                m_rgBlobs.Add(strName, rg);
            else
                m_rgBlobs[strName] = rg;
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

            if (!BaseParameter.TryParse(strVal, out dfVal))
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

        /// <summary>
        /// Returns an enumerator of the key/value pairs.
        /// </summary>
        /// <returns>The key/value pair enumerator is returned.</returns>
        public IEnumerator<KeyValuePair<string, string>> GetEnumerator()
        {
            return m_rgProperties.GetEnumerator();
        }

        /// <summary>
        /// Returns an enumerator of the key/value pairs.
        /// </summary>
        /// <returns>The key/value pair enumerator is returned.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_rgProperties.GetEnumerator();
        }
    }
}
