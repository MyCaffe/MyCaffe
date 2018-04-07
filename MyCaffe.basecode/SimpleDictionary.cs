using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyCaffe.basecode
{
    /// <summary>
    /// The SimpleDictionary is a dictionary used to store a set of key/value pairs, primarily as the DICTIONARY Data Criteria type.
    /// </summary>
    public class SimpleDictionary
    {
        Dictionary<string, object> m_rgValues = new Dictionary<string, object>();
        Dictionary<string, TYPE> m_rgTypes = new Dictionary<string, TYPE>();

        /// <summary>
        /// Defines the value type of each element.
        /// </summary>
        public enum TYPE
        {
            /// <summary>
            /// Specifies no type.
            /// </summary>
            NONE = -1,
            /// <summary>
            /// Specifies a text string value.
            /// </summary>
            STRING = 0,
            /// <summary>
            /// Specifies a <i>double</i> value.
            /// </summary>
            NUMERIC = 1,
            /// <summary>
            /// Specifies a 32-bit <i>integer</i> value.
            /// </summary>
            INTEGER = 2
        }

        /// <summary>
        /// The constructor.
        /// </summary>
        public SimpleDictionary()
        {
        }

        /// <summary>
        /// Returns the type of a given item.
        /// </summary>
        /// <param name="strName">Specifies the name of the item.</param>
        /// <param name="type">Returns the type of the item, if found in the dictionary.</param>
        /// <returns>If the variable 'strName' exists in the dictionary, <i>true</i> is returned, otherwise <i>false</i> is returned.</returns>
        public bool GetType(string strName, out TYPE type)
        {
            type = TYPE.NONE;

            if (!m_rgTypes.ContainsKey(strName))
                return false;

            type = m_rgTypes[strName];

            return true;
        }

        /// <summary>
        /// Returns the numeric value of an item.
        /// </summary>
        /// <param name="strName">Specifies the name of the item to get.</param>
        /// <returns>The numeric value of the item is returned.</returns>
        public double GetNumeric(string strName)
        {
            if (m_rgTypes[strName] != TYPE.NUMERIC)
                throw new Exception("Invalid type, expected '" + m_rgTypes[strName].ToString() + "'");

            return (double)m_rgValues[strName];
        }

        /// <summary>
        /// Returns the integer value of an item.
        /// </summary>
        /// <param name="strName">Specifies the name of the item to get.</param>
        /// <param name="nDefault">Specifies a default value, which when specified is returned if no value is found.</param>
        /// <returns>The integer value of the item is returned.</returns>
        public int GetInteger(string strName, int? nDefault = null)
        {
            if (!m_rgTypes.ContainsKey(strName))
            {
                if (nDefault.HasValue)
                    return nDefault.Value;

                throw new Exception("The variable '" + strName + "' is not in the dictionary.");
            }

            if (m_rgTypes[strName] != TYPE.INTEGER)
                throw new Exception("Invalid type, expected '" + m_rgTypes[strName].ToString() + "'");

            return (int)m_rgValues[strName];
        }

        /// <summary>
        /// Returns the string value of an item.
        /// </summary>
        /// <param name="strName">Specifies the name of the item to get.</param>
        /// <returns>The string value of the item is returned.</returns>
        public string GetString(string strName)
        {
            if (m_rgTypes[strName] != TYPE.STRING)
                throw new Exception("Invalid type, expected '" + m_rgTypes[strName].ToString() + "'");

            return (string)m_rgValues[strName];
        }

        /// <summary>
        /// Add a new string item to the dictionary.
        /// </summary>
        /// <param name="strName">Specifies the item name.</param>
        /// <param name="strVal">Specifies the item value.</param>
        public void Add(string strName, string strVal)
        {
            m_rgValues.Add(strName, strVal);
            m_rgTypes.Add(strName, TYPE.STRING);
        }

        /// <summary>
        /// Add a new integer item to the dictionary.
        /// </summary>
        /// <param name="strName">Specifies the item name.</param>
        /// <param name="nVal">Specifies the item value.</param>
        public void Add(string strName, int nVal)
        {
            m_rgValues.Add(strName, nVal);
            m_rgTypes.Add(strName, TYPE.INTEGER);
        }

        /// <summary>
        /// Add a new numeric item to the dictionary.
        /// </summary>
        /// <param name="strName">Specifies the item name.</param>
        /// <param name="dfVal">Specifies the item value.</param>
        public void Add(string strName, double dfVal)
        {
            m_rgValues.Add(strName, dfVal);
            m_rgTypes.Add(strName, TYPE.NUMERIC);
        }

        /// <summary>
        /// Get the list of values in the dictionary.
        /// </summary>
        /// <returns>The list of values is returned.</returns>
        public List<KeyValuePair<string, object>> ToList()
        {
            return m_rgValues.ToList();
        }

        /// <summary>
        /// Converts the dictionary to a byte array.
        /// </summary>
        /// <returns>The byte array is returned.</returns>
        public byte[] ToByteArray()
        {
            using (MemoryStream ms = new MemoryStream())
            {
                BinaryWriter bw = new BinaryWriter(ms);

                bw.Write(m_rgValues.Count);

                List<KeyValuePair<string, object>> rgValues = m_rgValues.ToList();
                List<KeyValuePair<string, TYPE>> rgTypes = m_rgTypes.ToList();

                for (int i = 0; i < rgValues.Count; i++)
                {
                    bw.Write(rgTypes[i].Key);
                    bw.Write((byte)rgTypes[i].Value);

                    switch (rgTypes[i].Value)
                    {
                        case TYPE.STRING:
                            bw.Write(rgValues[i].Value.ToString());
                            break;

                        case TYPE.INTEGER:
                            bw.Write((int)rgValues[i].Value);
                            break;

                        case TYPE.NUMERIC:
                            bw.Write((double)rgValues[i].Value);
                            break;

                        default:
                            break;
                    }
                }

                return ms.ToArray();
            }
        }

        /// <summary>
        /// Creates a new dictionary from a byte array.
        /// </summary>
        /// <param name="rg">Specifies the byte array to load.</param>
        /// <returns>The new dictionary is returned.</returns>
        public static SimpleDictionary FromByteArray(byte[] rg)
        {
            SimpleDictionary dictionary = new basecode.SimpleDictionary();

            using (MemoryStream ms = new MemoryStream(rg))
            {
                BinaryReader br = new BinaryReader(ms);
                int nCount = br.ReadInt32();

                for (int i = 0; i < nCount; i++)
                {
                    string strName = br.ReadString();
                    TYPE type = (TYPE)br.ReadByte();

                    switch (type)
                    {
                        case TYPE.STRING:
                            string strVal = br.ReadString();
                            dictionary.Add(strName, strVal);
                            break;

                        case TYPE.INTEGER:
                            int nVal = br.ReadInt32();
                            dictionary.Add(strName, nVal);
                            break;

                        case TYPE.NUMERIC:
                            double dfVal = br.ReadDouble();
                            dictionary.Add(strName, dfVal);                           
                            break;

                        default:
                            break;
                    }
                }
            }

            return dictionary;
        }

        /// <summary>
        /// Convert all numeric values into a standard Dictionary.
        /// </summary>
        /// <param name="nCount">Specifies the number of numeric values in the SimpleDictionary.</param>
        /// <param name="strKey">Specifies the base key used, such as 'A'.</param>
        /// <returns>A Dictionary of string-double pairs is returned.</returns>
        public Dictionary<string, double> ToNumericValues(int nCount, string strKey)
        {
            Dictionary<string, double> rg = new Dictionary<string, double>();

            for (int i = 0; i < nCount; i++)
            {
                string strKeyName = strKey + i.ToString() + "_name";
                string strKeyVal = strKey + i.ToString() + "_val";
                TYPE typeName;
                TYPE typeVal;

                if (GetType(strKeyName, out typeName) && GetType(strKeyVal, out typeVal))
                {
                    if (typeName == TYPE.STRING && typeVal != TYPE.STRING)
                    {
                        string strName = GetString(strKeyName);
                        double dfVal = GetNumeric(strKeyVal);
                        rg.Add(strName, dfVal);
                    }
                }
            }

            return rg;
        }

        /// <summary>
        /// Convert all string values into a standard Dictionary.
        /// </summary>
        /// <param name="nCount">Specifies the number of string values in the SimpleDictionary.</param>
        /// <param name="strKey">Specifies the base key used, such as 'Atx'.</param>
        /// <returns>A Dictionary of string-string pairs is returned.</returns>
        public Dictionary<string, string> ToStringValues(int nCount, string strKey)
        {
            Dictionary<string, string> rg = new Dictionary<string, string>();

            for (int i = 0; i < nCount; i++)
            {
                string strKeyName = strKey + i.ToString() + "_name";
                string strKeyVal = strKey + i.ToString() + "_val";
                TYPE typeName;
                TYPE typeVal;

                if (GetType(strKeyName, out typeName) && GetType(strKeyVal, out typeVal))
                {
                    if (typeName == TYPE.STRING && typeVal == TYPE.STRING)
                    {
                        string strName = GetString(strKeyName);
                        string strVal = GetString(strKeyVal);
                        rg.Add(strName, strVal);
                    }
                }
            }

            return rg;
        }
    }
}
