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
            /// Specifies a text string value.
            /// </summary>
            STRING,
            /// <summary>
            /// Specifies a <i>double</i> value.
            /// </summary>
            NUMERIC,
            /// <summary>
            /// Specifies a 32-bit <i>integer</i> value.
            /// </summary>
            INTEGER
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
        /// <returns>The TYPE of the item is returned.</returns>
        public TYPE GetType(string strName)
        {
            return m_rgTypes[strName];
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
        /// <returns>The integer value of the item is returned.</returns>
        public int GetInteger(string strName)
        {
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
                        case TYPE.INTEGER:
                            bw.Write((int)rgValues[i].Value);
                            break;

                        case TYPE.NUMERIC:
                            bw.Write((double)rgValues[i].Value);
                            break;

                        default:
                            bw.Write(rgValues[i].Value.ToString());
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
                        case TYPE.INTEGER:
                            int nVal = br.ReadInt32();
                            dictionary.Add(strName, nVal);
                            break;

                        case TYPE.NUMERIC:
                            double dfVal = br.ReadDouble();
                            dictionary.Add(strName, dfVal);                           
                            break;

                        default:
                            string strVal = br.ReadString();
                            dictionary.Add(strName, strVal);
                            break;
                    }
                }
            }

            return dictionary;
        }
    }
}
